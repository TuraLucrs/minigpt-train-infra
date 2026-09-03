"""单设备推理 Benchmark 与统一指标定义。

普通 ``generate()`` 不为测量而逐 token 同步；Benchmark 路径为了得到“第一个 token 何时
真正可用”和每次 Decode 的同步延迟，会在这些用户可见边界等待设备完成。两条路径分开，
避免为了漂亮指标改变正常推理热路径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import platform
import statistics
import time
from typing import Any, Sequence

import torch

from .inference import GenerationConfig, GenerationResult, InferenceEngine


@dataclass(frozen=True)
class InferenceRunMetrics:
    """一次同步单请求生成的原始测量值，时间统一使用毫秒。"""

    input_tokens: int
    prefill_tokens: int
    output_tokens: int
    tokenization_ms: float
    prefill_ms: float
    ttft_ms: float
    decode_ms: float
    tpot_ms: float | None
    e2e_latency_ms: float
    input_tokens_per_second: float
    output_tokens_per_second: float
    decode_tokens_per_second: float | None
    peak_device_memory_mb: float
    decode_step_ms: list[float]


@dataclass(frozen=True)
class TimedGeneration:
    result: GenerationResult
    metrics: InferenceRunMetrics


def percentile(values: Sequence[float], quantile: float) -> float:
    """使用相邻点线性插值计算分位数，单个样本时返回它本身。"""

    if not values:
        raise ValueError("分位数至少需要一个样本")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须在 [0,1] 内")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """保留原始 runs 之外，再给出便于比较的稳定统计量。"""

    if not values:
        raise ValueError("统计至少需要一个样本")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p50": percentile(numeric, 0.50),
        "p90": percentile(numeric, 0.90),
        "p99": percentile(numeric, 0.99),
    }


@torch.inference_mode()
def timed_generate(engine: InferenceEngine, prompt: str, config: GenerationConfig) -> TimedGeneration:
    """执行一次可测量的同步请求。

    TTFT 从请求开始计到第一个 token 已取回并 decode 为文本，因此包含 tokenizer、Host 调度、
    Prefill forward、token 选择和首 token detokenization。``prefill_ms`` 单独记录从模型阶段
    开始到首 token 在 device 上计算完成。TPOT 不包含第一个 token，定义为后续每个 token
    计算、取回并 decode 为文本的平均同步间隔。
    """

    config.validate()
    stop_token_ids = frozenset(config.stop_token_ids())
    if any(token_id >= engine.tokenizer.vocab_size for token_id in stop_token_ids):
        raise ValueError("停止 token 超出 tokenizer 词表")
    runtime = engine.runner.runtime
    runtime.synchronize()
    runtime.reset_peak_memory()

    request_start = time.perf_counter()
    tokenization_start = request_start
    input_ids, attention_mask, prompt_rows = engine.encode_prompts([prompt])
    tokenization_end = time.perf_counter()
    prompt_ids = prompt_rows[0]
    prompt_length = len(prompt_ids)
    engine.validate_generation_capacity(prompt_rows, config.max_new_tokens)
    generator = engine.make_generator(config)

    prefill_start = time.perf_counter()
    next_logits = engine.runner.prefill(input_ids, attention_mask)
    next_id = engine.select_next_token(next_logits, config, generator)
    runtime.synchronize()
    prefill_end = time.perf_counter()
    generated_ids = [int(next_id[0, 0].item())]
    engine.tokenizer.decode(generated_ids)
    first_token_ready = time.perf_counter()

    decode_step_seconds: list[float] = []
    stop_reason = "length"
    finished = generated_ids[-1] in stop_token_ids
    if finished:
        stop_reason = "eos"
    for _ in range(1, config.max_new_tokens):
        if finished: break
        decode_start = time.perf_counter()
        next_logits = engine.runner.decode(next_id, torch.ones(1, dtype=torch.bool, device=next_id.device))
        next_id = engine.select_next_token(next_logits, config, generator)
        runtime.synchronize()
        token_id = int(next_id[0, 0].item())
        generated_ids.append(token_id)
        engine.tokenizer.decode([token_id])
        decode_step_seconds.append(time.perf_counter() - decode_start)
        if token_id in stop_token_ids:
            finished = True
            stop_reason = "eos"

    all_ids = prompt_ids + generated_ids
    completion_text = engine.tokenizer.decode(generated_ids)
    full_text = engine.tokenizer.decode(all_ids)
    request_end = time.perf_counter()

    tokenization_seconds = tokenization_end - tokenization_start
    prefill_seconds = prefill_end - prefill_start
    ttft_seconds = first_token_ready - request_start
    decode_seconds = sum(decode_step_seconds)
    e2e_seconds = request_end - request_start
    decode_token_count = len(decode_step_seconds)
    tpot_seconds = decode_seconds / decode_token_count if decode_token_count else None
    _, peak_memory_mb = runtime.memory_stats_mb()
    prefill_tokens = min(prompt_length, engine.runner.block_size)

    result = GenerationResult(
        prompt_text=prompt,
        completion_text=completion_text,
        full_text=full_text,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        all_ids=all_ids,
        prefill_tokens=prefill_tokens,
        stop_reason=stop_reason,
    )
    metrics = InferenceRunMetrics(
        input_tokens=prompt_length,
        prefill_tokens=prefill_tokens,
        output_tokens=len(generated_ids),
        tokenization_ms=tokenization_seconds * 1000.0,
        prefill_ms=prefill_seconds * 1000.0,
        ttft_ms=ttft_seconds * 1000.0,
        decode_ms=decode_seconds * 1000.0,
        tpot_ms=tpot_seconds * 1000.0 if tpot_seconds is not None else None,
        e2e_latency_ms=e2e_seconds * 1000.0,
        input_tokens_per_second=prefill_tokens / max(prefill_seconds, 1e-12),
        output_tokens_per_second=len(generated_ids) / max(e2e_seconds, 1e-12),
        decode_tokens_per_second=(1.0 / tpot_seconds) if tpot_seconds is not None else None,
        peak_device_memory_mb=peak_memory_mb,
        decode_step_ms=[seconds * 1000.0 for seconds in decode_step_seconds],
    )
    return TimedGeneration(result=result, metrics=metrics)


def benchmark_generation(
    engine: InferenceEngine,
    prompt: str,
    config: GenerationConfig,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    """先 warmup，再重复测量并返回可直接写入 JSON 的完整报告。"""

    if warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")

    for _ in range(warmup):
        engine.generate(prompt, config)
        engine.runner.runtime.synchronize()

    timed_runs = [timed_generate(engine, prompt, config) for _ in range(repeats)]
    expected_ids = timed_runs[0].result.generated_ids
    if any(run.result.generated_ids != expected_ids for run in timed_runs[1:]):
        raise AssertionError("重复 benchmark 生成结果不稳定")
    run_dicts = [asdict(run.metrics) for run in timed_runs]

    summary_fields = (
        "tokenization_ms",
        "prefill_ms",
        "ttft_ms",
        "decode_ms",
        "e2e_latency_ms",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "peak_device_memory_mb",
    )
    summary = {field: summarize([float(run[field]) for run in run_dicts]) for field in summary_fields}

    tpot_values = [float(run["tpot_ms"]) for run in run_dicts if run["tpot_ms"] is not None]
    decode_throughput_values = [
        float(run["decode_tokens_per_second"])
        for run in run_dicts
        if run["decode_tokens_per_second"] is not None
    ]
    if tpot_values:
        summary["tpot_ms"] = summarize(tpot_values)
        summary["decode_tokens_per_second"] = summarize(decode_throughput_values)

    runtime = engine.runner.runtime
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": f"single_request_{runtime.device.type}_{engine.runner.implementation_name}",
        "measurement_scope": {
            "prefill": "Prefill 开始到首 token 在 device 上计算完成，包含 token 选择",
            "ttft": "请求开始到第一个 token 取回并 decode 为文本",
            "tpot": "除第一个 token 外，后续 token 计算、取回并 decode 为文本的平均同步间隔",
            "e2e_latency": "请求开始到完整 token 序列 decode 为文本",
            "decode_implementation": engine.runner.implementation_name,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            **runtime.backend_metadata(),
        },
        "request": {
            "prompt": prompt,
            "max_new_tokens": config.max_new_tokens,
            "strategy": config.strategy,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "seed": config.seed,
            "eos_token_ids": list(config.stop_token_ids()),
            "warmup": warmup,
            "repeats": repeats,
        },
        "result": {
            "completion_text": timed_runs[-1].result.completion_text,
            "generated_ids": timed_runs[-1].result.generated_ids,
            "stop_reason": timed_runs[-1].result.stop_reason,
        },
        "runs": run_dicts,
        "summary": summary,
    }


def compare_decode_modes(
    recompute_engine: InferenceEngine,
    cached_engine: InferenceEngine,
    prompt: str,
    config: GenerationConfig,
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    """先验证生成结果一致，再比较 recompute 与 KV Cache 的指标。"""

    release = getattr(cached_engine.runner, "release_cache", None)
    if callable(release):
        cached_engine.runner.runtime.synchronize()
        release()
    recompute = benchmark_generation(recompute_engine, prompt, config, warmup, repeats)
    cached = benchmark_generation(cached_engine, prompt, config, warmup, repeats)
    recompute_ids = recompute["result"]["generated_ids"]
    cached_ids = cached["result"]["generated_ids"]
    if recompute_ids != cached_ids:
        raise AssertionError("KV Cache 与 recompute 生成结果不一致")

    recompute_tpot = recompute["summary"].get("tpot_ms", {}).get("median")
    cached_tpot = cached["summary"].get("tpot_ms", {}).get("median")
    return {
        "schema_version": 2,
        "benchmark": "recompute_vs_kv_cache",
        "correctness_gate": {
            "generated_ids_equal": True,
            "generated_ids": cached_ids,
        },
        "recompute": recompute,
        "kv_cache": cached,
        "comparison": {
            "median_tpot_speedup": (
                None
                if recompute_tpot is None or cached_tpot is None
                else float(recompute_tpot) / max(float(cached_tpot), 1e-12)
            ),
            "median_peak_memory_delta_mb": (
                float(cached["summary"]["peak_device_memory_mb"]["median"])
                - float(recompute["summary"]["peak_device_memory_mb"]["median"])
            ),
        },
    }


@torch.inference_mode()
def benchmark_static_batch(
    engine: InferenceEngine,
    prompts: Sequence[str],
    config: GenerationConfig,
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    """测量固定请求集合在静态 batch 下的 E2E 吞吐和设备内存。"""

    if not prompts:
        raise ValueError("prompts 不能为空")
    if warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")
    config.validate()

    for _ in range(warmup):
        engine.generate_batch(prompts, config)
        engine.runner.runtime.synchronize()

    runs: list[dict[str, float | int]] = []
    expected_ids: list[list[int]] | None = None
    for _ in range(repeats):
        runtime = engine.runner.runtime
        runtime.synchronize()
        runtime.reset_peak_memory()
        started = time.perf_counter()
        results = engine.generate_batch(prompts, config)
        runtime.synchronize()
        elapsed = time.perf_counter() - started
        _, peak_memory_mb = runtime.memory_stats_mb()

        generated_ids = [result.generated_ids for result in results]
        if expected_ids is None:
            expected_ids = generated_ids
        elif generated_ids != expected_ids:
            raise AssertionError("重复静态 batch 生成结果不稳定")
        input_tokens = sum(len(result.prompt_ids) for result in results)
        output_tokens = sum(len(result.generated_ids) for result in results)
        runs.append(
            {
                "e2e_latency_ms": elapsed * 1000.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / max(elapsed, 1e-12),
                "peak_device_memory_mb": peak_memory_mb,
            }
        )

    runtime = engine.runner.runtime
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": f"static_batch_{engine.runner.implementation_name}",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            **runtime.backend_metadata(),
        },
        "request": {
            "batch_size": len(prompts),
            "prompt_lengths": [len(engine.tokenizer.encode(prompt)) for prompt in prompts],
            "max_new_tokens": config.max_new_tokens,
            "strategy": config.strategy,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "seed": config.seed,
            "eos_token_id": config.eos_token_id,
            "eos_token_ids": list(config.stop_token_ids()),
            "warmup": warmup,
            "repeats": repeats,
        },
        "result": {"generated_ids": expected_ids},
        "runs": runs,
        "summary": {
            "e2e_latency_ms": summarize(
                [float(run["e2e_latency_ms"]) for run in runs]
            ),
            "output_tokens_per_second": summarize(
                [float(run["output_tokens_per_second"]) for run in runs]
            ),
            "peak_device_memory_mb": summarize(
                [float(run["peak_device_memory_mb"]) for run in runs]
            ),
        },
    }
