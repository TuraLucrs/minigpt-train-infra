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
    runtime = engine.runner.runtime
    runtime.synchronize()
    runtime.reset_peak_memory()

    request_start = time.perf_counter()
    tokenization_start = request_start
    input_ids = engine.encode_prompt(prompt)
    tokenization_end = time.perf_counter()
    prompt_length = input_ids.shape[1]
    generator = engine.make_generator(config)

    prefill_start = time.perf_counter()
    next_logits = engine.runner.prefill(input_ids)
    next_id = engine.select_next_token(next_logits, config, generator)
    input_ids = torch.cat((input_ids, next_id), dim=1)
    runtime.synchronize()
    prefill_end = time.perf_counter()
    engine.tokenizer.decode([int(next_id[0, 0].item())])
    first_token_ready = time.perf_counter()

    decode_step_seconds: list[float] = []
    for _ in range(1, config.max_new_tokens):
        decode_start = time.perf_counter()
        next_logits = engine.runner.decode(input_ids)
        next_id = engine.select_next_token(next_logits, config, generator)
        input_ids = torch.cat((input_ids, next_id), dim=1)
        runtime.synchronize()
        engine.tokenizer.decode([int(next_id[0, 0].item())])
        decode_step_seconds.append(time.perf_counter() - decode_start)

    all_ids = input_ids[0].tolist()
    prompt_ids = all_ids[:prompt_length]
    generated_ids = all_ids[prompt_length:]
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
        stop_reason="length",
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
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "single_request_recompute_decode",
        "measurement_scope": {
            "prefill": "Prefill 开始到首 token 在 device 上计算完成，包含 token 选择",
            "ttft": "请求开始到第一个 token 取回并 decode 为文本",
            "tpot": "除第一个 token 外，后续 token 计算、取回并 decode 为文本的平均同步间隔",
            "e2e_latency": "请求开始到完整 token 序列 decode 为文本",
            "decode_implementation": "v0.3 每步重算当前完整有效上下文，不使用 KV Cache",
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(runtime.device),
            "device_name": runtime.device_name(),
            "precision": runtime.precision,
        },
        "request": {
            "prompt": prompt,
            "max_new_tokens": config.max_new_tokens,
            "strategy": config.strategy,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "seed": config.seed,
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
