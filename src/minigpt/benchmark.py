"""单设备推理 Benchmark 与统一指标定义。

普通 ``generate()`` 不为测量而逐 token 同步；Benchmark 路径为了得到“第一个 token 何时
真正可用”和每次 Decode 的同步延迟，会在这些用户可见边界等待设备完成。两条路径分开，
避免为了漂亮指标改变正常推理热路径。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import statistics
import time
from typing import Any, Callable, Sequence

import torch

from .inference import GenerationConfig, GenerationResult, InferenceEngine
from .backends.profile_types import StepProfiler


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


@dataclass(frozen=True)
class TimedStaticBatch:
    results: list[GenerationResult]
    metrics: dict[str, object]
    requests: list[dict[str, object]]


def _request_record(
    row: int, result: GenerationResult, *, ttft_ms: float,
    tpot_ms: float | None, e2e_latency_ms: float,
) -> dict[str, object]:
    return {
        "request_id": str(row), "row_index": row, "state": "finished",
        "prompt_ids": list(result.prompt_ids), "generated_ids": list(result.generated_ids),
        "stop_reason": result.stop_reason, "input_tokens": len(result.prompt_ids),
        "output_tokens": len(result.generated_ids), "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms, "e2e_latency_ms": e2e_latency_ms,
    }


def generation_output_digest(requests: Sequence[dict[str, object]]) -> str:
    """Hash request identity, input, output and terminal state in row order.

    Recompute after replacing benchmark row IDs with workload request IDs,
    including the corresponding independent profiling replay records.
    """
    canonical = [{key: row[key] for key in (
        "request_id", "prompt_ids", "generated_ids", "stop_reason", "state",
    )} for row in requests]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


_output_digest = generation_output_digest


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
def timed_generate(
    engine: InferenceEngine, prompt: str, config: GenerationConfig, *,
    after_step: Callable[[dict[str, object]], None] | None = None,
) -> TimedGeneration:
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
    with torch.profiler.record_function("minigpt::prefill_phase"):
        with torch.profiler.record_function("minigpt::prefill_model"):
            next_logits = engine.runner.prefill(input_ids, attention_mask)
        with torch.profiler.record_function("minigpt::prefill_token_selection"):
            next_id = engine.select_next_token(next_logits, config, generator)
            next_id = engine.synchronize_next_ids(next_id)
            runtime.synchronize()
        prefill_end = time.perf_counter()
        generated_ids = [int(next_id[0, 0].item())]
        engine.tokenizer.decode(generated_ids)
        first_token_ready = time.perf_counter()
    if after_step is not None:
        after_step({"prefill_batch_size": 1, "decode_batch_size": 0})

    decode_step_seconds: list[float] = []
    stop_reason = "length"
    finished = generated_ids[-1] in stop_token_ids
    if finished:
        stop_reason = "eos"
    for _ in range(1, config.max_new_tokens):
        if finished:
            break
        decode_start = time.perf_counter()
        with torch.profiler.record_function("minigpt::decode_phase"):
            with torch.profiler.record_function("minigpt::decode_model"):
                next_logits = engine.runner.decode(
                    next_id,
                    torch.ones(1, dtype=torch.bool, device=next_id.device),
                )
            with torch.profiler.record_function("minigpt::decode_token_selection"):
                next_id = engine.select_next_token(next_logits, config, generator)
                next_id = engine.synchronize_next_ids(next_id)
                runtime.synchronize()
            token_id = int(next_id[0, 0].item())
            generated_ids.append(token_id)
            engine.tokenizer.decode([token_id])
            decode_step_seconds.append(time.perf_counter() - decode_start)
        if after_step is not None:
            after_step({"prefill_batch_size": 0, "decode_batch_size": 1})
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
    *,
    profiling_session: StepProfiler | None = None,
) -> dict[str, Any]:
    """先 warmup，再重复测量并返回可直接写入 JSON 的完整报告。"""

    if warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")

    for _ in range(warmup):
        engine.generate(prompt, config)
        engine.runner.runtime.synchronize()

    timed_runs: list[TimedGeneration] = []
    measured_memory: list[dict[str, object]] = []
    for _ in range(repeats):
        timed_runs.append(timed_generate(engine, prompt, config))
        # Capture before the next run resets allocator peaks or a profiler
        # replay allocates trace buffers on the device.
        measured_memory.append(engine.runner.runtime.memory_snapshot().to_dict())
    expected_ids = timed_runs[0].result.generated_ids
    if any(run.result.generated_ids != expected_ids for run in timed_runs[1:]):
        raise AssertionError("重复 benchmark 生成结果不稳定")
    run_dicts = []
    for run, memory_snapshot in zip(timed_runs, measured_memory):
        requests = [_request_record(
            0, run.result, ttft_ms=run.metrics.ttft_ms,
            tpot_ms=run.metrics.tpot_ms, e2e_latency_ms=run.metrics.e2e_latency_ms,
        )]
        run_dicts.append({**asdict(run.metrics), "requests": requests,
                         "memory_snapshot": memory_snapshot, "output_sha256": _output_digest(requests)})

    profiling = None
    if profiling_session is not None:
        steps = 0

        def after_step(record: dict[str, object]) -> None:
            nonlocal steps
            steps += 1
            profiling_session.step(record)

        with profiling_session:
            profile_run = timed_generate(engine, prompt, config, after_step=after_step)
        profile_requests = [_request_record(
            0, profile_run.result, ttft_ms=profile_run.metrics.ttft_ms,
            tpot_ms=profile_run.metrics.tpot_ms, e2e_latency_ms=profile_run.metrics.e2e_latency_ms,
        )]
        digest = _output_digest(profile_requests)
        if digest != run_dicts[0]["output_sha256"]:
            raise AssertionError("profiling replay 与 measured single-request 输出不一致")
        profiling = {
            **profiling_session.metadata(), "measurement_excluded": True,
            "replay": {"scheduler_steps": steps, "wall_time_ms": profile_run.metrics.e2e_latency_ms,
                       "output_sha256": digest, "requests": profile_requests},
        }

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
    report = {
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
    if profiling is not None:
        report["profiling"] = profiling
    return report


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
def timed_static_batch(
    engine: InferenceEngine, prompts: Sequence[str], config: GenerationConfig, *,
    after_step: Callable[[dict[str, object]], None] | None = None,
) -> TimedStaticBatch:
    """Measure each row's actual first token and terminal delivery boundary.

All rows arrive together. A completed row is detokenized and timestamped at
that step, while the remaining rows continue through masked Decode. Batch E2E
is kept separately from per-request E2E. Sampling consumes generators in the
same row order as InferenceEngine.generate_batch, including inactive rows.
"""
    if not prompts:
        raise ValueError("prompts 不能为空")
    config.validate()
    stop_tokens = frozenset(config.stop_token_ids())
    if any(token >= engine.tokenizer.vocab_size for token in stop_tokens):
        raise ValueError("停止 token 超出 tokenizer 词表")
    runtime = engine.runner.runtime
    runtime.synchronize()
    runtime.reset_peak_memory()
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    input_ids, attention_mask, prompt_rows = engine.encode_prompts(prompts)
    engine.validate_generation_capacity(prompt_rows, config.max_new_tokens)
    tokenization_end = time.perf_counter()
    batch_size = len(prompts)
    generators = engine.make_generators(config, batch_size)
    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * batch_size
    first_ready: list[float | None] = [None] * batch_size
    last_ready: list[float | None] = [None] * batch_size
    token_intervals: list[list[float]] = [[] for _ in prompts]
    results: list[GenerationResult | None] = [None] * batch_size
    requests: list[dict[str, object] | None] = [None] * batch_size
    decode_step_ms: list[float] = []

    def select(logits: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([
            engine.select_next_token(logits[row : row + 1], config, generators[row])
            for row in range(batch_size)
        ])
        ids = engine.synchronize_next_ids(ids)
        runtime.synchronize()
        return ids

    def deliver(next_ids: torch.Tensor, step: int) -> None:
        for row, prompt in enumerate(prompts):
            if finished[row]:
                next_ids[row, 0] = engine.pad_token_id
                continue
            token_id = int(next_ids[row, 0].item())
            generated[row].append(token_id)
            engine.tokenizer.decode([token_id])
            ready = time.perf_counter()
            if first_ready[row] is None:
                first_ready[row] = ready
            else:
                token_intervals[row].append((ready - float(last_ready[row])) * 1000.0)
            last_ready[row] = ready
            is_eos = token_id in stop_tokens
            if not is_eos and step + 1 < config.max_new_tokens:
                continue
            finished[row] = True
            all_ids = prompt_rows[row] + generated[row]
            result = GenerationResult(
                prompt_text=prompt, completion_text=engine.tokenizer.decode(generated[row]),
                full_text=engine.tokenizer.decode(all_ids), prompt_ids=list(prompt_rows[row]),
                generated_ids=list(generated[row]), all_ids=all_ids,
                prefill_tokens=min(len(prompt_rows[row]), engine.runner.block_size),
                stop_reason="eos" if is_eos else "length",
            )
            completed = time.perf_counter()
            results[row] = result
            request = _request_record(
                row, result, ttft_ms=(float(first_ready[row]) - started) * 1000.0,
                tpot_ms=statistics.fmean(token_intervals[row]) if token_intervals[row] else None,
                e2e_latency_ms=(completed - started) * 1000.0,
            )
            request.update({"first_token_ready_ms": (float(first_ready[row]) - started) * 1000.0,
                            "completed_at_ms": (completed - started) * 1000.0,
                            "decode_step_ms": list(token_intervals[row])})
            requests[row] = request

    prefill_started = time.perf_counter()
    with torch.profiler.record_function("minigpt::prefill_phase"):
        with torch.profiler.record_function("minigpt::prefill_model"):
            logits = engine.runner.prefill(input_ids, attention_mask)
        with torch.profiler.record_function("minigpt::prefill_token_selection"):
            next_ids = select(logits)
        prefill_ended = time.perf_counter()
        deliver(next_ids, 0)
    if after_step is not None:
        after_step({"prefill_batch_size": batch_size, "decode_batch_size": 0})
    for step in range(1, config.max_new_tokens):
        if all(finished):
            break
        active_rows = sum(not value for value in finished)
        decode_started = time.perf_counter()
        with torch.profiler.record_function("minigpt::decode_phase"):
            active_mask = torch.tensor([not value for value in finished], dtype=torch.bool, device=input_ids.device)
            with torch.profiler.record_function("minigpt::decode_model"):
                logits = engine.runner.decode(next_ids, active_mask)
            with torch.profiler.record_function("minigpt::decode_token_selection"):
                next_ids = select(logits)
            deliver(next_ids, step)
            decode_step_ms.append((time.perf_counter() - decode_started) * 1000.0)
        if after_step is not None:
            after_step({"prefill_batch_size": 0, "decode_batch_size": active_rows})
    runtime.synchronize()
    ended = time.perf_counter()
    ended_unix_ns = time.time_ns()
    _, peak_memory_mb = runtime.memory_stats_mb()
    if any(result is None for result in results) or any(request is None for request in requests):
        raise AssertionError("static batch ended with nonterminal requests")
    final_results = [result for result in results if result is not None]
    final_requests = [request for request in requests if request is not None]
    elapsed = ended - started
    output_tokens = sum(len(result.generated_ids) for result in final_results)
    metrics: dict[str, object] = {
        "started_at_unix_ns": started_unix_ns, "ended_at_unix_ns": ended_unix_ns,
        "e2e_latency_ms": elapsed * 1000.0,
        "tokenization_ms": (tokenization_end - started) * 1000.0,
        "prefill_ms": (prefill_ended - prefill_started) * 1000.0,
        "decode_ms": sum(decode_step_ms), "decode_step_ms": decode_step_ms,
        "input_tokens": sum(len(result.prompt_ids) for result in final_results),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / max(elapsed, 1e-12),
        "peak_device_memory_mb": peak_memory_mb,
    }
    metrics["memory_snapshot"] = runtime.memory_snapshot().to_dict()
    return TimedStaticBatch(final_results, metrics, final_requests)


@torch.inference_mode()
def benchmark_static_batch(
    engine: InferenceEngine,
    prompts: Sequence[str],
    config: GenerationConfig,
    *,
    warmup: int = 2,
    repeats: int = 5,
    profiling_session: StepProfiler | None = None,
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

    runs: list[dict[str, object]] = []
    expected_ids: list[list[int]] | None = None
    for _ in range(repeats):
        timed = timed_static_batch(engine, prompts, config)
        generated_ids = [result.generated_ids for result in timed.results]
        if expected_ids is None:
            expected_ids = generated_ids
        elif generated_ids != expected_ids:
            raise AssertionError("重复静态 batch 生成结果不稳定")
        runs.append({**timed.metrics, "requests": timed.requests, "output_sha256": _output_digest(timed.requests)})

    profiling = None
    if profiling_session is not None:
        steps = 0

        def after_step(record: dict[str, object]) -> None:
            nonlocal steps
            steps += 1
            profiling_session.step(record)

        with profiling_session:
            profile_run = timed_static_batch(engine, prompts, config, after_step=after_step)
        digest = _output_digest(profile_run.requests)
        if digest != runs[0]["output_sha256"]:
            raise AssertionError("profiling replay 与 measured static-batch 输出不一致")
        profiling = {
            **profiling_session.metadata(), "measurement_excluded": True,
            "replay": {"scheduler_steps": steps, "wall_time_ms": profile_run.metrics["e2e_latency_ms"],
                       "output_sha256": digest, "requests": profile_run.requests},
        }

    runtime = engine.runner.runtime
    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": f"static_batch_{engine.runner.implementation_name}",
        "measurement_scope": {
            "arrival": "all rows arrive at batch start, before shared tokenization",
            "ttft": "batch start to this row's first token decoded for delivery",
            "tpot": "mean interval between this row's delivered tokens, excluding its first token",
            "request_e2e": "batch start to this row's terminal completion/full-text detokenization; finished rows do not wait for remaining rows",
            "e2e_latency": "whole-batch wall time, retained separately from per-request E2E",
            "profiling": "a separate output-equivalent replay excluded from measured runs",
        },
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
            "ttft_ms": summarize([float(row["ttft_ms"]) for run in runs for row in run["requests"]]),
            "request_e2e_latency_ms": summarize([float(row["e2e_latency_ms"]) for run in runs for row in run["requests"]]),
        },
    }
    tpot_values = [float(row["tpot_ms"]) for run in runs for row in run["requests"] if row["tpot_ms"] is not None]
    report["summary"]["tpot_ms"] = summarize(tpot_values) if tpot_values else None
    if profiling is not None:
        report["profiling"] = profiling
    return report
