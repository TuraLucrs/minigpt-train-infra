"""Continuous Batching trace benchmark 与跨重复统计。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
import statistics
import time
from typing import Callable, Sequence

import torch

from .benchmark import percentile
from .replay import OfflineTraceReplayer, ReplayCollectives
from .serving import ContinuousBatchEngine
from .workload import WorkloadTrace


def summarize_samples(values: Sequence[float]) -> dict[str, float | int] | None:
    """服务指标统一使用 p50/p95/p99；空集合显式返回 null。"""

    if not values:
        return None
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "p99": percentile(numeric, 0.99),
    }


def serving_output_digest(serving_report: dict[str, object]) -> str:
    """由逐请求终态与实际生成 token 重新计算可审计输出摘要。"""

    requests = serving_report["requests"]
    canonical = [
        {
            "request_id": request["request_id"],
            "state": request["state"],
            "stop_reason": request["stop_reason"],
            "generated_ids": request["generated_ids"],
        }
        for request in sorted(requests, key=lambda item: str(item["request_id"]))
    ]
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _run_throughput(
    serving_report: dict[str, object],
    wall_time_ms: float,
) -> dict[str, float | int]:
    summary = serving_report["summary"]
    duration_seconds = max(wall_time_ms / 1000.0, 1e-12)
    completed = int(summary["completed_requests"])
    good = int(summary["good_requests"])
    completed_requests = [
        request
        for request in serving_report["requests"]
        if request["state"] == "finished"
    ]
    input_tokens = sum(int(request["input_tokens"]) for request in completed_requests)
    output_tokens = sum(int(request["output_tokens"]) for request in completed_requests)
    return {
        "completed_requests": completed,
        "good_requests": good,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "completed_requests_per_second": completed / duration_seconds,
        "goodput_requests_per_second": good / duration_seconds,
        "input_tokens_per_second": input_tokens / duration_seconds,
        "output_tokens_per_second": output_tokens / duration_seconds,
    }


def _aggregate_runs(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    throughput_fields = (
        "completed_requests_per_second",
        "goodput_requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
    )
    summary: dict[str, object] = {
        field: summarize_samples(
            [float(run["throughput"][field]) for run in runs]
        )
        for field in throughput_fields
    }
    for field in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms"):
        values = [
            float(request[field])
            for run in runs
            for request in run["serving"]["requests"]
            if request["state"] == "finished" and request[field] is not None
        ]
        summary[field] = summarize_samples(values)
    summary["active_batch_size"] = summarize_samples(
        [
            float(step["active_after"])
            for run in runs
            for step in run["serving"]["steps"]
        ]
    )
    summary["decode_batch_size"] = summarize_samples(
        [
            float(step["decode_batch_size"])
            for run in runs
            for step in run["serving"]["steps"]
        ]
    )
    summary["prefill_batch_size"] = summarize_samples(
        [
            float(step["prefill_batch_size"])
            for run in runs
            for step in run["serving"]["steps"]
        ]
    )
    summary["wall_time_ms"] = summarize_samples(
        [float(run["replay"]["wall_time_ms"]) for run in runs]
    )
    summary["peak_device_memory_mb"] = summarize_samples(
        [float(run["memory"]["peak_device_memory_mb"]) for run in runs]
    )
    summary["peak_kv_slots_used"] = max(
        int(run["serving"]["kv_cache"]["peak_slots_used"]) for run in runs
    )
    summary["peak_kv_used_tokens"] = max(
        int(run["serving"]["kv_cache"]["peak_used_tokens"]) for run in runs
    )
    summary["peak_kv_internal_waste_tokens"] = max(
        int(run["serving"]["kv_cache"]["peak_internal_waste_tokens"])
        for run in runs
    )
    summary["state_counts_per_run"] = [
        dict(run["serving"]["summary"]["state_counts"]) for run in runs
    ]
    return summary


@torch.inference_mode()
def benchmark_trace_replay(
    engine: ContinuousBatchEngine,
    trace: WorkloadTrace,
    *,
    mode: str,
    closed_loop_clients: int | None,
    warmup: int,
    repeats: int,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
    distributed: ReplayCollectives | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    before_replay: Callable[[], None] | None = None,
    after_replay: Callable[[], None] | None = None,
) -> dict[str, object]:
    """重放同一 trace，保存每次原始请求/step，并验证输出可重复。"""

    if warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")
    trace.validate()

    def replay_once() -> tuple[dict[str, object], dict[str, object]]:
        if before_replay is not None:
            before_replay()
        engine.runner.runtime.reset_peak_memory()
        replay = OfflineTraceReplayer(
            engine,
            trace,
            mode=mode,
            closed_loop_clients=closed_loop_clients,
            distributed=distributed,
            control_device=engine.runner.runtime.device,
            clock=clock,
            sleeper=sleeper,
        ).run()
        engine.runner.runtime.synchronize()
        current_memory_mb, peak_memory_mb = engine.runner.runtime.memory_stats_mb()
        if after_replay is not None:
            after_replay()
        serving = engine.report(
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            e2e_slo_ms=e2e_slo_ms,
        )
        run = {
            "replay": asdict(replay),
            "throughput": _run_throughput(serving, replay.wall_time_ms),
            "memory": {
                "current_device_memory_mb": current_memory_mb,
                "peak_device_memory_mb": peak_memory_mb,
            },
            "output_sha256": serving_output_digest(serving),
            "serving": serving,
        }
        return run, serving

    for _ in range(warmup):
        replay_once()
        engine.reset()

    runs: list[dict[str, object]] = []
    for repeat in range(repeats):
        run, _serving = replay_once()
        run["repeat"] = repeat
        runs.append(run)
        if repeat + 1 < repeats:
            engine.reset()

    output_digests = {str(run["output_sha256"]) for run in runs}
    if len(output_digests) != 1:
        raise AssertionError("相同 trace 的重复运行产生了不同请求输出")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "continuous_batching_trace_replay",
        "measurement_scope": {
            "open_loop": (
                "按 trace 到达时间提交；调度 step 期间到达的请求在下一轮可见"
            ),
            "closed_loop": (
                "固定 client 数；请求终止后才由同等 client 容量补入下一个请求"
            ),
            "throughput": "从每次 replay 起点到全部请求终止的 wall time",
            "goodput": "同时满足所配置 TTFT、TPOT、request/E2E SLO 的完成请求",
            "latency_percentiles": "合并全部 measured repeats 的完成请求后计算",
        },
        "protocol": {
            "mode": mode,
            "closed_loop_clients": closed_loop_clients,
            "warmup": warmup,
            "repeats": repeats,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            "e2e_slo_ms": e2e_slo_ms,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            **engine.runner.runtime.backend_metadata(),
        },
        "workload": {
            "workload_id": trace.workload_id,
            "workload_class": trace.workload_class,
            "request_sha256": trace.request_sha256,
            "source_sha256": trace.source_sha256,
            "request_count": len(trace.requests),
            "metadata": dict(trace.metadata),
            "partition": None if trace.partition is None else asdict(trace.partition),
        },
        "engine": {
            "runner": engine.runner.implementation_name,
            "max_slots": engine.runner.max_slots,
            "max_seq_len": engine.runner.max_seq_len,
            "max_queue_size": engine.max_queue_size,
        },
        "summary": _aggregate_runs(runs),
        "runs": runs,
    }
