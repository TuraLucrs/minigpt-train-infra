"""v0.7 workload、replay 与 least-loaded 多副本路由正确性门。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.inference import (  # noqa: E402
    GenerationConfig,
    SlotCachedMiniGPTModelRunner,
)
from minigpt.model import MiniGPT, MiniGPTConfig  # noqa: E402
from minigpt.replica import (  # noqa: E402
    LeastLoadedRouter,
    MultiReplicaServing,
    ReplicaSnapshot,
    partition_workload_by_projected_load,
)
from minigpt.replay import OfflineTraceReplayer  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402
from minigpt.serving import ContinuousBatchEngine, RequestSpec, RequestState  # noqa: E402
from minigpt.serving_acceptance import summarize_v07_acceptance  # noqa: E402
from minigpt.serving_benchmark import (  # noqa: E402
    benchmark_trace_replay,
    serving_output_digest,
)
from minigpt.serving_layout import (  # noqa: E402
    load_layout_manifest,
    load_serving_report,
    summarize_serving_layouts,
)
from minigpt.serving_telemetry import (  # noqa: E402
    load_telemetry,
    parse_npu_smi_common,
    parse_npu_smi_usages,
    summarize_telemetry,
)
from minigpt.tokenizer import CharTokenizer  # noqa: E402
from minigpt.workload import WorkloadTrace, generate_workload  # noqa: E402


class ManualClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        if seconds < 0.0:
            raise ValueError("不能让时钟倒退")
        self.seconds += seconds


class AdvancingRunner:
    """为 deterministic replay 测试给每次模型调用增加固定虚拟耗时。"""

    def __init__(
        self,
        delegate: SlotCachedMiniGPTModelRunner,
        clock: ManualClock,
    ) -> None:
        self.delegate = delegate
        self.clock = clock
        self.runtime = delegate.runtime
        self.implementation_name = f"timed_{delegate.implementation_name}"
        self.max_slots = delegate.max_slots
        self.max_seq_len = delegate.max_seq_len

    def validate_request(self, prompt_length: int, max_new_tokens: int) -> None:
        self.delegate.validate_request(prompt_length, max_new_tokens)

    def prefill_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.delegate.prefill_slots(slot_ids, input_ids, attention_mask)
        self.clock.advance(0.001)
        return logits

    def decode_slots(
        self,
        slot_ids: Sequence[int],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.delegate.decode_slots(slot_ids, input_ids)
        self.clock.advance(0.001)
        return logits

    def release_slots(self, slot_ids: Sequence[int]) -> None:
        self.delegate.release_slots(slot_ids)

    def cache_lengths(self, slot_ids: Sequence[int]) -> list[int]:
        return self.delegate.cache_lengths(slot_ids)


class RecordingStepProfiler:
    def __init__(self) -> None:
        self.started = False
        self.finished = False
        self.records: list[dict[str, object]] = []

    def __enter__(self) -> "RecordingStepProfiler":
        assert self.started is False
        self.started = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self.started is True
        self.finished = True

    def step(self, record: dict[str, object]) -> None:
        assert self.started is True and self.finished is False
        self.records.append(record)

    def metadata(self) -> dict[str, object]:
        assert self.finished is True
        return {
            "collector": "test",
            "selected": True,
            "observed_scheduler_steps": len(self.records),
            "protocol": {"kind": "test"},
        }


def build_engine(
    clock: ManualClock,
    *,
    seed: int,
    max_slots: int = 2,
    max_queue_size: int = 16,
) -> ContinuousBatchEngine:
    torch.manual_seed(seed)
    tokenizer = CharTokenizer.train_from_text("abcdefghijklmnopqrstuvwxyz \n")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=24,
            n_layer=1,
            n_head=2,
            n_embd=24,
            dropout=0.0,
        )
    ).eval()
    runtime = RuntimeContext.create("cpu", "fp32")
    delegate = SlotCachedMiniGPTModelRunner(model, runtime, max_slots=max_slots)
    return ContinuousBatchEngine(
        AdvancingRunner(delegate, clock),
        tokenizer,
        max_queue_size=max_queue_size,
        clock=clock,
    )


def tiny_trace(*, arrival_interval_ms: float = 2.0) -> WorkloadTrace:
    requests = tuple(
        RequestSpec(
            request_id=f"request-{index}",
            prompt="ab" + "c" * (index % 3),
            config=GenerationConfig(max_new_tokens=2 + index % 3),
            arrival_time_ms=index * arrival_interval_ms,
        )
        for index in range(8)
    )
    trace = WorkloadTrace(
        workload_id="tiny-mixed",
        workload_class="mixed",
        requests=requests,
        metadata={"purpose": "cpu correctness"},
    )
    trace.validate()
    return trace


def check_workload_schema_and_presets() -> None:
    for preset in ("short_short", "long_prefill_short_decode", "mixed"):
        generated = generate_workload(
            preset,
            request_count=8,
            arrival_interval_ms=1.5,
            seed=7,
        )
        assert generated.workload_class == preset
        assert len(generated.request_sha256) == 64

    trace = tiny_trace()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.json"
        trace.save(path)
        restored = WorkloadTrace.load(path)
        assert restored == trace
        assert restored.request_sha256 == trace.request_sha256

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["requests"][0]["prompt"] = "changed"
        try:
            WorkloadTrace.from_dict(tampered)
        except ValueError as exc:
            assert "sha256" in str(exc).lower()
        else:
            raise AssertionError("被篡改的 workload 必须被 digest 门禁拒绝")


def check_router_scoring_and_live_replicas() -> None:
    router = LeastLoadedRouter()
    selected = router.choose(
        [
            ReplicaSnapshot("replica-b", 1, 0, 2, 10),
            ReplicaSnapshot("replica-a", 1, 0, 2, 10),
            ReplicaSnapshot("replica-c", 2, 1, 2, 30),
        ]
    )
    assert selected == "replica-a"

    clock = ManualClock()
    serving = MultiReplicaServing(
        {
            "replica-0": build_engine(clock, seed=11),
            "replica-1": build_engine(clock, seed=11),
        },
        clock=clock,
    )
    for request in tiny_trace(arrival_interval_ms=0.0).requests:
        serving.submit(request, now_ms=0.0)
    routed = list(serving.request_to_replica.values())
    assert set(routed) == {"replica-0", "replica-1"}
    assert abs(routed.count("replica-0") - routed.count("replica-1")) <= 1
    serving.run_until_idle()
    assert all(
        request.state == RequestState.FINISHED
        for engine in serving.replicas.values()
        for request in engine.requests.values()
    )
    report = serving.report()
    assert report["replica_count"] == 2
    assert report["completed_requests_per_second"] > 0.0


def check_open_and_closed_loop_replay() -> None:
    trace = tiny_trace(arrival_interval_ms=5.0)
    open_clock = ManualClock()
    open_engine = build_engine(open_clock, seed=17, max_slots=1)
    open_result = OfflineTraceReplayer(
        open_engine,
        trace,
        mode="open_loop",
        clock=open_clock,
        sleeper=open_clock.advance,
    ).run()
    assert open_result.submitted_requests == len(trace.requests)
    assert open_result.wait_count > 0
    assert open_result.scheduler_steps > 0
    assert open_engine.is_idle
    assert all(
        request.state == RequestState.FINISHED
        for request in open_engine.requests.values()
    )

    closed_clock = ManualClock()
    closed_serving = MultiReplicaServing(
        {
            "replica-0": build_engine(closed_clock, seed=23, max_slots=1),
            "replica-1": build_engine(closed_clock, seed=23, max_slots=1),
        },
        clock=closed_clock,
    )
    closed_result = OfflineTraceReplayer(
        closed_serving,
        trace,
        mode="closed_loop",
        closed_loop_clients=3,
        clock=closed_clock,
        sleeper=closed_clock.advance,
    ).run()
    assert closed_result.submitted_requests == len(trace.requests)
    assert closed_result.closed_loop_clients == 3
    assert closed_serving.is_idle
    assert len(closed_serving.routes) == len(trace.requests)


def check_projected_load_partitions() -> None:
    trace = tiny_trace()
    partitions, manifest = partition_workload_by_projected_load(
        trace,
        replica_count=4,
        max_slots_per_replica=2,
    )
    assert len(partitions) == 4
    assert len(manifest) == len(trace.requests)
    assert {partition.partition.replica_index for partition in partitions} == {
        0,
        1,
        2,
        3,
    }
    assert all(
        partition.source_sha256 == trace.request_sha256
        for partition in partitions
    )
    recovered_ids = {
        request.request_id
        for partition in partitions
        for request in partition.requests
    }
    assert recovered_ids == {request.request_id for request in trace.requests}


def check_replay_benchmark_protocol() -> None:
    clock = ManualClock()
    engine = build_engine(clock, seed=31, max_slots=2)
    report = benchmark_trace_replay(
        engine,
        tiny_trace(arrival_interval_ms=1.0),
        mode="open_loop",
        closed_loop_clients=None,
        warmup=1,
        repeats=2,
        ttft_slo_ms=1_000.0,
        tpot_slo_ms=1_000.0,
        e2e_slo_ms=1_000.0,
        clock=clock,
        sleeper=clock.advance,
    )
    assert report["protocol"]["warmup"] == 1
    assert report["protocol"]["open_loop_admission_scripted"] is False
    assert len(report["runs"]) == 2
    assert report["runs"][0]["output_sha256"] == report["runs"][1]["output_sha256"]
    assert report["summary"]["goodput_requests_per_second"]["median"] > 0.0
    assert report["summary"]["ttft_ms"]["p95"] >= 0.0
    assert report["summary"]["active_batch_size"]["p99"] >= 1.0
    assert report["summary"]["scheduler_phase_fraction"][
        "decode_phase_ms"
    ]["median"] >= 0.0

    scripted_clock = ManualClock()
    scripted_report = benchmark_trace_replay(
        build_engine(scripted_clock, seed=31, max_slots=2),
        tiny_trace(arrival_interval_ms=1.0),
        mode="open_loop",
        closed_loop_clients=None,
        warmup=1,
        repeats=2,
        ttft_slo_ms=1_000.0,
        tpot_slo_ms=1_000.0,
        e2e_slo_ms=1_000.0,
        clock=scripted_clock,
        sleeper=scripted_clock.advance,
        deterministic_open_loop=True,
    )
    assert scripted_report["protocol"]["open_loop_admission_scripted"] is True
    assert (
        scripted_report["runs"][0]["output_sha256"]
        == scripted_report["runs"][1]["output_sha256"]
    )

    profiled_clock = ManualClock()
    profiler = RecordingStepProfiler()
    profiled_report = benchmark_trace_replay(
        build_engine(profiled_clock, seed=31, max_slots=2),
        tiny_trace(arrival_interval_ms=1.0),
        mode="open_loop",
        closed_loop_clients=None,
        warmup=1,
        repeats=2,
        ttft_slo_ms=1_000.0,
        tpot_slo_ms=1_000.0,
        e2e_slo_ms=1_000.0,
        clock=profiled_clock,
        sleeper=profiled_clock.advance,
        deterministic_open_loop=True,
        profiling_session=profiler,
    )
    assert len(profiled_report["runs"]) == 2
    assert profiled_report["profiling"]["measurement_excluded"] is True
    assert profiled_report["profiling"]["observed_scheduler_steps"] == len(
        profiler.records
    )
    assert profiled_report["profiling"]["replay"]["output_sha256"] in {
        run["output_sha256"] for run in profiled_report["runs"]
    }

    try:
        benchmark_trace_replay(
            build_engine(ManualClock(), seed=31, max_slots=2),
            tiny_trace(),
            mode="closed_loop",
            closed_loop_clients=2,
            warmup=1,
            repeats=1,
            deterministic_open_loop=True,
        )
    except ValueError as exc:
        assert "仅支持 open_loop" in str(exc)
    else:
        raise AssertionError("closed_loop 不能启用 deterministic_open_loop")


def fake_layout_report(
    layout_id: str,
    *,
    tp_size: int,
    replica_count: int,
    replica_index: int,
    request_ids: list[str],
    routing_assignments: list[dict[str, object]],
    duration_ms: float,
    workload_class: str = "mixed",
    mode: str = "closed_loop",
) -> dict[str, object]:
    local_devices = list(
        range(replica_index * tp_size, (replica_index + 1) * tp_size)
    )
    max_slots = 8 // replica_count
    max_queue_size = 16 // replica_count
    runs = []
    for repeat in range(3):
        start_ns = 1_000_000_000 + repeat * 2_000_000_000 + replica_index * 1_000
        requests = [
            {
                "request_id": request_id,
                "state": "finished",
                "stop_reason": "max_new_tokens",
                "slo_met": True,
                "input_tokens": 10,
                "output_tokens": 4,
                "prompt_ids": list(range(10)),
                "generated_ids": [1, 2, 3, 4],
                "queue_ms": 1.0,
                "ttft_ms": 10.0,
                "tpot_ms": 5.0,
                "e2e_latency_ms": 30.0,
                "deadline_ms": None,
            }
            for request_id in request_ids
        ]
        serving = {
            "requests": requests,
            "steps": [
                {
                    "duration_ms": 10.0,
                    "decode_phase_ms": 6.0,
                    "prefill_phase_ms": 3.0,
                    "scheduler_bookkeeping_ms": 1.0,
                    "active_after": len(requests),
                    "decode_batch_size": len(requests),
                    "prefill_batch_size": len(requests),
                }
            ],
            "kv_cache": {
                "slot_capacity": max_slots,
                "peak_slots_used": min(len(requests), max_slots),
                "peak_used_tokens": 64,
                "peak_internal_waste_tokens": 16,
                "peak_internal_waste_capacity_ratio": 16 / (max_slots * 4096),
                "mean_active_internal_waste_ratio": 0.5,
                "external_fragmentation_tokens": 0,
            },
        }
        runs.append(
            {
                "repeat": repeat,
                "replay": {
                    "started_at_unix_ns": start_ns,
                    "ended_at_unix_ns": start_ns + int(duration_ms * 1_000_000),
                },
                "output_sha256": serving_output_digest(serving),
                "serving": serving,
            }
        )
    report = {
        "benchmark": "continuous_batching_trace_replay",
        "protocol": {
            "mode": mode,
            "open_loop_admission_scripted": mode == "open_loop",
            "closed_loop_clients": (
                None if mode == "open_loop" else 8 // replica_count
            ),
            "warmup": 1,
            "repeats": 3,
            "ttft_slo_ms": 100.0,
            "tpot_slo_ms": 20.0,
            "e2e_slo_ms": 500.0,
        },
        "environment": {
            "python": "3.12.14",
            "pytorch": "2.10.0",
            "torch": "2.10.0",
            "torch_npu": "2.10.0",
            "cann_version": "CANN-test-version",
            "device_type": "npu",
            "device_name": "Ascend",
            "precision": "bf16",
            "total_memory_mb": 65536.0,
        },
        "engine": {
            "runner": "SlotCachedTensorParallelQwen3ModelRunner",
            "max_slots": max_slots,
            "max_seq_len": 4096,
            "max_queue_size": max_queue_size,
        },
        "workload": {
            "workload_class": workload_class,
            "source_sha256": "a" * 64,
            "source_file_sha256": "9" * 64,
            "request_sha256": "b" * 64,
            "request_count": len(request_ids),
            "partition": (
                None
                if replica_count == 1
                else {
                    "source_sha256": "a" * 64,
                    "replica_count": replica_count,
                    "replica_index": replica_index,
                }
            ),
            "routing": {
                "router": "least_projected_load",
                "estimator": "prompt_unicode_codepoints_plus_max_new_tokens",
                "assignment_sha256": hashlib.sha256(
                    json.dumps(
                        routing_assignments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "assignments": routing_assignments,
            },
        },
        "model": {
            "type": "TensorParallelQwen3ForCausalLM",
            "config": {"hidden_size": 5120},
            "full_parameter_count": 32_762_123_264,
        },
        "distributed": {
            "backend": "hccl",
            "layout_id": layout_id,
            "tp_size": tp_size,
            "replica_count": replica_count,
            "replica_index": replica_index,
            "global_world_size": 8,
            "logical_device_ids": local_devices,
            "global_logical_device_ids": list(range(8)),
            "global_closed_loop_clients": None if mode == "open_loop" else 8,
            "replica_closed_loop_clients": (
                None if mode == "open_loop" else 8 // replica_count
            ),
            "physical_card_count": 4,
            "chips_per_card": 2,
            "interconnect_topology": "four dual-chip cards",
            "hostname": "ascend-test-host",
            "visible_device_count": 8,
            "per_rank": [
                {
                    "logical_device_id": device_id,
                    "max_measured_peak_mb": 10_000.0 / tp_size,
                }
                for device_id in local_devices
            ],
        },
        "provenance": {
            "git": {"commit": "c" * 40, "dirty": False},
            "config_sha256": "d" * 64,
            "metadata_sha256": {"tokenizer.json": "e" * 64},
            "weight_hashes_included": True,
            "weights": [
                {
                    "name": "model.safetensors",
                    "size_bytes": 1024,
                    "sha256": "f" * 64,
                }
            ],
        },
        "evidence_class": "qwen3_32b_ascend_continuous_batching_candidate",
        "runs": runs,
    }
    profile_windows = {
        "short_short": (8, 2, 4),
        "long_prefill_short_decode": (6, 1, 4),
        "mixed": (14, 1, 4),
    }
    profile_skip, profile_warmup, profile_active = profile_windows[workload_class]
    profile_protocol = {
        "skip_steps": profile_skip,
        "warmup_steps": profile_warmup,
        "active_steps": profile_active,
        "profiler_level": "level1",
        "aic_metrics": "pipe_utilization",
        "record_shapes": True,
        "profile_memory": False,
        "with_stack": False,
        "sys_interconnection": True,
    }
    observed_scheduler_steps = sum(
        int(profile_protocol[field])
        for field in ("skip_steps", "warmup_steps", "active_steps")
    )
    report["profiling"] = {
        "collector": "torch_npu.profiler",
        "selected": True,
        "global_rank": replica_index * tp_size,
        "logical_device_id": local_devices[0],
        "observed_scheduler_steps": observed_scheduler_steps,
        "measurement_excluded": True,
        "protocol": profile_protocol,
        "captured_scheduler_window": {
            "start_step": (
                int(profile_protocol["skip_steps"])
                + int(profile_protocol["warmup_steps"])
            ),
            "end_step_exclusive": observed_scheduler_steps,
            "observed_active_steps": 4,
            "step_indices": list(
                range(observed_scheduler_steps - 4, observed_scheduler_steps)
            ),
            "prefill_steps": 0 if workload_class == "short_short" else 1,
            "decode_steps": 4,
            "mixed_steps": 0 if workload_class == "short_short" else 1,
        },
        "replay": {
            "scheduler_steps": observed_scheduler_steps,
            "wall_time_ms": duration_ms,
            "output_sha256": runs[0]["output_sha256"],
        },
    }
    return report


def fake_source_trace(
    request_ids: list[str],
    *,
    workload_class: str = "mixed",
) -> WorkloadTrace:
    trace = WorkloadTrace(
        workload_id=f"fake-{workload_class}",
        workload_class=workload_class,
        requests=tuple(
            RequestSpec(
                request_id=request_id,
                prompt="abc",
                config=GenerationConfig(max_new_tokens=7),
                arrival_time_ms=float(index),
            )
            for index, request_id in enumerate(request_ids)
        ),
        metadata={"purpose": "layout evidence tests"},
    )
    trace.validate()
    return trace


def fake_layout_reports(
    source_trace: WorkloadTrace,
    layout_id: str,
    *,
    tp_size: int,
    replica_count: int,
    duration_ms: float,
    mode: str = "closed_loop",
) -> list[dict[str, object]]:
    partitions, routing_assignments = partition_workload_by_projected_load(
        source_trace,
        replica_count=replica_count,
        max_slots_per_replica=8 // replica_count,
    )
    reports = []
    for replica_index, partition in enumerate(partitions):
        report = fake_layout_report(
            layout_id,
            tp_size=tp_size,
            replica_count=replica_count,
            replica_index=replica_index,
            request_ids=[request.request_id for request in partition.requests],
            routing_assignments=routing_assignments,
            duration_ms=duration_ms,
            workload_class=source_trace.workload_class,
            mode=mode,
        )
        report["workload"]["source_sha256"] = source_trace.request_sha256
        report["workload"]["request_sha256"] = partition.request_sha256
        if report["workload"]["partition"] is not None:
            report["workload"]["partition"][
                "source_sha256"
            ] = source_trace.request_sha256
        reports.append(report)
    return reports


def write_and_load_layout(
    root: Path,
    layout_id: str,
    reports: list[dict[str, object]],
    source_trace: WorkloadTrace,
) -> list[tuple[Path, dict[str, object]]]:
    layout_root = root / layout_id
    layout_root.mkdir(parents=True)
    source_path = layout_root / "source_workload.json"
    source_trace.save(source_path)
    source_payload = source_path.read_bytes()
    source_file_sha256 = hashlib.sha256(source_payload).hexdigest()
    entries = []
    for replica_index, report in enumerate(reports):
        report["workload"]["source_file_sha256"] = source_file_sha256
        report_path = layout_root / f"replica-{replica_index:02d}.json"
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        report_path.write_bytes(payload)
        entries.append(
            {
                "replica_index": replica_index,
                "name": report_path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    first = reports[0]
    distributed = first["distributed"]
    workload = first["workload"]
    manifest = {
        "schema_version": 1,
        "layout_id": layout_id,
        "tp_size": distributed["tp_size"],
        "replica_count": distributed["replica_count"],
        "global_world_size": distributed["global_world_size"],
        "global_logical_device_ids": distributed["global_logical_device_ids"],
        "source_workload_sha256": workload["source_sha256"],
        "source_workload_file_sha256": workload["source_file_sha256"],
        "workload_class": workload["workload_class"],
        "source_workload": {
            "name": source_path.name,
            "sha256": source_file_sha256,
            "size_bytes": len(source_payload),
        },
        "routing_assignment_sha256": workload["routing"]["assignment_sha256"],
        "reports": entries,
    }
    manifest_path = layout_root / "layout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded_layout_id, loaded = load_layout_manifest(manifest_path)
    assert loaded_layout_id == layout_id
    return loaded


def fake_telemetry() -> dict[str, object]:
    samples = []
    for repeat in range(3):
        start_ns = 1_000_000_000 + repeat * 2_000_000_000
        for device_id in range(8):
            for offset_ms in (10, 20):
                samples.append(
                    {
                        "timestamp_unix_ns": start_ns + offset_ms * 1_000_000,
                        "collection_latency_ms": 1.0,
                        "logical_device_id": device_id,
                        "npu_id": device_id // 2,
                        "chip_id": device_id % 2,
                        "memory_capacity_mb": 65_536.0,
                        "memory_usage_percent": 50.0,
                        "aicore_usage_percent": 75.0 + device_id,
                        "aicpu_usage_percent": 3.0,
                        "ctrlcpu_usage_percent": 4.0,
                        "memory_bandwidth_usage_percent": 40.0,
                        "aicore_rated_frequency_mhz": 1800.0,
                        "aicore_current_frequency_mhz": 1750.0,
                        "temperature_celsius": 42.0 + device_id,
                        "power_watts": 220.0 + device_id,
                    }
                )
    return {
        "schema_version": 1,
        "collector": "npu-smi_info_usages",
        "sample_interval_ms": 10.0,
        "targets": [
            {
                "logical_device_id": device_id,
                "npu_id": device_id // 2,
                "chip_id": device_id % 2,
            }
            for device_id in range(8)
        ],
        "samples": samples,
        "errors": [],
        "complete": True,
    }


def write_and_load_telemetry(root: Path, layout_id: str) -> dict[str, object]:
    path = root / layout_id / "telemetry.json"
    path.write_text(
        json.dumps(fake_telemetry(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return load_telemetry(path)


def check_npu_telemetry() -> None:
    parsed = parse_npu_smi_usages(
        """
        Memory Capacity(MB)            : 65536
        Memory Usage Rate(%)           : 48
        Aicore Usage Rate(%)           : 76
        Aicpu Usage Rate(%)            : 3
        Ctrlcpu Usage Rate(%)          : 4
        Memory Bandwidth Usage Rate(%) : 51
        """
    )
    assert parsed["memory_capacity_mb"] == 65_536.0
    assert parsed["aicore_usage_percent"] == 76.0
    common = parse_npu_smi_common(
        """
        NPU ID                         : 0
        Chip Count                     : 2
        Chip ID                        : 0
        HBM Usage Rate(%)              : 40
        AICore Usage Rate(%)           : 70
        Chip ID                        : 1
        HBM Usage Rate(%)              : 55
        Aicore Usage Rate(%)           : 81
        """,
        chip_id=1,
    )
    assert common["hbm_usage_percent"] == 55.0
    assert common["aicore_usage_percent"] == 81.0
    summary = summarize_telemetry(
        fake_telemetry(),
        run_intervals=[
            (1_000_000_000, 1_300_000_000),
            (3_000_000_000, 3_300_000_000),
            (5_000_000_000, 5_300_000_000),
        ],
        expected_logical_device_ids=list(range(8)),
        min_samples_per_device_per_run=2,
    )
    assert summary["all_runs_covered"] is True
    assert summary["source_file_verified"] is False
    assert summary["overall"]["aicore_usage_percent"]["count"] == 48
    assert summary["overall"]["aicore_current_frequency_mhz"]["median"] == 1750.0
    assert summary["overall"]["temperature_celsius"]["count"] == 48
    try:
        summarize_telemetry(
            fake_telemetry(),
            run_intervals=[
                (1_000_000_000, 1_300_000_000),
                (1_200_000_000, 1_400_000_000),
            ],
            expected_logical_device_ids=list(range(8)),
        )
    except ValueError as exc:
        assert "重叠" in str(exc)
    else:
        raise AssertionError("重叠 measured run 区间必须被拒绝")

    nonfinite = fake_telemetry()
    nonfinite["samples"][0]["memory_capacity_mb"] = float("nan")
    try:
        summarize_telemetry(
            nonfinite,
            run_intervals=[(1_000_000_000, 1_300_000_000)],
            expected_logical_device_ids=list(range(8)),
        )
    except ValueError as exc:
        assert "有限值" in str(exc)
    else:
        raise AssertionError("非有限 telemetry 数值必须被拒绝")


def check_layout_manifest_hash_gate() -> None:
    source_trace = fake_source_trace(["manifest-request"])
    report = fake_layout_reports(
        source_trace,
        "tp8",
        tp_size=8,
        replica_count=1,
        duration_ms=300.0,
    )[0]
    report["_layout_manifest_verified"] = True
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_path = root / "source_workload.json"
        source_trace.save(source_path)
        source_payload = source_path.read_bytes()
        source_file_sha256 = hashlib.sha256(source_payload).hexdigest()
        report["workload"]["source_file_sha256"] = source_file_sha256
        report_path = root / "replica-00.json"
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        report_path.write_bytes(payload)
        manifest_path = root / "layout_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layout_id": "tp8",
                    "tp_size": 8,
                    "replica_count": 1,
                    "global_world_size": 8,
                    "global_logical_device_ids": list(range(8)),
                    "source_workload_sha256": source_trace.request_sha256,
                    "source_workload_file_sha256": source_file_sha256,
                    "workload_class": source_trace.workload_class,
                    "source_workload": {
                        "name": source_path.name,
                        "sha256": source_file_sha256,
                        "size_bytes": len(source_payload),
                    },
                    "routing_assignment_sha256": report["workload"]["routing"][
                        "assignment_sha256"
                    ],
                    "reports": [
                        {
                            "replica_index": 0,
                            "name": report_path.name,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        layout_id, loaded = load_layout_manifest(manifest_path)
        assert layout_id == "tp8"
        assert loaded[0][1]["_layout_manifest_verified"] is not True

        directly_loaded = load_serving_report(report_path)
        assert "_layout_manifest_verified" not in directly_loaded

        source_path.write_bytes(source_payload + b"\n")
        try:
            load_layout_manifest(manifest_path)
        except ValueError as exc:
            assert "source workload" in str(exc)
        else:
            raise AssertionError("被篡改的源 workload 必须被 manifest 拒绝")
        source_path.write_bytes(source_payload)

        report_path.write_text("{}\n", encoding="utf-8")
        try:
            load_layout_manifest(manifest_path)
        except ValueError as exc:
            assert "size" in str(exc) or "SHA-256" in str(exc)
        else:
            raise AssertionError("被篡改的 replica report 必须被 manifest 拒绝")


def check_layout_comparison() -> None:
    request_ids = [f"layout-request-{index}" for index in range(8)]
    source_trace = fake_source_trace(request_ids)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        raw_layouts: dict[str, list[tuple[Path, dict[str, object]]]] = {}
        layouts: dict[str, list[tuple[Path, dict[str, object]]]] = {}
        for layout_id, replica_count, tp_size, duration_ms in (
            ("tp8", 1, 8, 800.0),
            ("2xtp4", 2, 4, 450.0),
            ("4xtp2", 4, 2, 300.0),
        ):
            raw_reports = fake_layout_reports(
                source_trace,
                layout_id,
                tp_size=tp_size,
                replica_count=replica_count,
                duration_ms=duration_ms,
            )
            raw_layouts[layout_id] = [
                (Path(f"{layout_id}-replica-{index}.json"), report)
                for index, report in enumerate(raw_reports)
            ]
            layouts[layout_id] = write_and_load_layout(
                root,
                layout_id,
                raw_reports,
                source_trace,
            )

        telemetry_by_layout = {
            layout_id: write_and_load_telemetry(root, layout_id)
            for layout_id in layouts
        }

        incomplete = summarize_serving_layouts(
            raw_layouts,
            baseline_layout="tp8",
        )
        assert incomplete["evidence_class"] == (
            "development_or_incomplete_layout_comparison"
        )
        assert incomplete["incomplete_reasons"]
        summary = summarize_serving_layouts(
            layouts,
            baseline_layout="tp8",
            telemetry_by_layout=telemetry_by_layout,
        )
        assert summary["evidence_class"] == (
            "formal_qwen3_32b_tp8_vs_2xtp4_vs_4xtp2"
        )
        assert summary["complete_layout_matrix"] is True
        rows = {row["layout_id"]: row for row in summary["rows"]}
        assert rows["4xtp2"]["speedup_vs_baseline"][
            "goodput_requests_per_second"
        ] > 1.0

        wrong_baseline = summarize_serving_layouts(
            layouts,
            baseline_layout="2xtp4",
            telemetry_by_layout=telemetry_by_layout,
        )
        assert wrong_baseline["evidence_class"] == (
            "development_or_incomplete_layout_comparison"
        )
        assert any(
            "tp8" in reason for reason in wrong_baseline["incomplete_reasons"]
        )

        original_runners = [
            report["engine"]["runner"] for _path, report in layouts["2xtp4"]
        ]
        for _path, report in layouts["2xtp4"]:
            report["engine"]["runner"] = "DifferentRunner"
        runner_mismatch = summarize_serving_layouts(
            layouts,
            baseline_layout="tp8",
            telemetry_by_layout=telemetry_by_layout,
        )
        assert runner_mismatch["same_scheduler_capacity"] is False
        assert runner_mismatch["evidence_class"] == (
            "development_or_incomplete_layout_comparison"
        )

        for (_path, report), runner in zip(
            layouts["2xtp4"],
            original_runners,
        ):
            report["engine"]["runner"] = runner

        layouts["2xtp4"][0][1]["protocol"][
            "open_loop_admission_scripted"
        ] = True
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "测量协议" in str(exc)
        else:
            raise AssertionError("脚本化与非脚本化准入报告不能混合比较")
        layouts["2xtp4"][0][1]["protocol"][
            "open_loop_admission_scripted"
        ] = False

        for reports in layouts.values():
            for _path, report in reports:
                report["model"]["full_parameter_count"] = 1
        forged_candidate = summarize_serving_layouts(
            layouts,
            baseline_layout="tp8",
            telemetry_by_layout=telemetry_by_layout,
        )
        assert forged_candidate["evidence_class"] == (
            "development_or_incomplete_layout_comparison"
        )
        for reports in layouts.values():
            for _path, report in reports:
                report["model"]["full_parameter_count"] = 32_762_123_264

        request = layouts["tp8"][0][1]["runs"][0]["serving"]["requests"][0]
        request["ttft_ms"] = 101.0
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "slo_met" in str(exc)
        else:
            raise AssertionError("伪造 request.slo_met 必须被重算门禁拒绝")
        request["ttft_ms"] = 10.0
        generated_ids = layouts["tp8"][0][1]["runs"][0]["serving"][
            "requests"
        ][0]["generated_ids"]
        generated_ids[0] = 999
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "output_sha256" in str(exc)
        else:
            raise AssertionError("逐请求输出被篡改后必须与 output digest 不一致")
        generated_ids[0] = 1

        request["ttft_ms"] = -1.0
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "ttft_ms" in str(exc)
        else:
            raise AssertionError("负延迟不能进入正式 layout comparison")
        request["ttft_ms"] = 10.0

        layouts["2xtp4"][1][1]["environment"]["precision"] = "fp16"
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "环境" in str(exc)
        else:
            raise AssertionError("不同精度的 layout 报告必须拒绝比较")
        layouts["2xtp4"][1][1]["environment"]["precision"] = "bf16"

        comparisons = {}
        for workload_class in (
            "short_short",
            "long_prefill_short_decode",
            "mixed",
        ):
            for mode in ("open_loop", "closed_loop"):
                acceptance_root = root / "acceptance" / f"{workload_class}-{mode}"
                acceptance_root.mkdir(parents=True)
                acceptance_request_ids = [
                    f"{workload_class}-request-{index}" for index in range(8)
                ]
                acceptance_trace = fake_source_trace(
                    acceptance_request_ids,
                    workload_class=workload_class,
                )
                acceptance_layouts = {}
                acceptance_telemetry = {}
                for layout_id, replica_count, tp_size, duration_ms in (
                    ("tp8", 1, 8, 800.0),
                    ("2xtp4", 2, 4, 450.0),
                    ("4xtp2", 4, 2, 300.0),
                ):
                    reports = fake_layout_reports(
                        acceptance_trace,
                        layout_id,
                        tp_size=tp_size,
                        replica_count=replica_count,
                        duration_ms=duration_ms,
                        mode=mode,
                    )
                    acceptance_layouts[layout_id] = write_and_load_layout(
                        acceptance_root,
                        layout_id,
                        reports,
                        acceptance_trace,
                    )
                    acceptance_telemetry[layout_id] = write_and_load_telemetry(
                        acceptance_root,
                        layout_id,
                    )
                comparison = summarize_serving_layouts(
                    acceptance_layouts,
                    baseline_layout="tp8",
                    telemetry_by_layout=acceptance_telemetry,
                )
                for row in comparison["rows"]:
                    for field in ("layout_manifest", "source_workload"):
                        artifact = row[field]
                        artifact["path"] = str(
                            Path(artifact["path"]).relative_to(acceptance_root)
                        )
                    telemetry_artifact = row["telemetry"]["source_artifact"]
                    telemetry_artifact["path"] = str(
                        Path(telemetry_artifact["path"]).relative_to(
                            acceptance_root
                        )
                    )
                path = acceptance_root / "comparison.json"
                path.write_text(
                    json.dumps(comparison, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                comparisons[(workload_class, mode)] = (path, comparison)
        acceptance = summarize_v07_acceptance(comparisons)
        assert acceptance["complete_matrix"] is True
        assert acceptance["evidence_class"] == (
            "formal_v0.7_qwen3_32b_ascend_continuous_batching_acceptance"
        )

        legacy_path, current_comparison = comparisons[("mixed", "closed_loop")]
        legacy_comparison = json.loads(json.dumps(current_comparison))
        del legacy_comparison["protocol"]["open_loop_admission_scripted"]
        legacy_path = legacy_path.parent / "legacy-comparison.json"
        legacy_path.write_text(
            json.dumps(legacy_comparison, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        legacy_acceptance = summarize_v07_acceptance(
            {("mixed", "closed_loop"): (legacy_path, legacy_comparison)}
        )
        assert legacy_acceptance["complete_matrix"] is False

        partial = summarize_v07_acceptance(
            {("mixed", "closed_loop"): comparisons[("mixed", "closed_loop")]}
        )
        assert partial["complete_matrix"] is False
        assert partial["incomplete_reasons"]

        forged = json.loads(json.dumps(summary))
        forged["rows"][0]["telemetry_formal"] = False
        forged_path = root / "forged-formal-comparison.json"
        forged_path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
        try:
            summarize_v07_acceptance(
                {("mixed", "closed_loop"): (forged_path, forged)}
            )
        except ValueError as exc:
            assert "telemetry_formal" in str(exc)
        else:
            raise AssertionError("只伪造 evidence_class 的 comparison 必须被拒绝")

        relabelled = json.loads(
            json.dumps(comparisons[("mixed", "closed_loop")][1])
        )
        relabelled["workload_class"] = "short_short"
        for row in relabelled["rows"]:
            row["workload_class"] = "short_short"
        relabelled_path = (
            comparisons[("mixed", "closed_loop")][0].parent
            / "relabelled-comparison.json"
        )
        relabelled_path.write_text(
            json.dumps(relabelled, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            summarize_v07_acceptance(
                {("short_short", "closed_loop"): (
                    relabelled_path,
                    relabelled,
                )}
            )
        except ValueError as exc:
            assert "重新计算结果不一致" in str(exc)
        else:
            raise AssertionError("重贴 workload 标签的 comparison 必须被拒绝")

        assignments = layouts["2xtp4"][0][1]["workload"]["routing"][
            "assignments"
        ]
        assignments[0]["request_id"] = "tampered-request"
        try:
            summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                telemetry_by_layout=telemetry_by_layout,
            )
        except ValueError as exc:
            assert "assignment_sha256" in str(exc)
        else:
            raise AssertionError("routing assignment 篡改必须被拒绝")

        acceptance_path, acceptance_comparison = comparisons[
            ("mixed", "closed_loop")
        ]
        telemetry_path = acceptance_path.parent / Path(
            acceptance_comparison["rows"][0]["telemetry"]["source_artifact"][
                "path"
            ]
        )
        telemetry_path.write_bytes(telemetry_path.read_bytes() + b"\n")
        try:
            summarize_v07_acceptance(comparisons)
        except ValueError as exc:
            assert "大小已变化" in str(exc) or "SHA-256 已变化" in str(exc)
        else:
            raise AssertionError("最终验收必须重新核验原始 telemetry 文件")


def main() -> None:
    check_workload_schema_and_presets()
    check_router_scoring_and_live_replicas()
    check_open_and_closed_loop_replay()
    check_projected_load_partitions()
    check_replay_benchmark_protocol()
    check_npu_telemetry()
    check_layout_manifest_hash_gate()
    check_layout_comparison()
    print("v0.7 workload replay and multi-replica routing tests passed.")


if __name__ == "__main__":
    main()
