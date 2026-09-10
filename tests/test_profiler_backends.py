"""Real CPU profiling, Kineto attribution, and legacy Ascend compatibility."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.backends.profiling import (  # noqa: E402
    ProfileProtocol,
    build_profile_manifest,
    create_step_profiler,
    load_profile_manifest,
    summarize_profile,
    write_profile_manifest,
)
from minigpt.ascend_profiling import (  # noqa: E402
    AscendProfileProtocol,
    analyze_profile_manifest as analyze_legacy_profile,
    build_profile_manifest as build_legacy_manifest,
)


def runtime(backend: str) -> SimpleNamespace:
    return SimpleNamespace(device=SimpleNamespace(type=backend))


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def manifest(root: Path, backend: str, protocol: ProfileProtocol, *, world_size: int = 1) -> Path:
    value = build_profile_manifest(
        root / "profiler", backend=backend, layout_id="test", workload_class="mixed",
        mode="open_loop", source_workload_sha256="a" * 64,
        source_workload_file_sha256="b" * 64, git_commit="c" * 40,
        selected_ranks=[0], logical_device_ids=list(range(world_size)), protocol=protocol,
    )
    path = root / "profile_manifest.json"
    write_profile_manifest(path, value)
    return path


def event(name: str, category: str, start: float, duration: float, **arguments: object) -> dict[str, object]:
    return {"name": name, "cat": category, "ph": "X", "ts": start, "dur": duration,
            "pid": 1 if category != "kernel" else 0, "tid": 1, "args": arguments}


def fixture(root: Path, backend: str = "cuda") -> tuple[Path, ProfileProtocol, list[dict[str, object]]]:
    protocol = ProfileProtocol(skip_steps=1, warmup_steps=1, active_steps=2)
    rank = root / "profiler" / "rank_000"
    write(rank / "capture.json", {
        "collector": "torch.profiler", "backend": backend, "global_rank": 0,
        "logical_device_id": 0, "selected": True, "capture_status": "complete",
        "trace_exports": 1, "protocol": asdict(protocol), "observed_scheduler_steps": 4,
        "captured_scheduler_window": {
            "start_step": 2, "end_step_exclusive": 4, "observed_active_steps": 2,
            "step_indices": [2, 3], "prefill_steps": 1, "decode_steps": 1, "mixed_steps": 0,
        },
    })
    events = [
        event("ProfilerStep#2", "user_annotation", 0, 100),
        event("ProfilerStep#3", "user_annotation", 100, 100),
        event("minigpt::decode_phase", "user_annotation", 0, 80),
        event("minigpt::prefill_phase", "user_annotation", 100, 90),
        event("cudaLaunchKernel", "cuda_runtime", 10, 2, correlation=1),
        event("cudaLaunchKernel", "cuda_runtime", 12, 2, correlation=2),
        event("cudaLaunchKernel", "cuda_runtime", 110, 1, correlation=3),
        event("ampere_sgemm", "kernel", 20, 50, correlation=1),
        event("ncclDevKernel_AllReduce", "kernel", 50, 40, correlation=2),
        event("ampere_sgemm", "kernel", 120, 60, correlation=3),
        event("Memcpy HtoD", "gpu_memcpy", 90, 5),
    ]
    if backend == "cpu":
        events = events[:4] + [event("gloo:all_reduce", "cpu_op", 10, 60)]
    write(rank / "trace.json", {"traceEvents": events, "displayTimeUnit": "ms"})
    return manifest(root, backend, protocol), protocol, events


def check_kineto_intervals_and_correlation(root: Path) -> None:
    path, _protocol, events = fixture(root)
    report = summarize_profile(path)
    assert report["complete"] is True, report["incomplete_reasons"]
    fractions = report["aggregate_step_fractions"]
    assert abs(fractions["communication_not_overlapped"]["median"] - 0.10) < 1e-12
    assert abs(fractions["computing"]["median"] - 0.55) < 1e-12
    assert abs(fractions["free"]["median"] - 0.325) < 1e-12
    assert abs(fractions["overlapped_of_communication"]["median"] - 0.5) < 1e-12
    rank = report["ranks"][0]
    assert rank["kernel"]["top"][0]["calls"] == 2
    assert rank["phase_attribution"]["mapped_kernel_count"] == 3
    assert rank["phase_attribution"]["phases"]["decode"]["device_duration_us"] == 70
    assert report["metric_capabilities"]["communication_not_overlapped"]["available"] is True
    # Kernel time overlap is insufficient to assign its originating phase.
    events[7]["args"] = {"correlation": 999}
    write(root / "profiler/rank_000/trace.json", {"traceEvents": events})
    manifest(root, "cuda", _protocol)
    uncorrelated = summarize_profile(path)
    assert uncorrelated["ranks"][0]["phase_attribution"]["unmapped_kernel_count"] == 1


def check_missing_metrics_and_invalid_evidence(root: Path) -> None:
    path, protocol, _events = fixture(root, "cpu")
    report = summarize_profile(path)
    assert report["complete"] is True
    assert all(value is None for value in report["aggregate_step_fractions"].values())
    assert report["ranks"][0]["kernel"]["total_duration_us"] is None
    assert report["metric_capabilities"]["communication_not_overlapped"]["available"] is False
    assert report["metric_capabilities"]["communication_not_overlapped"]["reason"]
    assert report["metric_capabilities"]["host_operator_time"]["available"] is True
    # Partial-rank collection is valid collection but not full-world evidence.
    manifest(root, "cpu", protocol, world_size=2)
    partial = summarize_profile(path)
    assert partial["collection_complete"] is True
    assert partial["complete"] is False and partial["all_ranks_covered"] is False
    manifest(root, "cpu", protocol)
    trace_path = root / "profiler/rank_000/trace.json"
    trace_path.write_text("tampered", encoding="utf-8")
    try:
        load_profile_manifest(path)
    except ValueError as exc:
        assert "hash or size mismatch" in str(exc)
    else:
        raise AssertionError("tampered trace was accepted")
    path, protocol, events = fixture(root, "cpu")
    events[0]["name"] = "ProfilerStep#9"
    write(trace_path, {"traceEvents": events})
    manifest(root, "cpu", protocol)
    mismatch = summarize_profile(path)
    assert mismatch["complete"] is False
    assert any("ProfilerStep" in reason for reason in mismatch["incomplete_reasons"])
    assert mismatch["metric_capabilities"]["communication_not_overlapped"]["available"] is False
    # A manifest may not delete required artifacts and then assert complete=True.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ranks"][0]["artifacts"] = [a for a in payload["ranks"][0]["artifacts"] if a["kind"] != "kineto_trace"]
    payload["complete"] = True
    write(path, payload)
    assert load_profile_manifest(path)["complete"] is False


def check_no_observed_collective(root: Path) -> None:
    path, protocol, events = fixture(root)
    events = [event for event in events if not str(event["name"]).startswith("nccl")]
    write(root / "profiler/rank_000/trace.json", {"traceEvents": events})
    manifest(root, "cuda", protocol)
    report = summarize_profile(path)
    assert report["complete"] is True
    assert report["aggregate_step_fractions"]["communication_not_overlapped"] is None
    assert report["ranks"][0]["kernel"]["communication_duration_us"] is None


def check_active_window_and_cpu_device_boundaries(root: Path) -> None:
    path, protocol, events = fixture(root)
    for entry in events:
        if str(entry["name"]).startswith("nccl"):
            entry["ts"] = 300
    write(root / "profiler/rank_000/trace.json", {"traceEvents": events})
    manifest(root, "cuda", protocol)
    report = summarize_profile(path)
    assert report["complete"] is True
    assert report["aggregate_step_fractions"]["communication_not_overlapped"] is None
    assert report["metric_capabilities"]["communication_not_overlapped"]["available"] is False
    assert report["ranks"][0]["kernel"]["outside_window_rows"] == 1
    # Even an inconsistent CPU trace containing GPU launches cannot enable
    # device phase metrics under a CPU-only collection contract.
    capture_path = root / "profiler/rank_000/capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["backend"] = "cpu"
    write(capture_path, capture)
    manifest(root, "cpu", protocol)
    cpu = summarize_profile(path)
    assert cpu["metric_capabilities"]["phase_device_time"]["available"] is False
    assert cpu["ranks"][0]["phase_attribution"]["mapped_kernel_count"] == 0
    assert all(value["device_duration_us"] is None for value in cpu["ranks"][0]["phase_attribution"]["phases"].values())


def check_legacy_ascend(root: Path) -> None:
    profile_root = root / "profiler"
    rank = profile_root / "rank_000" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    write(rank / "profiler_info_0.json", {"rank_id": 0})
    write(rank / "trace_view.json", [])
    write(rank / "communication.json", {})
    (rank / "operator_details.csv").write_text("Name,Device Self Duration (us)\nMatMul,600\n", encoding="utf-8")
    (rank / "kernel_details.csv").write_text("Name,Type,Duration(us)\nMatMul,AI_CORE,600\nHcclAllReduce,HCCL,250\n", encoding="utf-8")
    (rank / "step_trace_time.csv").write_text("Step,Computing,Communication (Not Overlapped),Free,Stage\nProfilerStep#0,600,250,150,1000\n", encoding="utf-8")
    legacy = build_legacy_manifest(
        profile_root, layout_id="test", workload_class="mixed", mode="open_loop",
        source_workload_sha256="a" * 64, source_workload_file_sha256="b" * 64,
        git_commit="c" * 40, selected_ranks=[0], logical_device_ids=[0],
        protocol=AscendProfileProtocol(skip_steps=0, warmup_steps=0, active_steps=1),
    )
    path = root / "profile_manifest.json"
    write_profile_manifest(path, legacy)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    old_report = analyze_legacy_profile(path)
    assert old_report["schema_version"] == 1 and old_report["complete"] is True
    assert old_report["aggregate_step_fractions"]["communication_not_overlapped"]["median"] == 0.25
    report = summarize_profile(path)
    assert report["schema_version"] == 2 and report["source_schema_version"] == 1
    assert report["complete"] is True
    assert report["metric_capabilities"]["phase_device_time"]["available"] is False
    assert "hashed only" in report["metric_capabilities"]["phase_device_time"]["source"]
    assert report["aggregate_step_fractions"]["overlapped_of_communication"] is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def check_protocol_and_unselected(root: Path) -> None:
    for values in ({"active_steps": 0}, {"skip_steps": -1}, {"active_steps": 1.5}, {"warmup_steps": True}):
        try:
            ProfileProtocol(**values).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid profile protocol accepted: {values}")
    protocol = ProfileProtocol(skip_steps=0, warmup_steps=0, active_steps=1)
    for backend in ("cpu", "cuda", "npu"):
        collector = create_step_profiler(runtime(backend), root / backend, global_rank=0, logical_device_id=0, selected=False, protocol=protocol)
        with collector:
            collector.step({"prefill_batch_size": 1, "decode_batch_size": 1})
        assert collector.metadata()["captured_scheduler_window"]["mixed_steps"] == 1
        assert not (root / backend).exists()
        try:
            collector.step({"prefill_batch_size": 0, "decode_batch_size": 1})
        except RuntimeError:
            pass
        else:
            raise AssertionError("a finished profiler accepted another step")
    assert "torch_npu" not in sys.modules


def check_real_cpu_capture(root: Path) -> None:
    import torch

    protocol = ProfileProtocol(skip_steps=1, warmup_steps=1, active_steps=3)
    collector = create_step_profiler(runtime("cpu"), root / "profiler", global_rank=0, logical_device_id=0, protocol=protocol)
    matrix = torch.ones((32, 32))
    with collector:
        for index in range(protocol.required_scheduler_steps + 1):
            with torch.profiler.record_function("minigpt::decode_phase"):
                result = torch.mm(matrix, matrix).sum().item()
            assert result > 0
            collector.step({"prefill_batch_size": 0, "decode_batch_size": 1})
    path = manifest(root, "cpu", protocol)
    report = summarize_profile(path)
    assert report["complete"] is True, report["incomplete_reasons"]
    assert report["ranks"][0]["step_trace"]["step_indices"] == [2, 3, 4]
    assert any(row["name"] == "minigpt::decode_phase" for row in report["ranks"][0]["host_annotations"])
    assert report["metric_capabilities"]["host_operator_time"]["available"] is True
    assert report["aggregate_step_fractions"]["communication_not_overlapped"] is None
    assert "torch_npu" not in sys.modules
    try:
        with collector:
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("a collector was started twice")


def check_cpu_exception_cleanup(root: Path) -> None:
    import torch

    protocol = ProfileProtocol(skip_steps=0, warmup_steps=0, active_steps=3)
    collector = create_step_profiler(runtime("cpu"), root / "profiler", global_rank=0, logical_device_id=0, protocol=protocol)
    try:
        with collector:
            torch.ones(4).sum()
            collector.step({"prefill_batch_size": 0, "decode_batch_size": 1})
            raise LookupError("injected replay failure")
    except LookupError:
        pass
    else:
        raise AssertionError("profiling suppressed the replay exception")
    assert collector.metadata()["capture_status"] == "failed"
    path = manifest(root, "cpu", protocol)
    assert summarize_profile(path)["complete"] is False
    # A failed capture must not leave the global Kineto profiler active.
    check_real_cpu_capture(root / "after_failure")

    short = create_step_profiler(runtime("cpu"), root / "short", global_rank=0, logical_device_id=0, protocol=protocol)
    try:
        with short:
            torch.ones(4).sum()
            short.step({"prefill_batch_size": 0, "decode_batch_size": 1})
    except RuntimeError as exc:
        assert "insufficient scheduler steps" in str(exc)
    else:
        raise AssertionError("a truncated successful replay was accepted as complete")
    assert short.metadata()["capture_status"] == "failed"


def check_ascend_failed_start_cleanup(root: Path) -> None:
    # This tests cleanup of the real adapter's lifecycle calls, not NPU hardware
    # collection. Its output is deliberately not sufficient for a valid manifest.
    class FailedSession:
        stopped = False

        def __enter__(self) -> None:
            raise RuntimeError("injected Ascend start failure")

        def __exit__(self, *args: object) -> None:
            self.stopped = True

    failed = FailedSession()
    fake_profiler = SimpleNamespace(
        AiCMetrics=SimpleNamespace(PipeUtilization=1, Memory=2, ArithmeticUtilization=3),
        ExportType=SimpleNamespace(Text=1), ProfilerLevel=SimpleNamespace(Level1=1),
        ProfilerActivity=SimpleNamespace(CPU=1, NPU=2),
        _ExperimentalConfig=lambda **kwargs: kwargs, schedule=lambda **kwargs: kwargs,
        tensorboard_trace_handler=lambda *args, **kwargs: None,
        profile=lambda **kwargs: failed,
    )
    protocol = ProfileProtocol(skip_steps=0, warmup_steps=0, active_steps=1)
    collector = create_step_profiler(runtime("npu"), root / "profiler", global_rank=0, logical_device_id=0, protocol=protocol)
    with patch.dict(sys.modules, {"torch_npu": SimpleNamespace(profiler=fake_profiler)}):
        try:
            with collector:
                raise AssertionError("failed profiler entered its replay")
        except RuntimeError as exc:
            assert "injected Ascend start failure" in str(exc)
    assert failed.stopped is True
    assert not build_profile_manifest(
        root / "profiler", backend="npu", layout_id="test", workload_class="mixed",
        mode="open_loop", source_workload_sha256="a" * 64,
        source_workload_file_sha256="b" * 64, git_commit="c" * 40,
        selected_ranks=[0], logical_device_ids=[0], protocol=protocol,
    )["complete"]


def check_service_replay_isolation(root: Path) -> None:
    from minigpt.serving_benchmark import benchmark_trace_replay
    from test_serving_workloads import ManualClock, build_engine, tiny_trace

    def run(clock: ManualClock, profiler: object | None = None) -> dict[str, object]:
        return benchmark_trace_replay(
            build_engine(clock, seed=31, max_slots=2), tiny_trace(arrival_interval_ms=1.0),
            mode="open_loop", closed_loop_clients=None, warmup=1, repeats=2,
            ttft_slo_ms=1000.0, tpot_slo_ms=1000.0, e2e_slo_ms=1000.0,
            clock=clock, sleeper=clock.advance, deterministic_open_loop=True,
            profiling_session=profiler,
        )

    baseline = run(ManualClock())
    clock = ManualClock()
    protocol = ProfileProtocol(skip_steps=0, warmup_steps=1, active_steps=2)
    collector = create_step_profiler(runtime("cpu"), root / "profiler", global_rank=0, logical_device_id=0, protocol=protocol)

    class DelayedCollector:
        def __enter__(self) -> "DelayedCollector":
            collector.__enter__()
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            collector.__exit__(exc_type, exc, traceback)

        def step(self, record: dict[str, object]) -> None:
            collector.step(record)
            clock.advance(1.0)

        def metadata(self) -> dict[str, object]:
            return collector.metadata()

    profiled = run(clock, DelayedCollector())
    assert len(profiled["runs"]) == len(baseline["runs"]) == 2
    assert profiled["summary"] == baseline["summary"]
    assert profiled["profiling"]["measurement_excluded"] is True
    assert profiled["profiling"]["replay"]["output_sha256"] == baseline["runs"][0]["output_sha256"]
    assert profiled["profiling"]["replay"]["wall_time_ms"] > profiled["summary"]["wall_time_ms"]["max"]
    assert summarize_profile(manifest(root, "cpu", protocol))["complete"] is True


def main() -> None:
    checks = [check_protocol_and_unselected, check_kineto_intervals_and_correlation,
              check_missing_metrics_and_invalid_evidence, check_no_observed_collective,
              check_active_window_and_cpu_device_boundaries,
              check_legacy_ascend, check_ascend_failed_start_cleanup,
              check_real_cpu_capture, check_cpu_exception_cleanup,
              check_service_replay_isolation]
    for check in checks:
        with tempfile.TemporaryDirectory() as directory:
            check(Path(directory))
    print("CPU/CUDA/Ascend profiler backend tests passed (CUDA trace fixtures; real CPU capture).")


if __name__ == "__main__":
    main()
