"""v0.8 Decode 词表通信 A/B 决策口径测试。"""

from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import json
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.decode_critical_path import (  # noqa: E402
    FORMAL_CAPACITY, FORMAL_CASES, FORMAL_PATH_SEQUENCE, FORMAL_PROTOCOL,
    _compare_case, _load_session, summarize_decode_ab,
    summarize_npu_preflight, write_markdown_report,
)
from minigpt.workload import WorkloadTrace  # noqa: E402
from test_serving_workloads import fake_layout_reports, fake_telemetry  # noqa: E402
from test_profiling_gate import write_profile_manifest_for_layout  # noqa: E402


def fake_session(
    path: str,
    goodput: float,
    completed: float,
    tpot_ms: float,
    communication: float,
    free: float,
) -> dict[str, object]:
    return {
        "configured_path": path,
        "service": {
            "goodput_requests_per_second": {"median": goodput},
            "completed_requests_per_second": {"median": completed},
            "tpot_ms": {"median": tpot_ms},
        },
        "profile": {
            "communication_not_overlapped_fraction": communication,
            "free_fraction": free,
        },
    }


def paired_sessions(
    baseline: tuple[float, float, float, float, float],
    candidate: tuple[float, float, float, float, float],
) -> list[dict[str, object]]:
    return [
        fake_session("full_gather", *baseline),
        fake_session("distributed_argmax", *candidate),
        fake_session("distributed_argmax", *candidate),
        fake_session("full_gather", *baseline),
    ]


def check_supported_candidate() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (9.4, 10.5, 108.0, 0.27, 0.25),
        )
    )
    assert comparison["candidate_supported"] is True
    assert comparison["candidate_regressed"] is False
    assert comparison["completed_throughput_speedup"] == 1.05
    assert comparison["communication_fraction_absolute_reduction"] > 0.07


def check_candidate_without_end_to_end_gain() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (9.0, 10.1, 119.0, 0.20, 0.35),
        )
    )
    assert comparison["candidate_supported"] is False
    assert comparison["candidate_regressed"] is False


def check_regression() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (8.5, 9.5, 130.0, 0.30, 0.25),
        )
    )
    assert comparison["candidate_supported"] is False
    assert comparison["candidate_regressed"] is True


def check_zero_and_unstable_metrics() -> None:
    rows = paired_sessions((0.0, 10.0, 0.0, 0.35, 0.2), (0.0, 10.5, 0.0, 0.25, 0.2))
    result = _compare_case(rows)
    assert result["goodput_speedup"] is None
    assert result["tpot_reduction"] is None
    assert result["candidate_supported"] is True
    assert result["candidate_regressed"] is False
    # A zero baseline cannot define a multiplicative raw throughput improvement.
    zero = _compare_case(paired_sessions((0, 0, 0, .35, .2), (0, 0, 0, .25, .2)))
    assert zero["completed_throughput_speedup"] is None
    assert zero["candidate_supported"] is False
    rows[0]["service"]["goodput_requests_per_second"]["median"] = 4.0
    assert _compare_case(rows)["stable"] is False
    # Relative regressions still have a valid definition when the candidate is zero.
    result = _compare_case(paired_sessions((1, 10, 2, .35, .2), (0, 11, 2, .25, .2)))
    assert result["candidate_regressed"] is True


def clean_preflight(start_ns: int = 100_000_000_000) -> dict[str, object]:
    report = fake_telemetry()
    template = report["samples"][0]
    report["sample_interval_ms"] = 200.0
    report["samples"] = [
        {**template, "logical_device_id": device, "npu_id": device // 2,
         "chip_id": device % 2, "timestamp_unix_ns": start_ns + offset,
         "hbm_usage_percent": 4.0, "aicore_usage_percent": 0.0}
        for offset in (100_000_000, 350_000_000, 600_000_000)
        for device in range(8)
    ]
    return report


def check_preflight_window() -> None:
    report = clean_preflight()
    assert summarize_npu_preflight(report)["clean"] is True
    late_load = copy.deepcopy(report)
    late_load["samples"][-1]["aicore_usage_percent"] = 40.0
    assert summarize_npu_preflight(late_load)["clean"] is False
    for field, value in (("complete", False), ("errors", [{"error": "timeout"}])):
        bad = {**report, field: value}
        assert summarize_npu_preflight(bad)["clean"] is False
    one_sample = {**report, "samples": report["samples"][:8]}
    assert summarize_npu_preflight(one_sample)["clean"] is False
    duplicate = {**report, "samples": report["samples"][:8] * 3}
    assert summarize_npu_preflight(duplicate)["clean"] is False
    missing = {**report, "samples": [s for s in report["samples"] if s["logical_device_id"] != 7]}
    assert summarize_npu_preflight(missing)["clean"] is False


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reference(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"name": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def write_session(root: Path, case_id: str, position: int, path: str, serial: int) -> Path:
    """合成测量 fixture；只在临时目录使用，源请求取冻结 workload 的真实字节。"""
    directory = root / case_id / f"session-{position:02d}-{path}"
    directory.mkdir(parents=True)
    workload_class = FORMAL_CASES[case_id]["workload_class"]
    with tarfile.open(ROOT / "v0.7_ascend_evidence.tar.gz") as archive:
        payload = archive.extractfile(f"runs/v07/workloads/{workload_class}.json").read()
    source_path = directory / "source_workload.json"
    source_path.write_bytes(payload)
    trace = WorkloadTrace.load(source_path)
    candidate = path == "distributed_argmax"
    report = fake_layout_reports(trace, "tp8", tp_size=8, replica_count=1,
                                 duration_ms=900.0 if candidate else 1000.0, mode="open_loop")[0]
    report["protocol"].update(FORMAL_PROTOCOL, greedy_token_path=path)
    report["engine"].update(max_slots=32, max_queue_size=128)
    communication = {
        "measurement_type": "estimate", "payload_scope": "collective_input_per_rank_per_row",
        "configured_greedy_path": path, "global_vocab_size": 151936, "local_vocab_size": 18992,
        "tp_size": 8, "logit_element_size_bytes": 2,
        "full_gather_input_bytes_per_rank_per_row": 37984,
        "distributed_argmax_input_bytes_per_rank_per_row": 8, "collective_input_reduction": 4748.0,
    }
    report["engine"]["token_selection"] = communication
    report["model"]["local_parameter_count"] = 4_095_265_408
    report["provenance"]["index_sha256"] = "a" * 64
    report["workload"]["source_file_sha256"] = hashlib.sha256(payload).hexdigest()
    start_ns = 100_000_000_000 + serial * 20_000_000_000
    telemetry = clean_preflight(start_ns)
    telemetry["samples"] = []
    for repeat, run in enumerate(report["runs"]):
        run_start = start_ns + 2_000_000_000 + repeat * 2_000_000_000
        duration_ns = 900_000_000 if candidate else 1_000_000_000
        run["replay"].update(started_at_unix_ns=run_start, ended_at_unix_ns=run_start + duration_ns)
        serving = run["serving"]
        serving["kv_cache"]["slot_capacity"] = 32
        for step in serving["steps"]:
            step.update(prefill_token_selection_path=path, decode_token_selection_path=path)
        serving["token_selection"] = {"actual_rows_by_path": {path: 2 * len(trace.requests)},
                                      "communication_model": communication}
        for sample in clean_preflight(run_start)["samples"][:16]:
            telemetry["samples"].append(sample)
    write_json(directory / "telemetry_before.json", clean_preflight(start_ns))
    write_json(directory / "telemetry.json", telemetry)
    write_json(directory / "session_status.json", {"started_at_unix_ns": start_ns,
               "ended_at_unix_ns": start_ns + 10_000_000_000, "exit_code": 0})
    report_path = directory / "replica-00.json"
    write_json(report_path, report)
    manifest = {"schema_version": 1, "layout_id": "tp8", "tp_size": 8,
                "replica_count": 1, "global_world_size": 8, "global_logical_device_ids": list(range(8)),
                "source_workload_sha256": trace.request_sha256,
                "source_workload_file_sha256": report["workload"]["source_file_sha256"],
                "workload_class": workload_class, "source_workload": reference(source_path),
                "routing_assignment_sha256": report["workload"]["routing"]["assignment_sha256"],
                "reports": [{"replica_index": 0, **reference(report_path)}]}
    write_json(directory / "layout_manifest.json", manifest)
    write_profile_manifest_for_layout(directory)
    return directory


def rewrite_report(directory: Path, mutate) -> None:
    path = directory / "replica-00.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    write_json(path, report)
    manifest_path = directory / "layout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"] = [{"replica_index": 0, **reference(path)}]
    write_json(manifest_path, manifest)


def check_evidence_gate(root: Path) -> None:
    directories = []
    for case_id in FORMAL_CASES:
        for position, path in enumerate(FORMAL_PATH_SEQUENCE, start=1):
            directories.append(write_session(root, case_id, position, path, len(directories)))
    summary = summarize_decode_ab(root)
    assert summary["complete"], summary["incomplete_reasons"]
    assert summary["decision"]["status"] == "full_vocab_gather_not_end_to_end_bottleneck"
    assert summary["sessions"][0]["service"]["scheduler_capacity"]["runner"] == (
        FORMAL_CAPACITY["runner"]
    )
    first = directories[0]
    report_path = first / "replica-00.json"
    manifest_path = first / "layout_manifest.json"
    original_report, original_manifest = report_path.read_bytes(), manifest_path.read_bytes()
    mutations = (
        lambda r: r["protocol"].update(ttft_slo_ms=14000),
        lambda r: r["engine"].update(max_queue_size=127),
        # The report contract uses implementation_name. Reintroducing the
        # Python class name must fail instead of letting synthetic fixtures
        # diverge from real serving reports again.
        lambda r: r["engine"].update(
            runner="SlotCachedTensorParallelQwen3ModelRunner"
        ),
        lambda r: r["runs"][0]["serving"]["token_selection"]["actual_rows_by_path"].update(full_gather=1),
        lambda r: r["engine"]["token_selection"].update(measurement_type="measured"),
        lambda r: r["engine"].update(token_selection=None),
        lambda r: r["environment"].update(precision="fp32"),
        lambda r: r["provenance"]["git"].update(dirty=True),
    )
    for mutate in mutations:
        rewrite_report(first, mutate)
        invalid = summarize_decode_ab(root)
        assert invalid["complete"] is False
        assert invalid["decision"]["status"] == "incomplete"
        report_path.write_bytes(original_report)
        manifest_path.write_bytes(original_manifest)
    # Mutating a report without rebinding its hash must be rejected.
    report_path.write_bytes(original_report + b" ")
    assert not summarize_decode_ab(root)["complete"]
    report_path.write_bytes(original_report)
    status_path = directories[1] / "session_status.json"
    original_status = status_path.read_bytes()
    status = json.loads(original_status)
    status["started_at_unix_ns"] -= 20_000_000_000
    write_json(status_path, status)
    invalid = summarize_decode_ab(root)
    assert not invalid["complete"]
    assert any("ABBA" in r for r in invalid["incomplete_reasons"])
    status_path.write_bytes(original_status)
    saved = [(d, (d / "replica-00.json").read_bytes(), (d / "layout_manifest.json").read_bytes())
             for d in directories[1:3]]
    def relabel_candidate(report):
        report["protocol"]["greedy_token_path"] = "full_gather"
        report["engine"]["token_selection"]["configured_greedy_path"] = "full_gather"
        for run in report["runs"]:
            selection = run["serving"]["token_selection"]
            count = selection["actual_rows_by_path"]["distributed_argmax"]
            selection["actual_rows_by_path"] = {"full_gather": count}
            selection["communication_model"]["configured_greedy_path"] = "full_gather"
            for step in run["serving"]["steps"]:
                step.update(prefill_token_selection_path="full_gather", decode_token_selection_path="full_gather")
    for directory, _, _ in saved:
        rewrite_report(directory, relabel_candidate)
    missing_path = summarize_decode_ab(root)
    assert not missing_path["complete"]
    assert missing_path["cases"][0]["comparison"] is None
    for directory, report_bytes, manifest_bytes in saved:
        (directory / "replica-00.json").write_bytes(report_bytes)
        (directory / "layout_manifest.json").write_bytes(manifest_bytes)
    # A slow second same-path session is insufficient evidence, not a negative conclusion.
    def slow_session(report):
        for run in report["runs"]:
            run["replay"]["ended_at_unix_ns"] += 600_000_000
    rewrite_report(directories[3], slow_session)
    invalid = summarize_decode_ab(root)
    assert not invalid["complete"]
    assert any("CV" in r for r in invalid["incomplete_reasons"])
    write_markdown_report(invalid, root / "unstable.md")
    # Missing and malformed artifacts still leave an auditable report.
    report_path.unlink()
    invalid = summarize_decode_ab(root)
    assert not invalid["complete"] and len(invalid["sessions"]) == 8
    write_markdown_report(invalid, root / "missing.md")
    report_path.write_text("not json", encoding="utf-8")
    assert not summarize_decode_ab(root)["complete"]


def main() -> None:
    check_supported_candidate()
    check_candidate_without_end_to_end_gain()
    check_regression()
    check_zero_and_unstable_metrics()
    check_preflight_window()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        empty = summarize_decode_ab(root)
        assert not empty["complete"] and len(empty["sessions"]) == 8
        write_markdown_report(empty, root / "empty.md")
        check_evidence_gate(root)
    print("v0.8 decode critical-path decision tests passed.")


if __name__ == "__main__":
    main()
