"""v0.7.1 Profiler artifact、phase 指标与选题 gate 回归。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.ascend_profiling import (  # noqa: E402
    AscendProfileProtocol,
    AscendStepProfiler,
    analyze_profile_manifest,
    build_profile_manifest,
    load_profile_manifest,
    parse_profile_ranks,
    write_profile_manifest,
)
from minigpt.profiling_gate import (  # noqa: E402
    EXPECTED_CASES,
    FORMAL_PROFILE_PROTOCOLS,
    summarize_profile_point,
    summarize_profiling_gate,
)
from test_serving_workloads import (  # noqa: E402
    fake_layout_reports,
    fake_source_trace,
    write_and_load_layout,
)


def write_fake_profile_outputs(profile_root: Path) -> None:
    for rank in range(8):
        output = profile_root / f"rank_{rank:03d}" / "worker" / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        (output.parent / f"profiler_info_{rank}.json").write_text(
            json.dumps({"rank_id": rank}) + "\n",
            encoding="utf-8",
        )
        (output / "operator_details.csv").write_text(
            "Name,Host Self Duration (µs),Device Self Duration (µs)\n"
            "minigpt::decode_model,20,600\n"
            "minigpt::prefill_model,10,200\n",
            encoding="utf-8",
        )
        (output / "kernel_details.csv").write_text(
            "Name,Type,Accelerator Core,Duration(us)\n"
            "MatMul,AI_CORE,AI Core,600\n"
            "HcclAllReduce,HCCL,HCCL,250\n",
            encoding="utf-8",
        )
        (output / "step_trace_time.csv").write_text(
            "Device_id,Step,Computing,Communication (Not Overlapped),"
            "Overlapped,Communication,Free,Stage,Bubble,Preparing\n"
            f"{rank},ProfilerStep#10,600,250,50,300,150,1000,0,20\n",
            encoding="utf-8",
        )
        (output / "trace_view.json").write_text("[]\n", encoding="utf-8")
        (output / "communication.json").write_text("{}\n", encoding="utf-8")


def write_profile_manifest_for_layout(layout_root: Path) -> Path:
    layout_manifest_path = layout_root / "layout_manifest.json"
    layout_manifest = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
    profile_root = layout_root / "profiler"
    write_fake_profile_outputs(profile_root)
    workload_class = str(layout_manifest["workload_class"])
    manifest = build_profile_manifest(
        profile_root,
        layout_id=str(layout_manifest["layout_id"]),
        workload_class=workload_class,
        mode=json.loads(
            (layout_root / "replica-00.json").read_text(encoding="utf-8")
        )["protocol"]["mode"],
        source_workload_sha256=str(layout_manifest["source_workload_sha256"]),
        source_workload_file_sha256=str(
            layout_manifest["source_workload_file_sha256"]
        ),
        git_commit="c" * 40,
        selected_ranks=range(8),
        logical_device_ids=range(8),
        protocol=AscendProfileProtocol(
            **FORMAL_PROFILE_PROTOCOLS[workload_class]
        ),
    )
    path = layout_root / "profile_manifest.json"
    write_profile_manifest(path, manifest)
    profile_payload = path.read_bytes()
    profile_ref = {
        "name": path.name,
        "sha256": hashlib.sha256(profile_payload).hexdigest(),
        "size_bytes": len(profile_payload),
    }
    for report_ref in layout_manifest["reports"]:
        report_path = layout_root / report_ref["name"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["profiling"]["profile_manifest"] = profile_ref
        report_payload = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        report_path.write_bytes(report_payload)
        report_ref["sha256"] = hashlib.sha256(report_payload).hexdigest()
        report_ref["size_bytes"] = len(report_payload)
    layout_manifest["profile"] = profile_ref
    layout_manifest_path.write_text(
        json.dumps(layout_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def make_point(
    root: Path,
    *,
    case_id: str,
    layout_id: str,
) -> dict[str, object]:
    case = EXPECTED_CASES[case_id]
    tp_size, replica_count = {
        "tp8": (8, 1),
        "2xtp4": (4, 2),
        "4xtp2": (2, 4),
    }[layout_id]
    trace = fake_source_trace(
        [f"request-{index}" for index in range(8)],
        workload_class=str(case["workload_class"]),
    )
    reports = fake_layout_reports(
        trace,
        layout_id,
        tp_size=tp_size,
        replica_count=replica_count,
        duration_ms=100.0,
        mode=str(case["mode"]),
    )
    case_root = root / case_id
    write_and_load_layout(case_root, layout_id, reports, trace)
    layout_root = case_root / layout_id
    profile_path = write_profile_manifest_for_layout(layout_root)
    point = summarize_profile_point(
        layout_root / "layout_manifest.json",
        profile_path,
    )
    assert point["complete"] is True
    return point


def check_profile_manifest_and_parser(root: Path) -> None:
    point = make_point(root, case_id="short_decode_replica", layout_id="2xtp4")
    manifest_path = (
        root
        / "short_decode_replica"
        / "2xtp4"
        / "profile_manifest.json"
    )
    loaded = load_profile_manifest(manifest_path)
    assert loaded["complete"] is True
    summary = analyze_profile_manifest(manifest_path, top_k=1)
    assert summary["complete"] is True
    assert summary["evidence_class"] == "complete_v0.7.1_ascend_profile"
    assert summary["aggregate_step_fractions"][
        "communication_not_overlapped"
    ]["median"] == 0.25
    assert point["service"]["scheduler_phases"]["fractions"][
        "decode_phase_ms"
    ] == 0.6

    kernel_path = next(
        manifest_path.parent.glob(
            "profiler/rank_000/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
        )
    )
    kernel_path.write_text("tampered\n", encoding="utf-8")
    try:
        load_profile_manifest(manifest_path)
    except ValueError as exc:
        assert "哈希或大小不一致" in str(exc)
    else:
        raise AssertionError("被修改的 profile artifact 不应通过 manifest")


def check_complete_gate(root: Path) -> None:
    points: dict[tuple[str, str], dict[str, object]] = {}
    for case_id, case in EXPECTED_CASES.items():
        for layout_id in case["layouts"]:
            points[(case_id, layout_id)] = make_point(
                root,
                case_id=case_id,
                layout_id=str(layout_id),
            )
    summary = summarize_profiling_gate(points)
    assert summary["complete"] is True
    assert summary["selection_ready"] is True
    assert summary["evidence_class"] == "complete_v0.7.1_ascend_profiling_gate"
    assert summary["signals"]["paged_kv_block_manager"]["triggered"] is True
    assert summary["signals"]["decode_communication_path"]["triggered"] is True
    assert summary["signals"]["speculative_decoding_or_mtp"]["triggered"] is False

    for layout_id in EXPECTED_CASES["short_decode_replica"]["layouts"]:
        point = points[("short_decode_replica", str(layout_id))]
        point["complete"] = False
        point["profile"]["aggregate_step_fractions"][
            "communication_not_overlapped"
        ]["median"] = None
        point["profile"]["aggregate_step_fractions"]["free"]["median"] = None
    incomplete = summarize_profiling_gate(points)
    assert incomplete["complete"] is False
    assert incomplete["selection_ready"] is False


def check_profile_rank_parser() -> None:
    assert parse_profile_ranks("all", 4) == (0, 1, 2, 3)
    assert parse_profile_ranks("3, 1", 4) == (1, 3)
    for invalid in ("", "1,1", "4", "-1"):
        try:
            parse_profile_ranks(invalid, 4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 profile ranks 未被拒绝：{invalid!r}")


def check_captured_scheduler_window() -> None:
    profiler = AscendStepProfiler(
        "unused",
        global_rank=0,
        logical_device_id=0,
        selected=False,
        protocol=AscendProfileProtocol(
            skip_steps=1,
            warmup_steps=1,
            active_steps=2,
        ),
    )
    records = [
        {"prefill_batch_size": 4, "decode_batch_size": 0},
        {"prefill_batch_size": 0, "decode_batch_size": 4},
        {"prefill_batch_size": 2, "decode_batch_size": 4},
        {"prefill_batch_size": 0, "decode_batch_size": 4},
    ]
    with profiler:
        for record in records:
            profiler.step(record)
    window = profiler.metadata()["captured_scheduler_window"]
    assert window == {
        "start_step": 2,
        "end_step_exclusive": 4,
        "observed_active_steps": 2,
        "step_indices": [2, 3],
        "prefill_steps": 1,
        "decode_steps": 2,
        "mixed_steps": 1,
    }


def main() -> None:
    check_profile_rank_parser()
    check_captured_scheduler_window()
    with tempfile.TemporaryDirectory() as directory:
        check_profile_manifest_and_parser(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        check_complete_gate(Path(directory))
    print("v0.7.1 Ascend profiling gate tests passed.")


if __name__ == "__main__":
    main()
