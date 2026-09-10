"""v0.8 TP8 Decode 词表通信 A/B 的证据校验与结论汇总。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from .mixed_repro import _global_model_identity, _summarize_initial_npu_state
from .profiling_gate import summarize_profile_point
from .serving_layout import load_layout_manifest, summarize_serving_layout
from .serving_telemetry import load_telemetry, summarize_telemetry


DECODE_AB_SCHEMA_VERSION = 1
FORMAL_PATH_SEQUENCE = (
    "full_gather",
    "distributed_argmax",
    "distributed_argmax",
    "full_gather",
)
FORMAL_CASES = {
    "short_decode": {
        "workload_class": "short_short",
        "mode": "open_loop",
    },
    "mixed_decode": {
        "workload_class": "mixed",
        "mode": "open_loop",
    },
}
DECISION_THRESHOLDS = {
    "max_session_cv": 0.10,
    "minimum_completed_throughput_speedup": 1.03,
    "minimum_communication_fraction_reduction": 0.03,
    "maximum_goodput_regression": 0.02,
}


def _artifact(path: Path, *, relative_to: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("统计样本不能为空")
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
    }


def _cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return 0.0 if mean == 0.0 else statistics.pstdev(values) / abs(mean)


def _request_work_shape_sha256(report: Mapping[str, object]) -> str:
    requests = report["runs"][0]["serving"]["requests"]
    canonical = [
        {
            "request_id": request["request_id"],
            "state": request["state"],
            "stop_reason": request["stop_reason"],
            "input_tokens": request["input_tokens"],
            "output_tokens": request["output_tokens"],
        }
        for request in sorted(requests, key=lambda item: str(item["request_id"]))
    ]
    return _canonical_sha256(canonical)


def _profile_fraction(point: Mapping[str, object], field: str) -> float:
    metric = point["profile"]["aggregate_step_fractions"][field]
    return float(metric["median"])


def _load_session(
    root: Path,
    *,
    case_id: str,
    position: int,
    expected_path: str,
) -> dict[str, object]:
    session_id = f"session-{position:02d}-{expected_path}"
    session_dir = root / case_id / session_id
    manifest_path = session_dir / "layout_manifest.json"
    telemetry_before_path = session_dir / "telemetry_before.json"
    telemetry_path = session_dir / "telemetry.json"
    point = summarize_profile_point(manifest_path)
    layout_id, reports = load_layout_manifest(manifest_path)
    row = summarize_serving_layout(layout_id, reports, max_start_skew_ms=100.0)
    first = reports[0][1]
    telemetry_before = load_telemetry(telemetry_before_path)
    telemetry = load_telemetry(telemetry_path)
    initial_npu_state = _summarize_initial_npu_state(
        telemetry_before,
        expected_logical_device_ids=range(8),
    )
    measured_telemetry = summarize_telemetry(
        telemetry,
        run_intervals=[
            (int(run["started_at_unix_ns"]), int(run["ended_at_unix_ns"]))
            for run in row["runs"]
        ],
        expected_logical_device_ids=range(8),
        min_samples_per_device_per_run=2,
    )
    model_identity, local_parameter_count = _global_model_identity(
        first["model"],
        first["provenance"],
    )
    actual_rows: dict[str, int] = {}
    for _path, report in reports:
        for run in report["runs"]:
            paths = run["serving"]["token_selection"]["actual_rows_by_path"]
            for path, count in paths.items():
                actual_rows[str(path)] = actual_rows.get(str(path), 0) + int(count)
    protocol = dict(first["protocol"])
    configured_path = str(protocol.pop("greedy_token_path", ""))
    communication_models = {
        _canonical_sha256(report["engine"]["token_selection"]): report["engine"][
            "token_selection"
        ]
        for _path, report in reports
    }
    return {
        "session_id": session_id,
        "case_id": case_id,
        "position": position,
        "layout_id": layout_id,
        "configured_path": configured_path,
        "actual_rows_by_path": actual_rows,
        "protocol_without_path": protocol,
        "git": first["provenance"]["git"],
        "environment": first["environment"],
        "machine": {
            field: first["distributed"].get(field)
            for field in (
                "hostname",
                "backend",
                "global_world_size",
                "global_logical_device_ids",
                "physical_card_count",
                "chips_per_card",
                "interconnect_topology",
            )
        },
        "source_workload_sha256": first["workload"]["source_sha256"],
        "source_workload_file_sha256": first["workload"]["source_file_sha256"],
        "workload_class": first["workload"]["workload_class"],
        "routing_assignment_sha256": first["workload"]["routing"][
            "assignment_sha256"
        ],
        "model_identity_sha256": model_identity,
        "local_parameter_count": local_parameter_count,
        "output_sha256_by_replica": {
            str(report["distributed"]["replica_index"]): str(
                report["runs"][0]["output_sha256"]
            )
            for _path, report in reports
        },
        "output_work_shape_sha256_by_replica": {
            str(report["distributed"]["replica_index"]): (
                _request_work_shape_sha256(report)
            )
            for _path, report in reports
        },
        "initial_npu_state": initial_npu_state,
        "preflight_telemetry_complete": bool(telemetry_before["complete"]),
        "preflight_telemetry_errors": list(telemetry_before["errors"]),
        "measured_telemetry": measured_telemetry,
        "communication_model": next(iter(communication_models.values())),
        "communication_model_count": len(communication_models),
        "service": {
            "goodput_requests_per_second": row["summary"][
                "goodput_requests_per_second"
            ],
            "completed_requests_per_second": row["summary"][
                "completed_requests_per_second"
            ],
            "tpot_ms": row["summary"]["tpot_ms"],
            "runs": row["runs"],
        },
        "profile": {
            "communication_not_overlapped_fraction": _profile_fraction(
                point, "communication_not_overlapped"
            ),
            "free_fraction": _profile_fraction(point, "free"),
            "computing_fraction": _profile_fraction(point, "computing"),
        },
        "point_complete": point["complete"],
        "point_incomplete_reasons": point["incomplete_reasons"],
        "artifacts": {
            "layout_manifest": _artifact(manifest_path, relative_to=root),
            "telemetry_before": _artifact(
                telemetry_before_path,
                relative_to=root,
            ),
            "telemetry": _artifact(telemetry_path, relative_to=root),
            "profile_manifest": _artifact(
                session_dir / "profile_manifest.json",
                relative_to=root,
            ),
        },
    }


def _variant_map(
    sessions: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, list[str]]:
    variants: dict[str, list[str]] = {}
    for session in sessions:
        digest = _canonical_sha256(session[field])
        variants.setdefault(digest, []).append(str(session["session_id"]))
    return variants


def _aggregate_path(
    sessions: Sequence[Mapping[str, object]],
    path: str,
) -> dict[str, object]:
    selected = [session for session in sessions if session["configured_path"] == path]
    goodput = [
        float(session["service"]["goodput_requests_per_second"]["median"])
        for session in selected
    ]
    completed = [
        float(session["service"]["completed_requests_per_second"]["median"])
        for session in selected
    ]
    tpot = [float(session["service"]["tpot_ms"]["median"]) for session in selected]
    communication = [
        float(session["profile"]["communication_not_overlapped_fraction"])
        for session in selected
    ]
    free = [float(session["profile"]["free_fraction"]) for session in selected]
    return {
        "session_count": len(selected),
        "goodput_requests_per_second": _summary(goodput),
        "completed_requests_per_second": _summary(completed),
        "tpot_ms": _summary(tpot),
        "communication_not_overlapped_fraction": _summary(communication),
        "free_fraction": _summary(free),
        "session_goodput_cv": _cv(goodput),
    }


def _compare_case(
    sessions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paths = {
        path: _aggregate_path(sessions, path)
        for path in ("full_gather", "distributed_argmax")
    }
    baseline = paths["full_gather"]
    candidate = paths["distributed_argmax"]
    completed_speedup = (
        candidate["completed_requests_per_second"]["median"]
        / baseline["completed_requests_per_second"]["median"]
    )
    goodput_speedup = (
        candidate["goodput_requests_per_second"]["median"]
        / baseline["goodput_requests_per_second"]["median"]
    )
    tpot_reduction = 1.0 - (
        candidate["tpot_ms"]["median"] / baseline["tpot_ms"]["median"]
    )
    communication_reduction = (
        baseline["communication_not_overlapped_fraction"]["median"]
        - candidate["communication_not_overlapped_fraction"]["median"]
    )
    stable = all(
        float(row["session_goodput_cv"])
        <= DECISION_THRESHOLDS["max_session_cv"]
        for row in paths.values()
    )
    supported = bool(
        stable
        and completed_speedup
        >= DECISION_THRESHOLDS["minimum_completed_throughput_speedup"]
        and communication_reduction
        >= DECISION_THRESHOLDS["minimum_communication_fraction_reduction"]
        and goodput_speedup
        >= 1.0 - DECISION_THRESHOLDS["maximum_goodput_regression"]
    )
    regressed = bool(
        completed_speedup
        < 1.0 - DECISION_THRESHOLDS["maximum_goodput_regression"]
        or goodput_speedup
        < 1.0 - DECISION_THRESHOLDS["maximum_goodput_regression"]
    )
    return {
        "paths": paths,
        "completed_throughput_speedup": completed_speedup,
        "goodput_speedup": goodput_speedup,
        "tpot_reduction": tpot_reduction,
        "communication_fraction_absolute_reduction": communication_reduction,
        "stable": stable,
        "candidate_supported": supported,
        "candidate_regressed": regressed,
    }


def summarize_decode_ab(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    sessions = [
        _load_session(
            root_path,
            case_id=case_id,
            position=position,
            expected_path=path,
        )
        for case_id in FORMAL_CASES
        for position, path in enumerate(FORMAL_PATH_SEQUENCE, start=1)
    ]
    incomplete_reasons: list[str] = []
    warnings: list[str] = []
    if len({str(session["git"]["commit"]) for session in sessions}) != 1:
        incomplete_reasons.append("八个 session 的 Git commit 不一致")
    if any(bool(session["git"]["dirty"]) for session in sessions):
        incomplete_reasons.append("至少一个 session 使用 dirty tracked worktree")
    if len({str(session["model_identity_sha256"]) for session in sessions}) != 1:
        incomplete_reasons.append("八个 session 的模型或权重不一致")
    if len({_canonical_sha256(session["environment"]) for session in sessions}) != 1:
        incomplete_reasons.append("八个 session 的软件或 NPU runtime 不一致")
    if len({_canonical_sha256(session["machine"]) for session in sessions}) != 1:
        incomplete_reasons.append("八个 session 没有使用同一机器与拓扑")
    if any(session["layout_id"] != "tp8" for session in sessions):
        incomplete_reasons.append("所有 A/B session 必须使用 TP8")
    if any(session["point_complete"] is not True for session in sessions):
        incomplete_reasons.append("至少一个 Profiler point 不完整")
    if any(session["communication_model_count"] != 1 for session in sessions):
        incomplete_reasons.append("至少一个 session 的 rank 间通信模型不一致")
    if any(session["initial_npu_state"]["clean"] is not True for session in sessions):
        incomplete_reasons.append("至少一个 session 在 NPU 非空闲状态启动")
    if any(
        not session["preflight_telemetry_complete"]
        or session["preflight_telemetry_errors"]
        for session in sessions
    ):
        incomplete_reasons.append("至少一个 session 的 NPU preflight 采集不完整")
    if any(
        not (
            session["measured_telemetry"]["complete"]
            and session["measured_telemetry"]["source_file_verified"]
            and session["measured_telemetry"]["all_runs_covered"]
            and session["measured_telemetry"]["error_free"]
        )
        for session in sessions
    ):
        incomplete_reasons.append("至少一个 session 的 measured NPU telemetry 不完整")

    cases: list[dict[str, object]] = []
    for case_id, expected in FORMAL_CASES.items():
        rows = [session for session in sessions if session["case_id"] == case_id]
        if [row["configured_path"] for row in rows] != list(FORMAL_PATH_SEQUENCE):
            incomplete_reasons.append(f"{case_id} 未按 ABBA 顺序运行")
        if len({str(row["source_workload_file_sha256"]) for row in rows}) != 1:
            incomplete_reasons.append(f"{case_id} 没有复用同一 workload 文件")
        if len({str(row["routing_assignment_sha256"]) for row in rows}) != 1:
            incomplete_reasons.append(f"{case_id} 的 replica 路由分配不一致")
        if len({_canonical_sha256(row["protocol_without_path"]) for row in rows}) != 1:
            incomplete_reasons.append(f"{case_id} 除 A/B 路径外的协议不一致")
        if any(
            row["protocol_without_path"]["mode"] != expected["mode"]
            or row["workload_class"] != expected["workload_class"]
            for row in rows
        ):
            incomplete_reasons.append(f"{case_id} workload 或 mode 不匹配")
        for row in rows:
            actual_paths = set(row["actual_rows_by_path"])
            if actual_paths != {row["configured_path"]}:
                incomplete_reasons.append(
                    f"{case_id}/{row['session_id']} 实际 token 路径与配置不一致"
                )
        work_shapes = _variant_map(rows, "output_work_shape_sha256_by_replica")
        outputs = _variant_map(rows, "output_sha256_by_replica")
        if len(work_shapes) != 1:
            incomplete_reasons.append(f"{case_id} 跨 A/B 的请求终态或计算长度不一致")
        if len(outputs) != 1:
            warnings.append(f"{case_id} 跨 A/B 存在逐 token 数值变体")
        cases.append(
            {
                "case_id": case_id,
                "workload_class": expected["workload_class"],
                "mode": expected["mode"],
                "output_variants": outputs,
                "work_shape_variants": work_shapes,
                "comparison": _compare_case(rows),
            }
        )

    complete = not incomplete_reasons
    comparisons = [case["comparison"] for case in cases]
    if not complete:
        status = "incomplete"
        conclusion = "证据门禁未通过，不能判断词表通信优化是否值得继续。"
    elif any(row["candidate_regressed"] for row in comparisons):
        status = "candidate_regressed"
        conclusion = "候选路径出现端到端回退，当前实现不能进入扩展实验。"
    elif any(row["candidate_supported"] for row in comparisons):
        status = "candidate_supported"
        conclusion = (
            "至少一个 Decode 场景同时出现原始吞吐提升和非重叠通信下降，"
            "可以继续扩展布局与负载矩阵。"
        )
    else:
        status = "full_vocab_gather_not_end_to_end_bottleneck"
        conclusion = (
            "减少词表 collective 没有转化为足够端到端收益，应停止该优化并定位"
            "计算、Host launch 或同步空洞。"
        )
    return {
        "schema_version": DECODE_AB_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.8_decode_vocab_collective_ab",
        "complete": complete,
        "incomplete_reasons": incomplete_reasons,
        "warnings": warnings,
        "thresholds": dict(DECISION_THRESHOLDS),
        "design": {
            "layout_id": "tp8",
            "path_sequence_per_case": list(FORMAL_PATH_SEQUENCE),
            "cases": FORMAL_CASES,
        },
        "sessions": sessions,
        "cases": cases,
        "decision": {"status": status, "conclusion": conclusion},
    }


def write_markdown_report(summary: Mapping[str, object], output: str | Path) -> None:
    lines = [
        "# v0.8 Decode vocab collective A/B",
        "",
        f"- complete: `{str(summary['complete']).lower()}`",
        f"- status: `{summary['decision']['status']}`",
        f"- 结论：{summary['decision']['conclusion']}",
        "",
        "| case | raw throughput speedup | goodput speedup | TPOT reduction | "
        "communication reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        comparison = case["comparison"]
        lines.append(
            f"| {case['case_id']} | "
            f"{comparison['completed_throughput_speedup']:.4f}× | "
            f"{comparison['goodput_speedup']:.4f}× | "
            f"{comparison['tpot_reduction']:.2%} | "
            f"{comparison['communication_fraction_absolute_reduction']:.2%} |"
        )
    if summary["incomplete_reasons"]:
        lines.extend(["", "## Incomplete", ""])
        lines.extend(f"- {reason}" for reason in summary["incomplete_reasons"])
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
