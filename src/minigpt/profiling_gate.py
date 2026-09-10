"""v0.7.1 三组诊断比较与 v0.8 选题信号。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .ascend_profiling import analyze_profile_manifest
from .serving_layout import load_layout_manifest, summarize_serving_layout


PROFILE_POINT_SCHEMA_VERSION = 1
EXPECTED_CASES = {
    "short_decode_replica": {
        "workload_class": "short_short",
        "mode": "open_loop",
        "layouts": frozenset({"2xtp4", "4xtp2"}),
        "baseline_layout": "2xtp4",
        "question": "为什么 short request 下 4×TP2 低于 2×TP4？",
    },
    "long_prefill_scaling": {
        "workload_class": "long_prefill_short_decode",
        "mode": "closed_loop",
        "layouts": frozenset({"tp8", "4xtp2"}),
        "baseline_layout": "tp8",
        "question": "长 Prefill 为什么更偏向更多 replica？",
    },
    "mixed_overload": {
        "workload_class": "mixed",
        "mode": "open_loop",
        "layouts": frozenset({"tp8", "4xtp2"}),
        "baseline_layout": "tp8",
        "question": "mixed open-loop 中 4×TP2 的 4.24× goodput 来自哪里？",
    },
}

TRIAGE_THRESHOLDS = {
    "paged_kv_mean_active_internal_waste_ratio": 0.50,
    "paged_kv_peak_capacity_waste_ratio": 0.25,
    "decode_nonoverlapped_communication_ratio": 0.20,
    "decode_phase_fraction": 0.60,
    "long_prefill_phase_delta_vs_short": 0.15,
    "host_or_runtime_free_ratio": 0.15,
}

FORMAL_PROFILE_PROTOCOL = {
    "skip_steps": 8,
    "warmup_steps": 2,
    "active_steps": 4,
    "profiler_level": "level1",
    "aic_metrics": "pipe_utilization",
    "record_shapes": True,
    "profile_memory": False,
    "with_stack": False,
    "sys_interconnection": True,
}


def _artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def summarize_profile_point(
    layout_manifest_path: str | Path,
    profile_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    layout_path = Path(layout_manifest_path)
    profile_path = (
        Path(profile_manifest_path)
        if profile_manifest_path is not None
        else layout_path.with_name("profile_manifest.json")
    )
    try:
        raw_layout_manifest = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 layout manifest：{layout_path}") from exc
    layout_id, reports = load_layout_manifest(layout_path)
    row = summarize_serving_layout(
        layout_id,
        reports,
        max_start_skew_ms=100.0,
    )
    profile = analyze_profile_manifest(profile_path)
    first = reports[0][1]
    workload = first["workload"]
    protocol = first["protocol"]
    git = first["provenance"]["git"]
    incomplete_reasons: list[str] = []
    profile_artifact = _artifact(profile_path)
    expected_profile_ref = {
        "name": profile_path.name,
        "sha256": profile_artifact["sha256"],
        "size_bytes": profile_artifact["size_bytes"],
    }
    if raw_layout_manifest.get("profile") != expected_profile_ref:
        incomplete_reasons.append("layout manifest 未绑定当前 profile manifest")
    expected_identity = {
        "layout_id": layout_id,
        "workload_class": workload["workload_class"],
        "mode": protocol["mode"],
        "source_workload_sha256": workload["source_sha256"],
        "source_workload_file_sha256": workload["source_file_sha256"],
        "git_commit": git["commit"],
        "global_world_size": first["distributed"]["global_world_size"],
        "logical_device_ids": first["distributed"]["global_logical_device_ids"],
    }
    for field, expected in expected_identity.items():
        if profile.get(field) != expected:
            incomplete_reasons.append(
                f"profile {field} 与 serving layout 不一致"
            )
    expected_ranks = list(range(int(expected_identity["global_world_size"])))
    if profile.get("selected_ranks") != expected_ranks:
        incomplete_reasons.append("正式 profile point 必须覆盖全部 global ranks")
    profile_protocol = profile.get("protocol")
    if not isinstance(profile_protocol, dict):
        incomplete_reasons.append("profile manifest 缺少 protocol")
    else:
        for field, expected in FORMAL_PROFILE_PROTOCOL.items():
            if profile_protocol.get(field) != expected:
                incomplete_reasons.append(
                    f"正式 profile protocol 的 {field} 必须为 {expected!r}"
                )
    if profile.get("complete") is not True:
        incomplete_reasons.extend(str(reason) for reason in profile["incomplete_reasons"])
    profile_protocols = set()
    for path, report in reports:
        metadata = report.get("profiling")
        if not isinstance(metadata, dict):
            incomplete_reasons.append(f"{path.name} 缺少独立 profiling replay")
            continue
        if metadata.get("measurement_excluded") is not True:
            incomplete_reasons.append(f"{path.name} 把 profiling 混入 measured runs")
        if metadata.get("selected") is not True:
            incomplete_reasons.append(f"{path.name} 的 replica primary rank 未采集")
        if metadata.get("profile_manifest") != expected_profile_ref:
            incomplete_reasons.append(f"{path.name} 未绑定当前 profile manifest")
        replay = metadata.get("replay")
        measured_digests = {str(run["output_sha256"]) for run in report["runs"]}
        if not isinstance(replay, dict) or str(replay.get("output_sha256")) not in measured_digests:
            incomplete_reasons.append(f"{path.name} profiling replay 输出不一致")
        profile_protocols.add(
            json.dumps(metadata.get("protocol"), sort_keys=True, separators=(",", ":"))
        )
    expected_protocol_json = json.dumps(
        FORMAL_PROFILE_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(profile_protocols) != 1:
        incomplete_reasons.append("各 replica 的 profiling protocol 不一致")
    elif next(iter(profile_protocols)) != expected_protocol_json:
        incomplete_reasons.append("serving reports 未使用正式 profiling protocol")
    if row["scheduler_phases"].get("available") is not True:
        incomplete_reasons.append("serving report 缺少 scheduler phase wall-time")

    service = {
        "summary": row["summary"],
        "batching": row["batching"],
        "scheduler_phases": row["scheduler_phases"],
        "kv_cache": row["kv_cache"],
        "scheduler_capacity": row["scheduler_capacity"],
        "max_rank_peak_device_memory_mb": row["max_rank_peak_device_memory_mb"],
        "sum_rank_peak_device_memory_mb": row["sum_rank_peak_device_memory_mb"],
    }
    complete = not incomplete_reasons
    return {
        "schema_version": PROFILE_POINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.7.1_profiling_point",
        "evidence_class": (
            "complete_v0.7.1_profiling_point"
            if complete
            else "development_or_incomplete_v0.7.1_profiling_point"
        ),
        **expected_identity,
        "complete": complete,
        "incomplete_reasons": incomplete_reasons,
        "layout_manifest": _artifact(layout_path),
        "profile_manifest": profile_artifact,
        "service": service,
        "profile": profile,
    }


def _median(point: Mapping[str, object], path: tuple[str, ...]) -> float:
    value: object = point
    for key in path:
        if not isinstance(value, Mapping):
            raise ValueError(f"point 缺少字段：{'.'.join(path)}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"point 字段不是数值：{'.'.join(path)}")
    return float(value)


def _optional_median(
    point: Mapping[str, object],
    path: tuple[str, ...],
) -> float | None:
    try:
        return _median(point, path)
    except (KeyError, ValueError):
        return None


def summarize_profiling_gate(
    points: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    expected_keys = {
        (case_id, layout)
        for case_id, case in EXPECTED_CASES.items()
        for layout in case["layouts"]
    }
    actual_keys = set(points)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"profiling gate point 集合不完整：missing={missing}, extra={extra}")

    incomplete_reasons: list[str] = []
    cases: list[dict[str, object]] = []
    for case_id, case in EXPECTED_CASES.items():
        case_points = {
            layout: points[(case_id, layout)] for layout in case["layouts"]
        }
        source_digests: set[str] = set()
        git_commits: set[str] = set()
        rows: list[dict[str, object]] = []
        for layout in sorted(case_points):
            point = case_points[layout]
            if point.get("complete") is not True:
                incomplete_reasons.append(f"{case_id}/{layout} 不完整")
            for field, expected in (
                ("layout_id", layout),
                ("workload_class", case["workload_class"]),
                ("mode", case["mode"]),
            ):
                if point.get(field) != expected:
                    raise ValueError(f"{case_id}/{layout} 的 {field} 不匹配")
            source_digests.add(str(point["source_workload_file_sha256"]))
            git_commits.add(str(point["git_commit"]))
            rows.append(
                {
                    "layout_id": layout,
                    "goodput_requests_per_second": _median(
                        point,
                        ("service", "summary", "goodput_requests_per_second", "median"),
                    ),
                    "queue_p95_ms": _median(
                        point,
                        ("service", "summary", "queue_ms", "p95"),
                    ),
                    "decode_phase_fraction": _median(
                        point,
                        ("service", "scheduler_phases", "fractions", "decode_phase_ms"),
                    ),
                    "prefill_phase_fraction": _median(
                        point,
                        ("service", "scheduler_phases", "fractions", "prefill_phase_ms"),
                    ),
                    "kv_mean_active_internal_waste_ratio": _median(
                        point,
                        ("service", "kv_cache", "mean_active_internal_waste_ratio"),
                    ),
                    "kv_peak_capacity_waste_ratio": _median(
                        point,
                        (
                            "service",
                            "kv_cache",
                            "max_replica_peak_internal_waste_capacity_ratio",
                        ),
                    ),
                    "nonoverlapped_communication_fraction": _optional_median(
                        point,
                        (
                            "profile",
                            "aggregate_step_fractions",
                            "communication_not_overlapped",
                            "median",
                        ),
                    ),
                    "free_fraction": _optional_median(
                        point,
                        (
                            "profile",
                            "aggregate_step_fractions",
                            "free",
                            "median",
                        ),
                    ),
                }
            )
        if len(source_digests) != 1:
            incomplete_reasons.append(f"{case_id} 两个 layout 未复用同一 workload")
        if len(git_commits) != 1:
            incomplete_reasons.append(f"{case_id} 两个 layout 不在同一 commit")
        baseline = str(case["baseline_layout"])
        baseline_goodput = next(
            float(row["goodput_requests_per_second"])
            for row in rows
            if row["layout_id"] == baseline
        )
        for row in rows:
            row["goodput_vs_baseline"] = (
                float(row["goodput_requests_per_second"]) / baseline_goodput
                if baseline_goodput > 0.0
                else None
            )
        cases.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "workload_class": case["workload_class"],
                "mode": case["mode"],
                "baseline_layout": baseline,
                "rows": rows,
            }
        )

    flat_rows = [row for case in cases for row in case["rows"]]
    max_mean_waste = max(
        float(row["kv_mean_active_internal_waste_ratio"]) for row in flat_rows
    )
    max_peak_waste = max(
        float(row["kv_peak_capacity_waste_ratio"]) for row in flat_rows
    )
    short_rows = next(
        case["rows"] for case in cases if case["case_id"] == "short_decode_replica"
    )
    long_rows = next(
        case["rows"] for case in cases if case["case_id"] == "long_prefill_scaling"
    )
    max_decode_fraction = max(float(row["decode_phase_fraction"]) for row in short_rows)
    short_comm_values = [
        float(row["nonoverlapped_communication_fraction"])
        for row in short_rows
        if row["nonoverlapped_communication_fraction"] is not None
    ]
    max_short_comm = max(short_comm_values) if short_comm_values else None
    max_short_prefill = max(float(row["prefill_phase_fraction"]) for row in short_rows)
    max_long_prefill = max(float(row["prefill_phase_fraction"]) for row in long_rows)
    free_values = [
        float(row["free_fraction"])
        for row in flat_rows
        if row["free_fraction"] is not None
    ]
    max_free = max(free_values) if free_values else None
    signals = {
        "paged_kv_block_manager": {
            "triggered": (
                max_mean_waste
                >= TRIAGE_THRESHOLDS["paged_kv_mean_active_internal_waste_ratio"]
                or max_peak_waste
                >= TRIAGE_THRESHOLDS["paged_kv_peak_capacity_waste_ratio"]
            ),
            "observed": {
                "max_mean_active_internal_waste_ratio": max_mean_waste,
                "max_peak_capacity_waste_ratio": max_peak_waste,
            },
        },
        "decode_communication_path": {
            "triggered": (
                max_decode_fraction >= TRIAGE_THRESHOLDS["decode_phase_fraction"]
                and max_short_comm is not None
                and max_short_comm
                >= TRIAGE_THRESHOLDS["decode_nonoverlapped_communication_ratio"]
            ),
            "observed": {
                "max_short_decode_phase_fraction": max_decode_fraction,
                "max_short_nonoverlapped_communication_fraction": max_short_comm,
            },
        },
        "chunked_prefill_or_prefix_cache": {
            "triggered": (
                max_long_prefill - max_short_prefill
                >= TRIAGE_THRESHOLDS["long_prefill_phase_delta_vs_short"]
            ),
            "observed": {
                "max_long_prefill_phase_fraction": max_long_prefill,
                "max_short_prefill_phase_fraction": max_short_prefill,
                "delta": max_long_prefill - max_short_prefill,
            },
        },
        "host_scheduler_path": {
            "triggered": (
                max_free is not None
                and max_free >= TRIAGE_THRESHOLDS["host_or_runtime_free_ratio"]
            ),
            "observed": {"max_profile_free_fraction": max_free},
        },
        "speculative_decoding_or_mtp": {
            "triggered": (
                max_decode_fraction >= TRIAGE_THRESHOLDS["decode_phase_fraction"]
                and max_short_comm is not None
                and max_short_comm
                < TRIAGE_THRESHOLDS["decode_nonoverlapped_communication_ratio"]
            ),
            "requires_model_feasibility_gate": True,
            "observed": {
                "max_short_decode_phase_fraction": max_decode_fraction,
                "max_short_nonoverlapped_communication_fraction": max_short_comm,
            },
        },
    }
    complete = not incomplete_reasons
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.7.1_ascend_profiling_gate",
        "evidence_class": (
            "complete_v0.7.1_ascend_profiling_gate"
            if complete
            else "development_or_incomplete_v0.7.1_ascend_profiling_gate"
        ),
        "complete": complete,
        "selection_ready": complete,
        "incomplete_reasons": incomplete_reasons,
        "thresholds": dict(TRIAGE_THRESHOLDS),
        "cases": cases,
        "signals": signals,
        "decision_policy": (
            "signals 只负责证明瓶颈是否存在，不按命中数量自动选题；优先选择能用单一改动、"
            "同模型同 workload A/B 证明收益的方向。Speculative/MTP 还必须先通过 draft/MTP "
            "模型与 acceptance-rate 可行性门禁。"
        ),
    }
