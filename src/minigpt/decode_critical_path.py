"""v0.8 TP8 Decode 词表通信 A/B 的证据校验与结论汇总。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from .mixed_repro import _global_model_identity
from .profiling_gate import summarize_profile_point
from .serving_layout import load_layout_manifest, summarize_serving_layout
from .serving_telemetry import load_telemetry, summarize_telemetry, validate_telemetry


DECODE_AB_SCHEMA_VERSION = 2
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
        "source_sha256": "380a9af8544252650e8464cf73b2a3f614170b1694c75d440abac3b43ec71fbf",
        "source_file_sha256": "aec029836323c65e98b8e9e0d87157e8f3c82fc2f27fef02829fdbfc09348439",
    },
    "mixed_decode": {
        "workload_class": "mixed",
        "mode": "open_loop",
        "source_sha256": "f376a6dcbfc41c465c023864a8a676672c539e9ac1086a0dd76db51e3f5668f8",
        "source_file_sha256": "0024963d8a4697a7ad3039a2560beb2be8efae07140b77a147f83a9c42a0f50a",
    },
}
DECISION_THRESHOLDS = {
    "max_session_cv": 0.10,
    "minimum_completed_throughput_speedup": 1.03,
    "minimum_communication_fraction_reduction": 0.03,
    "maximum_goodput_regression": 0.02,
}
FORMAL_PROTOCOL = {
    "mode": "open_loop",
    "open_loop_admission_scripted": True,
    "warmup": 1,
    "repeats": 3,
    "ttft_slo_ms": 15000.0,
    "tpot_slo_ms": 500.0,
    "e2e_slo_ms": 30000.0,
}
FORMAL_CAPACITY = {
    "runner": "SlotCachedTensorParallelQwen3ModelRunner",
    "total_max_slots": 32,
    "total_max_queue_size": 128,
    "max_seq_len": 4096,
}


def summarize_npu_preflight(
    telemetry: Mapping[str, object],
    *,
    expected_logical_device_ids: Sequence[int] = tuple(range(8)),
    min_samples_per_device: int = 3,
    min_window_ms: float = 400.0,
) -> dict[str, object]:
    """检查整个启动前窗口；采集失败、缺设备或单张快照均不能证明空闲。"""

    validate_telemetry(telemetry)
    if min_samples_per_device < 2 or min_window_ms <= 0:
        raise ValueError("preflight 需要至少两个样本和正的观察窗口")
    expected = {int(value) for value in expected_logical_device_ids}
    if not expected:
        raise ValueError("preflight devices 不能为空")
    reasons = []
    if {int(item["logical_device_id"]) for item in telemetry["targets"]} != expected:
        reasons.append("preflight targets 必须恰好覆盖指定 logical devices")
    if telemetry["complete"] is not True or telemetry["errors"]:
        reasons.append("preflight 采集未完整结束或存在查询错误")
    per_device = {}
    for device_id in sorted(expected):
        samples = [s for s in telemetry["samples"] if int(s["logical_device_id"]) == device_id]
        timestamps = {int(s["timestamp_unix_ns"]) for s in samples}
        window_ms = (max(timestamps) - min(timestamps)) / 1e6 if timestamps else 0.0
        if len(timestamps) < min_samples_per_device or window_ms < min_window_ms:
            reasons.append(f"device {device_id} 的 preflight 样本数量或时间窗口不足")
        per_device[str(device_id)] = {
            "distinct_sample_count": len(timestamps),
            "window_ms": window_ms,
            "hbm_usage_percent": _summary([float(s["hbm_usage_percent"]) for s in samples]) if samples else None,
            "aicore_usage_percent": _summary([float(s["aicore_usage_percent"]) for s in samples]) if samples else None,
        }
    overall = {
        field: _summary([float(s[field]) for s in telemetry["samples"]]) if telemetry["samples"] else None
        for field in ("hbm_usage_percent", "aicore_usage_percent")
    }
    for field, maximum in (("hbm_usage_percent", 10.0), ("aicore_usage_percent", 5.0)):
        if overall[field] is not None and overall[field]["max"] > maximum:
            reasons.append(f"preflight {field} 超过 {maximum}%")
    return {"clean": not reasons, "incomplete_reasons": reasons, "per_device": per_device, "overall": overall}


def _actual_token_rows(report: Mapping[str, object]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in report["runs"]:
        observed: dict[str, int] = {}
        for step in run["serving"]["steps"]:
            for phase in ("prefill", "decode"):
                count = step[f"{phase}_batch_size"]
                path = step[f"{phase}_token_selection_path"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("step token row count 必须是非负整数")
                if count == 0:
                    if path is not None:
                        raise ValueError("空 batch 不能记录实际 token 路径")
                    continue
                if path not in {"full_gather", "distributed_argmax"}:
                    raise ValueError("非空 batch 缺少有效的实际 token 路径")
                observed[path] = observed.get(path, 0) + count
        declared = run["serving"]["token_selection"]["actual_rows_by_path"]
        if observed != declared or not observed:
            raise ValueError("token_selection 汇总与逐 step 路径/行数不一致")
        if run["serving"]["token_selection"]["communication_model"] != report["engine"]["token_selection"]:
            raise ValueError("run 与 engine 的通信 payload 口径不一致")
        for path, count in observed.items():
            totals[path] = totals.get(path, 0) + count
    return totals


def _validate_communication_model(model: Mapping[str, object], path: str) -> None:
    if not isinstance(model, Mapping):
        raise ValueError("通信模型必须是包含 payload 口径的对象")
    expected = {
        "measurement_type": "estimate",
        "payload_scope": "collective_input_per_rank_per_row",
        "configured_greedy_path": path,
        "global_vocab_size": 151936,
        "local_vocab_size": 18992,
        "tp_size": 8,
        "logit_element_size_bytes": 2,
        "full_gather_input_bytes_per_rank_per_row": 37984,
        "distributed_argmax_input_bytes_per_rank_per_row": 8,
        "collective_input_reduction": 4748.0,
    }
    for field, value in expected.items():
        if model.get(field) != value:
            raise ValueError(f"通信模型 {field} 必须为 {value!r}，理论 payload 不能冒充实测")


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
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise ValueError("统计样本必须为有限非负值")
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
    status_path = session_dir / "session_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    session_start = int(status["started_at_unix_ns"])
    session_end = int(status["ended_at_unix_ns"])
    if status["exit_code"] != 0 or session_start <= 0 or session_end <= session_start:
        raise ValueError("session 没有成功结束或时间边界无效")
    point = summarize_profile_point(manifest_path)
    layout_id, reports = load_layout_manifest(manifest_path)
    row = summarize_serving_layout(layout_id, reports, max_start_skew_ms=100.0)
    first = reports[0][1]
    telemetry_before = load_telemetry(telemetry_before_path)
    telemetry = load_telemetry(telemetry_path)
    initial_npu_state = summarize_npu_preflight(
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
    for data in (telemetry_before, telemetry):
        if float(data["sample_interval_ms"]) != 200.0:
            raise ValueError("正式 A/B telemetry 的 interval 必须为 200 ms")
        if data["targets"] != [
            {"logical_device_id": d, "npu_id": d // 2, "chip_id": d % 2}
            for d in range(8)
        ]:
            raise ValueError("正式 A3 TP8 telemetry 设备映射不匹配")
        if any(not session_start <= int(s["timestamp_unix_ns"]) <= session_end for s in data["samples"]):
            raise ValueError("telemetry 时间戳位于 session 外")
    first_run_start = min(int(run["started_at_unix_ns"]) for run in row["runs"])
    last_run_end = max(int(run["ended_at_unix_ns"]) for run in row["runs"])
    if not session_start < first_run_start < last_run_end < session_end:
        raise ValueError("measured runs 不在 session 时间区间内")
    if any(int(s["timestamp_unix_ns"]) >= first_run_start for s in telemetry_before["samples"]):
        raise ValueError("preflight 不是在 measured runs 前采集")
    model_identity, local_parameter_count = _global_model_identity(
        first["model"],
        first["provenance"],
    )
    actual_rows: dict[str, int] = {}
    for _path, report in reports:
        for path, count in _actual_token_rows(report).items():
            actual_rows[path] = actual_rows.get(path, 0) + count
    protocol = dict(first["protocol"])
    configured_path = str(protocol.pop("greedy_token_path", ""))
    communication_models = {
        _canonical_sha256(report["engine"]["token_selection"]): report["engine"][
            "token_selection"
        ]
        for _path, report in reports
    }
    _validate_communication_model(next(iter(communication_models.values())), configured_path)
    errors = []
    for field, expected in FORMAL_PROTOCOL.items():
        if protocol.get(field) != expected:
            errors.append(f"正式协议 {field} 必须为 {expected!r}")
    for field, expected in FORMAL_CAPACITY.items():
        if row["scheduler_capacity"].get(field) != expected:
            errors.append(f"正式 scheduler capacity {field} 必须为 {expected!r}")
    if not row["all_replica_reports_formal_candidates"]:
        errors.append("未满足真实 Qwen3-32B/NPU/BF16/HCCL/权重哈希的正式条件")
    if first["distributed"]["global_logical_device_ids"] != list(range(8)):
        errors.append("正式 TP8 logical devices 必须为 0..7")
    if first["distributed"]["physical_card_count"] != 4 or first["distributed"]["chips_per_card"] != 2:
        errors.append("正式 A3 TP8 必须记录 4 张双芯物理卡")
    if not first["distributed"].get("interconnect_topology"):
        errors.append("缺少实际互连拓扑")
    expected_workload = FORMAL_CASES[case_id]
    for field in ("source_sha256", "source_file_sha256"):
        if first["workload"][field] != expected_workload[field]:
            errors.append(f"workload {field} 与冻结 v0.7 源 workload 不一致")
    return {
        "session_id": session_id,
        "case_id": case_id,
        "position": position,
        "loaded": True,
        "validation_errors": errors,
        "started_at_unix_ns": session_start,
        "ended_at_unix_ns": session_end,
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
            "scheduler_capacity": row["scheduler_capacity"],
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
            "session_status": _artifact(status_path, relative_to=root),
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
    def ratio(field: str) -> float | None:
        denominator = baseline[field]["median"]
        return candidate[field]["median"] / denominator if denominator > 0.0 else None

    completed_speedup = ratio("completed_requests_per_second")
    goodput_speedup = ratio("goodput_requests_per_second")
    tpot_ratio = ratio("tpot_ms")
    tpot_reduction = 1.0 - tpot_ratio if tpot_ratio is not None else None
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
        and completed_speedup is not None
        and completed_speedup
        >= DECISION_THRESHOLDS["minimum_completed_throughput_speedup"]
        and communication_reduction
        >= DECISION_THRESHOLDS["minimum_communication_fraction_reduction"]
        and candidate["goodput_requests_per_second"]["median"]
        >= baseline["goodput_requests_per_second"]["median"]
        * (1.0 - DECISION_THRESHOLDS["maximum_goodput_regression"])
    )
    regressed = bool(
        any(
            candidate[field]["median"] < baseline[field]["median"]
            * (1.0 - DECISION_THRESHOLDS["maximum_goodput_regression"])
            for field in ("completed_requests_per_second", "goodput_requests_per_second")
        )
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
        "ratio_unavailable_reasons": [
            f"baseline {field} 为 0，倍率未定义"
            for field in ("completed_requests_per_second", "goodput_requests_per_second", "tpot_ms")
            if baseline[field]["median"] == 0.0
        ],
    }


def summarize_decode_ab(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    incomplete_reasons: list[str] = []
    warnings: list[str] = []
    all_sessions: list[dict[str, object]] = []
    for case_id in FORMAL_CASES:
        for position, path in enumerate(FORMAL_PATH_SEQUENCE, start=1):
            session_id = f"session-{position:02d}-{path}"
            try:
                session = _load_session(root_path, case_id=case_id, position=position, expected_path=path)
            except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                reason = f"{case_id}/{session_id} 无法校验证据：{type(exc).__name__}: {exc}"
                incomplete_reasons.append(reason)
                session = {"case_id": case_id, "session_id": session_id, "position": position, "loaded": False, "validation_errors": [reason]}
            all_sessions.append(session)
    sessions = [session for session in all_sessions if session["loaded"]]
    for session in sessions:
        incomplete_reasons.extend(f"{session['case_id']}/{session['session_id']}: {reason}" for reason in session["validation_errors"])
        incomplete_reasons.extend(f"{session['case_id']}/{session['session_id']}: {reason}" for reason in session["point_incomplete_reasons"])
    for previous, current in zip(sessions, sessions[1:]):
        if int(current["started_at_unix_ns"]) <= int(previous["ended_at_unix_ns"]):
            incomplete_reasons.append("session 时间戳未按实际 ABBA 顺序串行执行，或存在重叠")
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
        if len(rows) != 4:
            incomplete_reasons.append(f"{case_id} 缺少可校验的四个独立 session")
            cases.append({"case_id": case_id, "workload_class": expected["workload_class"], "mode": expected["mode"], "comparison": None})
            continue
        if [row["configured_path"] for row in rows] != list(FORMAL_PATH_SEQUENCE):
            incomplete_reasons.append(f"{case_id} 未按 ABBA 顺序运行")
        if any(sum(row["configured_path"] == path for row in rows) != 2
               for path in ("full_gather", "distributed_argmax")):
            incomplete_reasons.append(f"{case_id} 每条路径必须恰好有两个独立 session")
            cases.append({"case_id": case_id, "workload_class": expected["workload_class"],
                          "mode": expected["mode"], "comparison": None})
            continue
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
        comparison = _compare_case(rows)
        if not comparison["stable"]:
            incomplete_reasons.append(f"{case_id} 同路径跨 session goodput CV 超过 10%，证据不稳定")
        if comparison["completed_throughput_speedup"] is None:
            incomplete_reasons.append(f"{case_id} 基线无完成请求，无法判断端到端吞吐收益")
        warnings.extend(f"{case_id}: {reason}" for reason in comparison["ratio_unavailable_reasons"])
        cases.append(
            {
                "case_id": case_id,
                "workload_class": expected["workload_class"],
                "mode": expected["mode"],
                "output_variants": outputs,
                "work_shape_variants": work_shapes,
                "comparison": comparison,
            }
        )

    complete = not incomplete_reasons
    comparisons = [case["comparison"] for case in cases if case["comparison"] is not None]
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
        "sessions": all_sessions,
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
        if comparison is None:
            lines.append(f"| {case['case_id']} | N/A | N/A | N/A | N/A |")
            continue

        def display(value: float | None, *, percent: bool = False) -> str:
            if value is None:
                return "N/A"
            return f"{value:.2%}" if percent else f"{value:.4f}×"

        lines.append(
            f"| {case['case_id']} | "
            f"{display(comparison['completed_throughput_speedup'])} | "
            f"{display(comparison['goodput_speedup'])} | "
            f"{display(comparison['tpot_reduction'], percent=True)} | "
            f"{display(comparison['communication_fraction_absolute_reduction'], percent=True)} |"
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
