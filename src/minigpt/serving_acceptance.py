"""汇总 v0.7 三类 workload × open/closed loop 的最终验收矩阵。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .serving_layout import load_layout_manifest, summarize_serving_layouts
from .serving_telemetry import load_telemetry


FORMAL_LAYOUT_COMPARISON = "formal_qwen3_32b_tp8_vs_2xtp4_vs_4xtp2"
FORMAL_V07_ACCEPTANCE = (
    "formal_v0.7_qwen3_32b_ascend_continuous_batching_acceptance"
)
WORKLOAD_CLASSES = (
    "short_short",
    "long_prefill_short_decode",
    "mixed",
)
REPLAY_MODES = ("open_loop", "closed_loop")
FORMAL_LAYOUT_IDS = {
    "tp8": (1, 8),
    "2xtp4": (2, 4),
    "4xtp2": (4, 2),
}


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象")
    return value


def load_layout_comparison(path: str | Path) -> dict[str, object]:
    comparison_path = Path(path)
    try:
        raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 layout comparison：{comparison_path}") from exc
    comparison = _require_mapping(raw, str(comparison_path))
    required = {
        "benchmark",
        "evidence_class",
        "workload_class",
        "source_workload_sha256",
        "source_workload_file_sha256",
        "protocol",
        "execution",
        "model",
        "baseline_layout",
        "max_allowed_start_skew_ms",
        "min_telemetry_samples_per_device_per_run",
        "rows",
    }
    if not required.issubset(comparison):
        raise ValueError(f"不是完整 layout comparison：{comparison_path}")
    if comparison["benchmark"] != (
        "qwen3_continuous_batching_replica_layout_comparison"
    ):
        raise ValueError(f"layout comparison benchmark 类型不正确：{comparison_path}")
    return comparison


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_artifact(value: object, label: str, *, path: Path) -> Path:
    artifact = _require_mapping(value, label)
    raw_path = str(artifact.get("path", ""))
    if not raw_path:
        raise ValueError(f"{label} 缺少 path：{path}")
    if not _is_sha256(artifact.get("sha256")):
        raise ValueError(f"{label} SHA-256 无效：{path}")
    declared_size = int(artifact.get("size_bytes", 0))
    if declared_size <= 0:
        raise ValueError(f"{label} size_bytes 无效：{path}")
    artifact_path = Path(raw_path)
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    try:
        payload = artifact_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} 原始文件不可读：{artifact_path}") from exc
    if len(payload) != declared_size:
        raise ValueError(f"{label} 原始文件大小已变化：{artifact_path}")
    if hashlib.sha256(payload).hexdigest() != artifact.get("sha256"):
        raise ValueError(f"{label} 原始文件 SHA-256 已变化：{artifact_path}")
    return artifact_path


def _comparison_evidence_payload(
    comparison: Mapping[str, object],
) -> dict[str, object]:
    """移除时间与可搬迁路径，只比较会影响结论的完整证据内容。"""

    payload = json.loads(
        json.dumps(comparison, ensure_ascii=False, sort_keys=True)
    )
    payload.pop("created_at_utc", None)
    rows = payload.get("rows")
    if isinstance(rows, list):
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                continue
            raw_row.pop("reports", None)
            for field in ("layout_manifest", "source_workload"):
                artifact = raw_row.get(field)
                if isinstance(artifact, dict):
                    artifact.pop("path", None)
            telemetry = raw_row.get("telemetry")
            if isinstance(telemetry, dict):
                artifact = telemetry.get("source_artifact")
                if isinstance(artifact, dict):
                    artifact.pop("path", None)
        rows.sort(key=lambda row: str(row.get("layout_id", "")))
    return payload


def _validate_formal_comparison(
    comparison: Mapping[str, object],
    *,
    path: Path,
) -> None:
    """拒绝只改 evidence_class 字符串、但内部门禁并未成立的报告。"""

    if comparison.get("complete_layout_matrix") is not True:
        raise ValueError(f"正式 comparison 的布局矩阵不完整：{path}")
    if comparison.get("same_eight_devices") is not True:
        raise ValueError(f"正式 comparison 未使用同一 8 devices：{path}")
    if comparison.get("same_scheduler_capacity") is not True:
        raise ValueError(f"正式 comparison 的全局调度容量不一致：{path}")
    if comparison.get("same_telemetry_mapping") is not True:
        raise ValueError(f"正式 comparison 的遥测映射不一致：{path}")
    if comparison.get("incomplete_reasons") != []:
        raise ValueError(f"正式 comparison 仍包含 incomplete reason：{path}")
    rows = comparison.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_LAYOUT_IDS):
        raise ValueError(f"正式 comparison 必须包含三个布局：{path}")
    actual_layouts: dict[str, tuple[int, int]] = {}
    layouts: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    telemetry_by_layout: dict[str, dict[str, object]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"rows[{index}]")
        layout_id = str(row.get("layout_id", ""))
        if row.get("workload_class") != comparison.get("workload_class"):
            raise ValueError(
                f"正式 comparison 的 row workload_class 不一致：{path}"
            )
        if row.get("source_workload_sha256") != comparison.get(
            "source_workload_sha256"
        ):
            raise ValueError(f"正式 comparison 的 row 源 trace 不一致：{path}")
        if row.get("source_workload_file_sha256") != comparison.get(
            "source_workload_file_sha256"
        ):
            raise ValueError(
                f"正式 comparison 的 row 源 workload 文件不一致：{path}"
            )
        actual_layouts[layout_id] = (
            int(row.get("replica_count", -1)),
            int(row.get("tp_size", -1)),
        )
        for field in (
            "layout_manifest_verified",
            "all_replica_reports_formal_candidates",
            "telemetry_formal",
            "start_skew_within_limit",
        ):
            if row.get(field) is not True:
                raise ValueError(
                    f"正式 comparison 的 {layout_id or index} 未通过 {field}：{path}"
                )
        manifest_path = _validate_artifact(
            row.get("layout_manifest"),
            f"{layout_id}.layout_manifest",
            path=path,
        )
        loaded_layout_id, reports = load_layout_manifest(manifest_path)
        if loaded_layout_id != layout_id:
            raise ValueError(f"正式 comparison 的 manifest layout_id 不一致：{path}")
        if layout_id in layouts:
            raise ValueError(f"正式 comparison 的 layout_id 重复：{path}")
        layouts[layout_id] = reports
        _validate_artifact(
            row.get("source_workload"),
            f"{layout_id}.source_workload",
            path=path,
        )
        row_telemetry = _require_mapping(row.get("telemetry"), "row.telemetry")
        if row_telemetry.get("source_file_verified") is not True:
            raise ValueError(f"正式 comparison 的 telemetry 未绑定源文件：{path}")
        telemetry_path = _validate_artifact(
            row_telemetry.get("source_artifact"),
            f"{layout_id}.telemetry.source_artifact",
            path=path,
        )
        telemetry_by_layout[layout_id] = load_telemetry(telemetry_path)
    if actual_layouts != FORMAL_LAYOUT_IDS:
        raise ValueError(f"正式 comparison 的 TP/replica 布局不正确：{path}")
    if comparison.get("baseline_layout") != "tp8":
        raise ValueError(f"正式 comparison 必须使用 tp8 baseline：{path}")

    recomputed = summarize_serving_layouts(
        layouts,
        baseline_layout="tp8",
        max_start_skew_ms=float(comparison["max_allowed_start_skew_ms"]),
        telemetry_by_layout=telemetry_by_layout,
        min_telemetry_samples_per_device_per_run=int(
            comparison["min_telemetry_samples_per_device_per_run"]
        ),
    )
    recomputed_payload = _comparison_evidence_payload(recomputed)
    comparison_payload = _comparison_evidence_payload(comparison)
    comparison_protocol = _require_mapping(
        comparison_payload["protocol"],
        "comparison.protocol",
    )
    if "open_loop_admission_scripted" not in comparison_protocol:
        # 0bad74e 首批 Atlas 证据的原始 replica reports 已记录该字段，但当时的
        # comparison fingerprint 尚未输出它。兼容这批不可重跑的原始证据时，从
        # 已完成 manifest/SHA 校验的 reports 重建字段；不能信任或猜测缺省值。
        scripted_values = {
            bool(
                _require_mapping(report["protocol"], "report.protocol").get(
                    "open_loop_admission_scripted",
                    False,
                )
            )
            for reports in layouts.values()
            for _report_path, report in reports
        }
        if len(scripted_values) != 1:
            raise ValueError(
                f"旧版 comparison 的原始 reports 混用了准入口径：{path}"
            )
        comparison_protocol["open_loop_admission_scripted"] = next(
            iter(scripted_values)
        )
    if recomputed_payload != comparison_payload:
        raise ValueError(
            f"正式 comparison 与原始 workload/manifest/report/telemetry "
            f"重新计算结果不一致：{path}"
        )


def _slo_fingerprint(comparison: Mapping[str, object]) -> tuple[object, ...]:
    protocol = _require_mapping(comparison["protocol"], "protocol")
    return tuple(
        protocol.get(field)
        for field in ("ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms")
    )


def _best_layout(comparison: Mapping[str, object]) -> dict[str, object]:
    rows = comparison["rows"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("正式 layout comparison 必须包含三行布局")

    def median(row: dict[str, object], field: str) -> float:
        summary = _require_mapping(row["summary"], "row.summary")
        metric = _require_mapping(summary[field], f"row.summary.{field}")
        return float(metric["median"])

    typed_rows = [
        _require_mapping(row, f"rows[{index}]")
        for index, row in enumerate(rows)
    ]
    best = max(
        typed_rows,
        key=lambda row: median(row, "goodput_requests_per_second"),
    )
    return {
        "layout_id": best["layout_id"],
        "goodput_requests_per_second": median(
            best,
            "goodput_requests_per_second",
        ),
        "completed_requests_per_second": median(
            best,
            "completed_requests_per_second",
        ),
        "output_tokens_per_second": median(
            best,
            "output_tokens_per_second",
        ),
    }


def summarize_v07_acceptance(
    comparisons: Mapping[
        tuple[str, str],
        tuple[Path, dict[str, object]],
    ],
) -> dict[str, object]:
    if not comparisons:
        raise ValueError("至少需要一份 layout comparison")
    expected = {
        (workload_class, mode)
        for workload_class in WORKLOAD_CLASSES
        for mode in REPLAY_MODES
    }
    unknown = set(comparisons) - expected
    if unknown:
        raise ValueError(f"包含未知 workload/mode：{sorted(unknown)}")

    ordered_keys = [
        (workload_class, mode)
        for workload_class in WORKLOAD_CLASSES
        for mode in REPLAY_MODES
        if (workload_class, mode) in comparisons
    ]
    first = comparisons[ordered_keys[0]][1]
    expected_execution = _require_mapping(first["execution"], "execution")
    expected_model = _require_mapping(first["model"], "model")
    records: list[dict[str, object]] = []
    incomplete_reasons: list[str] = []
    if set(comparisons) != expected:
        missing = sorted(expected - set(comparisons))
        incomplete_reasons.append(f"缺少 workload/mode 组合：{missing}")

    for key in ordered_keys:
        workload_class, mode = key
        path, comparison = comparisons[key]
        if comparison.get("workload_class") != workload_class:
            raise ValueError(f"workload_class 与输入 key 不一致：{path}")
        protocol = _require_mapping(comparison["protocol"], "protocol")
        if protocol.get("mode") != mode:
            raise ValueError(f"replay mode 与输入 key 不一致：{path}")
        if _require_mapping(comparison["execution"], "execution") != (
            expected_execution
        ):
            raise ValueError(f"验收矩阵的硬件/软件环境不一致：{path}")
        if _require_mapping(comparison["model"], "model") != expected_model:
            raise ValueError(f"验收矩阵的模型/权重/提交不一致：{path}")
        if comparison.get("baseline_layout") != "tp8":
            incomplete_reasons.append(f"{workload_class}/{mode} 未使用 tp8 baseline")
        if comparison.get("evidence_class") != FORMAL_LAYOUT_COMPARISON:
            incomplete_reasons.append(
                f"{workload_class}/{mode} 不是正式三布局比较"
            )
        else:
            _validate_formal_comparison(comparison, path=path)
        if not _is_sha256(comparison.get("source_workload_sha256")):
            raise ValueError(f"源 workload SHA-256 无效：{path}")
        if not _is_sha256(comparison.get("source_workload_file_sha256")):
            raise ValueError(f"源 workload 文件 SHA-256 无效：{path}")

        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"无法读取验收输入文件：{path}") from exc
        records.append(
            {
                "workload_class": workload_class,
                "mode": mode,
                "source_workload_sha256": comparison[
                    "source_workload_sha256"
                ],
                "source_workload_file_sha256": comparison[
                    "source_workload_file_sha256"
                ],
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "evidence_class": comparison["evidence_class"],
                "best_layout": _best_layout(comparison),
            }
        )

    for workload_class in WORKLOAD_CLASSES:
        present = [
            comparisons[(workload_class, mode)][1]
            for mode in REPLAY_MODES
            if (workload_class, mode) in comparisons
        ]
        if len(present) != 2:
            continue
        if len(
            {str(item["source_workload_sha256"]) for item in present}
        ) != 1:
            raise ValueError(f"{workload_class} 的 open/closed 源 trace 不一致")
        if len(
            {str(item["source_workload_file_sha256"]) for item in present}
        ) != 1:
            raise ValueError(
                f"{workload_class} 的 open/closed 源 workload 文件不一致"
            )
        if len({_slo_fingerprint(item) for item in present}) != 1:
            raise ValueError(f"{workload_class} 的 open/closed SLO 不一致")

    formal = set(comparisons) == expected and not incomplete_reasons
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.7_continuous_batching_acceptance",
        "evidence_class": (
            FORMAL_V07_ACCEPTANCE
            if formal
            else "development_or_incomplete_v0.7_acceptance"
        ),
        "expected_workloads": list(WORKLOAD_CLASSES),
        "expected_modes": list(REPLAY_MODES),
        "complete_matrix": set(comparisons) == expected,
        "incomplete_reasons": incomplete_reasons,
        "execution": expected_execution,
        "model": expected_model,
        "comparisons": records,
    }
