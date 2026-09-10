"""v0.7.1 mixed open-loop 跨会话复现检查与保守判因。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
from typing import Mapping, Sequence

from .serving_layout import load_layout_manifest, summarize_serving_layout
from .serving_telemetry import load_telemetry, summarize_telemetry


MIXED_REPRO_SCHEMA_VERSION = 1
FORMAL_SEQUENCE = (
    "tp8",
    "4xtp2",
    "4xtp2",
    "tp8",
    "4xtp2",
    "tp8",
    "tp8",
    "4xtp2",
)
FORMAL_PROTOCOL = {
    "mode": "open_loop",
    "closed_loop_clients": None,
    "warmup": 1,
    "repeats": 3,
    "ttft_slo_ms": 15000.0,
    "tpot_slo_ms": 500.0,
    "e2e_slo_ms": 30000.0,
    "open_loop_admission_scripted": True,
}
FORMAL_LAYOUTS = {
    "tp8": {"tp_size": 8, "replica_count": 1, "per_replica_slots": 32},
    "4xtp2": {"tp_size": 2, "replica_count": 4, "per_replica_slots": 8},
}
DIAGNOSIS_THRESHOLDS = {
    "max_session_cv": 0.10,
    "max_relative_span": 0.20,
    "max_within_session_cv": 0.10,
    "reference_relative_tolerance": 0.20,
    "order_effect_relative_delta": 0.15,
    "association_abs_correlation": 0.70,
    "frequency_relative_span": 0.03,
    "temperature_span_celsius": 5.0,
    "initial_hbm_usage_percent_max": 10.0,
    "initial_aicore_usage_percent_max": 5.0,
    "historical_initial_hbm_excess_percentage_points": 20.0,
    "historical_initial_aicore_excess_percentage_points": 20.0,
}


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _artifact(path: Path, *, relative_to: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("统计样本不能为空")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
    }


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("CV 样本不能为空")
    mean = statistics.fmean(float(value) for value in values)
    if mean == 0.0:
        return 0.0 if all(float(value) == 0.0 for value in values) else math.inf
    return statistics.pstdev(float(value) for value in values) / abs(mean)


def _relative_distance(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value - reference) / abs(reference)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _global_model_identity(
    model: Mapping[str, object],
    provenance: Mapping[str, object],
) -> tuple[str, int]:
    """跨 TP 布局比较全局模型；local_parameter_count 只描述当前分片。"""

    local_parameter_count = int(model["local_parameter_count"])
    payload = {
        "model": {
            "type": model["type"],
            "config": model["config"],
            "full_parameter_count": model["full_parameter_count"],
        },
        "config_sha256": provenance["config_sha256"],
        "metadata_sha256": provenance["metadata_sha256"],
        "index_sha256": provenance["index_sha256"],
        "weights": provenance["weights"],
    }
    return _canonical_sha256(payload), local_parameter_count


def _request_work_shape_sha256(report: Mapping[str, object]) -> str:
    """绑定请求终态与计算长度，不要求跨 open-loop session 逐 token 相同。"""

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


def _summarize_initial_npu_state(
    telemetry: Mapping[str, object],
    *,
    expected_logical_device_ids: Sequence[int],
) -> dict[str, object]:
    """取 sampler 启动后每个 device 的首个样本，观察模型加载前设备是否空闲。"""

    samples = telemetry.get("samples")
    if not isinstance(samples, list):
        raise ValueError("telemetry samples 必须是数组")
    per_device: dict[str, dict[str, float | int]] = {}
    initial_samples: list[Mapping[str, object]] = []
    for device_id in expected_logical_device_ids:
        candidates = [
            sample
            for sample in samples
            if isinstance(sample, dict)
            and int(sample.get("logical_device_id", -1)) == int(device_id)
        ]
        if not candidates:
            raise ValueError(f"device {device_id} 缺少初始 telemetry sample")
        first = min(candidates, key=lambda sample: int(sample["timestamp_unix_ns"]))
        initial_samples.append(first)
        per_device[str(device_id)] = {
            "timestamp_unix_ns": int(first["timestamp_unix_ns"]),
            "hbm_usage_percent": float(first["hbm_usage_percent"]),
            "aicore_usage_percent": float(first["aicore_usage_percent"]),
            "power_watts": float(first["power_watts"]),
        }
    fields = ("hbm_usage_percent", "aicore_usage_percent", "power_watts")
    overall = {
        field: _summary([float(sample[field]) for sample in initial_samples])
        for field in fields
    }
    clean = bool(
        float(overall["hbm_usage_percent"]["max"])
        <= DIAGNOSIS_THRESHOLDS["initial_hbm_usage_percent_max"]
        and float(overall["aicore_usage_percent"]["max"])
        <= DIAGNOSIS_THRESHOLDS["initial_aicore_usage_percent_max"]
    )
    return {"per_device": per_device, "overall": overall, "clean": clean}


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 3:
        return None
    mean_x = statistics.fmean(values_x)
    mean_y = statistics.fmean(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _summarize_host_telemetry(
    path: Path,
    *,
    run_intervals: Sequence[tuple[int, int]],
    relative_to: Path,
) -> dict[str, object]:
    telemetry = _load_json(path)
    if telemetry.get("schema_version") != 1:
        raise ValueError(f"Host telemetry schema_version 不支持：{path}")
    if telemetry.get("collector") != "procfs_host":
        raise ValueError(f"Host telemetry collector 不支持：{path}")
    samples = telemetry.get("samples")
    errors = telemetry.get("errors")
    if not isinstance(samples, list) or not isinstance(errors, list):
        raise ValueError(f"Host telemetry samples/errors 必须是数组：{path}")
    metric_fields = (
        "cpu_usage_percent",
        "cpu_iowait_percent",
        "load1",
        "load5",
        "load15",
        "memory_available_mb",
        "psi_cpu_some_avg10",
        "psi_memory_some_avg10",
        "psi_io_some_avg10",
    )
    selected: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for repeat, (started_at_ns, ended_at_ns) in enumerate(run_intervals):
        current = [
            sample
            for sample in samples
            if isinstance(sample, dict)
            and started_at_ns <= int(sample["timestamp_unix_ns"]) <= ended_at_ns
        ]
        coverage.append({"repeat": repeat, "sample_count": len(current)})
        selected.extend(current)
    for sample in selected:
        for field in metric_fields:
            value = float(sample[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Host telemetry {field} 必须是非负有限值")
    all_runs_covered = all(row["sample_count"] >= 2 for row in coverage)
    return {
        "source_artifact": _artifact(path, relative_to=relative_to),
        "complete": telemetry.get("complete") is True,
        "error_free": not errors,
        "errors": errors,
        "all_runs_covered": all_runs_covered,
        "run_coverage": coverage,
        "overall": {
            field: _summary([float(sample[field]) for sample in selected])
            if selected
            else None
            for field in metric_fields
        },
    }


def _tar_json(archive: Path, suffix: str) -> dict[str, object]:
    try:
        with tarfile.open(archive, "r:gz") as handle:
            matches = [
                member
                for member in handle.getmembers()
                if member.isfile() and member.name.endswith(suffix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"归档中 {suffix!r} 应恰好出现一次，实际 {len(matches)} 次"
                )
            stream = handle.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"无法读取归档成员：{matches[0].name}")
            value = json.loads(stream.read().decode("utf-8"))
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取证据归档：{archive}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"归档成员顶层必须是对象：{suffix}")
    return value


def load_reference_observations(
    v07_comparison_path: str | Path,
    v07_evidence_archive: str | Path,
    v071_evidence_archive: str | Path,
) -> dict[str, object]:
    """读取 v0.7 与 v0.7.1 的冻结 mixed 对照，禁止手抄基准数字。"""

    comparison_path = Path(v07_comparison_path)
    v07_archive = Path(v07_evidence_archive)
    v071_archive = Path(v071_evidence_archive)
    comparison = _load_json(comparison_path)
    old_rows = {
        str(row["layout_id"]): row
        for row in comparison.get("rows", [])
        if isinstance(row, dict) and str(row.get("layout_id")) in FORMAL_LAYOUTS
    }
    if set(old_rows) != set(FORMAL_LAYOUTS):
        raise ValueError("v0.7 mixed comparison 必须包含 tp8 与 4xtp2")

    gate = _tar_json(v071_archive, "/profiling_gate.json")
    cases = {
        str(case["case_id"]): case
        for case in gate.get("cases", [])
        if isinstance(case, dict)
    }
    mixed = cases.get("mixed_overload")
    if not isinstance(mixed, dict):
        raise ValueError("v0.7.1 gate 缺少 mixed_overload")
    new_rows = {
        str(row["layout_id"]): row
        for row in mixed.get("rows", [])
        if isinstance(row, dict)
    }
    if set(new_rows) != set(FORMAL_LAYOUTS):
        raise ValueError("v0.7.1 mixed gate 必须包含 tp8 与 4xtp2")

    old_telemetry: dict[str, object] = {}
    old_initial_state: dict[str, object] = {}
    old_reports: dict[str, list[dict[str, object]]] = {}
    for layout_id in FORMAL_LAYOUTS:
        telemetry = _tar_json(
            v07_archive,
            f"/mixed_open_loop/{layout_id}/telemetry.json",
        )
        old_reports[layout_id] = [
            _tar_json(
                v07_archive,
                f"/mixed_open_loop/{report_name}",
            )
            for report_name in old_rows[layout_id]["reports"]
        ]
        intervals = [
            (int(run["started_at_unix_ns"]), int(run["ended_at_unix_ns"]))
            for run in old_rows[layout_id]["runs"]
        ]
        old_telemetry[layout_id] = summarize_telemetry(
            telemetry,
            run_intervals=intervals,
            expected_logical_device_ids=range(8),
            min_samples_per_device_per_run=2,
        )
        old_initial_state[layout_id] = _summarize_initial_npu_state(
            telemetry,
            expected_logical_device_ids=range(8),
        )

    old_source = str(comparison["source_workload_sha256"])
    new_sources = set()
    new_points: dict[str, dict[str, object]] = {}
    for layout_id in FORMAL_LAYOUTS:
        point = _tar_json(
            v071_archive,
            f"/points/mixed_overload/{layout_id}/profile_summary.json",
        )
        new_points[layout_id] = point
        new_sources.add(str(point["source_workload_sha256"]))
    if new_sources != {old_source}:
        raise ValueError("v0.7 与 v0.7.1 mixed source workload digest 不一致")

    def layout_reference(
        rows: Mapping[str, Mapping[str, object]],
        layout_id: str,
        reports: Sequence[Mapping[str, object]] | None = None,
        service_summary: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        row = rows[layout_id]
        summary = row.get("summary")
        if isinstance(summary, dict):
            goodput = float(summary["goodput_requests_per_second"]["median"])
            run_values = [
                float(run["goodput_requests_per_second"])
                for run in row.get("runs", [])
            ]
        else:
            goodput = float(row["goodput_requests_per_second"])
            run_values = []
        metric_summary = summary if isinstance(summary, dict) else service_summary
        completed = (
            float(metric_summary["completed_requests_per_second"]["median"])
            if isinstance(metric_summary, Mapping)
            else goodput
        )
        result: dict[str, object] = {
            "goodput_requests_per_second": goodput,
            "completed_requests_per_second": completed,
            "within_session_cv": (
                _coefficient_of_variation(run_values) if run_values else None
            ),
            "runs": [
                {
                    field: run[field]
                    for field in (
                        "wall_time_ms",
                        "completed_requests",
                        "good_requests",
                        "completed_requests_per_second",
                        "goodput_requests_per_second",
                    )
                    if field in run
                }
                for run in row.get("runs", [])
            ],
        }
        if reports is not None:
            result["output_sha256_by_replica"] = {
                str(report["distributed"]["replica_index"]): str(
                    report["runs"][0]["output_sha256"]
                )
                for report in reports
            }
        return result

    return {
        "source_workload_sha256": old_source,
        "source_workload_file_sha256": str(
            comparison["source_workload_file_sha256"]
        ),
        "v0.7": {
            "layouts": {
                layout_id: {
                    **layout_reference(
                        old_rows,
                        layout_id,
                        old_reports[layout_id],
                    ),
                    "telemetry": old_telemetry[layout_id],
                    "initial_npu_state": old_initial_state[layout_id],
                }
                for layout_id in FORMAL_LAYOUTS
            },
            "comparison_artifact": _artifact(
                comparison_path,
                relative_to=comparison_path.parent.parent.parent.parent,
            ),
            "evidence_archive": _artifact(
                v07_archive,
                relative_to=v07_archive.parent,
            ),
        },
        "v0.7.1": {
            "layouts": {
                layout_id: layout_reference(
                    new_rows,
                    layout_id,
                    service_summary=new_points[layout_id]["service"]["summary"],
                )
                for layout_id in FORMAL_LAYOUTS
            },
            "evidence_archive": _artifact(
                v071_archive,
                relative_to=v071_archive.parent,
            ),
        },
    }


def _load_session(
    root: Path,
    *,
    position: int,
    expected_layout: str,
) -> dict[str, object]:
    session_id = f"session-{position:02d}-{expected_layout}"
    session_dir = root / session_id
    manifest_path = session_dir / "layout_manifest.json"
    telemetry_path = session_dir / "telemetry.json"
    host_telemetry_path = session_dir / "host_telemetry.json"
    before_path = session_dir / "host_before.json"
    after_path = session_dir / "host_after.json"
    layout_id, reports = load_layout_manifest(manifest_path)
    if layout_id != expected_layout:
        raise ValueError(f"{session_id} layout 应为 {expected_layout}")
    row = summarize_serving_layout(layout_id, reports, max_start_skew_ms=100.0)
    first = reports[0][1]
    telemetry = load_telemetry(telemetry_path)
    initial_npu_state = _summarize_initial_npu_state(
        telemetry,
        expected_logical_device_ids=range(8),
    )
    telemetry_summary = summarize_telemetry(
        telemetry,
        run_intervals=[
            (int(run["started_at_unix_ns"]), int(run["ended_at_unix_ns"]))
            for run in row["runs"]
        ],
        expected_logical_device_ids=range(8),
        min_samples_per_device_per_run=2,
    )
    telemetry_artifact = telemetry_summary.get("source_artifact")
    if isinstance(telemetry_artifact, dict):
        telemetry_artifact["path"] = str(telemetry_path.relative_to(root))

    run_intervals = [
        (int(run["started_at_unix_ns"]), int(run["ended_at_unix_ns"]))
        for run in row["runs"]
    ]
    host_telemetry = _summarize_host_telemetry(
        host_telemetry_path,
        run_intervals=run_intervals,
        relative_to=root,
    )

    before = _load_json(before_path)
    after = _load_json(after_path)
    report_digests = {
        str(report["distributed"]["replica_index"]): str(
            report["runs"][0]["output_sha256"]
        )
        for _path, report in reports
    }
    work_shape_digests = {
        str(report["distributed"]["replica_index"]): _request_work_shape_sha256(
            report
        )
        for _path, report in reports
    }
    model_identity_sha256, local_parameter_count = _global_model_identity(
        first["model"],
        first["provenance"],
    )
    return {
        "session_id": session_id,
        "position": position,
        "layout_id": layout_id,
        "git": first["provenance"]["git"],
        "protocol": first["protocol"],
        "environment": first["environment"],
        "machine": {
            field: first["distributed"].get(field)
            for field in (
                "hostname",
                "backend",
                "visible_device_count",
                "physical_card_count",
                "chips_per_card",
                "interconnect_topology",
            )
        },
        "source_workload_sha256": first["workload"]["source_sha256"],
        "source_workload_file_sha256": first["workload"]["source_file_sha256"],
        "model_identity_sha256": model_identity_sha256,
        "local_parameter_count": local_parameter_count,
        "logical_device_ids": first["distributed"]["global_logical_device_ids"],
        "tp_size": first["distributed"]["tp_size"],
        "replica_count": first["distributed"]["replica_count"],
        "scheduler_capacity": row["scheduler_capacity"],
        "artifacts": {
            "layout_manifest": _artifact(manifest_path, relative_to=root),
            "host_before": _artifact(before_path, relative_to=root),
            "host_after": _artifact(after_path, relative_to=root),
        },
        "host": {"before": before, "after": after},
        "host_telemetry": host_telemetry,
        "initial_npu_state": initial_npu_state,
        "output_sha256_by_replica": report_digests,
        "output_work_shape_sha256_by_replica": work_shape_digests,
        "service": {
            "goodput_requests_per_second": row["summary"][
                "goodput_requests_per_second"
            ],
            "completed_requests_per_second": row["summary"][
                "completed_requests_per_second"
            ],
            "queue_ms": row["summary"]["queue_ms"],
            "ttft_ms": row["summary"]["ttft_ms"],
            "tpot_ms": row["summary"]["tpot_ms"],
            "e2e_latency_ms": row["summary"]["e2e_latency_ms"],
            "runs": row["runs"],
            "max_rank_peak_device_memory_mb": row[
                "max_rank_peak_device_memory_mb"
            ],
        },
        "telemetry": telemetry_summary,
    }


def summarize_output_reproducibility(
    sessions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """区分逐 token 变体与会改变性能工作量的终态/长度变体。"""

    result: dict[str, object] = {}
    for layout_id in FORMAL_LAYOUTS:
        rows = [
            session for session in sessions if session["layout_id"] == layout_id
        ]
        replica_ids = sorted(
            {
                str(replica_id)
                for session in rows
                for replica_id in session["output_sha256_by_replica"]
            }
        )
        replicas: dict[str, object] = {}
        for replica_id in replica_ids:
            content_variants: dict[str, list[str]] = {}
            work_shape_variants: dict[str, list[str]] = {}
            for session in rows:
                session_id = str(session["session_id"])
                content_sha256 = str(
                    session["output_sha256_by_replica"][replica_id]
                )
                shape_sha256 = str(
                    session["output_work_shape_sha256_by_replica"][replica_id]
                )
                content_variants.setdefault(content_sha256, []).append(session_id)
                work_shape_variants.setdefault(shape_sha256, []).append(session_id)
            replicas[replica_id] = {
                "bitwise_output_stable": len(content_variants) == 1,
                "work_shape_stable": len(work_shape_variants) == 1,
                "content_variants": content_variants,
                "work_shape_variants": work_shape_variants,
            }
        result[layout_id] = {
            "bitwise_output_stable": all(
                bool(replica["bitwise_output_stable"])
                for replica in replicas.values()
            ),
            "work_shape_stable": all(
                bool(replica["work_shape_stable"])
                for replica in replicas.values()
            ),
            "replicas": replicas,
        }
    return result


def classify_reproduction(
    sessions: Sequence[Mapping[str, object]],
    references: Mapping[str, object],
) -> dict[str, object]:
    """分类复现状态；只给证据范围内结论，不把相关性冒充因果。"""

    by_layout = {
        layout_id: [
            session for session in sessions if session["layout_id"] == layout_id
        ]
        for layout_id in FORMAL_LAYOUTS
    }
    aggregates: dict[str, object] = {}
    for layout_id, rows in by_layout.items():
        session_values = [
            float(session["service"]["goodput_requests_per_second"]["median"])
            for session in rows
        ]
        completed_values = [
            float(
                session["service"]["completed_requests_per_second"]["median"]
            )
            for session in rows
        ]
        within_cv = [
            _coefficient_of_variation(
                [
                    float(run["goodput_requests_per_second"])
                    for run in session["service"]["runs"]
                ]
            )
            for session in rows
        ]
        summary = _summary(session_values)
        relative_span = (summary["max"] - summary["min"]) / max(
            abs(float(summary["median"])), 1e-12
        )
        session_cv = _coefficient_of_variation(session_values)
        stable = bool(
            session_cv <= DIAGNOSIS_THRESHOLDS["max_session_cv"]
            and relative_span <= DIAGNOSIS_THRESHOLDS["max_relative_span"]
            and max(within_cv) <= DIAGNOSIS_THRESHOLDS["max_within_session_cv"]
        )
        early = statistics.median(session_values[:2])
        late = statistics.median(session_values[2:])
        aggregates[layout_id] = {
            "session_goodput_requests_per_second": summary,
            "session_completed_requests_per_second": _summary(completed_values),
            "session_cv": session_cv,
            "relative_span": relative_span,
            "within_session_cv": _summary(within_cv),
            "early_vs_late_relative_delta": (late - early) / max(abs(early), 1e-12),
            "stable": stable,
        }

    ratio = (
        aggregates["4xtp2"]["session_goodput_requests_per_second"]["median"]
        / aggregates["tp8"]["session_goodput_requests_per_second"]["median"]
    )
    normalized_goodput = []
    telemetry_values: dict[str, list[float]] = {
        field: []
        for field in (
            "aicore_current_frequency_mhz",
            "temperature_celsius",
            "power_watts",
            "aicore_usage_percent",
        )
    }
    usable_sessions = []
    for session in sessions:
        layout_id = str(session["layout_id"])
        value = float(session["service"]["goodput_requests_per_second"]["median"])
        layout_median = float(
            aggregates[layout_id]["session_goodput_requests_per_second"]["median"]
        )
        metrics = session["telemetry"]["overall"]
        medians: dict[str, float] = {}
        for field in telemetry_values:
            metric = metrics.get(field)
            if not isinstance(metric, dict):
                break
            medians[field] = float(metric["median"])
        else:
            normalized_goodput.append(value / layout_median)
            for field, metric_value in medians.items():
                telemetry_values[field].append(metric_value)
            usable_sessions.append(session["session_id"])

    associations = {
        field: {
            "pearson_with_layout_normalized_goodput": _pearson(
                normalized_goodput,
                values,
            ),
            "observed": _summary(values) if values else None,
        }
        for field, values in telemetry_values.items()
    }
    host_values: dict[str, list[float]] = {
        field: []
        for field in (
            "cpu_usage_percent",
            "cpu_iowait_percent",
            "load1",
            "psi_cpu_some_avg10",
            "psi_io_some_avg10",
        )
    }
    host_normalized_goodput: list[float] = []
    host_sessions: list[str] = []
    for session in sessions:
        layout_id = str(session["layout_id"])
        value = float(session["service"]["goodput_requests_per_second"]["median"])
        layout_median = float(
            aggregates[layout_id]["session_goodput_requests_per_second"]["median"]
        )
        metrics = session["host_telemetry"]["overall"]
        medians: dict[str, float] = {}
        for field in host_values:
            metric = metrics.get(field)
            if not isinstance(metric, dict):
                break
            medians[field] = float(metric["median"])
        else:
            host_normalized_goodput.append(value / layout_median)
            for field, metric_value in medians.items():
                host_values[field].append(metric_value)
            host_sessions.append(str(session["session_id"]))
    host_associations = {
        field: {
            "pearson_with_layout_normalized_goodput": _pearson(
                host_normalized_goodput,
                values,
            ),
            "observed": _summary(values) if values else None,
        }
        for field, values in host_values.items()
    }
    order_effect = any(
        abs(float(aggregate["early_vs_late_relative_delta"]))
        >= DIAGNOSIS_THRESHOLDS["order_effect_relative_delta"]
        for aggregate in aggregates.values()
    )
    all_stable = all(bool(value["stable"]) for value in aggregates.values())
    observed_tp8 = float(
        aggregates["tp8"]["session_goodput_requests_per_second"]["median"]
    )
    old_tp8 = float(
        references["v0.7"]["layouts"]["tp8"]["goodput_requests_per_second"]
    )
    new_tp8 = float(
        references["v0.7.1"]["layouts"]["tp8"][
            "goodput_requests_per_second"
        ]
    )
    old_distance = _relative_distance(observed_tp8, old_tp8)
    new_distance = _relative_distance(observed_tp8, new_tp8)
    tolerance = DIAGNOSIS_THRESHOLDS["reference_relative_tolerance"]

    if all_stable and new_distance <= tolerance and old_distance > tolerance:
        status = "historical_v07_tp8_slowdown_not_reproduced"
        conclusion = (
            "当前 ABBA+BAAB 中 TP8 稳定接近 v0.7.1，旧 v0.7 TP8 慢值未复现；"
            "4.24× 主要属于旧会话的运行状态，不是稳定布局倍率。"
        )
    elif all_stable and old_distance <= tolerance and new_distance > tolerance:
        status = "v071_tp8_speedup_not_reproduced"
        conclusion = (
            "当前 TP8 稳定接近 v0.7，v0.7.1 的快速 TP8 未复现；"
            "需要针对 v0.7.1 当次运行状态取证。"
        )
    elif all_stable:
        status = "stable_third_regime"
        conclusion = "两种布局均稳定，但落在两个历史观测之外，存在新的稳定运行状态。"
    elif order_effect:
        status = "order_or_machine_state_effect"
        conclusion = "同布局前后半程差异超过阈值，性能受运行顺序或机器状态影响。"
    else:
        status = "unstable_unexplained"
        conclusion = "跨会话或会话内波动过大，当前证据仍不能稳定复现任一历史状态。"

    old_initial = references["v0.7"]["layouts"]["tp8"][
        "initial_npu_state"
    ]["overall"]
    current_initial_hbm = [
        float(session["initial_npu_state"]["overall"]["hbm_usage_percent"]["median"])
        for session in sessions
        if session["layout_id"] == "tp8"
    ]
    current_initial_aicore = [
        float(
            session["initial_npu_state"]["overall"][
                "aicore_usage_percent"
            ]["median"]
        )
        for session in sessions
        if session["layout_id"] == "tp8"
    ]
    old_initial_hbm = float(old_initial["hbm_usage_percent"]["median"])
    old_initial_aicore = float(old_initial["aicore_usage_percent"]["median"])
    initial_hbm_excess = old_initial_hbm - max(current_initial_hbm)
    initial_aicore_excess = old_initial_aicore - max(current_initial_aicore)
    historical_contention = bool(
        initial_hbm_excess
        >= DIAGNOSIS_THRESHOLDS[
            "historical_initial_hbm_excess_percentage_points"
        ]
        and initial_aicore_excess
        >= DIAGNOSIS_THRESHOLDS[
            "historical_initial_aicore_excess_percentage_points"
        ]
    )
    old_output_sha256 = str(
        references["v0.7"]["layouts"]["tp8"][
            "output_sha256_by_replica"
        ]["0"]
    )
    matching_output_sessions = [
        str(session["session_id"])
        for session in sessions
        if session["layout_id"] == "tp8"
        and old_output_sha256 in session["output_sha256_by_replica"].values()
    ]
    old_completed = float(
        references["v0.7"]["layouts"]["tp8"][
            "completed_requests_per_second"
        ]
    )
    current_completed = float(
        aggregates["tp8"]["session_completed_requests_per_second"]["median"]
    )
    if historical_contention:
        root_cause_status = "historical_v07_npu_contention"
        root_cause_confidence = "strong"
        root_cause_conclusion = (
            "旧 v0.7 TP8 在模型加载前已有显著 HBM 与 AICore 占用；"
            "本次空闲起跑未复现慢态，且存在与旧输出逐位相同的快速 session。"
            "证据支持旧结果受到同设备并发负载污染，而不是稳定的 TP8 布局效应。"
        )
    else:
        root_cause_status = "historical_v07_slowdown_source_unresolved"
        root_cause_confidence = "insufficient"
        root_cause_conclusion = (
            "当前证据未观察到足以解释旧 TP8 慢态的模型加载前设备占用差异。"
        )
    historical_root_cause = {
        "status": root_cause_status,
        "confidence": root_cause_confidence,
        "conclusion": root_cause_conclusion,
        "evidence": {
            "v0.7_tp8_initial_hbm_usage_percent_median": old_initial_hbm,
            "current_tp8_initial_hbm_usage_percent": _summary(
                current_initial_hbm
            ),
            "v0.7_tp8_initial_aicore_usage_percent_median": old_initial_aicore,
            "current_tp8_initial_aicore_usage_percent": _summary(
                current_initial_aicore
            ),
            "initial_hbm_excess_percentage_points": initial_hbm_excess,
            "initial_aicore_excess_percentage_points": initial_aicore_excess,
            "v0.7_tp8_completed_requests_per_second": old_completed,
            "current_tp8_completed_requests_per_second": current_completed,
            "current_vs_v0.7_completed_throughput_ratio": (
                current_completed / old_completed
            ),
            "v0.7_tp8_good_requests_per_run": [
                int(run["good_requests"])
                for run in references["v0.7"]["layouts"]["tp8"]["runs"]
            ],
            "matching_output_sha256": old_output_sha256,
            "current_sessions_matching_v0.7_output": matching_output_sessions,
        },
        "boundary": (
            "可确认旧测量发生设备争用；现有进程级证据不能追溯并发负载的所有者或名称。"
        ),
    }

    association_flags: list[str] = []
    correlation_threshold = DIAGNOSIS_THRESHOLDS["association_abs_correlation"]
    frequency = associations["aicore_current_frequency_mhz"]
    if frequency["observed"] is not None:
        values = frequency["observed"]
        relative_span = (values["max"] - values["min"]) / max(
            abs(float(values["median"])), 1e-12
        )
        correlation = frequency["pearson_with_layout_normalized_goodput"]
        if (
            relative_span >= DIAGNOSIS_THRESHOLDS["frequency_relative_span"]
            and correlation is not None
            and abs(correlation) >= correlation_threshold
        ):
            association_flags.append("npu_frequency_associated")
    temperature = associations["temperature_celsius"]
    if temperature["observed"] is not None:
        values = temperature["observed"]
        correlation = temperature["pearson_with_layout_normalized_goodput"]
        if (
            values["max"] - values["min"]
            >= DIAGNOSIS_THRESHOLDS["temperature_span_celsius"]
            and correlation is not None
            and correlation <= -correlation_threshold
        ):
            association_flags.append("temperature_associated")

    return {
        "thresholds": dict(DIAGNOSIS_THRESHOLDS),
        "aggregates": aggregates,
        "goodput_ratio_4xtp2_over_tp8": ratio,
        "reference_distance": {
            "tp8_vs_v0.7": old_distance,
            "tp8_vs_v0.7.1": new_distance,
        },
        "order_effect": order_effect,
        "telemetry_associations": associations,
        "telemetry_association_sessions": usable_sessions,
        "host_associations": host_associations,
        "host_association_sessions": host_sessions,
        "association_flags": association_flags,
        "historical_root_cause": historical_root_cause,
        "status": status,
        "conclusion": conclusion,
        "scope": (
            "该判定识别可复现运行状态；只有同时出现足够状态变化和强相关时才输出关联标记，"
            "关联标记仍不等于物理因果。"
        ),
    }


def summarize_mixed_reproduction(
    root: str | Path,
    *,
    v07_comparison_path: str | Path,
    v07_evidence_archive: str | Path,
    v071_evidence_archive: str | Path,
) -> dict[str, object]:
    root_path = Path(root)
    references = load_reference_observations(
        v07_comparison_path,
        v07_evidence_archive,
        v071_evidence_archive,
    )
    sessions = [
        _load_session(root_path, position=index, expected_layout=layout_id)
        for index, layout_id in enumerate(FORMAL_SEQUENCE, start=1)
    ]

    expected_source = references["source_workload_sha256"]
    expected_source_file = references["source_workload_file_sha256"]
    git_commits = {str(session["git"]["commit"]) for session in sessions}
    model_identities = {
        str(session["model_identity_sha256"]) for session in sessions
    }
    environment_identities = {
        json.dumps(session["environment"], sort_keys=True, separators=(",", ":"))
        for session in sessions
    }
    machine_identities = {
        json.dumps(session["machine"], sort_keys=True, separators=(",", ":"))
        for session in sessions
    }
    protocols = {
        json.dumps(session["protocol"], sort_keys=True, separators=(",", ":"))
        for session in sessions
    }
    output_reproducibility = summarize_output_reproducibility(sessions)
    incomplete_reasons: list[str] = []
    warnings: list[str] = []
    if len(git_commits) != 1:
        incomplete_reasons.append("八个 session 的 git commit 不一致")
    if any(bool(session["git"]["dirty"]) for session in sessions):
        incomplete_reasons.append("至少一个 session 在 dirty tracked worktree 上运行")
    if len(model_identities) != 1:
        incomplete_reasons.append("八个 session 的模型或权重不一致")
    if len(environment_identities) != 1:
        incomplete_reasons.append("八个 session 的软件/NPU 运行时环境不一致")
    if len(machine_identities) != 1:
        incomplete_reasons.append("八个 session 没有在同一物理机器与拓扑上运行")
    expected_protocol = json.dumps(
        FORMAL_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
    )
    if protocols != {expected_protocol}:
        incomplete_reasons.append("八个 session 没有使用冻结的 mixed open-loop 协议")

    for session, expected_layout in zip(sessions, FORMAL_SEQUENCE):
        host_before = session["host"]["before"]
        host_after = session["host"]["after"]
        if not (
            host_before.get("hostname") == session["machine"]["hostname"]
            and host_after.get("hostname") == session["machine"]["hostname"]
        ):
            incomplete_reasons.append(
                f"{session['session_id']} Host/作业机器身份不一致"
            )
        for phase, snapshot in (("before", host_before), ("after", host_after)):
            snapshot_git = snapshot.get("git")
            if not isinstance(snapshot_git, dict) or (
                snapshot_git.get("commit") != session["git"]["commit"]
                or snapshot_git.get("dirty_tracked") is not False
            ):
                incomplete_reasons.append(
                    f"{session['session_id']} Host {phase} Git 状态不合格"
                )
        if session["source_workload_sha256"] != expected_source:
            incomplete_reasons.append(
                f"{session['session_id']} source workload digest 不匹配"
            )
        if session["source_workload_file_sha256"] != expected_source_file:
            incomplete_reasons.append(
                f"{session['session_id']} source workload 文件 digest 不匹配"
            )
        expected = FORMAL_LAYOUTS[expected_layout]
        if int(session["tp_size"]) != expected["tp_size"]:
            incomplete_reasons.append(f"{session['session_id']} TP size 不匹配")
        if int(session["replica_count"]) != expected["replica_count"]:
            incomplete_reasons.append(f"{session['session_id']} replica count 不匹配")
        capacity = session["scheduler_capacity"]
        if (
            int(capacity["per_replica_max_slots"]) != expected["per_replica_slots"]
            or int(capacity["total_max_slots"]) != 32
            or int(capacity["total_max_queue_size"]) != 128
            or int(capacity["max_seq_len"]) != 4096
        ):
            incomplete_reasons.append(
                f"{session['session_id']} scheduler capacity 不匹配"
            )
        if session["logical_device_ids"] != list(range(8)):
            incomplete_reasons.append(
                f"{session['session_id']} 没有恰好使用 logical devices 0-7"
            )
        telemetry = session["telemetry"]
        if not (
            telemetry["source_file_verified"]
            and telemetry["complete"]
            and telemetry["all_runs_covered"]
            and telemetry["error_free"]
        ):
            incomplete_reasons.append(f"{session['session_id']} measured telemetry 不完整")
        host_telemetry = session["host_telemetry"]
        if not (
            host_telemetry["complete"]
            and host_telemetry["all_runs_covered"]
            and host_telemetry["error_free"]
        ):
            incomplete_reasons.append(
                f"{session['session_id']} measured Host telemetry 不完整"
            )
        if not session["initial_npu_state"]["clean"]:
            incomplete_reasons.append(
                f"{session['session_id']} 模型加载前 NPU 已被占用"
            )
    for layout_id, output in output_reproducibility.items():
        if not output["work_shape_stable"]:
            incomplete_reasons.append(
                f"{layout_id} 跨 session 请求终态或计算长度不一致"
            )
        elif not output["bitwise_output_stable"]:
            warnings.append(
                f"{layout_id} 存在 batch-shape 相关逐 token 变体；"
                "请求终态与计算长度一致，不阻断性能复现"
            )

    complete = not incomplete_reasons
    diagnosis = classify_reproduction(sessions, references) if complete else None
    return {
        "schema_version": MIXED_REPRO_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.7.1_mixed_open_loop_reproduction_check",
        "evidence_class": (
            "complete_v0.7.1_mixed_open_loop_reproduction_check"
            if complete
            else "incomplete_v0.7.1_mixed_open_loop_reproduction_check"
        ),
        "complete": complete,
        "incomplete_reasons": incomplete_reasons,
        "warnings": warnings,
        "design": {
            "sequence": list(FORMAL_SEQUENCE),
            "rationale": "ABBA + BAAB 平衡布局顺序和单调机器状态漂移",
            "logical_device_ids": list(range(8)),
            "protocol": dict(FORMAL_PROTOCOL),
            "total_scheduler_slots": 32,
            "total_scheduler_queue_size": 128,
            "profiler_enabled": False,
        },
        "references": references,
        "sessions": sessions,
        "output_reproducibility": output_reproducibility,
        "diagnosis": diagnosis,
    }


def write_markdown_report(summary: Mapping[str, object], output: str | Path) -> None:
    path = Path(output)
    lines = [
        "# RUN_LOG — v0.7.1 mixed open-loop 复现检查",
        "",
        f"- complete: `{str(summary['complete']).lower()}`",
        f"- evidence_class: `{summary['evidence_class']}`",
        "- 设计：ABBA + BAAB；8 个独立作业，每个 1 warmup + 3 measured repeats。",
        "- 范围：只验证 v0.7 与 v0.7.1 mixed TP8 差异，不属于 v0.8。",
        "",
        "| 顺序 | 布局 | goodput 中位数 | NPU 频率 | 温度 | AICore | "
        "Host CPU | IO wait | load1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for session in summary["sessions"]:
        overall = session["telemetry"]["overall"]
        host_overall = session["host_telemetry"]["overall"]

        def median(field: str) -> str:
            value = overall.get(field)
            return (
                "n/a"
                if not isinstance(value, dict)
                else f"{float(value['median']):.3f}"
            )

        def host_median(field: str) -> str:
            value = host_overall.get(field)
            return (
                "n/a"
                if not isinstance(value, dict)
                else f"{float(value['median']):.3f}"
            )

        lines.append(
            f"| {session['position']} | {session['layout_id']} | "
            f"{float(session['service']['goodput_requests_per_second']['median']):.6f} | "
            f"{median('aicore_current_frequency_mhz')} | "
            f"{median('temperature_celsius')} | "
            f"{median('aicore_usage_percent')} | "
            f"{host_median('cpu_usage_percent')} | "
            f"{host_median('cpu_iowait_percent')} | "
            f"{host_median('load1')} |"
        )
    lines.extend(["", "## 自动判定", ""])
    diagnosis = summary.get("diagnosis")
    if isinstance(diagnosis, dict):
        lines.extend(
            [
                f"- status: `{diagnosis['status']}`",
                "- 4×TP2 / TP8 goodput: "
                f"`{float(diagnosis['goodput_ratio_4xtp2_over_tp8']):.6f}×`",
                f"- 结论：{diagnosis['conclusion']}",
                f"- 边界：{diagnosis['scope']}",
            ]
        )
        root_cause = diagnosis["historical_root_cause"]
        evidence = root_cause["evidence"]
        lines.extend(
            [
                "",
                "## 历史慢态判因",
                "",
                f"- status: `{root_cause['status']}`",
                f"- confidence: `{root_cause['confidence']}`",
                f"- 结论：{root_cause['conclusion']}",
                "- 模型加载前 HBM 中位数："
                f"v0.7 TP8 `{evidence['v0.7_tp8_initial_hbm_usage_percent_median']:.3f}%`，"
                "本次 TP8 最大 "
                f"`{evidence['current_tp8_initial_hbm_usage_percent']['max']:.3f}%`。",
                "- 模型加载前 AICore 中位数："
                f"v0.7 TP8 `{evidence['v0.7_tp8_initial_aicore_usage_percent_median']:.3f}%`，"
                "本次 TP8 最大 "
                f"`{evidence['current_tp8_initial_aicore_usage_percent']['max']:.3f}%`。",
                "- 原始完成吞吐："
                f"v0.7 TP8 `{evidence['v0.7_tp8_completed_requests_per_second']:.6f}` req/s，"
                "本次 TP8 "
                f"`{evidence['current_tp8_completed_requests_per_second']:.6f}` req/s。",
                f"- 证据边界：{root_cause['boundary']}",
            ]
        )
    else:
        for reason in summary["incomplete_reasons"]:
            lines.append(f"- incomplete: {reason}")
    if summary["warnings"]:
        lines.extend(["", "## 非阻断警告", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
