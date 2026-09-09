"""汇总 TP8、2×TP4、4×TP2 同设备 Continuous Batching 报告。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .replica import partition_workload_by_projected_load
from .serving_benchmark import serving_output_digest, summarize_samples
from .serving_telemetry import summarize_telemetry
from .workload import WorkloadTrace


FORMAL_REPLICA_EVIDENCE = "qwen3_32b_ascend_continuous_batching_candidate"
QWEN3_32B_PARAMETERS = 32_762_123_264
FORMAL_LAYOUT_IDS = {
    "tp8": (1, 8),
    "2xtp4": (2, 4),
    "4xtp2": (4, 2),
}
WORKLOAD_CLASSES = frozenset(
    {"short_short", "long_prefill_short_decode", "mixed"}
)
_MANIFEST_VERIFIED = object()


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0.0


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_serving_report(path: str | Path) -> dict[str, object]:
    report_path = Path(path)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 Continuous Batching 报告：{report_path}") from exc
    report = _require_mapping(raw, str(report_path))
    required = {
        "benchmark",
        "protocol",
        "environment",
        "workload",
        "model",
        "distributed",
        "provenance",
        "runs",
        "evidence_class",
    }
    if not required.issubset(report):
        raise ValueError(f"不是完整 Continuous Batching 报告：{report_path}")
    if report["benchmark"] != "continuous_batching_trace_replay":
        raise ValueError(f"benchmark 类型不正确：{report_path}")
    # 这是加载器内部的信任标记，不能允许报告 JSON 自己声明已经通过 manifest。
    report.pop("_layout_manifest_verified", None)
    report.pop("_layout_manifest_artifact", None)
    report.pop("_source_workload_artifact", None)
    report.pop("_source_workload_trace", None)
    return report


def load_layout_manifest(
    path: str | Path,
) -> tuple[str, list[tuple[Path, dict[str, object]]]]:
    """按 manifest 的相对路径与 SHA-256 加载一个完整 layout。"""

    manifest_path = Path(path)
    try:
        manifest_payload = manifest_path.read_bytes()
        raw = json.loads(manifest_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 layout manifest：{manifest_path}") from exc
    manifest = _require_mapping(raw, str(manifest_path))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"不支持的 layout manifest：{manifest_path}")
    layout_id = str(manifest.get("layout_id", ""))
    entries = manifest.get("reports")
    if not layout_id or not isinstance(entries, list) or not entries:
        raise ValueError(f"layout manifest 缺少 layout_id/reports：{manifest_path}")

    source_entry = _require_mapping(
        manifest.get("source_workload"),
        "manifest.source_workload",
    )
    source_name = str(source_entry.get("name", ""))
    if not source_name or Path(source_name).name != source_name:
        raise ValueError("layout manifest source workload name 必须是单个相对文件名")
    source_path = manifest_path.parent / source_name
    try:
        source_payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取 manifest source workload：{source_path}") from exc
    if int(source_entry.get("size_bytes", -1)) != len(source_payload):
        raise ValueError(f"manifest source workload size 不一致：{source_path}")
    source_file_sha256 = hashlib.sha256(source_payload).hexdigest()
    if source_entry.get("sha256") != source_file_sha256:
        raise ValueError(f"manifest source workload SHA-256 不一致：{source_path}")
    try:
        source_trace = WorkloadTrace.from_dict(
            json.loads(source_payload.decode("utf-8"))
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"manifest source workload 内容无效：{source_path}") from exc
    if source_trace.partition is not None:
        raise ValueError("layout manifest 必须绑定未分片的源 workload")
    if manifest.get("source_workload_sha256") != source_trace.request_sha256:
        raise ValueError("layout manifest source workload 语义 SHA-256 不一致")
    if manifest.get("source_workload_file_sha256") != source_file_sha256:
        raise ValueError("layout manifest source workload 文件 SHA-256 不一致")
    if manifest.get("workload_class") != source_trace.workload_class:
        raise ValueError("layout manifest workload_class 与源 workload 不一致")
    source_artifact = {
        "path": str(source_path),
        "sha256": source_file_sha256,
        "size_bytes": len(source_payload),
    }

    reports: list[tuple[Path, dict[str, object]]] = []
    indices: set[int] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"manifest.reports[{index}]")
        name = str(entry.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError("layout manifest report name 必须是单个相对文件名")
        report_path = manifest_path.parent / name
        try:
            payload = report_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"无法读取 manifest report：{report_path}") from exc
        if int(entry.get("size_bytes", -1)) != len(payload):
            raise ValueError(f"manifest report size 不一致：{report_path}")
        if entry.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"manifest report SHA-256 不一致：{report_path}")
        replica_index = int(entry.get("replica_index", -1))
        if replica_index in indices:
            raise ValueError("layout manifest replica_index 重复")
        indices.add(replica_index)
        report = load_serving_report(report_path)
        distributed = _require_mapping(report["distributed"], "distributed")
        if int(distributed.get("replica_index", -1)) != replica_index:
            raise ValueError(
                f"manifest replica_index 与报告身份不一致：{report_path}"
            )
        report["_layout_manifest_verified"] = _MANIFEST_VERIFIED
        report["_layout_manifest_artifact"] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "size_bytes": len(manifest_payload),
        }
        report["_source_workload_artifact"] = dict(source_artifact)
        report["_source_workload_trace"] = source_trace
        reports.append((report_path, report))
    if indices != set(range(len(entries))):
        raise ValueError("layout manifest replica_index 不连续")

    first_report = reports[0][1]
    first_distributed = _require_mapping(
        first_report["distributed"],
        "distributed",
    )
    first_workload = _require_mapping(first_report["workload"], "workload")
    expected_manifest = {
        "layout_id": first_distributed.get("layout_id"),
        "tp_size": first_distributed.get("tp_size"),
        "replica_count": first_distributed.get("replica_count"),
        "global_world_size": first_distributed.get("global_world_size"),
        "global_logical_device_ids": first_distributed.get(
            "global_logical_device_ids"
        ),
        "source_workload_sha256": first_workload.get("source_sha256"),
        "source_workload_file_sha256": first_workload.get(
            "source_file_sha256"
        ),
        "workload_class": first_workload.get("workload_class"),
        "routing_assignment_sha256": _require_mapping(
            first_workload.get("routing"),
            "workload.routing",
        ).get("assignment_sha256"),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"layout manifest {field} 与报告不一致")
    if layout_id != expected_manifest["layout_id"]:
        raise ValueError("layout manifest layout_id 与报告不一致")
    return layout_id, reports


def _routing_assignments(
    routing: dict[str, object],
    *,
    replica_count: int,
) -> dict[int, set[str]]:
    if routing.get("router") != "least_projected_load":
        raise ValueError("routing.router 必须是 least_projected_load")
    if routing.get("estimator") != (
        "prompt_unicode_codepoints_plus_max_new_tokens"
    ):
        raise ValueError("routing.estimator 不受支持")
    raw_assignments = routing.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("routing.assignments 必须是非空数组")
    if routing.get("assignment_sha256") != _canonical_sha256(raw_assignments):
        raise ValueError("routing assignment_sha256 与 assignments 不一致")

    by_replica = {index: set() for index in range(replica_count)}
    all_request_ids: set[str] = set()
    for index, raw_assignment in enumerate(raw_assignments):
        assignment = _require_mapping(
            raw_assignment,
            f"routing.assignments[{index}]",
        )
        request_id = str(assignment.get("request_id", ""))
        replica_index = int(assignment.get("replica_index", -1))
        estimated_work = int(assignment.get("estimated_work", 0))
        if not request_id or request_id in all_request_ids:
            raise ValueError("routing assignments 的 request_id 为空或重复")
        if not 0 <= replica_index < replica_count:
            raise ValueError("routing assignment replica_index 越界")
        if estimated_work <= 0:
            raise ValueError("routing assignment estimated_work 必须大于 0")
        all_request_ids.add(request_id)
        by_replica[replica_index].add(request_id)
    if any(not request_ids for request_ids in by_replica.values()):
        raise ValueError("routing assignments 不能产生空 replica")
    return by_replica


def _protocol_fingerprint(report: dict[str, object]) -> dict[str, object]:
    protocol = _require_mapping(report["protocol"], "protocol")
    distributed = _require_mapping(report["distributed"], "distributed")
    return {
        "mode": protocol.get("mode"),
        "open_loop_admission_scripted": protocol.get(
            "open_loop_admission_scripted",
            False,
        ),
        "warmup": protocol.get("warmup"),
        "repeats": protocol.get("repeats"),
        "ttft_slo_ms": protocol.get("ttft_slo_ms"),
        "tpot_slo_ms": protocol.get("tpot_slo_ms"),
        "e2e_slo_ms": protocol.get("e2e_slo_ms"),
        "global_closed_loop_clients": distributed.get(
            "global_closed_loop_clients"
        ),
    }


def _execution_fingerprint(report: dict[str, object]) -> dict[str, object]:
    environment = _require_mapping(report["environment"], "environment")
    distributed = _require_mapping(report["distributed"], "distributed")
    return {
        "python": environment.get("python"),
        "pytorch": environment.get("pytorch"),
        "torch": environment.get("torch"),
        "torch_npu": environment.get("torch_npu"),
        "cann_version": environment.get("cann_version"),
        "device_type": environment.get("device_type"),
        "device_name": environment.get("device_name"),
        "precision": environment.get("precision"),
        "total_memory_mb": environment.get("total_memory_mb"),
        "backend": distributed.get("backend"),
        "hostname": distributed.get("hostname"),
        "visible_device_count": distributed.get("visible_device_count"),
        "physical_card_count": distributed.get("physical_card_count"),
        "chips_per_card": distributed.get("chips_per_card"),
        "interconnect_topology": distributed.get("interconnect_topology"),
        "global_logical_device_ids": distributed.get("global_logical_device_ids"),
    }


def _model_fingerprint(report: dict[str, object]) -> dict[str, object]:
    model = _require_mapping(report["model"], "model")
    provenance = _require_mapping(report["provenance"], "provenance")
    git = _require_mapping(provenance.get("git"), "provenance.git")
    return {
        "type": model.get("type"),
        "config": model.get("config"),
        "full_parameter_count": model.get("full_parameter_count"),
        "config_sha256": provenance.get("config_sha256"),
        "metadata_sha256": provenance.get("metadata_sha256"),
        "weights": provenance.get("weights"),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
    }


def _engine_fingerprint(report: dict[str, object]) -> dict[str, int | str]:
    engine = _require_mapping(report.get("engine"), "engine")
    return {
        "runner": str(engine.get("runner", "")),
        "max_slots": int(engine.get("max_slots", -1)),
        "max_seq_len": int(engine.get("max_seq_len", -1)),
        "max_queue_size": int(engine.get("max_queue_size", -1)),
    }


def _is_formal_replica_report(report: dict[str, object]) -> bool:
    """从原始字段重算候选资格，不能信任报告自带的 evidence_class。"""

    protocol = _require_mapping(report["protocol"], "protocol")
    environment = _require_mapping(report["environment"], "environment")
    distributed = _require_mapping(report["distributed"], "distributed")
    workload = _require_mapping(report["workload"], "workload")
    model = _require_mapping(report["model"], "model")
    provenance = _require_mapping(report["provenance"], "provenance")
    git = _require_mapping(provenance.get("git"), "provenance.git")
    weights = provenance.get("weights")
    valid_weights = (
        provenance.get("weight_hashes_included") is True
        and isinstance(weights, list)
        and bool(weights)
        and all(
            isinstance(weight, dict)
            and bool(str(weight.get("name", "")))
            and int(weight.get("size_bytes", 0)) > 0
            and _is_sha256(weight.get("sha256"))
            for weight in weights
        )
    )
    commit = git.get("commit")
    valid_commit = (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    )
    return bool(
        report.get("evidence_class") == FORMAL_REPLICA_EVIDENCE
        and model.get("type") == "TensorParallelQwen3ForCausalLM"
        and int(model.get("full_parameter_count", -1)) == QWEN3_32B_PARAMETERS
        and environment.get("device_type") == "npu"
        and environment.get("precision") == "bf16"
        and bool(str(environment.get("torch_npu") or "").strip())
        and str(environment.get("torch_npu")).lower() != "unknown"
        and bool(str(environment.get("cann_version") or "").strip())
        and distributed.get("backend") == "hccl"
        and int(distributed.get("global_world_size", -1)) == 8
        and valid_weights
        and valid_commit
        and git.get("dirty") is False
        and int(protocol.get("warmup", -1)) >= 1
        and int(protocol.get("repeats", -1)) >= 3
        and protocol.get("mode") in {"open_loop", "closed_loop"}
        and workload.get("workload_class") in WORKLOAD_CLASSES
        and _is_sha256(workload.get("source_file_sha256"))
        and all(
            _is_positive_finite(protocol.get(field))
            for field in ("ttft_slo_ms", "tpot_slo_ms", "e2e_slo_ms")
        )
    )


def _request_metrics(run: dict[str, object]) -> list[dict[str, object]]:
    serving = _require_mapping(run["serving"], "run.serving")
    requests = serving.get("requests")
    if not isinstance(requests, list) or any(
        not isinstance(request, dict) for request in requests
    ):
        raise ValueError("run.serving.requests 必须是对象数组")
    return requests


def _validate_request_metric(request: dict[str, object], *, path: Path) -> None:
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"request_id 为空或类型错误：{path}")
    state = request.get("state")
    if state not in {"finished", "cancelled", "rejected", "failed"}:
        raise ValueError(f"measured run 包含非终态请求：{path}")
    stop_reason = request.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError(f"终态请求缺少 stop_reason：{path}")
    if not isinstance(request.get("slo_met"), bool):
        raise ValueError(f"request.slo_met 必须是布尔值：{path}")
    if state != "finished" and request["slo_met"] is not False:
        raise ValueError(f"未完成请求不能标记为满足 SLO：{path}")

    for count_field, ids_field in (
        ("input_tokens", "prompt_ids"),
        ("output_tokens", "generated_ids"),
    ):
        count = request.get(count_field)
        ids = request.get(ids_field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{count_field} 必须是非负整数：{path}")
        if not isinstance(ids, list) or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in ids
        ):
            raise ValueError(f"{ids_field} 必须是非负整数数组：{path}")
        if len(ids) != count:
            raise ValueError(f"{count_field} 与 {ids_field} 长度不一致：{path}")
    if int(request["input_tokens"]) <= 0:
        raise ValueError(f"请求 input_tokens 必须大于 0：{path}")

    for field in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms"):
        value = request.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} 必须是数值或 null：{path}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{field} 必须是非负有限值：{path}")
    if state == "finished" and any(
        request.get(field) is None
        for field in ("queue_ms", "ttft_ms", "e2e_latency_ms")
    ):
        raise ValueError(f"完成请求缺少 queue/TTFT/E2E 指标：{path}")
    deadline_ms = request.get("deadline_ms")
    if deadline_ms is not None and not _is_positive_finite(deadline_ms):
        raise ValueError(f"request.deadline_ms 必须是正有限值或 null：{path}")


def _request_meets_slo(
    request: dict[str, object],
    protocol: Mapping[str, object],
) -> bool:
    if request["state"] != "finished":
        return False
    request_e2e_limit = request.get("deadline_ms") or protocol.get(
        "e2e_slo_ms"
    )
    checks = (
        (protocol.get("ttft_slo_ms"), request.get("ttft_ms")),
        (protocol.get("tpot_slo_ms"), request.get("tpot_ms")),
        (request_e2e_limit, request.get("e2e_latency_ms")),
    )
    return all(
        limit is None
        or (value is not None and float(value) <= float(limit))
        for limit, value in checks
    )


def _summarize_layout(
    layout_id: str,
    reports: Sequence[tuple[Path, dict[str, object]]],
    *,
    max_start_skew_ms: float,
    telemetry: Mapping[str, object] | None,
    min_telemetry_samples_per_device_per_run: int,
) -> dict[str, object]:
    if not reports:
        raise ValueError(f"layout {layout_id!r} 没有报告")
    first = reports[0][1]
    first_distributed = _require_mapping(first["distributed"], "distributed")
    replica_count = int(first_distributed["replica_count"])
    tp_size = int(first_distributed["tp_size"])
    global_world_size = int(first_distributed["global_world_size"])
    if len(reports) != replica_count:
        raise ValueError(
            f"layout {layout_id!r} 需要 {replica_count} 份 replica 报告，"
            f"实际 {len(reports)}"
        )

    first_workload = _require_mapping(first["workload"], "workload")
    source_sha256 = first_workload.get("source_sha256")
    source_file_sha256 = first_workload.get("source_file_sha256")
    if not _is_sha256(source_sha256):
        raise ValueError(f"layout {layout_id!r} 的 source_sha256 无效")
    if not _is_sha256(source_file_sha256):
        raise ValueError(f"layout {layout_id!r} 的 source_file_sha256 无效")
    workload_class = str(first_workload.get("workload_class", ""))
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError(f"layout {layout_id!r} 的 workload_class 不受支持")
    first_routing = _require_mapping(
        first_workload.get("routing"),
        "workload.routing",
    )
    routing_fingerprint = {
        "router": first_routing.get("router"),
        "estimator": first_routing.get("estimator"),
        "assignment_sha256": first_routing.get("assignment_sha256"),
    }
    protocol_fingerprint = _protocol_fingerprint(first)
    execution_fingerprint = _execution_fingerprint(first)
    if not str(execution_fingerprint.get("hostname") or ""):
        raise ValueError(f"layout {layout_id!r} 缺少 hostname")
    if int(execution_fingerprint.get("visible_device_count") or 0) < (
        global_world_size
    ):
        raise ValueError(f"layout {layout_id!r} 可见设备数少于 global world_size")
    if (
        int(execution_fingerprint.get("physical_card_count") or 0)
        * int(execution_fingerprint.get("chips_per_card") or 0)
        != global_world_size
    ):
        raise ValueError(f"layout {layout_id!r} 的物理卡/芯片拓扑不守恒")
    model_fingerprint = _model_fingerprint(first)
    engine_fingerprint = _engine_fingerprint(first)
    if min(
        int(engine_fingerprint["max_slots"]),
        int(engine_fingerprint["max_seq_len"]),
        int(engine_fingerprint["max_queue_size"]),
    ) <= 0:
        raise ValueError(f"layout {layout_id!r} 的 engine 容量无效")
    assigned_request_ids = _routing_assignments(
        first_routing,
        replica_count=replica_count,
    )
    source_trace = first.get("_source_workload_trace")
    expected_partition_sha256: dict[int, str] = {}
    if isinstance(source_trace, WorkloadTrace):
        expected_partitions, expected_assignments = (
            partition_workload_by_projected_load(
                source_trace,
                replica_count=replica_count,
                max_slots_per_replica=int(engine_fingerprint["max_slots"]),
            )
        )
        if first_routing.get("assignments") != expected_assignments:
            raise ValueError(
                f"layout {layout_id!r} 的 routing 不是源 workload 的确定性 "
                "least-projected-load 结果"
            )
        expected_partition_sha256 = {
            int(partition.partition.replica_index): partition.request_sha256
            for partition in expected_partitions
            if partition.partition is not None
        }
    repeats = int(first["protocol"]["repeats"])
    expected_indices = set(range(replica_count))
    actual_indices: set[int] = set()
    used_devices: set[int] = set()

    for path, report in reports:
        distributed = _require_mapping(report["distributed"], "distributed")
        workload = _require_mapping(report["workload"], "workload")
        if distributed.get("layout_id") != layout_id:
            raise ValueError(f"layout_id 与分组名称不一致：{path}")
        if (
            int(distributed.get("replica_count", -1)) != replica_count
            or int(distributed.get("tp_size", -1)) != tp_size
            or int(distributed.get("global_world_size", -1)) != global_world_size
        ):
            raise ValueError(f"同一 layout 的 TP/replica 拓扑不一致：{path}")
        actual_indices.add(int(distributed["replica_index"]))
        local_devices = distributed.get("logical_device_ids")
        if not isinstance(local_devices, list) or len(local_devices) != tp_size:
            raise ValueError(f"logical_device_ids 数量与 TP size 不一致：{path}")
        for device_id in local_devices:
            numeric_id = int(device_id)
            if numeric_id in used_devices:
                raise ValueError(f"同一 layout 重复使用 device {numeric_id}：{path}")
            used_devices.add(numeric_id)
        per_rank = distributed.get("per_rank")
        if not isinstance(per_rank, list) or len(per_rank) != tp_size:
            raise ValueError(f"per_rank 数量与 TP size 不一致：{path}")
        per_rank_devices: set[int] = set()
        for raw_rank in per_rank:
            rank = _require_mapping(raw_rank, "distributed.per_rank")
            per_rank_devices.add(int(rank.get("logical_device_id", -1)))
            peak_mb = float(rank.get("max_measured_peak_mb", -1.0))
            if not math.isfinite(peak_mb) or peak_mb < 0.0:
                raise ValueError(f"per_rank peak memory 无效：{path}")
        if per_rank_devices != {int(value) for value in local_devices}:
            raise ValueError(f"per_rank device 与 logical_device_ids 不一致：{path}")
        if workload.get("source_sha256") != source_sha256:
            raise ValueError(f"同一 layout 的源 workload 不一致：{path}")
        if workload.get("source_file_sha256") != source_file_sha256:
            raise ValueError(f"同一 layout 的源 workload 文件不一致：{path}")
        if workload.get("workload_class") != workload_class:
            raise ValueError(f"同一 layout 的 workload_class 不一致：{path}")
        routing = _require_mapping(workload.get("routing"), "workload.routing")
        if {
            "router": routing.get("router"),
            "estimator": routing.get("estimator"),
            "assignment_sha256": routing.get("assignment_sha256"),
        } != routing_fingerprint:
            raise ValueError(f"同一 layout 的 routing manifest 不一致：{path}")
        _routing_assignments(routing, replica_count=replica_count)
        partition = workload.get("partition")
        if replica_count == 1:
            if partition is not None:
                raise ValueError(f"单副本 layout 不应带 workload partition：{path}")
        else:
            partition_mapping = _require_mapping(
                partition,
                "workload.partition",
            )
            if (
                int(partition_mapping.get("replica_count", -1)) != replica_count
                or int(partition_mapping.get("replica_index", -1))
                != int(distributed["replica_index"])
                or partition_mapping.get("source_sha256") != source_sha256
            ):
                raise ValueError(f"workload partition 与 replica 身份不一致：{path}")
        if _protocol_fingerprint(report) != protocol_fingerprint:
            raise ValueError(f"同一 layout 的测量协议不一致：{path}")
        if _execution_fingerprint(report) != execution_fingerprint:
            raise ValueError(f"同一 layout 的硬件/软件环境不一致：{path}")
        if _model_fingerprint(report) != model_fingerprint:
            raise ValueError(f"同一 layout 的模型/权重/提交不一致：{path}")
        if _engine_fingerprint(report) != engine_fingerprint:
            raise ValueError(f"同一 layout 的 engine 容量或 runner 不一致：{path}")
        runs = report["runs"]
        if not isinstance(runs, list) or len(runs) != repeats:
            raise ValueError(f"measured repeats 数量不正确：{path}")
        replica_index = int(distributed["replica_index"])
        expected_ids = assigned_request_ids[replica_index]
        declared_request_count = int(workload.get("request_count", -1))
        if declared_request_count != len(expected_ids):
            raise ValueError(f"workload request_count 与 routing 分配不一致：{path}")
        if expected_partition_sha256 and workload.get("request_sha256") != (
            expected_partition_sha256[replica_index]
        ):
            raise ValueError(f"workload request_sha256 与源 workload 分片不一致：{path}")
        previous_end = 0
        expected_output_sha256: str | None = None
        for repeat, run in enumerate(runs):
            run_mapping = _require_mapping(run, f"runs[{repeat}]")
            if int(run_mapping.get("repeat", -1)) != repeat:
                raise ValueError(f"run repeat 编号不连续：{path}")
            replay = _require_mapping(run_mapping.get("replay"), "run.replay")
            started_at_ns = int(replay.get("started_at_unix_ns", -1))
            ended_at_ns = int(replay.get("ended_at_unix_ns", -1))
            if started_at_ns <= 0 or ended_at_ns <= started_at_ns:
                raise ValueError(f"measured run 时间区间无效：{path}")
            if started_at_ns < previous_end:
                raise ValueError(f"measured runs 时间区间重叠：{path}")
            previous_end = ended_at_ns
            serving = _require_mapping(run_mapping.get("serving"), "run.serving")
            request_metrics = _request_metrics(run_mapping)
            for request in request_metrics:
                _validate_request_metric(request, path=path)
                if request["slo_met"] is not _request_meets_slo(
                    request,
                    protocol_fingerprint,
                ):
                    raise ValueError(
                        f"request.slo_met 与延迟/SLO 重算结果不一致：{path}"
                    )
            run_request_ids = {
                str(request.get("request_id", "")) for request in request_metrics
            }
            if run_request_ids != expected_ids:
                raise ValueError(f"run 请求集合与 routing assignment 不一致：{path}")
            kv_cache = _require_mapping(serving.get("kv_cache"), "serving.kv_cache")
            if int(kv_cache.get("slot_capacity", -1)) != int(
                engine_fingerprint["max_slots"]
            ):
                raise ValueError(f"KV slot capacity 与 engine.max_slots 不一致：{path}")
            output_sha256 = run_mapping.get("output_sha256")
            if not _is_sha256(output_sha256):
                raise ValueError(f"run output_sha256 无效：{path}")
            if serving_output_digest(serving) != output_sha256:
                raise ValueError(f"run output_sha256 与逐请求输出不一致：{path}")
            if expected_output_sha256 is None:
                expected_output_sha256 = str(output_sha256)
            elif output_sha256 != expected_output_sha256:
                raise ValueError(f"同一 replica 的重复运行输出不一致：{path}")

    if actual_indices != expected_indices:
        raise ValueError(f"layout {layout_id!r} 的 replica index 不完整")
    expected_devices = {
        int(value) for value in first_distributed["global_logical_device_ids"]
    }
    if used_devices != expected_devices or len(used_devices) != global_world_size:
        raise ValueError(f"layout {layout_id!r} 没有恰好覆盖声明的全部 devices")

    if protocol_fingerprint["mode"] == "closed_loop":
        total_clients = sum(
            int(report["distributed"]["replica_closed_loop_clients"])
            for _path, report in reports
        )
        if total_clients != int(protocol_fingerprint["global_closed_loop_clients"]):
            raise ValueError(f"layout {layout_id!r} 的 closed-loop client 拆分不守恒")

    layout_runs: list[dict[str, object]] = []
    expected_request_ids: set[str] | None = None
    max_observed_skew_ms = 0.0
    for repeat in range(repeats):
        replica_runs = [report["runs"][repeat] for _path, report in reports]
        starts = [int(run["replay"]["started_at_unix_ns"]) for run in replica_runs]
        ends = [int(run["replay"]["ended_at_unix_ns"]) for run in replica_runs]
        start_skew_ms = (max(starts) - min(starts)) / 1_000_000.0
        max_observed_skew_ms = max(max_observed_skew_ms, start_skew_ms)
        duration_seconds = max((max(ends) - min(starts)) / 1_000_000_000.0, 1e-12)
        requests = [
            request
            for run in replica_runs
            for request in _request_metrics(run)
        ]
        request_ids = [str(request["request_id"]) for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"layout {layout_id!r} 的 replica workload 有重复请求")
        current_ids = set(request_ids)
        if expected_request_ids is None:
            expected_request_ids = current_ids
        elif current_ids != expected_request_ids:
            raise ValueError(f"layout {layout_id!r} 不同 repeat 的请求集合不一致")

        completed = [request for request in requests if request["state"] == "finished"]
        good = [request for request in completed if request["slo_met"]]
        input_tokens = sum(int(request["input_tokens"]) for request in completed)
        output_tokens = sum(int(request["output_tokens"]) for request in completed)
        latency: dict[str, object] = {}
        for field in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms"):
            latency[field] = summarize_samples(
                [
                    float(request[field])
                    for request in completed
                    if request[field] is not None
                ]
            )
        layout_runs.append(
            {
                "repeat": repeat,
                "started_at_unix_ns": min(starts),
                "ended_at_unix_ns": max(ends),
                "start_skew_ms": start_skew_ms,
                "wall_time_ms": duration_seconds * 1000.0,
                "requests": len(requests),
                "completed_requests": len(completed),
                "good_requests": len(good),
                "state_counts": {
                    state: sum(request["state"] == state for request in requests)
                    for state in (
                        "finished",
                        "cancelled",
                        "rejected",
                        "failed",
                    )
                },
                "completed_requests_per_second": len(completed) / duration_seconds,
                "goodput_requests_per_second": len(good) / duration_seconds,
                "input_tokens_per_second": input_tokens / duration_seconds,
                "output_tokens_per_second": output_tokens / duration_seconds,
                "latency": latency,
            }
        )

    metric_fields = (
        "completed_requests_per_second",
        "goodput_requests_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "wall_time_ms",
        "start_skew_ms",
    )
    summary: dict[str, object] = {
        field: summarize_samples([float(run[field]) for run in layout_runs])
        for field in metric_fields
    }
    for field in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms"):
        summary[field] = summarize_samples(
            [
                float(request[field])
                for _path, report in reports
                for run in report["runs"]
                for request in _request_metrics(run)
                if request["state"] == "finished" and request[field] is not None
            ]
        )
    per_rank_memory = [
        rank
        for _path, report in reports
        for rank in report["distributed"]["per_rank"]
    ]
    serving_runs = [
        run["serving"]
        for _path, report in reports
        for run in report["runs"]
    ]
    batching = {
        field: summarize_samples(
            [
                float(step[field])
                for serving in serving_runs
                for step in serving["steps"]
            ]
        )
        for field in (
            "active_after",
            "decode_batch_size",
            "prefill_batch_size",
        )
    }
    kv_runs = [serving["kv_cache"] for serving in serving_runs]
    kv_cache = {
        "total_slot_capacity": sum(
            int(report["runs"][0]["serving"]["kv_cache"]["slot_capacity"])
            for _path, report in reports
        ),
        "max_replica_peak_slots_used": max(
            int(kv["peak_slots_used"]) for kv in kv_runs
        ),
        "max_replica_peak_used_tokens": max(
            int(kv["peak_used_tokens"]) for kv in kv_runs
        ),
        "max_replica_internal_waste_tokens": max(
            int(kv["peak_internal_waste_tokens"]) for kv in kv_runs
        ),
        "external_fragmentation_tokens": max(
            int(kv["external_fragmentation_tokens"]) for kv in kv_runs
        ),
    }
    scheduler_capacity = {
        "runner": str(engine_fingerprint["runner"]),
        "total_max_slots": sum(
            int(_engine_fingerprint(report)["max_slots"])
            for _path, report in reports
        ),
        "total_max_queue_size": sum(
            int(_engine_fingerprint(report)["max_queue_size"])
            for _path, report in reports
        ),
        "per_replica_max_slots": int(engine_fingerprint["max_slots"]),
        "per_replica_max_queue_size": int(
            engine_fingerprint["max_queue_size"]
        ),
        "max_seq_len": int(engine_fingerprint["max_seq_len"]),
    }
    max_rank_peak_mb = max(
        float(rank["max_measured_peak_mb"]) for rank in per_rank_memory
    )
    sum_rank_peak_mb = sum(
        float(rank["max_measured_peak_mb"]) for rank in per_rank_memory
    )
    manifest_verified = all(
        report.get("_layout_manifest_verified") is _MANIFEST_VERIFIED
        for _path, report in reports
    )
    manifest_artifacts = {
        json.dumps(
            report.get("_layout_manifest_artifact"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for _path, report in reports
        if report.get("_layout_manifest_verified") is _MANIFEST_VERIFIED
    }
    if manifest_verified and len(manifest_artifacts) != 1:
        raise ValueError(f"layout {layout_id!r} 的 manifest artifact 不一致")
    manifest_artifact = (
        json.loads(next(iter(manifest_artifacts)))
        if manifest_verified
        else None
    )
    source_workload_artifacts = {
        json.dumps(
            report.get("_source_workload_artifact"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for _path, report in reports
        if report.get("_layout_manifest_verified") is _MANIFEST_VERIFIED
    }
    if manifest_verified and len(source_workload_artifacts) != 1:
        raise ValueError(f"layout {layout_id!r} 的 source workload artifact 不一致")
    source_workload_artifact = (
        json.loads(next(iter(source_workload_artifacts)))
        if manifest_verified
        else None
    )
    all_formal_candidates = all(
        _is_formal_replica_report(report)
        for _path, report in reports
    )
    run_intervals = [
        (int(run["started_at_unix_ns"]), int(run["ended_at_unix_ns"]))
        for run in layout_runs
    ]
    telemetry_summary: dict[str, object]
    if telemetry is None:
        telemetry_summary = {
            "available": False,
            "all_runs_covered": False,
            "error_free": False,
            "reason": "没有提供与本 layout measured runs 对齐的设备遥测",
        }
    else:
        telemetry_summary = {
            "available": True,
            **summarize_telemetry(
                telemetry,
                run_intervals=run_intervals,
                expected_logical_device_ids=sorted(used_devices),
                min_samples_per_device_per_run=(
                    min_telemetry_samples_per_device_per_run
                ),
            ),
        }
    telemetry_formal = bool(
        telemetry_summary["available"]
        and telemetry_summary["source_file_verified"]
        and telemetry_summary["complete"]
        and telemetry_summary["all_runs_covered"]
        and telemetry_summary["error_free"]
        and telemetry_summary["overall"]["aicore_usage_percent"] is not None
    )
    return {
        "layout_id": layout_id,
        "tp_size": tp_size,
        "replica_count": replica_count,
        "global_world_size": global_world_size,
        "logical_device_ids": sorted(used_devices),
        "source_workload_sha256": source_sha256,
        "source_workload_file_sha256": source_file_sha256,
        "workload_class": workload_class,
        "routing": routing_fingerprint,
        "max_start_skew_ms": max_observed_skew_ms,
        "start_skew_within_limit": max_observed_skew_ms <= max_start_skew_ms,
        "max_rank_peak_device_memory_mb": max_rank_peak_mb,
        "sum_rank_peak_device_memory_mb": sum_rank_peak_mb,
        "all_replica_reports_formal_candidates": all_formal_candidates,
        "layout_manifest_verified": manifest_verified,
        "layout_manifest": manifest_artifact,
        "source_workload": source_workload_artifact,
        "telemetry_formal": telemetry_formal,
        "telemetry": telemetry_summary,
        "batching": batching,
        "scheduler_capacity": scheduler_capacity,
        "kv_cache": kv_cache,
        "summary": summary,
        "runs": layout_runs,
        "reports": [str(path) for path, _report in reports],
    }


def summarize_serving_layouts(
    layouts: Mapping[str, Sequence[tuple[Path, dict[str, object]]]],
    *,
    baseline_layout: str,
    max_start_skew_ms: float = 100.0,
    telemetry_by_layout: Mapping[str, Mapping[str, object]] | None = None,
    min_telemetry_samples_per_device_per_run: int = 2,
) -> dict[str, object]:
    if len(layouts) < 2:
        raise ValueError("至少需要两个 layouts 才能比较")
    if baseline_layout not in layouts:
        raise ValueError("baseline_layout 不在输入 layouts 中")
    if max_start_skew_ms < 0.0:
        raise ValueError("max_start_skew_ms 不能小于 0")
    if min_telemetry_samples_per_device_per_run <= 0:
        raise ValueError("min_telemetry_samples_per_device_per_run 必须大于 0")
    unknown_telemetry = set(telemetry_by_layout or {}) - set(layouts)
    if unknown_telemetry:
        raise ValueError(
            f"telemetry 包含未提供报告的 layout：{sorted(unknown_telemetry)}"
        )
    rows = [
        _summarize_layout(
            name,
            reports,
            max_start_skew_ms=max_start_skew_ms,
            telemetry=(telemetry_by_layout or {}).get(name),
            min_telemetry_samples_per_device_per_run=(
                min_telemetry_samples_per_device_per_run
            ),
        )
        for name, reports in layouts.items()
    ]
    baseline = next(row for row in rows if row["layout_id"] == baseline_layout)
    baseline_request_rate = float(
        baseline["summary"]["completed_requests_per_second"]["median"]
    )
    baseline_goodput = float(
        baseline["summary"]["goodput_requests_per_second"]["median"]
    )

    first_report = next(iter(layouts.values()))[0][1]
    protocol_fingerprint = _protocol_fingerprint(first_report)
    execution_fingerprint = _execution_fingerprint(first_report)
    model_fingerprint = _model_fingerprint(first_report)
    source_sha256 = first_report["workload"]["source_sha256"]
    source_file_sha256 = first_report["workload"]["source_file_sha256"]
    workload_class = first_report["workload"]["workload_class"]
    for reports in layouts.values():
        for path, report in reports:
            if _protocol_fingerprint(report) != protocol_fingerprint:
                raise ValueError(f"不同 layout 的测量协议不一致：{path}")
            if _execution_fingerprint(report) != execution_fingerprint:
                raise ValueError(f"不同 layout 的硬件/软件环境不一致：{path}")
            if _model_fingerprint(report) != model_fingerprint:
                raise ValueError(f"不同 layout 的模型/权重/提交不一致：{path}")
            if report["workload"]["source_sha256"] != source_sha256:
                raise ValueError(f"不同 layout 的源 workload 不一致：{path}")
            if report["workload"]["source_file_sha256"] != source_file_sha256:
                raise ValueError(f"不同 layout 的源 workload 文件不一致：{path}")
            if report["workload"]["workload_class"] != workload_class:
                raise ValueError(f"不同 layout 的 workload_class 不一致：{path}")

    baseline_request_ids = {
        str(request["request_id"])
        for _path, report in layouts[baseline_layout]
        for run in report["runs"][:1]
        for request in _request_metrics(run)
    }
    for layout_name, reports in layouts.items():
        ids = {
            str(request["request_id"])
            for _path, report in reports
            for run in report["runs"][:1]
            for request in _request_metrics(run)
        }
        if ids != baseline_request_ids:
            raise ValueError(f"layout {layout_name!r} 与 baseline 请求集合不一致")

    for row in rows:
        request_rate = float(
            row["summary"]["completed_requests_per_second"]["median"]
        )
        goodput = float(row["summary"]["goodput_requests_per_second"]["median"])
        row["speedup_vs_baseline"] = {
            "completed_requests_per_second": (
                None
                if baseline_request_rate == 0.0
                else request_rate / baseline_request_rate
            ),
            "goodput_requests_per_second": (
                None if baseline_goodput == 0.0 else goodput / baseline_goodput
            ),
        }

    actual_layouts = {
        str(row["layout_id"]): (
            int(row["replica_count"]),
            int(row["tp_size"]),
        )
        for row in rows
    }
    same_devices = len({tuple(row["logical_device_ids"]) for row in rows}) == 1
    same_eight_devices = same_devices and len(rows[0]["logical_device_ids"]) == 8
    capacity_fingerprints = {
        (
            str(row["scheduler_capacity"]["runner"]),
            int(row["scheduler_capacity"]["total_max_slots"]),
            int(row["scheduler_capacity"]["total_max_queue_size"]),
            int(row["scheduler_capacity"]["max_seq_len"]),
        )
        for row in rows
    }
    same_scheduler_capacity = len(capacity_fingerprints) == 1
    telemetry_mappings = set()
    for row in rows:
        if not row["telemetry"]["available"]:
            continue
        mapping = row["telemetry"].get("target_mapping")
        telemetry_mappings.add(
            (
                row["telemetry"]["collector"],
                tuple(
                    sorted(
                        (
                            int(target["logical_device_id"]),
                            int(target["npu_id"]),
                            int(target["chip_id"]),
                        )
                        for target in mapping
                    )
                )
            )
        )
    same_telemetry_mapping = (
        len(telemetry_mappings) == 1
        and all(row["telemetry"]["available"] for row in rows)
    )
    complete_matrix = actual_layouts == FORMAL_LAYOUT_IDS
    formal = (
        complete_matrix
        and baseline_layout == "tp8"
        and same_eight_devices
        and same_scheduler_capacity
        and same_telemetry_mapping
        and all(row["layout_manifest_verified"] for row in rows)
        and all(row["all_replica_reports_formal_candidates"] for row in rows)
        and all(row["telemetry_formal"] for row in rows)
        and all(row["start_skew_within_limit"] for row in rows)
    )
    incomplete_reasons: list[str] = []
    if not complete_matrix:
        incomplete_reasons.append("缺少 TP8、2×TP4、4×TP2 三种完整布局")
    if baseline_layout != "tp8":
        incomplete_reasons.append("正式 comparison 必须使用 tp8 作为 baseline")
    if not same_eight_devices:
        incomplete_reasons.append("三种布局没有使用相同的 8 个 logical devices")
    if not same_scheduler_capacity:
        incomplete_reasons.append(
            "三种布局的 runner、全局 slot/queue 容量或 max_seq_len 不一致"
        )
    if not same_telemetry_mapping:
        incomplete_reasons.append("三种布局缺少一致的 device/chip 遥测映射")
    if not all(row["layout_manifest_verified"] for row in rows):
        incomplete_reasons.append("至少一个布局未通过 manifest 文件哈希校验")
    if not all(row["all_replica_reports_formal_candidates"] for row in rows):
        incomplete_reasons.append("至少一份 replica 报告未达到正式候选门槛")
    if not all(row["telemetry_formal"] for row in rows):
        incomplete_reasons.append(
            "至少一个布局的设备遥测缺失、有错误或覆盖不足"
        )
    if not all(row["start_skew_within_limit"] for row in rows):
        incomplete_reasons.append("至少一个布局的 replica measured run 起点偏差超限")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "qwen3_continuous_batching_replica_layout_comparison",
        "evidence_class": (
            "formal_qwen3_32b_tp8_vs_2xtp4_vs_4xtp2"
            if formal
            else "development_or_incomplete_layout_comparison"
        ),
        "baseline_layout": baseline_layout,
        "comparison_question": (
            "同一 8 devices、模型、workload 和 SLO 下，TP8、2×TP4、4×TP2 "
            "谁提供最高吞吐与 goodput？"
        ),
        "source_workload_sha256": source_sha256,
        "source_workload_file_sha256": source_file_sha256,
        "workload_class": workload_class,
        "protocol": protocol_fingerprint,
        "execution": execution_fingerprint,
        "model": model_fingerprint,
        "same_devices": same_devices,
        "same_eight_devices": same_eight_devices,
        "same_scheduler_capacity": same_scheduler_capacity,
        "same_telemetry_mapping": same_telemetry_mapping,
        "complete_layout_matrix": complete_matrix,
        "max_allowed_start_skew_ms": max_start_skew_ms,
        "min_telemetry_samples_per_device_per_run": (
            min_telemetry_samples_per_device_per_run
        ),
        "incomplete_reasons": incomplete_reasons,
        "rows": rows,
    }
