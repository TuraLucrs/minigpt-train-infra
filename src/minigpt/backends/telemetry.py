"""Fail-closed accelerator idle checks taken before loading a model.

CUDA device IDs here are host NVML indices, before CUDA_VISIBLE_DEVICES
renumbers a process's visible devices. Full CUDA cards must use that same
index as physical_card_id and chip_id=0. Ascend maps each logical device to
the explicit npu-smi physical-card/chip pair. No vendor Python SDK is loaded.
"""

from __future__ import annotations

import csv
import io
import math
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from minigpt.serving_telemetry import NpuTelemetryTarget, sample_npu_smi_target


PREFLIGHT_SCHEMA_VERSION = 1
MINIMUM_SAMPLES = 3
MINIMUM_SPAN_SECONDS = 1.0
MEMORY_LIMIT_PERCENT = 10.0
COMPUTE_LIMIT_PERCENT = 5.0

_DEVICE_FIELDS = {"logical_device_id", "physical_card_id", "chip_id"}
_CUDA_QUERY = "--query-gpu=index,memory.used,memory.total,utilization.gpu"


def _number(value: object, name: str, *, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _protocol(samples: int, interval_seconds: float, timeout_seconds: float) -> dict[str, Any]:
    _integer(samples, "samples", minimum=MINIMUM_SAMPLES)
    interval = _number(interval_seconds, "interval_seconds")
    timeout = _number(timeout_seconds, "query_timeout_seconds")
    if interval <= 0 or (samples - 1) * interval < MINIMUM_SPAN_SECONDS:
        raise ValueError("sampling intervals must span at least one second across at least three rounds")
    if timeout <= 0:
        raise ValueError("query_timeout_seconds must be greater than zero")
    return {
        "samples": samples,
        "interval_seconds": interval,
        "interval_policy": "minimum_wait_between_completed_rounds",
        "query_timeout_seconds": timeout,
        "minimum_samples": MINIMUM_SAMPLES,
        "minimum_span_seconds": MINIMUM_SPAN_SECONDS,
        "memory_limit_percent": MEMORY_LIMIT_PERCENT,
        "compute_limit_percent": COMPUTE_LIMIT_PERCENT,
        "aggregation": "maximum_over_all_samples_per_device",
        "collection_phase": "before_model_launch",
    }


def _description(device_type: str) -> dict[str, Any]:
    if device_type not in {"cpu", "cuda", "npu"}:
        raise ValueError("device_type must be cpu, cuda, or npu")
    supported = device_type != "cpu"
    return {
        "supported": supported,
        "collector": {"cpu": "unavailable", "cuda": "nvidia-smi_query_gpu",
                      "npu": "npu-smi_info_common"}[device_type],
        "units": {"memory_usage_percent": "percent", "compute_utilization_percent": "percent",
                  "cuda_memory": "MiB", "wall_clock": "unix_nanoseconds",
                  "sampling_clock": "monotonic_nanoseconds"},
        "capabilities": {
            "device_memory_utilization": supported,
            "device_compute_utilization": supported,
            "per_device_raw_samples": supported,
            "raw_command_output": device_type == "cuda",
            "memory_source": {"cpu": None, "cuda": "memory.used / memory.total",
                              "npu": "HBM Usage Rate(%)"}[device_type],
            "compute_source": {"cpu": None, "cuda": "utilization.gpu",
                               "npu": "AICore Usage Rate(%)"}[device_type],
        },
        "index_mapping": {
            "cpu": "No accelerator devices are claimed for CPU process ranks.",
            "cuda": "logical_device_id and physical_card_id are the same host NVML GPU index; "
                    "chip_id is zero. The selected list order becomes runtime visible ordinal order. "
                    "MIG and unrelated CUDA/NVML index remapping are unsupported.",
            "npu": "logical_device_id maps explicitly to npu-smi -i physical_card_id and Chip ID chip_id.",
        }[device_type],
    }


def _validate_devices(device_type: str, devices: object) -> list[dict[str, int]]:
    if not isinstance(devices, list) or not devices:
        raise ValueError("accelerator preflight requires a nonempty explicit device list")
    normalized: list[dict[str, int]] = []
    logical_ids: set[int] = set()
    physical_targets: set[tuple[int, int]] = set()
    for item in devices:
        if not isinstance(item, Mapping) or set(item) != _DEVICE_FIELDS:
            raise ValueError("each device requires exactly logical_device_id, physical_card_id, and chip_id")
        row = {key: _integer(item[key], key) for key in sorted(_DEVICE_FIELDS)}
        physical = (row["physical_card_id"], row["chip_id"])
        if row["logical_device_id"] in logical_ids or physical in physical_targets:
            raise ValueError("device mapping has duplicate logical IDs or physical targets")
        if device_type == "cuda" and (row["physical_card_id"] != row["logical_device_id"] or row["chip_id"] != 0):
            raise ValueError("CUDA requires logical_device_id == physical_card_id == host NVML index and chip_id=0")
        logical_ids.add(row["logical_device_id"])
        physical_targets.add(physical)
        normalized.append(row)
    return normalized


def _stamp() -> dict[str, int]:
    return {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _query(device_type: str, devices: list[dict[str, int]], *, executable: str,
           timeout_seconds: float) -> dict[str, Any]:
    logical_ids = [row["logical_device_id"] for row in devices]
    if device_type == "cuda":
        command = [executable, _CUDA_QUERY, "--format=csv,noheader,nounits", "-i",
                   ",".join(str(value) for value in logical_ids)]
    else:
        command = [executable, "info", "-t", "common", "-i", str(devices[0]["physical_card_id"])]
    result: dict[str, Any] = {"logical_device_ids": logical_ids, "command": command,
                              "started": _stamp(), "returncode": None, "error": None}
    try:
        if device_type == "cuda":
            completed = subprocess.run(command, check=False, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                                       shell=False, timeout=timeout_seconds)
            result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"nvidia-smi exited with code {completed.returncode}")
        else:
            row = devices[0]
            result["raw_sample"] = sample_npu_smi_target(
                NpuTelemetryTarget(row["logical_device_id"], row["physical_card_id"], row["chip_id"]),
                executable=executable, query_type="common", timeout_seconds=timeout_seconds)
            result["returncode"] = 0
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, subprocess.TimeoutExpired):
            result.update(stdout=_output_text(exc.stdout), stderr=_output_text(exc.stderr))
    finally:
        result["ended"] = _stamp()
    return result


def _timestamps(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    start, end = value["started"], value["ended"]
    unix_start = _integer(start["unix_ns"], "started.unix_ns", minimum=1)
    unix_end = _integer(end["unix_ns"], "ended.unix_ns", minimum=unix_start)
    mono_start = _integer(start["monotonic_ns"], "started.monotonic_ns", minimum=0)
    mono_end = _integer(end["monotonic_ns"], "ended.monotonic_ns", minimum=mono_start)
    return unix_start, unix_end, mono_start, mono_end


def _parse_query(device_type: str, query: Mapping[str, Any],
                 devices: list[dict[str, int]]) -> list[dict[str, Any]]:
    if query.get("error") is not None or type(query.get("returncode")) is not int or query["returncode"] != 0:
        raise ValueError(f"device query failed: {query.get('error') or query.get('returncode')}")
    unix_start, unix_end, mono_start, mono_end = _timestamps(query)
    logical_ids = [row["logical_device_id"] for row in devices]
    if query.get("logical_device_ids") != logical_ids:
        raise ValueError("query device IDs do not match the expected device mapping")
    command = query.get("command")
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise ValueError("query command must be recorded as an argument list")
    parsed: list[dict[str, Any]] = []
    if device_type == "cuda":
        if command[1:] != [_CUDA_QUERY, "--format=csv,noheader,nounits", "-i",
                           ",".join(str(value) for value in logical_ids)]:
            raise ValueError("recorded nvidia-smi query does not match the required device IDs and metrics")
        if not isinstance(query.get("stdout"), str) or not isinstance(query.get("stderr"), str):
            raise ValueError("nvidia-smi raw stdout and stderr must be preserved")
        by_index = {row["logical_device_id"]: row for row in devices}
        seen: set[int] = set()
        for columns in csv.reader(io.StringIO(query["stdout"])):
            if not columns or not any(column.strip() for column in columns):
                continue
            if len(columns) != 4 or re.fullmatch(r"[0-9]+", columns[0].strip()) is None:
                raise ValueError("nvidia-smi must return index,memory.used,memory.total,utilization.gpu")
            index = int(columns[0].strip())
            if index in seen or index not in by_index:
                raise ValueError("nvidia-smi returned a duplicate or unrequested NVML index")
            seen.add(index)
            used = _number(float(columns[1]), "memory.used")
            total = _number(float(columns[2]), "memory.total")
            compute = _number(float(columns[3]), "utilization.gpu", maximum=100.0)
            if total <= 0 or used > total:
                raise ValueError("nvidia-smi memory.used must not exceed a positive memory.total")
            parsed.append({**by_index[index], "nvml_index": index,
                           "timestamp_unix_ns": (unix_start + unix_end) // 2,
                           "timestamp_monotonic_ns": (mono_start + mono_end) // 2,
                           "memory_used_mib": used, "memory_total_mib": total,
                           "memory_usage_percent": (used / total) * 100.0,
                           "compute_utilization_percent": compute})
        if seen != set(by_index):
            raise ValueError("nvidia-smi response does not cover every requested NVML device")
    else:
        row = devices[0]
        if command[1:] != ["info", "-t", "common", "-i", str(row["physical_card_id"])]:
            raise ValueError("recorded npu-smi query does not match the required physical card and common metrics")
        raw = query.get("raw_sample")
        if not isinstance(raw, Mapping):
            raise ValueError("npu-smi original per-chip sample must be preserved")
        for key, expected in (("logical_device_id", row["logical_device_id"]),
                              ("npu_id", row["physical_card_id"]), ("chip_id", row["chip_id"])):
            if _integer(raw.get(key), key) != expected:
                raise ValueError(f"npu-smi {key} does not match the requested device")
        timestamp = _integer(raw.get("timestamp_unix_ns"), "timestamp_unix_ns", minimum=unix_start)
        if timestamp > unix_end:
            raise ValueError("npu-smi sample timestamp is outside its recorded query interval")
        _number(raw.get("collection_latency_ms"), "collection_latency_ms")
        memory = _number(raw.get("hbm_usage_percent"), "HBM usage", maximum=100.0)
        compute = _number(raw.get("aicore_usage_percent"), "AICore usage", maximum=100.0)
        parsed.append({**row, "timestamp_unix_ns": timestamp,
                       "timestamp_monotonic_ns": (mono_start + mono_end) // 2,
                       "memory_usage_percent": memory, "compute_utilization_percent": compute})
    return parsed


def summarize_device_preflight(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute coverage, the sampling window, maxima, and clean from raw evidence.

    Stored clean/complete/maxima are deliberately ignored. This function is
    read-only and can be used by an artifact verifier before accepting a run.
    """
    device_type = record.get("device_type")
    description = _description(device_type)
    result: dict[str, Any] = {"supported": description["supported"], "clean": None,
                              "complete": False, "sample_rounds": 0, "sample_span_seconds": None,
                              "device_maxima": [], "errors": [], "violations": []}
    if device_type == "cpu":
        result["reason"] = "CPU process ranks have no accelerator idle telemetry"
        return result
    result["clean"] = False
    errors = result["errors"]
    try:
        if record.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("unsupported device preflight schema_version")
        for key, expected in description.items():
            if record.get(key) != expected:
                raise ValueError(f"device preflight {key} does not match its backend")
        devices = _validate_devices(device_type, record.get("devices"))
        protocol = record["protocol"]
        expected_protocol = _protocol(protocol["samples"], protocol["interval_seconds"],
                                      protocol["query_timeout_seconds"])
        if protocol != expected_protocol:
            raise ValueError("device preflight protocol or fixed idle thresholds do not match")
        _, _, capture_start, capture_end = _timestamps(record)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append({"stage": "metadata", "message": str(exc)})
        return result
    rounds = record.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != protocol["samples"]:
        errors.append({"stage": "coverage", "message": "preflight must preserve every requested independent sampling round"})
        return result
    result["sample_rounds"] = len(rounds)
    samples_by_device: dict[int, list[dict[str, Any]]] = {row["logical_device_id"]: [] for row in devices}
    round_starts: list[int] = []
    previous_end: int | None = None
    for index, round_record in enumerate(rounds):
        try:
            if _integer(round_record["round_index"], "round_index") != index:
                raise ValueError("sampling round indices must be consecutive and unique")
            _, _, round_start, round_end = _timestamps(round_record)
            if round_start < capture_start or round_end > capture_end:
                raise ValueError("sampling round is outside the collection interval")
            if previous_end is not None and round_start - previous_end < round(protocol["interval_seconds"] * 1e9):
                raise ValueError("sampling rounds do not have the required independent interval")
            previous_end = round_end
            round_starts.append(round_start)
            query_targets = [devices] if device_type == "cuda" else [[row] for row in devices]
            queries = round_record["queries"]
            if not isinstance(queries, list) or len(queries) != len(query_targets):
                raise ValueError("sampling round does not have exact query coverage")
            for query_index, (query, targets) in enumerate(zip(queries, query_targets)):
                try:
                    _, _, query_start, query_end = _timestamps(query)
                    if query_start < round_start or query_end > round_end:
                        raise ValueError("device query is outside its sampling round")
                    for sample in _parse_query(device_type, query, targets):
                        samples_by_device[sample["logical_device_id"]].append(sample)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append({"stage": "query", "round_index": index,
                                   "query_index": query_index, "message": str(exc)})
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"stage": "round", "round_index": index, "message": str(exc)})
    if len(round_starts) == len(rounds):
        result["sample_span_seconds"] = (round_starts[-1] - round_starts[0]) / 1e9
        if result["sample_span_seconds"] < MINIMUM_SPAN_SECONDS:
            errors.append({"stage": "window", "message": "sampling rounds span less than one second"})
    for ordinal, row in enumerate(devices):
        samples = samples_by_device[row["logical_device_id"]]
        span = (samples[-1]["timestamp_monotonic_ns"] - samples[0]["timestamp_monotonic_ns"]) / 1e9 if len(samples) >= 2 else None
        maximum_memory = max((sample["memory_usage_percent"] for sample in samples), default=None)
        maximum_compute = max((sample["compute_utilization_percent"] for sample in samples), default=None)
        covered = len(samples) == protocol["samples"] and span is not None and span >= MINIMUM_SPAN_SECONDS
        if not covered:
            errors.append({"stage": "coverage", "logical_device_id": row["logical_device_id"],
                           "message": "device needs every sample and at least one second of independently timed coverage"})
        maxima = {**row, "sample_count": len(samples), "sample_span_seconds": span,
                  "max_memory_usage_percent": maximum_memory,
                  "max_compute_utilization_percent": maximum_compute,
                  "clean": bool(covered and maximum_memory <= MEMORY_LIMIT_PERCENT and maximum_compute <= COMPUTE_LIMIT_PERCENT)}
        if device_type == "cuda":
            maxima.update(nvml_index=row["logical_device_id"], runtime_visible_ordinal=ordinal)
        result["device_maxima"].append(maxima)
        for name, value, limit in (("memory_usage_percent", maximum_memory, MEMORY_LIMIT_PERCENT),
                                   ("compute_utilization_percent", maximum_compute, COMPUTE_LIMIT_PERCENT)):
            if value is not None and value > limit:
                result["violations"].append({"logical_device_id": row["logical_device_id"],
                                             "metric": name, "maximum_percent": value, "limit_percent": limit})
    result["complete"] = not errors
    result["clean"] = result["complete"] and not result["violations"]
    return result


def collect_device_preflight(device_type: str, devices: Sequence[Mapping[str, Any]], *,
                             samples: int = MINIMUM_SAMPLES, interval_seconds: float = 0.5,
                             query_timeout_seconds: float = 10.0,
                             nvidia_smi_executable: str = "nvidia-smi",
                             npu_smi_executable: str = "npu-smi") -> dict[str, Any]:
    """Collect three or more bounded rounds and reject busy or incomplete evidence.

    CUDA preserves command stdout/stderr. Ascend preserves each original
    sample returned by the common-query collector; that legacy collector
    exposes parsed HBM/AICore fields, not command stdout (declared explicitly
    in capabilities). The caller owns saving the JSON before launching work.
    CPU returns supported=False and clean=None without a device query or wait.
    """
    description = _description(device_type)
    record: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION, "device_type": device_type,
        **description, "devices": [dict(row) if isinstance(row, Mapping) else row for row in devices],
        "protocol": _protocol(samples, interval_seconds, query_timeout_seconds),
        "started": _stamp(), "rounds": [],
    }
    try:
        targets = _validate_devices(device_type, record["devices"]) if description["supported"] else []
    except ValueError:
        targets = []  # Metadata validation below records the exact failure; no query is made.
    if targets:
        for index in range(samples):
            if index:
                time.sleep(interval_seconds)
            round_record: dict[str, Any] = {"round_index": index, "started": _stamp(), "queries": []}
            query_targets = [targets] if device_type == "cuda" else [[row] for row in targets]
            executable = nvidia_smi_executable if device_type == "cuda" else npu_smi_executable
            for selected in query_targets:
                round_record["queries"].append(_query(device_type, selected, executable=executable,
                                                      timeout_seconds=query_timeout_seconds))
            round_record["ended"] = _stamp()
            record["rounds"].append(round_record)
    record["ended"] = _stamp()
    record.update(summarize_device_preflight(record))
    return record
