"""Ascend serving telemetry schema、npu-smi 解析与测量区间汇总。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Mapping, Sequence


TELEMETRY_SCHEMA_VERSION = 1
_TELEMETRY_SOURCE_VERIFIED = object()


@dataclass(frozen=True)
class NpuTelemetryTarget:
    """报告中的 logical device 到 npu-smi device/chip 的显式映射。"""

    logical_device_id: int
    npu_id: int
    chip_id: int

    def validate(self) -> None:
        if min(self.logical_device_id, self.npu_id, self.chip_id) < 0:
            raise ValueError("telemetry target id 不能小于 0")

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return asdict(self)


def parse_npu_target(value: str) -> NpuTelemetryTarget:
    """解析 ``logical_device_id=npu_id:chip_id``。"""

    match = re.fullmatch(r"\s*(\d+)\s*=\s*(\d+)\s*:\s*(\d+)\s*", value)
    if match is None:
        raise ValueError("--target 必须使用 logical_device_id=npu_id:chip_id")
    target = NpuTelemetryTarget(*(int(item) for item in match.groups()))
    target.validate()
    return target


_NPU_SMI_FIELDS = {
    "memorycapacity(mb)": "memory_capacity_mb",
    "memoryusagerate(%)": "memory_usage_percent",
    "hbmusagerate(%)": "hbm_usage_percent",
    "aicoreusagerate(%)": "aicore_usage_percent",
    "aicpuusagerate(%)": "aicpu_usage_percent",
    "ctrlcpuusagerate(%)": "ctrlcpu_usage_percent",
    "memorybandwidthusagerate(%)": "memory_bandwidth_usage_percent",
    "aicorefreq(mhz)": "aicore_rated_frequency_mhz",
    "aicorecurfreq(mhz)": "aicore_current_frequency_mhz",
    "temperature(c)": "temperature_celsius",
    "npureal-timepower(w)": "power_watts",
}


def _parse_npu_smi_metrics(output: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw_line in output.splitlines():
        if ":" not in raw_line:
            continue
        raw_name, raw_value = raw_line.split(":", 1)
        name = re.sub(r"\s+", "", raw_name).casefold()
        field = _NPU_SMI_FIELDS.get(name)
        if field is None:
            continue
        value_match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", raw_value)
        if value_match is None:
            continue
        parsed[field] = float(value_match.group(0))

    if "aicore_usage_percent" not in parsed:
        raise ValueError("npu-smi 输出缺少 Aicore Usage Rate(%)")
    if not {"memory_usage_percent", "hbm_usage_percent"} & set(parsed):
        raise ValueError("npu-smi 输出缺少 Memory/HBM Usage Rate(%)")
    if parsed.get("memory_capacity_mb", 1.0) <= 0.0:
        raise ValueError("npu-smi Memory Capacity(MB) 必须大于 0")
    for field, value in parsed.items():
        if field.endswith("_percent") and not 0.0 <= value <= 100.0:
            raise ValueError(f"npu-smi {field} 必须位于 [0, 100]")
    return parsed


def parse_npu_smi_usages(output: str) -> dict[str, float]:
    """解析 ``npu-smi info -t usages`` 的指定 chip 输出。"""

    return _parse_npu_smi_metrics(output)


def parse_npu_smi_common(output: str, *, chip_id: int) -> dict[str, float]:
    """从 ``-t common`` 的多个 Chip ID block 中选择目标 chip。"""

    selected_lines: list[str] = []
    in_selected_block = False
    found = False
    for raw_line in output.splitlines():
        if ":" in raw_line:
            raw_name, raw_value = raw_line.split(":", 1)
            name = re.sub(r"\s+", "", raw_name).casefold()
            if name == "chipid":
                value_match = re.search(r"\d+", raw_value)
                current_chip = (
                    None if value_match is None else int(value_match.group(0))
                )
                in_selected_block = current_chip == chip_id
                found = found or in_selected_block
        if in_selected_block:
            selected_lines.append(raw_line)
    if not found:
        raise ValueError(f"npu-smi common 输出中没有 Chip ID {chip_id}")
    return _parse_npu_smi_metrics("\n".join(selected_lines))


def sample_npu_smi_target(
    target: NpuTelemetryTarget,
    *,
    executable: str = "npu-smi",
    query_type: str = "usages",
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    """采集一个 chip；时间戳取命令调用前后的中点。"""

    target.validate()
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds 必须大于 0")
    if query_type not in {"usages", "common"}:
        raise ValueError("query_type 必须是 usages 或 common")
    started_at_ns = time.time_ns()
    command = [
        executable,
        "info",
        "-t",
        query_type,
        "-i",
        str(target.npu_id),
    ]
    if query_type == "usages":
        command.extend(("-c", str(target.chip_id)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    ended_at_ns = time.time_ns()
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"npu-smi target {target.logical_device_id} 失败，"
            f"returncode={completed.returncode}：{detail}"
        )
    usages = (
        parse_npu_smi_usages(completed.stdout)
        if query_type == "usages"
        else parse_npu_smi_common(completed.stdout, chip_id=target.chip_id)
    )
    return {
        "timestamp_unix_ns": (started_at_ns + ended_at_ns) // 2,
        "collection_latency_ms": (ended_at_ns - started_at_ns) / 1_000_000.0,
        **target.to_dict(),
        **usages,
    }


def _require_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是对象")
    return value


def load_telemetry(path: str | Path) -> dict[str, object]:
    telemetry_path = Path(path)
    try:
        payload = telemetry_path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 telemetry：{telemetry_path}") from exc
    telemetry = _require_mapping(raw, str(telemetry_path))
    telemetry.pop("_source_verified", None)
    telemetry.pop("_source_artifact", None)
    validate_telemetry(telemetry)
    telemetry["_source_verified"] = _TELEMETRY_SOURCE_VERIFIED
    telemetry["_source_artifact"] = {
        "path": str(telemetry_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return telemetry


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数值")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} 必须是有限值")
    return numeric


def validate_telemetry(telemetry: Mapping[str, object]) -> None:
    if telemetry.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        raise ValueError("不支持的 telemetry schema_version")
    if telemetry.get("collector") not in {
        "npu-smi_info_usages",
        "npu-smi_info_common",
    }:
        raise ValueError("telemetry collector 必须是受支持的 npu-smi info 查询")
    if not isinstance(telemetry.get("complete"), bool):
        raise ValueError("telemetry.complete 必须是布尔值")
    sample_interval_ms = _number(
        telemetry.get("sample_interval_ms"),
        "telemetry.sample_interval_ms",
    )
    if sample_interval_ms <= 0.0:
        raise ValueError("telemetry.sample_interval_ms 必须大于 0")
    targets = telemetry.get("targets")
    samples = telemetry.get("samples")
    errors = telemetry.get("errors")
    if not isinstance(targets, list) or not targets:
        raise ValueError("telemetry.targets 必须是非空数组")
    if not isinstance(samples, list):
        raise ValueError("telemetry.samples 必须是数组")
    if not isinstance(errors, list):
        raise ValueError("telemetry.errors 必须是数组")

    logical_ids: set[int] = set()
    physical_targets: set[tuple[int, int]] = set()
    target_by_logical_id: dict[int, tuple[int, int]] = {}
    for index, raw_target in enumerate(targets):
        target = _require_mapping(raw_target, f"telemetry.targets[{index}]")
        parsed = NpuTelemetryTarget(
            int(target["logical_device_id"]),
            int(target["npu_id"]),
            int(target["chip_id"]),
        )
        parsed.validate()
        if parsed.logical_device_id in logical_ids:
            raise ValueError("telemetry target logical_device_id 重复")
        physical = (parsed.npu_id, parsed.chip_id)
        if physical in physical_targets:
            raise ValueError("telemetry target npu_id/chip_id 重复")
        logical_ids.add(parsed.logical_device_id)
        physical_targets.add(physical)
        target_by_logical_id[parsed.logical_device_id] = physical

    previous_timestamp: dict[int, int] = {}
    for index, raw_sample in enumerate(samples):
        sample = _require_mapping(raw_sample, f"telemetry.samples[{index}]")
        logical_id = int(sample["logical_device_id"])
        timestamp = int(sample["timestamp_unix_ns"])
        if logical_id not in logical_ids:
            raise ValueError("telemetry sample 引用了未声明的 logical device")
        sample_physical = (int(sample["npu_id"]), int(sample["chip_id"]))
        if sample_physical != target_by_logical_id[logical_id]:
            raise ValueError("telemetry sample 的 npu_id/chip_id 与 target 不一致")
        if timestamp <= 0:
            raise ValueError("telemetry timestamp_unix_ns 必须大于 0")
        if timestamp < previous_timestamp.get(logical_id, 0):
            raise ValueError("同一 logical device 的 telemetry 时间戳必须非递减")
        previous_timestamp[logical_id] = timestamp
        if _number(
            sample.get("collection_latency_ms"),
            "collection_latency_ms",
        ) < 0.0:
            raise ValueError("collection_latency_ms 不能小于 0")
        if "aicore_usage_percent" not in sample:
            raise ValueError("telemetry sample 缺少 aicore_usage_percent")
        if not {"memory_usage_percent", "hbm_usage_percent"} & set(sample):
            raise ValueError("telemetry sample 缺少 Memory/HBM usage")
        if (
            "memory_capacity_mb" in sample
            and _number(sample["memory_capacity_mb"], "memory_capacity_mb") <= 0.0
        ):
            raise ValueError("memory_capacity_mb 必须大于 0")
        for field in (
            "memory_usage_percent",
            "hbm_usage_percent",
            "aicore_usage_percent",
            "aicpu_usage_percent",
            "ctrlcpu_usage_percent",
            "memory_bandwidth_usage_percent",
        ):
            if field not in sample:
                continue
            value = _number(sample[field], field)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{field} 必须位于 [0, 100]")
        for field in (
            "aicore_rated_frequency_mhz",
            "aicore_current_frequency_mhz",
            "temperature_celsius",
            "power_watts",
        ):
            if field in sample and _number(sample[field], field) < 0.0:
                raise ValueError(f"{field} 不能小于 0")


def _sample_summary(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    ordered = sorted(numeric)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def summarize_telemetry(
    telemetry: Mapping[str, object],
    *,
    run_intervals: Sequence[tuple[int, int]],
    expected_logical_device_ids: Sequence[int],
    min_samples_per_device_per_run: int = 2,
) -> dict[str, object]:
    """只汇总 measured run 区间内采样，并显式给出逐 run/device 覆盖率。"""

    validate_telemetry(telemetry)
    if not run_intervals:
        raise ValueError("run_intervals 不能为空")
    if min_samples_per_device_per_run <= 0:
        raise ValueError("min_samples_per_device_per_run 必须大于 0")
    expected_devices = {int(value) for value in expected_logical_device_ids}
    if not expected_devices:
        raise ValueError("expected_logical_device_ids 不能为空")
    targets = telemetry["targets"]
    target_devices = {int(target["logical_device_id"]) for target in targets}
    if target_devices != expected_devices:
        raise ValueError("telemetry target 没有恰好覆盖 layout 的 logical devices")

    normalized_intervals: list[tuple[int, int]] = []
    previous_end = 0
    for started_at_ns, ended_at_ns in run_intervals:
        start = int(started_at_ns)
        end = int(ended_at_ns)
        if start <= 0 or end <= start:
            raise ValueError("telemetry measured run 区间无效")
        if previous_end and start <= previous_end:
            raise ValueError("telemetry measured run 区间必须有序且不能重叠")
        normalized_intervals.append((start, end))
        previous_end = end

    samples = telemetry["samples"]
    in_any_interval: dict[tuple[int, int], dict[str, object]] = {}
    run_coverage: list[dict[str, object]] = []
    all_runs_covered = True
    for repeat, (started_at_ns, ended_at_ns) in enumerate(normalized_intervals):
        counts: dict[str, int] = {}
        for device_id in sorted(expected_devices):
            selected = [
                sample
                for sample in samples
                if int(sample["logical_device_id"]) == device_id
                and started_at_ns <= int(sample["timestamp_unix_ns"]) <= ended_at_ns
            ]
            counts[str(device_id)] = len(selected)
            for sample in selected:
                key = (
                    int(sample["logical_device_id"]),
                    int(sample["timestamp_unix_ns"]),
                )
                if key in in_any_interval:
                    raise ValueError(
                        "telemetry 同一 device/timestamp 出现重复 sample"
                    )
                in_any_interval[key] = sample
        covered = all(
            count >= min_samples_per_device_per_run for count in counts.values()
        )
        all_runs_covered = all_runs_covered and covered
        run_coverage.append(
            {
                "repeat": repeat,
                "started_at_unix_ns": started_at_ns,
                "ended_at_unix_ns": ended_at_ns,
                "samples_per_device": counts,
                "covered": covered,
            }
        )

    metric_fields = (
        "aicore_usage_percent",
        "memory_usage_percent",
        "hbm_usage_percent",
        "memory_bandwidth_usage_percent",
        "aicpu_usage_percent",
        "ctrlcpu_usage_percent",
        "aicore_rated_frequency_mhz",
        "aicore_current_frequency_mhz",
        "temperature_celsius",
        "power_watts",
        "memory_capacity_mb",
    )
    per_device: dict[str, object] = {}
    for device_id in sorted(expected_devices):
        device_samples = [
            sample
            for sample in in_any_interval.values()
            if int(sample["logical_device_id"]) == device_id
        ]
        per_device[str(device_id)] = {
            "sample_count": len(device_samples),
            **{
                field: _sample_summary(
                    [
                        float(sample[field])
                        for sample in device_samples
                        if field in sample
                    ]
                )
                for field in metric_fields
            },
        }
    overall = {
        field: _sample_summary(
            [
                float(sample[field])
                for sample in in_any_interval.values()
                if field in sample
            ]
        )
        for field in metric_fields
    }
    source_verified = (
        telemetry.get("_source_verified") is _TELEMETRY_SOURCE_VERIFIED
    )
    source_artifact = telemetry.get("_source_artifact")
    if source_verified:
        source_artifact = dict(
            _require_mapping(source_artifact, "telemetry._source_artifact")
        )
    else:
        source_artifact = None
    return {
        "collector": telemetry["collector"],
        "source_file_verified": source_verified,
        "source_artifact": source_artifact,
        "sample_interval_ms": telemetry.get("sample_interval_ms"),
        "target_mapping": targets,
        "errors": telemetry["errors"],
        "complete": telemetry["complete"],
        "min_samples_per_device_per_run": min_samples_per_device_per_run,
        "all_runs_covered": all_runs_covered,
        "error_free": not telemetry["errors"],
        "run_coverage": run_coverage,
        "overall": overall,
        "per_device": per_device,
    }
