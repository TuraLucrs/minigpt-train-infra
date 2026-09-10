"""Ascend scheduler-step Profiler collection、artifact manifest 与摘要解析。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Mapping, Sequence


PROFILE_SCHEMA_VERSION = 1
REQUIRED_LEVEL1_ARTIFACTS = frozenset(
    {
        "profiler_info",
        "operator_details",
        "kernel_details",
        "step_trace_time",
        "trace_view",
        "communication",
    }
)
_PROFILE_MANIFEST_VERIFIED = object()


@dataclass(frozen=True)
class AscendProfileProtocol:
    """把一个 scheduler step 作为 Profiler step 的有界采集协议。"""

    skip_steps: int = 8
    warmup_steps: int = 2
    active_steps: int = 4
    profiler_level: str = "level1"
    aic_metrics: str = "pipe_utilization"
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
    sys_interconnection: bool = True

    def validate(self) -> None:
        if self.skip_steps < 0 or self.warmup_steps < 0:
            raise ValueError("profile skip/warmup steps 不能小于 0")
        if self.active_steps <= 0:
            raise ValueError("profile active steps 必须大于 0")
        if self.profiler_level != "level1":
            raise ValueError("v0.7.1 Profiling Gate 固定使用 Level1")
        if self.aic_metrics not in {
            "pipe_utilization",
            "memory",
            "arithmetic_utilization",
        }:
            raise ValueError("不支持的 profile aic_metrics")

    @property
    def required_scheduler_steps(self) -> int:
        return self.skip_steps + self.warmup_steps + self.active_steps


class AscendStepProfiler:
    """延迟导入 torch_npu；未选中的 rank 仍参与相同 replay，但不采集。"""

    def __init__(
        self,
        profile_root: str | Path,
        *,
        global_rank: int,
        logical_device_id: int,
        selected: bool,
        protocol: AscendProfileProtocol,
    ) -> None:
        protocol.validate()
        if global_rank < 0 or logical_device_id < 0:
            raise ValueError("profile rank/device id 不能小于 0")
        self.profile_root = Path(profile_root)
        self.global_rank = global_rank
        self.logical_device_id = logical_device_id
        self.selected = selected
        self.protocol = protocol
        self.observed_steps = 0
        self._profile: object | None = None
        self._started = False
        self._finished = False

    @property
    def rank_dir(self) -> Path:
        return self.profile_root / f"rank_{self.global_rank:03d}"

    def __enter__(self) -> "AscendStepProfiler":
        if self._started:
            raise RuntimeError("AscendStepProfiler 不能重复启动")
        self._started = True
        if not self.selected:
            return self
        if self.rank_dir.exists():
            raise FileExistsError(f"profile rank 输出目录已经存在：{self.rank_dir}")
        self.rank_dir.mkdir(parents=True)
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("Ascend profiling 需要安装 torch_npu") from exc

        profiler = torch_npu.profiler
        metric_by_name = {
            "pipe_utilization": profiler.AiCMetrics.PipeUtilization,
            "memory": profiler.AiCMetrics.Memory,
            "arithmetic_utilization": profiler.AiCMetrics.ArithmeticUtilization,
        }
        experimental_config = profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=metric_by_name[self.protocol.aic_metrics],
            data_simplification=True,
            sys_interconnection=self.protocol.sys_interconnection,
        )
        schedule = profiler.schedule(
            wait=0,
            warmup=self.protocol.warmup_steps,
            active=self.protocol.active_steps,
            repeat=1,
            skip_first=self.protocol.skip_steps,
        )
        handler = profiler.tensorboard_trace_handler(
            str(self.rank_dir),
            worker_name=f"rank_{self.global_rank:03d}",
            analyse_flag=True,
        )
        self._profile = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU,
            ],
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=self.protocol.record_shapes,
            profile_memory=self.protocol.profile_memory,
            with_stack=self.protocol.with_stack,
            with_modules=False,
            experimental_config=experimental_config,
        )
        self._profile.__enter__()
        return self

    def step(self, _record: dict[str, object]) -> None:
        if not self._started or self._finished:
            raise RuntimeError("profiler step 必须位于活动 session 内")
        self.observed_steps += 1
        if self.selected:
            assert self._profile is not None
            self._profile.step()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._finished:
            raise RuntimeError("AscendStepProfiler 不能重复结束")
        self._finished = True
        if self.selected:
            assert self._profile is not None
            self._profile.__exit__(exc_type, exc, traceback)
        if (
            exc_type is None
            and self.selected
            and self.observed_steps < self.protocol.required_scheduler_steps
        ):
            raise RuntimeError(
                "profiling replay 的 scheduler steps 不足："
                f"observed={self.observed_steps}, "
                f"required={self.protocol.required_scheduler_steps}"
            )

    def metadata(self) -> dict[str, object]:
        if not self._finished:
            raise RuntimeError("profiler metadata 只能在 session 完成后读取")
        return {
            "collector": "torch_npu.profiler",
            "selected": self.selected,
            "global_rank": self.global_rank,
            "logical_device_id": self.logical_device_id,
            "observed_scheduler_steps": self.observed_steps,
            "protocol": asdict(self.protocol),
            "rank_output_dir": (
                str(self.rank_dir) if self.selected else None
            ),
        }


def parse_profile_ranks(value: str, world_size: int) -> tuple[int, ...]:
    if world_size <= 0:
        raise ValueError("world_size 必须大于 0")
    if value.strip().lower() == "all":
        return tuple(range(world_size))
    try:
        ranks = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("profile-ranks 必须为 all 或逗号分隔整数") from exc
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("profile-ranks 不能为空或重复")
    if min(ranks) < 0 or max(ranks) >= world_size:
        raise ValueError("profile-ranks 超出 global world size")
    return tuple(sorted(ranks))


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _artifact_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("profiler_info") and name.endswith(".json"):
        return "profiler_info"
    exact = {
        "operator_details.csv": "operator_details",
        "kernel_details.csv": "kernel_details",
        "step_trace_time.csv": "step_trace_time",
        "trace_view.json": "trace_view",
        "communication.json": "communication",
        "communication_matrix.json": "communication_matrix",
        "hccs.csv": "hccs",
        "pcie.csv": "pcie",
        "api_statistic.csv": "api_statistic",
        "op_statistic.csv": "op_statistic",
    }
    return exact.get(name)


def build_profile_manifest(
    profile_root: str | Path,
    *,
    layout_id: str,
    workload_class: str,
    mode: str,
    source_workload_sha256: str,
    source_workload_file_sha256: str,
    git_commit: str,
    selected_ranks: Sequence[int],
    logical_device_ids: Sequence[int],
    protocol: AscendProfileProtocol,
) -> dict[str, object]:
    """为解析后的关键文件建立哈希索引；原始 PROF 数据保留但不重复全量哈希。"""

    protocol.validate()
    root = Path(profile_root)
    raw_selected = tuple(int(rank) for rank in selected_ranks)
    selected = tuple(sorted(raw_selected))
    if not layout_id or not workload_class or mode not in {"open_loop", "closed_loop"}:
        raise ValueError("profile manifest 的 layout/workload/mode 无效")
    for label, digest in (
        ("source_workload_sha256", source_workload_sha256),
        ("source_workload_file_sha256", source_workload_file_sha256),
        ("git_commit", git_commit),
    ):
        if len(digest) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"profile manifest 的 {label} 无效")
    if len(source_workload_sha256) != 64 or len(source_workload_file_sha256) != 64:
        raise ValueError("profile manifest 的 workload digest 必须是 SHA-256")
    if len(git_commit) != 40:
        raise ValueError("profile manifest 的 git_commit 必须是完整 commit SHA")
    if len(logical_device_ids) == 0:
        raise ValueError("profile manifest 缺少 logical devices")
    if (
        not selected
        or len(set(selected)) != len(selected)
        or min(selected) < 0
    ):
        raise ValueError("profile manifest 的 selected ranks 为空、重复或小于 0")
    logical_devices = [int(value) for value in logical_device_ids]
    if min(logical_devices) < 0 or len(set(logical_devices)) != len(logical_devices):
        raise ValueError("profile manifest 的 logical devices 必须非负且互不重复")
    rank_entries: list[dict[str, object]] = []
    incomplete_reasons: list[str] = []
    for rank in selected:
        if rank >= len(logical_device_ids):
            raise ValueError("profile rank 没有对应 logical device")
        rank_dir = root / f"rank_{rank:03d}"
        artifacts: list[dict[str, object]] = []
        kinds: set[str] = set()
        if rank_dir.is_dir():
            for path in sorted(rank_dir.rglob("*")):
                if not path.is_file():
                    continue
                kind = _artifact_kind(path)
                if kind is None:
                    continue
                digest, size = _sha256(path)
                artifacts.append(
                    {
                        "kind": kind,
                        "path": str(path.relative_to(root.parent)),
                        "sha256": digest,
                        "size_bytes": size,
                    }
                )
                kinds.add(kind)
        missing = sorted(REQUIRED_LEVEL1_ARTIFACTS - kinds)
        if missing:
            incomplete_reasons.append(f"rank {rank} 缺少 {', '.join(missing)}")
        rank_entries.append(
            {
                "global_rank": rank,
                "logical_device_id": int(logical_device_ids[rank]),
                "complete": not missing,
                "missing": missing,
                "artifacts": artifacts,
            }
        )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector": "torch_npu.profiler",
        "layout_id": layout_id,
        "workload_class": workload_class,
        "mode": mode,
        "source_workload_sha256": source_workload_sha256,
        "source_workload_file_sha256": source_workload_file_sha256,
        "git_commit": git_commit,
        "global_world_size": len(logical_device_ids),
        "logical_device_ids": logical_devices,
        "selected_ranks": list(selected),
        "protocol": asdict(protocol),
        "required_artifact_kinds": sorted(REQUIRED_LEVEL1_ARTIFACTS),
        "complete": not incomplete_reasons,
        "incomplete_reasons": incomplete_reasons,
        "ranks": rank_entries,
    }


def write_profile_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_profile_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        payload = manifest_path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 profile manifest：{manifest_path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("不支持的 profile manifest")
    ranks = raw.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("profile manifest 缺少 rank entries")
    root = manifest_path.parent
    resolved_root = root.resolve()
    seen_paths: set[str] = set()
    seen_ranks: set[int] = set()
    for rank_entry in ranks:
        if not isinstance(rank_entry, dict):
            raise ValueError("profile rank entry 必须是对象")
        rank = int(rank_entry.get("global_rank", -1))
        if rank < 0 or rank in seen_ranks:
            raise ValueError("profile rank 非法或重复")
        seen_ranks.add(rank)
        artifacts = rank_entry.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("profile artifacts 必须是数组")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("profile artifact 必须是对象")
            relative = str(artifact.get("path", ""))
            relative_path = Path(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative in seen_paths
            ):
                raise ValueError("profile artifact path 非法或重复")
            seen_paths.add(relative)
            artifact_path = root / relative_path
            if not artifact_path.resolve().is_relative_to(resolved_root):
                raise ValueError("profile artifact path 越出 manifest 目录")
            digest, size = _sha256(artifact_path)
            if digest != artifact.get("sha256") or size != int(
                artifact.get("size_bytes", -1)
            ):
                raise ValueError(f"profile artifact 哈希或大小不一致：{artifact_path}")
            artifact["_verified_path"] = artifact_path
    selected = {int(value) for value in raw.get("selected_ranks", [])}
    if selected != seen_ranks:
        raise ValueError("profile selected_ranks 与 rank entries 不一致")
    raw["_source_verified"] = _PROFILE_MANIFEST_VERIFIED
    raw["_source_artifact"] = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return raw


def _header_key(value: str) -> str:
    value = value.replace("µ", "u").replace("μ", "u").casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def _column(fieldnames: Sequence[str], *aliases: str, required: bool = True) -> str | None:
    by_key = {_header_key(field): field for field in fieldnames}
    for alias in aliases:
        if _header_key(alias) in by_key:
            return by_key[_header_key(alias)]
    if required:
        raise ValueError(f"Profiler CSV 缺少列：{aliases[0]}")
    return None


def _finite_number(value: object, label: str) -> float:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数值") from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} 必须是非负有限值")
    return numeric


def _csv_rows(paths: Iterable[Path]) -> tuple[list[str], list[dict[str, str]]]:
    common_fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Profiler CSV 没有表头：{path}")
            fields = [str(field) for field in reader.fieldnames]
            if common_fields is None:
                common_fields = fields
            elif {_header_key(field) for field in common_fields} != {
                _header_key(field) for field in fields
            }:
                raise ValueError(f"同类 Profiler CSV 表头不一致：{path}")
            rows.extend({str(key): str(value) for key, value in row.items()} for row in reader)
    if common_fields is None:
        raise ValueError("没有 Profiler CSV 输入")
    return common_fields, rows


def _summary(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _top_durations(
    names: Sequence[str],
    durations: Sequence[float],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    totals: dict[str, list[float]] = {}
    for name, duration in zip(names, durations):
        totals.setdefault(name, []).append(duration)
    rows = [
        {
            "name": name,
            "calls": len(samples),
            "total_us": sum(samples),
            "mean_us": statistics.fmean(samples),
        }
        for name, samples in totals.items()
    ]
    rows.sort(key=lambda row: (-float(row["total_us"]), str(row["name"])))
    return rows[:top_k]


def _analyze_rank(
    rank_entry: Mapping[str, object],
    *,
    top_k: int,
) -> dict[str, object]:
    artifacts = rank_entry["artifacts"]
    by_kind: dict[str, list[Path]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or "_verified_path" not in artifact:
            raise ValueError("profile artifact 未通过 manifest 验证")
        by_kind.setdefault(str(artifact["kind"]), []).append(
            Path(artifact["_verified_path"])
        )

    kernel_fields, kernel_rows = _csv_rows(by_kind["kernel_details"])
    if not kernel_rows:
        raise ValueError("kernel_details 没有数据行")
    kernel_name = _column(kernel_fields, "Name", "Op Name")
    kernel_type = _column(kernel_fields, "Type", "OP Type", required=False)
    accelerator = _column(
        kernel_fields,
        "Accelerator Core",
        "Task Type",
        required=False,
    )
    kernel_duration = _column(
        kernel_fields,
        "Duration(us)",
        "Task Duration(us)",
    )
    kernel_names = [row[kernel_name] for row in kernel_rows]
    kernel_durations = [
        _finite_number(row[kernel_duration], "kernel duration")
        for row in kernel_rows
    ]
    communication_durations: list[float] = []
    compute_durations: list[float] = []
    for row, duration in zip(kernel_rows, kernel_durations):
        identity = " ".join(
            str(row.get(field, ""))
            for field in (kernel_name, kernel_type, accelerator)
            if field is not None
        ).casefold()
        is_communication = any(
            token in identity
            for token in (
                "hccl",
                "hcom",
                "allreduce",
                "allgather",
                "reducescatter",
                "reduce_scatter",
                "alltoall",
            )
        )
        (communication_durations if is_communication else compute_durations).append(
            duration
        )

    operator_fields, operator_rows = _csv_rows(by_kind["operator_details"])
    if not operator_rows:
        raise ValueError("operator_details 没有数据行")
    operator_name = _column(operator_fields, "Name", "Op Name")
    device_self = _column(
        operator_fields,
        "Device Self Duration (us)",
        "Device Self Duration",
        required=False,
    )
    host_self = _column(
        operator_fields,
        "Host Self Duration (us)",
        "Host Self Duration",
        required=False,
    )
    operator_duration_column = device_self or host_self
    if operator_duration_column is None:
        raise ValueError("operator_details 缺少 Host/Device Self Duration")
    operator_names = [row[operator_name] for row in operator_rows]
    operator_durations = [
        _finite_number(row[operator_duration_column], "operator duration")
        for row in operator_rows
    ]

    step_fields, step_rows = _csv_rows(by_kind["step_trace_time"])
    if not step_rows:
        raise ValueError("step_trace_time 没有数据行")
    component_aliases = {
        "computing": ("Computing",),
        "communication_not_overlapped": ("Communication (Not Overlapped)",),
        "overlapped": ("Overlapped",),
        "communication": ("Communication",),
        "free": ("Free",),
        "stage": ("Stage",),
        "bubble": ("Bubble",),
        "preparing": ("Preparing",),
    }
    components: dict[str, list[float]] = {}
    for key, aliases in component_aliases.items():
        field = _column(step_fields, *aliases, required=False)
        if field is not None:
            components[key] = [
                _finite_number(row[field], f"step_trace {key}") for row in step_rows
            ]
    required_components = {"computing", "communication_not_overlapped", "free"}
    if not required_components.issubset(components):
        raise ValueError("step_trace_time 缺少 compute/communication/free")
    totals = {key: sum(values) for key, values in components.items()}
    denominator = totals.get("stage") or sum(
        totals[key] for key in ("computing", "communication_not_overlapped", "free")
    )
    denominator = max(denominator, 1e-12)
    return {
        "global_rank": int(rank_entry["global_rank"]),
        "logical_device_id": int(rank_entry["logical_device_id"]),
        "kernel": {
            "rows": len(kernel_rows),
            "total_duration_us": sum(kernel_durations),
            "communication_duration_us": sum(communication_durations),
            "compute_duration_us": sum(compute_durations),
            "top": _top_durations(kernel_names, kernel_durations, top_k=top_k),
        },
        "operator": {
            "rows": len(operator_rows),
            "duration_field": operator_duration_column,
            "top": _top_durations(operator_names, operator_durations, top_k=top_k),
        },
        "step_trace": {
            "rows": len(step_rows),
            "totals_us": totals,
            "fractions": {
                "computing": totals["computing"] / denominator,
                "communication_not_overlapped": (
                    totals["communication_not_overlapped"] / denominator
                ),
                "free": totals["free"] / denominator,
                "overlapped_of_communication": (
                    totals.get("overlapped", 0.0)
                    / max(totals.get("communication", 0.0), 1e-12)
                ),
            },
        },
    }


def analyze_profile_manifest(
    path: str | Path,
    *,
    top_k: int = 20,
) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    manifest = load_profile_manifest(path)
    if manifest.get("_source_verified") is not _PROFILE_MANIFEST_VERIFIED:
        raise ValueError("profile manifest 未验证")
    parse_errors: list[str] = []
    rank_summaries: list[dict[str, object]] = []
    for rank_entry in manifest["ranks"]:
        if not rank_entry.get("complete"):
            parse_errors.append(
                f"rank {rank_entry['global_rank']} artifact 不完整"
            )
            continue
        try:
            rank_summaries.append(_analyze_rank(rank_entry, top_k=top_k))
        except (KeyError, OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            parse_errors.append(f"rank {rank_entry['global_rank']} 解析失败：{exc}")
    fraction_fields = (
        "computing",
        "communication_not_overlapped",
        "free",
        "overlapped_of_communication",
    )
    aggregate_fractions = {
        field: _summary(
            [float(rank["step_trace"]["fractions"][field]) for rank in rank_summaries]
        )
        for field in fraction_fields
    }
    source_artifact = dict(manifest["_source_artifact"])
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "v0.7.1_ascend_profile_summary",
        "evidence_class": (
            "complete_v0.7.1_ascend_profile"
            if manifest.get("complete")
            and not parse_errors
            and len(rank_summaries) == int(manifest["global_world_size"])
            else "development_or_incomplete_v0.7.1_profile"
        ),
        "layout_id": manifest["layout_id"],
        "workload_class": manifest["workload_class"],
        "mode": manifest["mode"],
        "source_workload_sha256": manifest["source_workload_sha256"],
        "source_workload_file_sha256": manifest[
            "source_workload_file_sha256"
        ],
        "git_commit": manifest["git_commit"],
        "global_world_size": manifest["global_world_size"],
        "logical_device_ids": manifest["logical_device_ids"],
        "selected_ranks": manifest["selected_ranks"],
        "protocol": manifest["protocol"],
        "profile_manifest": source_artifact,
        "complete": (
            bool(manifest.get("complete"))
            and not parse_errors
            and len(rank_summaries) == int(manifest["global_world_size"])
        ),
        "incomplete_reasons": list(manifest.get("incomplete_reasons", []))
        + parse_errors,
        "aggregate_step_fractions": aggregate_fractions,
        "ranks": rank_summaries,
    }
