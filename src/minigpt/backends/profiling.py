"""Backend-neutral production profiling, manifests, and evidence summaries.

Version 1 Ascend APIs and artifacts remain supported by profile_ascend. New
sessions use version 2, including per-rank scheduler-window metadata. A complete
CPU capture is valid collection evidence but cannot establish device time or
communication performance; inspect metric_capabilities before comparing values.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

from . import profile_ascend
from .profile_types import (
    PROFILE_SCHEMA_VERSION, ProfileProtocol, StepProfiler, capability, summarize_numbers,
)
from .profile_torch import TORCH_ARTIFACTS, TorchStepProfiler, analyze_torch_rank, write_json


parse_profile_ranks = profile_ascend.parse_profile_ranks
_MANIFEST_VERIFIED = object()
_FRACTIONS = ("computing", "communication_not_overlapped", "free", "overlapped_of_communication")


def _backend_name(value: object) -> str:
    if not isinstance(value, str):
        device = getattr(value, "device", None)
        value = getattr(device, "type", device)
    value = str(value).split(":", 1)[0].lower()
    value = {"ascend": "npu", "torch_npu": "npu"}.get(value, value)
    if value not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"unsupported profiler backend: {value!r}")
    return value


def _protocol(value: object) -> ProfileProtocol:
    if isinstance(value, ProfileProtocol):
        value.validate()
        return value
    if is_dataclass(value):
        return ProfileProtocol.from_dict(asdict(value))
    if isinstance(value, Mapping):
        return ProfileProtocol.from_dict(value)
    raise TypeError("profile protocol must be a ProfileProtocol or a protocol mapping")


def _ascend_protocol(protocol: ProfileProtocol) -> profile_ascend.AscendProfileProtocol:
    return profile_ascend.AscendProfileProtocol(
        skip_steps=protocol.skip_steps, warmup_steps=protocol.warmup_steps,
        active_steps=protocol.active_steps, record_shapes=protocol.record_shapes,
        profile_memory=protocol.profile_memory, with_stack=protocol.with_stack,
        profiler_level=protocol.ascend_profiler_level, aic_metrics=protocol.ascend_aic_metrics,
        sys_interconnection=protocol.ascend_sys_interconnection,
    )


class AscendProfilerSession:
    """Reuse the real Ascend collector and add portable per-rank capture metadata."""

    def __init__(
        self, profile_root: str | Path, *, global_rank: int, logical_device_id: int,
        selected: bool, protocol: ProfileProtocol,
    ) -> None:
        self.protocol = protocol
        self._collector = profile_ascend.AscendStepProfiler(
            profile_root, global_rank=global_rank, logical_device_id=logical_device_id,
            selected=selected, protocol=_ascend_protocol(protocol),
        )
        self._finished = False
        self._failure: str | None = None

    @property
    def rank_dir(self) -> Path:
        return self._collector.rank_dir

    def __enter__(self) -> "AscendProfilerSession":
        self._collector.__enter__()
        return self

    def step(self, record: dict[str, object]) -> None:
        self._collector.step(record)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        cleanup_error: BaseException | None = None
        try:
            self._collector.__exit__(exc_type, exc, traceback)
        except BaseException as error:
            cleanup_error = error
            self._failure = f"{type(error).__name__}: {error}"
        self._finished = True
        if exc is not None:
            self._failure = f"{type(exc).__name__}: {exc}"
        if self._collector.selected and self.rank_dir.is_dir():
            try:
                write_json(self.rank_dir / "capture.json", self.metadata())
            except Exception:
                if exc_type is None and cleanup_error is None:
                    raise
        if exc_type is None and cleanup_error is not None:
            raise cleanup_error

    def metadata(self) -> dict[str, object]:
        if not self._finished:
            raise RuntimeError("profile metadata is available after session completion")
        metadata = self._collector.metadata()
        metadata["backend_protocol"] = metadata["protocol"]
        metadata.update({
            "backend": "npu", "protocol": self.protocol.as_dict(),
            "capture_status": "failed" if self._failure else "complete",
            "failure_reason": self._failure, "trace_time_unit": "us",
        })
        return metadata


def create_step_profiler(
    runtime: object, profile_root: str | Path, *, global_rank: int,
    logical_device_id: int, selected: bool = True,
    protocol: ProfileProtocol = ProfileProtocol(),
) -> StepProfiler:
    """Configure, but do not start, a profiler for an already-resolved runtime."""
    backend = _backend_name(runtime)
    protocol = _protocol(protocol)
    if type(selected) is not bool:
        raise ValueError("profile selected must be boolean")
    if backend == "npu":
        return AscendProfilerSession(
            profile_root, global_rank=global_rank, logical_device_id=logical_device_id,
            selected=selected, protocol=protocol,
        )
    return TorchStepProfiler(
        profile_root, backend=backend, global_rank=global_rank,
        logical_device_id=logical_device_id, selected=selected, protocol=protocol,
    )


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _artifact_kind(path: Path, backend: str) -> str | None:
    if path.name == "capture.json":
        return "capture_metadata"
    if backend == "npu":
        return profile_ascend._artifact_kind(path)
    return "kineto_trace" if path.name == "trace.json" else None


def _required(backend: str) -> frozenset[str]:
    if backend == "npu":
        return profile_ascend.REQUIRED_LEVEL1_ARTIFACTS | {"capture_metadata"}
    return TORCH_ARTIFACTS


def _identity(
    *, layout_id: str, workload_class: str, mode: str,
    source_workload_sha256: str, source_workload_file_sha256: str,
    git_commit: str, selected_ranks: Sequence[int], logical_device_ids: Sequence[int],
) -> dict[str, object]:
    if not layout_id or not workload_class or mode not in {"open_loop", "closed_loop", "single_request", "static_batch"}:
        raise ValueError("invalid profile layout, workload, or mode")
    for name, value, length in (
        ("source_workload_sha256", source_workload_sha256, 64),
        ("source_workload_file_sha256", source_workload_file_sha256, 64),
        ("git_commit", git_commit, 40),
    ):
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid profile {name}")
    devices = list(logical_device_ids)
    selected = list(selected_ranks)
    if not devices or any(type(value) is not int or value < 0 for value in devices) or len(set(devices)) != len(devices):
        raise ValueError("profile logical devices must be unique nonnegative integers")
    if not selected or any(type(value) is not int or value < 0 or value >= len(devices) for value in selected) or len(set(selected)) != len(selected):
        raise ValueError("invalid or duplicate selected profile ranks")
    return {
        "layout_id": layout_id, "workload_class": workload_class, "mode": mode,
        "source_workload_sha256": source_workload_sha256,
        "source_workload_file_sha256": source_workload_file_sha256,
        "git_commit": git_commit, "selected_ranks": sorted(selected),
        "logical_device_ids": devices, "global_world_size": len(devices),
    }


def build_profile_manifest(
    profile_root: str | Path, *, runtime: object | None = None, backend: str | None = None,
    layout_id: str, workload_class: str, mode: str,
    source_workload_sha256: str, source_workload_file_sha256: str, git_commit: str,
    selected_ranks: Sequence[int], logical_device_ids: Sequence[int],
    protocol: ProfileProtocol = ProfileProtocol(),
) -> dict[str, object]:
    root = Path(profile_root)
    protocol = _protocol(protocol)
    if backend is None and runtime is None:
        inferred = {
            str(json.loads(path.read_text(encoding="utf-8")).get("backend"))
            for path in root.glob("rank_*/capture.json")
        }
        if len(inferred) != 1:
            raise ValueError("pass runtime/backend when profile capture metadata is absent or ambiguous")
        backend = inferred.pop()
    actual_backend = _backend_name(runtime if backend is None else backend)
    if runtime is not None and _backend_name(runtime) != actual_backend:
        raise ValueError("profile backend does not match runtime")
    identity = _identity(
        layout_id=layout_id, workload_class=workload_class, mode=mode,
        source_workload_sha256=source_workload_sha256,
        source_workload_file_sha256=source_workload_file_sha256, git_commit=git_commit,
        selected_ranks=selected_ranks, logical_device_ids=logical_device_ids,
    )
    entries: list[dict[str, object]] = []
    reasons: list[str] = []
    required = _required(actual_backend)
    resolved_parent = root.parent.resolve()
    for rank in identity["selected_ranks"]:
        artifacts = []
        kinds: set[str] = set()
        for path in sorted((root / f"rank_{rank:03d}").rglob("*")):
            if not path.is_file():
                continue
            kind = _artifact_kind(path, actual_backend)
            if kind is None:
                continue
            if not path.resolve().is_relative_to(resolved_parent):
                raise ValueError("profile artifact resolves outside its output directory")
            digest, size = _sha256(path)
            artifacts.append({"kind": kind, "path": path.relative_to(root.parent).as_posix(), "sha256": digest, "size_bytes": size})
            kinds.add(kind)
        missing = sorted(required - kinds)
        if missing:
            reasons.append(f"rank {rank} missing artifacts: {', '.join(missing)}")
        entries.append({"global_rank": rank, "logical_device_id": identity["logical_device_ids"][rank],
                        "complete": not missing, "missing": missing, "artifacts": artifacts})
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": actual_backend,
        "collector": "torch_npu.profiler" if actual_backend == "npu" else "torch.profiler",
        "trace_format": "ascend_level1" if actual_backend == "npu" else "kineto_chrome_json",
        **identity, "protocol": protocol.as_dict(),
        "required_artifact_kinds": sorted(required),
        "complete": not reasons, "incomplete_reasons": reasons, "ranks": entries,
    }


def write_profile_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)


def load_profile_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        payload = manifest_path.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read profile manifest: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("profile manifest must be an object")
    if raw.get("schema_version") == 1:
        return profile_ascend.load_profile_manifest(manifest_path)
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported profile manifest schema")
    backend = _backend_name(raw.get("backend"))
    collector = "torch_npu.profiler" if backend == "npu" else "torch.profiler"
    trace_format = "ascend_level1" if backend == "npu" else "kineto_chrome_json"
    if raw.get("collector") != collector or raw.get("trace_format") != trace_format:
        raise ValueError("profile backend, collector, and trace format disagree")
    try:
        identity = _identity(**{key: raw[key] for key in (
            "layout_id", "workload_class", "mode", "source_workload_sha256",
            "source_workload_file_sha256", "git_commit", "selected_ranks", "logical_device_ids",
        )})
        _protocol(raw["protocol"])
    except (KeyError, TypeError) as exc:
        raise ValueError("profile manifest lacks valid identity/protocol") from exc
    if raw.get("global_world_size") != identity["global_world_size"]:
        raise ValueError("profile global world size does not match logical devices")
    if raw.get("selected_ranks") != identity["selected_ranks"]:
        raise ValueError("profile selected ranks must be sorted")
    if raw.get("required_artifact_kinds") != sorted(_required(backend)):
        raise ValueError("profile required artifacts do not match backend")
    if not isinstance(raw.get("ranks"), list) or not raw["ranks"]:
        raise ValueError("profile manifest has no rank entries")
    resolved_root = manifest_path.parent.resolve()
    seen_paths: set[Path] = set()
    seen_ranks: set[int] = set()
    reasons: list[str] = []
    for entry in raw["ranks"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid profile rank entry")
        rank = entry.get("global_rank")
        if type(rank) is not int or rank not in identity["selected_ranks"] or rank in seen_ranks:
            raise ValueError("invalid or duplicate profile rank")
        seen_ranks.add(rank)
        if entry.get("logical_device_id") != identity["logical_device_ids"][rank]:
            raise ValueError("profile rank maps to the wrong logical device")
        if not isinstance(entry.get("artifacts"), list):
            raise ValueError("profile artifacts must be an array")
        kinds: set[str] = set()
        for artifact in entry["artifacts"]:
            if not isinstance(artifact, dict):
                raise ValueError("invalid profile artifact")
            relative = str(artifact.get("path", "")).replace("\\", "/")
            relative_path = Path(relative)
            if not relative or relative_path.is_absolute() or PureWindowsPath(relative).drive or ".." in relative_path.parts:
                raise ValueError("invalid or duplicate profile artifact path")
            artifact_path = manifest_path.parent / relative_path
            resolved_path = artifact_path.resolve()
            if resolved_path in seen_paths:
                raise ValueError("duplicate profile artifact path")
            if not resolved_path.is_relative_to(resolved_root):
                raise ValueError("profile artifact path escapes manifest directory")
            if f"rank_{rank:03d}" not in relative_path.parts:
                raise ValueError("profile artifact belongs to a different rank directory")
            kind = _artifact_kind(artifact_path, backend)
            if kind is None or artifact.get("kind") != kind:
                raise ValueError("profile artifact kind does not match its file")
            try:
                digest, size = _sha256(artifact_path)
            except OSError as exc:
                raise ValueError(f"cannot read profile artifact: {artifact_path}") from exc
            if digest != artifact.get("sha256") or type(artifact.get("size_bytes")) is not int or size != artifact["size_bytes"]:
                raise ValueError(f"profile artifact hash or size mismatch: {artifact_path}")
            seen_paths.add(resolved_path)
            kinds.add(kind)
            artifact["_verified_path"] = artifact_path
        missing = sorted(_required(backend) - kinds)
        entry["complete"] = not missing
        entry["missing"] = missing
        if missing:
            reasons.append(f"rank {rank} missing artifacts: {', '.join(missing)}")
    if seen_ranks != set(identity["selected_ranks"]):
        raise ValueError("profile rank entries do not match selected ranks")
    # Recompute completeness instead of trusting the manifest's boolean.
    raw["complete"] = not reasons
    raw["incomplete_reasons"] = reasons
    raw["_source_verified"] = _MANIFEST_VERIFIED
    raw["_source_artifact"] = {"path": str(manifest_path), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    return raw


def _ascend_rank_capabilities(rank: dict[str, object]) -> dict[str, object]:
    totals = rank["step_trace"]["totals_us"]
    if "overlapped" not in totals or not totals.get("communication"):
        # The frozen parser historically defaulted absent overlap columns to 0.
        # Preserve that legacy API, but do not fabricate this metric in v2.
        rank["step_trace"]["fractions"]["overlapped_of_communication"] = None
    rank["step_trace"]["time_basis"] = "Ascend step_trace_time.csv Stage, or Computing + Communication (Not Overlapped) + Free when Stage is absent"
    rank["step_trace"]["duration_method"] = "ascend_level1_native_csv"
    rank["phase_attribution"] = {"available": False, "method": None,
                                  "reason": "trace_view.json and communication.json are hash-bound artifacts; this adapter does not parse flow/phase correlation"}
    return {
        "host_operator_time": capability("Host" in str(rank["operator"]["duration_field"]), source="ascend.operator_details.csv", reason="the legacy parser selected Device Self Duration", semantics="operator self duration from the explicitly named CSV column"),
        "device_kernel_time": capability(True, source="ascend.kernel_details.csv", semantics="sum of exported kernel durations, not interval-unioned wall time"),
        "communication_not_overlapped": capability(True, source="ascend.step_trace_time.csv", semantics="native Communication (Not Overlapped), divided by the recorded Stage/fallback denominator"),
        "phase_device_time": capability(False, source="ascend.trace_view.json (hashed only)", reason="no trace correlation/phase parser is implemented for Ascend", semantics="must not infer device phase attribution from host markers alone"),
        "device_memory": capability(False, source="runtime measured memory report", reason="allocator peak/reserved memory is not inferred from these Level1 summaries", unit="bytes", semantics="use runtime memory measurement with its allocator and peak definitions"),
    }


def _capture(entry: Mapping[str, object], manifest: Mapping[str, object]) -> dict[str, object]:
    paths = [artifact["_verified_path"] for artifact in entry["artifacts"] if artifact["kind"] == "capture_metadata"]
    if len(paths) != 1:
        raise ValueError("a rank needs exactly one capture metadata artifact")
    capture = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    if not isinstance(capture, dict):
        raise ValueError("capture metadata must be an object")
    for field, expected in (
        ("backend", manifest["backend"]), ("collector", manifest["collector"]),
        ("global_rank", entry["global_rank"]), ("logical_device_id", entry["logical_device_id"]),
        ("selected", True), ("capture_status", "complete"),
    ):
        if capture.get(field) != expected:
            raise ValueError(f"capture metadata has mismatched {field}")
    protocol = _protocol(manifest["protocol"])
    if _protocol(capture.get("protocol")).as_dict() != protocol.as_dict():
        raise ValueError("capture and manifest protocols disagree")
    window = capture.get("captured_scheduler_window")
    expected_start = protocol.skip_steps + protocol.warmup_steps
    expected_indices = list(range(expected_start, protocol.required_scheduler_steps))
    if not isinstance(window, dict):
        raise ValueError("capture has no scheduler window metadata")
    for field, expected in (
        ("start_step", expected_start), ("end_step_exclusive", protocol.required_scheduler_steps),
        ("observed_active_steps", protocol.active_steps), ("step_indices", expected_indices),
    ):
        if window.get(field) != expected:
            raise ValueError(f"capture scheduler window has mismatched {field}")
    for name in ("prefill_steps", "decode_steps", "mixed_steps"):
        value = window.get(name)
        if type(value) is not int or value < 0 or value > protocol.active_steps:
            raise ValueError(f"invalid captured {name}")
    if window["mixed_steps"] > min(window["prefill_steps"], window["decode_steps"]):
        raise ValueError("captured mixed steps exceed phase coverage")
    if type(capture.get("observed_scheduler_steps")) is not int or capture["observed_scheduler_steps"] < protocol.required_scheduler_steps:
        raise ValueError("insufficient observed scheduler steps")
    return capture


def summarize_profile(path: str | Path, *, top_k: int = 20) -> dict[str, object]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    manifest = load_profile_manifest(path)
    legacy = manifest["schema_version"] == 1
    ranks: list[dict[str, object]] = []
    errors = list(manifest.get("incomplete_reasons", []))
    backend = "npu" if legacy else str(manifest["backend"])
    if legacy:
        legacy_report = profile_ascend.analyze_profile_manifest(path, top_k=top_k)
        ranks = legacy_report["ranks"]
        errors = list(legacy_report["incomplete_reasons"])
        for entry in manifest["ranks"]:
            kinds = {artifact["kind"] for artifact in entry["artifacts"]}
            missing = profile_ascend.REQUIRED_LEVEL1_ARTIFACTS - kinds
            if missing:
                errors.append(f"legacy rank {entry['global_rank']} missing artifacts: {', '.join(sorted(missing))}")
        for rank in ranks:
            rank["metric_capabilities"] = _ascend_rank_capabilities(rank)
    else:
        if manifest.get("_source_verified") is not _MANIFEST_VERIFIED:
            raise ValueError("profile manifest has not been verified")
        for entry in manifest["ranks"]:
            if not entry["complete"]:
                continue
            try:
                capture = _capture(entry, manifest)
                if backend == "npu":
                    rank = profile_ascend._analyze_rank(entry, top_k=top_k)
                    rank["metric_capabilities"] = _ascend_rank_capabilities(rank)
                else:
                    rank = analyze_torch_rank(entry, backend=backend, top_k=top_k)
                rank["capture"] = capture
                ranks.append(rank)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"rank {entry['global_rank']} profile analysis failed: {exc}")
    collection_complete = not errors and len(ranks) == len(manifest["selected_ranks"])
    all_ranks = sorted(manifest["selected_ranks"]) == list(range(int(manifest["global_world_size"])))
    if not all_ranks:
        errors.append("profile does not cover every global rank")
    aggregate: dict[str, object] = {}
    coverage: dict[str, object] = {}
    for field in _FRACTIONS:
        values = [rank["step_trace"]["fractions"].get(field) for rank in ranks]
        numeric = [float(value) for value in values if value is not None]
        aggregate[field] = summarize_numbers(numeric) if collection_complete and len(numeric) == len(ranks) else None
        coverage[field] = {"ranks_with_metric": len(numeric), "selected_ranks": len(manifest["selected_ranks"]), "global_world_size": manifest["global_world_size"]}
    metric_names = {name for rank in ranks for name in rank["metric_capabilities"]}
    caps: dict[str, object] = {}
    for name in sorted(metric_names):
        values = [rank["metric_capabilities"][name] for rank in ranks]
        first = dict(values[0])
        available = collection_complete and all(value["available"] for value in values)
        first["available"] = available
        first["status"] = "available" if available else "unsupported_or_unobserved"
        first["reason"] = None if available else "; ".join(sorted({str(value["reason"]) for value in values if value.get("reason")})) or "not all selected ranks have verified metric evidence"
        first["ranks_with_metric"] = sum(bool(value["available"]) for value in values)
        caps[name] = first
    if not ranks:
        for name in (
            "host_operator_time", "device_kernel_time", "communication_not_overlapped",
            "phase_device_time", "device_memory",
        ):
            caps[name] = capability(
                False, source="unverified profile artifacts",
                reason="no selected rank produced a verified parseable profile",
                unit="bytes" if name == "device_memory" else "us",
                semantics="unavailable evidence must not be interpreted as zero",
            )
            caps[name]["ranks_with_metric"] = 0
    complete = collection_complete and all_ranks
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "backend_profile_summary",
        "evidence_class": "complete_backend_profile" if complete else "incomplete_backend_profile",
        "backend": backend,
        "collector": "torch_npu.profiler" if backend == "npu" else "torch.profiler",
        "source_schema_version": manifest["schema_version"],
        "parser": "ascend_level1_csv_v1" if backend == "npu" else "kineto_interval_correlation_v1",
        **{key: manifest[key] for key in (
            "layout_id", "workload_class", "mode", "source_workload_sha256",
            "source_workload_file_sha256", "git_commit", "global_world_size",
            "logical_device_ids", "selected_ranks",
        )},
        "protocol": _protocol(manifest["protocol"]).as_dict(),
        "profile_manifest": dict(manifest["_source_artifact"]),
        "complete": complete, "collection_complete": collection_complete,
        "all_ranks_covered": all_ranks, "incomplete_reasons": errors,
        "aggregate_step_fractions": aggregate, "metric_coverage": coverage,
        "metric_capabilities": caps, "ranks": ranks,
        "metric_units": {"kernel_duration": "us", "operator_duration": "us",
                         "step_trace_totals": "us", "step_fractions": "ratio",
                         "phase_device_duration": "us"},
        "aggregation_scope": "per-rank ratios summarized across selected ranks; a missing rank metric makes its aggregate null",
        "measurement_scope": "separate bounded profiler replay; latency/goodput must come from measured serving runs",
    }


analyze_profile_manifest = summarize_profile

__all__ = [
    "ProfileProtocol", "StepProfiler", "create_step_profiler", "build_profile_manifest",
    "write_profile_manifest", "load_profile_manifest", "summarize_profile",
    "analyze_profile_manifest", "parse_profile_ranks",
]
