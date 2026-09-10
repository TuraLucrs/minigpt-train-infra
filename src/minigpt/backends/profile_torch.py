"""Real torch.profiler sessions and offline Kineto trace analysis.

No vendor extension is imported here. CUDA launch correlation is used for
phase attribution; a GPU event is not assigned to a phase merely because
their timestamps overlap.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Mapping, Sequence

from .profile_types import ProfileProtocol, SchedulerWindow, capability


TORCH_ARTIFACTS = frozenset({"kineto_trace", "capture_metadata"})


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class TorchStepProfiler:
    def __init__(
        self, profile_root: str | Path, *, backend: str, global_rank: int,
        logical_device_id: int, selected: bool, protocol: ProfileProtocol,
    ) -> None:
        protocol.validate()
        if backend not in {"cpu", "cuda"}:
            raise ValueError("TorchStepProfiler requires cpu or cuda")
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("invalid profile global_rank")
        if type(logical_device_id) is not int or logical_device_id < 0:
            raise ValueError("invalid profile logical_device_id")
        self.profile_root = Path(profile_root)
        self.backend = backend
        self.global_rank = global_rank
        self.logical_device_id = logical_device_id
        self.selected = selected
        self.protocol = protocol
        self.window = SchedulerWindow(protocol)
        self._state = "new"
        self._profile = None
        self._exports = 0
        self._torch_version: str | None = None
        self._failure: str | None = None

    @property
    def rank_dir(self) -> Path:
        return self.profile_root / f"rank_{self.global_rank:03d}"

    def _export(self, profiler: object) -> None:
        if self._exports:
            raise RuntimeError("a bounded profile must export exactly one trace")
        profiler.export_chrome_trace(str(self.rank_dir / "trace.json"))
        self._exports += 1

    def __enter__(self) -> "TorchStepProfiler":
        if self._state != "new":
            raise RuntimeError("a profiler session cannot be started twice")
        self._state = "starting"
        if not self.selected:
            self._state = "active"
            return self
        # Check availability before claiming a capture directory. In particular,
        # torch.profiler must never silently drop a requested CUDA activity.
        try:
            import torch

            self._torch_version = str(torch.__version__)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.backend == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA profiling requires an available CUDA device")
                if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError("this PyTorch build cannot collect CUDA profiler activities")
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if self.rank_dir.exists():
                raise FileExistsError(f"profile output directory already exists: {self.rank_dir}")
            self.rank_dir.mkdir(parents=True)
            self._profile = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.protocol.warmup_steps,
                    active=self.protocol.active_steps, repeat=1,
                    skip_first=self.protocol.skip_steps,
                ),
                on_trace_ready=self._export,
                record_shapes=self.protocol.record_shapes,
                profile_memory=self.protocol.profile_memory,
                with_stack=self.protocol.with_stack,
                with_modules=False,
            )
            self._profile.__enter__()
        except BaseException as exc:
            self._state = "failed"
            self._failure = f"{type(exc).__name__}: {exc}"
            if self._profile is not None:
                try:
                    self._profile.__exit__(*sys.exc_info())
                except BaseException:
                    pass
                self._persist_metadata()
            raise
        self._state = "active"
        return self

    def step(self, record: dict[str, object]) -> None:
        if self._state != "active":
            raise RuntimeError("profiler step requires an active session")
        self.window.step(record)
        if self.selected:
            self._profile.step()

    def _persist_metadata(self) -> None:
        if self.selected and self.rank_dir.is_dir():
            write_json(self.rank_dir / "capture.json", self.metadata())

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._state != "active":
            raise RuntimeError("profiler exit requires an active session")
        cleanup_error: BaseException | None = None
        try:
            if self.selected:
                self._profile.__exit__(exc_type, exc, traceback)
        except BaseException as error:
            cleanup_error = error
        if exc is not None:
            self._failure = f"{type(exc).__name__}: {exc}"
        elif cleanup_error is not None:
            self._failure = f"{type(cleanup_error).__name__}: {cleanup_error}"
        elif self.selected and self.window.observed_steps < self.protocol.required_scheduler_steps:
            self._failure = (
                "profiling replay has insufficient scheduler steps: "
                f"observed={self.window.observed_steps}, required={self.protocol.required_scheduler_steps}"
            )
        elif self.selected and self._exports != 1:
            self._failure = "profiler did not export its bounded active window"
        self._state = "failed" if self._failure else "finished"
        try:
            self._persist_metadata()
        except Exception:
            if exc_type is None and cleanup_error is None:
                raise
        if exc_type is None:
            if cleanup_error is not None:
                raise cleanup_error
            if self._failure:
                raise RuntimeError(self._failure)

    def metadata(self) -> dict[str, object]:
        if self._state not in {"finished", "failed"}:
            raise RuntimeError("profile metadata is available after session completion")
        return {
            "collector": "torch.profiler", "backend": self.backend,
            "collector_version": self._torch_version,
            "selected": self.selected, "global_rank": self.global_rank,
            "logical_device_id": self.logical_device_id,
            "protocol": self.protocol.as_dict(),
            "observed_scheduler_steps": self.window.observed_steps,
            "captured_scheduler_window": self.window.metadata(),
            "rank_output_dir": str(self.rank_dir) if self.selected else None,
            "capture_status": "complete" if self._state == "finished" else "failed",
            "failure_reason": self._failure, "trace_exports": self._exports,
            "trace_time_unit": "us",
        }


def _number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"invalid {label}")
    return result


def _merged(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _intersection(
    first: Sequence[tuple[float, float]], second: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    left, right = _merged(first), _merged(second)
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def _duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merged(intervals))


def _top(events: Sequence[Mapping[str, object]], top_k: int) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        grouped[str(event["name"])].append(float(event["dur"]))
    rows = [
        {"name": name, "calls": len(samples), "total_us": sum(samples),
         "mean_us": statistics.fmean(samples)}
        for name, samples in grouped.items()
    ]
    return sorted(rows, key=lambda row: (-float(row["total_us"]), str(row["name"])))[:top_k]


def _argument(event: Mapping[str, object], *names: str) -> str | None:
    arguments = event.get("args", {})
    if not isinstance(arguments, dict):
        return None
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in arguments.items()}
    for name in names:
        value = normalized.get(name)
        if value is not None:
            return str(value)
    return None


def _phase_for_launch(
    launch: Mapping[str, object], annotations: Sequence[Mapping[str, object]],
) -> str | None:
    phases = {
        str(annotation["name"]).removeprefix("minigpt::").removesuffix("_phase")
        for annotation in annotations
        if annotation.get("pid") == launch.get("pid")
        and annotation.get("tid") == launch.get("tid")
        and float(annotation["ts"]) <= float(launch["ts"])
        and float(launch["ts"]) + float(launch["dur"]) <= float(annotation["ts"]) + float(annotation["dur"]) + 1e-6
    }
    return next(iter(phases)) if len(phases) == 1 else None


def analyze_torch_rank(
    rank_entry: Mapping[str, object], *, backend: str, top_k: int,
) -> dict[str, object]:
    by_kind: dict[str, list[Path]] = defaultdict(list)
    for artifact in rank_entry["artifacts"]:
        if "_verified_path" not in artifact:
            raise ValueError("profile artifact has not been verified")
        by_kind[str(artifact["kind"])].append(Path(artifact["_verified_path"]))
    if any(len(by_kind[kind]) != 1 for kind in TORCH_ARTIFACTS):
        raise ValueError("each rank needs exactly one Kineto trace and capture metadata")
    capture = json.loads(by_kind["capture_metadata"][0].read_text(encoding="utf-8"))
    if not isinstance(capture, dict):
        raise ValueError("capture metadata must be an object")
    for field, expected in (
        ("backend", backend), ("collector", "torch.profiler"),
        ("global_rank", rank_entry["global_rank"]),
        ("logical_device_id", rank_entry["logical_device_id"]),
        ("selected", True), ("capture_status", "complete"), ("trace_exports", 1),
    ):
        if capture.get(field) != expected:
            raise ValueError(f"capture metadata has mismatched {field}")
    protocol = ProfileProtocol.from_dict(capture["protocol"])
    payload = json.loads(by_kind["kineto_trace"][0].read_text(encoding="utf-8"))
    raw_events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Kineto trace has no traceEvents")
    events: list[dict[str, object]] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            raise ValueError("invalid Kineto trace event")
        if raw.get("ph") != "X":
            continue
        event = dict(raw)
        event["name"] = str(raw.get("name", ""))
        event["cat"] = str(raw.get("cat", "")).lower()
        event["ts"] = _number(raw.get("ts"), "event timestamp")
        event["dur"] = _number(raw.get("dur"), "event duration", nonnegative=True)
        events.append(event)
    if not events:
        raise ValueError("Kineto trace contains no complete duration events")
    steps = [event for event in events if re.fullmatch(r"ProfilerStep#\d+", str(event["name"]))]
    indices = sorted(int(str(event["name"]).split("#")[1]) for event in steps)
    expected = list(range(protocol.skip_steps + protocol.warmup_steps, protocol.required_scheduler_steps))
    window = capture.get("captured_scheduler_window")
    if indices != expected or not isinstance(window, dict) or window.get("step_indices") != expected:
        raise ValueError("Kineto ProfilerStep indices do not match the captured scheduler window")
    if window.get("observed_active_steps") != protocol.active_steps:
        raise ValueError("capture metadata has incomplete active scheduler steps")
    if int(capture.get("observed_scheduler_steps", -1)) < protocol.required_scheduler_steps:
        raise ValueError("capture metadata has insufficient scheduler steps")
    windows = _merged([(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in steps])
    stage = _duration(windows)
    if stage <= 0.0:
        raise ValueError("captured scheduler window has no positive duration")
    host = [event for event in events if event["cat"] in {"cpu_op", "user_annotation", "python_function"}
            and not str(event["name"]).startswith("ProfilerStep#")]
    annotations = [event for event in host if str(event["name"]).startswith("minigpt::")]
    phases = [event for event in annotations if event["name"] in {"minigpt::decode_phase", "minigpt::prefill_phase"}]
    kernels = [event for event in events if event["cat"] == "kernel"]
    transfers = [event for event in events if event["cat"] in {"gpu_memcpy", "gpu_memset"}]
    communication = [event for event in kernels if re.search(r"nccl|rccl", str(event["name"]), re.IGNORECASE)]
    communication_ids = {id(event) for event in communication}
    compute = [event for event in kernels if id(event) not in communication_ids]

    def clipped(selected: Sequence[Mapping[str, object]]) -> list[tuple[float, float]]:
        return _intersection(
            [(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in selected], windows
        )

    outside_kernel_count = sum(_duration(clipped([event])) <= 0.0 for event in kernels)
    captured_kernel_count = len(kernels)
    # Availability must be established inside the selected scheduler window.
    # A kernel elsewhere in the trace is not evidence of zero active-window work.
    kernels = [event for event in kernels if _duration(clipped([event])) > 0.0] if backend == "cuda" else []
    communication = [event for event in kernels if id(event) in communication_ids]
    compute = [event for event in kernels if id(event) not in communication_ids]
    has_device = bool(kernels)
    has_communication = has_device and bool(communication)
    compute_us = _duration(clipped(compute)) if has_device else None
    communication_us = _duration(clipped(communication)) if has_communication else None
    overlap_us = _duration(_intersection(clipped(compute), clipped(communication))) if has_communication else None
    nonoverlap_us = max(communication_us - overlap_us, 0.0) if has_communication else None
    free_us = max(stage - _duration(clipped(kernels + transfers)), 0.0) if has_device else None
    fractions = {
        "computing": compute_us / stage if compute_us is not None else None,
        "communication_not_overlapped": nonoverlap_us / stage if nonoverlap_us is not None else None,
        "free": free_us / stage if free_us is not None else None,
        "overlapped_of_communication": overlap_us / communication_us if communication_us else None,
    }
    launches = [event for event in events if event["cat"] in {"cuda_runtime", "cuda_driver"}]
    by_correlation: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_external_id: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for launch in launches:
        key = _argument(launch, "correlation", "correlationid")
        if key is not None:
            by_correlation[key].append(launch)
    for event in host + launches:
        key = _argument(event, "externalid")
        if key is not None:
            by_external_id[key].append(event)
    assigned: dict[str, list[Mapping[str, object]]] = {"prefill": [], "decode": []}
    for kernel in kernels:
        correlated = by_correlation.get(_argument(kernel, "correlation", "correlationid"), [])
        if not correlated:
            correlated = by_external_id.get(_argument(kernel, "externalid"), [])
        identities = {_phase_for_launch(launch, phases) for launch in correlated}
        if len(identities) == 1 and None not in identities:
            assigned[next(iter(identities))].append(kernel)
    matched = sum(len(value) for value in assigned.values())
    phase_rows = {
        phase: {
            "kernel_count": len(selected),
            "device_duration_us": _duration(clipped(selected)) if selected else None,
            "communication_duration_us": _duration(clipped([event for event in selected if id(event) in communication_ids])) if selected and has_communication else None,
        }
        for phase, selected in assigned.items()
    }
    device_reason = "CPU captures have no device kernel evidence" if backend == "cpu" else "no positive-duration CUDA kernel activity was captured inside the active scheduler window"
    communication_reason = device_reason if not has_device else "no positive-duration NCCL kernels were captured inside the active scheduler window; communication coverage is unconfirmed"
    caps = {
        "host_operator_time": capability(bool(host), source="kineto.cpu_op/user_annotation", reason="no host operator events", semantics="sum of inclusive host event durations; nested events can overlap"),
        "device_kernel_time": capability(has_device, source="kineto.kernel", reason=device_reason, semantics="kernel event durations; top-event sums are not wall time"),
        "communication_not_overlapped": capability(has_communication, source="kineto.kernel/NCCL", reason=communication_reason, semantics="union of NCCL intervals minus their intersection with compute kernels, clipped to ProfilerStep windows"),
        "phase_device_time": capability(matched > 0, source="kineto CUDA correlation/External id and minigpt phase annotations", reason="no CUDA kernels could be correlated to a phase launch", semantics="union of phase-associated kernel intervals; coverage counts are reported explicitly"),
        "device_memory": capability(False, source="runtime measured memory report", reason="allocator peak/reserved memory is not inferred from this trace parser", unit="bytes", semantics="use runtime memory measurement with its allocator and peak definitions"),
    }
    if backend == "cuda" and not kernels:
        raise ValueError("CUDA profile has no positive-duration device kernel events inside its active scheduler window")
    return {
        "global_rank": int(rank_entry["global_rank"]),
        "logical_device_id": int(rank_entry["logical_device_id"]),
        "capture": capture,
        "kernel": {
            "rows": len(kernels),
            "raw_trace_rows": captured_kernel_count,
            "outside_window_rows": outside_kernel_count,
            "total_duration_us": sum(float(event["dur"]) for event in kernels) if has_device else None,
            "communication_duration_us": sum(float(event["dur"]) for event in communication) if has_communication else None,
            "compute_duration_us": sum(float(event["dur"]) for event in compute) if has_device else None,
            "top": _top(kernels, top_k),
        },
        "operator": {"rows": len(host), "duration_field": "Host Duration (inclusive, us)", "top": _top(host, top_k)},
        "host_annotations": _top(annotations, max(len(annotations), 1)),
        "step_trace": {
            "rows": len(steps), "step_indices": indices,
            "totals_us": {"stage": stage, "computing": compute_us, "communication": communication_us,
                          "communication_not_overlapped": nonoverlap_us, "overlapped": overlap_us, "free": free_us},
            "fractions": fractions,
            "time_basis": "union of active Kineto ProfilerStep host windows",
            "duration_method": "interval_union_clipped_to_scheduler_windows",
        },
        "phase_attribution": {
            "available": matched > 0, "method": "CUDA launch correlation/External id",
            "mapped_kernel_count": matched, "unmapped_kernel_count": len(kernels) - matched,
            "coverage_fraction": matched / len(kernels) if kernels else None,
            "phases": phase_rows,
        },
        "metric_capabilities": caps,
    }
