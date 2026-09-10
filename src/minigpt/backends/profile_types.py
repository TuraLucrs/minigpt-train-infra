"""Device-independent scheduler profiling contracts and metadata.

Durations exported by the profilers are microseconds. Scheduler/serving wall
times retain their existing millisecond units and are never mixed with them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
from typing import Mapping, Protocol, Sequence


PROFILE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ProfileProtocol:
    skip_steps: int = 8
    warmup_steps: int = 2
    active_steps: int = 4
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
    ascend_profiler_level: str = "level1"
    ascend_aic_metrics: str = "pipe_utilization"
    ascend_sys_interconnection: bool = True

    def validate(self) -> None:
        for name in ("skip_steps", "warmup_steps", "active_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "active_steps" else 0):
                raise ValueError(f"invalid profile {name}: {value!r}")
        for name in (
            "record_shapes", "profile_memory", "with_stack", "ascend_sys_interconnection"
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"profile {name} must be boolean")
        if self.ascend_profiler_level != "level1":
            raise ValueError("the Ascend collector supports Level1 profiling")
        if self.ascend_aic_metrics not in {
            "pipe_utilization", "memory", "arithmetic_utilization"
        }:
            raise ValueError("unsupported Ascend AiC metric")

    @property
    def required_scheduler_steps(self) -> int:
        return self.skip_steps + self.warmup_steps + self.active_steps

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProfileProtocol":
        # Legacy manifests remain readable without changing their on-disk schema.
        values = dict(value)
        for old, new in (
            ("profiler_level", "ascend_profiler_level"),
            ("aic_metrics", "ascend_aic_metrics"),
            ("sys_interconnection", "ascend_sys_interconnection"),
        ):
            if old in values:
                if new in values and values[new] != values[old]:
                    raise ValueError(f"conflicting profile options: {old}/{new}")
                values[new] = values.pop(old)
        try:
            protocol = cls(**values)
        except TypeError as exc:
            raise ValueError("invalid profile protocol fields") from exc
        protocol.validate()
        return protocol


class StepProfiler(Protocol):
    def __enter__(self) -> "StepProfiler": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def step(self, record: dict[str, object]) -> None: ...

    def metadata(self) -> dict[str, object]: ...


class SchedulerWindow:
    """Track the executed schedule, including ranks not selected for collection."""

    def __init__(self, protocol: ProfileProtocol) -> None:
        protocol.validate()
        self.protocol = protocol
        self.observed_steps = 0
        self.step_indices: list[int] = []
        self.prefill_steps = 0
        self.decode_steps = 0
        self.mixed_steps = 0

    def step(self, record: Mapping[str, object]) -> None:
        index = self.observed_steps
        start = self.protocol.skip_steps + self.protocol.warmup_steps
        if start <= index < self.protocol.required_scheduler_steps:
            sizes: list[int] = []
            for name in ("prefill_batch_size", "decode_batch_size"):
                value = record.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"profile step requires a nonnegative integer {name}")
                sizes.append(value)
            prefill, decode = sizes
            self.step_indices.append(index)
            self.prefill_steps += int(prefill > 0)
            self.decode_steps += int(decode > 0)
            self.mixed_steps += int(prefill > 0 and decode > 0)
        self.observed_steps += 1

    def metadata(self) -> dict[str, object]:
        return {
            "start_step": self.protocol.skip_steps + self.protocol.warmup_steps,
            "end_step_exclusive": self.protocol.required_scheduler_steps,
            "observed_active_steps": len(self.step_indices),
            "step_indices": list(self.step_indices),
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "mixed_steps": self.mixed_steps,
        }


def summarize_numbers(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def capability(
    available: bool, *, source: str, reason: str | None = None,
    unit: str = "us", semantics: str,
) -> dict[str, object]:
    return {
        "available": available,
        "status": "available" if available else "unsupported_or_unobserved",
        "source": source,
        "unit": unit,
        "semantics": semantics,
        "reason": None if available else reason,
    }
