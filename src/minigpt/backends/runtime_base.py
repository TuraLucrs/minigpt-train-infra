"""The device operations needed by the existing PyTorch inference engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ContextManager, Protocol

import torch


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    """Allocator measurements in bytes; unavailable measurements stay null.

    These are PyTorch allocator statistics, not total process RSS or an
    estimate of model/KV memory. ``total_bytes`` is physical device capacity.
    """

    supported: bool
    allocated_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    total_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "unit": "bytes",
            "source": "pytorch_allocator" if self.supported else "unavailable",
        }


class DeviceEvent(Protocol):
    def record(self) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, end_event: DeviceEvent) -> float: ...


class RuntimeBackend(Protocol):
    """No Tensor/operator wrappers: only genuinely device-specific operations."""

    device_type: str
    distributed_backend: str
    supports_device_events: bool
    supports_memory_stats: bool
    supports_fused_adamw: bool

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def set_device(self, device: torch.device) -> None: ...

    def supported_autocast_dtypes(self) -> frozenset[torch.dtype]: ...

    def autocast(self, dtype: torch.dtype) -> ContextManager[None]: ...

    def synchronize(self, device: torch.device) -> None: ...

    def manual_seed_all(self, seed: int) -> None: ...

    def reset_peak_memory(self, device: torch.device) -> None: ...

    def memory_snapshot(self, device: torch.device) -> DeviceMemorySnapshot: ...

    def device_name(self, device: torch.device) -> str: ...

    def total_memory_bytes(self, device: torch.device) -> int | None: ...

    def empty_cache(self) -> None: ...

    def create_event(self) -> DeviceEvent: ...

    def version_metadata(self) -> dict[str, str | None]: ...
