"""CPU correctness backend; accelerator memory/events are unavailable."""

from __future__ import annotations

from typing import ContextManager

import torch

from .runtime_base import DeviceEvent, DeviceMemorySnapshot


class CpuRuntimeBackend:
    device_type = "cpu"
    distributed_backend = "gloo"
    supports_device_events = False
    supports_memory_stats = False
    supports_fused_adamw = False

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        # CPU ranks are processes, not physical accelerator devices.
        return 0

    def set_device(self, device: torch.device) -> None:
        return None

    def supported_autocast_dtypes(self) -> frozenset[torch.dtype]:
        # Preserve the existing CPU/fp32 correctness contract.
        return frozenset()

    def autocast(self, dtype: torch.dtype) -> ContextManager[None]:
        raise RuntimeError("CPU correctness backend does not enable autocast")

    def synchronize(self, device: torch.device) -> None:
        return None

    def manual_seed_all(self, seed: int) -> None:
        # RuntimeContext already seeds the CPU generator with torch.manual_seed.
        return None

    def reset_peak_memory(self, device: torch.device) -> None:
        return None

    def memory_snapshot(self, device: torch.device) -> DeviceMemorySnapshot:
        return DeviceMemorySnapshot(supported=False)

    def device_name(self, device: torch.device) -> str:
        return "CPU"

    def total_memory_bytes(self, device: torch.device) -> int | None:
        return None

    def empty_cache(self) -> None:
        return None

    def create_event(self) -> DeviceEvent:
        raise RuntimeError("CPU backend does not provide device timing events")

    def version_metadata(self) -> dict[str, str | None]:
        return {"torch_npu": None}
