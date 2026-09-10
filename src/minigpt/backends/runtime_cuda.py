"""Production CUDA runtime operations exposed through the narrow backend API."""

from __future__ import annotations

from typing import ContextManager

import torch

from .runtime_base import DeviceEvent, DeviceMemorySnapshot


class CudaRuntimeBackend:
    device_type = "cuda"
    distributed_backend = "nccl"
    supports_device_events = True
    supports_memory_stats = True
    supports_fused_adamw = True

    def is_available(self) -> bool:
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def set_device(self, device: torch.device) -> None:
        torch.cuda.set_device(device)

    def supported_autocast_dtypes(self) -> frozenset[torch.dtype]:
        dtypes = {torch.float16}
        if torch.cuda.is_bf16_supported():
            dtypes.add(torch.bfloat16)
        return frozenset(dtypes)

    def autocast(self, dtype: torch.dtype) -> ContextManager[None]:
        return torch.amp.autocast(device_type=self.device_type, dtype=dtype)

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def reset_peak_memory(self, device: torch.device) -> None:
        torch.cuda.reset_peak_memory_stats(device)

    def memory_snapshot(self, device: torch.device) -> DeviceMemorySnapshot:
        return DeviceMemorySnapshot(
            supported=True,
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            total_bytes=self.total_memory_bytes(device),
        )

    def device_name(self, device: torch.device) -> str:
        return str(torch.cuda.get_device_name(device))

    def total_memory_bytes(self, device: torch.device) -> int:
        return int(torch.cuda.get_device_properties(device).total_memory)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def create_event(self) -> DeviceEvent:
        return torch.cuda.Event(enable_timing=True)

    def version_metadata(self) -> dict[str, str | None]:
        nccl_version: str | None = None
        nccl = getattr(torch.cuda, "nccl", None)
        getter = getattr(nccl, "version", None)
        if callable(getter):
            try:
                value = getter()
                nccl_version = (
                    ".".join(str(part) for part in value)
                    if isinstance(value, (tuple, list))
                    else str(value)
                )
            except (AttributeError, RuntimeError):
                pass
        return {
            "torch_npu": None,
            "cuda_version": torch.version.cuda,
            "nccl_version": nccl_version,
        }
