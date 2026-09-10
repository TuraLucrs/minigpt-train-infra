"""Ascend runtime operations with lazy torch_npu registration.

Importing this module is safe on a CPU-only machine. Only checking or using
the Ascend backend imports the optional extension and its CANN libraries.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, ContextManager

import torch

from .runtime_base import DeviceEvent, DeviceMemorySnapshot


class AscendRuntimeBackend:
    device_type = "npu"
    distributed_backend = "hccl"
    supports_device_events = True
    supports_memory_stats = True
    supports_fused_adamw = False

    @staticmethod
    def _extension() -> Any:
        # Python caches successful imports; failed availability probes are not
        # cached, so installing/registering the extension can be retried.
        return importlib.import_module("torch_npu")

    def _api(self) -> Any:
        self._extension()
        api = getattr(torch, "npu", None)
        if api is None:
            raise RuntimeError("torch_npu did not register the torch.npu runtime")
        return api

    def is_available(self) -> bool:
        try:
            return bool(self._api().is_available())
        except (ImportError, OSError, RuntimeError):
            # An optional installation with missing CANN shared libraries must
            # not prevent auto device selection from using the CPU backend.
            return False

    def device_count(self) -> int:
        return int(self._api().device_count())

    def set_device(self, device: torch.device) -> None:
        self._api().set_device(device)

    def supported_autocast_dtypes(self) -> frozenset[torch.dtype]:
        api = self._api()
        getter = getattr(api, "get_amp_supported_dtype", None)
        if callable(getter):
            return frozenset(getter())
        dtypes = {torch.float16}
        checker = getattr(api, "is_bf16_supported", None)
        if callable(checker) and checker():
            dtypes.add(torch.bfloat16)
        return frozenset(dtypes)

    def autocast(self, dtype: torch.dtype) -> ContextManager[None]:
        self._api()
        return torch.amp.autocast(device_type=self.device_type, dtype=dtype)

    def synchronize(self, device: torch.device) -> None:
        self._api().synchronize(device)

    def manual_seed_all(self, seed: int) -> None:
        self._api().manual_seed_all(seed)

    def reset_peak_memory(self, device: torch.device) -> None:
        self._api().reset_peak_memory_stats(device)

    def memory_snapshot(self, device: torch.device) -> DeviceMemorySnapshot:
        api = self._api()
        reserved = getattr(api, "memory_reserved", None)
        peak_reserved = getattr(api, "max_memory_reserved", None)
        return DeviceMemorySnapshot(
            supported=True,
            allocated_bytes=int(api.memory_allocated(device)),
            peak_allocated_bytes=int(api.max_memory_allocated(device)),
            reserved_bytes=int(reserved(device)) if callable(reserved) else None,
            peak_reserved_bytes=(
                int(peak_reserved(device)) if callable(peak_reserved) else None
            ),
            total_bytes=self.total_memory_bytes(device),
        )

    def device_name(self, device: torch.device) -> str:
        return str(self._api().get_device_name(device))

    def total_memory_bytes(self, device: torch.device) -> int:
        return int(self._api().get_device_properties(device).total_memory)

    def empty_cache(self) -> None:
        self._api().empty_cache()

    def create_event(self) -> DeviceEvent:
        return self._api().Event(enable_timing=True)

    def version_metadata(self) -> dict[str, str | None]:
        extension = self._extension()
        return {
            "torch_npu": str(getattr(extension, "__version__", "unknown")),
            # The experiment CLI may supply versions from a captured host
            # snapshot. Do not infer a CANN/HCCL version from torch_npu's version.
            "cann_version": os.environ.get("CANN_VERSION") or None,
            "hccl_version": os.environ.get("HCCL_VERSION") or None,
        }
