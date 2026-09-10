"""Lazy runtime backend selection; does not import accelerator SDKs eagerly."""

from __future__ import annotations

from functools import lru_cache

from .runtime_base import RuntimeBackend


@lru_cache(maxsize=3)
def get_runtime_backend(device_type: str) -> RuntimeBackend:
    if device_type == "cpu":
        from .runtime_cpu import CpuRuntimeBackend

        return CpuRuntimeBackend()
    if device_type == "cuda":
        from .runtime_cuda import CudaRuntimeBackend

        return CudaRuntimeBackend()
    if device_type == "npu":
        from .runtime_ascend import AscendRuntimeBackend

        return AscendRuntimeBackend()
    raise ValueError(f"Unsupported runtime backend: {device_type!r}")
