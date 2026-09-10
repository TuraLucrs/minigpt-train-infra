"""The public device runtime shared by models, engines, and benchmarks.

Device-specific operations live in minigpt.backends. Tensor, Module, and
mathematical operations remain ordinary PyTorch code.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Callable, ContextManager

import torch

from .backends.runtime import get_runtime_backend
from .backends.runtime_base import DeviceEvent, DeviceMemorySnapshot, RuntimeBackend


def _npu_available() -> bool:
    """Compatibility helper; the optional extension is loaded by its backend."""

    return get_runtime_backend("npu").is_available()


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Capabilities actually used by this project's runtime."""

    supports_autocast: bool
    supports_bf16: bool
    supports_device_events: bool
    supports_memory_stats: bool
    supports_fused_adamw: bool


@dataclass(frozen=True)
class RuntimeContext:
    """A resolved device, precision, and backend capability set.

    The existing constructor and public methods remain compatible. Accelerator
    SDK imports happen only while resolving or using that accelerator.
    """

    device: torch.device
    precision: str
    amp_dtype: torch.dtype | None
    capabilities: RuntimeCapabilities

    @property
    def backend(self) -> RuntimeBackend:
        return get_runtime_backend(self.device.type)

    @classmethod
    def create(
        cls,
        requested_device: str,
        requested_precision: str,
        warn: Callable[[str], None] = print,
        device_index: int | None = None,
        allow_accelerator_fallback: bool = True,
        allow_precision_fallback: bool = True,
    ) -> "RuntimeContext":
        if device_index is not None and device_index < 0:
            raise ValueError("device_index 不能小于 0")
        device_name = requested_device.lower()
        if device_name == "auto":
            device_name = next(
                name
                for name in ("cuda", "npu", "cpu")
                if get_runtime_backend(name).is_available()
            )
        elif device_name not in {"cpu", "cuda", "npu"}:
            raise ValueError("device 必须是 auto、cpu、cuda 或 npu")

        backend = get_runtime_backend(device_name)
        if not backend.is_available():
            if not allow_accelerator_fallback:
                raise RuntimeError(f"请求了 {device_name.upper()}，但当前后端不可用")
            warn(f"[warning] 请求了 {device_name.upper()}，但当前不可用；回退到 CPU。")
            device_name = "cpu"
            backend = get_runtime_backend(device_name)
        device = (
            torch.device("cpu")
            if device_name == "cpu"
            else torch.device(device_name, device_index)
        )
        if device.type != "cpu" and device.index is not None:
            visible_count = backend.device_count()
            if device.index >= visible_count:
                raise ValueError(
                    f"{device.type.upper()} device_index={device.index} "
                    f"超出可见设备数 {visible_count}"
                )
            backend.set_device(device)

        precision_name = requested_precision.lower()
        if precision_name not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision 必须是 fp32、fp16 或 bf16")
        supported_dtypes = backend.supported_autocast_dtypes()
        requested_dtype = {
            "fp32": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision_name]
        amp_dtype = requested_dtype
        if requested_dtype is not None and requested_dtype not in supported_dtypes:
            if not allow_precision_fallback:
                raise RuntimeError(
                    f"当前 {device.type} 后端不支持请求的 {precision_name}"
                )
            warn(
                f"[warning] 当前 {device.type} 后端不支持 {precision_name} autocast；"
                "回退到 fp32。"
            )
            precision_name = "fp32"
            amp_dtype = None
        capabilities = RuntimeCapabilities(
            supports_autocast=amp_dtype is not None,
            supports_bf16=torch.bfloat16 in supported_dtypes,
            supports_device_events=backend.supports_device_events,
            supports_memory_stats=backend.supports_memory_stats,
            supports_fused_adamw=backend.supports_fused_adamw,
        )
        return cls(
            device=device,
            precision=precision_name,
            amp_dtype=amp_dtype,
            capabilities=capabilities,
        )

    def autocast(self) -> ContextManager[None]:
        if self.amp_dtype is None:
            return nullcontext()
        return self.backend.autocast(self.amp_dtype)

    def synchronize(self) -> None:
        self.backend.synchronize(self.device)

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.backend.manual_seed_all(seed)

    def reset_peak_memory(self) -> None:
        self.backend.reset_peak_memory(self.device)

    def memory_snapshot(self) -> DeviceMemorySnapshot:
        """Return explicit allocator measurements, in bytes, with null for N/A."""

        return self.backend.memory_snapshot(self.device)

    def memory_stats_mb(self) -> tuple[float, float]:
        """Legacy (allocated, peak allocated) API, using MiB despite its name.

        CPU retains the historical (0, 0) result for existing callers. New
        evidence writers should use memory_snapshot() and its supported flag
        instead of presenting this compatibility sentinel as a measurement.
        """

        snapshot = self.memory_snapshot()
        if not snapshot.supported:
            return 0.0, 0.0
        if snapshot.allocated_bytes is None or snapshot.peak_allocated_bytes is None:
            raise RuntimeError("设备 allocator 未提供当前/峰值内存")
        scale = 1024 * 1024
        return (
            float(snapshot.allocated_bytes / scale),
            float(snapshot.peak_allocated_bytes / scale),
        )

    def device_name(self) -> str:
        return self.backend.device_name(self.device)

    def visible_device_count(self) -> int:
        """Accelerator device count; CPU process ranks are not device counts."""

        return self.backend.device_count()

    def total_memory_mb(self) -> float:
        total = self.backend.total_memory_bytes(self.device)
        return 0.0 if total is None else float(total / (1024 * 1024))

    def empty_cache(self) -> None:
        self.backend.empty_cache()

    def create_device_event(self) -> DeviceEvent:
        return self.backend.create_event()

    def backend_metadata(self) -> dict[str, str | bool | float | None]:
        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "device_name": self.device_name(),
            "precision": self.precision,
            "supports_bf16": self.capabilities.supports_bf16,
            "supports_memory_stats": self.capabilities.supports_memory_stats,
            "supports_device_events": self.capabilities.supports_device_events,
            "memory_statistics_source": (
                "pytorch_allocator"
                if self.capabilities.supports_memory_stats
                else "unavailable"
            ),
            "total_memory_mb": self.total_memory_mb(),
            "torch": torch.__version__,
            **self.backend.version_metadata(),
        }


class DeviceIntervalTimer:
    """Measure one window, synchronizing device events only at its end."""

    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self._wall_start: float | None = None
        self._device_start: DeviceEvent | None = None
        self.last_device_seconds: float | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            raise RuntimeError("计时窗口已经启动")
        self._wall_start = time.perf_counter()
        if self.runtime.capabilities.supports_device_events:
            event = self.runtime.create_device_event()
            event.record()
            self._device_start = event
        self._running = True

    def elapsed_seconds(self) -> float:
        if not self._running or self._wall_start is None:
            raise RuntimeError("计时窗口尚未启动")
        try:
            if self.runtime.capabilities.supports_device_events:
                if self._device_start is None:
                    raise RuntimeError("设备计时起始 Event 缺失")
                end = self.runtime.create_device_event()
                end.record()
                end.synchronize()
                self.last_device_seconds = self._device_start.elapsed_time(end) / 1000.0
            else:
                self.last_device_seconds = None
            return time.perf_counter() - self._wall_start
        finally:
            self._device_start = None
            self._wall_start = None
            self._running = False
