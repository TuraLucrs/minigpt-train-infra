"""设备相关能力的最小运行时边界。

模型、训练循环和推理引擎只需要知道“在哪个 device 上运行、使用什么精度、何时必须同步、
怎样读取设备内存”。CUDA 细节集中在这里，后续接入 Ascend 时可以增加后端实现，而不是把
``torch_npu`` 和 HCCL 判断散落到业务代码中。

这个抽象故意很窄：Tensor、Module 和数学算子仍然直接使用 PyTorch。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Callable, ContextManager

import torch


@dataclass(frozen=True)
class RuntimeCapabilities:
    """当前后端实际提供的、项目此阶段会使用的能力。"""

    supports_autocast: bool
    supports_bf16: bool
    supports_device_events: bool
    supports_memory_stats: bool
    supports_fused_adamw: bool


@dataclass(frozen=True)
class RuntimeContext:
    """一次运行解析后的 device、precision 和后端能力。"""

    device: torch.device
    precision: str
    amp_dtype: torch.dtype | None
    capabilities: RuntimeCapabilities

    @classmethod
    def create(
        cls,
        requested_device: str,
        requested_precision: str,
        warn: Callable[[str], None] = print,
    ) -> "RuntimeContext":
        """把用户请求解析成当前机器真正可以执行的运行时。

        v0.3 先提供 CPU/CUDA。未来 Ascend 后端应在这一边界增加能力，而不是让推理引擎
        根据厂商名称到处写分支。
        """

        device_name = requested_device.lower()
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_name == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                warn("[warning] 请求了 CUDA，但当前不可用；回退到 CPU。")
                device = torch.device("cpu")
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError("device 必须是 auto、cpu 或 cuda")

        precision_name = requested_precision.lower()
        if precision_name not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision 必须是 fp32、fp16 或 bf16")

        supports_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
        amp_dtype: torch.dtype | None = None
        if precision_name != "fp32":
            if device.type != "cuda":
                warn(f"[warning] 当前 {device.type} 路径不启用 {precision_name} autocast；回退到 fp32。")
                precision_name = "fp32"
            elif precision_name == "bf16" and not supports_bf16:
                warn("[warning] 当前 CUDA 设备不支持 bf16；回退到 fp32。")
                precision_name = "fp32"
            else:
                amp_dtype = torch.float16 if precision_name == "fp16" else torch.bfloat16

        capabilities = RuntimeCapabilities(
            supports_autocast=amp_dtype is not None,
            supports_bf16=supports_bf16,
            supports_device_events=device.type == "cuda",
            supports_memory_stats=device.type == "cuda",
            supports_fused_adamw=device.type == "cuda",
        )
        return cls(
            device=device,
            precision=precision_name,
            amp_dtype=amp_dtype,
            capabilities=capabilities,
        )

    def autocast(self) -> ContextManager[None]:
        """为一次 forward 创建新的 autocast context。"""

        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def synchronize(self) -> None:
        """等待当前 device 已提交的计算完成。CPU 路径不需要操作。"""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def manual_seed(self, seed: int) -> None:
        """设置 PyTorch CPU RNG，并在 CUDA 路径设置所有可见设备 RNG。"""

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def reset_peak_memory(self) -> None:
        """从当前时刻重新统计峰值设备内存。"""

        if self.capabilities.supports_memory_stats:
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats_mb(self) -> tuple[float, float]:
        """返回当前和峰值设备内存，单位 MB；不支持时返回 0。"""

        if not self.capabilities.supports_memory_stats:
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return float(allocated), float(peak)

    def device_name(self) -> str:
        """返回适合写入实验记录的设备名称。"""

        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return "CPU"


class DeviceIntervalTimer:
    """在不逐 step 同步设备的情况下测量一个统计窗口。

    wall time 表示调用方实际等待的端到端时间。CUDA 路径额外用 Event 记录设备工作时间，
    只在窗口结束时同步一次。
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self._wall_start: float | None = None
        self._device_start: torch.cuda.Event | None = None
        self.last_device_seconds: float | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            raise RuntimeError("计时窗口已经启动")
        self._wall_start = time.perf_counter()
        if self.runtime.capabilities.supports_device_events:
            self._device_start = torch.cuda.Event(enable_timing=True)
            self._device_start.record()
        self._running = True

    def elapsed_seconds(self) -> float:
        if not self._running or self._wall_start is None:
            raise RuntimeError("计时窗口尚未启动")

        if self.runtime.capabilities.supports_device_events:
            if self._device_start is None:
                raise RuntimeError("设备计时起始 Event 缺失")
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            self.last_device_seconds = self._device_start.elapsed_time(end) / 1000.0
            self._device_start = None
        else:
            self.last_device_seconds = None

        elapsed = time.perf_counter() - self._wall_start
        self._wall_start = None
        self._running = False
        return elapsed
