"""训练日志辅助函数。

本项目不引入 TensorBoard / WandB，是为了保持依赖最少。
训练日志会同时：
1. 打印到控制台，方便你边跑边看；
2. 写入 CSV，方便后续画 loss curve 或做实验对比。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable

import torch


class CSVLogger:
    """一个极简 CSV logger。"""

    def __init__(self, path: str | Path, fieldnames: Iterable[str], append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        file_exists = append and self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if append else "w"
        self.file = self.path.open(mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, row: Dict[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats_mb(device: torch.device) -> tuple[float, float]:
    """返回当前显存和峰值显存，单位 MB。

    CPU 训练时没有 GPU memory，返回 0。
    """

    if device.type != "cuda":
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return float(allocated), float(peak)


class DeviceIntervalTimer:
    """在不逐step同步加速器的情况下测量一个统计窗口。

    ``tokens_per_sec``使用纯训练窗口的端到端wall time。CUDA路径用Event界定
    已排队的设备工作，只在日志、验证或checkpoint边界关闭窗口时等待；纯设备
    时间保存在``last_device_seconds``，供后续profiler/metrics扩展。
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._wall_start: float | None = None
        self._cuda_start: torch.cuda.Event | None = None
        self.last_device_seconds: float | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Timer window is already running")
        self._wall_start = time.perf_counter()
        if self.device.type == "cuda":
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        self._running = True

    def elapsed_seconds(self) -> float:
        if not self._running:
            raise RuntimeError("Timer window has not been started")
        if self._wall_start is None:
            raise RuntimeError("Wall timer start value is missing")

        if self.device.type == "cuda":
            if self._cuda_start is None:
                raise RuntimeError("CUDA timer start event is missing")
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            self.last_device_seconds = self._cuda_start.elapsed_time(end) / 1000.0
            self._cuda_start = None
        else:
            self.last_device_seconds = None

        elapsed = time.perf_counter() - self._wall_start
        self._wall_start = None
        self._running = False
        return elapsed
