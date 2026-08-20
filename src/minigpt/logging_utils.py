"""Logging helpers for training.

本项目不引入 TensorBoard / WandB，是为了保持依赖最少。
训练日志会同时：
1. 打印到控制台，方便你边跑边看；
2. 写入 CSV，方便后续画 loss curve 或做实验对比。
"""

from __future__ import annotations

import csv
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


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA kernels before timing boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
