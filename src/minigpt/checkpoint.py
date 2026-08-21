"""Checkpoint save/load helpers.

训练 infra 里，checkpoint 不是“可选装饰”，而是核心能力。

一个能断点续训的 checkpoint 至少应该包含：
- model weights
- optimizer state
- mixed precision scaler state
- current training step
- config
- random number generator states

如果只保存 model weights，恢复后虽然能继续跑，但 optimizer 动量、学习率进度、
随机采样位置等都变了，训练曲线可能不连续。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scaler: Any,
    step: int,
    config: Dict[str, Any],
    best_val_loss: float | None,
) -> Dict[str, Any]:
    """组装 checkpoint dict。"""

    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "config": config,
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    return payload


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """Atomically save a single-file checkpoint.

    torch.save 底层使用 pickle + tensor storage。真实大模型会用分片 checkpoint，
    但单卡学习项目先用一个 .pt 文件最清楚。数据先写入同目录临时文件，成功 flush/fsync
    后再用 ``os.replace`` 原子替换目标，避免进程中断留下“名字正常、内容损坏”的 checkpoint。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        # fdopen owns the descriptor after it succeeds.  If it failed before
        # taking ownership, close the descriptor here; an already-closed fd is
        # harmlessly ignored.
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def update_latest_checkpoint(latest_path: str | Path, checkpoint_path: str | Path) -> str:
    """Atomically point ``latest.pt`` at an already-saved numbered checkpoint.

    A hard link avoids serializing and writing the same payload twice.  Some
    filesystems do not support hard links; those environments fall back to an
    atomic file copy while still avoiding a second ``torch.save`` call.

    Returns ``"hardlink"`` or ``"copy"`` so tests and logs can record the path
    actually used.
    """

    latest_path = Path(latest_path)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=latest_path.parent,
        prefix=f".{latest_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()

    method = "hardlink"
    try:
        try:
            os.link(checkpoint_path, temporary_path)
        except OSError:
            method = "copy"
            with checkpoint_path.open("rb") as source, temporary_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        os.replace(temporary_path, latest_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return method


def load_checkpoint(path: str | Path, map_location: torch.device | str) -> Dict[str, Any]:
    """读取 checkpoint。"""

    return torch.load(Path(path), map_location=map_location)


def restore_rng_state(payload: Dict[str, Any]) -> None:
    """恢复随机数状态。

    恢复 RNG 后，随机 batch 采样、dropout 等会更接近断点前的行为。
    """

    if "rng_state" in payload:
        torch.set_rng_state(payload["rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
