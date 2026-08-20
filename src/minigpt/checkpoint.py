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
    """保存 checkpoint。

    torch.save 底层使用 pickle + tensor storage。真实大模型会用分片 checkpoint，
    但单卡学习项目先用一个 .pt 文件最清楚。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
