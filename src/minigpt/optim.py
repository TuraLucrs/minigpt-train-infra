"""Optimizer helpers that remain specific to this training loop.

AdamW, gradient clipping, and fp16 scaling now use PyTorch's maintained
implementations. The original teaching implementations remain available at
Git tag ``baseline-v0.1``.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the scheduled learning rate on every optimizer parameter group."""

    for group in optimizer.param_groups:
        group["lr"] = lr


def load_optimizer_state(optimizer: torch.optim.Optimizer, payload: dict[str, Any]) -> None:
    """Load native state or migrate a ``baseline-v0.1`` MiniAdamW state.

    The teaching optimizer stored state as a list aligned with model parameter
    order. Native PyTorch optimizers store parameter IDs plus parameter-group
    metadata. Supporting both formats keeps old checkpoints resumable.
    """

    if "param_groups" in payload:
        optimizer.load_state_dict(payload)
        return

    old_states = payload.get("state")
    if not isinstance(old_states, list):
        raise ValueError("Unsupported optimizer checkpoint format")

    params = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(old_states) != len(params):
        raise ValueError("Optimizer state does not match model parameter count")

    for group in optimizer.param_groups:
        group["lr"] = float(payload["lr"])
        group["betas"] = (float(payload["beta1"]), float(payload["beta2"]))
        group["eps"] = float(payload["eps"])
        group["weight_decay"] = float(payload["weight_decay"])

    step = float(payload["step_num"])
    for parameter, old_state in zip(params, old_states):
        optimizer.state[parameter] = {
            "step": torch.tensor(step, dtype=torch.float32, device=parameter.device),
            "exp_avg": old_state["exp_avg"].to(device=parameter.device, dtype=parameter.dtype),
            "exp_avg_sq": old_state["exp_avg_sq"].to(device=parameter.device, dtype=parameter.dtype),
        }


def load_grad_scaler_state(scaler: torch.amp.GradScaler, payload: dict[str, Any]) -> None:
    """Load native GradScaler state or migrate the teaching scaler format."""

    if not scaler.is_enabled() or not payload:
        return
    if "_growth_tracker" in payload:
        scaler.load_state_dict(payload)
        return
    if not payload.get("enabled", False):
        return

    scaler.load_state_dict(
        {
            "scale": float(payload["scale"]),
            "growth_factor": float(payload["growth_factor"]),
            "backoff_factor": float(payload["backoff_factor"]),
            "growth_interval": int(payload["growth_interval"]),
            "_growth_tracker": int(payload["growth_tracker"]),
        }
    )


def cosine_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Warm up linearly, then decay from ``base_lr`` to ``min_lr`` with cosine."""

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)
