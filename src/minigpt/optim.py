"""Small training utilities implemented from scratch.

这个文件覆盖训练 infra 里很常见的三个点：

1. AdamW optimizer
2. cosine learning rate schedule with warmup
3. fp16 gradient scaling

真实项目里你通常会直接用 torch.optim.AdamW 和 torch.cuda.amp.GradScaler。
这里手写一版，是为了学习它们的核心机制。
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch


def _trainable_params(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


class MiniAdamW:
    """一个教学版 AdamW optimizer。

    AdamW 的核心状态：
    - exp_avg: 梯度的一阶动量，类似“最近梯度方向的滑动平均”
    - exp_avg_sq: 梯度平方的二阶动量，类似“每个参数历史梯度大小”

    W 表示 decoupled weight decay：权重衰减不混进梯度里，而是直接对参数做缩放。

    这个类没有继承 torch.optim.Optimizer，是为了让代码更直白。代价是少了一些
    PyTorch optimizer 的高级功能，但足够本项目学习使用。
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = _trainable_params(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0

        self.state = []
        for p in self.params:
            self.state.append(
                {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
            )

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清空梯度。

        set_to_none=True 会把 grad 设成 None，通常更省一点内存。
        """

        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        """执行一次 AdamW 参数更新。"""

        self.step_num += 1

        beta1 = self.beta1
        beta2 = self.beta2

        for p, state in zip(self.params, self.state):
            if p.grad is None:
                continue

            grad = p.grad
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # decoupled weight decay: 直接缩小参数本身，而不是改 gradient。
            if self.weight_decay != 0:
                p.mul_(1.0 - self.lr * self.weight_decay)

            # 更新一阶/二阶动量。
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            # bias correction: 训练初期 exp_avg / exp_avg_sq 会偏小，需要修正。
            bias_correction1 = 1.0 - beta1**self.step_num
            bias_correction2 = 1.0 - beta2**self.step_num

            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1
            denom = exp_avg_sq.sqrt().add_(self.eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_num": self.step_num,
            "state": self.state,
        }

    def load_state_dict(self, payload: dict) -> None:
        if len(payload["state"]) != len(self.state):
            raise ValueError("Optimizer state does not match model parameter count")
        self.lr = payload["lr"]
        self.beta1 = payload["beta1"]
        self.beta2 = payload["beta2"]
        self.eps = payload["eps"]
        self.weight_decay = payload["weight_decay"]
        self.step_num = payload["step_num"]
        self.state = payload["state"]


def cosine_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """warmup + cosine decay 学习率。

    训练刚开始时参数是随机的，直接上大学习率容易不稳定，所以先 warmup。
    warmup 后用 cosine 慢慢衰减到 min_lr。
    """

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


@torch.no_grad()
def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """手写 gradient clipping。

    如果梯度整体 norm 太大，参数更新会很猛，训练可能 NaN 或发散。
    clipping 会把所有梯度按同一个比例缩小，让总 norm 不超过 max_norm。
    """

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    total = torch.zeros((), device=params[0].grad.device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    total_norm = torch.sqrt(total).item()

    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(scale)

    return float(total_norm)


class SimpleGradScaler:
    """教学版 fp16 loss scaler。

    为什么 fp16 需要 loss scaling？
    -------------------------------
    fp16 表示范围比 fp32 小很多。反向传播里有些梯度会非常小，小到 fp16 表示不了，
    就会 underflow 成 0。loss scaling 的做法是：

    1. backward 前，把 loss 乘以一个较大的 scale；
    2. backward 后，梯度也被同样放大；
    3. optimizer step 前，再把梯度除以 scale；
    4. 如果发现梯度里有 inf/nan，说明 scale 太大，跳过本次 step 并降低 scale。

    bf16 通常不需要 loss scaling，因为它的指数范围和 fp32 更接近。
    """

    def __init__(
        self,
        enabled: bool,
        init_scale: float = 2.0**12,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self.enabled = enabled
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return loss
        return loss * self.scale

    @torch.no_grad()
    def unscale_(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        if not self.enabled:
            return
        inv_scale = 1.0 / self.scale
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(inv_scale)

    @torch.no_grad()
    def has_inf_or_nan(self, parameters: Iterable[torch.nn.Parameter]) -> bool:
        if not self.enabled:
            return False
        for p in parameters:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def update(self, found_inf: bool) -> None:
        if not self.enabled:
            return
        if found_inf:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self.growth_tracker = 0
            return
        self.growth_tracker += 1
        if self.growth_tracker >= self.growth_interval:
            self.scale *= self.growth_factor
            self.growth_tracker = 0

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "growth_tracker": self.growth_tracker,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.enabled = payload["enabled"]
        self.scale = payload["scale"]
        self.growth_factor = payload["growth_factor"]
        self.backoff_factor = payload["backoff_factor"]
        self.growth_interval = payload["growth_interval"]
        self.growth_tracker = payload["growth_tracker"]
