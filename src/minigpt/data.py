"""数据加载辅助函数。

这个项目没有使用 torch.utils.data.DataLoader。不是因为 DataLoader 不好，而是因为
第一个训练 infra 项目最好亲眼看到 batch 是怎么切出来的。

GPT 预训练通常把文本看作一个很长的 token 序列：

tokens = [t0, t1, t2, t3, ...]

如果 block_size = 4，那么一个训练样本可以是：

x = [t0, t1, t2, t3]
y = [t1, t2, t3, t4]

也就是说，模型在每个位置都预测“下一个 token”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def split_train_val(tokens: torch.Tensor, val_fraction: float, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """把一条 token 序列切成 train/val 两段。

    注意：val 至少要有 block_size + 1 个 token，否则无法构造 x/y。
    如果语料太小，这里会主动报错，而不是让后面的训练循环出现奇怪 shape。
    """

    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1-D tensor")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # 一个GPT样本需要block_size+1个token：
    # x = tokens[start : start + block_size]
    # y = tokens[start + 1 : start + block_size + 1]
    min_required = block_size + 1
    if tokens.numel() < min_required * 2:
        raise ValueError(
            f"Corpus is too small for block_size={block_size}. "
            f"Need at least {min_required * 2} tokens, got {tokens.numel()}."
        )

    val_tokens = max(min_required, int(tokens.numel() * val_fraction))
    train_tokens = tokens.numel() - val_tokens
    if train_tokens < min_required:
        raise ValueError("Training split is too small. Reduce val_fraction or block_size.")

    return tokens[:train_tokens].contiguous(), tokens[train_tokens:].contiguous()


@dataclass
class RandomTokenBatcher:
    """从连续 token 序列里随机采样 GPT 训练 batch。

    这个类就是一个极简 data loader：
    - 不做多进程；
    - 不做复杂 shuffle；
    - 不做磁盘 streaming；
    - 每次随机选 batch_size 个起点，然后切 block_size 长度。

    这样写的好处是非常透明，适合你先理解“语言模型训练样本”到底长什么样。
    """

    tokens: torch.Tensor
    batch_size: int
    block_size: int
    device: torch.device
    seed: int

    def __post_init__(self) -> None:
        if self.tokens.ndim != 1:
            raise ValueError("RandomTokenBatcher expects a 1-D token tensor")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.tokens.numel() < self.block_size + 1:
            raise ValueError("Not enough tokens to sample one training block")

        # 采样起点在 CPU 上生成即可。真正的 x/y 会在最后搬到训练设备。
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回一个 batch: x 和 y。

        x shape: [batch_size, block_size]
        y shape: [batch_size, block_size]

        y 比 x 向右移动一格，所以 y[b, i] 是 x[b, i] 的下一个 token。
        """

        # 合法起点为闭区间[0, len(tokens)-block_size-1]。
        # torch.randint的high不包含在取值范围内，因此high应为len(tokens)-block_size。
        num_possible_starts = self.tokens.numel() - self.block_size
        starts = torch.randint(
            low=0,
            high=num_possible_starts,
            size=(self.batch_size,),
            generator=self.generator,
        )

        offsets = torch.arange(self.block_size)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        x = self.tokens[indices].to(self.device, non_blocking=True)
        y = self.tokens[indices + 1].to(self.device, non_blocking=True)
        return x, y
