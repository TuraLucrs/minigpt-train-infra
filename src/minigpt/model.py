"""MiniGPT model implemented with explicit Transformer building blocks.

这个文件是项目最重要的学习材料之一。它刻意不用：
- torch.nn.Transformer
- torch.nn.TransformerEncoder
- torch.nn.MultiheadAttention
- torch.nn.LayerNorm
- torch.nn.functional.cross_entropy

我们自己写：
- LayerNorm
- GELU
- causal self-attention
- Transformer block
- GPT forward
- next-token cross entropy loss

但我们仍然使用 PyTorch 的基础能力：
- Tensor 运算
- nn.Linear / nn.Embedding / nn.Dropout
- autograd 自动求导

完全手写矩阵乘和反向传播不是这个阶段的重点；理解训练系统和 Transformer 结构才是。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn


@dataclass
class MiniGPTConfig:
    """模型配置。

    vocab_size: tokenizer 的词表大小。
    block_size: 模型一次最多看多少个 token，也叫 context length。
    n_layer: Transformer block 层数。
    n_head: attention head 数量。
    n_embd: 每个 token 的隐藏向量维度。
    dropout: 训练时随机丢弃一部分激活，帮助小模型不要太快过拟合。
    """

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1


class MiniLayerNorm(nn.Module):
    """手写 LayerNorm。

    LayerNorm 对每个 token 的最后一维 hidden dimension 做归一化。

    输入 x shape: [B, T, C]
    B = batch size
    T = sequence length
    C = hidden dimension / n_embd

    对于每个 [B, T] 位置上的 C 维向量：
    1. 减去均值；
    2. 除以标准差；
    3. 乘可学习参数 gamma；
    4. 加可学习参数 beta。
    """

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.weight * normalized + self.bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    """手写 GELU 激活函数的 tanh 近似版本。

    GPT 系列模型常用 GELU，而不是 ReLU。这里不用 torch.nn.GELU，
    是为了让你看到它本质上只是一个逐元素非线性变换。
    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    """GPT 的 masked multi-head self-attention。

    self-attention 做的事情：
    - 每个 token 生成 query/key/value 三个向量；
    - query 和 key 做点积，得到“我应该看谁”的分数；
    - causal mask 禁止当前位置看到未来 token；
    - softmax 把分数变成概率；
    - 用概率加权 value，得到新的 token 表示。

    multi-head 的意思是把 n_embd 切成多个 head，每个 head 学不同关系。
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # 这里不用 nn.MultiheadAttention，而是显式写出 Q/K/V 三个投影。
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal_mask shape: [1, 1, block_size, block_size]
        # 下三角为 True，表示当前位置可以看自己和过去；上三角为 False，表示不能看未来。
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # q/k/v 原始 shape 都是 [B, T, C]。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 把 C 拆成 n_head * head_dim。
        # 变换后 shape: [B, n_head, T, head_dim]
        # transpose 的目的：让每个 head 可以独立做 attention。
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # attention scores shape: [B, n_head, T, T]
        # 最后两个维度表示：每个 query token 对每个 key token 的分数。
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 把未来位置的分数设成 -inf，softmax 后概率会变成 0。
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # attention weights shape: [B, n_head, T, T]
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # weighted sum of values:
        # [B, n_head, T, T] @ [B, n_head, T, head_dim]
        # -> [B, n_head, T, head_dim]
        y = weights @ v

        # 把多头拼回 C 维：先转回 [B, T, n_head, head_dim]，再 view 成 [B, T, C]。
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    """Transformer block 里的 MLP。

    常见结构是：
    Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd)

    中间扩大 4 倍，是为了给模型更多非线性表达能力。
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """一个 GPT Transformer block。

    这里使用 Pre-LN 结构：

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

    残差连接非常重要：深层网络如果没有残差，梯度传播会困难得多。
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.ln_1 = MiniLayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = MiniLayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """一个最小 GPT 语言模型。

    forward 输入:
        input_ids shape [B, T]

    forward 输出:
        logits shape [B, T, vocab_size]

    logits[b, t, v] 表示第 b 个样本、第 t 个位置，模型认为下一个 token 是 v 的原始分数。
    """

    def __init__(self, config: MiniGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = MiniLayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：输入 token embedding 和输出 lm_head 使用同一份权重。
        # 这是 GPT 系列常见技巧，可以减少参数量，也常常带来一点效果提升。
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化参数。

        小模型对初始化也敏感。这里使用 GPT 常见的 normal(0, 0.02)。
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence length {T}; block_size is {self.config.block_size}")

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # token_emb shape: [B, T, C]
        # pos_emb shape: [T, C]，会自动 broadcast 到 [B, T, C]
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """简单自回归采样，用来感受训练后的模型会输出什么。

        这不是训练主线，但很适合调试：如果 loss 下降了，生成文本通常会变得
        更像训练语料，即使这个小模型不会真的“聪明”。
        """

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        for _ in range(max_new_tokens):
            # 如果上下文太长，只取最后 block_size 个 token。
            context = input_ids[:, -self.config.block_size :]
            logits = self(context)

            # 只看最后一个位置的 logits，因为它预测下一个 token。
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, debug_checks: bool = False) -> torch.Tensor:
    """手写 next-token cross entropy。

    PyTorch 里通常会写：
        F.cross_entropy(logits.view(-1, V), targets.view(-1))

    这里不用它，而是自己展开公式，方便你理解 loss 到底在算什么。

    对一个位置来说：
        loss = -log softmax(logits)[target]
             = log(sum(exp(logits))) - logits[target]

    为了数值稳定，logsumexp 用“减最大值”的方式计算。
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [B, T]")

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(f"targets shape must be [B, T], got {tuple(targets.shape)} for logits {tuple(logits.shape)}")

    # Under fp16/bf16 autocast, logits can be low precision. Keep the loss math in fp32:
    # exp/sum/log are exactly where low precision tends to hurt numerical stability.
    logits_flat = logits.reshape(B * T, V).float()
    targets_flat = targets.reshape(B * T)
    if debug_checks and torch.any((targets_flat < 0) | (targets_flat >= V)):
        raise ValueError("targets contain token ids outside the vocabulary range")

    max_logits = logits_flat.max(dim=-1, keepdim=True).values
    shifted = logits_flat - max_logits
    logsumexp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))

    target_logits = logits_flat.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
    losses = logsumexp - target_logits
    return losses.mean()


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
