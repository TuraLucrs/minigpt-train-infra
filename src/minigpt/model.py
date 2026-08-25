"""显式保留Transformer结构并使用PyTorch原生基础算子的MiniGPT模型。

最初的教学基线刻意不用：
- torch.nn.Transformer
- torch.nn.TransformerEncoder
- torch.nn.MultiheadAttention

当前工程化版本仍然显式保留：
- causal self-attention
- Transformer block
- GPT forward

已经学完且官方实现更成熟的 LayerNorm、GELU 和 cross entropy 改用
PyTorch 原生算子。原始手写版本保存在 Git 标签 ``baseline-v0.1``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


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
        self.dropout = config.dropout

        # 一次 GEMM 同时产生 Q/K/V，减少 kernel launch 和重复读取 x。
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # qkv shape [B, T, 3*C]，按最后一维切成三份 [B, T, C]。
        q, k, v = self.qkv_proj(x).split(C, dim=-1)

        # 把 C 拆成 n_head * head_dim。
        # 变换后 shape: [B, n_head, T, head_dim]
        # transpose 的目的：让每个 head 可以独立做 attention。
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch SDPA会为当前device/dtype选择可用的最佳Attention kernel。
        # is_causal=True在保留GPT不可看未来规则的同时，避免物化[T,T] mask buffer。
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

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
        x = F.gelu(x, approximate="tanh")
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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
        self.ln_f = nn.LayerNorm(config.n_embd)
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


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, debug_checks: bool = False) -> torch.Tensor:
    """使用PyTorch优化后的cross-entropy kernel计算next-token loss。"""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [B, T]")

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(f"targets shape must be [B, T], got {tuple(targets.shape)} for logits {tuple(logits.shape)}")

    logits_flat = logits.reshape(B * T, V).float()
    targets_flat = targets.reshape(B * T)
    if debug_checks and torch.any((targets_flat < 0) | (targets_flat >= V)):
        raise ValueError("targets contain token ids outside the vocabulary range")
    return F.cross_entropy(logits_flat, targets_flat)


def migrate_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把教学版或分离QKV checkpoint迁移到当前模型布局。"""

    migrated = dict(state_dict)
    for key in tuple(migrated):
        if key.endswith(".attn.causal_mask"):
            del migrated[key]

    q_weight_suffix = ".attn.q_proj.weight"
    for q_weight_key in tuple(migrated):
        if not q_weight_key.endswith(q_weight_suffix):
            continue
        prefix = q_weight_key[: -len("q_proj.weight")]
        q_bias_key = prefix + "q_proj.bias"
        k_weight_key = prefix + "k_proj.weight"
        k_bias_key = prefix + "k_proj.bias"
        v_weight_key = prefix + "v_proj.weight"
        v_bias_key = prefix + "v_proj.bias"
        migrated[prefix + "qkv_proj.weight"] = torch.cat(
            [migrated[q_weight_key], migrated[k_weight_key], migrated[v_weight_key]],
            dim=0,
        )
        migrated[prefix + "qkv_proj.bias"] = torch.cat(
            [migrated[q_bias_key], migrated[k_bias_key], migrated[v_bias_key]],
            dim=0,
        )
        for old_key in (q_weight_key, q_bias_key, k_weight_key, k_bias_key, v_weight_key, v_bias_key):
            del migrated[old_key]
    return migrated


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
