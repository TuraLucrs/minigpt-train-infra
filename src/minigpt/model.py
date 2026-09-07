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
import torch
from torch import nn
from torch.nn import functional as F

from .cache_utils import normalize_cache_rows


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


@dataclass
class MiniGPTLayerKVCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class MiniGPTKVCache:
    layers: list[MiniGPTLayerKVCache]
    lengths: torch.Tensor
    max_batch_size: int
    max_seq_len: int
    batch_size: int = 0
    current_max_length: int = 0

    def reset(self, batch_size: int) -> None:
        if batch_size <= 0 or batch_size > self.max_batch_size:
            raise ValueError("batch_size 超出 KV Cache 容量")
        self.batch_size = batch_size
        self.current_max_length = 0
        self.lengths[:batch_size].zero_()


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

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return q, k, v

    def _output(self, y: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = y.shape
        C = self.n_head * self.head_dim
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return self.resid_dropout(y)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project(x)

        sdpa_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, T):
                raise ValueError("attention_mask 必须是 [B,T]")
            causal = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            sdpa_mask = causal[None, None] & attention_mask[:, None, None, :].bool()

        # PyTorch SDPA会为当前device/dtype选择可用的最佳Attention kernel。
        # is_causal=True在保留GPT不可看未来规则的同时，避免物化[T,T] mask buffer。
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            attn_mask=sdpa_mask,
            is_causal=sdpa_mask is None,
        )
        return self._output(y)

    def prefill_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: MiniGPTLayerKVCache,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project(x)
        rows, tokens = torch.where(attention_mask.bool())
        positions = position_ids[rows, tokens]
        target_rows = cache_rows[rows]
        cache.key[target_rows, :, positions] = k[rows, :, tokens]
        cache.value[target_rows, :, positions] = v[rows, :, tokens]
        all_valid = bool(torch.all(attention_mask).item())
        sdpa_mask = None
        if not all_valid:
            causal = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            sdpa_mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=sdpa_mask is None,
        )
        return self._output(y)

    def decode_with_cache(
        self,
        x: torch.Tensor,
        cache: MiniGPTLayerKVCache,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        new_lengths: torch.Tensor,
        max_length: int,
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._project(x)
        batch_size = x.shape[0]
        rows = torch.arange(batch_size, device=x.device)[active_mask]
        if rows.numel():
            write_positions = positions[active_mask]
            target_rows = cache_rows[rows]
            cache.key[target_rows, :, write_positions] = k[active_mask, :, 0]
            cache.value[target_rows, :, write_positions] = v[active_mask, :, 0]
        key_positions = torch.arange(max_length, device=x.device)
        allowed = key_positions[None] < new_lengths[:, None]
        y = F.scaled_dot_product_attention(
            q,
            cache.key[cache_rows, :, :max_length],
            cache.value[cache_rows, :, :max_length],
            attn_mask=allowed[:, None, None],
            dropout_p=0.0,
            is_causal=False,
        )
        return self._output(y)


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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def prefill_with_cache(self, x, attention_mask, position_ids, cache, cache_rows):
        x = x + self.attn.prefill_with_cache(
            self.ln_1(x), attention_mask, position_ids, cache, cache_rows
        )
        return x + self.mlp(self.ln_2(x))

    def decode_with_cache(
        self, x, cache, positions, active_mask, new_lengths, max_length, cache_rows
    ):
        x = x + self.attn.decode_with_cache(
            self.ln_1(x),
            cache,
            positions,
            active_mask,
            new_lengths,
            max_length,
            cache_rows,
        )
        return x + self.mlp(self.ln_2(x))


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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence length {T}; block_size is {self.config.block_size}")

        if attention_mask is None:
            positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask 必须与 input_ids 同为 [B,T]")
            positions = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)

        # token_emb shape: [B, T, C]
        # pos_emb shape: [T, C]，会自动 broadcast 到 [B, T, C]
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def allocate_kv_cache(
        self,
        max_batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MiniGPTKVCache:
        shape = (
            max_batch_size,
            self.config.n_head,
            self.config.block_size,
            self.config.n_embd // self.config.n_head,
        )
        layers = [
            MiniGPTLayerKVCache(
                key=torch.zeros(shape, device=device, dtype=dtype),
                value=torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(self.config.n_layer)
        ]
        return MiniGPTKVCache(
            layers=layers,
            lengths=torch.zeros(max_batch_size, dtype=torch.long, device=device),
            max_batch_size=max_batch_size,
            max_seq_len=self.config.block_size,
        )

    def prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: MiniGPTKVCache,
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask 必须与 input_ids 同为 [B,T]")
        lengths = attention_mask.long().sum(dim=1)
        if torch.any(lengths <= 0) or T > self.config.block_size:
            raise ValueError("无效 Prefill 输入")
        rows = normalize_cache_rows(
            cache_rows,
            batch_size=B,
            max_batch_size=cache.max_batch_size,
            device=input_ids.device,
        )
        if cache_rows is None:
            cache.reset(B)
        else:
            cache.lengths[rows].zero_()
            cache.batch_size = max(cache.batch_size, int(rows.max().item()) + 1)
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))
        for block, layer_cache in zip(self.blocks, cache.layers):
            x = block.prefill_with_cache(
                x, attention_mask, position_ids, layer_cache, rows
            )
        cache.lengths[rows] = lengths
        cache.current_max_length = int(cache.lengths.max().item())
        return self.lm_head(self.ln_f(x))

    def decode_with_cache(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: MiniGPTKVCache,
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = cache.batch_size if cache_rows is None else input_ids.shape[0]
        if input_ids.shape != (B, 1) or active_mask.shape != (B,):
            raise ValueError("Decode 需要 [B,1] input_ids 和 [B] active_mask")
        rows = normalize_cache_rows(
            cache_rows,
            batch_size=B,
            max_batch_size=cache.max_batch_size,
            device=input_ids.device,
        )
        positions = cache.lengths[rows].clone()
        new_lengths = positions + active_mask.long()
        max_length = int(new_lengths.max().item())
        if max_length > self.config.block_size:
            raise ValueError("Decode 超过 block_size")
        # 已完成的请求可能已经占满窗口，但它们本轮不写缓存。给这些 inactive row 使用安全的
        # 占位位置，避免 position == block_size 时发生越界；其输出随后不会被消费。
        safe_positions = torch.where(active_mask.bool(), positions, torch.zeros_like(positions))
        x = self.dropout(
            self.token_embedding(input_ids)
            + self.position_embedding(safe_positions[:, None])
        )
        for block, layer_cache in zip(self.blocks, cache.layers):
            x = block.decode_with_cache(
                x,
                layer_cache,
                positions,
                active_mask.bool(),
                new_lengths,
                max_length,
                rows,
            )
        cache.lengths[rows] = new_lengths
        cache.current_max_length = int(cache.lengths.max().item())
        return self.lm_head(self.ln_f(x))


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
