"""可加载 Hugging Face safetensors 的 Qwen3 dense decoder reference 实现。

本模块保留模型数学和 KV Cache 数据流，tokenizer、生成编排与 Benchmark 放在其他模块。
当前只支持默认 RoPE 和全注意力；不静默接受尚未实现的 sliding window/rope scaling。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    torch_dtype: str | None = None
    initializer_range: float = 0.02
    use_sliding_window: bool = False
    sliding_window: int | None = None
    rope_scaling: dict | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "Qwen3Config":
        model_type = raw.get("model_type")
        if model_type is not None and model_type != "qwen3":
            raise ValueError(f"只支持 model_type=qwen3，收到 {model_type!r}")
        architectures = raw.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or "Qwen3ForCausalLM" not in architectures
        ):
            raise ValueError("只支持 Qwen3ForCausalLM causal-LM checkpoint")
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"Qwen3 config 缺少字段：{', '.join(missing)}")
        config = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            attention_bias=bool(raw.get("attention_bias", False)),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            bos_token_id=_optional_int(raw.get("bos_token_id")),
            eos_token_id=_optional_int(raw.get("eos_token_id")),
            pad_token_id=_optional_int(raw.get("pad_token_id")),
            torch_dtype=raw.get("torch_dtype"),
            initializer_range=float(raw.get("initializer_range", 0.02)),
            use_sliding_window=bool(raw.get("use_sliding_window", False)),
            sliding_window=_optional_int(raw.get("sliding_window")),
            rope_scaling=raw.get("rope_scaling"),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Qwen3 config.json 必须是 JSON object")
        return cls.from_dict(raw)

    def validate(self) -> None:
        positive_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} 必须大于 0")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        if self.hidden_act != "silu":
            raise ValueError("当前 Qwen3 实现只支持 hidden_act=silu")
        if self.attention_dropout < 0.0 or self.attention_dropout >= 1.0:
            raise ValueError("attention_dropout 必须位于 [0,1)")
        if self.use_sliding_window:
            raise ValueError("当前 Qwen3 实现尚不支持 sliding window attention")
        if self.rope_scaling is not None:
            raise ValueError("当前 Qwen3 实现尚不支持 rope_scaling")

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


@dataclass
class Qwen3LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class Qwen3KVCache:
    layers: list[Qwen3LayerKVCache]
    lengths: torch.Tensor
    max_batch_size: int
    max_seq_len: int
    batch_size: int = 0
    current_max_length: int = 0

    def reset(self, batch_size: int) -> None:
        if batch_size <= 0 or batch_size > self.max_batch_size:
            raise ValueError("batch_size 超出 Qwen3 KV Cache 容量")
        self.batch_size = batch_size
        self.current_max_length = 0
        self.lengths[:batch_size].zero_()


class Qwen3RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer(
            "inv_freq",
            self._build_inv_freq(device=device),
            persistent=False,
        )

    def _build_inv_freq(
        self,
        *,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        dimensions = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.theta ** (dimensions / self.head_dim))

    def reset_inv_freq(self, device: torch.device | str) -> None:
        self.inv_freq = self._build_inv_freq(device=device)

    @torch.no_grad()
    def forward(
        self,
        reference: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = reference.device.type if reference.device.type != "mps" else "cpu"
        # 与 Transformers reference 一致：RoPE 相位始终在 FP32 中计算，不能被外层
        # BF16/FP16 autocast 降精度，长上下文尤其依赖这一点。
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = torch.einsum(
                "d,bt->btd",
                self.inv_freq.float(),
                position_ids.float(),
            )
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos()
            sine = embedding.sin()
        return cosine.to(reference.dtype), sine.to(reference.dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine[:, None]
    sine = sine[:, None]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _causal_key_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length = attention_mask.shape
    causal = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=attention_mask.device,
    ).tril()
    return causal.view(1, 1, sequence_length, sequence_length).expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    ) & attention_mask[:, None, None, :].bool()


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.query_width,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.kv_width,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.kv_width,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            config.query_width,
            config.hidden_size,
            bias=config.attention_bias,
            **factory_kwargs,
        )
        self.q_norm = Qwen3RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
            **factory_kwargs,
        )
        self.k_norm = Qwen3RMSNorm(
            config.head_dim,
            config.rms_norm_eps,
            **factory_kwargs,
        )

    def _project(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        return query, key, value

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.config.num_attention_heads != self.config.num_key_value_heads,
        )
        batch_size, _, sequence_length, _ = output.shape
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.config.query_width,
        )
        return self.o_proj(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden_states)
        query, key = _apply_rope(query, key, *position_embeddings)
        sdpa_mask = None if attention_mask is None else _causal_key_mask(attention_mask)
        return self._attend(
            query,
            key,
            value,
            attention_mask=sdpa_mask,
            is_causal=sdpa_mask is None,
        )

    def prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Qwen3LayerKVCache,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden_states)
        query, key = _apply_rope(query, key, *position_embeddings)

        rows, token_positions = torch.where(attention_mask.bool())
        cache_positions = position_ids[rows, token_positions]
        cache.key[rows, :, cache_positions] = key[rows, :, token_positions]
        cache.value[rows, :, cache_positions] = value[rows, :, token_positions]

        sdpa_mask = _causal_key_mask(attention_mask)
        return self._attend(
            query,
            key,
            value,
            attention_mask=sdpa_mask,
            is_causal=False,
        )

    def decode_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: Qwen3LayerKVCache,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        new_lengths: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden_states)
        query, key = _apply_rope(query, key, *position_embeddings)
        batch_size = hidden_states.shape[0]
        active_rows = torch.arange(batch_size, device=hidden_states.device)[active_mask]
        if active_rows.numel():
            write_positions = positions[active_mask]
            cache.key[active_rows, :, write_positions] = key[active_mask, :, 0]
            cache.value[active_rows, :, write_positions] = value[active_mask, :, 0]

        key_positions = torch.arange(max_length, device=hidden_states.device)
        allowed = key_positions[None] < new_lengths[:, None]
        return self._attend(
            query,
            cache.key[:batch_size, :, :max_length],
            cache.value[:batch_size, :, :max_length],
            attention_mask=allowed[:, None, None],
            is_causal=False,
        )


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Qwen3Attention(config, **factory_kwargs)
        self.mlp = Qwen3MLP(config, **factory_kwargs)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **factory_kwargs,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Qwen3LayerKVCache,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.prefill_with_cache(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            position_ids,
            cache,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def decode_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: Qwen3LayerKVCache,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        new_lengths: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.decode_with_cache(
            self.input_layernorm(hidden_states),
            position_embeddings,
            cache,
            positions,
            active_mask,
            new_lengths,
            max_length,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, **factory_kwargs) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **factory_kwargs,
        )
        self.rotary_emb = Qwen3RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            device=device,
        )

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        return (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Qwen3 input_ids 必须是 [B,T]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Qwen3 输入超过 max_position_embeddings")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask 必须与 input_ids 同为 [B,T]")
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                ).unsqueeze(0)
            else:
                position_ids = self._position_ids(attention_mask)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        return self.norm(hidden_states)

    def prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Qwen3KVCache,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Qwen3 Prefill 需要同形状 [B,T] input_ids/attention_mask")
        batch_size, _ = input_ids.shape
        lengths = attention_mask.long().sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("每个 Qwen3 prompt 至少需要一个有效 token")
        if int(lengths.max().item()) > cache.max_seq_len:
            raise ValueError("Qwen3 Prefill 超过 KV Cache 容量")

        cache.reset(batch_size)
        position_ids = self._position_ids(attention_mask)
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer, layer_cache in zip(self.layers, cache.layers):
            hidden_states = layer.prefill_with_cache(
                hidden_states,
                position_embeddings,
                attention_mask,
                position_ids,
                layer_cache,
            )
        cache.lengths[:batch_size].copy_(lengths)
        cache.current_max_length = int(lengths.max().item())
        return self.norm(hidden_states)

    def decode_with_cache(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: Qwen3KVCache,
    ) -> torch.Tensor:
        batch_size = cache.batch_size
        if input_ids.shape != (batch_size, 1) or active_mask.shape != (batch_size,):
            raise ValueError("Qwen3 Decode 需要 [B,1] input_ids 和 [B] active_mask")
        active_mask = active_mask.bool()
        positions = cache.lengths[:batch_size].clone()
        new_lengths = positions + active_mask.long()
        max_length = int(new_lengths.max().item())
        if max_length > cache.max_seq_len:
            raise ValueError("Qwen3 Decode 超过 KV Cache 容量")

        safe_positions = torch.where(active_mask, positions, torch.zeros_like(positions))
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, safe_positions[:, None])
        for layer, layer_cache in zip(self.layers, cache.layers):
            hidden_states = layer.decode_with_cache(
                hidden_states,
                position_embeddings,
                layer_cache,
                positions,
                active_mask,
                new_lengths,
                max_length,
            )
        cache.lengths[:batch_size].copy_(new_lengths)
        cache.current_max_length = max_length
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.model = Qwen3Model(config, **factory_kwargs)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            **factory_kwargs,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        if device != "meta" and not (
            isinstance(device, torch.device) and device.type == "meta"
        ):
            self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        logit_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        if logit_positions is not None:
            if logit_positions.shape != (input_ids.shape[0],):
                raise ValueError("logit_positions 必须是 [B]")
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden_states = hidden_states[batch_indices, logit_positions]
        return self.lm_head(hidden_states)

    def allocate_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen3KVCache:
        if max_seq_len <= 0 or max_seq_len > self.config.max_position_embeddings:
            raise ValueError("max_seq_len 超出 Qwen3 上下文容量")
        shape = (
            max_batch_size,
            self.config.num_key_value_heads,
            max_seq_len,
            self.config.head_dim,
        )
        layers = [
            Qwen3LayerKVCache(
                # 不同 prompt 长度共享 max_length 时，较短行的尾部仍会参加 QK
                # matmul 后再被 mask。必须初始化为有限值，避免未初始化 NaN 在 mask
                # 生效前污染注意力分数。
                key=torch.zeros(shape, device=device, dtype=dtype),
                value=torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        return Qwen3KVCache(
            layers=layers,
            lengths=torch.zeros(max_batch_size, dtype=torch.long, device=device),
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    def prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Qwen3KVCache,
        logit_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model.prefill_with_cache(input_ids, attention_mask, cache)
        if logit_positions is not None:
            if logit_positions.shape != (input_ids.shape[0],):
                raise ValueError("logit_positions 必须是 [B]")
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden_states = hidden_states[batch_indices, logit_positions]
        return self.lm_head(hidden_states)

    def decode_with_cache(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: Qwen3KVCache,
    ) -> torch.Tensor:
        hidden_states = self.model.decode_with_cache(input_ids, active_mask, cache)
        return self.lm_head(hidden_states)

    def reset_non_persistent_buffers(self, device: torch.device) -> None:
        self.model.rotary_emb.reset_inv_freq(device)


def count_qwen3_parameters(model_or_config: Qwen3ForCausalLM | Qwen3Config) -> int:
    if isinstance(model_or_config, Qwen3ForCausalLM):
        return sum(parameter.numel() for parameter in model_or_config.parameters())
    config = model_or_config
    attention = (
        config.hidden_size * config.query_width
        + 2 * config.hidden_size * config.kv_width
        + config.query_width * config.hidden_size
        + 2 * config.head_dim
    )
    if config.attention_bias:
        attention += config.query_width + 2 * config.kv_width + config.hidden_size
    mlp = 3 * config.hidden_size * config.intermediate_size
    norms = 2 * config.hidden_size
    layers = config.num_hidden_layers * (attention + mlp + norms)
    embeddings = config.vocab_size * config.hidden_size
    output = 0 if config.tie_word_embeddings else config.vocab_size * config.hidden_size
    return embeddings + layers + config.hidden_size + output


def _safetensor_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index 缺少 weight_map")
        return [model_dir / name for name in sorted(set(weight_map.values()))]
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"{model_dir} 中没有 model.safetensors 或 sharded index")


def iter_safetensor_weights(model_dir: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("加载 Qwen3 权重需要安装 safetensors") from exc

    directory = Path(model_dir)
    for shard_path in _safetensor_files(directory):
        if not shard_path.is_file():
            raise FileNotFoundError(f"缺少 safetensors shard：{shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def load_qwen3_from_pretrained(
    model_dir: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3ForCausalLM:
    """用 meta 初始化和直接 CPU→target copy 控制 32B 权重加载峰值。"""

    directory = Path(model_dir)
    config = Qwen3Config.from_json(directory / "config.json")
    model = Qwen3ForCausalLM(config, device="meta", dtype=dtype)
    model.to_empty(device=device)
    model.reset_non_persistent_buffers(device)

    targets = dict(model.named_parameters(remove_duplicate=False))
    loaded: set[str] = set()
    with torch.no_grad():
        for name, source in iter_safetensor_weights(directory):
            target = targets.get(name)
            if target is None:
                raise ValueError(f"checkpoint 包含未知参数：{name}")
            if name in loaded:
                raise ValueError(f"checkpoint 重复包含参数：{name}")
            if tuple(target.shape) != tuple(source.shape):
                raise ValueError(
                    f"参数 {name} shape 不匹配：checkpoint={tuple(source.shape)}，"
                    f"model={tuple(target.shape)}"
                )
            target.copy_(source)
            loaded.add(name)

    missing = sorted(set(targets) - loaded)
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"checkpoint 缺少 {len(missing)} 个参数：{preview}")
    model.eval()
    return model
