"""Qwen3 dense decoder 的 Tensor Parallel 推理与分片 safetensors 加载。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cache_utils import normalize_cache_rows
from .distributed import DistributedContext
from .inference import GenerationConfig, InferenceEngine
from .qwen3 import (
    Qwen3Config,
    Qwen3KVCache,
    Qwen3LayerKVCache,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    _apply_rope,
    _causal_key_mask,
)
from .qwen3_inference import (
    CachedQwen3ModelRunner,
    Qwen3Tokenizer,
    RecomputeQwen3ModelRunner,
    SlotCachedQwen3ModelRunner,
    validate_qwen3_tokenizer_config,
)


@dataclass(frozen=True)
class Qwen3TensorParallelPlan:
    rank: int
    world_size: int
    head_dim: int
    query_head_start: int
    query_heads: int
    kv_head_start: int
    kv_heads: int
    intermediate_start: int
    intermediate_size: int
    vocab_start: int
    vocab_size: int

    @classmethod
    def create(
        cls,
        config: Qwen3Config,
        rank: int,
        world_size: int,
    ) -> "Qwen3TensorParallelPlan":
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("无效的 TP rank/world_size")
        if config.attention_bias:
            raise ValueError("当前 Qwen3 TP 路径只支持 attention_bias=false")
        if config.tie_word_embeddings:
            raise ValueError("当前 Qwen3 TP 路径只支持未绑定的 embedding/lm_head")
        if config.num_attention_heads % world_size != 0:
            raise ValueError("Qwen3 query heads 必须能被 TP world_size 整除")
        if config.intermediate_size % world_size != 0:
            raise ValueError("Qwen3 intermediate_size 必须能被 TP world_size 整除")
        if config.vocab_size % world_size != 0:
            raise ValueError("Qwen3 vocab_size 必须能被 TP world_size 整除")

        query_heads = config.num_attention_heads // world_size
        query_head_start = rank * query_heads
        if config.num_key_value_heads % world_size == 0:
            kv_heads = config.num_key_value_heads // world_size
            kv_head_start = rank * kv_heads
        elif world_size % config.num_key_value_heads == 0:
            # TP rank 多于 KV heads 时，让处理同一组 query heads 的 rank 复制对应 KV head。
            ranks_per_kv_head = world_size // config.num_key_value_heads
            kv_heads = 1
            kv_head_start = rank // ranks_per_kv_head
        else:
            raise ValueError("当前 TP world_size 无法均匀分配或复制 Qwen3 KV heads")

        intermediate_size = config.intermediate_size // world_size
        vocab_size = config.vocab_size // world_size
        return cls(
            rank=rank,
            world_size=world_size,
            head_dim=config.head_dim,
            query_head_start=query_head_start,
            query_heads=query_heads,
            kv_head_start=kv_head_start,
            kv_heads=kv_heads,
            intermediate_start=rank * intermediate_size,
            intermediate_size=intermediate_size,
            vocab_start=rank * vocab_size,
            vocab_size=vocab_size,
        )

    @property
    def query_width_start(self) -> int:
        return self.query_head_start * self.head_dim

    @property
    def query_width(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_width_start(self) -> int:
        return self.kv_head_start * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


def _create_plan(
    config: Qwen3Config,
    distributed: DistributedContext,
) -> Qwen3TensorParallelPlan:
    return Qwen3TensorParallelPlan.create(
        config,
        distributed.rank,
        distributed.world_size,
    )


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        plan: Qwen3TensorParallelPlan,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.vocab_start = plan.vocab_start
        self.vocab_end = plan.vocab_start + plan.vocab_size
        self.full_vocab_size = config.vocab_size
        self.distributed = distributed
        self.weight = nn.Parameter(
            torch.empty(plan.vocab_size, config.hidden_size, device=device, dtype=dtype)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.full_vocab_size):
            raise ValueError("Qwen3 input_ids 超出模型词表")
        owned = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~owned, 0)
        hidden_states = F.embedding(local_ids, self.weight)
        hidden_states = hidden_states.masked_fill(~owned.unsqueeze(-1), 0.0)
        return self.distributed.all_reduce_sum(hidden_states)


class TensorParallelQwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        plan: Qwen3TensorParallelPlan,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.plan = plan
        self.distributed = distributed
        self.q_proj = nn.Linear(
            config.hidden_size,
            plan.query_width,
            bias=False,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            plan.kv_width,
            bias=False,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            plan.kv_width,
            bias=False,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            plan.query_width,
            config.hidden_size,
            bias=False,
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
            self.plan.query_heads,
            self.config.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.plan.kv_heads,
            self.config.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.plan.kv_heads,
            self.config.head_dim,
        )
        return (
            self.q_norm(query).transpose(1, 2),
            self.k_norm(key).transpose(1, 2),
            value.transpose(1, 2),
        )

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
            enable_gqa=self.plan.query_heads != self.plan.kv_heads,
        )
        batch_size, _, sequence_length, _ = output.shape
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.plan.query_width,
            )
        )
        output = self.o_proj(output)
        return self.distributed.all_reduce_sum(output)

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
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden_states)
        query, key = _apply_rope(query, key, *position_embeddings)
        rows, token_positions = torch.where(attention_mask.bool())
        cache_positions = position_ids[rows, token_positions]
        target_rows = cache_rows[rows]
        cache.key[target_rows, :, cache_positions] = key[rows, :, token_positions]
        cache.value[target_rows, :, cache_positions] = value[rows, :, token_positions]
        return self._attend(
            query,
            key,
            value,
            attention_mask=_causal_key_mask(attention_mask),
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
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self._project(hidden_states)
        query, key = _apply_rope(query, key, *position_embeddings)
        batch_size = hidden_states.shape[0]
        active_rows = torch.arange(batch_size, device=hidden_states.device)[active_mask]
        if active_rows.numel():
            write_positions = positions[active_mask]
            target_rows = cache_rows[active_rows]
            cache.key[target_rows, :, write_positions] = key[active_mask, :, 0]
            cache.value[target_rows, :, write_positions] = value[active_mask, :, 0]
        key_positions = torch.arange(max_length, device=hidden_states.device)
        allowed = key_positions[None] < new_lengths[:, None]
        return self._attend(
            query,
            cache.key[cache_rows, :, :max_length],
            cache.value[cache_rows, :, :max_length],
            attention_mask=allowed[:, None, None],
            is_causal=False,
        )


class TensorParallelQwen3MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        plan: Qwen3TensorParallelPlan,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.distributed = distributed
        self.gate_proj = nn.Linear(
            config.hidden_size,
            plan.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            plan.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            plan.intermediate_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return self.distributed.all_reduce_sum(hidden_states)


class TensorParallelQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        plan: Qwen3TensorParallelPlan,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = TensorParallelQwen3Attention(
            config,
            plan,
            distributed,
            **factory_kwargs,
        )
        self.mlp = TensorParallelQwen3MLP(
            config,
            plan,
            distributed,
            **factory_kwargs,
        )
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
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.prefill_with_cache(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            position_ids,
            cache,
            cache_rows,
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
        cache_rows: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn.decode_with_cache(
            self.input_layernorm(hidden_states),
            position_embeddings,
            cache,
            positions,
            active_mask,
            new_lengths,
            max_length,
            cache_rows,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TensorParallelQwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        plan: Qwen3TensorParallelPlan,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.plan = plan
        self.embed_tokens = VocabParallelEmbedding(
            config,
            plan,
            distributed,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                TensorParallelQwen3DecoderLayer(
                    config,
                    plan,
                    distributed,
                    **factory_kwargs,
                )
                for _ in range(config.num_hidden_layers)
            ]
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
            raise ValueError("Qwen3 TP input_ids 必须是 [B,T]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Qwen3 TP 输入超过 max_position_embeddings")
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
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Qwen3 TP Prefill 需要同形状 [B,T] input_ids/attention_mask"
            )
        batch_size = input_ids.shape[0]
        lengths = attention_mask.long().sum(dim=1)
        if torch.any(lengths <= 0):
            raise ValueError("每个 Qwen3 TP prompt 至少需要一个有效 token")
        if int(lengths.max().item()) > cache.max_seq_len:
            raise ValueError("Qwen3 TP Prefill 超过 KV Cache 容量")
        rows = normalize_cache_rows(
            cache_rows,
            batch_size=batch_size,
            max_batch_size=cache.max_batch_size,
            device=input_ids.device,
        )
        if cache_rows is None:
            cache.reset(batch_size)
        else:
            cache.lengths[rows].zero_()
            cache.batch_size = max(cache.batch_size, int(rows.max().item()) + 1)
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
                rows,
            )
        cache.lengths[rows] = lengths
        cache.current_max_length = int(cache.lengths.max().item())
        return self.norm(hidden_states)

    def decode_with_cache(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: Qwen3KVCache,
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = cache.batch_size if cache_rows is None else input_ids.shape[0]
        if input_ids.shape != (batch_size, 1) or active_mask.shape != (batch_size,):
            raise ValueError("Qwen3 TP Decode 需要 [B,1] input_ids 和 [B] active_mask")
        active_mask = active_mask.bool()
        rows = normalize_cache_rows(
            cache_rows,
            batch_size=batch_size,
            max_batch_size=cache.max_batch_size,
            device=input_ids.device,
        )
        positions = cache.lengths[rows].clone()
        new_lengths = positions + active_mask.long()
        max_length = int(new_lengths.max().item())
        if max_length > cache.max_seq_len:
            raise ValueError("Qwen3 TP Decode 超过 KV Cache 容量")
        safe_positions = torch.where(
            active_mask, positions, torch.zeros_like(positions)
        )
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
                rows,
            )
        cache.lengths[rows] = new_lengths
        cache.current_max_length = int(cache.lengths.max().item())
        return self.norm(hidden_states)


class TensorParallelQwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        distributed: DistributedContext,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.distributed = distributed
        self.plan = _create_plan(config, distributed)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.model = TensorParallelQwen3Model(
            config,
            self.plan,
            distributed,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            self.plan.vocab_size,
            bias=False,
            **factory_kwargs,
        )
        if device != "meta" and not (
            isinstance(device, torch.device) and device.type == "meta"
        ):
            self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, VocabParallelEmbedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _gather_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = self.lm_head(hidden_states)
        logits = self.distributed.all_gather_last_dim(local_logits)
        return logits[..., : self.config.vocab_size]

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
        return self._gather_logits(hidden_states)

    def allocate_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen3KVCache:
        if max_seq_len <= 0 or max_seq_len > self.config.max_position_embeddings:
            raise ValueError("max_seq_len 超出 Qwen3 TP 上下文容量")
        shape = (
            max_batch_size,
            self.plan.kv_heads,
            max_seq_len,
            self.config.head_dim,
        )
        return Qwen3KVCache(
            layers=[
                Qwen3LayerKVCache(
                    key=torch.zeros(shape, device=device, dtype=dtype),
                    value=torch.zeros(shape, device=device, dtype=dtype),
                )
                for _ in range(self.config.num_hidden_layers)
            ],
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
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model.prefill_with_cache(
            input_ids,
            attention_mask,
            cache,
            cache_rows=cache_rows,
        )
        if logit_positions is not None:
            if logit_positions.shape != (input_ids.shape[0],):
                raise ValueError("logit_positions 必须是 [B]")
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            hidden_states = hidden_states[batch_indices, logit_positions]
        return self._gather_logits(hidden_states)

    def decode_with_cache(
        self,
        input_ids: torch.Tensor,
        active_mask: torch.Tensor,
        cache: Qwen3KVCache,
        *,
        cache_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model.decode_with_cache(
            input_ids,
            active_mask,
            cache,
            cache_rows=cache_rows,
        )
        return self._gather_logits(hidden_states)

    def reset_non_persistent_buffers(self, device: torch.device) -> None:
        self.model.rotary_emb.reset_inv_freq(device)


def expected_qwen3_parameter_shapes(config: Qwen3Config) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
        "lm_head.weight": (config.vocab_size, config.hidden_size),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (
                    config.query_width,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.k_proj.weight": (
                    config.kv_width,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.v_proj.weight": (
                    config.kv_width,
                    config.hidden_size,
                ),
                f"{prefix}.self_attn.o_proj.weight": (
                    config.hidden_size,
                    config.query_width,
                ),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
            }
        )
    return shapes


def _parameter_slice(
    name: str,
    plan: Qwen3TensorParallelPlan,
) -> tuple[slice, ...]:
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return (
            slice(plan.vocab_start, plan.vocab_start + plan.vocab_size),
            slice(None),
        )
    if name.endswith(".self_attn.q_proj.weight"):
        return (
            slice(
                plan.query_width_start,
                plan.query_width_start + plan.query_width,
            ),
            slice(None),
        )
    if name.endswith((".self_attn.k_proj.weight", ".self_attn.v_proj.weight")):
        return (
            slice(plan.kv_width_start, plan.kv_width_start + plan.kv_width),
            slice(None),
        )
    if name.endswith(".self_attn.o_proj.weight"):
        return (
            slice(None),
            slice(
                plan.query_width_start,
                plan.query_width_start + plan.query_width,
            ),
        )
    if name.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight")):
        return (
            slice(
                plan.intermediate_start,
                plan.intermediate_start + plan.intermediate_size,
            ),
            slice(None),
        )
    if name.endswith(".mlp.down_proj.weight"):
        return (
            slice(None),
            slice(
                plan.intermediate_start,
                plan.intermediate_start + plan.intermediate_size,
            ),
        )
    return (slice(None),)


def _checkpoint_weight_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError("safetensors index 缺少 weight_map")
        weight_map: dict[str, Path] = {}
        for name, filename in raw_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise ValueError("safetensors weight_map 必须把参数名映射到文件名")
            shard_path = (model_dir / filename).resolve()
            if shard_path.parent != model_dir.resolve():
                raise ValueError("safetensors shard 必须直接位于模型目录")
            weight_map[name] = shard_path
        return weight_map

    single = (model_dir / "model.safetensors").resolve()
    if not single.is_file():
        raise FileNotFoundError("模型目录缺少 safetensors 或 sharded index")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("加载 Qwen3 TP 权重需要安装 safetensors") from exc
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single for name in handle.keys()}


def load_tp_qwen3_from_pretrained(
    model_dir: str | Path,
    distributed: DistributedContext,
    *,
    dtype: torch.dtype,
) -> TensorParallelQwen3ForCausalLM:
    """每个 rank 只读取自己负责的参数切片，不创建完整 Qwen3 权重副本。"""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("加载 Qwen3 TP 权重需要安装 safetensors") from exc

    directory = Path(model_dir).resolve()
    config = Qwen3Config.from_json(directory / "config.json")
    model = TensorParallelQwen3ForCausalLM(
        config,
        distributed,
        device="meta",
        dtype=dtype,
    )
    model.to_empty(device=distributed.runtime.device)
    model.reset_non_persistent_buffers(distributed.runtime.device)

    expected_shapes = expected_qwen3_parameter_shapes(config)
    targets = dict(model.named_parameters())
    if set(targets) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(targets))
        extra = sorted(set(targets) - set(expected_shapes))
        raise RuntimeError(
            f"TP 模型参数结构不一致：missing={missing[:4]}, extra={extra[:4]}"
        )

    weight_map = _checkpoint_weight_map(directory)
    if set(weight_map) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(weight_map))
        extra = sorted(set(weight_map) - set(expected_shapes))
        raise ValueError(
            f"checkpoint 参数不一致：missing={missing[:4]}, extra={extra[:4]}"
        )

    shard_to_names: dict[Path, list[str]] = {}
    for name, shard_path in weight_map.items():
        shard_to_names.setdefault(shard_path, []).append(name)

    loaded: set[str] = set()
    with torch.no_grad():
        for shard_path, names in sorted(
            shard_to_names.items(), key=lambda item: str(item[0])
        ):
            if not shard_path.is_file():
                raise FileNotFoundError(f"缺少 safetensors shard：{shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                mapped_keys = {
                    name for name, path in weight_map.items() if path == shard_path
                }
                if actual_keys != mapped_keys:
                    raise ValueError(f"shard 内容与 index 不一致：{shard_path.name}")
                for name in sorted(names):
                    tensor_slice = handle.get_slice(name)
                    source_shape = tuple(tensor_slice.get_shape())
                    if source_shape != expected_shapes[name]:
                        raise ValueError(
                            f"参数 {name} shape 不匹配：checkpoint={source_shape}, "
                            f"expected={expected_shapes[name]}"
                        )
                    source = tensor_slice[_parameter_slice(name, model.plan)]
                    target = targets[name]
                    if tuple(source.shape) != tuple(target.shape):
                        raise ValueError(
                            f"参数 {name} 分片 shape 不匹配：slice={tuple(source.shape)}, "
                            f"target={tuple(target.shape)}"
                        )
                    target.copy_(source)
                    loaded.add(name)
                    del source

    if loaded != set(targets):
        missing = sorted(set(targets) - loaded)
        raise RuntimeError(f"TP loader 缺少 {len(missing)} 个参数：{missing[:4]}")
    model.eval()
    return model


class CachedTensorParallelQwen3ModelRunner(CachedQwen3ModelRunner):
    """复用 v0.5 KV Cache 生命周期，但执行 TP 分片模型。"""

    implementation_name = "qwen3_tp_kv_cache"


class RecomputeTensorParallelQwen3ModelRunner(RecomputeQwen3ModelRunner):
    """TP full-forward oracle；主要用于 tiny correctness，不用于正式 32B 性能。"""

    implementation_name = "qwen3_tp_recompute"


class SlotCachedTensorParallelQwen3ModelRunner(SlotCachedQwen3ModelRunner):
    """复用相同 slot 生命周期，底层执行 TP Qwen3 模型。"""

    implementation_name = "qwen3_tp_slot_kv_cache"


class TensorParallelInferenceEngine(InferenceEngine):
    """只让 rank 0 选择 token，再广播给所有 rank，防止采样路径发生分叉。"""

    def __init__(
        self,
        runner: CachedTensorParallelQwen3ModelRunner
        | RecomputeTensorParallelQwen3ModelRunner,
        tokenizer: Qwen3Tokenizer,
        distributed: DistributedContext,
    ) -> None:
        super().__init__(runner, tokenizer, pad_token_id=tokenizer.pad_token_id)
        self.distributed = distributed

    def select_next_token(
        self,
        next_logits: torch.Tensor,
        config: GenerationConfig,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.distributed.is_primary:
            return super().select_next_token(next_logits, config, generator)
        if next_logits.ndim != 2:
            raise ValueError("next_logits 必须是 [B,V]")
        return torch.zeros(
            (next_logits.shape[0], 1),
            dtype=torch.long,
            device=next_logits.device,
        )

    def synchronize_next_ids(self, next_ids: torch.Tensor) -> torch.Tensor:
        return self.distributed.broadcast(next_ids, src=0)


def load_tp_qwen3_engine(
    model_dir: str | Path,
    distributed: DistributedContext,
    *,
    decode_mode: str = "kv_cache",
    use_chat_template: bool = False,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> TensorParallelInferenceEngine:
    """从本地 HF 目录为当前 rank 加载 Qwen3 参数分片和统一 tokenizer。"""

    runtime = distributed.runtime
    dtype = runtime.amp_dtype or torch.float32
    tokenizer = Qwen3Tokenizer(
        model_dir,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
    )
    model = load_tp_qwen3_from_pretrained(
        model_dir,
        distributed,
        dtype=dtype,
    )
    validate_qwen3_tokenizer_config(tokenizer, model.config)
    if decode_mode == "kv_cache":
        runner = CachedTensorParallelQwen3ModelRunner(model, runtime)
    elif decode_mode == "recompute":
        runner = RecomputeTensorParallelQwen3ModelRunner(model, runtime)
    else:
        raise ValueError("decode_mode 必须是 kv_cache 或 recompute")
    return TensorParallelInferenceEngine(runner, tokenizer, distributed)
