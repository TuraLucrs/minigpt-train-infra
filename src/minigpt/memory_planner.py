"""Qwen3 dense 推理的静态 HBM 估算；结果用于运行前规划，不冒充实测显存。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .qwen3 import Qwen3Config, count_qwen3_parameters


MIB = 1024 * 1024


@dataclass(frozen=True)
class Qwen3MemoryEstimate:
    tensor_parallel_size: int
    batch_size: int
    max_sequence_length: int
    dtype_bytes: int
    parameter_count: int
    local_parameter_count: int
    query_heads_per_rank: int
    kv_heads_per_rank: int
    ideal_balanced_weight_memory_mb: float
    replicated_weight_overhead_mb: float
    weight_memory_mb: float
    kv_cache_memory_mb: float
    workspace_memory_mb: float
    runtime_reserve_mb: float
    estimated_total_mb: float
    device_memory_mb: float
    estimated_utilization: float
    fits: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def estimate_qwen3_memory(
    config: Qwen3Config,
    *,
    tensor_parallel_size: int,
    batch_size: int,
    max_sequence_length: int,
    device_memory_mb: float,
    dtype_bytes: int = 2,
    workspace_memory_mb: float = 1024.0,
    runtime_reserve_mb: float = 2048.0,
) -> Qwen3MemoryEstimate:
    """估算每个 TP rank 的权重、KV Cache、workspace 和保留空间。"""

    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size 必须大于 0")
    if config.num_attention_heads % tensor_parallel_size != 0:
        raise ValueError("当前规划要求 num_attention_heads 能被 TP 整除")
    if batch_size <= 0 or max_sequence_length <= 0:
        raise ValueError("batch_size/max_sequence_length 必须大于 0")
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("max_sequence_length 超过模型上下文上限")
    if dtype_bytes <= 0 or device_memory_mb <= 0:
        raise ValueError("dtype_bytes/device_memory_mb 必须大于 0")

    parameter_count = count_qwen3_parameters(config)
    query_heads_per_rank = config.num_attention_heads // tensor_parallel_size
    # 当 TP 大于 KV head 数时，每个 rank 至少持有一个 KV head，K/V projection 与 cache
    # 会在部分 rank 间复制。词表/FFN 使用向上取整表示最重 rank 的保守容量。
    kv_heads_per_rank = max(1, math.ceil(config.num_key_value_heads / tensor_parallel_size))
    query_width_per_rank = query_heads_per_rank * config.head_dim
    kv_width_per_rank = kv_heads_per_rank * config.head_dim
    intermediate_per_rank = math.ceil(config.intermediate_size / tensor_parallel_size)
    vocab_per_rank = math.ceil(config.vocab_size / tensor_parallel_size)

    local_attention_parameters = (
        config.hidden_size * query_width_per_rank
        + 2 * config.hidden_size * kv_width_per_rank
        + query_width_per_rank * config.hidden_size
        + 2 * config.head_dim
    )
    if config.attention_bias:
        local_attention_parameters += (
            query_width_per_rank
            + 2 * kv_width_per_rank
            + config.hidden_size
        )
    local_mlp_parameters = 3 * config.hidden_size * intermediate_per_rank
    replicated_layer_norms = 2 * config.hidden_size
    local_layer_parameters = config.num_hidden_layers * (
        local_attention_parameters + local_mlp_parameters + replicated_layer_norms
    )
    local_embeddings = vocab_per_rank * config.hidden_size
    local_output = 0 if config.tie_word_embeddings else local_embeddings
    local_parameter_count = (
        local_embeddings + local_layer_parameters + config.hidden_size + local_output
    )
    ideal_weight_memory_mb = (
        parameter_count * dtype_bytes / tensor_parallel_size / MIB
    )
    weight_memory_mb = local_parameter_count * dtype_bytes / MIB
    replicated_weight_overhead_mb = weight_memory_mb - ideal_weight_memory_mb

    kv_cache_elements = (
        2
        * config.num_hidden_layers
        * kv_heads_per_rank
        * config.head_dim
        * batch_size
        * max_sequence_length
    )
    kv_cache_memory_mb = kv_cache_elements * dtype_bytes / MIB
    estimated_total_mb = (
        weight_memory_mb
        + kv_cache_memory_mb
        + workspace_memory_mb
        + runtime_reserve_mb
    )
    utilization = estimated_total_mb / device_memory_mb
    return Qwen3MemoryEstimate(
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        dtype_bytes=dtype_bytes,
        parameter_count=parameter_count,
        local_parameter_count=local_parameter_count,
        query_heads_per_rank=query_heads_per_rank,
        kv_heads_per_rank=kv_heads_per_rank,
        ideal_balanced_weight_memory_mb=ideal_weight_memory_mb,
        replicated_weight_overhead_mb=replicated_weight_overhead_mb,
        weight_memory_mb=weight_memory_mb,
        kv_cache_memory_mb=kv_cache_memory_mb,
        workspace_memory_mb=workspace_memory_mb,
        runtime_reserve_mb=runtime_reserve_mb,
        estimated_total_mb=estimated_total_mb,
        device_memory_mb=device_memory_mb,
        estimated_utilization=utilization,
        fits=estimated_total_mb <= device_memory_mb,
    )
