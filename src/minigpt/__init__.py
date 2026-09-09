"""MiniGPT 推理 Infra 学习包。

训练基础保留 tokenizer、data、model、optim；推理主线从 runtime、inference、benchmark
继续扩展。
"""

from .config import ExperimentConfig, ModelConfig, TrainConfig
from .distributed import DistributedContext, TensorParallelReplicaContext
from .inference import (
    CachedMiniGPTModelRunner,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
    MiniGPTModelRunner,
    RecomputeMiniGPTModelRunner,
    SlotCachedMiniGPTModelRunner,
)
from .model import MiniGPT, MiniGPTConfig, MiniGPTKVCache
from .qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3KVCache
from .qwen3_inference import (
    CachedQwen3ModelRunner,
    RecomputeQwen3ModelRunner,
    SlotCachedQwen3ModelRunner,
    load_qwen3_slot_runner,
)
from .qwen3_tp import (
    CachedTensorParallelQwen3ModelRunner,
    Qwen3TensorParallelPlan,
    TensorParallelInferenceEngine,
    TensorParallelQwen3ForCausalLM,
    SlotCachedTensorParallelQwen3ModelRunner,
    load_tp_qwen3_slot_runner,
)
from .runtime import RuntimeContext
from .replay import OfflineTraceReplayer, ReplaySummary
from .replica import (
    LeastLoadedRouter,
    MultiReplicaServing,
    ReplicaSnapshot,
    partition_workload_by_projected_load,
)
from .serving import (
    ContinuousBatchEngine,
    KVSlotAllocator,
    RequestSpec,
    RequestState,
    ServingRequest,
)
from .serving_benchmark import benchmark_trace_replay, summarize_samples
from .serving_acceptance import (
    load_layout_comparison,
    summarize_v07_acceptance,
)
from .serving_layout import (
    load_layout_manifest,
    load_serving_report,
    summarize_serving_layouts,
)
from .serving_telemetry import (
    NpuTelemetryTarget,
    load_telemetry,
    parse_npu_smi_common,
    parse_npu_smi_usages,
    parse_npu_target,
    summarize_telemetry,
)
from .tokenizer import CharTokenizer
from .workload import WorkloadPartition, WorkloadTrace, generate_workload

__all__ = [
    "CharTokenizer",
    "CachedMiniGPTModelRunner",
    "CachedTensorParallelQwen3ModelRunner",
    "ContinuousBatchEngine",
    "DistributedContext",
    "ExperimentConfig",
    "GenerationConfig",
    "GenerationResult",
    "InferenceEngine",
    "MiniGPT",
    "MiniGPTConfig",
    "MiniGPTKVCache",
    "MiniGPTModelRunner",
    "ModelConfig",
    "MultiReplicaServing",
    "LeastLoadedRouter",
    "OfflineTraceReplayer",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3KVCache",
    "Qwen3TensorParallelPlan",
    "KVSlotAllocator",
    "RequestSpec",
    "RequestState",
    "ReplaySummary",
    "ReplicaSnapshot",
    "ServingRequest",
    "CachedQwen3ModelRunner",
    "RecomputeQwen3ModelRunner",
    "RuntimeContext",
    "SlotCachedMiniGPTModelRunner",
    "SlotCachedQwen3ModelRunner",
    "SlotCachedTensorParallelQwen3ModelRunner",
    "load_qwen3_slot_runner",
    "load_tp_qwen3_slot_runner",
    "TensorParallelInferenceEngine",
    "TensorParallelReplicaContext",
    "TensorParallelQwen3ForCausalLM",
    "RecomputeMiniGPTModelRunner",
    "TrainConfig",
    "WorkloadPartition",
    "WorkloadTrace",
    "generate_workload",
    "partition_workload_by_projected_load",
    "benchmark_trace_replay",
    "summarize_samples",
    "load_layout_comparison",
    "summarize_v07_acceptance",
    "load_serving_report",
    "load_layout_manifest",
    "summarize_serving_layouts",
    "NpuTelemetryTarget",
    "load_telemetry",
    "parse_npu_smi_common",
    "parse_npu_smi_usages",
    "parse_npu_target",
    "summarize_telemetry",
]
