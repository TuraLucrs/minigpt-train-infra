"""Qwen3 Tensor Parallel 规划、分片加载、collective、KV Cache 与生成门禁。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Sequence

import torch
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.distributed import (  # noqa: E402
    DistributedContext,
    TensorParallelReplicaContext,
)
from minigpt.inference import GenerationConfig, InferenceEngine  # noqa: E402
from minigpt.qwen3 import (  # noqa: E402
    Qwen3Config,
    Qwen3ForCausalLM,
    load_qwen3_from_pretrained,
)
from minigpt.qwen3_inference import CachedQwen3ModelRunner  # noqa: E402
from minigpt.qwen3_tp import (  # noqa: E402
    CachedTensorParallelQwen3ModelRunner,
    Qwen3TensorParallelPlan,
    SlotCachedTensorParallelQwen3ModelRunner,
    TensorParallelQwen3ForCausalLM,
    TensorParallelInferenceEngine,
    load_tp_qwen3_from_pretrained,
)
from minigpt.serving import ContinuousBatchEngine, RequestSpec, RequestState  # noqa: E402


def tiny_config_dict() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 40,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "float32",
        "initializer_range": 0.02,
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
    }


class IntegerTokenizer:
    """仅供 TP 生成编排测试使用；空格分隔 token id。"""

    vocab_size = 40
    pad_token_id = 0

    @staticmethod
    def encode(text: str) -> list[int]:
        return [int(value) for value in text.split()]

    @staticmethod
    def decode(token_ids: Sequence[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-4):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{label} 最大绝对差为 {max_diff}")


class _RecordingCollectives:
    """记录模型实际发送的张量，避免只检查 scheduler 自报的路径标签。"""

    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.events: list[tuple[str, tuple[int, ...], torch.dtype]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        self.events.append(("all_gather", tuple(tensor.shape), tensor.dtype))
        return self.delegate.all_gather_last_dim(tensor)  # type: ignore[attr-defined]

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        assert src == 0
        self.events.append(("broadcast", tuple(tensor.shape), tensor.dtype))
        return self.delegate.broadcast(tensor, src=src)  # type: ignore[attr-defined]


def save_checkpoint(directory: Path) -> Qwen3ForCausalLM:
    from safetensors.torch import save_file

    torch.manual_seed(2026)
    raw = tiny_config_dict()
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(raw)).eval()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )
    state = {
        name: tensor.detach().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, directory / "model.safetensors")
    return model


def check_plan_boundaries() -> None:
    tiny = Qwen3Config.from_dict(tiny_config_dict())
    plans = [Qwen3TensorParallelPlan.create(tiny, rank, 4) for rank in range(4)]
    assert [plan.query_head_start for plan in plans] == [0, 1, 2, 3]
    assert [plan.kv_head_start for plan in plans] == [0, 0, 1, 1]
    assert all(plan.query_heads == 1 for plan in plans)
    assert all(plan.kv_heads == 1 for plan in plans)
    assert [plan.intermediate_start for plan in plans] == [0, 12, 24, 36]
    assert [plan.vocab_start for plan in plans] == [0, 10, 20, 30]

    official = Qwen3Config.from_json(
        PROJECT_ROOT / "configs" / "qwen3_32b_official.json"
    )
    for world_size in (1, 2, 4, 8, 16):
        world_plans = [
            Qwen3TensorParallelPlan.create(official, rank, world_size)
            for rank in range(world_size)
        ]
        assert sum(plan.query_heads for plan in world_plans) == 64
        assert (
            sum(plan.intermediate_size for plan in world_plans)
            == official.intermediate_size
        )
        assert sum(plan.vocab_size for plan in world_plans) == official.vocab_size
    tp16 = [Qwen3TensorParallelPlan.create(official, rank, 16) for rank in range(16)]
    assert [plan.kv_head_start for plan in tp16] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
    ]

    for invalid_world_size in (3, 5):
        try:
            Qwen3TensorParallelPlan.create(official, 0, invalid_world_size)
        except ValueError:
            pass
        else:
            raise AssertionError("非法 TP world_size 必须被拒绝")


def check_distributed_boundaries() -> None:
    single = DistributedContext.create(
        "cpu",
        "fp32",
        rank=0,
        local_rank=0,
        world_size=1,
    )
    assert not single.is_distributed
    assert single.is_primary
    assert not single.owns_process_group
    value = torch.tensor([3.0])
    assert single.all_reduce_sum(value) is value
    assert single.all_gather_last_dim(value) is value
    assert single.all_gather_floats([1.5]) == [[1.5]]
    single_replica = TensorParallelReplicaContext.create(single, tp_size=1)
    assert single_replica.rank == 0
    assert single_replica.replica_count == 1
    assert single_replica.group_ranks == (0,)
    assert single_replica.all_gather_last_dim(value) is value
    try:
        single.broadcast(torch.tensor([1]), src=1)
    except ValueError:
        pass
    else:
        raise AssertionError("越界 broadcast src 必须被拒绝")

    try:
        DistributedContext.create("cpu", "fp32", backend="nccl")
    except ValueError:
        pass
    else:
        raise AssertionError("CPU TP 不能接受 NCCL backend")

    if not torch.cuda.is_available():
        try:
            DistributedContext.create("cuda", "fp32", warn=lambda _message: None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("显式 CUDA 分布式任务不能静默回退 CPU")


def _tp_worker(
    rank: int,
    world_size: int,
    model_dir: str,
    rendezvous_path: str,
) -> None:
    torch.set_num_threads(1)
    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            "cpu",
            "fp32",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            init_method=Path(rendezvous_path).resolve().as_uri(),
            timeout_seconds=60,
        )
        assert distributed.backend == "gloo"
        assert distributed.is_distributed
        assert distributed.metadata()["process_group_initialized"] is True

        reduced = torch.tensor([float(rank + 1)])
        distributed.all_reduce_sum(reduced)
        assert reduced.item() == 3.0
        gathered = distributed.all_gather_last_dim(torch.tensor([[float(rank)]]))
        assert gathered.tolist() == [[0.0, 1.0]]
        measurements = distributed.all_gather_floats([rank + 0.25])
        assert measurements == [[0.25], [1.25]]
        broadcast = torch.tensor([17 if rank == 0 else -1], dtype=torch.long)
        distributed.broadcast(broadcast)
        assert broadcast.item() == 17

        replica = TensorParallelReplicaContext.create(distributed, tp_size=1)
        assert replica.rank == 0
        assert replica.replica_count == world_size
        assert replica.replica_index == rank
        assert replica.group_ranks == (rank,)
        local = torch.tensor([float(rank)])
        assert replica.all_reduce_sum(local).item() == float(rank)
        replica.global_barrier()

        _check_distributed_argmax(distributed)
        _check_distributed_argmax(replica)
        _check_tp_model(distributed, model_dir)
        for mismatch in ("request", "greedy_token_path", "capability"):
            _check_scheduler_mismatch_rank(distributed, mismatch)
    finally:
        if distributed is not None:
            distributed.close()


def _replica_group_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
) -> None:
    torch.set_num_threads(1)
    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            "cpu",
            "fp32",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            init_method=Path(rendezvous_path).resolve().as_uri(),
            timeout_seconds=60,
        )
        replica = TensorParallelReplicaContext.create(distributed, tp_size=2)
        expected_replica = rank // 2
        expected_group = (expected_replica * 2, expected_replica * 2 + 1)
        assert replica.replica_index == expected_replica
        assert replica.replica_count == 2
        assert replica.rank == rank % 2
        assert replica.group_ranks == expected_group

        reduced = torch.tensor([float(rank + 1)])
        replica.all_reduce_sum(reduced)
        assert reduced.item() == (3.0 if expected_replica == 0 else 7.0)
        gathered = replica.all_gather_last_dim(torch.tensor([[float(rank)]]))
        assert gathered.tolist() == [[float(value) for value in expected_group]]
        broadcast = torch.tensor(
            [100 + expected_replica if replica.rank == 0 else -1],
            dtype=torch.long,
        )
        replica.broadcast(broadcast, src=0)
        assert broadcast.item() == 100 + expected_replica
        measurements = replica.all_gather_floats([rank + 0.5])
        assert measurements == [[value + 0.5] for value in expected_group]

        _check_distributed_argmax(replica)
        for mismatch in ("request", "greedy_token_path", "capability"):
            _check_scheduler_mismatch_rank(replica, mismatch)
        replica.global_barrier()
        global_value = torch.tensor([1.0])
        distributed.all_reduce_sum(global_value)
        assert global_value.item() == float(world_size)
    finally:
        if distributed is not None:
            distributed.close()


def _check_distributed_argmax(distributed: object) -> None:
    raw = tiny_config_dict()
    raw["vocab_size"] = 131_072
    recorded = _RecordingCollectives(distributed)
    # 只测选 token 原语，meta 权重避免为大 token id 用例分配完整 embedding。
    model = TensorParallelQwen3ForCausalLM(
        Qwen3Config.from_dict(raw),
        recorded,  # type: ignore[arg-type]
        device="meta",
    )
    vocab_size = model.config.vocab_size
    replica_winner = vocab_size - 1 - int(getattr(distributed, "replica_index", 0))
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        logits = torch.full((9, vocab_size), -10.0, dtype=dtype)
        logits[0] = -7.0
        logits[1, [1, 3]] = 4.0
        logits[2, 3] = 7.0
        for rank in range(1, model.plan.world_size):
            logits[2, rank * model.plan.vocab_size] = 7.0
        logits[3, replica_winner] = 3.0
        logits[4] = float("-inf")
        logits[5, [2, vocab_size - 1]] = float("inf")
        logits[6, 0] = -0.0
        logits[6, -1] = 0.0
        logits[7, [4, vocab_size - 1]] = float("nan")
        logits[8, 0] = 1.0
        logits[8, -1] = 1.0 + torch.finfo(dtype).eps
        local = logits.narrow(-1, model.plan.vocab_start, model.plan.vocab_size)
        before = len(recorded.events)
        with torch.inference_mode():
            actual = model.distributed_greedy_argmax(local)
        expected = torch.argmax(logits, dim=-1, keepdim=True)
        assert torch.equal(actual, expected), (dtype, model.plan.rank, actual, expected)
        assert actual[:, 0].tolist() == [
            0, 1, 3, replica_winner, 0, 2, 0, 4, vocab_size - 1
        ]
        assert recorded.events[before:] == [
            ("all_gather", (9, 2), torch.float32),
            ("broadcast", (9, 1), torch.long),
        ]

    # 1 + 2**-30 在 FP64 大于 1，但打包成 FP32 后会与较小 token 的 1 并列。
    # API 必须明确拒绝这种会改变 argmax 的输入，不能静默宣称等价。
    local_fp64 = torch.full((1, model.plan.vocab_size), -10.0, dtype=torch.float64)
    local_fp64[0, 0] = 1.0 + model.plan.rank * 2**-30
    before = len(recorded.events)
    try:
        model.distributed_greedy_argmax(local_fp64)
    except ValueError as exc:
        assert "FP16、BF16、FP32" in str(exc)
    else:
        raise AssertionError("distributed argmax 必须拒绝会丢失分数精度的 FP64 输入")
    assert len(recorded.events) == before


def _check_tp_model(distributed: object, model_dir: str) -> None:
    rank = int(distributed.rank)  # type: ignore[attr-defined]
    runtime = distributed.runtime  # type: ignore[attr-defined]

    reference = load_qwen3_from_pretrained(
        model_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model_collectives = _RecordingCollectives(distributed)
    tp_model = load_tp_qwen3_from_pretrained(
        model_dir,
        model_collectives,  # type: ignore[arg-type]
        dtype=torch.float32,
    )
    assert tp_model.plan.rank == rank
    assert tp_model.plan.kv_heads == 1
    assert tp_model.plan.query_heads == 4 // distributed.world_size  # type: ignore[attr-defined]
    assert sum(parameter.numel() for parameter in tp_model.parameters()) < sum(
        parameter.numel() for parameter in reference.parameters()
    )

    input_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]])
    attention_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0]],
        dtype=torch.bool,
    )
    with torch.inference_mode():
        expected_full = reference(input_ids, attention_mask)
        actual_full = tp_model(input_ids, attention_mask)
    assert_close(actual_full, expected_full, "TP=2 full logits")

    cache = tp_model.allocate_kv_cache(
        2,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert cache.layers[0].key.shape == (2, 1, 8, 8)
    last_positions = torch.tensor([3, 1])
    with torch.inference_mode():
        actual_prefill = tp_model.prefill_with_cache(
            input_ids,
            attention_mask,
            cache,
            logit_positions=last_positions,
        )
        expected_prefill = expected_full[
            torch.arange(input_ids.shape[0]),
            last_positions,
        ]
    assert_close(actual_prefill, expected_prefill, "TP=2 cached Prefill")

    next_ids = torch.argmax(expected_prefill, dim=-1, keepdim=True)
    distributed_cache = tp_model.allocate_kv_cache(
        2,
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        local_prefill = tp_model.prefill_with_cache(
            input_ids,
            attention_mask,
            distributed_cache,
            logit_positions=last_positions,
            gather_logits=False,
        )
        distributed_next_ids = tp_model.distributed_greedy_argmax(local_prefill)
    assert torch.equal(distributed_next_ids, next_ids)

    tie_logits = torch.full(
        (2, tp_model.plan.vocab_size),
        -10.0,
        dtype=torch.float32,
    )
    tie_logits[0, 3 if rank == 0 else 0] = 7.0
    tie_logits[1, 1] = float(rank)
    with torch.inference_mode():
        tie_ids = tp_model.distributed_greedy_argmax(tie_logits)
    assert tie_ids[0, 0].item() == 3
    assert tie_ids[1, 0].item() == (
        (distributed.world_size - 1) * tp_model.plan.vocab_size + 1
    )

    active_mask = torch.tensor([True, True])
    with torch.inference_mode():
        actual_decode = tp_model.decode_with_cache(next_ids, active_mask, cache)[:, 0]
        local_decode = tp_model.decode_with_cache(
            next_ids,
            active_mask,
            distributed_cache,
            gather_logits=False,
        )[:, 0]
        distributed_decode_ids = tp_model.distributed_greedy_argmax(local_decode)
        row0 = torch.cat((input_ids[0, :4], next_ids[0])).unsqueeze(0)
        row1 = torch.cat((input_ids[1, :2], next_ids[1])).unsqueeze(0)
        expected_row0 = reference(row0)[:, -1]
        expected_row1 = reference(row1)[:, -1]
    assert_close(actual_decode[0:1], expected_row0, "TP=2 cached Decode row 0")
    assert_close(actual_decode[1:2], expected_row1, "TP=2 cached Decode row 1")
    assert torch.equal(
        distributed_decode_ids,
        torch.argmax(actual_decode, dim=-1, keepdim=True),
    )

    tokenizer = IntegerTokenizer()
    reference_engine = InferenceEngine(
        CachedQwen3ModelRunner(reference, runtime),
        tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )
    tp_engine = TensorParallelInferenceEngine(
        CachedTensorParallelQwen3ModelRunner(tp_model, runtime),
        tokenizer,  # type: ignore[arg-type]
        distributed,  # type: ignore[arg-type]
    )
    config = GenerationConfig(max_new_tokens=4, strategy="greedy")
    prompts = ["4 5 6", "7 8"]
    expected_generation = reference_engine.generate_batch(prompts, config)
    actual_generation = tp_engine.generate_batch(prompts, config)
    assert [item.generated_ids for item in actual_generation] == [
        item.generated_ids for item in expected_generation
    ]
    sample_config = GenerationConfig(
        max_new_tokens=3,
        strategy="sample",
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        seed=7,
    )
    expected_sample = reference_engine.generate_batch(prompts, sample_config)
    actual_sample = tp_engine.generate_batch(prompts, sample_config)
    assert [item.generated_ids for item in actual_sample] == [
        item.generated_ids for item in expected_sample
    ]
    assert [item.generated_ids for item in actual_sample] != [
        item.generated_ids[:3] for item in expected_generation
    ]

    def run_slot_case(
        path: str,
        specs: Sequence[RequestSpec],
    ) -> tuple[ContinuousBatchEngine, list[tuple[str, tuple[int, ...], torch.dtype]]]:
        engine = ContinuousBatchEngine(
            SlotCachedTensorParallelQwen3ModelRunner(
                tp_model,
                runtime,
                max_slots=2,
                max_seq_len=12,
                greedy_token_path=path,
            ),
            tokenizer,
            distributed=distributed,  # type: ignore[arg-type]
        )
        model_collectives.events.clear()
        for spec in specs:
            engine.submit(spec)
        if rank != 0:
            # follower 的本地 deque 故意不同，batch 仍须服从 rank 0 的 plan。
            engine._waiting.rotate(1)
        first_step = engine.step()
        assert first_step["prefill_batch_size"] == 2
        engine.run_until_idle()
        assert engine.allocator.used == 0
        assert engine.runner.cache_lengths([0, 1]) == [0, 0]
        return engine, list(model_collectives.events)

    fixed_specs = [
        RequestSpec("fixed-a", "4 5 6", GenerationConfig(max_new_tokens=4)),
        RequestSpec("fixed-b", "7 8", GenerationConfig(max_new_tokens=4)),
    ]
    fixed_baseline, baseline_events = run_slot_case("full_gather", fixed_specs)
    fixed_candidate, candidate_events = run_slot_case("distributed_argmax", fixed_specs)
    for spec, expected in zip(fixed_specs, expected_generation):
        baseline = fixed_baseline.requests[spec.request_id]
        candidate = fixed_candidate.requests[spec.request_id]
        assert baseline.state == candidate.state == RequestState.FINISHED
        assert baseline.generated_ids == candidate.generated_ids == expected.generated_ids
    assert fixed_baseline.report()["token_selection"]["actual_rows_by_path"] == {
        "full_gather": 8
    }
    assert fixed_candidate.report()["token_selection"]["actual_rows_by_path"] == {
        "distributed_argmax": 8
    }
    assert baseline_events == [
        ("all_gather", (2, tp_model.plan.vocab_size), torch.float32)
    ] + [
        ("all_gather", (2, 1, tp_model.plan.vocab_size), torch.float32)
    ] * 3, baseline_events
    assert candidate_events == [
        ("all_gather", (2, 2), torch.float32),
        ("broadcast", (2, 1), torch.long),
    ] * 4, candidate_events

    dynamic_specs = [
        RequestSpec("dynamic-a", "4 5 6", GenerationConfig(max_new_tokens=5)),
        RequestSpec(
            "dynamic-b",
            "7 8",
            GenerationConfig(
                max_new_tokens=2,
                strategy="sample",
                temperature=0.8,
                top_k=8,
                top_p=0.9,
                seed=17,
            ),
        ),
        RequestSpec("dynamic-c", "9 10 11 12", GenerationConfig(max_new_tokens=3)),
    ]
    expected_dynamic = {
        spec.request_id: reference_engine.generate(spec.prompt, spec.config).generated_ids
        for spec in dynamic_specs
    }
    dynamic_baseline, baseline_events = run_slot_case("full_gather", dynamic_specs)
    slot_engine, candidate_events = run_slot_case("distributed_argmax", dynamic_specs)
    assert slot_engine.report()["token_selection"]["actual_rows_by_path"] == {
        "full_gather_sampling_fallback": 4,
        "distributed_argmax": 6,
    }
    assert dynamic_baseline.report()["token_selection"]["actual_rows_by_path"] == {
        "full_gather": 10
    }
    assert slot_engine.steps[0]["prefill_token_selection_path"] == "full_gather_sampling_fallback"
    assert slot_engine.steps[1]["decode_token_selection_path"] == "full_gather_sampling_fallback"
    assert slot_engine.steps[1]["prefill_token_selection_path"] == "distributed_argmax"
    assert all(
        step["decode_token_selection_path"] == "distributed_argmax"
        for step in slot_engine.steps[2:]
    )
    phase_batches = [
        ("prefill", 2, False),
        ("decode", 2, False),
        ("prefill", 1, True),
        ("decode", 2, True),
        ("decode", 2, True),
        ("decode", 1, True),
    ]
    expected_events = []
    for phase, rows, distributed_greedy in phase_batches:
        width = 2 if distributed_greedy else tp_model.plan.vocab_size
        shape = (rows, width) if distributed_greedy or phase == "prefill" else (rows, 1, width)
        expected_events.append(("all_gather", shape, torch.float32))
        if distributed_greedy:
            expected_events.append(("broadcast", (rows, 1), torch.long))
    assert candidate_events == expected_events, (candidate_events, expected_events)
    assert baseline_events == [
        (
            "all_gather",
            (rows, tp_model.plan.vocab_size)
            if phase == "prefill"
            else (rows, 1, tp_model.plan.vocab_size),
            torch.float32,
        )
        for phase, rows, _distributed_greedy in phase_batches
    ], baseline_events
    for request_id, expected_ids in expected_dynamic.items():
        request = slot_engine.requests[request_id]
        baseline = dynamic_baseline.requests[request_id]
        assert request.state == baseline.state == RequestState.FINISHED
        assert request.generated_ids == expected_ids
        assert request.generated_ids == baseline.generated_ids
        if request.spec.config.strategy == "sample" and distributed.is_primary:
            assert request.generator is not None and baseline.generator is not None
            assert torch.equal(request.generator.get_state(), baseline.generator.get_state())
        else:
            assert request.generator is None and baseline.generator is None
    greedy_b = reference_engine.generate("7 8", GenerationConfig(max_new_tokens=2))
    assert slot_engine.requests["dynamic-b"].generated_ids != greedy_b.generated_ids


class _ThreadCollectives:
    """无 socket 的确定性 collective，只用于当前 CI 的多 rank 数学验证。"""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.condition = threading.Condition()
        self.calls = {
            "all_reduce": {rank: 0 for rank in range(world_size)},
            "all_gather": {rank: 0 for rank in range(world_size)},
            "broadcast": {rank: 0 for rank in range(world_size)},
        }
        self.slots: dict[tuple[str, int], dict[int, torch.Tensor]] = {}
        self.results: dict[tuple[str, int], torch.Tensor] = {}
        self.errors: dict[tuple[str, int], str] = {}
        self.readers: dict[tuple[str, int], int] = {}

    def run(
        self,
        operation: str,
        rank: int,
        tensor: torch.Tensor,
        *,
        src: int = 0,
    ) -> torch.Tensor:
        call_index = self.calls[operation][rank]
        self.calls[operation][rank] += 1
        key = (operation, call_index)
        with self.condition:
            slot = self.slots.setdefault(key, {})
            slot[rank] = tensor.detach().clone()
            if len(slot) == self.world_size:
                ordered = [slot[index] for index in range(self.world_size)]
                if any(
                    value.shape != ordered[0].shape or value.dtype != ordered[0].dtype
                    for value in ordered[1:]
                ):
                    self.errors[key] = f"collective tensor shape/dtype 不一致：{key}"
                elif operation == "all_reduce":
                    result = torch.stack(ordered).sum(dim=0)
                    self.results[key] = result
                elif operation == "all_gather":
                    result = torch.cat(ordered, dim=-1)
                    self.results[key] = result
                else:
                    result = slot[src]
                    self.results[key] = result
                self.readers[key] = 0
                self.condition.notify_all()
            ready = self.condition.wait_for(
                lambda: key in self.results or key in self.errors,
                timeout=30,
            )
            if not ready:
                raise TimeoutError(f"线程 collective 超时：{key}")
            error = self.errors.get(key)
            result = self.results[key].clone() if error is None else None
            self.readers[key] += 1
            if self.readers[key] == self.world_size:
                del self.slots[key]
                self.results.pop(key, None)
                self.errors.pop(key, None)
                del self.readers[key]
            if error is not None:
                raise RuntimeError(error)
            assert result is not None
            return result


class _ThreadDistributedContext:
    def __init__(
        self,
        rank: int,
        world_size: int,
        runtime: object,
        collectives: _ThreadCollectives,
    ) -> None:
        self.rank = rank
        self.local_rank = rank
        self.world_size = world_size
        self.runtime = runtime
        self.collectives = collectives

    @property
    def is_distributed(self) -> bool:
        return True

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        result = self.collectives.run("all_reduce", self.rank, tensor)
        tensor.copy_(result)
        return tensor

    def all_gather_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.collectives.run("all_gather", self.rank, tensor)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        result = self.collectives.run("broadcast", self.rank, tensor, src=src)
        tensor.copy_(result)
        return tensor


def check_threaded_tp(model_dir: str, world_size: int) -> None:
    from minigpt.runtime import RuntimeContext

    runtime = RuntimeContext.create("cpu", "fp32")
    collectives = _ThreadCollectives(world_size)
    contexts = [
        _ThreadDistributedContext(rank, world_size, runtime, collectives)
        for rank in range(world_size)
    ]
    with ThreadPoolExecutor(max_workers=world_size) as executor:
        futures = [
            executor.submit(_check_tp_model, context, model_dir) for context in contexts
        ]
        for future in futures:
            future.result()


def check_threaded_argmax(world_size: int) -> None:
    from minigpt.runtime import RuntimeContext

    runtime = RuntimeContext.create("cpu", "fp32")
    collectives = _ThreadCollectives(world_size)
    contexts = [
        _ThreadDistributedContext(rank, world_size, runtime, collectives)
        for rank in range(world_size)
    ]
    with ThreadPoolExecutor(max_workers=world_size) as executor:
        futures = [executor.submit(_check_distributed_argmax, context) for context in contexts]
        for future in futures:
            future.result()


def check_token_selection_metadata() -> None:
    from minigpt.runtime import RuntimeContext

    runtime = RuntimeContext.create("cpu", "fp32")
    context = _ThreadDistributedContext(0, 8, runtime, _ThreadCollectives(8))
    config = Qwen3Config.from_json(PROJECT_ROOT / "configs" / "qwen3_32b_official.json")
    model = TensorParallelQwen3ForCausalLM(
        config,
        context,  # type: ignore[arg-type]
        device="meta",
        dtype=torch.bfloat16,
    )
    runner = SlotCachedTensorParallelQwen3ModelRunner(
        model,
        runtime,
        max_slots=1,
        max_seq_len=1,
        greedy_token_path="distributed_argmax",
    )
    expected = {
        "measurement_type": "estimate",
        "payload_scope": "collective_input_per_rank_per_row",
        "configured_greedy_path": "distributed_argmax",
        "global_vocab_size": 151936,
        "local_vocab_size": 18992,
        "tp_size": 8,
        "logit_element_size_bytes": 2,
        "full_gather_input_bytes_per_rank_per_row": 37984,
        "distributed_argmax_input_bytes_per_rank_per_row": 8,
        "collective_input_reduction": 4748.0,
    }
    assert runner.token_selection_metadata() == expected
    # FP32 lm_head 在 BF16 autocast 下产生 BF16 logits，payload 不能按权重计为 4 字节。
    model.lm_head.to(dtype=torch.float32)
    runner.runtime = replace(runtime, precision="bf16", amp_dtype=torch.bfloat16)
    assert runner.token_selection_metadata() == expected
    try:
        runner.greedy_token_path = "full_gather"  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("runner 的选 token 通信协议不能在启动后被修改")


def _check_scheduler_mismatch_rank(distributed: object, mismatch: str) -> str:
    runtime = distributed.runtime  # type: ignore[attr-defined]
    rank = int(distributed.rank)  # type: ignore[attr-defined]

    class NeverCalledRunner:
        implementation_name = "never_called"
        max_slots = 1
        max_seq_len = 8
        greedy_token_path = "full_gather"

        def __init__(self) -> None:
            self.runtime = runtime
            self.model_calls = 0

        @staticmethod
        def validate_request(prompt_length: int, max_new_tokens: int) -> None:
            if prompt_length + max_new_tokens - 1 > 8:
                raise ValueError("request 太长")

        def prefill_slots(self, *_args: object) -> torch.Tensor:
            self.model_calls += 1
            raise AssertionError("control plan 不一致时不能进入模型")

        def decode_slots(self, *_args: object) -> torch.Tensor:
            self.model_calls += 1
            raise AssertionError("control plan 不一致时不能进入模型")

        def prefill_slots_local_logits(self, *_args: object) -> torch.Tensor:
            return self.prefill_slots()

        def decode_slots_local_logits(self, *_args: object) -> torch.Tensor:
            return self.decode_slots()

        def select_greedy_tokens(self, *_args: object) -> torch.Tensor:
            self.model_calls += 1
            raise AssertionError("control plan 不一致时不能进入选 token collective")

        @staticmethod
        def release_slots(_slot_ids: Sequence[int]) -> None:
            return None

        @staticmethod
        def cache_lengths(slot_ids: Sequence[int]) -> list[int]:
            return [0 for _slot_id in slot_ids]

    runner = NeverCalledRunner()
    if mismatch == "greedy_token_path":
        if rank == 0:
            runner.greedy_token_path = "distributed_argmax"
    elif mismatch == "capability":
        runner.greedy_token_path = "distributed_argmax"
        if rank == 1:
            runner.select_greedy_tokens = None  # type: ignore[assignment]
    elif mismatch != "request":
        raise ValueError(f"未知 mismatch case：{mismatch}")

    engine = ContinuousBatchEngine(
        runner,
        IntegerTokenizer(),
        distributed=distributed,  # type: ignore[arg-type]
    )
    prompt = "6 7" if mismatch == "request" and rank == 1 else "4 5"
    engine.submit(RequestSpec("same-id", prompt, GenerationConfig(max_new_tokens=2)))
    try:
        engine.step()
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError(f"不同 rank 的 {mismatch} 不一致时必须共同失败")
    assert "control plan" in error
    if mismatch != "request":
        assert "token_selection" in error
    assert runner.model_calls == 0
    return error


def check_scheduler_mismatch_fails_all_ranks(mismatch: str = "request") -> None:
    from minigpt.runtime import RuntimeContext

    runtime = RuntimeContext.create("cpu", "fp32")
    collectives = _ThreadCollectives(2)
    contexts = [
        _ThreadDistributedContext(rank, 2, runtime, collectives)
        for rank in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_check_scheduler_mismatch_rank, context, mismatch)
            for context in contexts
        ]
        for future in futures:
            future.result()


def main() -> None:
    torch.set_num_threads(1)
    check_plan_boundaries()
    check_distributed_boundaries()
    for world_size in (1, 2, 4):
        check_threaded_argmax(world_size)
    check_token_selection_metadata()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model_dir = root / "model"
        save_checkpoint(model_dir)
        check_threaded_tp(str(model_dir), 2)
        # TP=4 走 world_size > KV heads 的复制分支；确保 replicated KV 与 query
        # group 的对应关系仍能恢复完整模型结果。
        check_threaded_tp(str(model_dir), 4)
        check_scheduler_mismatch_fails_all_ranks()
        check_scheduler_mismatch_fails_all_ranks("greedy_token_path")
        check_scheduler_mismatch_fails_all_ranks("capability")
        if os.environ.get("MINIGPT_RUN_GLOO_TESTS") == "1":
            rendezvous_path = root / "gloo-rendezvous"
            mp.spawn(
                _tp_worker,
                args=(2, str(model_dir), str(rendezvous_path)),
                nprocs=2,
                join=True,
            )
            print("v0.8 Gloo argmax, fixed-batch and sampling fallback tests passed.")
            replica_rendezvous = root / "gloo-replica-rendezvous"
            mp.spawn(
                _replica_group_worker,
                args=(4, str(replica_rendezvous)),
                nprocs=4,
                join=True,
            )
            print("v0.8 Gloo TP subgroup argmax tests passed.")
        else:
            print("Gloo test skipped; set MINIGPT_RUN_GLOO_TESTS=1 to enable it.")
    print("Qwen3 TP and v0.8 distributed token selection tests passed.")


if __name__ == "__main__":
    main()
