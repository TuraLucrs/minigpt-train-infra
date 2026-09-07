"""Qwen3 Tensor Parallel 规划、分片加载、collective、KV Cache 与生成门禁。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

from minigpt.distributed import DistributedContext  # noqa: E402
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
    TensorParallelInferenceEngine,
    load_tp_qwen3_from_pretrained,
)


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
            init_method=f"file://{rendezvous_path}",
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

        _check_tp_model(distributed, model_dir)
    finally:
        if distributed is not None:
            distributed.close()


def _check_tp_model(distributed: object, model_dir: str) -> None:
    rank = int(distributed.rank)  # type: ignore[attr-defined]
    runtime = distributed.runtime  # type: ignore[attr-defined]

    reference = load_qwen3_from_pretrained(
        model_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tp_model = load_tp_qwen3_from_pretrained(
        model_dir,
        distributed,  # type: ignore[arg-type]
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
    active_mask = torch.tensor([True, True])
    with torch.inference_mode():
        actual_decode = tp_model.decode_with_cache(next_ids, active_mask, cache)[:, 0]
        row0 = torch.cat((input_ids[0, :4], next_ids[0])).unsqueeze(0)
        row1 = torch.cat((input_ids[1, :2], next_ids[1])).unsqueeze(0)
        expected_row0 = reference(row0)[:, -1]
        expected_row1 = reference(row1)[:, -1]
    assert_close(actual_decode[0:1], expected_row0, "TP=2 cached Decode row 0")
    assert_close(actual_decode[1:2], expected_row1, "TP=2 cached Decode row 1")

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
        top_k=1,
        seed=7,
    )
    expected_sample = reference_engine.generate_batch(prompts, sample_config)
    actual_sample = tp_engine.generate_batch(prompts, sample_config)
    assert [item.generated_ids for item in actual_sample] == [
        item.generated_ids for item in expected_sample
    ]


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
                if operation == "all_reduce":
                    result = torch.stack(ordered).sum(dim=0)
                elif operation == "all_gather":
                    result = torch.cat(ordered, dim=-1)
                else:
                    result = slot[src]
                self.results[key] = result
                self.readers[key] = 0
                self.condition.notify_all()
            ready = self.condition.wait_for(lambda: key in self.results, timeout=30)
            if not ready:
                raise TimeoutError(f"线程 collective 超时：{key}")
            result = self.results[key].clone()
            self.readers[key] += 1
            if self.readers[key] == self.world_size:
                del self.slots[key]
                del self.results[key]
                del self.readers[key]
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


def main() -> None:
    check_plan_boundaries()
    check_distributed_boundaries()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model_dir = root / "model"
        save_checkpoint(model_dir)
        check_threaded_tp(str(model_dir), 2)
        # TP=4 走 world_size > KV heads 的复制分支；确保 replicated KV 与 query
        # group 的对应关系仍能恢复完整模型结果。
        check_threaded_tp(str(model_dir), 4)
        if os.environ.get("MINIGPT_RUN_GLOO_TESTS") == "1":
            rendezvous_path = root / "gloo-rendezvous"
            mp.spawn(
                _tp_worker,
                args=(2, str(model_dir), str(rendezvous_path)),
                nprocs=2,
                join=True,
            )
            print("v0.6 Gloo process-group tests passed.")
        else:
            print("v0.6 Gloo test skipped; set MINIGPT_RUN_GLOO_TESTS=1 to enable it.")
    print("v0.6 Qwen3 Tensor Parallel simulation tests passed.")


if __name__ == "__main__":
    main()
