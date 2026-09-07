"""v0.7 请求状态、动态 batching、slot KV 和清理路径正确性门。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.inference import (  # noqa: E402
    CachedMiniGPTModelRunner,
    GenerationConfig,
    InferenceEngine,
    SlotCachedMiniGPTModelRunner,
)
from minigpt.model import MiniGPT, MiniGPTConfig  # noqa: E402
from minigpt.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from minigpt.qwen3_inference import (  # noqa: E402
    CachedQwen3ModelRunner,
    SlotCachedQwen3ModelRunner,
)
from minigpt.runtime import RuntimeContext  # noqa: E402
from minigpt.serving import (  # noqa: E402
    ContinuousBatchEngine,
    KVSlotAllocator,
    RequestSpec,
    RequestState,
)
from minigpt.tokenizer import CharTokenizer  # noqa: E402


class IntegerTokenizer:
    vocab_size = 40
    pad_token_id = 0

    @staticmethod
    def encode(text: str) -> list[int]:
        return [int(value) for value in text.split()]

    @staticmethod
    def decode(token_ids: Sequence[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config.from_dict(
        {
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
    )


def check_allocator_ownership() -> None:
    allocator = KVSlotAllocator(2)
    assert allocator.allocate("a") == 0
    assert allocator.allocate("b") == 1
    assert allocator.used == 2 and allocator.free == 0
    try:
        allocator.allocate("c")
    except RuntimeError:
        pass
    else:
        raise AssertionError("满容量 allocator 必须拒绝继续分配")
    try:
        allocator.release(0, "b")
    except RuntimeError:
        pass
    else:
        raise AssertionError("非 owner 不能释放 KV slot")
    allocator.release(0, "a")
    assert allocator.allocate("c") == 0
    assert allocator.peak_used == 2


def build_minigpt() -> tuple[MiniGPT, RuntimeContext, CharTokenizer]:
    torch.manual_seed(2026)
    tokenizer = CharTokenizer.train_from_text("abcdefghijklmnop\n")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
    ).eval()
    return model, RuntimeContext.create("cpu", "fp32"), tokenizer


def check_dynamic_minigpt() -> None:
    model, runtime, tokenizer = build_minigpt()
    reference = InferenceEngine(CachedMiniGPTModelRunner(model, runtime), tokenizer)
    runner = SlotCachedMiniGPTModelRunner(model, runtime, max_slots=2)
    serving = ContinuousBatchEngine(
        runner,
        tokenizer,
        max_queue_size=4,
    )
    specs = [
        RequestSpec("a", "ab", GenerationConfig(max_new_tokens=5)),
        RequestSpec("b", "abc", GenerationConfig(max_new_tokens=2)),
        RequestSpec("c", "abcd", GenerationConfig(max_new_tokens=3)),
    ]
    for spec in specs:
        assert serving.submit(spec).state == RequestState.WAITING

    first = serving.step()
    assert first["prefill_batch_size"] == 2
    assert first["active_after"] == 2
    second = serving.step()
    assert second["decode_batch_size"] == 2
    assert "b" in second["finished_request_ids"]
    assert "c" in second["admitted_request_ids"]
    serving.run_until_idle()

    for spec in specs:
        actual = serving.requests[spec.request_id]
        expected = reference.generate(spec.prompt, spec.config)
        assert actual.state == RequestState.FINISHED
        assert actual.generated_ids == expected.generated_ids
    assert serving.allocator.used == 0
    assert serving.allocator.allocations == serving.allocator.releases == 3
    assert runner.cache_lengths([0, 1]) == [0, 0]
    assert serving.requests["c"].admitted_at_ms is not None
    assert (
        serving.requests["c"].admitted_at_ms
        >= serving.requests["c"].submitted_at_ms
    )

    report = serving.report(ttft_slo_ms=60_000.0, e2e_slo_ms=60_000.0)
    assert report["summary"]["state_counts"]["finished"] == 3
    assert report["summary"]["output_tokens_per_second"] > 0.0
    assert report["kv_cache"]["peak_slots_used"] == 2
    assert report["kv_cache"]["allocations"] == 3
    assert report["kv_cache"]["releases"] == 3


def run_sample_with_optional_distractor(include_distractor: bool) -> list[int]:
    model, runtime, tokenizer = build_minigpt()
    runner = SlotCachedMiniGPTModelRunner(model, runtime, max_slots=2)
    serving = ContinuousBatchEngine(runner, tokenizer)
    if include_distractor:
        serving.submit(
            RequestSpec(
                "distractor",
                "abcd",
                GenerationConfig(max_new_tokens=4, strategy="sample", seed=99),
            )
        )
    serving.submit(
        RequestSpec(
            "sample",
            "abc",
            GenerationConfig(
                max_new_tokens=5,
                strategy="sample",
                temperature=0.8,
                top_k=5,
                seed=7,
            ),
        )
    )
    serving.run_until_idle()
    return serving.requests["sample"].generated_ids


def check_per_request_sampling() -> None:
    assert run_sample_with_optional_distractor(False) == run_sample_with_optional_distractor(
        True
    )


def check_cancel_reject_and_error_cleanup() -> None:
    model, runtime, tokenizer = build_minigpt()
    runner = SlotCachedMiniGPTModelRunner(model, runtime, max_slots=1)
    serving = ContinuousBatchEngine(runner, tokenizer, max_queue_size=1)
    active = serving.submit(
        RequestSpec("active", "ab", GenerationConfig(max_new_tokens=4))
    )
    rejected = serving.submit(
        RequestSpec("overflow", "abc", GenerationConfig(max_new_tokens=2))
    )
    assert rejected.state == RequestState.REJECTED
    assert rejected.stop_reason == "queue_full"
    serving.step()
    assert active.slot_id == 0
    assert serving.cancel("active")
    assert active.state == RequestState.CANCELLED
    assert serving.allocator.free == 1
    assert runner.cache_lengths([0]) == [0]

    too_large = serving.submit(
        RequestSpec("too-large", "abcdefghijklmno", GenerationConfig(max_new_tokens=3))
    )
    assert too_large.state == RequestState.REJECTED
    assert too_large.stop_reason == "invalid_request"

    class FailingRunner:
        implementation_name = "failing_slot_runner"

        def __init__(self, delegate: SlotCachedMiniGPTModelRunner) -> None:
            self.delegate = delegate
            self.runtime = delegate.runtime
            self.max_slots = delegate.max_slots
            self.max_seq_len = delegate.max_seq_len

        def validate_request(self, prompt_length: int, max_new_tokens: int) -> None:
            self.delegate.validate_request(prompt_length, max_new_tokens)

        def prefill_slots(self, *args: object, **kwargs: object) -> torch.Tensor:
            return self.delegate.prefill_slots(*args, **kwargs)  # type: ignore[arg-type]

        def decode_slots(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise RuntimeError("injected decode failure")

        def release_slots(self, slot_ids: Sequence[int]) -> None:
            self.delegate.release_slots(slot_ids)

        def cache_lengths(self, slot_ids: Sequence[int]) -> list[int]:
            return self.delegate.cache_lengths(slot_ids)

    failure_engine = ContinuousBatchEngine(FailingRunner(runner), tokenizer)
    failed = failure_engine.submit(
        RequestSpec("failed", "ab", GenerationConfig(max_new_tokens=3))
    )
    failure_engine.step()
    try:
        failure_engine.step()
    except RuntimeError as exc:
        assert "injected decode failure" in str(exc)
    else:
        raise AssertionError("注入的 Decode 故障必须传播给调用方")
    assert failed.state == RequestState.FAILED
    assert failure_engine.allocator.free == 1
    assert runner.cache_lengths([0]) == [0]


def check_dynamic_qwen3() -> None:
    torch.manual_seed(2026)
    runtime = RuntimeContext.create("cpu", "fp32")
    tokenizer = IntegerTokenizer()
    model = Qwen3ForCausalLM(tiny_qwen3_config()).eval()
    reference = InferenceEngine(
        CachedQwen3ModelRunner(model, runtime),
        tokenizer,
        pad_token_id=tokenizer.pad_token_id,
    )
    runner = SlotCachedQwen3ModelRunner(
        model,
        runtime,
        max_slots=2,
        max_seq_len=12,
    )
    serving = ContinuousBatchEngine(runner, tokenizer)
    specs = [
        RequestSpec("q0", "4 5 6", GenerationConfig(max_new_tokens=5)),
        RequestSpec("q1", "7 8", GenerationConfig(max_new_tokens=2)),
        RequestSpec("q2", "9 10 11 12", GenerationConfig(max_new_tokens=3)),
    ]
    for spec in specs:
        serving.submit(spec)
    serving.run_until_idle()
    for spec in specs:
        assert serving.requests[spec.request_id].generated_ids == reference.generate(
            spec.prompt,
            spec.config,
        ).generated_ids
    assert runner.cache_lengths([0, 1]) == [0, 0]


def main() -> None:
    check_allocator_ownership()
    check_dynamic_minigpt()
    check_per_request_sampling()
    check_cancel_reject_and_error_cleanup()
    check_dynamic_qwen3()
    print("v0.7 continuous batching scheduler and KV slot tests passed.")


if __name__ == "__main__":
    main()
