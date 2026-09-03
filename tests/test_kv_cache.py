"""KV Cache、静态 batch、EOS 和 benchmark 正确性门。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_static_batch, compare_decode_modes  # noqa: E402
from minigpt.inference import (  # noqa: E402
    CachedMiniGPTModelRunner,
    GenerationConfig,
    InferenceEngine,
    RecomputeMiniGPTModelRunner,
)
from minigpt.model import MiniGPT, MiniGPTConfig  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402
from minigpt.tokenizer import CharTokenizer  # noqa: E402


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, atol=2e-6, rtol=1e-5):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{label} 最大绝对差为 {max_diff}")


def build_engines() -> tuple[InferenceEngine, InferenceEngine, MiniGPT, CharTokenizer]:
    torch.manual_seed(2026)
    tokenizer = CharTokenizer.train_from_text("abcdefghijklmnop\n")
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=12,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
    ).eval()
    runtime = RuntimeContext.create("cpu", "fp32")
    recompute = InferenceEngine(RecomputeMiniGPTModelRunner(model, runtime), tokenizer)
    cached = InferenceEngine(CachedMiniGPTModelRunner(model, runtime), tokenizer)
    return recompute, cached, model, tokenizer


def main() -> None:
    recompute, cached, model, tokenizer = build_engines()
    prompts = ["ab", "abcdef", "abcd"]
    input_ids, attention_mask, _ = cached.encode_prompts(prompts)

    cached_prefill = cached.runner.prefill(input_ids, attention_mask)
    for row, prompt in enumerate(prompts):
        unpadded_ids = torch.tensor([tokenizer.encode(prompt)])
        expected = model(unpadded_ids)[:, -1]
        assert_close(cached_prefill[row : row + 1], expected, f"prefill row {row}")
    recompute_prefill = recompute.runner.prefill(input_ids, attention_mask)
    assert_close(cached_prefill, recompute_prefill, "batched prefill")

    next_ids = torch.argmax(recompute_prefill, dim=-1, keepdim=True)
    active_mask = torch.tensor([True, False, True])
    recompute_decode = recompute.runner.decode(next_ids, active_mask)
    cached_decode = cached.runner.decode(next_ids, active_mask)
    assert_close(cached_decode[active_mask], recompute_decode[active_mask], "masked decode")

    cache = cached.runner.cache
    assert cache is not None
    assert cache.lengths[:3].tolist() == [3, 6, 5]
    assert torch.count_nonzero(cache.layers[0].key[:, :, 6:]).item() == 0
    cache_pointers = [
        (layer.key.data_ptr(), layer.value.data_ptr())
        for layer in cache.layers
    ]

    config = GenerationConfig(max_new_tokens=5)
    recompute_ids = [
        result.generated_ids
        for result in recompute.generate_batch(prompts, config)
    ]
    cached_ids = [
        result.generated_ids
        for result in cached.generate_batch(prompts, config)
    ]
    assert cached_ids == recompute_ids
    assert cache_pointers == [
        (layer.key.data_ptr(), layer.value.data_ptr())
        for layer in cache.layers
    ]

    # learned absolute position 的窗口滚动会触发 Prefill 重建，结果仍须与重算一致。
    rollover_prompts = ["abcdefghijk", "abc"]
    recompute_rollover = [
        result.generated_ids
        for result in recompute.generate_batch(rollover_prompts, config)
    ]
    cached_rollover = [
        result.generated_ids
        for result in cached.generate_batch(rollover_prompts, config)
    ]
    assert cached_rollover == recompute_rollover

    # 一个已停止请求占满窗口时，另一个短请求仍应可以继续 Decode，不能因 inactive row 的
    # position == block_size 而访问越界。
    edge_prompts = ["abcdefghijkl", "ab"]
    edge_ids, edge_mask, _ = cached.encode_prompts(edge_prompts)
    edge_logits = cached.runner.prefill(edge_ids, edge_mask)
    edge_next = torch.argmax(edge_logits, dim=-1, keepdim=True)
    edge_active = torch.tensor([False, True])
    cached.runner.decode(edge_next, edge_active)

    first_token = cached.generate(
        "ab",
        GenerationConfig(max_new_tokens=1),
    ).generated_ids[0]
    eos_result = cached.generate(
        "ab",
        GenerationConfig(max_new_tokens=8, eos_token_id=first_token),
    )
    assert eos_result.generated_ids == [first_token]
    assert eos_result.stop_reason == "eos"

    multi_eos_result = cached.generate(
        "ab",
        GenerationConfig(
            max_new_tokens=8,
            eos_token_ids=((first_token + 1) % tokenizer.vocab_size, first_token),
        ),
    )
    assert multi_eos_result.generated_ids == [first_token]
    assert multi_eos_result.stop_reason == "eos"

    comparison = compare_decode_modes(
        recompute,
        cached,
        "abcdef",
        config,
        warmup=0,
        repeats=2,
    )
    assert comparison["correctness_gate"]["generated_ids_equal"]
    batch_report = benchmark_static_batch(
        cached,
        prompts,
        config,
        warmup=0,
        repeats=2,
    )
    assert batch_report["request"]["batch_size"] == 3

    print("v0.4 KV Cache and static batching tests passed.")


if __name__ == "__main__":
    main()
