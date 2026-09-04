"""Qwen3 数学、HF 对齐、KV Cache、权重加载与显存规划测试。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.memory_planner import estimate_qwen3_memory  # noqa: E402
from minigpt.qwen3 import (  # noqa: E402
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
    count_qwen3_parameters,
    load_qwen3_from_pretrained,
)
from minigpt.qwen3_inference import (  # noqa: E402
    CachedQwen3ModelRunner,
    Qwen3Tokenizer,
    _load_generation_eos_token_ids,
    _last_valid_positions,
)
from minigpt.runtime import RuntimeContext  # noqa: E402


def tiny_config_dict() -> dict:
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


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, atol=2e-5, rtol=2e-4):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{label} 最大绝对差为 {max_diff}")


def save_single_checkpoint(directory: Path, model: Qwen3ForCausalLM, raw: dict) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )
    state = {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, directory / "model.safetensors")


def save_sharded_checkpoint(directory: Path, model: Qwen3ForCausalLM, raw: dict) -> None:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(raw, indent=2),
        encoding="utf-8",
    )
    state = list(model.state_dict().items())
    shards = (state[::2], state[1::2])
    weight_map = {}
    total_size = 0
    for shard_index, items in enumerate(shards, start=1):
        filename = f"model-{shard_index:05d}-of-00002.safetensors"
        tensors = {}
        for name, tensor in items:
            tensors[name] = tensor.detach().contiguous()
            weight_map[name] = filename
            total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, directory / filename)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8",
    )


def compare_with_transformers(
    model: Qwen3ForCausalLM,
    raw_config: dict,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    from transformers import Qwen3Config as HFQwen3Config
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    reference = HFQwen3ForCausalLM(HFQwen3Config(**raw_config)).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
    with torch.inference_mode():
        actual = model(input_ids, attention_mask, position_ids)
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
    assert_close(actual[attention_mask.bool()], expected[attention_mask.bool()], "HF full logits")


def check_chat_template_result_normalization() -> None:
    """锁住 Transformers 4.x list 与 5.x BatchEncoding 两种返回形态。"""

    class FakeTokenizer:
        def __init__(self, result: object) -> None:
            self.result = result

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> object:
            return self.result

    def encode(result: object) -> list[int]:
        tokenizer = object.__new__(Qwen3Tokenizer)
        tokenizer._tokenizer = FakeTokenizer(result)
        tokenizer.use_chat_template = True
        tokenizer.system_prompt = None
        tokenizer.enable_thinking = False
        return tokenizer.encode("hello")

    assert encode([11, 12, 13]) == [11, 12, 13]
    assert encode({"input_ids": torch.tensor([[21, 22, 23]])}) == [21, 22, 23]
    try:
        encode({"input_ids": [[31, 32], [41, 42]]})
    except ValueError:
        pass
    else:
        raise AssertionError("chat template 的多行 token 结果必须被拒绝")


def main() -> None:
    torch.manual_seed(2026)
    check_chat_template_result_normalization()
    raw_config = tiny_config_dict()
    config = Qwen3Config.from_dict(raw_config)
    assert config.query_width == 32
    assert config.hidden_size == 24
    model = Qwen3ForCausalLM(config).eval()

    rotary = Qwen3RotaryEmbedding(config.head_dim, config.rope_theta)
    rotary_reference = torch.zeros(1, 2, config.head_dim, dtype=torch.bfloat16)
    high_positions = torch.tensor([[30_000, 40_000]])
    expected_rope = rotary(rotary_reference, high_positions)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_rope = rotary(rotary_reference, high_positions)
    assert torch.equal(autocast_rope[0], expected_rope[0])
    assert torch.equal(autocast_rope[1], expected_rope[1])

    input_ids = torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    compare_with_transformers(model, raw_config, input_ids, attention_mask)

    left_padded_ids = torch.tensor([[4, 5, 6, 7], [0, 0, 8, 9]])
    left_padded_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    compare_with_transformers(model, raw_config, left_padded_ids, left_padded_mask)
    assert _last_valid_positions(left_padded_mask).tolist() == [3, 3]

    runtime = RuntimeContext.create("cpu", "fp32")
    runner = CachedQwen3ModelRunner(model, runtime)
    runner.validate_generation([4, 2], max_new_tokens=4)
    runner.validate_generation(
        [config.max_position_embeddings],
        max_new_tokens=1,
    )
    try:
        runner.validate_generation(
            [config.max_position_embeddings],
            max_new_tokens=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("满上下文只能直接产生 Prefill 的第一个输出 token")
    lm_head_inputs: list[tuple[int, ...]] = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: lm_head_inputs.append(tuple(inputs[0].shape))
    )
    with torch.inference_mode():
        cached_prefill = runner.prefill(input_ids, attention_mask)
    hook.remove()
    assert cached_prefill.shape == (2, config.vocab_size)
    assert lm_head_inputs == [(2, config.hidden_size)]
    assert runner.cache is not None
    assert torch.count_nonzero(runner.cache.layers[0].key[1, :, 2:]).item() == 0

    last_positions = _last_valid_positions(attention_mask)
    with torch.inference_mode():
        full_prefill = model(
            input_ids,
            attention_mask,
            logit_positions=last_positions,
        )
    assert_close(cached_prefill, full_prefill, "cached prefill")

    # KV 容量按有效 token 数计算，不应被无效 padding 的物理宽度放大或拒绝。
    overpadded_ids = torch.tensor([[0, 0, 4, 5, 0, 0]])
    overpadded_mask = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    compact_cache = model.allocate_kv_cache(
        1,
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        overpadded_cached = model.prefill_with_cache(
            overpadded_ids,
            overpadded_mask,
            compact_cache,
            logit_positions=torch.tensor([3]),
        )
        overpadded_full = model(
            overpadded_ids,
            overpadded_mask,
            logit_positions=torch.tensor([3]),
        )
    assert_close(overpadded_cached, overpadded_full, "overpadded cached prefill")

    next_ids = torch.argmax(full_prefill, dim=-1, keepdim=True)
    active_mask = torch.tensor([True, True])
    with torch.inference_mode():
        cached_decode = runner.decode(next_ids, active_mask)
        row0 = torch.cat((input_ids[0, :4], next_ids[0])).unsqueeze(0)
        row1 = torch.cat((input_ids[1, :2], next_ids[1])).unsqueeze(0)
        expected0 = model(row0)[:, -1]
        expected1 = model(row1)[:, -1]
    assert_close(cached_decode[0:1], expected0, "cached decode row 0")
    assert_close(cached_decode[1:2], expected1, "cached decode row 1")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        generation_dir = root / "generation"
        generation_dir.mkdir()
        (generation_dir / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [2, 1, 2]}),
            encoding="utf-8",
        )
        assert _load_generation_eos_token_ids(generation_dir, None) == (2, 1)

        single = root / "single"
        save_single_checkpoint(single, model, raw_config)
        loaded = load_qwen3_from_pretrained(
            single,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            assert_close(
                loaded(input_ids, attention_mask),
                model(input_ids, attention_mask),
                "single safetensors load",
            )

        sharded = root / "sharded"
        save_sharded_checkpoint(sharded, model, raw_config)
        loaded_sharded = load_qwen3_from_pretrained(
            sharded,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            assert_close(
                loaded_sharded(input_ids, attention_mask),
                model(input_ids, attention_mask),
                "sharded safetensors load",
            )

    official = Qwen3Config.from_json(PROJECT_ROOT / "configs" / "qwen3_32b_official.json")
    assert count_qwen3_parameters(official) == 32_762_123_264
    estimate = estimate_qwen3_memory(
        official,
        tensor_parallel_size=1,
        batch_size=1,
        max_sequence_length=128,
        device_memory_mb=65_536,
    )
    assert abs(estimate.weight_memory_mb - 62_488.791015625) < 1e-9
    assert abs(estimate.kv_cache_memory_mb - 32.0) < 1e-9
    assert not estimate.fits
    tp16 = estimate_qwen3_memory(
        official,
        tensor_parallel_size=16,
        batch_size=1,
        max_sequence_length=128,
        device_memory_mb=65_536,
    )
    assert tp16.query_heads_per_rank == 4
    assert tp16.kv_heads_per_rank == 1
    assert tp16.replicated_weight_overhead_mb > 0.0
    assert tp16.weight_memory_mb > tp16.ideal_balanced_weight_memory_mb

    invalid = dict(raw_config, use_sliding_window=True, sliding_window=8)
    try:
        Qwen3Config.from_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("未实现的 sliding window 必须被显式拒绝")

    try:
        Qwen3Config.from_dict(dict(raw_config, model_type="qwen2"))
    except ValueError:
        pass
    else:
        raise AssertionError("非 Qwen3 checkpoint 必须被显式拒绝")

    print("Qwen3 parity, cache, loading and memory planning tests passed.")


if __name__ == "__main__":
    main()
