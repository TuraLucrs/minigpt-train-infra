"""v0.3 单设备推理、阶段边界和指标测试。"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_generation, percentile, timed_generate  # noqa: E402
from minigpt.checkpoint import save_checkpoint  # noqa: E402
from minigpt.experiment import sha256_file  # noqa: E402
from minigpt.inference import (  # noqa: E402
    GenerationConfig,
    InferenceEngine,
    MiniGPTModelRunner,
    load_minigpt_engine,
)
from minigpt.model import MiniGPT, MiniGPTConfig  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402
from minigpt.tokenizer import CharTokenizer  # noqa: E402


def build_engine() -> tuple[InferenceEngine, MiniGPT, CharTokenizer, RuntimeContext]:
    torch.manual_seed(123)
    text = "hello minigpt inference\n"
    tokenizer = CharTokenizer.train_from_text(text)
    model = MiniGPT(
        MiniGPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
    )
    runtime = RuntimeContext.create("cpu", "fp32")
    engine = InferenceEngine(MiniGPTModelRunner(model, runtime), tokenizer)
    return engine, model, tokenizer, runtime


def main() -> None:
    engine, model, tokenizer, runtime = build_engine()

    prompt = "hello"
    input_ids = engine.encode_prompt(prompt)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.inference_mode():
        full_logits = model(input_ids)
        prefill_logits = engine.runner.prefill(input_ids, attention_mask)
    assert torch.equal(prefill_logits, full_logits[:, -1, :])
    assert prefill_logits.shape == (1, tokenizer.vocab_size)

    # v0.3 的 Decode 明确仍重算完整上下文；它必须与直接 model forward 的最后位置一致。
    extended_ids = torch.cat((input_ids, torch.tensor([[1]], dtype=torch.long)), dim=1)
    with torch.inference_mode():
        decode_logits = engine.runner.decode(
            torch.tensor([[1]], dtype=torch.long), torch.ones(1, dtype=torch.bool)
        )
        expected_decode = model(extended_ids)[:, -1, :]
    assert torch.equal(decode_logits, expected_decode)

    greedy_config = GenerationConfig(max_new_tokens=4, strategy="greedy")
    greedy_a = engine.generate(prompt, greedy_config)
    greedy_b = engine.generate(prompt, greedy_config)
    assert greedy_a.generated_ids == greedy_b.generated_ids
    assert len(greedy_a.prompt_ids) == len(tokenizer.encode(prompt))
    assert len(greedy_a.generated_ids) == 4
    assert greedy_a.all_ids == greedy_a.prompt_ids + greedy_a.generated_ids
    assert greedy_a.full_text == tokenizer.decode(greedy_a.all_ids)
    assert greedy_a.prefill_tokens == len(greedy_a.prompt_ids)

    long_prompt = "hellohello"
    long_input_ids = engine.encode_prompt(long_prompt)
    long_mask = torch.ones_like(long_input_ids, dtype=torch.bool)
    with torch.inference_mode():
        long_prefill = engine.runner.prefill(long_input_ids, long_mask)
        cropped_expected = model(long_input_ids[:, -model.config.block_size :])[
            :, -1, :
        ]
    assert torch.equal(long_prefill, cropped_expected)
    assert (
        engine.generate(long_prompt, GenerationConfig(max_new_tokens=1)).prefill_tokens
        == model.config.block_size
    )

    sample_config = GenerationConfig(
        max_new_tokens=5, strategy="sample", temperature=0.8, top_k=4, seed=7
    )
    assert (
        engine.generate(prompt, sample_config).generated_ids
        == engine.generate(prompt, sample_config).generated_ids
    )

    nucleus_config = GenerationConfig(strategy="sample", top_p=0.5, seed=11)
    nucleus_generator = engine.make_generator(nucleus_config)
    selected = engine.select_next_token(
        torch.tensor([[10.0, 0.0, 0.0]]),
        nucleus_config,
        nucleus_generator,
    )
    assert selected.item() == 0

    for invalid_config in (
        GenerationConfig(top_p=0.0),
        GenerationConfig(eos_token_id=1, eos_token_ids=(2,)),
    ):
        try:
            invalid_config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("非法采样/停止配置必须被拒绝")

    try:
        engine.generate("", greedy_config)
    except ValueError:
        pass
    else:
        raise AssertionError("空 prompt 应被拒绝")

    timed = timed_generate(engine, prompt, greedy_config)
    metrics = timed.metrics
    assert metrics.input_tokens == len(tokenizer.encode(prompt))
    assert metrics.output_tokens == 4
    assert len(metrics.decode_step_ms) == 3
    assert metrics.tpot_ms is not None
    assert abs(metrics.tpot_ms * 3 - metrics.decode_ms) < 1e-6
    assert metrics.ttft_ms >= metrics.prefill_ms
    assert metrics.e2e_latency_ms >= metrics.ttft_ms
    assert metrics.output_tokens_per_second > 0

    report = benchmark_generation(engine, prompt, greedy_config, warmup=1, repeats=3)
    assert report["benchmark"] == "single_request_cpu_recompute"
    assert len(report["runs"]) == 3
    assert report["summary"]["ttft_ms"]["count"] == 3
    assert report["summary"]["tpot_ms"]["median"] > 0
    assert report["result"]["generated_ids"] == greedy_a.generated_ids
    assert abs(percentile([1.0, 2.0, 3.0], 0.9) - 2.8) < 1e-12

    warnings: list[str] = []
    cpu_fallback = RuntimeContext.create("cpu", "bf16", warn=warnings.append)
    assert cpu_fallback.precision == "fp32"
    assert warnings
    assert runtime.device_name() == "CPU"

    unavailable_accelerator = None
    if not torch.cuda.is_available():
        unavailable_accelerator = "cuda"
    elif not hasattr(torch, "npu"):
        unavailable_accelerator = "npu"
    if unavailable_accelerator is not None:
        try:
            RuntimeContext.create(
                unavailable_accelerator,
                "fp32",
                allow_accelerator_fallback=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("严格设备 smoke 不能回退到 CPU 后报告成功")

    try:
        RuntimeContext.create(
            "cpu",
            "bf16",
            allow_precision_fallback=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("严格精度 smoke 不能回退到 fp32 后报告成功")

    # 独立推理入口必须能直接读取训练版本保存的 config/model/tokenizer。
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "run"
        run_dir.mkdir(parents=True)
        fingerprint_file = run_dir / "fingerprint.bin"
        fingerprint_file.write_bytes(b"abc")
        assert sha256_file(fingerprint_file) == hashlib.sha256(b"abc").hexdigest()
        tokenizer.save(run_dir / "tokenizer.json")
        checkpoint_path = run_dir / "latest.pt"
        save_checkpoint(
            checkpoint_path,
            {
                "model_state": model.state_dict(),
                "config": {
                    "model": {
                        "block_size": model.config.block_size,
                        "n_layer": model.config.n_layer,
                        "n_head": model.config.n_head,
                        "n_embd": model.config.n_embd,
                        "dropout": model.config.dropout,
                    }
                },
            },
        )
        loaded_engine = load_minigpt_engine(checkpoint_path, runtime)
        loaded_result = loaded_engine.generate(prompt, greedy_config)
        assert loaded_result.generated_ids == greedy_a.generated_ids

    print("v0.3 inference and benchmark tests passed.")


if __name__ == "__main__":
    main()
