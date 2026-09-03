"""创建只用于离线正确性/CLI smoke test 的 tiny Qwen3 Hugging Face 目录。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="创建 tiny Qwen3 safetensors/tokenizer 夹具（不能用于性能结论）。"
    )
    parser.add_argument("--output", default="runs/tiny_qwen3_fixture")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        torch_dtype="float32",
    )
    raw_config = {
        **asdict(config),
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "use_cache": True,
    }
    (output / "config.json").write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 1,
                "eos_token_id": [2, 1],
                "pad_token_id": 0,
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.manual_seed(args.seed)
    model = Qwen3ForCausalLM(config).eval()
    from safetensors.torch import save_file

    save_file(
        {
            name: tensor.detach().contiguous()
            for name, tensor in model.state_dict().items()
        },
        output / "model.safetensors",
    )

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    words = [
        "Hello",
        "world",
        "MiniGPT",
        "Qwen",
        "inference",
        "cache",
        "test",
        "one",
        "two",
        "three",
    ]
    vocabulary = {token: index for index, token in enumerate(special + words)}
    for index in range(len(vocabulary), config.vocab_size):
        vocabulary[f"<unused_{index}>"] = index
    backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.decoder = decoders.WordPiece(prefix="##")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(output)

    manifest = {
        "purpose": "correctness_and_cli_smoke_only",
        "formal_performance_model": False,
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "fixture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"tiny Qwen3 fixture: {output}")
    print("用途：正确性和 CLI smoke test；禁止作为性能证据。")


if __name__ == "__main__":
    main()
