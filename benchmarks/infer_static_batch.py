"""MiniGPT 静态 batch 推理 Benchmark。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_static_batch  # noqa: E402
from minigpt.inference import GenerationConfig, load_minigpt_engine  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量 MiniGPT 静态 batch 推理吞吐和显存。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eos-token-id", type=int, action="append", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="runs/static_batch_benchmark.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    runtime = RuntimeContext.create(args.device, args.precision)
    engine = load_minigpt_engine(
        checkpoint_path=project_path(args.checkpoint),
        runtime=runtime,
        tokenizer_path=project_path(args.tokenizer) if args.tokenizer else None,
        decode_mode="kv_cache",
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        eos_token_ids=None if args.eos_token_id is None else tuple(args.eos_token_id),
    )
    report = benchmark_static_batch(
        engine,
        args.prompt,
        generation_config,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
