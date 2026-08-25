"""MiniGPT 独立单设备推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.inference import GenerationConfig, load_minigpt_engine  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练完成的 MiniGPT checkpoint 生成文本。")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径，例如 runs/tiny_cpu/latest.pt")
    parser.add_argument("--tokenizer", default="", help="可选 tokenizer.json；默认从 checkpoint 目录寻找")
    parser.add_argument("--prompt", default="MiniGPT", help="输入文本")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="最多生成多少个新 token")
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy", help="token 选择策略")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top-k", type=int, default=None, help="只从分数最高的 k 个 token 中采样")
    parser.add_argument("--seed", type=int, default=1337, help="sample 策略的随机种子")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    runtime = RuntimeContext.create(args.device, args.precision)
    tokenizer_path = project_path(args.tokenizer) if args.tokenizer else None
    engine = load_minigpt_engine(
        checkpoint_path=project_path(args.checkpoint),
        runtime=runtime,
        tokenizer_path=tokenizer_path,
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    result = engine.generate(args.prompt, config)
    runtime.synchronize()

    print("=" * 80)
    print("MiniGPT 单设备推理")
    print(f"device          : {runtime.device}")
    print(f"precision       : {runtime.precision}")
    print(f"input tokens    : {len(result.prompt_ids)}")
    print(f"prefill tokens  : {result.prefill_tokens}")
    print(f"output tokens   : {len(result.generated_ids)}")
    print(f"stop reason     : {result.stop_reason}")
    print("-" * 80)
    print(result.full_text)


if __name__ == "__main__":
    main()
