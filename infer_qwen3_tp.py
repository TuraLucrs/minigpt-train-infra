"""通过 torchrun 从本地 Hugging Face Qwen3 目录执行 Tensor Parallel 推理。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.inference import GenerationConfig  # noqa: E402
from minigpt.qwen3_tp import load_tp_qwen3_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 torchrun 和本地 Qwen3 safetensors 执行 Tensor Parallel 推理。"
    )
    parser.add_argument("--model-dir", required=True, help="本地 Hugging Face 模型目录")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己。")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--eos-token-id",
        type=int,
        action="append",
        default=None,
        help="停止 token，可重复传入；默认读取 generation_config.json",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "npu"), default="auto"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--backend", choices=("auto", "gloo", "nccl", "hccl"), default="auto"
    )
    parser.add_argument(
        "--decode-mode", choices=("kv_cache", "recompute"), default="kv_cache"
    )
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            args.device,
            args.precision,
            backend=args.backend,
            timeout_seconds=args.distributed_timeout_seconds,
        )
        engine = load_tp_qwen3_engine(
            project_path(args.model_dir),
            distributed,
            decode_mode=args.decode_mode,
            use_chat_template=args.chat_template,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
        )
        eos_token_ids = (
            tuple(args.eos_token_id)
            if args.eos_token_id is not None
            else engine.tokenizer.generation_eos_token_ids
        )
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            eos_token_ids=eos_token_ids or None,
        )
        result = engine.generate(args.prompt, config)
        distributed.runtime.synchronize()

        if distributed.is_primary:
            print("=" * 80)
            print("Qwen3 Tensor Parallel 推理")
            print(f"world size      : {distributed.world_size}")
            print(f"backend         : {distributed.backend}")
            print(
                f"device          : {distributed.runtime.device_name()} "
                f"({distributed.runtime.device})"
            )
            print(f"precision       : {distributed.runtime.precision}")
            print(f"decode mode     : {args.decode_mode}")
            print(f"input tokens    : {len(result.prompt_ids)}")
            print(f"output tokens   : {len(result.generated_ids)}")
            print(f"stop reason     : {result.stop_reason}")
            print("-" * 80)
            print(result.full_text)
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
