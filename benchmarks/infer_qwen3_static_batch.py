"""Qwen3 静态 batch 吞吐与显存 Benchmark。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_static_batch  # noqa: E402
from minigpt.experiment import build_model_directory_provenance  # noqa: E402
from minigpt.inference import GenerationConfig  # noqa: E402
from minigpt.qwen3 import count_qwen3_parameters  # noqa: E402
from minigpt.qwen3_inference import load_qwen3_engine  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量 Qwen3 静态 batch 吞吐和显存。")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", action="append", required=True, help="可重复传入")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "npu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--decode-mode", choices=("kv_cache", "recompute"), default="kv_cache")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--output", default="runs/qwen3_static_batch.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    runtime = RuntimeContext.create(args.device, args.precision)
    model_dir = project_path(args.model_dir)
    engine = load_qwen3_engine(
        model_dir,
        runtime,
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
    report = benchmark_static_batch(
        engine,
        args.prompt,
        config,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    model = engine.runner.model
    parameter_count = count_qwen3_parameters(model)
    report["model"] = {
        "type": "Qwen3ForCausalLM",
        "config": asdict(model.config),
        "parameters": parameter_count,
    }
    if parameter_count == 32_762_123_264 and args.hash_weights:
        report["evidence_class"] = "formal_qwen3_32b_hashed"
    elif parameter_count == 32_762_123_264:
        report["evidence_class"] = "qwen3_32b_unhashed"
    else:
        report["evidence_class"] = "correctness_or_nonformal_model"
    report["provenance"] = build_model_directory_provenance(
        PROJECT_ROOT,
        model_dir,
        sys.argv,
        hash_weights=args.hash_weights,
    )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print("=" * 80)
    print("Qwen3 静态 batch Benchmark（中位数）")
    print(f"batch size       : {len(args.prompt)}")
    print(f"evidence class   : {report['evidence_class']}")
    print(f"E2E latency      : {summary['e2e_latency_ms']['median']:.3f} ms")
    print(f"output tok/s     : {summary['output_tokens_per_second']['median']:.3f}")
    print(f"peak device MB   : {summary['peak_device_memory_mb']['median']:.3f}")
    print(f"报告              : {output}")


if __name__ == "__main__":
    main()
