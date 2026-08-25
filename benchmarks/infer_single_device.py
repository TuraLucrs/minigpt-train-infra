"""MiniGPT 单请求、单设备推理 Benchmark。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_generation  # noqa: E402
from minigpt.experiment import build_provenance  # noqa: E402
from minigpt.inference import GenerationConfig, load_minigpt_engine  # noqa: E402
from minigpt.model import count_parameters  # noqa: E402
from minigpt.runtime import RuntimeContext  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测量 MiniGPT 单设备推理 TTFT、TPOT、吞吐和显存。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--prompt", default="MiniGPT")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="runs/inference_benchmark.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    runtime = RuntimeContext.create(args.device, args.precision)
    checkpoint_path = project_path(args.checkpoint)
    engine = load_minigpt_engine(
        checkpoint_path=checkpoint_path,
        runtime=runtime,
        tokenizer_path=project_path(args.tokenizer) if args.tokenizer else None,
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    report = benchmark_generation(
        engine=engine,
        prompt=args.prompt,
        config=config,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    report["model"] = {
        "type": "MiniGPT",
        "config": asdict(engine.runner.model.config),
        "parameters": count_parameters(engine.runner.model),
    }
    report["provenance"] = build_provenance(PROJECT_ROOT, checkpoint_path, sys.argv)

    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("=" * 80)
    print("MiniGPT 单设备推理 Benchmark（中位数）")
    print(f"device          : {report['environment']['device_name']}")
    print(f"precision       : {report['environment']['precision']}")
    print(f"TTFT            : {summary['ttft_ms']['median']:.3f} ms")
    if "tpot_ms" in summary:
        print(f"TPOT            : {summary['tpot_ms']['median']:.3f} ms")
    print(f"E2E latency     : {summary['e2e_latency_ms']['median']:.3f} ms")
    print(f"output tok/s    : {summary['output_tokens_per_second']['median']:.3f}")
    print(f"peak device MB  : {summary['peak_device_memory_mb']['median']:.3f}")
    print(f"报告             : {output_path}")


if __name__ == "__main__":
    main()
