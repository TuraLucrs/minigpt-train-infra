"""Qwen3 Tensor Parallel 单请求/静态 batch 的 TTFT、TPOT、吞吐与逐 rank HBM。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import benchmark_generation, benchmark_static_batch  # noqa: E402
from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.experiment import build_model_directory_provenance  # noqa: E402
from minigpt.inference import GenerationConfig  # noqa: E402
from minigpt.qwen3 import count_qwen3_parameters  # noqa: E402
from minigpt.qwen3_tp import load_tp_qwen3_engine  # noqa: E402


QWEN3_32B_PARAMETERS = 32_762_123_264


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 torchrun 测量 Qwen3 TP 推理；重复 --prompt 即测静态 batch。"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", action="append", default=None, help="可重复传入")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eos-token-id", type=int, action="append", default=None)
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--physical-card-count", type=int, default=None)
    parser.add_argument("--chips-per-card", type=int, default=None)
    parser.add_argument("--interconnect-topology", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _validate_hardware_description(args: argparse.Namespace, world_size: int) -> bool:
    supplied = (
        args.physical_card_count is not None,
        args.chips_per_card is not None,
        args.interconnect_topology is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "硬件拓扑必须同时提供 physical-card-count、chips-per-card 和 "
            "interconnect-topology"
        )
    if not all(supplied):
        return False
    if args.physical_card_count <= 0 or args.chips_per_card <= 0:
        raise ValueError("physical-card-count/chips-per-card 必须大于 0")
    logical_devices = args.physical_card_count * args.chips_per_card
    if logical_devices != world_size:
        raise ValueError(
            f"physical_card_count × chips_per_card={logical_devices}，"
            f"但 torchrun world_size={world_size}"
        )
    return True


def _result_digest(report: dict[str, object]) -> tuple[float, float]:
    result = report.get("result")
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest_24_bits = int(hashlib.sha256(serialized).hexdigest()[:6], 16)
    return float(digest_24_bits), float(len(serialized))


def _text_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _attach_distributed_measurements(
    report: dict[str, object],
    distributed: DistributedContext,
    *,
    local_parameter_count: int,
    full_parameter_count: int,
    topology_complete: bool,
    load_seconds: float,
    model_loaded_allocated_mb: float,
    model_loaded_peak_mb: float,
    args: argparse.Namespace,
) -> None:
    summary = report["summary"]
    peak_summary = summary["peak_device_memory_mb"]
    median_peak_mb = float(peak_summary["median"])
    current_mb, last_peak_mb = distributed.runtime.memory_stats_mb()
    digest, digest_size = _result_digest(report)
    rank_values = distributed.all_gather_floats(
        [
            median_peak_mb,
            current_mb,
            last_peak_mb,
            digest,
            digest_size,
            load_seconds,
            model_loaded_allocated_mb,
            model_loaded_peak_mb,
        ]
    )
    digests = {(round(values[3]), round(values[4])) for values in rank_values}
    if len(digests) != 1:
        raise AssertionError("不同 TP rank 的生成结果摘要不一致")

    report["distributed"] = {
        **distributed.metadata(),
        "physical_card_count": args.physical_card_count,
        "chips_per_card": args.chips_per_card,
        "interconnect_topology": args.interconnect_topology,
        "topology_complete": topology_complete,
        "run_label": args.run_label,
        "full_parameter_count": full_parameter_count,
        "local_parameter_count": local_parameter_count,
        "ideal_parameter_fraction": 1.0 / distributed.world_size,
        "actual_parameter_fraction": local_parameter_count / full_parameter_count,
        "rank_results_consistent": True,
        "per_rank_memory": [
            {
                "rank": rank,
                "median_request_peak_mb": values[0],
                "post_benchmark_allocated_mb": values[1],
                "last_request_peak_mb": values[2],
                "model_load_seconds": values[5],
                "model_loaded_allocated_mb": values[6],
                "model_loaded_peak_mb": values[7],
            }
            for rank, values in enumerate(rank_values)
        ],
        "max_rank_median_request_peak_mb": max(values[0] for values in rank_values),
        "sum_rank_median_request_peak_mb": sum(values[0] for values in rank_values),
        "max_rank_model_load_seconds": max(values[5] for values in rank_values),
        "max_rank_model_loaded_allocated_mb": max(values[6] for values in rank_values),
    }


def _evidence_class(
    report: dict[str, object],
    distributed: DistributedContext,
    *,
    full_parameter_count: int,
    topology_complete: bool,
    hash_weights: bool,
) -> str:
    provenance = report["provenance"]
    git = provenance["git"]
    clean_commit = git["commit"] != "unknown" and git["dirty"] is False
    if (
        full_parameter_count == QWEN3_32B_PARAMETERS
        and distributed.world_size >= 2
        and distributed.runtime.device.type in {"cuda", "npu"}
        and topology_complete
        and hash_weights
        and clean_commit
    ):
        return "formal_qwen3_32b_tp_hashed"
    if full_parameter_count == QWEN3_32B_PARAMETERS:
        return "qwen3_32b_tp_incomplete_evidence"
    return "correctness_or_nonformal_model"


def main() -> None:
    args = parse_args()
    prompts = args.prompt or ["你好，请介绍一下你自己。"]
    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            args.device,
            args.precision,
            backend=args.backend,
            timeout_seconds=args.distributed_timeout_seconds,
        )
        topology_complete = _validate_hardware_description(args, distributed.world_size)
        model_dir = project_path(args.model_dir)
        distributed.runtime.synchronize()
        distributed.runtime.reset_peak_memory()
        load_started = time.perf_counter()
        engine = load_tp_qwen3_engine(
            model_dir,
            distributed,
            decode_mode=args.decode_mode,
            use_chat_template=args.chat_template,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
        )
        distributed.runtime.synchronize()
        load_seconds = time.perf_counter() - load_started
        model_loaded_allocated_mb, model_loaded_peak_mb = (
            distributed.runtime.memory_stats_mb()
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
        if len(prompts) == 1:
            report = benchmark_generation(
                engine,
                prompts[0],
                config,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        else:
            report = benchmark_static_batch(
                engine,
                prompts,
                config,
                warmup=args.warmup,
                repeats=args.repeats,
            )

        model = engine.runner.model
        local_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        full_parameter_count = count_qwen3_parameters(model.config)
        report["workload_fingerprint"] = {
            "prompt_sha256": [_text_sha256(prompt) for prompt in prompts],
            "encoded_prompt_lengths": [
                len(engine.tokenizer.encode(prompt)) for prompt in prompts
            ],
            "system_prompt_sha256": _text_sha256(args.system_prompt),
            "use_chat_template": args.chat_template,
            "enable_thinking": args.enable_thinking,
            "decode_mode": args.decode_mode,
        }
        report["model"] = {
            "type": "TensorParallelQwen3ForCausalLM",
            "config": asdict(model.config),
            "parameters": full_parameter_count,
        }
        _attach_distributed_measurements(
            report,
            distributed,
            local_parameter_count=local_parameter_count,
            full_parameter_count=full_parameter_count,
            topology_complete=topology_complete,
            load_seconds=load_seconds,
            model_loaded_allocated_mb=model_loaded_allocated_mb,
            model_loaded_peak_mb=model_loaded_peak_mb,
            args=args,
        )
        if not distributed.is_primary:
            return

        report["provenance"] = build_model_directory_provenance(
            PROJECT_ROOT,
            model_dir,
            sys.argv,
            hash_weights=args.hash_weights,
        )
        report["evidence_class"] = _evidence_class(
            report,
            distributed,
            full_parameter_count=full_parameter_count,
            topology_complete=topology_complete,
            hash_weights=args.hash_weights,
        )
        output = (
            project_path(args.output)
            if args.output is not None
            else PROJECT_ROOT / "runs" / f"qwen3_tp{distributed.world_size}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        summary = report["summary"]
        print("=" * 80)
        print("Qwen3 Tensor Parallel Benchmark（中位数）")
        print(f"world size       : {distributed.world_size}")
        print(f"backend          : {distributed.backend}")
        print(f"evidence class   : {report['evidence_class']}")
        if "ttft_ms" in summary:
            print(f"TTFT             : {summary['ttft_ms']['median']:.3f} ms")
        if "tpot_ms" in summary:
            print(f"TPOT             : {summary['tpot_ms']['median']:.3f} ms")
        print(f"E2E latency      : {summary['e2e_latency_ms']['median']:.3f} ms")
        print(f"output tok/s     : {summary['output_tokens_per_second']['median']:.3f}")
        print(
            "model load       : "
            f"{report['distributed']['max_rank_model_load_seconds']:.3f} s"
        )
        print(
            "max rank peak MB : "
            f"{report['distributed']['max_rank_median_request_peak_mb']:.3f}"
        )
        print(f"报告              : {output}")
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
