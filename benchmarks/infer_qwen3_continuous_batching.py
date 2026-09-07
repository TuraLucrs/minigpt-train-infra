"""在一个 Qwen3 TP replica 上重放 v0.7 Continuous Batching workload。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.experiment import build_model_directory_provenance  # noqa: E402
from minigpt.qwen3 import count_qwen3_parameters  # noqa: E402
from minigpt.qwen3_tp import load_tp_qwen3_slot_runner  # noqa: E402
from minigpt.serving import ContinuousBatchEngine  # noqa: E402
from minigpt.serving_benchmark import benchmark_trace_replay  # noqa: E402
from minigpt.workload import WorkloadTrace  # noqa: E402


QWEN3_32B_PARAMETERS = 32_762_123_264


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--mode", choices=("open_loop", "closed_loop"), required=True)
    parser.add_argument("--closed-loop-clients", type=int, default=None)
    parser.add_argument("--max-slots", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--ttft-slo-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-ms", type=float, default=None)
    parser.add_argument("--e2e-slo-ms", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "npu"), default="auto"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--backend", choices=("auto", "gloo", "nccl", "hccl"), default="auto"
    )
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--layout-id", required=True)
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument(
        "--logical-device-ids",
        required=True,
        help="当前 TP replica 实际使用的全局 logical device id，逗号分隔",
    )
    parser.add_argument("--physical-card-count", type=int, required=True)
    parser.add_argument("--chips-per-card", type=int, required=True)
    parser.add_argument("--interconnect-topology", required=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_device_ids(value: str, world_size: int) -> list[int]:
    try:
        device_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("logical-device-ids 必须是逗号分隔整数") from exc
    if len(device_ids) != world_size:
        raise ValueError(
            f"logical-device-ids 数量 {len(device_ids)} 与 TP world_size "
            f"{world_size} 不一致"
        )
    if len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise ValueError("logical-device-ids 必须是互不重复的非负整数")
    return device_ids


def validate_layout(
    args: argparse.Namespace,
    trace: WorkloadTrace,
    world_size: int,
) -> list[int]:
    if args.replica_count <= 0:
        raise ValueError("replica-count 必须大于 0")
    if not 0 <= args.replica_index < args.replica_count:
        raise ValueError("replica-index 必须位于 [0, replica-count)")
    if args.physical_card_count <= 0 or args.chips_per_card <= 0:
        raise ValueError("physical-card-count/chips-per-card 必须大于 0")
    if trace.partition is None:
        if args.replica_count != 1 or args.replica_index != 0:
            raise ValueError("多副本运行必须使用带 partition 身份的 workload")
    elif (
        trace.partition.replica_count != args.replica_count
        or trace.partition.replica_index != args.replica_index
    ):
        raise ValueError("workload partition 与 replica-index/count 不一致")
    device_ids = parse_device_ids(args.logical_device_ids, world_size)
    declared_total_devices = args.physical_card_count * args.chips_per_card
    if max(device_ids) >= declared_total_devices:
        raise ValueError("logical-device-ids 超出声明的物理拓扑")
    return device_ids


def _digest_numbers(value: str) -> tuple[float, float]:
    return float(int(value[:6], 16)), float(len(value))


def evidence_class(
    report: dict[str, object],
    *,
    parameter_count: int,
    hash_weights: bool,
) -> str:
    git = report["provenance"]["git"]
    protocol = report["protocol"]
    environment = report["environment"]
    formal = (
        parameter_count == QWEN3_32B_PARAMETERS
        and environment["device_type"] in {"cuda", "npu"}
        and environment["precision"] == "bf16"
        and hash_weights
        and git["commit"] != "unknown"
        and git["dirty"] is False
        and int(protocol["warmup"]) >= 1
        and int(protocol["repeats"]) >= 3
    )
    if formal:
        return "formal_qwen3_32b_continuous_batching"
    if parameter_count == QWEN3_32B_PARAMETERS:
        return "qwen3_32b_continuous_batching_incomplete_evidence"
    return "correctness_or_nonformal_model"


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
        trace = WorkloadTrace.load(project_path(args.workload))
        logical_device_ids = validate_layout(args, trace, distributed.world_size)
        model_dir = project_path(args.model_dir)

        distributed.runtime.synchronize()
        distributed.runtime.reset_peak_memory()
        load_started = time.perf_counter()
        runner, tokenizer = load_tp_qwen3_slot_runner(
            model_dir,
            distributed,
            max_slots=args.max_slots,
            max_seq_len=args.max_seq_len,
            use_chat_template=args.chat_template,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
        )
        distributed.runtime.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        model_loaded_memory_mb, model_loaded_peak_mb = (
            distributed.runtime.memory_stats_mb()
        )
        encoded_prompt_lengths = [
            len(tokenizer.encode(request.prompt)) for request in trace.requests
        ]
        for request, prompt_length in zip(trace.requests, encoded_prompt_lengths):
            runner.validate_request(prompt_length, request.config.max_new_tokens)

        engine = ContinuousBatchEngine(
            runner,
            tokenizer,
            max_queue_size=args.max_queue_size,
            distributed=distributed,
        )
        report = benchmark_trace_replay(
            engine,
            trace,
            mode=args.mode,
            closed_loop_clients=args.closed_loop_clients,
            warmup=args.warmup,
            repeats=args.repeats,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            e2e_slo_ms=args.e2e_slo_ms,
            distributed=distributed,
        )

        local_peaks = [
            float(run["memory"]["peak_device_memory_mb"])
            for run in report["runs"]
        ]
        output_digest, output_digest_length = _digest_numbers(
            str(report["runs"][0]["output_sha256"])
        )
        rank_values = distributed.all_gather_floats(
            [
                max(local_peaks),
                statistics.median(local_peaks),
                model_load_seconds,
                model_loaded_memory_mb,
                model_loaded_peak_mb,
                output_digest,
                output_digest_length,
            ]
        )
        if len({(round(values[5]), round(values[6])) for values in rank_values}) != 1:
            raise AssertionError("不同 TP rank 的 trace 输出摘要不一致")

        model = runner.model
        full_parameter_count = count_qwen3_parameters(model.config)
        local_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        report["model"] = {
            "type": "TensorParallelQwen3ForCausalLM",
            "config": asdict(model.config),
            "full_parameter_count": full_parameter_count,
            "local_parameter_count": local_parameter_count,
        }
        report["workload"]["encoded_prompt_lengths"] = encoded_prompt_lengths
        report["distributed"] = {
            **distributed.metadata(),
            "tp_size": distributed.world_size,
            "layout_id": args.layout_id,
            "replica_index": args.replica_index,
            "replica_count": args.replica_count,
            "logical_device_ids": logical_device_ids,
            "physical_card_count": args.physical_card_count,
            "chips_per_card": args.chips_per_card,
            "interconnect_topology": args.interconnect_topology,
            "run_label": args.run_label,
            "rank_results_consistent": True,
            "per_rank": [
                {
                    "tp_rank": rank,
                    "logical_device_id": logical_device_ids[rank],
                    "max_measured_peak_mb": values[0],
                    "median_measured_peak_mb": values[1],
                    "model_load_seconds": values[2],
                    "model_loaded_memory_mb": values[3],
                    "model_loaded_peak_mb": values[4],
                }
                for rank, values in enumerate(rank_values)
            ],
        }
        if not distributed.is_primary:
            return

        report["provenance"] = build_model_directory_provenance(
            PROJECT_ROOT,
            model_dir,
            sys.argv,
            hash_weights=args.hash_weights,
        )
        report["evidence_class"] = evidence_class(
            report,
            parameter_count=full_parameter_count,
            hash_weights=args.hash_weights,
        )
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary = report["summary"]
        print("=" * 80)
        print("Qwen3 Continuous Batching Benchmark（measured repeats 中位数）")
        print(f"layout / replica : {args.layout_id} / {args.replica_index}")
        print(f"TP size          : {distributed.world_size}")
        print(f"workload         : {trace.workload_id}")
        print(f"mode             : {args.mode}")
        print(
            "request/s        : "
            f"{summary['completed_requests_per_second']['median']:.3f}"
        )
        print(
            "goodput request/s: "
            f"{summary['goodput_requests_per_second']['median']:.3f}"
        )
        print(
            "output token/s   : "
            f"{summary['output_tokens_per_second']['median']:.3f}"
        )
        print(f"evidence class   : {report['evidence_class']}")
        print(f"report           : {output}")
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
