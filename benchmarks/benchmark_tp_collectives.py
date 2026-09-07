"""在 torchrun 任务内测量 TP 使用的 AllReduce/AllGather collective。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.benchmark import summarize  # noqa: E402
from minigpt.distributed import DistributedContext  # noqa: E402
from minigpt.experiment import git_snapshot  # noqa: E402


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测量 TP AllReduce/AllGather 延迟和估算链路流量。"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "npu"), default="auto"
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--backend", choices=("auto", "gloo", "nccl", "hccl"), default="auto"
    )
    parser.add_argument("--message-mb", type=float, action="append", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=600)
    parser.add_argument("--physical-card-count", type=int, default=None)
    parser.add_argument("--chips-per-card", type=int, default=None)
    parser.add_argument("--interconnect-topology", default=None)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _topology(
    args: argparse.Namespace, world_size: int
) -> tuple[bool, dict[str, object]]:
    fields = (
        args.physical_card_count,
        args.chips_per_card,
        args.interconnect_topology,
    )
    if any(value is not None for value in fields) and not all(
        value is not None for value in fields
    ):
        raise ValueError("硬件拓扑三个字段必须同时提供")
    complete = all(value is not None for value in fields)
    if complete:
        if args.physical_card_count <= 0 or args.chips_per_card <= 0:
            raise ValueError("physical-card-count/chips-per-card 必须大于 0")
        described_world_size = args.physical_card_count * args.chips_per_card
        if described_world_size != world_size:
            raise ValueError(
                f"拓扑描述包含 {described_world_size} 个 logical devices，"
                f"torchrun world_size={world_size}"
            )
    return complete, {
        "physical_card_count": args.physical_card_count,
        "chips_per_card": args.chips_per_card,
        "interconnect_topology": args.interconnect_topology,
        "topology_complete": complete,
    }


def _time_collective(
    distributed: DistributedContext,
    operation: str,
    tensor: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], int]:
    if operation == "all_reduce":

        def run() -> torch.Tensor:
            return distributed.all_reduce_sum(tensor)

        output_bytes = tensor.numel() * tensor.element_size()
    elif operation == "all_gather":

        def run() -> torch.Tensor:
            return distributed.all_gather_last_dim(tensor)

        output_bytes = tensor.numel() * tensor.element_size() * distributed.world_size
    else:
        raise ValueError(f"未知 collective：{operation}")

    for _ in range(warmup):
        run()
    distributed.runtime.synchronize()
    distributed.barrier()

    local_seconds: list[float] = []
    for _ in range(repeats):
        distributed.barrier()
        distributed.runtime.synchronize()
        started = time.perf_counter()
        result = run()
        distributed.runtime.synchronize()
        elapsed = time.perf_counter() - started
        del result
        rank_elapsed = distributed.all_gather_floats([elapsed])
        local_seconds.append(max(values[0] for values in rank_elapsed))
    return local_seconds, output_bytes


def _measurement(
    operation: str,
    requested_mb: float,
    tensor: torch.Tensor,
    seconds: list[float],
    output_bytes: int,
    world_size: int,
) -> dict[str, object]:
    input_bytes = tensor.numel() * tensor.element_size()
    latency_ms = [value * 1000.0 for value in seconds]
    median_seconds = statistics.median(seconds)
    algorithmic_gib_s = input_bytes / max(median_seconds, 1e-12) / (1024**3)
    if operation == "all_reduce":
        estimated_rank_traffic_bytes = input_bytes * 2.0 * (world_size - 1) / world_size
    else:
        estimated_rank_traffic_bytes = input_bytes * (world_size - 1)
    return {
        "operation": operation,
        "requested_message_mb": requested_mb,
        "input_bytes_per_rank": input_bytes,
        "output_bytes_per_rank": output_bytes,
        "latency_ms": latency_ms,
        "summary_ms": summarize(latency_ms),
        "median_algorithmic_input_gib_per_second": algorithmic_gib_s,
        "median_estimated_rank_traffic_gib_per_second": (
            estimated_rank_traffic_bytes / max(median_seconds, 1e-12) / (1024**3)
        ),
        "traffic_formula": (
            "2*(world_size-1)/world_size*input_bytes"
            if operation == "all_reduce"
            else "(world_size-1)*input_bytes"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
    message_sizes = args.message_mb or [0.25, 1.0, 4.0]
    if any(size <= 0 for size in message_sizes):
        raise ValueError("message-mb 必须大于 0")

    distributed: DistributedContext | None = None
    try:
        distributed = DistributedContext.create(
            args.device,
            args.precision,
            backend=args.backend,
            timeout_seconds=args.distributed_timeout_seconds,
        )
        topology_complete, topology = _topology(args, distributed.world_size)
        dtype = distributed.runtime.amp_dtype or torch.float32
        measurements: list[dict[str, object]] = []

        correctness = torch.tensor(
            [float(distributed.rank + 1)],
            device=distributed.runtime.device,
            dtype=dtype,
        )
        distributed.all_reduce_sum(correctness)
        expected = distributed.world_size * (distributed.world_size + 1) / 2
        if not math.isclose(float(correctness.item()), expected, rel_tol=1e-3):
            raise AssertionError("AllReduce correctness probe 失败")

        for requested_mb in message_sizes:
            element_count = max(
                1,
                math.ceil(
                    requested_mb * MIB / torch.empty((), dtype=dtype).element_size()
                ),
            )
            for operation in ("all_reduce", "all_gather"):
                tensor = torch.zeros(
                    (1, element_count),
                    dtype=dtype,
                    device=distributed.runtime.device,
                )
                seconds, output_bytes = _time_collective(
                    distributed,
                    operation,
                    tensor,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                if distributed.is_primary:
                    measurements.append(
                        _measurement(
                            operation,
                            requested_mb,
                            tensor,
                            seconds,
                            output_bytes,
                            distributed.world_size,
                        )
                    )
                del tensor
                distributed.runtime.empty_cache()

        if not distributed.is_primary:
            return
        git = git_snapshot(PROJECT_ROOT)
        evidence_class = (
            "formal_collective_measurement"
            if distributed.world_size >= 2
            and distributed.runtime.device.type in {"cuda", "npu"}
            and topology_complete
            and git["commit"] != "unknown"
            and git["dirty"] is False
            else "development_or_incomplete_collective_measurement"
        )
        report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "tensor_parallel_collectives",
            "evidence_class": evidence_class,
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                **distributed.runtime.backend_metadata(),
            },
            "distributed": {
                **distributed.metadata(),
                **topology,
                "run_label": args.run_label,
            },
            "request": {
                "message_mb": message_sizes,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "dtype": str(dtype),
            },
            "provenance": {"git": git, "command": list(sys.argv)},
            "measurements": measurements,
        }
        output = (
            project_path(args.output)
            if args.output is not None
            else PROJECT_ROOT / "runs" / f"tp_collectives_{distributed.world_size}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("=" * 80)
        print("Tensor Parallel collective Benchmark")
        print(f"world size       : {distributed.world_size}")
        print(f"backend          : {distributed.backend}")
        print(f"evidence class   : {evidence_class}")
        for item in measurements:
            print(
                f"{item['operation']:10s} {item['requested_message_mb']:7.2f} MiB "
                f"median={item['summary_ms']['median']:.3f} ms"
            )
        print(f"报告              : {output}")
    finally:
        if distributed is not None:
            distributed.close()


if __name__ == "__main__":
    main()
