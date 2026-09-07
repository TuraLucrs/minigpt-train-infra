"""在分配模型前估算 Qwen3-32B 每个 TP rank 的最低 HBM 需求。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.memory_planner import estimate_qwen3_memory  # noqa: E402
from minigpt.qwen3 import Qwen3Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="估算 Qwen3 dense 推理每 rank HBM。")
    parser.add_argument("--config", default="configs/qwen3_32b_official.json")
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--device-memory-mb", type=float, default=65536.0)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--workspace-memory-mb", type=float, default=1024.0)
    parser.add_argument("--runtime-reserve-mb", type=float, default=2048.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = Qwen3Config.from_json(config_path)
    estimates = [
        estimate_qwen3_memory(
            config,
            tensor_parallel_size=tp,
            batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length,
            device_memory_mb=args.device_memory_mb,
            dtype_bytes=args.dtype_bytes,
            workspace_memory_mb=args.workspace_memory_mb,
            runtime_reserve_mb=args.runtime_reserve_mb,
        ).to_dict()
        for tp in args.tp
    ]
    report = {
        "scope": "static_estimate_not_measured_memory",
        "weight_sharding_assumption": (
            "vocab/LM-head sharded; Q and FFN column/row parallel; RMSNorm replicated; "
            "KV heads replicated when TP exceeds num_key_value_heads"
        ),
        "config": str(config_path),
        "estimates": estimates,
    }
    if args.output:
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TP  weights_MiB  KV_MiB  total_MiB  util  fits")
    for row in estimates:
        print(
            f"{row['tensor_parallel_size']:>2}  "
            f"{row['weight_memory_mb']:>11.1f}  "
            f"{row['kv_cache_memory_mb']:>6.1f}  "
            f"{row['estimated_total_mb']:>9.1f}  "
            f"{row['estimated_utilization']:>5.1%}  "
            f"{str(row['fits']):>5}"
        )
    print("注意：这是运行前静态估算，不是实测峰值显存。")


if __name__ == "__main__":
    main()
