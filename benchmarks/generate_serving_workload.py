"""生成 v0.7 三类可重放 Continuous Batching workload。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.workload import WORKLOAD_PRESETS, generate_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=WORKLOAD_PRESETS, required=True)
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--arrival-interval-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = generate_workload(
        args.preset,
        request_count=args.request_count,
        arrival_interval_ms=args.arrival_interval_ms,
        seed=args.seed,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    trace.save(output)
    file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"workload        : {trace.workload_id}")
    print(f"requests        : {len(trace.requests)}")
    print(f"request SHA-256 : {trace.request_sha256}")
    print(f"file SHA-256    : {file_sha256}")
    print(f"output          : {output}")


if __name__ == "__main__":
    main()
