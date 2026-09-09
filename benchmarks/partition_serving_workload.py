"""按 least projected load 为多个独立 TP replicas 生成 trace 分片。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.replica import partition_workload_by_projected_load  # noqa: E402
from minigpt.workload import WorkloadTrace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--replica-count", type=int, required=True)
    parser.add_argument("--max-slots-per-replica", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    workload_path = project_path(args.workload)
    source_file_sha256 = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    trace = WorkloadTrace.load(workload_path)
    partitions, assignments = partition_workload_by_projected_load(
        trace,
        replica_count=args.replica_count,
        max_slots_per_replica=args.max_slots_per_replica,
    )
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_files: list[str] = []
    for partition in partitions:
        index = partition.partition.replica_index
        path = output_dir / f"replica-{index:02d}.json"
        partition.save(path)
        partition_files.append(path.name)
    manifest = {
        "schema_version": 1,
        "method": "least_projected_character_plus_output_work",
        "source_workload_id": trace.workload_id,
        "source_request_sha256": trace.request_sha256,
        "source_file_sha256": source_file_sha256,
        "replica_count": args.replica_count,
        "max_slots_per_replica": args.max_slots_per_replica,
        "partition_files": partition_files,
        "assignments": assignments,
    }
    manifest_path = output_dir / "partition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"source SHA-256 : {trace.request_sha256}")
    print(f"file SHA-256   : {source_file_sha256}")
    print(f"replicas       : {args.replica_count}")
    print(f"manifest       : {manifest_path}")


if __name__ == "__main__":
    main()
