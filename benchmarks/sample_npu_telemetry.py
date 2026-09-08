"""周期采集 npu-smi usages，供 v0.7 服务基准按 measured run 对齐。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.serving_telemetry import (  # noqa: E402
    TELEMETRY_SCHEMA_VERSION,
    NpuTelemetryTarget,
    parse_npu_target,
    sample_npu_smi_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        help="可重复；格式 logical_device_id=npu_id:chip_id",
    )
    parser.add_argument("--interval-ms", type=float, default=200.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument(
        "--query-type",
        choices=("common", "usages"),
        default="common",
        help="A3/训练服务器优先 common；确认支持指定 chip usages 时可切换",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_targets(values: list[str]) -> list[NpuTelemetryTarget]:
    targets = [parse_npu_target(value) for value in values]
    logical_ids = [target.logical_device_id for target in targets]
    physical_ids = [(target.npu_id, target.chip_id) for target in targets]
    if len(set(logical_ids)) != len(logical_ids):
        raise ValueError("--target 的 logical_device_id 不能重复")
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("--target 的 npu_id:chip_id 不能重复")
    return sorted(targets, key=lambda target: target.logical_device_id)


def main() -> None:
    args = parse_args()
    if args.interval_ms <= 0.0:
        raise ValueError("interval-ms 必须大于 0")
    if args.timeout_seconds <= 0.0:
        raise ValueError("timeout-seconds 必须大于 0")
    if args.duration_seconds is not None and args.duration_seconds <= 0.0:
        raise ValueError("duration-seconds 必须大于 0")
    targets = _validate_targets(args.target)
    output = project_path(args.output)
    stopped = threading.Event()
    termination_reason = "duration_elapsed"

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal termination_reason
        termination_reason = signal.Signals(signum).name.lower()
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    report: dict[str, object] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "collector": f"npu-smi_info_{args.query_type}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_interval_ms": args.interval_ms,
        "command_template": (
            "npu-smi info -t common -i <npu_id>"
            if args.query_type == "common"
            else "npu-smi info -t usages -i <npu_id> -c <chip_id>"
        ),
        "targets": [target.to_dict() for target in targets],
        "samples": [],
        "errors": [],
        "complete": False,
        "termination_reason": None,
    }
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        while not stopped.is_set():
            cycle_started = time.monotonic()
            futures = [
                executor.submit(
                    sample_npu_smi_target,
                    target,
                    executable=args.npu_smi,
                    query_type=args.query_type,
                    timeout_seconds=args.timeout_seconds,
                )
                for target in targets
            ]
            for target, future in zip(targets, futures):
                try:
                    report["samples"].append(future.result())
                except Exception as exc:
                    report["errors"].append(
                        {
                            "timestamp_unix_ns": time.time_ns(),
                            "logical_device_id": target.logical_device_id,
                            "npu_id": target.npu_id,
                            "chip_id": target.chip_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            _write_report(output, report)

            elapsed = time.monotonic() - started
            if args.duration_seconds is not None and elapsed >= args.duration_seconds:
                break
            remaining = args.interval_ms / 1000.0 - (
                time.monotonic() - cycle_started
            )
            if remaining > 0.0:
                stopped.wait(remaining)

    report["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["complete"] = True
    report["termination_reason"] = termination_reason
    _write_report(output, report)
    samples = report["samples"]
    errors = report["errors"]
    print(f"telemetry samples: {len(samples)}")
    print(f"telemetry errors : {len(errors)}")
    print(f"telemetry report : {output}")
    if not samples or errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
