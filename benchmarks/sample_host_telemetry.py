"""周期采集无进程名、无秘密的 procfs Host 状态，供复现检查对齐 measured runs。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=float, default=500.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_cpu_ticks() -> tuple[int, int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:9]]
    total = sum(values)
    idle = values[3]
    iowait = values[4]
    return total, idle, iowait


def _read_loadavg() -> tuple[float, float, float]:
    fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    return float(fields[0]), float(fields[1]), float(fields[2])


def _read_memory_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    raise ValueError("/proc/meminfo 缺少 MemAvailable")


def _read_pressure_avg10(resource: str) -> float:
    path = Path("/proc/pressure") / resource
    if not path.is_file():
        return 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("some "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            return float(fields["avg10"])
    return 0.0


def _write(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.interval_ms <= 0.0:
        raise ValueError("interval-ms 必须大于 0")
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    stopped = threading.Event()
    termination_reason = "unknown"

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal termination_reason
        termination_reason = signal.Signals(signum).name.lower()
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    report: dict[str, object] = {
        "schema_version": 1,
        "collector": "procfs_host",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_interval_ms": args.interval_ms,
        "cpu_count": os.cpu_count(),
        "samples": [],
        "errors": [],
        "complete": False,
        "termination_reason": None,
    }
    previous_total, previous_idle, previous_iowait = _read_cpu_ticks()
    while not stopped.wait(args.interval_ms / 1000.0):
        try:
            total, idle, iowait = _read_cpu_ticks()
            delta_total = total - previous_total
            delta_idle = idle - previous_idle
            delta_iowait = iowait - previous_iowait
            if delta_total <= 0:
                raise ValueError("/proc/stat CPU tick 没有前进")
            load1, load5, load15 = _read_loadavg()
            report["samples"].append(
                {
                    "timestamp_unix_ns": time.time_ns(),
                    "cpu_usage_percent": 100.0
                    * (delta_total - delta_idle - delta_iowait)
                    / delta_total,
                    "cpu_iowait_percent": 100.0 * delta_iowait / delta_total,
                    "load1": load1,
                    "load5": load5,
                    "load15": load15,
                    "memory_available_mb": _read_memory_available_mb(),
                    "psi_cpu_some_avg10": _read_pressure_avg10("cpu"),
                    "psi_memory_some_avg10": _read_pressure_avg10("memory"),
                    "psi_io_some_avg10": _read_pressure_avg10("io"),
                }
            )
            previous_total, previous_idle, previous_iowait = total, idle, iowait
            _write(output, report)
        except Exception as exc:
            report["errors"].append(
                {
                    "timestamp_unix_ns": time.time_ns(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write(output, report)

    report["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["complete"] = True
    report["termination_reason"] = termination_reason
    _write(output, report)
    print(f"host telemetry samples: {len(report['samples'])}")
    print(f"host telemetry errors : {len(report['errors'])}")
    print(f"host telemetry report : {output}")
    if not report["samples"] or report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
