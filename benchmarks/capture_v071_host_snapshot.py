"""采集不含进程名和环境秘密的 v0.7.1 复现 Host 状态快照。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _loadavg() -> dict[str, object] | None:
    value = _read_text(Path("/proc/loadavg"))
    if value is None:
        return None
    fields = value.split()
    running, total = fields[3].split("/", 1)
    return {
        "load1": float(fields[0]),
        "load5": float(fields[1]),
        "load15": float(fields[2]),
        "running_processes": int(running),
        "total_processes": int(total),
    }


def _meminfo() -> dict[str, int] | None:
    value = _read_text(Path("/proc/meminfo"))
    if value is None:
        return None
    wanted = {
        "MemTotal",
        "MemAvailable",
        "Cached",
        "Buffers",
        "SwapTotal",
        "SwapFree",
    }
    result: dict[str, int] = {}
    for line in value.splitlines():
        name, raw = line.split(":", 1)
        if name in wanted:
            result[f"{name}_kb"] = int(raw.strip().split()[0])
    return result


def _pressure() -> dict[str, str | None]:
    return {
        name: _read_text(Path("/proc/pressure") / name)
        for name in ("cpu", "memory", "io")
    }


def _cpu_stat() -> dict[str, int] | None:
    value = _read_text(Path("/proc/stat"))
    if value is None:
        return None
    first = value.splitlines()[0].split()
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    return {name: int(raw) for name, raw in zip(names, first[1:])}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    relevant_environment = {
        name: value
        for name, value in os.environ.items()
        if name in {
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "PYTORCH_NPU_ALLOC_CONF",
            "TASK_QUEUE_ENABLE",
            "OMP_NUM_THREADS",
        }
        or name.startswith("HCCL_")
    }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix_ns": time.time_ns(),
        "phase": args.phase,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "loadavg": _loadavg(),
        "meminfo": _meminfo(),
        "pressure": _pressure(),
        "cpu_stat": _cpu_stat(),
        "environment": relevant_environment,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty_tracked": bool(
                _git("status", "--porcelain", "--untracked-files=no")
            ),
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"host snapshot: {output}")


if __name__ == "__main__":
    main()
