"""检查 v0.8 A/B 作业启动前八个 NPU logical device 是否空闲。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.mixed_repro import _summarize_initial_npu_state  # noqa: E402
from minigpt.serving_telemetry import load_telemetry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True)
    args = parser.parse_args()
    path = Path(args.telemetry)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    summary = _summarize_initial_npu_state(
        load_telemetry(path),
        expected_logical_device_ids=range(8),
    )
    hbm = summary["overall"]["hbm_usage_percent"]
    aicore = summary["overall"]["aicore_usage_percent"]
    print(f"preflight HBM max   : {hbm['max']:.3f}%")
    print(f"preflight AICore max: {aicore['max']:.3f}%")
    if not summary["clean"]:
        raise SystemExit("NPU preflight 非空闲，拒绝启动正式 A/B")


if __name__ == "__main__":
    main()
