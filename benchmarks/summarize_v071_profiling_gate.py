"""从六个 layout manifests 重算 v0.7.1 Profiling Gate。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.profiling_gate import (  # noqa: E402
    EXPECTED_CASES,
    summarize_profile_point,
    summarize_profiling_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="格式：case_id/layout_id=layout_manifest.json；profile manifest 必须同目录",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    points: dict[tuple[str, str], dict[str, object]] = {}
    for value in args.point:
        if "=" not in value or "/" not in value.split("=", 1)[0]:
            raise ValueError("--point 必须使用 case_id/layout_id=路径")
        identity, raw_path = value.split("=", 1)
        case_id, layout_id = identity.split("/", 1)
        if case_id not in EXPECTED_CASES or not layout_id or not raw_path:
            raise ValueError(f"未知或不完整的 --point：{identity}")
        key = (case_id, layout_id)
        if key in points:
            raise ValueError(f"重复 --point：{identity}")
        points[key] = summarize_profile_point(project_path(raw_path))
    summary = summarize_profiling_gate(points)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print("v0.7.1 Ascend Profiling Gate")
    print(f"evidence class : {summary['evidence_class']}")
    print(f"selection ready: {summary['selection_ready']}")
    for name, signal in summary["signals"].items():
        print(f"{name:<34}: {signal['triggered']}")
    for reason in summary["incomplete_reasons"]:
        print(f"incomplete     : {reason}")
    print(f"report         : {output}")


if __name__ == "__main__":
    main()
