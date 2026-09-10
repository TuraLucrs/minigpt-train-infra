"""校验并汇总一个 v0.7.1 Ascend Profiler layout。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.profiling_gate import summarize_profile_point  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--profile-manifest", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    summary = summarize_profile_point(
        project_path(args.layout_manifest),
        (
            None
            if args.profile_manifest is None
            else project_path(args.profile_manifest)
        ),
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print("v0.7.1 Ascend Profile Summary")
    print(f"case           : {summary['workload_class']}/{summary['mode']}")
    print(f"layout         : {summary['layout_id']}")
    print(f"evidence class : {summary['evidence_class']}")
    for reason in summary["incomplete_reasons"]:
        print(f"incomplete     : {reason}")
    print(f"report         : {output}")


if __name__ == "__main__":
    main()
