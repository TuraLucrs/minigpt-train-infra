"""汇总三类 workload × open/closed loop 的 v0.7 最终验收。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.serving_acceptance import (  # noqa: E402
    load_layout_comparison,
    summarize_v07_acceptance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="可重复；格式 workload_class/mode=layout_comparison.json",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    comparisons: dict[
        tuple[str, str],
        tuple[Path, dict[str, object]],
    ] = {}
    for value in args.comparison:
        if "=" not in value or "/" not in value.split("=", 1)[0]:
            raise ValueError(
                "--comparison 必须使用 workload_class/mode=路径"
            )
        raw_key, raw_path = value.split("=", 1)
        workload_class, mode = raw_key.split("/", 1)
        key = (workload_class, mode)
        if not all(key) or not raw_path:
            raise ValueError("--comparison 的 workload、mode、路径不能为空")
        if key in comparisons:
            raise ValueError(f"重复 comparison：{workload_class}/{mode}")
        path = project_path(raw_path)
        comparisons[key] = (path, load_layout_comparison(path))

    summary = summarize_v07_acceptance(comparisons)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for comparison in summary["comparisons"]:
        comparison["path"] = os.path.relpath(
            Path(str(comparison["path"])),
            start=output.parent,
        )
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"evidence class : {summary['evidence_class']}")
    for reason in summary["incomplete_reasons"]:
        print(f"incomplete     : {reason}")
    for comparison in summary["comparisons"]:
        best = comparison["best_layout"]
        print(
            f"{comparison['workload_class']}/{comparison['mode']}: "
            f"best={best['layout_id']} goodput="
            f"{best['goodput_requests_per_second']:.3f} request/s"
        )
    print(f"report         : {output}")


if __name__ == "__main__":
    main()
