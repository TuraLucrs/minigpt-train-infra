"""汇总 v0.7.1 mixed TP8/4×TP2 的 ABBA+BAAB 复现检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.mixed_repro import (  # noqa: E402
    summarize_mixed_reproduction,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--v07-comparison",
        default=(
            "artifacts/v0.7_qwen3_continuous_batching_acceptance/"
            "comparisons/mixed_open_loop.json"
        ),
    )
    parser.add_argument(
        "--v07-evidence-archive",
        default="v0.7_ascend_evidence.tar.gz",
    )
    parser.add_argument(
        "--v071-evidence-archive",
        default="v0.7.1_ascend_profiling_gate_evidence.tar.gz",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    summary = summarize_mixed_reproduction(
        project_path(args.root),
        v07_comparison_path=project_path(args.v07_comparison),
        v07_evidence_archive=project_path(args.v07_evidence_archive),
        v071_evidence_archive=project_path(args.v071_evidence_archive),
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output = project_path(args.markdown_output)
    write_markdown_report(summary, markdown_output)
    print(f"complete       : {summary['complete']}")
    print(f"evidence class : {summary['evidence_class']}")
    diagnosis = summary.get("diagnosis")
    if isinstance(diagnosis, dict):
        print(f"status         : {diagnosis['status']}")
        print(
            "4xtp2 / tp8   : "
            f"{diagnosis['goodput_ratio_4xtp2_over_tp8']:.6f}x"
        )
        print(f"conclusion     : {diagnosis['conclusion']}")
    else:
        for reason in summary["incomplete_reasons"]:
            print(f"incomplete     : {reason}")
    print(f"json report    : {output}")
    print(f"run log        : {markdown_output}")
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
