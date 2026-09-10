"""Verify saved matrix artifacts and compare available CUDA/Ascend observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.experiment_matrix import summarize_matrices, write_matrix_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="matrix output directory; repeat for cross-backend comparisons")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--require-complete", action="store_true", help="exit nonzero for missing/failed/incomparable points")
    parser.add_argument("--require-formal", action="store_true", help="require complete clean-commit Qwen3-32B hardware evidence")
    args = parser.parse_args()
    summary = summarize_matrices(args.input)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_matrix_markdown(summary, Path(args.markdown).resolve() if args.markdown else output.with_suffix(".md"))
    print(f"complete={summary['complete']}; verified matrices={len(summary['matrices'])}; report={output}")
    invalid_formal = args.require_formal and not all(matrix["formal_performance_evidence"] for matrix in summary["matrices"])
    return 2 if (args.require_complete and not summary["complete"]) or invalid_formal else 0


if __name__ == "__main__":
    raise SystemExit(main())
