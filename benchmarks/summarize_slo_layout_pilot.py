"""Validate and summarize the A3 SLO-aware TP/replica layout pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.slo_layout_pilot import summarize_pilot, write_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/slo_layout_pilot_a3.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--require-selection-ready", action="store_true")
    args = parser.parse_args()
    summary = summarize_pilot(args.root, args.config)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown_report(summary, Path(args.markdown).resolve())
    print(f"complete={summary['complete']}")
    print(f"selection_ready={summary['selection_ready']}")
    print(f"status={summary['classification']['status']}")
    print(f"report={output}")
    return 2 if args.require_selection_ready and not summary["selection_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
