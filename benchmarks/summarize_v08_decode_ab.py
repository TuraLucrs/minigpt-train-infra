"""汇总 v0.8 TP8 Decode full-gather / distributed-argmax A/B。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.decode_critical_path import (  # noqa: E402
    summarize_decode_ab,
    write_markdown_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    markdown = Path(args.markdown_output)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not markdown.is_absolute():
        markdown = PROJECT_ROOT / markdown
    summary = summarize_decode_ab(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(summary, markdown)
    print(f"complete: {summary['complete']}")
    print(f"status  : {summary['decision']['status']}")
    print(f"report  : {output}")
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
