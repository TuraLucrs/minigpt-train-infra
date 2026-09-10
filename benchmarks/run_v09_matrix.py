"""Run all or selected v0.9 benchmark points in independent real processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.experiment_matrix import (  # noqa: E402
    expand_matrix, load_matrix_config, run_matrix, summarize_matrices, write_matrix_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device-map", help="explicit logical/physical/chip mapping; required for accelerator runs")
    parser.add_argument("--interconnect-topology")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--case", action="append", default=[], help="case ID from --list-points; repeatable")
    parser.add_argument("--point", action="append", default=[], help="point ID from --list-points; repeatable")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="save a pending plan; no point is marked executed")
    parser.add_argument("--list-points", action="store_true", help="inspect complete matrix without model weights or hardware")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if args.list_points:
        plan = expand_matrix(load_matrix_config(config_path))
        for point in plan["points"]:
            print(f"{point['point_id']}  axis={point['axis']} TP={point['tp_size']} {point['entrypoint']}/{point['decode_mode']}")
        print(f"total: {len(plan['points'])} independent processes; cases: {len(plan['cases'])}")
        return 0
    if not args.model_dir or not args.output_dir:
        parser.error("--model-dir and --output-dir are required for plan/run")
    state = run_matrix(config_path, args.model_dir, args.output_dir, project_root=PROJECT_ROOT,
                       device_map_path=args.device_map, interconnect_topology=args.interconnect_topology,
                       python_executable=args.python, resume=args.resume, dry_run=args.dry_run,
                       selected_cases=args.case, selected_points=args.point)
    if args.dry_run:
        print("Plan saved with pending points. No benchmark or hardware result is claimed.")
        return 0
    summary = summarize_matrices([args.output_dir])
    output = Path(args.output_dir).resolve()
    (output / "matrix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_matrix_markdown(summary, output / "MATRIX_REPORT.md")
    selected = [
        point for point in state["plan"]["points"]
        if (not args.case or point["case_id"] in args.case) and (not args.point or point["point_id"] in args.point)
    ]
    verified = summary["matrices"][0]
    point_status = {point["point_id"]: point["status"] for point in verified["points"]}
    failed = [point["point_id"] for point in selected if point_status[point["point_id"]] != "succeeded"]
    selected_ids = {point["point_id"] for point in selected}
    invalid_cases = [case["case_id"] for case in verified["cases"]
                     if set(case["point_ids"]).issubset(selected_ids) and not case["complete"]]
    print(f"Selected points complete: {len(selected) - len(failed)}/{len(selected)}")
    print(f"Full matrix complete: {summary['complete']}; report: {output / 'MATRIX_REPORT.md'}")
    if invalid_cases:
        print("Selected A/B cases failed validation: " + ", ".join(invalid_cases))
    return 2 if failed or invalid_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
