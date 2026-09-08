"""汇总 v0.7 TP8、2×TP4、4×TP2 多副本正式报告。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from minigpt.serving_layout import (  # noqa: E402
    load_layout_manifest,
    load_serving_report,
    summarize_serving_layouts,
)
from minigpt.serving_telemetry import load_telemetry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="格式为 layout_id=报告路径；同一 layout 可重复传入",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="推荐；可重复传入每个 layout 的 layout_manifest.json",
    )
    parser.add_argument("--baseline-layout", required=True)
    parser.add_argument("--max-start-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--telemetry",
        action="append",
        default=[],
        help="格式为 layout_id=telemetry 路径；每个 layout 恰好一份",
    )
    parser.add_argument(
        "--min-telemetry-samples-per-device-per-run",
        type=int,
        default=2,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_artifact_paths_portable(
    summary: dict[str, object],
    *,
    output_dir: Path,
) -> None:
    """将可搬迁证据引用写成相对 comparison 文件的路径。"""

    for row in summary["rows"]:
        for field in ("layout_manifest", "source_workload"):
            artifact = row.get(field)
            if isinstance(artifact, dict) and artifact.get("path"):
                artifact["path"] = os.path.relpath(
                    Path(str(artifact["path"])),
                    start=output_dir,
                )
        telemetry = row.get("telemetry")
        if isinstance(telemetry, dict):
            artifact = telemetry.get("source_artifact")
            if isinstance(artifact, dict) and artifact.get("path"):
                artifact["path"] = os.path.relpath(
                    Path(str(artifact["path"])),
                    start=output_dir,
                )
        reports = row.get("reports")
        if isinstance(reports, list):
            row["reports"] = [
                os.path.relpath(Path(str(path)), start=output_dir)
                for path in reports
            ]


def main() -> None:
    args = parse_args()
    if not args.report and not args.manifest:
        raise ValueError("至少提供一个 --manifest 或 --report")
    layouts: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    manifest_layouts: set[str] = set()
    for raw_path in args.manifest:
        path = project_path(raw_path)
        layout_id, reports = load_layout_manifest(path)
        if layout_id in layouts:
            raise ValueError(f"layout {layout_id!r} 重复提供 manifest/report")
        layouts[layout_id] = reports
        manifest_layouts.add(layout_id)
    for value in args.report:
        if "=" not in value:
            raise ValueError("--report 必须使用 layout_id=路径")
        layout_id, raw_path = value.split("=", 1)
        if not layout_id or not raw_path:
            raise ValueError("--report 的 layout_id 和路径都不能为空")
        path = project_path(raw_path)
        if layout_id in manifest_layouts:
            raise ValueError(f"layout {layout_id!r} 不能混用 manifest 和 report")
        layouts.setdefault(layout_id, []).append((path, load_serving_report(path)))
    telemetry_by_layout: dict[str, dict[str, object]] = {}
    for value in args.telemetry:
        if "=" not in value:
            raise ValueError("--telemetry 必须使用 layout_id=路径")
        layout_id, raw_path = value.split("=", 1)
        if not layout_id or not raw_path:
            raise ValueError("--telemetry 的 layout_id 和路径都不能为空")
        if layout_id in telemetry_by_layout:
            raise ValueError(f"layout {layout_id!r} 重复提供 telemetry")
        telemetry_by_layout[layout_id] = load_telemetry(project_path(raw_path))
    summary = summarize_serving_layouts(
        layouts,
        baseline_layout=args.baseline_layout,
        max_start_skew_ms=args.max_start_skew_ms,
        telemetry_by_layout=telemetry_by_layout,
        min_telemetry_samples_per_device_per_run=(
            args.min_telemetry_samples_per_device_per_run
        ),
    )
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_artifact_paths_portable(summary, output_dir=output.parent)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=" * 80)
    print("Qwen3 Continuous Batching Replica Layout Comparison")
    print(f"evidence class : {summary['evidence_class']}")
    if summary["incomplete_reasons"]:
        for reason in summary["incomplete_reasons"]:
            print(f"incomplete     : {reason}")
    for row in summary["rows"]:
        print(
            f"{row['layout_id']:<12} replicas={row['replica_count']} "
            f"TP={row['tp_size']} "
            f"request/s={row['summary']['completed_requests_per_second']['median']:.3f} "
            f"goodput={row['summary']['goodput_requests_per_second']['median']:.3f}"
        )
    print(f"report         : {output}")


if __name__ == "__main__":
    main()
