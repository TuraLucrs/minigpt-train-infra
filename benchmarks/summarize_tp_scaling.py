"""合并 TP=2/4/8 等独立报告，计算相对最小 TP 基线的 Scaling。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Qwen3 TP benchmark scaling。")
    parser.add_argument(
        "--report", action="append", required=True, help="可重复传入 TP 报告"
    )
    parser.add_argument("--output", default="runs/qwen3_tp_scaling.json")
    return parser.parse_args()


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_report(path: Path) -> dict[str, object]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 TP 报告：{path}") from exc
    required = {
        "benchmark",
        "environment",
        "summary",
        "distributed",
        "provenance",
        "request",
        "model",
        "workload_fingerprint",
    }
    if not isinstance(report, dict) or not required.issubset(report):
        raise ValueError(f"不是完整 Qwen3 TP 报告：{path}")
    return report


def _median(report: dict[str, object], field: str) -> float | None:
    summary = report["summary"]
    if field not in summary:
        return None
    return float(summary[field]["median"])


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / max(denominator, 1e-12)


def _execution_fingerprint(report: dict[str, object]) -> dict[str, object]:
    environment = report["environment"]
    distributed = report["distributed"]
    return {
        "device_type": environment.get("device_type"),
        "device_name": environment.get("device_name"),
        "precision": environment.get("precision"),
        "torch": environment.get("torch"),
        "torch_npu": environment.get("torch_npu"),
        "total_memory_mb": environment.get("total_memory_mb"),
        "backend": distributed.get("backend"),
        "chips_per_card": distributed.get("chips_per_card"),
    }


def summarize_scaling(
    entries: list[tuple[Path, dict[str, object]]],
) -> dict[str, object]:
    if len(entries) < 2:
        raise ValueError("Scaling 至少需要两个不同 TP world_size 的报告")
    ordered = sorted(
        entries, key=lambda item: int(item[1]["distributed"]["world_size"])
    )
    world_sizes = [int(report["distributed"]["world_size"]) for _, report in ordered]
    if len(set(world_sizes)) != len(world_sizes):
        raise ValueError("每个 TP world_size 只能提供一份报告")

    first = ordered[0][1]
    for path, report in ordered[1:]:
        if report["benchmark"] != first["benchmark"]:
            raise ValueError(f"benchmark 类型不一致：{path}")
        if report["model"]["config"] != first["model"]["config"]:
            raise ValueError(f"模型 config 不一致：{path}")
        if report["request"] != first["request"]:
            raise ValueError(f"请求配置不一致：{path}")
        if _execution_fingerprint(report) != _execution_fingerprint(first):
            raise ValueError(f"设备、精度或软件环境不一致：{path}")
        if report["workload_fingerprint"] != first["workload_fingerprint"]:
            raise ValueError(f"prompt/tokenizer workload 指纹不一致：{path}")
        if (
            report["provenance"]["config_sha256"]
            != first["provenance"]["config_sha256"]
        ):
            raise ValueError(f"config SHA-256 不一致：{path}")

    baseline_world_size = world_sizes[0]
    baseline_latency = {
        field: _median(first, field)
        for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms")
    }
    baseline_throughput = _median(first, "output_tokens_per_second")
    baseline_peak = float(first["distributed"]["max_rank_median_request_peak_mb"])
    rows = []
    for path, report in ordered:
        world_size = int(report["distributed"]["world_size"])
        scale = world_size / baseline_world_size
        latency = {
            field: _median(report, field)
            for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms")
        }
        throughput = _median(report, "output_tokens_per_second")
        latency_scaling = {}
        for field, current in latency.items():
            speedup = _ratio(baseline_latency[field], current)
            latency_scaling[field] = {
                "median": current,
                "speedup_vs_baseline": speedup,
                "scaling_efficiency": None if speedup is None else speedup / scale,
            }
        throughput_speedup = _ratio(throughput, baseline_throughput)
        max_rank_peak = float(report["distributed"]["max_rank_median_request_peak_mb"])
        rows.append(
            {
                "report": str(path),
                "world_size": world_size,
                "relative_device_count": scale,
                "latency": latency_scaling,
                "throughput": {
                    "median_output_tokens_per_second": throughput,
                    "speedup_vs_baseline": throughput_speedup,
                    "scaling_efficiency": (
                        None
                        if throughput_speedup is None
                        else throughput_speedup / scale
                    ),
                },
                "memory": {
                    "max_rank_median_request_peak_mb": max_rank_peak,
                    "per_rank_reduction_vs_baseline": _ratio(
                        baseline_peak, max_rank_peak
                    ),
                    "sum_rank_median_request_peak_mb": float(
                        report["distributed"]["sum_rank_median_request_peak_mb"]
                    ),
                },
                "evidence_class": report.get("evidence_class"),
                "git_commit": report["provenance"]["git"]["commit"],
            }
        )

    all_formal = all(
        report.get("evidence_class") == "formal_qwen3_32b_tp_hashed"
        for _, report in ordered
    )
    same_weights = all(
        report["provenance"]["weights"] == first["provenance"]["weights"]
        for _, report in ordered[1:]
    )
    same_commit = (
        len({str(report["provenance"]["git"]["commit"]) for _, report in ordered}) == 1
    )
    evidence_class = (
        "formal_qwen3_32b_tp_scaling"
        if all_formal and same_weights and same_commit
        else "development_or_incomparable_tp_scaling"
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "qwen3_tensor_parallel_scaling",
        "evidence_class": evidence_class,
        "baseline_world_size": baseline_world_size,
        "world_sizes": world_sizes,
        "same_weight_manifest": same_weights,
        "same_git_commit": same_commit,
        "comparison_rule": (
            "latency speedup=baseline/current; throughput speedup=current/baseline; "
            "efficiency=speedup/(world_size/baseline_world_size)"
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    entries = [
        (project_path(path), _load_report(project_path(path))) for path in args.report
    ]
    summary = summarize_scaling(entries)
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 80)
    print("Qwen3 Tensor Parallel Scaling")
    print(f"evidence class : {summary['evidence_class']}")
    for row in summary["rows"]:
        throughput = row["throughput"]
        print(
            f"TP={row['world_size']:<2d} output_tok/s="
            f"{throughput['median_output_tokens_per_second']:.3f} "
            f"efficiency={throughput['scaling_efficiency']:.4f}"
        )
    print(f"报告            : {output}")


if __name__ == "__main__":
    main()
