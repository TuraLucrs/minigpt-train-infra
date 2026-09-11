"""Evidence-bound A3 pilot for SLO-aware TP/replica layout selection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .decode_critical_path import summarize_npu_preflight
from .serving_layout import load_layout_manifest, summarize_serving_layouts
from .serving_telemetry import load_telemetry


SCHEMA_VERSION = 1
FORMAL_LAYOUTS = {
    "tp8": (1, 8, 32, 128),
    "2xtp4": (2, 4, 16, 64),
    "4xtp2": (4, 2, 8, 32),
}
FORMAL_ROUND_ORDERS = (
    ("tp8", "2xtp4", "4xtp2"),
    ("4xtp2", "2xtp4", "tp8"),
)


def _identifier(value: object, label: str) -> str:
    text = str(value)
    if not text or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in text):
        raise ValueError(f"{label} must be a lowercase filename-safe identifier")
    return text


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported SLO layout pilot schema")
    required = {
        "schema_version", "pilot_id", "hardware_family", "global_world_size",
        "physical_card_count", "chips_per_card", "warmup", "repeats",
        "max_seq_len", "max_start_skew_ms",
        "min_telemetry_samples_per_device_per_run", "max_session_cv",
        "min_winner_margin", "layouts", "round_orders", "scenarios",
    }
    allowed = required | {"description"}
    if required - set(raw) or set(raw) - allowed:
        raise ValueError(
            f"invalid pilot fields: missing={sorted(required - set(raw))}, "
            f"unknown={sorted(set(raw) - allowed)}"
        )
    config = json.loads(json.dumps(raw))
    _identifier(config["pilot_id"], "pilot_id")
    if config["hardware_family"] != "ascend_a3":
        raise ValueError("this pilot is restricted to Ascend A3")
    integer_contract = {
        "global_world_size": 8,
        "physical_card_count": 4,
        "chips_per_card": 2,
        "warmup": 1,
        "repeats": 3,
        "max_seq_len": 4096,
    }
    for field, expected in integer_contract.items():
        if type(config[field]) is not int or config[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    if (
        type(config["min_telemetry_samples_per_device_per_run"]) is not int
        or config["min_telemetry_samples_per_device_per_run"] < 2
    ):
        raise ValueError("telemetry coverage must require at least two samples")
    _positive(config["max_start_skew_ms"], "max_start_skew_ms")
    _positive(config["max_session_cv"], "max_session_cv")
    _positive(config["min_winner_margin"], "min_winner_margin")

    layouts = config["layouts"]
    if not isinstance(layouts, list) or len(layouts) != 3:
        raise ValueError("pilot requires exactly three layouts")
    actual_layouts: dict[str, tuple[int, int, int, int]] = {}
    for row in layouts:
        if not isinstance(row, dict) or set(row) != {
            "id", "tp_size", "replica_count", "per_replica_max_slots",
            "per_replica_max_queue_size",
        }:
            raise ValueError("invalid layout entry")
        layout_id = _identifier(row["id"], "layout.id")
        actual_layouts[layout_id] = (
            int(row["replica_count"]), int(row["tp_size"]),
            int(row["per_replica_max_slots"]),
            int(row["per_replica_max_queue_size"]),
        )
    if actual_layouts != FORMAL_LAYOUTS:
        raise ValueError("pilot layouts must be TP8, 2xTP4, and 4xTP2 with equal global capacity")
    if tuple(tuple(order) for order in config["round_orders"]) != FORMAL_ROUND_ORDERS:
        raise ValueError("pilot requires the declared forward/reverse counterbalanced orders")

    scenarios = config["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 2:
        raise ValueError("pilot requires exactly two workload/SLO operating points")
    scenario_ids: set[str] = set()
    workload_classes: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {
            "id", "workload_class", "mode", "closed_loop_clients",
            "deterministic_open_loop", "slo",
        }:
            raise ValueError("invalid scenario entry")
        scenario_id = _identifier(scenario["id"], "scenario.id")
        if scenario_id in scenario_ids:
            raise ValueError("duplicate scenario id")
        scenario_ids.add(scenario_id)
        workload_class = str(scenario["workload_class"])
        if workload_class not in {"short_short", "mixed"}:
            raise ValueError("pilot only accepts the frozen short_short and mixed workloads")
        workload_classes.add(workload_class)
        mode = scenario["mode"]
        clients = scenario["closed_loop_clients"]
        scripted = scenario["deterministic_open_loop"]
        if mode == "closed_loop":
            if type(clients) is not int or clients < 4 or scripted is not False:
                raise ValueError("closed-loop scenario requires >=4 clients and no open-loop scripting")
        elif mode == "open_loop":
            if clients is not None or scripted is not True:
                raise ValueError("open-loop scenario requires deterministic replay and no client count")
        else:
            raise ValueError("scenario mode must be open_loop or closed_loop")
        slo = scenario["slo"]
        if not isinstance(slo, dict) or set(slo) != {"ttft_ms", "tpot_ms", "e2e_ms"}:
            raise ValueError("scenario SLO requires ttft_ms/tpot_ms/e2e_ms")
        for field, value in slo.items():
            _positive(value, f"scenario.slo.{field}")
    if workload_classes != {"short_short", "mixed"}:
        raise ValueError("pilot must cover both short_short and mixed")
    return config


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_session_plan(config: Mapping[str, Any]) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    position = 0
    for scenario in config["scenarios"]:
        for round_index, order in enumerate(config["round_orders"], start=1):
            for order_index, layout_id in enumerate(order, start=1):
                position += 1
                plan.append({
                    "position": position,
                    "scenario_id": scenario["id"],
                    "round": round_index,
                    "order_in_round": order_index,
                    "layout_id": layout_id,
                    "session_id": f"session-{position:02d}-{layout_id}",
                })
    return plan


def coefficient_of_variation(values: Sequence[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("CV requires at least one value")
    mean = statistics.fmean(numbers)
    if mean == 0.0:
        return 0.0 if all(value == 0.0 for value in numbers) else math.inf
    return statistics.pstdev(numbers) / abs(mean)


def classify_layout_observations(
    scenario_layout_values: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    max_session_cv: float,
    min_winner_margin: float,
) -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for scenario_id, layouts in scenario_layout_values.items():
        if set(layouts) != set(FORMAL_LAYOUTS):
            raise ValueError(f"{scenario_id} does not contain all three layouts")
        aggregates: dict[str, object] = {}
        for layout_id, raw_values in layouts.items():
            values = [float(value) for value in raw_values]
            if len(values) != 2 or any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("each layout requires two finite nonnegative session values")
            median = statistics.median(values)
            session_cv = coefficient_of_variation(values)
            aggregates[layout_id] = {
                "session_values": values,
                "median": median,
                "session_cv": session_cv,
                "stable": session_cv <= max_session_cv,
            }
        ordered = sorted(
            aggregates.items(), key=lambda item: float(item[1]["median"]), reverse=True
        )
        winner_value = float(ordered[0][1]["median"])
        runner_up_value = float(ordered[1][1]["median"])
        margin = (
            0.0 if winner_value == 0.0
            else (winner_value - runner_up_value) / winner_value
        )
        scenarios[scenario_id] = {
            "layouts": aggregates,
            "best_layout": ordered[0][0],
            "winner_margin": margin,
            "winner_clear": margin >= min_winner_margin,
            "stable": all(bool(row["stable"]) for row in aggregates.values()),
        }

    stable = all(bool(row["stable"]) for row in scenarios.values())
    winners = {str(row["best_layout"]) for row in scenarios.values()}
    clear = all(bool(row["winner_clear"]) for row in scenarios.values())
    if not stable:
        status = "inconclusive_session_variation"
    elif len(winners) > 1 and clear:
        status = "layout_flip_supported"
    elif len(winners) > 1:
        status = "layout_flip_weak_margin"
    elif not clear:
        status = "same_layout_weak_margin"
    else:
        status = "same_layout_wins_tested_points"

    normalized_scores: dict[str, float] = {}
    for layout_id in FORMAL_LAYOUTS:
        ratios = []
        for scenario in scenarios.values():
            best = max(float(row["median"]) for row in scenario["layouts"].values())
            value = float(scenario["layouts"][layout_id]["median"])
            ratios.append(0.0 if best == 0.0 else value / best)
        normalized_scores[layout_id] = math.prod(ratios) ** (1.0 / len(ratios))
    best_fixed_layout = max(normalized_scores, key=normalized_scores.get)
    best_fixed_score = normalized_scores[best_fixed_layout]
    return {
        "status": status,
        "layout_flip_observed": len(winners) > 1,
        "all_sessions_stable": stable,
        "all_winner_margins_clear": clear,
        "scenarios": scenarios,
        "best_fixed_layout": best_fixed_layout,
        "best_fixed_normalized_score": best_fixed_score,
        "oracle_advantage_over_best_fixed": (
            None if best_fixed_score == 0.0 else 1.0 / best_fixed_score - 1.0
        ),
    }


def _artifact(path: Path, *, relative_to: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _verify_session_marker(
    root: Path,
    session_dir: Path,
    expected: Mapping[str, object],
    *,
    config_sha256: str,
) -> dict[str, object]:
    marker_path = session_dir / "session_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or marker.get("schema_version") != 1:
        raise ValueError(f"invalid session marker: {marker_path}")
    if marker.get("config_sha256") != config_sha256:
        raise ValueError(f"session marker config changed: {marker_path}")
    for field in (
        "position", "scenario_id", "round", "order_in_round", "layout_id",
        "session_id",
    ):
        if marker.get(field) != expected.get(field):
            raise ValueError(f"session marker {field} differs from plan: {marker_path}")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"session marker has no artifacts: {marker_path}")
    required = {
        "layout_manifest.json", "source_workload.json", "device_preflight.json",
        "telemetry.json", "host_telemetry.json", "host_before.json",
        "host_after.json", "benchmark.log",
    }
    if not required.issubset(artifacts):
        raise ValueError(f"session marker misses required artifacts: {marker_path}")
    for label, raw in artifacts.items():
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"invalid marker artifact {label}: {marker_path}")
        artifact_path = (root / str(raw["path"])).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(f"marker artifact escapes pilot root: {marker_path}")
        payload = artifact_path.read_bytes()
        if len(payload) != int(raw["size_bytes"]):
            raise ValueError(f"marker artifact size changed: {artifact_path}")
        if hashlib.sha256(payload).hexdigest() != raw["sha256"]:
            raise ValueError(f"marker artifact SHA-256 changed: {artifact_path}")
    return marker


def _validate_host_telemetry(
    path: Path,
    *,
    run_intervals: Sequence[tuple[int, int]],
) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("collector") != "procfs_host"
        or raw.get("complete") is not True
        or raw.get("errors") != []
        or not isinstance(raw.get("samples"), list)
    ):
        raise ValueError(f"invalid host telemetry: {path}")
    coverage = []
    for repeat, (started_at_ns, ended_at_ns) in enumerate(run_intervals):
        count = sum(
            isinstance(sample, dict)
            and started_at_ns <= int(sample.get("timestamp_unix_ns", -1)) <= ended_at_ns
            for sample in raw["samples"]
        )
        coverage.append({"repeat": repeat, "sample_count": count})
    if any(row["sample_count"] < 2 for row in coverage):
        raise ValueError(f"host telemetry does not cover every measured repeat: {path}")
    return {"complete": True, "error_free": True, "run_coverage": coverage}


def summarize_pilot(root: str | Path, config_path: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = load_pilot_config(config_path)
    config_sha256 = canonical_sha256(config)
    plan = build_session_plan(config)
    plan_by_key = {
        (str(row["scenario_id"]), int(row["round"]), str(row["layout_id"])): row
        for row in plan
    }
    session_rows: list[dict[str, object]] = []
    scenario_layout_values: dict[str, dict[str, list[float]]] = {
        str(scenario["id"]): {layout_id: [] for layout_id in FORMAL_LAYOUTS}
        for scenario in config["scenarios"]
    }
    identity: tuple[str, str] | None = None
    formal_rounds = True

    for scenario in config["scenarios"]:
        scenario_id = str(scenario["id"])
        for round_index in range(1, 3):
            layouts = {}
            telemetry = {}
            preflights = {}
            paths = {}
            for layout_id in FORMAL_LAYOUTS:
                planned = plan_by_key[(scenario_id, round_index, layout_id)]
                session_dir = root / scenario_id / f"round-{round_index:02d}" / str(planned["session_id"])
                _verify_session_marker(
                    root, session_dir, planned, config_sha256=config_sha256
                )
                manifest_path = session_dir / "layout_manifest.json"
                loaded_layout_id, reports = load_layout_manifest(manifest_path)
                if loaded_layout_id != layout_id:
                    raise ValueError(f"unexpected layout in {manifest_path}")
                layouts[layout_id] = reports
                telemetry[layout_id] = load_telemetry(session_dir / "telemetry.json")
                preflight = json.loads((session_dir / "device_preflight.json").read_text(encoding="utf-8"))
                preflight_summary = summarize_npu_preflight(
                    preflight, expected_logical_device_ids=range(8)
                )
                if not preflight_summary["clean"]:
                    raise ValueError(f"unclean device preflight in {session_dir}")
                preflights[layout_id] = preflight_summary
                paths[layout_id] = session_dir
            comparison = summarize_serving_layouts(
                layouts,
                baseline_layout="tp8",
                max_start_skew_ms=float(config["max_start_skew_ms"]),
                telemetry_by_layout=telemetry,
                min_telemetry_samples_per_device_per_run=int(
                    config["min_telemetry_samples_per_device_per_run"]
                ),
            )
            expected_protocol = {
                "mode": scenario["mode"],
                "open_loop_admission_scripted": scenario[
                    "deterministic_open_loop"
                ],
                "warmup": config["warmup"],
                "repeats": config["repeats"],
                "ttft_slo_ms": float(scenario["slo"]["ttft_ms"]),
                "tpot_slo_ms": float(scenario["slo"]["tpot_ms"]),
                "e2e_slo_ms": float(scenario["slo"]["e2e_ms"]),
                "global_closed_loop_clients": scenario[
                    "closed_loop_clients"
                ],
            }
            if comparison["protocol"] != expected_protocol:
                raise ValueError(
                    f"{scenario_id} round {round_index} protocol differs "
                    "from the preregistered config"
                )
            actual_order = tuple(
                str(row["layout_id"])
                for row in sorted(
                    comparison["rows"],
                    key=lambda row: min(
                        int(run["started_at_unix_ns"])
                        for run in row["runs"]
                    ),
                )
            )
            expected_order = tuple(config["round_orders"][round_index - 1])
            if actual_order != expected_order:
                raise ValueError(
                    f"{scenario_id} round {round_index} execution order "
                    f"{actual_order} differs from {expected_order}"
                )
            formal = comparison["evidence_class"] == "formal_qwen3_32b_tp8_vs_2xtp4_vs_4xtp2"
            formal_rounds = formal_rounds and formal
            current_identity = (
                canonical_sha256(comparison["model"]),
                canonical_sha256(comparison["execution"]),
            )
            if identity is None:
                identity = current_identity
            elif current_identity != identity:
                raise ValueError("pilot sessions changed model, code commit, or execution environment")
            for row in comparison["rows"]:
                layout_id = str(row["layout_id"])
                session_dir = paths[layout_id]
                goodput = float(row["summary"]["goodput_requests_per_second"]["median"])
                within_cv = coefficient_of_variation(
                    [float(run["goodput_requests_per_second"]) for run in row["runs"]]
                )
                scenario_layout_values[scenario_id][layout_id].append(goodput)
                session_rows.append({
                    **plan_by_key[(scenario_id, round_index, layout_id)],
                    "formal_layout_comparison": formal,
                    "preflight": preflights[layout_id],
                    "within_session_goodput_cv": within_cv,
                    "host_telemetry": _validate_host_telemetry(
                        session_dir / "host_telemetry.json",
                        run_intervals=[
                            (
                                int(run["started_at_unix_ns"]),
                                int(run["ended_at_unix_ns"]),
                            )
                            for run in row["runs"]
                        ],
                    ),
                    "metrics": {
                        field: row["summary"][field]
                        for field in (
                            "completed_requests_per_second",
                            "goodput_requests_per_second",
                            "output_tokens_per_second",
                            "queue_ms", "ttft_ms", "tpot_ms", "e2e_latency_ms",
                        )
                    },
                    "max_rank_peak_device_memory_mb": row["max_rank_peak_device_memory_mb"],
                    "telemetry": row["telemetry"],
                    "artifacts": {
                        "layout_manifest": _artifact(session_dir / "layout_manifest.json", relative_to=root),
                        "device_preflight": _artifact(session_dir / "device_preflight.json", relative_to=root),
                        "telemetry": _artifact(session_dir / "telemetry.json", relative_to=root),
                        "host_telemetry": _artifact(session_dir / "host_telemetry.json", relative_to=root),
                        "session_complete": _artifact(session_dir / "session_complete.json", relative_to=root),
                    },
                })

    classification = classify_layout_observations(
        scenario_layout_values,
        max_session_cv=float(config["max_session_cv"]),
        min_winner_margin=float(config["min_winner_margin"]),
    )
    within_stable = all(
        float(row["within_session_goodput_cv"]) <= float(config["max_session_cv"])
        for row in session_rows
    )
    selection_ready = bool(
        formal_rounds
        and classification["all_sessions_stable"]
        and classification["all_winner_margins_clear"]
        and within_stable
    )
    if not within_stable:
        classification["status"] = "inconclusive_within_session_variation"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "slo_aware_tp_replica_layout_research_pilot",
        "evidence_class": (
            "formal_qwen3_32b_a3_research_pilot"
            if formal_rounds else "development_or_incomplete_research_pilot"
        ),
        "complete": len(session_rows) == 12,
        "selection_ready": selection_ready,
        "scope": (
            "Two predeclared operating points and two independent sessions per layout; "
            "sufficient to choose the next implementation candidate, not a paper-level evaluation."
        ),
        "config": _artifact(config_path, relative_to=config_path.parent.parent),
        "config_sha256": config_sha256,
        "session_count": len(session_rows),
        "sessions": session_rows,
        "classification": classification,
        "within_session_stable": within_stable,
    }


def write_markdown_report(summary: Mapping[str, object], path: str | Path) -> None:
    classification = summary["classification"]
    lines = [
        "# A3 SLO-aware TP / Replica Layout Pilot",
        "",
        f"- complete: `{str(summary['complete']).lower()}`",
        f"- selection_ready: `{str(summary['selection_ready']).lower()}`",
        f"- evidence_class: `{summary['evidence_class']}`",
        f"- status: `{classification['status']}`",
        f"- best fixed layout: `{classification['best_fixed_layout']}`",
        "",
        "| scenario | TP8 goodput | 2×TP4 goodput | 4×TP2 goodput | best | margin |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for scenario_id, scenario in classification["scenarios"].items():
        values = scenario["layouts"]
        lines.append(
            f"| {scenario_id} | {values['tp8']['median']:.4f} | "
            f"{values['2xtp4']['median']:.4f} | {values['4xtp2']['median']:.4f} | "
            f"{scenario['best_layout']} | {scenario['winner_margin']:.2%} |"
        )
    lines += [
        "",
        "`layout_flip_supported` only means the predeclared A3 pilot observed a stable, "
        "clear winner change. It does not by itself validate an online controller or generalize "
        "to other models, hardware, arrival rates, or SLOs.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
