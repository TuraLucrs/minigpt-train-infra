"""v0.7.1 mixed open-loop 复现设计、历史引用与保守判因测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.mixed_repro import (  # noqa: E402
    FORMAL_SEQUENCE,
    _summarize_host_telemetry,
    classify_reproduction,
    load_reference_observations,
    summarize_mixed_reproduction,
    summarize_output_reproducibility,
)


def fake_session(
    position: int,
    layout_id: str,
    goodput: float,
    *,
    frequency_mhz: float = 1800.0,
    temperature_celsius: float = 40.0,
    output_variant: str = "a",
) -> dict[str, object]:
    runs = [
        {"goodput_requests_per_second": goodput * multiplier}
        for multiplier in (0.99, 1.0, 1.01)
    ]
    return {
        "session_id": f"session-{position:02d}-{layout_id}",
        "position": position,
        "layout_id": layout_id,
        "service": {
            "goodput_requests_per_second": {"median": goodput},
            "completed_requests_per_second": {"median": goodput},
            "runs": runs,
        },
        "initial_npu_state": {
            "clean": True,
            "overall": {
                "hbm_usage_percent": {"min": 4.0, "median": 4.0, "max": 4.0},
                "aicore_usage_percent": {
                    "min": 0.0,
                    "median": 0.0,
                    "max": 0.0,
                },
            },
        },
        "output_sha256_by_replica": {"0": f"{layout_id}-{output_variant}"},
        "output_work_shape_sha256_by_replica": {"0": f"{layout_id}-shape"},
        "telemetry": {
            "overall": {
                "aicore_current_frequency_mhz": {
                    "min": frequency_mhz,
                    "median": frequency_mhz,
                    "max": frequency_mhz,
                },
                "temperature_celsius": {
                    "min": temperature_celsius,
                    "median": temperature_celsius,
                    "max": temperature_celsius,
                },
                "power_watts": {"min": 220.0, "median": 230.0, "max": 240.0},
                "aicore_usage_percent": {
                    "min": 60.0,
                    "median": 70.0,
                    "max": 80.0,
                },
            }
        },
        "host_telemetry": {
            "overall": {
                "cpu_usage_percent": {
                    "min": 35.0,
                    "median": 40.0,
                    "max": 45.0,
                },
                "cpu_iowait_percent": {
                    "min": 0.0,
                    "median": 0.1,
                    "max": 0.2,
                },
                "load1": {"min": 3.0, "median": 4.0, "max": 5.0},
                "psi_cpu_some_avg10": {
                    "min": 0.0,
                    "median": 0.0,
                    "max": 0.1,
                },
                "psi_io_some_avg10": {
                    "min": 0.0,
                    "median": 0.0,
                    "max": 0.1,
                },
            }
        },
    }


def fake_references() -> dict[str, object]:
    return {
        "v0.7": {
            "layouts": {
                "tp8": {
                    "goodput_requests_per_second": 1.179,
                    "completed_requests_per_second": 2.433,
                    "initial_npu_state": {
                        "overall": {
                            "hbm_usage_percent": {"median": 39.5},
                            "aicore_usage_percent": {"median": 45.5},
                        }
                    },
                    "output_sha256_by_replica": {"0": "tp8-b"},
                    "runs": [
                        {"good_requests": 33},
                        {"good_requests": 31},
                        {"good_requests": 15},
                    ],
                },
                "4xtp2": {"goodput_requests_per_second": 5.004},
            }
        },
        "v0.7.1": {
            "layouts": {
                "tp8": {"goodput_requests_per_second": 4.005},
                "4xtp2": {"goodput_requests_per_second": 4.719},
            }
        },
    }


def check_reference_loader() -> None:
    references = load_reference_observations(
        ROOT
        / "artifacts/v0.7_qwen3_continuous_batching_acceptance/comparisons/"
        "mixed_open_loop.json",
        ROOT / "v0.7_ascend_evidence.tar.gz",
        ROOT / "v0.7.1_ascend_profiling_gate_evidence.tar.gz",
    )
    assert references["source_workload_sha256"] == (
        "f376a6dcbfc41c465c023864a8a676672c539e9ac1086a0dd76db51e3f5668f8"
    )
    assert references["v0.7"]["layouts"]["tp8"][
        "goodput_requests_per_second"
    ] == 1.1792534087782434
    assert references["v0.7.1"]["layouts"]["tp8"][
        "goodput_requests_per_second"
    ] == 4.00521534109121
    assert references["v0.7"]["layouts"]["tp8"]["within_session_cv"] > 0.25
    assert references["v0.7"]["layouts"]["tp8"]["telemetry"]["overall"][
        "aicore_current_frequency_mhz"
    ]["median"] == 1800.0
    assert references["v0.7"]["layouts"]["tp8"]["initial_npu_state"][
        "overall"
    ]["hbm_usage_percent"]["median"] == 39.5


def check_current_regime_classification() -> None:
    values = {
        "tp8": iter((3.96, 4.04, 4.01, 3.99)),
        "4xtp2": iter((4.68, 4.74, 4.70, 4.76)),
    }
    sessions = [
        fake_session(position, layout_id, next(values[layout_id]))
        for position, layout_id in enumerate(FORMAL_SEQUENCE, start=1)
    ]
    diagnosis = classify_reproduction(sessions, fake_references())
    assert diagnosis["status"] == "historical_v07_tp8_slowdown_not_reproduced"
    assert 1.15 < diagnosis["goodput_ratio_4xtp2_over_tp8"] < 1.25
    assert diagnosis["order_effect"] is False
    assert len(diagnosis["host_association_sessions"]) == len(FORMAL_SEQUENCE)
    assert diagnosis["historical_root_cause"]["status"] == (
        "historical_v07_npu_contention"
    )


def check_output_reproducibility_warning() -> None:
    sessions = [
        fake_session(
            position,
            layout_id,
            4.0,
            output_variant="b" if position == 4 else "a",
        )
        for position, layout_id in enumerate(FORMAL_SEQUENCE, start=1)
    ]
    output = summarize_output_reproducibility(sessions)
    assert output["tp8"]["bitwise_output_stable"] is False
    assert output["tp8"]["work_shape_stable"] is True
    assert output["4xtp2"]["bitwise_output_stable"] is True


def check_host_telemetry_summary() -> None:
    samples = []
    for timestamp_ns in (110, 120, 210, 220):
        samples.append(
            {
                "timestamp_unix_ns": timestamp_ns,
                "cpu_usage_percent": 40.0,
                "cpu_iowait_percent": 0.5,
                "load1": 4.0,
                "load5": 3.0,
                "load15": 2.0,
                "memory_available_mb": 1024.0,
                "psi_cpu_some_avg10": 0.1,
                "psi_memory_some_avg10": 0.0,
                "psi_io_some_avg10": 0.0,
            }
        )
    payload = {
        "schema_version": 1,
        "collector": "procfs_host",
        "complete": True,
        "samples": samples,
        "errors": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "host_telemetry.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        summary = _summarize_host_telemetry(
            path,
            run_intervals=((100, 150), (200, 250)),
            relative_to=root,
        )
    assert summary["complete"] is True
    assert summary["all_runs_covered"] is True
    assert summary["error_free"] is True
    assert summary["overall"]["cpu_usage_percent"]["count"] == 4


def check_order_effect_classification() -> None:
    values = {
        "tp8": iter((2.0, 2.2, 3.8, 4.0)),
        "4xtp2": iter((3.0, 3.2, 4.8, 5.0)),
    }
    sessions = [
        fake_session(
            position,
            layout_id,
            next(values[layout_id]),
            frequency_mhz=1500.0 + position * 30.0,
        )
        for position, layout_id in enumerate(FORMAL_SEQUENCE, start=1)
    ]
    diagnosis = classify_reproduction(sessions, fake_references())
    assert diagnosis["status"] == "order_or_machine_state_effect"
    assert diagnosis["order_effect"] is True


def check_real_mixed_reproduction_package() -> None:
    archive = ROOT / "v0.7.1_mixed_repro_evidence.tar.gz"
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(temporary, filter="data")
        root = Path(temporary) / "v071_mixed_repro_check"
        summary = summarize_mixed_reproduction(
            root,
            v07_comparison_path=(
                ROOT
                / "artifacts/v0.7_qwen3_continuous_batching_acceptance/"
                "comparisons/mixed_open_loop.json"
            ),
            v07_evidence_archive=ROOT / "v0.7_ascend_evidence.tar.gz",
            v071_evidence_archive=(
                ROOT / "v0.7.1_ascend_profiling_gate_evidence.tar.gz"
            ),
        )
    assert summary["complete"] is True
    assert summary["diagnosis"]["status"] == (
        "historical_v07_tp8_slowdown_not_reproduced"
    )
    assert summary["diagnosis"]["historical_root_cause"]["status"] == (
        "historical_v07_npu_contention"
    )
    assert summary["output_reproducibility"]["tp8"][
        "bitwise_output_stable"
    ] is False
    assert summary["output_reproducibility"]["tp8"]["work_shape_stable"] is True


def main() -> None:
    check_reference_loader()
    check_host_telemetry_summary()
    check_output_reproducibility_warning()
    check_current_regime_classification()
    check_order_effect_classification()
    if os.environ.get("MINIGPT_SKIP_FULL_MIXED_PACKAGE") != "1":
        check_real_mixed_reproduction_package()
    print("v0.7.1 mixed reproduction tests passed.")


if __name__ == "__main__":
    main()
