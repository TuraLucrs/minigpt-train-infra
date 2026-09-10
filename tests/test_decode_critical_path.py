"""v0.8 Decode 词表通信 A/B 决策口径测试。"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minigpt.decode_critical_path import _compare_case  # noqa: E402


def fake_session(
    path: str,
    goodput: float,
    completed: float,
    tpot_ms: float,
    communication: float,
    free: float,
) -> dict[str, object]:
    return {
        "configured_path": path,
        "service": {
            "goodput_requests_per_second": {"median": goodput},
            "completed_requests_per_second": {"median": completed},
            "tpot_ms": {"median": tpot_ms},
        },
        "profile": {
            "communication_not_overlapped_fraction": communication,
            "free_fraction": free,
        },
    }


def paired_sessions(
    baseline: tuple[float, float, float, float, float],
    candidate: tuple[float, float, float, float, float],
) -> list[dict[str, object]]:
    return [
        fake_session("full_gather", *baseline),
        fake_session("distributed_argmax", *candidate),
        fake_session("distributed_argmax", *candidate),
        fake_session("full_gather", *baseline),
    ]


def check_supported_candidate() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (9.4, 10.5, 108.0, 0.27, 0.25),
        )
    )
    assert comparison["candidate_supported"] is True
    assert comparison["candidate_regressed"] is False
    assert comparison["completed_throughput_speedup"] == 1.05
    assert comparison["communication_fraction_absolute_reduction"] > 0.07


def check_candidate_without_end_to_end_gain() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (9.0, 10.1, 119.0, 0.20, 0.35),
        )
    )
    assert comparison["candidate_supported"] is False
    assert comparison["candidate_regressed"] is False


def check_regression() -> None:
    comparison = _compare_case(
        paired_sessions(
            (9.0, 10.0, 120.0, 0.35, 0.20),
            (8.5, 9.5, 130.0, 0.30, 0.25),
        )
    )
    assert comparison["candidate_supported"] is False
    assert comparison["candidate_regressed"] is True


def main() -> None:
    check_supported_candidate()
    check_candidate_without_end_to_end_gain()
    check_regression()
    print("v0.8 decode critical-path decision tests passed.")


if __name__ == "__main__":
    main()
