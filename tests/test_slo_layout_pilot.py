from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile

from minigpt.slo_layout_pilot import (
    build_session_plan,
    classify_layout_observations,
    load_pilot_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def expect_error(callback, text: str) -> None:
    try:
        callback()
    except ValueError as exc:
        assert text in str(exc)
    else:
        raise AssertionError("expected ValueError")


def main() -> None:
    config = load_pilot_config(PROJECT_ROOT / "configs" / "slo_layout_pilot_a3.json")
    plan = build_session_plan(config)
    assert len(plan) == 12
    assert [row["layout_id"] for row in plan[:6]] == [
        "tp8", "2xtp4", "4xtp2", "4xtp2", "2xtp4", "tp8"
    ]
    assert set(row["scenario_id"] for row in plan) == {
        "short_strict_closed", "mixed_standard_open"
    }
    assert len({row["session_id"] for row in plan}) == 12

    values = {
        "short": {"tp8": [8.0, 8.1], "2xtp4": [10.0, 10.1], "4xtp2": [7.0, 7.1]},
        "mixed": {"tp8": [4.0, 4.1], "2xtp4": [5.0, 5.1], "4xtp2": [7.0, 7.1]},
    }
    result = classify_layout_observations(values, max_session_cv=0.10, min_winner_margin=0.05)
    assert result["status"] == "layout_flip_supported"
    assert result["layout_flip_observed"] is True
    assert result["scenarios"]["short"]["best_layout"] == "2xtp4"
    assert result["scenarios"]["mixed"]["best_layout"] == "4xtp2"
    assert result["oracle_advantage_over_best_fixed"] > 0.0

    unstable = deepcopy(values)
    unstable["short"]["tp8"] = [1.0, 3.0]
    result = classify_layout_observations(unstable, max_session_cv=0.10, min_winner_margin=0.05)
    assert result["status"] == "inconclusive_session_variation"

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "invalid.json"
        invalid = deepcopy(config)
        invalid["round_orders"][1] = ["tp8", "2xtp4", "4xtp2"]
        import json
        path.write_text(json.dumps(invalid), encoding="utf-8")
        expect_error(lambda: load_pilot_config(path), "counterbalanced")

    print("test_slo_layout_pilot: OK")


if __name__ == "__main__":
    main()
