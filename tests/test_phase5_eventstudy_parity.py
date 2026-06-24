from __future__ import annotations

import json
from pathlib import Path

import pytest

from contdid import (
    ContDIDSpec,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
    simulate_contdid_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_FIXTURE = REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase5_eventstudy_expected.json"

_POINT_ESTIMATE_ATOL = 0.08
_SUMMARY_ATOL = 0.05


def _load_fixture() -> dict:
    return json.loads(PACKAGE_FIXTURE.read_text(encoding="utf-8"))


def test_phase5_eventstudy_parity_matches_fixture_targets() -> None:
    fixture = _load_fixture()
    panel = simulate_contdid_data(
        n=16000,
        dgp_id=fixture["scenario_id"],
        seed=fixture["default_seed"],
    )

    level_result = estimate_eventstudy_effects(
        panel,
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        degree=1,
    )
    slope_result = estimate_eventstudy_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        degree=1,
    )

    assert level_result.event_time == fixture["event_time_grid"]
    assert slope_result.event_time == fixture["event_time_grid"]
    assert level_result.metadata["support"] == fixture["support"]
    assert slope_result.metadata["support"] == fixture["support"]
    assert level_result.estimate == pytest.approx(fixture["level_curve"], abs=_POINT_ESTIMATE_ATOL)
    assert slope_result.estimate == pytest.approx(fixture["slope_curve"], abs=_POINT_ESTIMATE_ATOL)
    assert level_result.metadata["summary_aggregates"]["overall_level"] == pytest.approx(
        fixture["summary_aggregates"]["overall_level"], abs=_SUMMARY_ATOL
    )
    assert slope_result.metadata["summary_aggregates"]["overall_slope"] == pytest.approx(
        fixture["summary_aggregates"]["overall_slope"], abs=_SUMMARY_ATOL
    )
    assert level_result.metadata["shape_constraints"] == fixture["shape_constraints"]
    assert slope_result.metadata["shape_constraints"] == fixture["shape_constraints"]
