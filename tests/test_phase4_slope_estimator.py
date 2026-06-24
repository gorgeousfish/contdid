from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from contdid import ContDIDSpec, simulate_contdid_data
from contdid.data import PanelData
from contdid.validation import ContDIDValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_FIXTURE_PATH = (
    REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase4_parity_expected.json"
)


def _load_fixture() -> dict:
    return json.loads(PARITY_FIXTURE_PATH.read_text(encoding="utf-8"))


def _make_slope_spec(
    *, aggregation: str = "dose", dose_est_method: str = "parametric"
) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter="slope",
        aggregation=aggregation,
        dose_est_method=dose_est_method,
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
    )


def _simulate_two_period_dose_panel(
    *, n: int, dgp_id: str, seed: int | None = None
) -> PanelData:
    kwargs: dict[str, object] = {
        "n": n,
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "dgp_id": dgp_id,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return simulate_contdid_data(**kwargs)


def _make_piecewise_linear_slope_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
    ]
    for unit, dose in enumerate((0.2, 0.4, 0.6, 0.8), start=1):
        effect = 2.0 * dose + 3.0 * max(dose - 0.5, 0.0)
        rows.extend(
            [
                (f"t{unit}", 1, 0.0, 2, dose),
                (f"t{unit}", 2, effect, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def test_slope_estimator_tracks_null_dose_curve_near_zero() -> None:
    from contdid import estimate_dose_slope_effects

    fixture = _load_fixture()
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-001-null-dose")

    result = estimate_dose_slope_effects(
        panel, _make_slope_spec(), dvals=fixture["evaluation_grid"], degree=1
    )

    assert result.estimand == "ACRT(d)"
    assert result.grid == fixture["evaluation_grid"]
    assert result.metadata["identification"] == {
        "paper_estimand": "ACRT(d)",
        "identifying_assumption": "SPT + continuous dose support",
        "ordinary_pt_interpretation": (
            "derivative of LATT(d|d) with local selection-bias contamination"
        ),
        "identification_note": (
            "Ordinary PT is not enough for a causal ACRT(d) interpretation; "
            "the public slope route reports the SPT-based causal-response label."
        ),
    }
    assert max(abs(value) for value in result.estimate) < 0.08
    assert abs(result.metadata["summary"]["overall_acrt_uniform_support"]) < 0.05


def test_slope_estimator_preserves_positive_linear_direction() -> None:
    from contdid import estimate_dose_slope_effects

    fixture = _load_fixture()
    oracle = fixture["scenarios"]["SIM-002-linear-dose"]
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-002-linear-dose")

    result = estimate_dose_slope_effects(
        panel, _make_slope_spec(), dvals=fixture["evaluation_grid"], degree=1
    )

    assert all(value > 0.0 for value in result.estimate)
    assert max(result.estimate) - min(result.estimate) < 0.08
    assert result.metadata["summary"]["overall_acrt"] == pytest.approx(
        oracle["summary_aggregates"]["overall_acrt"], abs=0.06
    )
    assert result.metadata["summary"]["overall_acrt_uniform_support"] == pytest.approx(
        oracle["summary_aggregates"]["overall_acrt_uniform_support"], abs=0.06
    )


def test_slope_estimator_recovers_increasing_quadratic_response() -> None:
    from contdid import estimate_dose_slope_effects

    fixture = _load_fixture()
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-003-quadratic-dose")

    result = estimate_dose_slope_effects(
        panel, _make_slope_spec(), dvals=fixture["evaluation_grid"], degree=2
    )

    first_differences = [
        next_value - value
        for value, next_value in zip(result.estimate, result.estimate[1:])
    ]

    assert all(value > 0.0 for value in result.estimate)
    assert all(step > 0.0 for step in first_differences)


def test_linear_spline_slope_keeps_hinge_derivative_off_below_the_knot() -> None:
    from contdid import estimate_dose_slope_effects

    result = estimate_dose_slope_effects(
        _make_piecewise_linear_slope_panel(),
        _make_slope_spec(),
        dvals=[0.3, 0.7],
        degree=1,
        num_knots=1,
    )

    assert result.metadata["basis"]["interior_knots"] == pytest.approx([0.5])
    assert result.estimate == pytest.approx([2.0, 5.0], abs=1e-12)


def test_linear_spline_slope_uses_right_derivative_at_the_knot() -> None:
    from contdid import estimate_dose_slope_effects

    result = estimate_dose_slope_effects(
        _make_piecewise_linear_slope_panel(),
        _make_slope_spec(),
        dvals=[0.5],
        degree=1,
        num_knots=1,
    )

    assert result.metadata["basis"]["interior_knots"] == pytest.approx([0.5])
    assert result.estimate == pytest.approx([5.0], abs=1e-12)


def test_slope_estimator_rejects_eventstudy_aggregation() -> None:
    from contdid import estimate_dose_slope_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")
    spec = _make_slope_spec(aggregation="eventstudy")

    with pytest.raises(
        ContDIDValidationError, match="Phase 4 only supports aggregation='dose'"
    ):
        estimate_dose_slope_effects(panel, spec)


def test_slope_estimator_routes_supported_cck_dose_path() -> None:
    from contdid import estimate_dose_slope_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")
    spec = _make_slope_spec(dose_est_method="cck")

    result = estimate_dose_slope_effects(panel, spec, dvals=[0.2, 0.5, 0.8])

    assert result.estimand == "ACRT(d)"
    assert result.metadata["dose_est_method"] == "cck"
    assert result.metadata["source_estimator"] == "phase6_cck_backend"
    assert result.metadata["identification"]["identifying_assumption"] == (
        "SPT + continuous dose support"
    )
    assert "selection-bias contamination" in result.metadata["identification"][
        "ordinary_pt_interpretation"
    ]
