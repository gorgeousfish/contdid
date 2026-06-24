from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def _make_level_spec(
    *,
    aggregation: str = "dose",
    dose_est_method: str = "parametric",
    bstrap: bool = True,
) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter="level",
        aggregation=aggregation,
        dose_est_method=dose_est_method,
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        bstrap=bstrap,
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


def _make_skewed_summary_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.04, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.2),
        ("t2", 2, 0.04, 2, 0.2),
        ("t3", 1, 0.0, 2, 0.5),
        ("t3", 2, 0.25, 2, 0.5),
        ("t4", 1, 0.0, 2, 0.8),
        ("t4", 2, 0.64, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


class _NonIterableArray(np.ndarray):
    def __iter__(self):
        raise AssertionError("ndarray dvals should use the NumPy coercion path")


def _make_underidentified_support_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.10, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.70, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_exact_fit_support_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.20, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.80, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_single_untreated_exact_fit_support_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.20, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.80, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_negative_dose_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("tneg", 1, 0.0, 2, -0.4),
        ("tneg", 2, -0.16, 2, -0.4),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.10, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.70, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_non_finite_dose_panel(dose_value: float) -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("tbad", 1, 0.0, 2, dose_value),
        ("tbad", 2, 0.10, 2, dose_value),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.70, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_invalid_group_timing_panel(group_value: float) -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("tbad", 1, 0.0, group_value, 0.2),
        ("tbad", 2, 0.20, group_value, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.50, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.80, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_non_finite_outcome_panel(outcome_value: float) -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("tbad", 1, 0.0, 2, 0.2),
        ("tbad", 2, outcome_value, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.50, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.80, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_missing_id_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.2, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.4),
        ("t2", 2, 0.4, 2, 0.4),
        ("t3", 1, 0.0, 2, 0.6),
        ("t3", 2, 0.6, 2, 0.6),
        ("t4", 1, 0.0, 2, 0.8),
        ("t4", 2, 0.8, 2, 0.8),
        (pd.NA, 1, 0.0, 2, 0.5),
        (pd.NA, 2, 50.0, 2, 0.5),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_treated_timing_zero_dose_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("bad", 1, 0.0, 2, 0.0),
        ("bad", 2, 10.0, 2, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.2, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.5, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.8, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_multi_period_staggered_dose_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u0", 3, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("u1", 3, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.2, 2, 0.2),
        ("t1", 3, 0.4, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.8),
        ("t2", 2, 0.8, 2, 0.8),
        ("t2", 3, 1.6, 2, 0.8),
        ("t3", 1, 0.0, 3, 0.5),
        ("t3", 2, 0.0, 3, 0.5),
        ("t3", 3, 0.5, 3, 0.5),
        ("t4", 1, 0.0, 3, 0.9),
        ("t4", 2, 0.0, 3, 0.9),
        ("t4", 3, 0.9, 3, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_baseline_treated_two_period_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 10.0, 1, 0.2),
        ("t1", 2, 10.2, 1, 0.2),
        ("t2", 1, 20.0, 1, 0.5),
        ("t2", 2, 20.5, 1, 0.5),
        ("t3", 1, 30.0, 1, 0.8),
        ("t3", 2, 30.8, 1, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_boolean_grid_support_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.5),
        ("t1", 2, 0.5, 2, 0.5),
        ("t2", 1, 0.0, 2, 1.0),
        ("t2", 2, 1.0, 2, 1.0),
        ("t3", 1, 0.0, 2, 1.5),
        ("t3", 2, 1.5, 2, 1.5),
        ("t4", 1, 0.0, 2, 2.0),
        ("t4", 2, 2.0, 2, 2.0),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def test_level_estimator_tracks_null_dose_curve_near_zero() -> None:
    from contdid import estimate_dose_effects

    fixture = _load_fixture()
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-001-null-dose")

    result = estimate_dose_effects(
        panel, _make_level_spec(), dvals=fixture["evaluation_grid"], degree=2
    )

    assert result.estimand == "ATT(d)"
    assert result.grid == fixture["evaluation_grid"]
    assert result.metadata["identification"] == {
        "paper_estimand": "ATT(d)",
        "identifying_assumption": "SPT",
        "ordinary_pt_interpretation": "LATT(d|d)",
        "identification_note": (
            "The same dose-specific contrast identifies LATT(d|d) under "
            "ordinary PT; interpreting it as ATT(d) requires SPT."
        ),
    }
    assert max(abs(value) for value in result.estimate) < 0.08
    assert abs(result.metadata["summary"]["overall_att_uniform_support"]) < 0.05
    assert result.metadata["untreated_benchmark"] == pytest.approx(1.0, abs=0.08)


def test_level_estimator_preserves_positive_linear_dose_ordering() -> None:
    from contdid import estimate_dose_effects

    fixture = _load_fixture()
    oracle = fixture["scenarios"]["SIM-002-linear-dose"]
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-002-linear-dose")

    result = estimate_dose_effects(
        panel, _make_level_spec(), dvals=fixture["evaluation_grid"], degree=1
    )

    assert all(
        next_value > value
        for value, next_value in zip(result.estimate, result.estimate[1:])
    )
    assert result.metadata["summary"]["overall_att"] == pytest.approx(
        oracle["summary_aggregates"]["overall_att"], abs=0.06
    )
    assert result.metadata["summary"]["overall_att_uniform_support"] == pytest.approx(
        oracle["summary_aggregates"]["overall_att_uniform_support"], abs=0.06
    )


def test_level_estimator_recovers_convex_quadratic_curve_shape() -> None:
    from contdid import estimate_dose_effects

    fixture = _load_fixture()
    panel = _simulate_two_period_dose_panel(n=16000, dgp_id="SIM-003-quadratic-dose")

    result = estimate_dose_effects(
        panel, _make_level_spec(), dvals=fixture["evaluation_grid"], degree=2
    )

    first_differences = [
        next_value - value
        for value, next_value in zip(result.estimate, result.estimate[1:])
    ]
    second_differences = [
        next_value - value
        for value, next_value in zip(first_differences, first_differences[1:])
    ]

    assert all(value > 0.0 for value in result.estimate)
    assert all(step > 0.0 for step in first_differences)
    assert all(step > 0.0 for step in second_differences)


def test_level_estimator_rejects_eventstudy_aggregation() -> None:
    from contdid import estimate_dose_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")
    spec = _make_level_spec(aggregation="eventstudy")

    with pytest.raises(
        ContDIDValidationError, match="Phase 4 only supports aggregation='dose'"
    ):
        estimate_dose_effects(panel, spec)


def test_level_estimator_routes_supported_cck_dose_path() -> None:
    from contdid import estimate_dose_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")
    spec = _make_level_spec(dose_est_method="cck", bstrap=False)

    result = estimate_dose_effects(panel, spec, dvals=[0.2, 0.5, 0.8])

    assert result.estimand == "ATT(d)"
    assert result.metadata["dose_est_method"] == "cck"
    assert result.metadata["source_estimator"] == "phase6_cck_backend"
    assert result.metadata["identification"]["identifying_assumption"] == "SPT"
    assert (
        result.metadata["identification"]["ordinary_pt_interpretation"]
        == "LATT(d|d)"
    )


def test_parametric_dose_routes_reject_negative_anticipation() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")
    level_spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=-1,
    )
    slope_spec = ContDIDSpec(
        target_parameter="slope",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=-1,
    )

    with pytest.raises(
        ContDIDValidationError, match="anticipation must be a non-negative integer"
    ):
        estimate_dose_effects(panel, level_spec)

    with pytest.raises(
        ContDIDValidationError, match="anticipation must be a non-negative integer"
    ):
        estimate_dose_slope_effects(panel, slope_spec)


def test_parametric_dose_routes_reject_unchecked_control_groups() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    control_group = "eventuallytreated"
    panel = _simulate_two_period_dose_panel(
        n=4000, dgp_id="SIM-002-linear-dose", seed=20260407
    )
    level_spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
    )
    slope_spec = ContDIDSpec(
        target_parameter="slope",
        aggregation="dose",
        dose_est_method="parametric",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
    )

    match = "is not supported"
    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(panel, level_spec, degree=1)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(panel, slope_spec, degree=1)


def test_level_estimator_exports_plan_locked_level_aliases_and_grid_builder() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_level_effects,
    )

    explicit_grid = [0.2, 0.4, 0.8]
    panel = _simulate_two_period_dose_panel(n=8000, dgp_id="SIM-002-linear-dose")

    computed_grid = build_dose_grid(panel, dvals=explicit_grid)
    aliased = estimate_dose_level_effects(
        panel, _make_level_spec(), dvals=explicit_grid, degree=1
    )
    canonical = estimate_dose_effects(
        panel, _make_level_spec(), dvals=explicit_grid, degree=1
    )

    assert computed_grid == explicit_grid
    assert aliased.estimand == "ATT(d)"
    assert aliased.grid == explicit_grid
    assert aliased.estimate == pytest.approx(canonical.estimate)
    assert aliased.metadata["dose_grid"] == explicit_grid


def test_parametric_dose_routes_reject_noninteger_basis_controls() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _simulate_two_period_dose_panel(n=4000, dgp_id="SIM-002-linear-dose")

    with pytest.raises(ContDIDValidationError, match="degree must be an integer"):
        estimate_dose_effects(panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1.5)

    with pytest.raises(ContDIDValidationError, match="num_knots must be an integer"):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
            num_knots=0.5,
        )


@pytest.mark.parametrize(
    "knots",
    [
        [0.2, True],
        [0.2, np.bool_(True)],
        ["0.2", "0.8"],
        np.asarray([0.2, True], dtype=object),
    ],
)
def test_shared_parametric_dose_fit_rejects_non_numeric_explicit_knots(
    knots: object,
) -> None:
    import contdid.estimation as estimation_module

    panel = _make_boolean_grid_support_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="interior knots must contain only finite non-boolean numeric values",
    ):
        estimation_module._fit_shared_dose_design(
            panel,
            _make_level_spec(bstrap=False),
            expected_target="level",
            dvals=[0.5, 1.0, 1.5],
            degree=1,
            num_knots=0,
            knots=knots,
        )


def test_level_estimator_respects_paneldata_column_overrides() -> None:
    from contdid import estimate_dose_effects

    explicit_grid = [0.2, 0.4, 0.8]
    canonical_panel = _simulate_two_period_dose_panel(
        n=8000, dgp_id="SIM-002-linear-dose"
    )
    custom_panel = PanelData(
        frame=canonical_panel.frame.rename(
            columns={
                canonical_panel.id_column: "unit_id",
                canonical_panel.time_column: "period",
                canonical_panel.outcome_column: "outcome",
                canonical_panel.group_column: "cohort",
                canonical_panel.dose_column: "dose",
            }
        ),
        id_column="unit_id",
        time_column="period",
        outcome_column="outcome",
        group_column="cohort",
        dose_column="dose",
    )

    canonical = estimate_dose_effects(
        canonical_panel,
        _make_level_spec(),
        dvals=explicit_grid,
        degree=1,
    )
    aliased = estimate_dose_effects(
        custom_panel,
        _make_level_spec(),
        dvals=explicit_grid,
        degree=1,
    )

    assert aliased.grid == canonical.grid
    assert aliased.estimate == pytest.approx(canonical.estimate)
    assert aliased.metadata["dose_grid"] == explicit_grid
    assert aliased.metadata["untreated_benchmark"] == pytest.approx(
        canonical.metadata["untreated_benchmark"]
    )
    assert set(aliased.metadata["summary"]) == set(canonical.metadata["summary"])
    for key, value in canonical.metadata["summary"].items():
        assert aliased.metadata["summary"][key] == pytest.approx(value)


def test_default_parametric_dose_grid_matches_contdid_r_quantile_contract() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _simulate_two_period_dose_panel(n=6000, dgp_id="SIM-002-linear-dose")
    unit_frame = panel.frame.drop_duplicates(panel.id_column)
    positive_dose = unit_frame.loc[
        unit_frame[panel.dose_column] > 0.0, panel.dose_column
    ].to_numpy(dtype=float)
    expected_grid = np.quantile(positive_dose, np.arange(0.10, 1.0, 0.01)).tolist()

    computed_grid = build_dose_grid(panel)
    level_result = estimate_dose_effects(panel, _make_level_spec(), degree=1)
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        ),
        degree=1,
    )

    assert len(computed_grid) == 90
    assert computed_grid == pytest.approx(expected_grid)
    assert level_result.grid == pytest.approx(expected_grid)
    assert slope_result.grid == pytest.approx(expected_grid)


def test_parametric_dose_summaries_follow_realized_treated_support_not_grid_mean() -> (
    None
):
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_skewed_summary_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    level_result = estimate_dose_effects(
        panel,
        _make_level_spec(),
        dvals=explicit_grid,
        degree=2,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        dvals=explicit_grid,
        degree=2,
    )
    expected_overall_att = (
        2.0 * level_result.estimate[0]
        + level_result.estimate[1]
        + level_result.estimate[2]
    ) / 4.0
    expected_overall_acrt = (
        2.0 * slope_result.estimate[0]
        + slope_result.estimate[1]
        + slope_result.estimate[2]
    ) / 4.0

    assert level_result.metadata["summary"]["overall_att"] == pytest.approx(
        expected_overall_att
    )
    assert level_result.metadata["summary"]["overall_att_uniform_support"] == pytest.approx(
        sum(level_result.estimate) / len(level_result.estimate)
    )
    assert level_result.metadata["summary"]["overall_att"] != pytest.approx(
        level_result.metadata["summary"]["overall_att_uniform_support"]
    )

    assert slope_result.metadata["summary"]["overall_acrt"] == pytest.approx(
        expected_overall_acrt
    )
    assert slope_result.metadata["summary"][
        "overall_acrt_uniform_support"
    ] == pytest.approx(sum(slope_result.estimate) / len(slope_result.estimate))
    assert slope_result.metadata["summary"]["overall_acrt"] != pytest.approx(
        slope_result.metadata["summary"]["overall_acrt_uniform_support"]
    )


def test_parametric_dose_routes_reject_underidentified_positive_dose_support() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_underidentified_support_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="underidentified positive-dose support"
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=explicit_grid,
            degree=3,
        )

    with pytest.raises(
        ContDIDValidationError, match="underidentified positive-dose support"
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=explicit_grid,
            degree=3,
        )


@pytest.mark.parametrize("bstrap", [True, False])
def test_parametric_dose_routes_reject_exact_fit_support_without_inference_df(
    bstrap: bool,
) -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_exact_fit_support_panel()
    explicit_grid = [0.2, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(bstrap=bstrap),
            dvals=explicit_grid,
            degree=1,
        )

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
                bstrap=bstrap,
            ),
            dvals=explicit_grid,
            degree=1,
        )


def test_parametric_level_exact_fit_support_takes_precedence_over_untreated_variance_guard() -> (
    None
):
    from contdid import estimate_dose_effects

    panel = _make_single_untreated_exact_fit_support_panel()
    explicit_grid = [0.2, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(bstrap=False),
            dvals=explicit_grid,
            degree=1,
        )


def test_parametric_dose_routes_reject_negative_dose_support() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_negative_dose_panel()

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative"
    ):
        build_dose_grid(panel)

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative"
    ):
        estimate_dose_effects(panel, _make_level_spec(), degree=1)

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative"
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            degree=1,
        )


@pytest.mark.parametrize("dose_value", [float("nan"), float("inf")])
def test_parametric_dose_routes_reject_non_finite_dose_support(
    dose_value: float,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_non_finite_dose_panel(dose_value)

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative and finite"
    ):
        build_dose_grid(panel)

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative and finite"
    ):
        estimate_dose_effects(panel, _make_level_spec(), degree=1)

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative and finite"
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            degree=1,
        )


@pytest.mark.parametrize("outcome_value", [float("nan"), float("inf")])
def test_parametric_dose_routes_reject_non_finite_outcomes(
    outcome_value: float,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_non_finite_outcome_panel(outcome_value)

    with pytest.raises(
        ContDIDValidationError, match="outcome values must be finite"
    ):
        build_dose_grid(panel)

    with pytest.raises(
        ContDIDValidationError, match="outcome values must be finite"
    ):
        estimate_dose_effects(panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1)

    with pytest.raises(
        ContDIDValidationError, match="outcome values must be finite"
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


@pytest.mark.parametrize(
    ("estimator_name", "target_parameter"),
    [
        ("estimate_dose_effects", "level"),
        ("estimate_dose_slope_effects", "slope"),
    ],
)
def test_public_dose_routes_reuse_single_validated_panel(
    monkeypatch: pytest.MonkeyPatch,
    estimator_name: str,
    target_parameter: str,
) -> None:
    import contdid.estimation as estimation_module
    import contdid.validation as validation_module
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _simulate_two_period_dose_panel(
        n=80,
        dgp_id="SIM-002-linear-dose",
        seed=1234,
    )
    spec = ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        bstrap=False,
    )
    estimator = {
        "estimate_dose_effects": estimate_dose_effects,
        "estimate_dose_slope_effects": estimate_dose_slope_effects,
    }[estimator_name]
    original_validate = validation_module.validate_panel_data
    validation_calls = 0

    def _counting_validate(panel_arg):
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(panel_arg)

    monkeypatch.setattr(estimation_module, "validate_panel_data", _counting_validate)
    monkeypatch.setattr(validation_module, "validate_panel_data", _counting_validate)

    estimator(panel, spec, dvals=[0.2, 0.5, 0.8], degree=1)

    assert validation_calls == 1


@pytest.mark.parametrize(
    "dvals",
    [
        "0.5",
        ["0.2", "0.5", "0.8"],
        np.asarray(["0.2", "0.5", "0.8"]),
        np.asarray([0.2, "0.5", 0.8], dtype=object),
    ],
)
def test_parametric_dose_routes_reject_string_valued_explicit_grids(
    dvals: object,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_skewed_summary_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        build_dose_grid(panel, dvals=dvals)

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=dvals,
            degree=1,
        )

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=dvals,
            degree=1,
        )


@pytest.mark.parametrize(
    "dvals",
    [
        True,
        np.bool_(True),
        [0.5, True, 1.5],
        [0.5, np.bool_(True), 1.5],
        np.asarray([False, True]),
        np.asarray([0.5, True, 1.5], dtype=object),
    ],
)
def test_parametric_dose_routes_reject_boolean_explicit_grids(
    dvals: object,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_boolean_grid_support_panel()
    match = "dose grid must contain only finite non-boolean numeric values"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel, dvals=dvals)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=dvals,
            degree=1,
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=dvals,
            degree=1,
        )


def test_parametric_dose_routes_accept_numpy_explicit_grid_without_iteration() -> None:
    from contdid import build_dose_grid, estimate_dose_effects

    panel = _make_skewed_summary_panel()
    dvals = np.asarray([0.2, 0.5, 0.8], dtype=float).view(_NonIterableArray)

    assert build_dose_grid(panel, dvals=dvals) == [0.2, 0.5, 0.8]

    result = estimate_dose_effects(
        panel,
        _make_level_spec(bstrap=False),
        dvals=dvals,
        degree=1,
    )
    assert result.grid == [0.2, 0.5, 0.8]


def test_parametric_dose_routes_reject_numeric_string_outcomes_before_delta_split() -> (
    None
):
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_skewed_summary_panel()
    panel.frame["Y"] = panel.frame["Y"].astype(str)
    match = "outcome values must be finite numeric values"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel, dvals=[0.2, 0.5, 0.8])

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=[0.2, 0.5, 0.8],
            degree=2,
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=2,
        )


def test_parametric_dose_routes_reject_missing_unit_ids_before_collapse() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_missing_id_panel()
    match = "id values must be nonmissing"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel, dvals=[0.2, 0.5, 0.8])

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


@pytest.mark.parametrize("group_value", [float("nan"), float("inf"), -1.0])
def test_parametric_dose_routes_reject_invalid_group_timing(
    group_value: float,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_invalid_group_timing_panel(group_value)
    match = "group timing values must be nonnegative and finite"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


def test_parametric_dose_routes_reject_treated_timing_without_positive_dose() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_treated_timing_zero_dose_panel()
    match = "positive treatment timing must have positive dose"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


def test_parametric_dose_routes_reject_multi_period_staggered_panels() -> None:
    from contdid import build_dose_grid, estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_multi_period_staggered_dose_panel()
    match = (
        "dose aggregation currently supports exactly two observed time periods only "
        "until checked multi-period dose timing semantics land on the public dose routes"
    )

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


def test_parametric_dose_routes_reject_treatment_timing_in_first_period() -> None:
    from contdid import build_dose_grid, estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_baseline_treated_two_period_panel()
    match = (
        "two-period dose aggregation requires positive treatment timing to start "
        "in the post period"
    )

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel, dvals=[0.2, 0.5, 0.8])

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


def test_parametric_dose_routes_reject_group_timing_off_the_observed_integer_grid() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_invalid_group_timing_panel(1.5)
    match = "group timing values must align with or follow the observed integer time-period grid"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel, _make_level_spec(), dvals=[0.2, 0.5, 0.8], degree=1
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=[0.2, 0.5, 0.8],
            degree=1,
        )


def test_parametric_dose_routes_reject_explicit_grid_outside_positive_support() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_skewed_summary_panel()
    explicit_grid = [0.0, 0.2, 0.8]

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must stay within the observed positive-dose treated support",
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=explicit_grid,
            degree=2,
        )

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must stay within the observed positive-dose treated support",
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=explicit_grid,
            degree=2,
        )


@pytest.mark.parametrize(
    ("explicit_grid", "label"),
    [
        ([0.2, 0.2, 0.8], "duplicate"),
        ([0.8, 0.2, 0.5], "nonincreasing"),
    ],
)
def test_parametric_dose_routes_reject_non_strict_explicit_grid_values(
    explicit_grid: list[float],
    label: str,
) -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_skewed_summary_panel()
    match = "explicit dvals must be strictly increasing with no duplicate dose values"

    with pytest.raises(ContDIDValidationError, match=match):
        build_dose_grid(panel, dvals=explicit_grid)

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=explicit_grid,
            degree=2,
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=explicit_grid,
            degree=2,
        )

    assert label in {"duplicate", "nonincreasing"}


def test_parametric_dose_routes_reject_non_finite_explicit_grid_values() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_skewed_summary_panel()
    explicit_grid = [0.2, float("nan"), 0.8]

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        build_dose_grid(panel, dvals=explicit_grid)

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_effects(
            panel,
            _make_level_spec(),
            dvals=explicit_grid,
            degree=2,
        )

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_slope_effects(
            panel,
            ContDIDSpec(
                target_parameter="slope",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                treatment_type="continuous",
                anticipation=0,
            ),
            dvals=explicit_grid,
            degree=2,
        )


def test_parametric_dose_routes_accept_scalar_explicit_grid_as_length_one() -> None:
    from contdid import (
        build_dose_grid,
        estimate_dose_effects,
        estimate_dose_slope_effects,
    )

    panel = _make_skewed_summary_panel()

    assert build_dose_grid(panel, dvals=0.5) == pytest.approx([0.5])

    level_result = estimate_dose_effects(panel, _make_level_spec(), dvals=0.5, degree=2)
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        dvals=0.5,
        degree=2,
    )

    assert level_result.grid == pytest.approx([0.5])
    assert slope_result.grid == pytest.approx([0.5])
    assert len(level_result.estimate) == 1
    assert len(slope_result.estimate) == 1


def _make_linear_bspline_reference_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []
    for unit in ["u0", "u1", "u2"]:
        rows.extend([(unit, 1, 0.0, 0, 0.0), (unit, 2, 0.0, 0, 0.0)])
    for index, (dose, delta) in enumerate(
        [(1.0, 0.2), (2.0, 0.7), (3.0, 1.3), (5.0, 2.8), (7.0, 3.2)],
        start=1,
    ):
        unit = f"t{index}"
        rows.extend([(unit, 1, 0.0, 2, dose), (unit, 2, delta, 2, dose)])
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cubic_bspline_reference_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []
    for unit in ["u0", "u1", "u2"]:
        rows.extend([(unit, 1, 0.0, 0, 0.0), (unit, 2, 0.0, 0, 0.0)])
    for index, dose in enumerate([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 7.0], start=1):
        delta = 0.4 + 0.25 * dose - 0.08 * dose**2 + 0.015 * dose**3
        unit = f"tc{index}"
        rows.extend([(unit, 1, 0.0, 2, dose), (unit, 2, delta, 2, dose)])
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _linear_bspline_no_intercept_columns(
    values: np.ndarray,
    *,
    lower: float,
    knot: float,
    upper: float,
) -> np.ndarray:
    left_middle = np.where(
        values < knot,
        (values - lower) / (knot - lower),
        (upper - values) / (upper - knot),
    )
    right = np.where(values < knot, 0.0, (values - knot) / (upper - knot))
    return np.column_stack([left_middle, right])


def _linear_bspline_no_intercept_derivative_columns(
    values: np.ndarray,
    *,
    lower: float,
    knot: float,
    upper: float,
) -> np.ndarray:
    left_middle = np.where(values < knot, 1.0 / (knot - lower), -1.0 / (upper - knot))
    right = np.where(values < knot, 0.0, 1.0 / (upper - knot))
    return np.column_stack([left_middle, right])


def _cubic_bspline_no_intercept_columns(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    scaled = (values - lower) / (upper - lower)
    middle_left = 3.0 * scaled * (1.0 - scaled) ** 2
    middle_right = 3.0 * scaled**2 * (1.0 - scaled)
    right = scaled**3
    return np.column_stack([middle_left, middle_right, right])


def _cubic_bspline_no_intercept_derivative_columns(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    scaled = (values - lower) / (upper - lower)
    scale = 1.0 / (upper - lower)
    middle_left = 3.0 * (1.0 - scaled) * (1.0 - 3.0 * scaled) * scale
    middle_right = 3.0 * scaled * (2.0 - 3.0 * scaled) * scale
    right = 3.0 * scaled**2 * scale
    return np.column_stack([middle_left, middle_right, right])


def test_linear_parametric_spline_matches_reference_bspline_span_and_derivative() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_linear_bspline_reference_panel()
    treated_dose = np.asarray([1.0, 2.0, 3.0, 5.0, 7.0])
    treated_delta = np.asarray([0.2, 0.7, 1.3, 2.8, 3.2])
    explicit_grid = [2.0, 3.0, 5.0, 6.0]
    grid = np.asarray(explicit_grid)
    knot = float(np.quantile(treated_dose, 0.5))

    reference_design = np.column_stack(
        [
            np.ones_like(treated_dose),
            _linear_bspline_no_intercept_columns(
                treated_dose,
                lower=float(treated_dose.min()),
                knot=knot,
                upper=float(treated_dose.max()),
            ),
        ]
    )
    reference_coefficients, *_ = np.linalg.lstsq(
        reference_design, treated_delta, rcond=None
    )
    reference_grid = np.column_stack(
        [
            np.ones_like(grid),
            _linear_bspline_no_intercept_columns(
                grid,
                lower=float(treated_dose.min()),
                knot=knot,
                upper=float(treated_dose.max()),
            ),
        ]
    )
    reference_derivative_grid = np.column_stack(
        [
            np.zeros_like(grid),
            _linear_bspline_no_intercept_derivative_columns(
                grid,
                lower=float(treated_dose.min()),
                knot=knot,
                upper=float(treated_dose.max()),
            ),
        ]
    )

    level_result = estimate_dose_effects(
        panel,
        _make_level_spec(bstrap=False),
        dvals=explicit_grid,
        degree=1,
        num_knots=1,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            bstrap=False,
        ),
        dvals=explicit_grid,
        degree=1,
        num_knots=1,
    )

    assert level_result.metadata["basis"]["interior_knots"] == pytest.approx([knot])
    assert slope_result.metadata["basis"]["interior_knots"] == pytest.approx([knot])
    assert level_result.estimate == pytest.approx(
        reference_grid @ reference_coefficients,
        abs=1e-12,
    )
    assert slope_result.estimate == pytest.approx(
        reference_derivative_grid @ reference_coefficients,
        abs=1e-12,
    )


def test_default_cubic_parametric_route_matches_reference_bspline_span_and_derivative() -> None:
    from contdid import estimate_dose_effects, estimate_dose_slope_effects

    panel = _make_cubic_bspline_reference_panel()
    treated_dose = np.asarray([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 7.0])
    treated_delta = (
        0.4
        + 0.25 * treated_dose
        - 0.08 * treated_dose**2
        + 0.015 * treated_dose**3
    )
    explicit_grid = [1.25, 2.5, 5.0, 6.5]
    grid = np.asarray(explicit_grid)
    lower = float(treated_dose.min())
    upper = float(treated_dose.max())

    reference_design = np.column_stack(
        [
            np.ones_like(treated_dose),
            _cubic_bspline_no_intercept_columns(
                treated_dose,
                lower=lower,
                upper=upper,
            ),
        ]
    )
    reference_coefficients, *_ = np.linalg.lstsq(
        reference_design, treated_delta, rcond=None
    )
    reference_grid = np.column_stack(
        [
            np.ones_like(grid),
            _cubic_bspline_no_intercept_columns(
                grid,
                lower=lower,
                upper=upper,
            ),
        ]
    )
    reference_derivative_grid = np.column_stack(
        [
            np.zeros_like(grid),
            _cubic_bspline_no_intercept_derivative_columns(
                grid,
                lower=lower,
                upper=upper,
            ),
        ]
    )

    level_result = estimate_dose_effects(
        panel,
        _make_level_spec(bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            bstrap=False,
        ),
        dvals=explicit_grid,
    )

    assert level_result.metadata["basis"] == {
        "type": "global_polynomial",
        "degree": 3,
        "num_knots": 0,
        "interior_knots": [],
    }
    assert slope_result.metadata["basis"] == level_result.metadata["basis"]
    assert level_result.estimate == pytest.approx(
        reference_grid @ reference_coefficients,
        abs=1e-12,
    )
    assert slope_result.estimate == pytest.approx(
        reference_derivative_grid @ reference_coefficients,
        abs=1e-12,
    )
