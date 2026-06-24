from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDSpec,
    estimate_dose_effects,
    estimate_dose_slope_effects,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
    simulate_contdid_data,
)
from contdid.data import PanelData
from contdid.validation import ContDIDValidationError


_NORMAL = NormalDist()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_R_CONTDID_PATH = _REPO_ROOT / "contdid-r" / "R" / "cont_did.R"


def _make_cck_dose_spec(*, target_parameter: str, bstrap: bool = True) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method="cck",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=bstrap,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )


def _make_cck_eventstudy_spec(*, target_parameter: str) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
    )


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


def _make_untreated_benchmark_variance_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, -0.2, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.2, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.04, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.4),
        ("t2", 2, 0.16, 2, 0.4),
        ("t3", 1, 0.0, 2, 0.6),
        ("t3", 2, 0.36, 2, 0.6),
        ("t4", 1, 0.0, 2, 0.8),
        ("t4", 2, 0.64, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_heteroskedastic_treated_fit_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("u2", 1, 0.0, 0, 0.0),
        ("u2", 2, 0.0, 0, 0.0),
        ("u3", 1, 0.0, 0, 0.0),
        ("u3", 2, 0.0, 0, 0.0),
    ]
    for unit, dose, residual in (
        ("t1", 0.2, 0.0),
        ("t2", 0.4, 0.0),
        ("t3", 0.6, -0.6),
        ("t4", 0.8, 0.6),
        ("t5", 1.0, -1.2),
        ("t6", 1.2, 1.2),
    ):
        rows.extend(
            [
                (unit, 1, 0.0, 2, dose),
                (unit, 2, dose + residual, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_combined_inference_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, -0.15, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.25, 0, 0.0),
        ("u2", 1, 0.0, 0, 0.0),
        ("u2", 2, 0.50, 0, 0.0),
    ]
    for unit, dose, residual in (
        ("tc1", 0.2, 0.00),
        ("tc2", 0.4, 0.18),
        ("tc3", 0.6, -0.27),
        ("tc4", 0.8, 0.36),
        ("tc5", 1.0, -0.21),
        ("tc6", 1.2, 0.24),
    ):
        signal = 0.3 + 0.7 * dose - 0.2 * dose**2
        rows.extend(
            [
                (unit, 1, 0.0, 2, dose),
                (unit, 2, signal + residual, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


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


def _make_single_untreated_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.1, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.14, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.35, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.74, 2, 0.8),
        ("t4", 1, 0.0, 2, 0.9),
        ("t4", 2, 0.89, 2, 0.9),
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
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.50, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.80, 2, 0.8),
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
        ("t4", 1, 40.0, 1, 0.9),
        ("t4", 2, 40.9, 1, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_staggered_two_period_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("g1_t1", 1, 0.0, 1, 0.2),
        ("g1_t1", 2, 0.2, 1, 0.2),
        ("g1_t2", 1, 0.0, 1, 0.5),
        ("g1_t2", 2, 0.5, 1, 0.5),
        ("g2_t1", 1, 0.0, 2, 0.8),
        ("g2_t1", 2, 0.8, 2, 0.8),
        ("g2_t2", 1, 0.0, 2, 0.9),
        ("g2_t2", 2, 0.9, 2, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_no_treated_cohort_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.1, 0, 0.0),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_staggered_three_period_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u0", 3, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("u1", 3, 0.0, 0, 0.0),
        ("g2_t1", 1, 0.0, 2, 0.2),
        ("g2_t1", 2, 0.1, 2, 0.2),
        ("g2_t1", 3, 0.2, 2, 0.2),
        ("g2_t2", 1, 0.0, 2, 0.5),
        ("g2_t2", 2, 0.25, 2, 0.5),
        ("g2_t2", 3, 0.5, 2, 0.5),
        ("g3_t1", 1, 0.0, 3, 0.8),
        ("g3_t1", 2, 0.0, 3, 0.8),
        ("g3_t1", 3, 0.64, 3, 0.8),
        ("g3_t2", 1, 0.0, 3, 0.9),
        ("g3_t2", 2, 0.0, 3, 0.9),
        ("g3_t2", 3, 0.81, 3, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_single_cohort_three_period_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u0", 3, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("u1", 3, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.1, 2, 0.2),
        ("t1", 3, 0.2, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.25, 2, 0.5),
        ("t2", 3, 0.5, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.64, 2, 0.8),
        ("t3", 3, 1.28, 2, 0.8),
        ("t4", 1, 0.0, 2, 0.9),
        ("t4", 2, 0.81, 2, 0.9),
        ("t4", 3, 1.62, 2, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cck_single_cohort_after_window_two_period_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 3, 0.2),
        ("t1", 2, 0.2, 3, 0.2),
        ("t2", 1, 0.0, 3, 0.5),
        ("t2", 2, 0.5, 3, 0.5),
        ("t3", 1, 0.0, 3, 0.8),
        ("t3", 2, 0.8, 3, 0.8),
        ("t4", 1, 0.0, 3, 0.9),
        ("t4", 2, 0.9, 3, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_single_untreated_exact_fit_support_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.04, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.25, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.64, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def test_supported_sim005_cck_dose_level_and_slope_paths_return_inference_payloads() -> (
    None
):
    explicit_grid = [0.2, 0.5, 0.8]
    panel = simulate_contdid_data(
        n=12000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level"),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope"),
        dvals=explicit_grid,
    )

    for result in (level_result, slope_result):
        assert result.metadata["bootstrap_type"] == "multiplier"
        assert result.metadata["bootstrap_seed"] == 20260407
        assert result.metadata["confidence_band_kind"] == "simultaneous_multiplier"
        assert "boot_type" not in result.metadata
        assert "backend_name" not in result.metadata
        assert result.metadata["summary_aggregates"] == result.metadata["summary"]
        assert result.critical_value is not None and result.critical_value > 0.0
        assert all(value > 0.0 for value in result.std_error)
        assert result.confidence_band is not None


def test_cck_level_and_slope_keep_separate_simultaneous_critical_values() -> None:
    panel = _make_cck_combined_inference_panel()
    explicit_grid = [0.2, 0.7, 1.2]

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level"),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope"),
        dvals=explicit_grid,
    )

    assert level_result.critical_value == pytest.approx(
        2.036123005241396,
        abs=1e-12,
    )
    assert slope_result.critical_value == pytest.approx(
        2.1261748516987153,
        abs=1e-12,
    )
    assert level_result.critical_value != pytest.approx(slope_result.critical_value)
    assert level_result.metadata["critical_value"] == level_result.critical_value
    assert slope_result.metadata["critical_value"] == slope_result.critical_value
    assert level_result.confidence_band["critical_value"] == level_result.critical_value
    assert slope_result.confidence_band["critical_value"] == slope_result.critical_value


def test_cck_default_dose_grid_matches_contdid_r_minmax_contract() -> None:
    panel = simulate_contdid_data(
        n=8000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )
    unit_frame = panel.frame.drop_duplicates(panel.id_column)
    positive_dose = unit_frame.loc[
        unit_frame[panel.dose_column] > 0.0, panel.dose_column
    ].to_numpy(dtype=float)
    expected_grid = np.linspace(positive_dose.min(), positive_dose.max(), 50).tolist()

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level"),
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope"),
    )

    assert len(level_result.grid) == 50
    assert level_result.grid == pytest.approx(expected_grid)
    assert slope_result.grid == pytest.approx(expected_grid)


def test_cck_dose_paths_honor_analytic_inference_when_bootstrap_disabled() -> None:
    explicit_grid = [0.2, 0.5, 0.8]
    panel = simulate_contdid_data(
        n=12000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )
    expected_critical = _NORMAL.inv_cdf(1.0 - 0.1 / 2.0)

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level", bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )

    for result in (level_result, slope_result):
        assert result.metadata["inference"] == "analytic"
        assert result.metadata["bootstrap_type"] == "analytic"
        assert result.metadata["bootstrap_seed"] is None
        assert result.metadata["confidence_band_kind"] == "pointwise_analytic"
        assert "boot_type" not in result.metadata
        assert result.metadata["summary_aggregates"] == result.metadata["summary"]
        assert result.metadata["bstrap"] is False
        assert result.critical_value == pytest.approx(expected_critical)
        assert result.confidence_band is not None
        assert result.confidence_band["critical_value"] == pytest.approx(
            expected_critical
        )
        assert all(value > 0.0 for value in result.std_error)


def test_contdid_r_cck_non_cband_branch_sets_level_and_slope_critical_values() -> None:
    source = _R_CONTDID_PATH.read_text(encoding="utf-8")
    guard_start = source.index("if (!cband) {")
    guard_end = source.index("}", guard_start)
    guard_body = source[guard_start:guard_end]

    assert guard_body.count("att.d_crit.val <- qnorm(1 - alp / 2)") == 1
    assert "acrt.d_crit.val <- qnorm(1 - alp / 2)" in guard_body


def test_cck_analytic_dose_results_collapse_intervals_on_perfect_fit() -> None:
    panel = _make_skewed_summary_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level", bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )

    for result in (level_result, slope_result):
        assert result.metadata["inference"] == "analytic"
        assert result.metadata["bootstrap_type"] == "analytic"
        assert "boot_type" not in result.metadata
        assert result.std_error == [0.0, 0.0, 0.0]
        assert result.confidence_interval == [
            [point, point] for point in result.estimate
        ]
        assert result.confidence_band is not None
        assert result.confidence_band["lower"] == result.estimate
        assert result.confidence_band["upper"] == result.estimate


def test_cck_level_inference_includes_untreated_benchmark_variance() -> None:
    panel = _make_untreated_benchmark_variance_panel()
    explicit_grid = [0.2, 0.4, 0.8]

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level", bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )

    assert level_result.metadata["inference"] == "analytic"
    assert level_result.std_error == pytest.approx([0.2, 0.2, 0.2])
    assert level_result.confidence_interval != [
        [point, point] for point in level_result.estimate
    ]

    assert slope_result.metadata["inference"] == "analytic"
    assert slope_result.std_error == pytest.approx([0.0, 0.0, 0.0])
    assert slope_result.confidence_interval == [
        [point, point] for point in slope_result.estimate
    ]


def test_cck_level_inference_requires_two_untreated_units() -> None:
    panel = _make_single_untreated_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="untreated benchmark variance requires at least two untreated units"
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level", bstrap=False),
            dvals=explicit_grid,
        )

    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )
    assert slope_result.metadata["inference"] == "analytic"


def test_cck_analytic_dose_inference_uses_treated_sandwich_covariance() -> None:
    panel = _make_cck_heteroskedastic_treated_fit_panel()
    explicit_grid = [0.2, 0.6, 1.2]

    frame = panel.frame.sort_values([panel.id_column, panel.time_column])
    collapsed = frame.groupby(panel.id_column, sort=False).agg(
        {
            panel.time_column: ["first", "last"],
            panel.outcome_column: ["first", "last"],
            panel.group_column: "first",
            panel.dose_column: "first",
        }
    )
    collapsed.columns = [
        "time_first",
        "time_last",
        "outcome_first",
        "outcome_last",
        panel.group_column,
        panel.dose_column,
    ]
    collapsed = collapsed.reset_index()
    collapsed["delta_outcome"] = collapsed["outcome_last"] - collapsed["outcome_first"]
    treated = collapsed.loc[
        collapsed[panel.dose_column] > 0.0,
        [panel.id_column, panel.dose_column, "delta_outcome"],
    ]
    treated_dose = treated[panel.dose_column].to_numpy(dtype=float)
    treated_delta = treated["delta_outcome"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones_like(treated_dose), treated_dose, treated_dose**2]
    )
    coefficients, *_ = np.linalg.lstsq(design, treated_delta, rcond=None)
    residual = treated_delta - design @ coefficients
    xtx_inv = np.linalg.pinv(design.T @ design)
    covariance = xtx_inv @ (design.T @ np.diag(residual**2) @ design) @ xtx_inv

    grid = np.asarray(explicit_grid, dtype=float)
    level_loadings = np.column_stack([np.ones_like(grid), grid, grid**2])
    slope_loadings = np.column_stack(
        [np.zeros_like(grid), np.ones_like(grid), 2.0 * grid]
    )
    expected_level_se = np.sqrt(
        np.einsum("ij,jk,ik->i", level_loadings, covariance, level_loadings)
    )
    expected_slope_se = np.sqrt(
        np.einsum("ij,jk,ik->i", slope_loadings, covariance, slope_loadings)
    )

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level", bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )

    assert level_result.metadata["inference"] == "analytic"
    assert level_result.std_error == pytest.approx(
        expected_level_se.tolist(), abs=1e-12
    )
    assert slope_result.metadata["inference"] == "analytic"
    assert slope_result.std_error == pytest.approx(
        expected_slope_se.tolist(), abs=1e-12
    )


def test_cck_level_inference_combines_treated_sandwich_and_untreated_variance() -> None:
    panel = _make_cck_combined_inference_panel()
    explicit_grid = [0.2, 0.7, 1.2]
    treated_dose = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    untreated_delta = np.asarray([-0.15, 0.25, 0.50])
    untreated_mean = float(np.mean(untreated_delta))
    untreated_mean_variance = float(
        np.var(untreated_delta, ddof=1) / untreated_delta.size
    )
    residual = np.asarray([0.0, 0.18, -0.27, 0.36, -0.21, 0.24])
    treated_delta = (
        0.3
        + 0.7 * treated_dose
        - 0.2 * treated_dose**2
        + residual
        - untreated_mean
    )
    design = np.column_stack(
        [np.ones_like(treated_dose), treated_dose, treated_dose**2]
    )
    coefficients, *_ = np.linalg.lstsq(design, treated_delta, rcond=None)
    regression_residual = treated_delta - design @ coefficients
    xtx_inv = np.linalg.pinv(design.T @ design)
    covariance = xtx_inv @ (
        design.T @ np.diag(regression_residual**2) @ design
    ) @ xtx_inv

    grid = np.asarray(explicit_grid, dtype=float)
    level_loadings = np.column_stack([np.ones_like(grid), grid, grid**2])
    slope_loadings = np.column_stack(
        [np.zeros_like(grid), np.ones_like(grid), 2.0 * grid]
    )
    expected_level_se = np.sqrt(
        np.einsum("ij,jk,ik->i", level_loadings, covariance, level_loadings)
        + untreated_mean_variance
    )
    expected_slope_se = np.sqrt(
        np.einsum("ij,jk,ik->i", slope_loadings, covariance, slope_loadings)
    )

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level", bstrap=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope", bstrap=False),
        dvals=explicit_grid,
    )

    assert level_result.metadata["basis"]["type"] == "cck_polynomial_backend"
    assert slope_result.metadata["basis"]["type"] == "cck_polynomial_backend"
    assert level_result.std_error == pytest.approx(
        expected_level_se.tolist(),
        abs=1e-12,
    )
    assert slope_result.std_error == pytest.approx(
        expected_slope_se.tolist(),
        abs=1e-12,
    )


def test_cck_dose_summaries_follow_realized_treated_support_not_grid_mean() -> None:
    panel = _make_skewed_summary_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    level_result = estimate_dose_effects(
        panel,
        _make_cck_dose_spec(target_parameter="level"),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_cck_dose_spec(target_parameter="slope"),
        dvals=explicit_grid,
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


def test_cck_dose_routes_reject_underidentified_positive_dose_support() -> None:
    panel = _make_underidentified_support_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="underidentified positive-dose support"
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level"),
            dvals=explicit_grid,
        )

    with pytest.raises(
        ContDIDValidationError, match="underidentified positive-dose support"
    ):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope"),
            dvals=explicit_grid,
        )


@pytest.mark.parametrize("bstrap", [True, False])
def test_cck_dose_routes_reject_exact_fit_support_without_inference_df(
    bstrap: bool,
) -> None:
    panel = _make_exact_fit_support_panel()

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level", bstrap=bstrap),
        )

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope", bstrap=bstrap),
        )


def test_cck_dose_routes_reject_treatment_timing_in_first_period() -> None:
    panel = _make_baseline_treated_two_period_panel()
    match = "cck estimator requires positive treatment timing to start in the post period"

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level"),
            dvals=[0.2, 0.5, 0.8],
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope"),
            dvals=[0.2, 0.5, 0.8],
        )


def test_cck_level_exact_fit_support_takes_precedence_over_untreated_variance_guard() -> (
    None
):
    panel = _make_single_untreated_exact_fit_support_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level", bstrap=False),
            dvals=explicit_grid,
        )


def test_cck_dose_routes_reject_non_finite_explicit_grid_values() -> None:
    panel = _make_skewed_summary_panel()
    explicit_grid = [0.2, float("nan"), 0.8]

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level"),
            dvals=explicit_grid,
        )

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope"),
            dvals=explicit_grid,
        )


@pytest.mark.parametrize(
    "explicit_grid",
    [
        [0.2, 0.2, 0.8],
        [0.8, 0.2, 0.5],
    ],
)
def test_cck_dose_routes_reject_non_strict_explicit_grid_values(
    explicit_grid: list[float],
) -> None:
    panel = _make_skewed_summary_panel()
    match = "explicit dvals must be strictly increasing with no duplicate dose values"

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level"),
            dvals=explicit_grid,
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope"),
            dvals=explicit_grid,
        )


@pytest.mark.parametrize(
    "dvals",
    [
        "0.5",
        ["0.2", "0.5", "0.8"],
    ],
)
def test_cck_dose_routes_reject_string_valued_explicit_grid_values(
    dvals: object,
) -> None:
    panel = simulate_contdid_data(n=4000, dgp_id="SIM-005-cck-two-period", seed=20261234)

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_effects(
            panel,
            _make_cck_dose_spec(target_parameter="level"),
            dvals=dvals,
        )

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimate_dose_slope_effects(
            panel,
            _make_cck_dose_spec(target_parameter="slope"),
            dvals=dvals,
        )


def test_cck_rejects_staggered_adoption_with_exact_manifest_substring() -> None:
    panel = _make_cck_staggered_two_period_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with staggered adoption yet",
    ):
        estimate_dose_effects(
            panel, _make_cck_dose_spec(target_parameter="level"), dvals=[0.2, 0.5]
        )


@pytest.mark.parametrize(
    ("estimator", "target_parameter"),
    [
        (estimate_dose_effects, "level"),
        (estimate_dose_slope_effects, "slope"),
    ],
)
def test_cck_requires_one_positive_treatment_timing_cohort(
    estimator,
    target_parameter: str,
) -> None:
    panel = _make_cck_no_treated_cohort_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator requires exactly one positive treatment timing cohort",
    ):
        estimator(
            panel, _make_cck_dose_spec(target_parameter=target_parameter), dvals=[0.2]
        )


def test_cck_staggered_adoption_guard_precedes_multi_period_guard() -> None:
    panel = _make_cck_staggered_three_period_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with staggered adoption yet",
    ):
        estimate_dose_effects(
            panel, _make_cck_dose_spec(target_parameter="level"), dvals=[0.2, 0.5]
        )

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with staggered adoption yet",
    ):
        estimate_dose_slope_effects(
            panel, _make_cck_dose_spec(target_parameter="slope"), dvals=[0.2, 0.5]
        )


@pytest.mark.parametrize(
    ("estimator", "target_parameter"),
    [
        (estimate_dose_effects, "level"),
        (estimate_dose_slope_effects, "slope"),
    ],
)
def test_cck_dose_staggered_guard_precedes_unchecked_control_group(
    estimator,
    target_parameter: str,
) -> None:
    panel = _make_cck_staggered_two_period_panel()
    spec = ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method="cck",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with staggered adoption yet",
    ):
        estimator(panel, spec)


def test_cck_requires_exactly_two_time_periods() -> None:
    panel = _make_cck_single_cohort_three_period_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with more than two time periods. consider averaging across pre and post treatment periods",
    ):
        estimate_dose_effects(
            panel, _make_cck_dose_spec(target_parameter="level"), dvals=[0.2, 0.5]
        )

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator not supported with more than two time periods. consider averaging across pre and post treatment periods",
    ):
        estimate_dose_slope_effects(
            panel, _make_cck_dose_spec(target_parameter="slope"), dvals=[0.2, 0.5]
        )


def test_cck_requires_positive_timing_in_post_period_with_cck_error() -> None:
    panel = _make_cck_single_cohort_after_window_two_period_panel()
    match = "cck estimator requires positive treatment timing to start in the post period"

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(
            panel, _make_cck_dose_spec(target_parameter="level"), dvals=[0.2, 0.5]
        )

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(
            panel, _make_cck_dose_spec(target_parameter="slope"), dvals=[0.2, 0.5]
        )


def test_cck_dose_routes_reject_unchecked_control_groups() -> None:
    control_group = "eventuallytreated"
    panel = simulate_contdid_data(
        n=4000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )
    level_spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="cck",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=False,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )
    slope_spec = ContDIDSpec(
        target_parameter="slope",
        aggregation="dose",
        dose_est_method="cck",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=False,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    match = "is not supported"
    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_effects(panel, level_spec, dvals=[0.2, 0.5])

    with pytest.raises(ContDIDValidationError, match=match):
        estimate_dose_slope_effects(panel, slope_spec, dvals=[0.2, 0.5])


def test_eventstudy_cck_path_succeeds_fixed_dimension() -> None:
    """CCK with fixed dimension should succeed in event study."""
    panel = simulate_contdid_data(
        n=4000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )

    result_level = estimate_eventstudy_effects(
        panel, _make_cck_eventstudy_spec(target_parameter="level"),
        degree=2, num_knots=0,
    )
    assert result_level.estimand == "ATT(event_time)"
    assert result_level.metadata["dose_est_method"] == "cck"

    result_slope = estimate_eventstudy_slope_effects(
        panel, _make_cck_eventstudy_spec(target_parameter="slope"),
        degree=2, num_knots=0,
    )
    assert result_slope.estimand == "ACRT(event_time)"
    assert result_slope.metadata["dose_est_method"] == "cck"


@pytest.mark.parametrize("base_period", ["universal", 1])
def test_eventstudy_cck_works_with_base_period(
    base_period: str | int,
) -> None:
    """CCK eventstudy should work with different base_period strategies."""
    panel = simulate_contdid_data(
        n=4000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )

    result = estimate_eventstudy_effects(
        panel,
        _make_cck_eventstudy_spec(target_parameter="level"),
        base_period=base_period,
        degree=2,
        num_knots=0,
    )
    assert result.estimand == "ATT(event_time)"
    assert result.metadata["dose_est_method"] == "cck"


def test_eventstudy_cck_rejects_invalid_control_group() -> None:
    """CCK eventstudy should reject invalid control_group."""
    panel = simulate_contdid_data(
        n=4000, dgp_id="SIM-005-cck-two-period", seed=20261234
    )
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="eventuallytreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ):
        estimate_eventstudy_effects(panel, spec)


def test_eventstudy_cck_staggered_small_sample_error() -> None:
    """CCK eventstudy with staggered tiny panel should fail due to sample size."""
    panel = _make_cck_staggered_two_period_panel()

    with pytest.raises(
        ContDIDValidationError,
    ):
        estimate_eventstudy_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="level")
        )

    with pytest.raises(
        ContDIDValidationError,
    ):
        estimate_eventstudy_slope_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="slope")
        )


def test_eventstudy_cck_staggered_rejects_invalid_control_group() -> None:
    """Staggered CCK eventstudy with invalid control_group should fail."""
    panel = _make_cck_staggered_two_period_panel()
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="eventuallytreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ):
        estimate_eventstudy_effects(panel, spec)


def test_eventstudy_cck_staggered_rejects_invalid_control_group_three_period() -> (
    None
):
    """CCK eventstudy with invalid control_group should fail."""
    panel = _make_cck_staggered_three_period_panel()
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="eventuallytreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ):
        estimate_eventstudy_effects(panel, spec)


def test_eventstudy_cck_after_window_timing_raises_appropriate_error() -> None:
    """CCK eventstudy where treatment is after observed window should fail."""
    panel = _make_cck_single_cohort_after_window_two_period_panel()

    with pytest.raises(ContDIDValidationError):
        estimate_eventstudy_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="level")
        )

    with pytest.raises(ContDIDValidationError):
        estimate_eventstudy_slope_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="slope")
        )


@pytest.mark.parametrize(
    ("estimator", "target_parameter"),
    [
        (estimate_eventstudy_effects, "level"),
        (estimate_eventstudy_slope_effects, "slope"),
    ],
)
@pytest.mark.parametrize("base_period", ["universal", 1])
def test_eventstudy_cck_rejects_invalid_control_group_with_base_period(
    estimator,
    target_parameter: str,
    base_period: str | int,
) -> None:
    """CCK eventstudy with invalid control_group should fail regardless of base_period."""
    panel = _make_cck_staggered_three_period_panel()
    spec = ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="eventuallytreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ):
        estimator(panel, spec, base_period=base_period)


def test_eventstudy_cck_multi_period_small_sample_raises_error() -> None:
    """CCK eventstudy with tiny multi-period panel should fail due to sample."""
    panel = _make_cck_single_cohort_three_period_panel()

    with pytest.raises(
        ContDIDValidationError,
    ):
        estimate_eventstudy_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="level")
        )

    with pytest.raises(
        ContDIDValidationError,
    ):
        estimate_eventstudy_slope_effects(
            panel, _make_cck_eventstudy_spec(target_parameter="slope")
        )
