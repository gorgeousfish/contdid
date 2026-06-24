"""Tests for multi-period 2x2 dose estimation."""
import numpy as np
import pandas as pd
import pytest

from contdid.multiperiod import (
    MultiPeriodDoseResult,
    estimate_multiperiod_dose,
    _identify_dose_timing_groups,
)
from contdid.validation import ContDIDValidationError


def _make_staggered_panel(
    n_units=200, n_periods=5, n_cohorts=3, seed=42
):
    """Create a synthetic staggered panel for testing.

    - Units 0..n_never are never-treated (group=0)
    - Remaining units are split across cohorts treated at times 2,3,4
    - Dose is drawn from Uniform(0.1, 1.0) for treated
    - True DGP: delta_Y = 1.0 + 2.0*dose + noise
    """
    rng = np.random.default_rng(seed)
    n_never = n_units // 4
    n_treated = n_units - n_never
    units_per_cohort = n_treated // n_cohorts

    records = []
    unit_id = 0

    # Never-treated units
    for i in range(n_never):
        dose = 0.0
        group = 0
        for t in range(1, n_periods + 1):
            y = rng.normal(0, 0.5)
            records.append({
                "id": unit_id, "time": t, "outcome": y,
                "dose": dose, "group": group,
            })
        unit_id += 1

    # Treated cohorts
    for c in range(n_cohorts):
        treatment_time = c + 2  # Treated at times 2, 3, 4
        for i in range(units_per_cohort):
            dose = rng.uniform(0.1, 1.0)
            group = treatment_time
            for t in range(1, n_periods + 1):
                if t >= treatment_time:
                    # Post-treatment: add treatment effect
                    effect = 1.0 + 2.0 * dose
                    y = effect + rng.normal(0, 0.3)
                else:
                    y = rng.normal(0, 0.5)
                records.append({
                    "id": unit_id, "time": t, "outcome": y,
                    "dose": dose if t >= treatment_time else 0.0, "group": group,
                })
            unit_id += 1

    return pd.DataFrame(records)


class TestIdentifyTimingGroups:
    def test_finds_all_cohorts(self):
        df = _make_staggered_panel()
        groups = _identify_dose_timing_groups(
            df, group_column="group", time_column="time",
            id_column="id", dose_column="dose", control_group="nevertreated"
        )
        # 3 cohorts (g=2,3,4), each with all t>=g in 5-period panel:
        # g=2: t=2,3,4,5 (4 pairs); g=3: t=3,4,5 (3 pairs); g=4: t=4,5 (2 pairs)
        assert len(groups) == 9
        # Verify all cohorts are represented
        unique_groups = set(g["group"] for g in groups)
        assert unique_groups == {2, 3, 4}
        # Verify event_time field is present
        for g_info in groups:
            assert "event_time" in g_info
            assert g_info["event_time"] == g_info["time"] - g_info["group"]

    def test_notyettreated_has_more_controls(self):
        df = _make_staggered_panel()
        groups_never = _identify_dose_timing_groups(
            df, group_column="group", time_column="time",
            id_column="id", dose_column="dose", control_group="nevertreated"
        )
        groups_nyt = _identify_dose_timing_groups(
            df, group_column="group", time_column="time",
            id_column="id", dose_column="dose", control_group="notyettreated"
        )
        # Not-yet-treated should have >= same number of controls for early cohorts
        assert groups_nyt[0]["n_control"] >= groups_never[0]["n_control"]

    def test_skips_groups_without_pre_period(self):
        """Group treated at time 1 (first period) should be skipped."""
        df = _make_staggered_panel(n_cohorts=1)
        # Override: make group treated at time 1
        df["group"] = df["group"].replace({2: 1})
        groups = _identify_dose_timing_groups(
            df, group_column="group", time_column="time",
            id_column="id", dose_column="dose", control_group="nevertreated"
        )
        # Should skip cohort treated at time 1 (no pre-period)
        assert len(groups) == 0


class TestMultiPeriodDoseEstimation:
    def test_basic_estimation(self):
        """Multi-period estimation runs and produces reasonable output."""
        df = _make_staggered_panel(n_units=200)
        dose_grid = np.linspace(0.2, 0.9, 10)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=200,
            cband=False,
            boot_seed=42,
        )

        assert isinstance(result, MultiPeriodDoseResult)
        assert len(result.dose_grid) == 10
        assert len(result.point_estimate) == 10
        assert len(result.standard_error) == 10

    def test_point_estimates_reasonable(self):
        """With known DGP (1+2d), estimates should be in ballpark."""
        df = _make_staggered_panel(n_units=500, seed=123)
        dose_grid = np.array([0.3, 0.5, 0.7])

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            cband=False,
            boot_seed=42,
        )

        # True effect at d: 1 + 2*d
        expected = 1.0 + 2.0 * dose_grid
        estimates = np.array(result.point_estimate)
        # Should be within 0.5 of true (generous for noise)
        np.testing.assert_allclose(estimates, expected, atol=0.5)

    def test_confidence_band(self):
        """Confidence band contains point estimates."""
        df = _make_staggered_panel(n_units=200)
        dose_grid = np.linspace(0.2, 0.9, 5)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            num_knots=0,
            biters=500,
            cband=True,
            alp=0.05,
            boot_seed=42,
        )

        for i in range(len(dose_grid)):
            assert result.confidence_band_lower[i] <= result.point_estimate[i]
            assert result.point_estimate[i] <= result.confidence_band_upper[i]

    def test_slope_estimation(self):
        """ACRT(d) estimation works."""
        df = _make_staggered_panel(n_units=300)
        dose_grid = np.linspace(0.2, 0.9, 5)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            num_knots=0,
            target="slope",
            biters=100,
            cband=False,
            boot_seed=42,
        )

        # With linear DGP (1+2d), the slope should be approximately 2 everywhere
        slopes = np.array(result.point_estimate)
        np.testing.assert_allclose(slopes, 2.0, atol=0.8)

    def test_notyettreated_control(self):
        """notyettreated control group works."""
        df = _make_staggered_panel(n_units=200)
        dose_grid = np.linspace(0.2, 0.9, 5)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            num_knots=0,
            control_group="notyettreated",
            biters=100,
            cband=False,
            boot_seed=42,
        )

        assert result.metadata["control_group"] == "notyettreated"
        assert len(result.point_estimate) == 5

    def test_metadata_completeness(self):
        """Result metadata has all expected fields."""
        df = _make_staggered_panel(n_units=200)
        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            degree=2,
            num_knots=1,
            biters=100,
            cband=False,
            boot_seed=42,
        )

        assert result.metadata["estimator"] == "multiperiod_dose_2x2"
        assert result.metadata["degree"] == 2
        assert result.metadata["num_knots"] == 1
        assert result.metadata["n_groups"] >= 1
        assert result.metadata["basis_type"] == "bspline"

    def test_empty_panel_raises(self):
        """Empty or invalid panel raises ContDIDValidationError."""
        df = pd.DataFrame({
            "id": [1, 1], "time": [1, 2], "outcome": [0.0, 1.0],
            "dose": [0.0, 0.0], "group": [0, 0]
        })

        with pytest.raises(ContDIDValidationError):
            estimate_multiperiod_dose(
                df,
                id_column="id",
                time_column="time",
                outcome_column="outcome",
                dose_column="dose",
                group_column="group",
            )

    def test_with_bspline_knots(self):
        """Estimation with interior knots works."""
        df = _make_staggered_panel(n_units=300)
        dose_grid = np.linspace(0.2, 0.9, 8)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=3,
            num_knots=2,
            biters=100,
            cband=False,
            boot_seed=42,
        )

        assert len(result.point_estimate) == 8
        assert result.metadata["num_knots"] == 2

    def test_default_dose_grid(self):
        """Default dose grid is auto-generated when not specified."""
        df = _make_staggered_panel(n_units=200)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            degree=1,
            num_knots=0,
            biters=100,
            cband=False,
            boot_seed=42,
        )

        assert len(result.dose_grid) == 20  # Default 20 quantile points

    def test_invalid_target_raises(self):
        """Invalid target parameter raises."""
        df = _make_staggered_panel(n_units=200)
        with pytest.raises(ContDIDValidationError):
            estimate_multiperiod_dose(
                df,
                id_column="id",
                time_column="time",
                outcome_column="outcome",
                dose_column="dose",
                group_column="group",
                target="invalid",
            )

    def test_invalid_control_group_raises(self):
        """Invalid control_group raises."""
        df = _make_staggered_panel(n_units=200)
        with pytest.raises(ContDIDValidationError):
            estimate_multiperiod_dose(
                df,
                id_column="id",
                time_column="time",
                outcome_column="outcome",
                dose_column="dose",
                group_column="group",
                control_group="invalid",
            )


class TestLocalResults:
    def test_local_results_populated(self):
        """Local (g,t) results are reported."""
        df = _make_staggered_panel(n_units=200, n_cohorts=3)
        dose_grid = np.linspace(0.2, 0.9, 5)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            biters=100,
            cband=False,
            boot_seed=42,
        )

        assert len(result.local_results) >= 1
        for lr in result.local_results:
            assert "group" in lr
            assert "time" in lr
            assert "n_treated" in lr
            assert "weight" in lr

    def test_weights_sum_to_one(self):
        """Local weights should sum to 1."""
        df = _make_staggered_panel(n_units=200, n_cohorts=3)
        dose_grid = np.linspace(0.2, 0.9, 5)

        result = estimate_multiperiod_dose(
            df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            dose_column="dose",
            group_column="group",
            dose_grid=dose_grid,
            degree=1,
            biters=100,
            cband=False,
            boot_seed=42,
        )

        total_weight = sum(lr["weight"] for lr in result.local_results)
        np.testing.assert_allclose(total_weight, 1.0, atol=1e-10)
