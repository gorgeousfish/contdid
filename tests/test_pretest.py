"""Tests for pre-trend testing module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from contdid.pretest import (
    PreTrendTestResult,
    pre_trend_test,
    _compute_wald_statistic,
)


def _make_panel_no_pretrend(n_control=80, n_treated=60, n_periods=4, treatment_time=3, seed=42):
    """Panel with parallel trends (no pre-trend). H0 should NOT be rejected.

    Dose is unit-level constant (same across all periods). Group encodes
    treatment timing. Outcome shifts only at/after treatment_time.
    """
    rng = np.random.default_rng(seed)
    records = []
    uid = 0
    for i in range(n_control):
        for t in range(1, n_periods + 1):
            records.append({"id": uid, "time": t, "outcome": rng.normal(0, 0.5),
                          "dose": 0.0, "group": 0})
        uid += 1
    for i in range(n_treated):
        dose_val = rng.uniform(0.1, 1.0)
        for t in range(1, n_periods + 1):
            effect = (1.0 + 2.0 * dose_val) if t >= treatment_time else 0.0
            records.append({"id": uid, "time": t, "outcome": rng.normal(0, 0.3) + effect,
                          "dose": dose_val, "group": treatment_time})
        uid += 1
    return pd.DataFrame(records)


def _make_panel_with_pretrend(n_control=80, n_treated=60, n_periods=5, treatment_time=4, seed=42):
    """Panel with pre-existing differential trends. H0 SHOULD be rejected.

    Dose is unit-level constant. Pre-trend is dose-proportional across all
    pre-treatment periods, making the parallel trends assumption violated.
    """
    rng = np.random.default_rng(seed)
    records = []
    uid = 0
    for i in range(n_control):
        for t in range(1, n_periods + 1):
            records.append({"id": uid, "time": t, "outcome": rng.normal(0, 0.3),
                          "dose": 0.0, "group": 0})
        uid += 1
    for i in range(n_treated):
        dose_val = rng.uniform(0.1, 1.0)
        for t in range(1, n_periods + 1):
            # Pre-trend: treated units have differential trend even before treatment
            pre_effect = 0.8 * dose_val * (t - 1)  # Growing pre-trend
            post_effect = (2.0 + 3.0 * dose_val) if t >= treatment_time else 0.0
            records.append({"id": uid, "time": t,
                          "outcome": rng.normal(0, 0.2) + pre_effect + post_effect,
                          "dose": dose_val, "group": treatment_time})
        uid += 1
    return pd.DataFrame(records)


class TestWaldStatistic:
    def test_zero_estimates_gives_zero_stat(self):
        """All zeros should give W=0, p=1."""
        est = np.zeros(3)
        cov = np.eye(3) * 0.01
        W, p, df = _compute_wald_statistic(est, cov)
        assert W == 0.0
        assert p == 1.0
        assert df == 3

    def test_known_statistic(self):
        """Known case: single parameter, theta=1, sigma^2=0.25 -> W=4."""
        est = np.array([1.0])
        cov = np.array([[0.25]])
        W, p, df = _compute_wald_statistic(est, cov)
        np.testing.assert_allclose(W, 4.0, atol=1e-10)
        assert df == 1
        # p-value for chi2(1) at 4 ~ 0.0455
        expected_p = 1.0 - stats.chi2.cdf(4.0, df=1)
        np.testing.assert_allclose(p, expected_p, atol=1e-10)

    def test_multivariate_known(self):
        """Multi-dimensional with identity covariance: W = ||theta||^2."""
        est = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        W, p, df = _compute_wald_statistic(est, cov)
        np.testing.assert_allclose(W, 14.0, atol=1e-10)  # 1+4+9
        assert df == 3

    def test_singular_covariance_handled(self):
        """Near-singular covariance should not crash (uses pinv)."""
        est = np.array([0.1, 0.2])
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        W, p, df = _compute_wald_statistic(est, cov)
        assert W >= 0
        assert 0 <= p <= 1


class TestPreTrendNoEffect:
    def test_parallel_trends_not_rejected(self):
        """With true parallel trends, test should usually not reject."""
        df = _make_panel_no_pretrend(n_control=100, n_treated=80, seed=42)
        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose", biters=200, 
        )
        assert isinstance(result, PreTrendTestResult)
        assert result.p_value > 0.01  # Should not strongly reject
        assert result.degrees_of_freedom >= 1

    def test_result_fields_populated(self):
        """All result fields should be properly populated."""
        df = _make_panel_no_pretrend(seed=123)
        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose", biters=100, 
        )
        assert result.test_statistic >= 0
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom == len(result.pre_period_estimates)
        assert len(result.pre_period_se) == result.degrees_of_freedom
        assert len(result.pre_period_event_times) == result.degrees_of_freedom
        assert result.covariance_matrix.shape == (result.degrees_of_freedom, result.degrees_of_freedom)
        assert all(e < 0 for e in result.pre_period_event_times)


class TestPreTrendWithEffect:
    def test_differential_pretrend_rejected(self):
        """With strong pre-existing differential trend, test should reject."""
        df = _make_panel_with_pretrend(n_control=120, n_treated=100, seed=42)
        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose", biters=300, 
        )
        # Strong pre-trend should be detected
        assert result.p_value < 0.10  # Should reject at 10% at least
        assert result.test_statistic > stats.chi2.ppf(0.90, df=result.degrees_of_freedom)


class TestEdgeCases:
    def test_single_pre_period(self):
        """With 2 pre-periods (4 total, treat at 3), one is base -> df=1."""
        df = _make_panel_no_pretrend(n_periods=4, treatment_time=3, seed=42)
        # periods 1,2,3,4; treated at 3; event_times: -2,-1(base),0,1
        # Only event_time=-2 is testable pre-period
        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose", biters=100, 
        )
        assert result.degrees_of_freedom >= 1

    def test_multiple_pre_periods(self):
        """With 3+ pre-periods, df should be > 1."""
        df = _make_panel_no_pretrend(n_periods=6, treatment_time=5, seed=42)
        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose", biters=100, 
        )
        assert result.degrees_of_freedom >= 2

    def test_insufficient_periods_raises(self):
        """Panel with only 2 periods should raise error (no pre-period available)."""
        rng = np.random.default_rng(42)
        records = []
        for i in range(50):
            records.append({"id": i, "time": 1, "outcome": rng.normal(), "dose": 0.0, "group": 0})
            records.append({"id": i, "time": 2, "outcome": rng.normal(), "dose": 0.0, "group": 0})
        for i in range(50, 100):
            dose_val = 0.5
            records.append({"id": i, "time": 1, "outcome": rng.normal(), "dose": dose_val, "group": 2})
            records.append({"id": i, "time": 2, "outcome": rng.normal() + 1, "dose": dose_val, "group": 2})
        df = pd.DataFrame(records)

        with pytest.raises(Exception):
            pre_trend_test(
                df, id_column="id", time_column="time", outcome_column="outcome",
                group_column="group", dose_column="dose", biters=50, 
            )

    def test_notyettreated_control(self):
        """Should work with notyettreated control group."""
        df = _make_panel_no_pretrend(n_periods=5, treatment_time=4, seed=42)
        # Add a later cohort (treated at time 5, with unit-level dose)
        rng = np.random.default_rng(99)
        extra = []
        uid = 200
        for i in range(30):
            dose_val = rng.uniform(0.1, 0.8)
            for t in range(1, 6):
                extra.append({"id": uid, "time": t, "outcome": rng.normal(0, 0.3),
                            "dose": dose_val, "group": 5})  # Treated at time 5
            uid += 1
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

        result = pre_trend_test(
            df, id_column="id", time_column="time", outcome_column="outcome",
            group_column="group", dose_column="dose",
            control_group="notyettreated", biters=100, 
        )
        assert isinstance(result, PreTrendTestResult)


class TestBackwardCompatibility:
    def test_existing_eventstudy_unchanged(self):
        """Pre-trend test should not affect existing event-study estimation."""
        from contdid import PanelData, ContDIDSpec
        from contdid.eventstudy import estimate_eventstudy_effects

        df = _make_panel_no_pretrend(n_periods=4, seed=42)
        panel = PanelData(frame=df, id_column="id", time_column="time",
                         outcome_column="outcome", group_column="group", dose_column="dose")
        spec = ContDIDSpec(
            target_parameter="level", aggregation="eventstudy",
            dose_est_method="parametric", control_group="nevertreated"
        )

        # This should still work exactly as before
        result = estimate_eventstudy_effects(panel, spec)
        assert result is not None
