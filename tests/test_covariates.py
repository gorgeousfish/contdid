"""Tests for covariate conditioning in dose estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Helper: Generate test data with known covariate effects
# ============================================================

def _make_panel_with_covariates(
    n_treated=150, n_untreated=50, seed=42, n_covariates=2
):
    """Create a 2-period panel with known covariate effects.

    DGP: ΔY_i = 1 + 2*D_i + 0.5*D_i^2 + 3*X1_i - X2_i + noise
    True dose effect: f(d) = 1 + 2d + 0.5d^2
    Covariate effects: 3*X1 - X2 (additive, not interacting with dose)

    Panel convention: dose column is unit-level (constant across time periods).
    Untreated units have dose=0; treated units carry their dose in all rows.
    Group column encodes treatment timing (0=never, 2=treated at t=2).
    """
    rng = np.random.default_rng(seed)
    n = n_treated + n_untreated
    records = []

    for i in range(n):
        x1 = rng.normal(0, 1)
        x2 = rng.uniform(-1, 1)

        if i < n_treated:
            dose = rng.uniform(0.1, 1.0)
            group = 2  # treated at time 2
        else:
            dose = 0.0
            group = 0  # never treated

        # Pre-period outcome
        y_pre = rng.normal(0, 0.5) + 0.5 * x1

        # Post-period outcome
        if dose > 0:
            effect = 1.0 + 2.0 * dose + 0.5 * dose**2 + 3.0 * x1 - x2
        else:
            effect = 0.0
        y_post = y_pre + effect + rng.normal(0, 0.2)

        # Dose is unit-level: constant across both time periods
        records.append({"id": i, "time": 1, "outcome": y_pre, "dose": dose,
                        "group": group, "X1": x1, "X2": x2})
        records.append({"id": i, "time": 2, "outcome": y_post, "dose": dose,
                        "group": group, "X1": x1, "X2": x2})

    return pd.DataFrame(records)


# ============================================================
# A. Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_no_covariates_same_as_before(self):
        """covariates=None should produce identical results to pre-covariate code."""
        from contdid import PanelData, ContDIDSpec
        from contdid.estimation import estimate_dose_effects

        df = _make_panel_with_covariates(n_treated=100, n_untreated=40)
        panel = PanelData(frame=df, id_column="id", time_column="time",
                         outcome_column="outcome", group_column="group",
                         dose_column="dose")

        # Without covariates
        spec_no_cov = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated"
        )

        # With covariates=None explicitly
        spec_none = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            covariates=None
        )

        # Both should work identically
        assert spec_no_cov.covariates is None
        assert spec_none.covariates is None

        r1 = estimate_dose_effects(panel, spec_no_cov, degree=2, num_knots=0)
        r2 = estimate_dose_effects(panel, spec_none, degree=2, num_knots=0)

        np.testing.assert_allclose(r1.estimate, r2.estimate)
        np.testing.assert_allclose(r1.std_error, r2.std_error)

    def test_spec_accepts_covariates_tuple(self):
        """ContDIDSpec should accept covariates as tuple of strings."""
        from contdid import ContDIDSpec
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            covariates=("X1", "X2")
        )
        assert spec.covariates == ("X1", "X2")


# ============================================================
# B. Known DGP Accuracy
# ============================================================

class TestKnownDGPAccuracy:
    def test_covariate_adjustment_runs_without_crash(self):
        """With covariates parameter, run_cck_backend should not crash."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 2))
        # True DGP with large covariate effect
        true_f = 1.0 + 2.0 * dose + 0.5 * dose**2
        dy = true_f + 5.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, 0.2, n)

        untreated = rng.normal(0, 0.5, 50)
        dvals = [0.3, 0.5, 0.7]

        # With covariates (should run without error)
        r_cov = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3, num_knots=0,
            covariates=X
        )

        # Check return is a dict with expected keys
        assert isinstance(r_cov, dict)
        assert "att_curve" in r_cov
        assert "att_se" in r_cov
        assert len(r_cov["att_curve"]) == 3

    def test_covariate_adjustment_improves_accuracy(self):
        """With covariates, ATT(d) curve should be closer to truth than without."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 2))
        # True DGP with large covariate effect
        true_f = 1.0 + 2.0 * dose + 0.5 * dose**2
        dy = true_f + 5.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, 0.2, n)

        untreated = rng.normal(0, 0.5, 50)
        dvals = [0.3, 0.5, 0.7]

        # Without covariates
        r_no_cov = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3, num_knots=0
        )

        # With covariates
        r_cov = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3, num_knots=0,
            covariates=X
        )

        # True values at grid
        true_at_grid = 1.0 + 2.0 * np.array(dvals) + 0.5 * np.array(dvals)**2

        err_no_cov = np.max(np.abs(np.array(r_no_cov["att_curve"]) - true_at_grid))
        err_cov = np.max(np.abs(np.array(r_cov["att_curve"]) - true_at_grid))

        # With covariates should be closer to truth
        assert err_cov < err_no_cov

    def test_covariates_reduce_standard_errors(self):
        """Controlling for covariates should reduce SE when covariates explain variance."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 1))
        # Large covariate effect = lots of residual variance without X
        dy = 1.0 + 2.0 * dose + 10.0 * X[:, 0] + rng.normal(0, 0.1, n)

        untreated = rng.normal(0, 0.5, 40)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r_no = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=100, degree=3, num_knots=0
        )

        r_yes = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=100, degree=3, num_knots=0,
            covariates=X
        )

        # SE with covariates should be smaller
        se_no = np.mean(r_no["att_se"])
        se_yes = np.mean(r_yes["att_se"])
        assert se_yes < se_no


# ============================================================
# C. Frisch-Waugh Equivalence
# ============================================================

class TestFrischWaugh:
    def test_manual_residualization_equivalent(self):
        """Partialling out covariates manually should give same dose coefficients.

        B-spline basis forms a partition of unity, so we do NOT add a separate
        intercept to X when residualizing — just residualize on X directly.
        """
        from contdid.bspline import build_bspline_design

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 2))
        dy = 1.0 + 2.0 * dose + 3.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, n)

        knots = []
        degree = 3

        # Method 1: Joint regression [B-spline, X]
        bs = build_bspline_design(dose, degree, knots)
        design_full = np.column_stack([bs, X])
        coef_full, *_ = np.linalg.lstsq(design_full, dy, rcond=None)
        coef_dose_full = coef_full[:bs.shape[1]]

        # Method 2: Frisch-Waugh (residualize on X — no intercept since
        # B-spline basis already spans constants)
        gamma, *_ = np.linalg.lstsq(X, dy, rcond=None)
        dy_resid = dy - X @ gamma
        # Residualize B-spline basis on X
        bs_resid = np.zeros_like(bs)
        for j in range(bs.shape[1]):
            g, *_ = np.linalg.lstsq(X, bs[:, j], rcond=None)
            bs_resid[:, j] = bs[:, j] - X @ g
        # Regress residualized dy on residualized basis
        coef_dose_fw, *_ = np.linalg.lstsq(bs_resid, dy_resid, rcond=None)

        # Coefficients should be identical (up to numerical precision)
        np.testing.assert_allclose(coef_dose_full, coef_dose_fw, atol=1e-6)


# ============================================================
# D. Standard Error Correctness
# ============================================================

class TestStandardErrors:
    def test_se_positive_with_covariates(self):
        """Standard errors should be positive with covariates."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 2))
        dy = 1.0 + 2.0 * dose + X[:, 0] + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 40)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=200, degree=3,
            covariates=X
        )

        assert all(s > 0 for s in r["att_se"])
        assert all(s > 0 for s in r["acrt_se"])

    def test_confidence_interval_contains_truth_with_covariates(self):
        """Confidence intervals should cover the true value at most grid points."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 400
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 1))
        true_f = 1.0 + 2.0 * dose
        dy = true_f + 3.0 * X[:, 0] + rng.normal(0, 0.1, n)
        untreated = rng.normal(0, 0.3, 60)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=500, degree=2,
            covariates=X
        )

        true_at_grid = 1.0 + 2.0 * np.array(dvals)
        intervals = r["att_interval"]
        covered = sum(
            1 for i, d in enumerate(dvals)
            if intervals[i][0] <= true_at_grid[i] <= intervals[i][1]
        )
        # At least 3 out of 5 should cover (relaxed for finite sample)
        assert covered >= 3


# ============================================================
# E. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_covariate(self):
        """Single covariate should work."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 1))
        dy = 1.0 + 2.0 * dose + 3.0 * X[:, 0] + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 30)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3,
            covariates=X
        )
        assert r is not None
        assert len(r["att_curve"]) == 5

    def test_many_covariates(self):
        """Multiple covariates should work."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=(n, 5))
        dy = 1.0 + dose + X @ rng.normal(size=5) + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 50)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3,
            covariates=X
        )
        assert r is not None
        assert len(r["att_curve"]) == 5

    def test_1d_covariate_array(self):
        """1D covariate array should be auto-promoted to 2D."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        X = rng.normal(size=n)  # 1D array
        dy = 1.0 + 2.0 * dose + 3.0 * X + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 30)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3,
            covariates=X
        )
        assert r is not None
        assert len(r["att_curve"]) == 5

    def test_multiperiod_with_covariates(self):
        """Multi-period estimation with covariates raises NotImplementedError.

        CGBS Theorem 3.1 identifies LATT under unconditional parallel trends;
        covariate adjustment in multi-period decomposition lacks theory support.
        """
        from contdid.multiperiod import estimate_multiperiod_dose

        rng = np.random.default_rng(42)
        records = []
        uid = 0
        for i in range(60):
            x1 = rng.normal()
            for t in [1, 2, 3]:
                records.append({"id": uid, "time": t,
                              "outcome": rng.normal(0, 0.5),
                              "dose": 0.0, "group": 0, "X1": x1})
            uid += 1
        for g_time in [2, 3]:
            for i in range(40):
                x1 = rng.normal()
                dose_val = rng.uniform(0.1, 1.0)
                for t in [1, 2, 3]:
                    effect = (1.0 + 2.0 * dose_val + x1) if t >= g_time else 0.0
                    records.append({"id": uid, "time": t,
                                  "outcome": rng.normal(0, 0.3) + effect,
                                  "dose": dose_val if t >= g_time else 0.0,
                                  "group": g_time, "X1": x1})
                uid += 1
        df = pd.DataFrame(records)

        import pytest
        with pytest.raises(NotImplementedError, match="Covariate adjustment"):
            estimate_multiperiod_dose(
                df, id_column="id", time_column="time", outcome_column="outcome",
                dose_column="dose", group_column="group",
                dose_grid=np.array([0.3, 0.5, 0.7]), degree=1, num_knots=0,
                covariates=["X1"], biters=100, cband=False, boot_seed=42,
            )

    def test_parametric_estimation_with_covariates_raises(self):
        """Full parametric estimation pipeline with covariates raises NotImplementedError.

        The paper (arXiv:2107.02637v7) does not provide the complete estimation
        theory for covariate conditioning (influence function correction formula
        and bootstrap coverage guarantees are missing).
        """
        from contdid import PanelData, ContDIDSpec
        from contdid.estimation import estimate_dose_effects

        df = _make_panel_with_covariates(n_treated=150, n_untreated=50)
        panel = PanelData(frame=df, id_column="id", time_column="time",
                         outcome_column="outcome", group_column="group",
                         dose_column="dose")
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            covariates=("X1", "X2")
        )

        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_dose_effects(panel, spec, degree=3, num_knots=0)

    def test_validation_missing_column_raises_not_implemented(self):
        """Specifying covariates should raise NotImplementedError at validation stage,
        regardless of whether the column exists."""
        from contdid import PanelData, ContDIDSpec
        from contdid.validation import validate_spec

        df = _make_panel_with_covariates(n_treated=50, n_untreated=20)
        panel = PanelData(frame=df, id_column="id", time_column="time",
                         outcome_column="outcome", group_column="group",
                         dose_column="dose")
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            covariates=("NONEXISTENT",)
        )

        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            validate_spec(spec, panel=panel)

    def test_validation_nan_covariate_raises_not_implemented(self):
        """NaN in covariate column is irrelevant: covariates are blocked at validation."""
        from contdid import PanelData, ContDIDSpec
        from contdid.validation import validate_spec

        df = _make_panel_with_covariates(n_treated=50, n_untreated=20)
        df.loc[0, "X1"] = np.nan  # Introduce NaN
        panel = PanelData(frame=df, id_column="id", time_column="time",
                         outcome_column="outcome", group_column="group",
                         dose_column="dose")
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            covariates=("X1",)
        )

        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            validate_spec(spec, panel=panel)

    def test_orthogonal_covariates_preserve_dose_curve(self):
        """Covariates uncorrelated with dose should barely change ATT(d) curve."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        # Covariates that are independent of dose and have tiny true coefficient
        X = rng.normal(size=(n, 2)) * 0.001
        dy = 1.0 + 2.0 * dose + rng.normal(0, 0.1, n)
        untreated = rng.normal(0, 0.3, 40)
        dvals = list(np.linspace(0.2, 0.8, 5))

        r_no_cov = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3, num_knots=0
        )
        r_with_cov = run_cck_backend(
            dy, dose, dvals, untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=False, cband=False, alp=0.05, degree=3, num_knots=0,
            covariates=X
        )

        # With near-zero covariates, the dose curve should be nearly identical
        np.testing.assert_allclose(
            r_no_cov["att_curve"], r_with_cov["att_curve"], atol=0.05
        )
