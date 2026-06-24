"""Tests for Lepski adaptive dimension selection."""
from __future__ import annotations

import numpy as np
import pytest
from contdid.lepski import (
    LepskiResult,
    select_lepski_dimension,
    _build_candidate_grid,
    _fit_sieve_at_dimension,
    _compute_pointwise_se,
    _compute_pairwise_variance,
    _multiplier_bootstrap_critical_value,
)


# ============================================================
# A. Candidate Grid Construction
# ============================================================

class TestBuildCandidateGrid:
    def test_grid_is_ascending(self):
        """Grid elements must be strictly increasing."""
        grid = _build_candidate_grid(500, degree=3)
        assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))

    def test_grid_dyadic_structure(self):
        """Grid elements follow {2^l + degree} pattern."""
        grid = _build_candidate_grid(1000, degree=3)
        # All elements must be from dyadic sequence 2^l + 3
        dyadic_set = {2**l + 3 for l in range(20)}
        for k in grid:
            assert k in dyadic_set, f"{k} not in dyadic set"

    def test_grid_respects_k_min(self):
        """All elements >= k_min."""
        grid = _build_candidate_grid(500, degree=3, k_min=7)
        assert all(k >= 7 for k in grid)

    def test_grid_respects_k_max(self):
        """All elements <= k_max."""
        grid = _build_candidate_grid(500, degree=3, k_max=20)
        assert all(k <= 20 for k in grid)

    def test_minimum_two_candidates(self):
        """Grid should have at least 2 candidates when possible."""
        grid = _build_candidate_grid(100, degree=3)
        assert len(grid) >= 2

    def test_larger_n_allows_higher_dimensions(self):
        """More data should allow higher K_max."""
        grid_small = _build_candidate_grid(100, degree=3)
        grid_large = _build_candidate_grid(5000, degree=3)
        assert max(grid_large) >= max(grid_small)

    def test_small_sample(self):
        """Very small n should still produce valid grid."""
        grid = _build_candidate_grid(30, degree=3)
        assert len(grid) >= 1
        assert all(k >= 4 for k in grid)  # minimum is degree + 1

    def test_very_large_n(self):
        """Large n should produce grid without error."""
        grid = _build_candidate_grid(10000, degree=3)
        assert len(grid) >= 3
        assert grid == sorted(grid)


# ============================================================
# B. Sieve Fitting
# ============================================================

class TestFitSieveAtDimension:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        delta_y = 1.0 + 2.0 * dose - 0.5 * dose**2 + rng.normal(0, 0.2, n)
        grid = np.linspace(0.15, 0.95, 20)
        return delta_y, dose, grid

    def test_output_shape(self, sample_data):
        """Fitted values shape matches grid."""
        dy, dose, grid = sample_data
        fit = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        assert fit["fitted_at_grid"].shape == (20,)
        assert fit["residuals"].shape == (200,)
        assert fit["design"].shape[0] == 200
        assert fit["design"].shape[1] == 5  # dimension = degree + 1 + num_interior_knots

    def test_output_keys(self, sample_data):
        """All expected keys are returned."""
        dy, dose, grid = sample_data
        fit = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        expected_keys = {
            "coefficients", "fitted_at_grid", "derivative_at_grid",
            "residuals", "design", "eval_design", "eval_deriv_design",
            "bread", "hat_values",
        }
        assert set(fit.keys()) == expected_keys

    def test_residuals_orthogonal_to_design(self, sample_data):
        """OLS residuals are orthogonal to design matrix."""
        dy, dose, grid = sample_data
        fit = _fit_sieve_at_dimension(dy, dose, dimension=7, degree=3, eval_grid=grid)
        ortho = fit["design"].T @ fit["residuals"]
        np.testing.assert_allclose(ortho, 0.0, atol=1e-8)

    def test_higher_dim_lower_residual(self, sample_data):
        """Higher dimension fits should have lower or equal training error."""
        dy, dose, grid = sample_data
        fit5 = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        fit7 = _fit_sieve_at_dimension(dy, dose, dimension=7, degree=3, eval_grid=grid)
        sse5 = np.sum(fit5["residuals"]**2)
        sse7 = np.sum(fit7["residuals"]**2)
        assert sse7 <= sse5 + 1e-10

    def test_known_polynomial_recovery(self):
        """Fit should recover known polynomial perfectly."""
        rng = np.random.default_rng(123)
        n = 500
        dose = rng.uniform(0.1, 1.0, n)
        # Quadratic: needs dim >= 3 in cubic spline space
        delta_y = 1.0 + 2.0 * dose + 3.0 * dose**2  # No noise
        grid = np.linspace(0.2, 0.9, 10)
        fit = _fit_sieve_at_dimension(delta_y, dose, dimension=4, degree=3, eval_grid=grid)
        true_vals = 1.0 + 2.0 * grid + 3.0 * grid**2
        np.testing.assert_allclose(fit["fitted_at_grid"], true_vals, atol=1e-4)

    def test_hat_values_in_range(self, sample_data):
        """Hat values should be in [0, 1]."""
        dy, dose, grid = sample_data
        fit = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        assert np.all(fit["hat_values"] >= -1e-10)
        assert np.all(fit["hat_values"] <= 1.0 + 1e-10)


# ============================================================
# C. Variance Estimation
# ============================================================

class TestVarianceEstimation:
    def test_pointwise_se_positive(self):
        """Standard errors must be positive."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)
        fit = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        se = _compute_pointwise_se(fit)
        assert np.all(se > 0)
        assert se.shape == (10,)

    def test_pairwise_variance_positive(self):
        """Pairwise standard deviation should be positive."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)
        fit5 = _fit_sieve_at_dimension(dy, dose, dimension=5, degree=3, eval_grid=grid)
        fit7 = _fit_sieve_at_dimension(dy, dose, dimension=7, degree=3, eval_grid=grid)
        sigma = _compute_pairwise_variance(fit5, fit7, n)
        assert np.all(sigma > 0)
        assert sigma.shape == (10,)

    def test_se_decreases_with_n(self):
        """Standard errors should decrease with larger samples."""
        rng = np.random.default_rng(42)
        grid = np.linspace(0.2, 0.8, 5)

        dose100 = rng.uniform(0.1, 1.0, 100)
        dy100 = 1.0 + dose100 + rng.normal(0, 0.5, 100)
        fit100 = _fit_sieve_at_dimension(dy100, dose100, dimension=5, degree=3, eval_grid=grid)
        se100 = _compute_pointwise_se(fit100)

        dose500 = rng.uniform(0.1, 1.0, 500)
        dy500 = 1.0 + dose500 + rng.normal(0, 0.5, 500)
        fit500 = _fit_sieve_at_dimension(dy500, dose500, dimension=5, degree=3, eval_grid=grid)
        se500 = _compute_pointwise_se(fit500)

        assert np.mean(se500) < np.mean(se100)

    def test_pointwise_se_shape_matches_grid(self):
        """SE shape should match eval_grid."""
        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)
        grid = np.linspace(0.2, 0.8, 15)
        fit = _fit_sieve_at_dimension(dy, dose, dimension=6, degree=3, eval_grid=grid)
        se = _compute_pointwise_se(fit)
        assert se.shape == (15,)


# ============================================================
# D. Bootstrap Critical Value
# ============================================================

class TestBootstrapCriticalValue:
    def test_critical_value_positive(self):
        """Critical value must be positive."""
        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)

        dims = [4, 5, 7]
        fits = {}
        for d in dims:
            fits[d] = _fit_sieve_at_dimension(dy, dose, dimension=d, degree=3, eval_grid=grid)

        cv, alpha = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=200, seed=42
        )
        assert cv > 0
        assert 0 < alpha <= 0.5

    def test_alpha_hat_range(self):
        """alpha_hat should be in (0, 0.5]."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)

        dims = [4, 5, 7, 11]
        fits = {
            d: _fit_sieve_at_dimension(dy, dose, dimension=d, degree=3, eval_grid=grid)
            for d in dims
        }

        _, alpha = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=100, seed=42
        )
        assert 0 < alpha <= 0.5

    def test_different_seeds_give_different_values(self):
        """Different seeds should produce (slightly) different critical values."""
        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)

        dims = [4, 5, 7]
        fits = {
            d: _fit_sieve_at_dimension(dy, dose, dimension=d, degree=3, eval_grid=grid)
            for d in dims
        }

        cv1, _ = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=200, seed=42
        )
        cv2, _ = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=200, seed=99
        )
        # They should be in similar range but not identical
        assert cv1 != cv2
        assert abs(cv1 - cv2) / max(cv1, cv2) < 0.5  # Within 50% of each other

    def test_same_seed_reproducible(self):
        """Same seed should produce identical critical values."""
        rng = np.random.default_rng(42)
        n = 150
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)
        grid = np.linspace(0.2, 0.8, 10)

        dims = [4, 5, 7]
        fits = {
            d: _fit_sieve_at_dimension(dy, dose, dimension=d, degree=3, eval_grid=grid)
            for d in dims
        }

        cv1, a1 = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=200, seed=42
        )
        cv2, a2 = _multiplier_bootstrap_critical_value(
            fits, dims, n, bootstrap_reps=200, seed=42
        )
        assert cv1 == cv2
        assert a1 == a2


# ============================================================
# E. Lepski Selection Rule
# ============================================================

class TestLepskiSelectionRule:
    def test_quadratic_dgp_selects_reasonable_dim(self):
        """For quadratic DGP, should select dim >= 3."""
        rng = np.random.default_rng(42)
        n = 400
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0 * dose + 0.5 * dose**2 + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=200, seed=42)
        assert result.selected_dimension >= 3

    def test_linear_dgp_selects_low_dim(self):
        """For simple linear DGP, should not select unnecessarily high dim."""
        rng = np.random.default_rng(42)
        n = 400
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 3.0 * dose + rng.normal(0, 0.2, n)

        result = select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=200, seed=42)
        # Linear needs only dim >= 2, Lepski should pick something modest
        assert result.selected_dimension <= 35  # Not too high

    def test_returns_lepski_result_type(self):
        """select_lepski_dimension returns a LepskiResult."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=100, seed=42)
        assert isinstance(result, LepskiResult)
        assert result.selected_dimension in result.candidate_dimensions
        assert result.bootstrap_critical_value > 0
        assert len(result.fitted_values) == len(result.eval_grid)
        assert len(result.standard_errors) == len(result.eval_grid)

    def test_fitted_values_reasonable(self):
        """Fitted values should be close to true function."""
        rng = np.random.default_rng(42)
        n = 500
        dose = rng.uniform(0.1, 1.0, n)
        true_func = lambda d: 1.0 + 2.0 * d + 0.5 * d**2
        dy = true_func(dose) + rng.normal(0, 0.2, n)

        result = select_lepski_dimension(
            dy, dose, degree=3, bootstrap_reps=200, seed=42, grid_size=10
        )
        true_at_grid = true_func(result.eval_grid)
        errors = np.abs(result.fitted_values - true_at_grid)
        # Most points should be within 2*SE of truth
        within_2se = np.sum(errors <= 2 * result.standard_errors)
        assert within_2se >= 7  # At least 70% within 2 SE

    def test_contrast_statistics_populated(self):
        """Contrast statistics should be computed for all candidates."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=100, seed=42)
        # contrast_statistics should have an entry for each candidate dimension
        for dim in result.candidate_dimensions:
            assert dim in result.contrast_statistics
        assert all(v >= 0 for v in result.contrast_statistics.values())

    def test_metadata_contains_params(self):
        """Metadata should contain key parameters."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(
            dy, dose, degree=3, bootstrap_reps=100, seed=42
        )
        assert result.metadata["n"] == 200
        assert result.metadata["degree"] == 3
        assert result.metadata["bootstrap_reps"] == 100
        assert result.metadata["seed"] == 42


# ============================================================
# F. End-to-end Integration
# ============================================================

class TestEndToEndIntegration:
    def test_cck_adaptive_mode(self):
        """run_cck_backend with adaptive=True works end-to-end."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0 * dose + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 50)
        dvals = np.linspace(0.2, 0.8, 10)

        result = run_cck_backend(
            dy, dose, dvals,
            untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=100,
            degree=3, num_knots=0,
            adaptive=True, adaptive_seed=42,
        )

        assert isinstance(result, dict)
        assert "lepski" in result
        assert result["lepski"]["adaptive"] is True
        assert result["lepski"]["selected_dimension"] >= 4

    def test_cck_non_adaptive_unchanged(self):
        """run_cck_backend with adaptive=False does not include lepski metadata."""
        from contdid.cck import run_cck_backend

        rng = np.random.default_rng(42)
        n = 100
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)
        untreated = rng.normal(0, 0.5, 30)
        dvals = np.linspace(0.2, 0.8, 5)

        result = run_cck_backend(
            dy, dose, dvals,
            untreated_delta=untreated,
            require_untreated_variance_df=True,
            bstrap=True, cband=False, alp=0.05, biters=100,
            degree=3, num_knots=0,
            adaptive=False,
        )

        assert isinstance(result, dict)
        assert "lepski" not in result


# ============================================================
# G. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_very_small_sample(self):
        """Algorithm should not crash with small n."""
        rng = np.random.default_rng(42)
        n = 25
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)

        result = select_lepski_dimension(dy, dose, degree=2, bootstrap_reps=50, seed=42)
        assert result.selected_dimension >= 3  # degree + 1

    def test_constant_dose_raises(self):
        """All-same dose values should raise an error."""
        n = 100
        dose = np.ones(n) * 0.5
        dy = np.random.default_rng(42).normal(size=n)

        with pytest.raises(Exception):  # ContDIDValidationError
            select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=50, seed=42)

    def test_mismatched_lengths_raises(self):
        """Mismatched delta_outcome and dose lengths should raise."""
        rng = np.random.default_rng(42)
        dose = rng.uniform(0.1, 1.0, 100)
        dy = rng.normal(size=50)

        with pytest.raises(Exception):
            select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=50, seed=42)

    def test_too_small_sample_raises(self):
        """Extremely small sample (< degree + 2) should raise."""
        dose = np.array([0.2, 0.5, 0.8])
        dy = np.array([1.0, 1.5, 2.0])

        with pytest.raises(Exception):
            select_lepski_dimension(dy, dose, degree=3, bootstrap_reps=50, seed=42)

    def test_single_candidate_shortcut(self):
        """When grid produces single candidate, should return without bootstrap."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        # Force single candidate by setting k_min == k_max very high
        # so that only one dimension fits in the grid
        grid = _build_candidate_grid(n, degree=3, k_min=67, k_max=67)
        # If grid has only 1 element, select_lepski_dimension uses shortcut
        if len(grid) == 1:
            result = select_lepski_dimension(
                dy, dose, degree=3, k_min=67, k_max=67,
                bootstrap_reps=50, seed=42
            )
            assert result.bootstrap_critical_value == 0.0
            assert result.alpha_hat == 0.5

    def test_custom_eval_grid(self):
        """Custom eval_grid should be respected."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)
        custom_grid = np.array([0.2, 0.4, 0.6, 0.8])

        result = select_lepski_dimension(
            dy, dose, degree=3, eval_grid=custom_grid, bootstrap_reps=100, seed=42
        )
        np.testing.assert_array_equal(result.eval_grid, custom_grid)
        assert len(result.fitted_values) == 4
        assert len(result.standard_errors) == 4

    def test_grid_size_parameter(self):
        """grid_size parameter controls eval_grid length when eval_grid is None."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(
            dy, dose, degree=3, grid_size=15, bootstrap_reps=100, seed=42
        )
        assert len(result.eval_grid) == 15
        assert len(result.fitted_values) == 15


# ============================================================
# H. Dimension Bounds Enforcement
# ============================================================

class TestDimensionBoundsEnforcement:
    def test_k_min_enforced(self):
        """k_min should be respected - no candidates below it."""
        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0*dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, k_min=7, bootstrap_reps=100, seed=42)
        assert result.selected_dimension >= 7
        assert all(k >= 7 for k in result.candidate_dimensions)

    def test_k_max_enforced(self):
        """k_max should be respected - no candidates above it."""
        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0*dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, k_max=11, bootstrap_reps=100, seed=42)
        assert result.selected_dimension <= 11
        assert all(k <= 11 for k in result.candidate_dimensions)

    def test_k_min_k_max_both(self):
        """Both bounds together constrain the search."""
        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0*dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, k_min=5, k_max=11, bootstrap_reps=100, seed=42)
        assert 5 <= result.selected_dimension <= 11
        assert all(5 <= k <= 11 for k in result.candidate_dimensions)

    def test_k_min_equals_k_max_shortcircuit(self):
        """When k_min == k_max, the function returns that dimension directly."""
        rng = np.random.default_rng(42)
        n = 300
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0*dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(dy, dose, degree=3, k_min=7, k_max=7, bootstrap_reps=50, seed=42)
        assert result.selected_dimension == 7
        assert result.candidate_dimensions == (7,)
        assert result.bootstrap_critical_value == 0.0

    def test_build_candidate_grid_bounds(self):
        """_build_candidate_grid respects k_min and k_max directly."""
        grid = _build_candidate_grid(300, degree=3, k_min=7, k_max=19)
        assert all(7 <= k <= 19 for k in grid)

        grid2 = _build_candidate_grid(500, degree=3, k_min=11, k_max=35)
        assert all(11 <= k <= 35 for k in grid2)


# ============================================================
# I. Infeasible Bounds
# ============================================================

class TestInfeasibleBounds:
    def test_k_min_greater_than_k_max_raises(self):
        """Contradictory bounds (k_min > k_max) should raise."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises((ValueError, Exception)):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=20, k_max=5,
                bootstrap_reps=50, seed=42
            )

    def test_k_min_too_large_for_sample_raises(self):
        """k_min larger than sample allows should raise."""
        rng = np.random.default_rng(42)
        n = 30  # Small sample
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises((ValueError, Exception)):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=25,
                bootstrap_reps=50, seed=42
            )

    def test_no_feasible_candidates_raises(self):
        """When no dyadic grid point falls in [k_min, k_max], should raise."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        # Dyadic grid for degree=3: [4, 5, 7, 11, 19, 35, ...]
        # Range [8, 10] has no dyadic points
        with pytest.raises((ValueError, Exception)):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=8, k_max=10,
                bootstrap_reps=50, seed=42
            )

    def test_selected_never_exceeds_k_max(self):
        """Selected dimension must NEVER exceed k_max."""
        rng = np.random.default_rng(42)
        n = 500
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0*dose + 3.0*dose**2 + rng.normal(0, 0.1, n)

        for k_max in [5, 7, 11]:
            result = select_lepski_dimension(
                dy, dose, degree=3, k_max=k_max,
                bootstrap_reps=100, seed=42
            )
            assert result.selected_dimension <= k_max, \
                f"selected {result.selected_dimension} > k_max {k_max}"

    def test_selected_never_below_k_min(self):
        """Selected dimension must NEVER be below k_min."""
        rng = np.random.default_rng(42)
        n = 500
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.5, n)

        for k_min in [5, 7, 11]:
            result = select_lepski_dimension(
                dy, dose, degree=3, k_min=k_min,
                bootstrap_reps=100, seed=42
            )
            assert result.selected_dimension >= k_min, \
                f"selected {result.selected_dimension} < k_min {k_min}"

    def test_build_grid_k_min_greater_k_max_raises(self):
        """_build_candidate_grid with k_min > k_max raises ValueError."""
        with pytest.raises(ValueError):
            _build_candidate_grid(500, degree=3, k_min=20, k_max=5)

    def test_build_grid_no_feasible_range_raises(self):
        """_build_candidate_grid with empty feasible range raises."""
        with pytest.raises(Exception):
            _build_candidate_grid(200, degree=3, k_min=8, k_max=10)

    def test_k_min_k_max_equal_but_infeasible_raises(self):
        """k_min == k_max but too large for sample should raise."""
        rng = np.random.default_rng(42)
        n = 30
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises(Exception):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=20, k_max=20,
                bootstrap_reps=50, seed=42
            )


# ============================================================
# J. Single-Dimension Validation
# ============================================================

class TestSingleDimensionValidation:
    """Tests for the k_min == k_max short-circuit validation gap fix."""

    def test_single_dim_below_minimum_raises(self):
        """k_min == k_max == 2 with degree=3 is invalid (needs >= 4)."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises(Exception):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=2, k_max=2,
                bootstrap_reps=50, seed=42
            )

    def test_single_dim_exceeds_sample_raises(self):
        """k_min == k_max == 100 with n=50 is invalid."""
        rng = np.random.default_rng(42)
        n = 50
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises(Exception):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=100, k_max=100,
                bootstrap_reps=50, seed=42
            )

    def test_single_dim_zero_raises(self):
        """k_min == k_max == 0 is invalid."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises(Exception):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=0, k_max=0,
                bootstrap_reps=50, seed=42
            )

    def test_single_dim_negative_raises(self):
        """k_min == k_max == -1 is invalid."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        with pytest.raises(Exception):
            select_lepski_dimension(
                dy, dose, degree=3, k_min=-1, k_max=-1,
                bootstrap_reps=50, seed=42
            )

    def test_single_dim_valid_works(self):
        """k_min == k_max == 7 with n=200 should work fine."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        result = select_lepski_dimension(
            dy, dose, degree=3, k_min=7, k_max=7,
            bootstrap_reps=50, seed=42
        )
        assert result.selected_dimension == 7

    def test_single_dim_exactly_at_minimum_works(self):
        """k_min == k_max == degree+1 is the minimum valid dimension."""
        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        # degree=3 -> minimum dimension is 4
        result = select_lepski_dimension(
            dy, dose, degree=3, k_min=4, k_max=4,
            bootstrap_reps=50, seed=42
        )
        assert result.selected_dimension == 4

    def test_build_grid_single_dim_below_minimum_raises(self):
        """_build_candidate_grid with k_min==k_max < degree+1 raises."""
        from contdid.validation import ContDIDValidationError

        with pytest.raises(ContDIDValidationError):
            _build_candidate_grid(200, degree=3, k_min=2, k_max=2)

    def test_build_grid_single_dim_zero_raises(self):
        """_build_candidate_grid with k_min==k_max==0 raises."""
        from contdid.validation import ContDIDValidationError

        with pytest.raises(ContDIDValidationError):
            _build_candidate_grid(200, degree=3, k_min=0, k_max=0)


# ============================================================
# K. Performance
# ============================================================

class TestPerformance:
    def test_moderate_sample_completes_quickly(self):
        """n=500, bootstrap_reps=200 should complete in reasonable time."""
        import time

        rng = np.random.default_rng(42)
        n = 500
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + 2.0 * dose + rng.normal(0, 0.3, n)

        start = time.time()
        result = select_lepski_dimension(
            dy, dose, degree=3, bootstrap_reps=200, seed=42, grid_size=30
        )
        elapsed = time.time() - start

        assert elapsed < 30.0  # Should complete within 30 seconds
        assert result.selected_dimension >= 4

    def test_small_bootstrap_is_fast(self):
        """Small bootstrap_reps should be fast."""
        import time

        rng = np.random.default_rng(42)
        n = 200
        dose = rng.uniform(0.1, 1.0, n)
        dy = 1.0 + dose + rng.normal(0, 0.3, n)

        start = time.time()
        result = select_lepski_dimension(
            dy, dose, degree=3, bootstrap_reps=50, seed=42, grid_size=10
        )
        elapsed = time.time() - start

        assert elapsed < 10.0
        assert isinstance(result, LepskiResult)
