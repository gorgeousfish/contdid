"""Tests for influence function infrastructure."""
import numpy as np
import pytest

from contdid.influence import (
    InfluenceFunction,
    aggregate_influence_functions,
    compute_dose_influence_function,
)


class TestInfluenceFunction:
    def test_creation(self):
        """Basic IF creation and validation."""
        values = np.random.default_rng(42).normal(size=(100, 5))
        ids = tuple(range(100))
        labels = tuple(f"d_{i}" for i in range(5))
        inf_func = InfluenceFunction(
            unit_ids=ids, values=values, estimand_labels=labels, n_total=100
        )
        assert inf_func.n_units == 100
        assert inf_func.n_estimands == 5

    def test_covariance_shape(self):
        """Covariance matrix has correct shape."""
        values = np.random.default_rng(42).normal(size=(200, 10))
        inf_func = InfluenceFunction(
            unit_ids=tuple(range(200)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(10)),
            n_total=200,
        )
        cov = inf_func.covariance()
        assert cov.shape == (10, 10)
        # PSD check
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-14)

    def test_standard_error_nonnegative(self):
        """Standard errors are nonnegative."""
        values = np.random.default_rng(42).normal(size=(100, 5))
        inf_func = InfluenceFunction(
            unit_ids=tuple(range(100)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(5)),
            n_total=100,
        )
        se = inf_func.standard_error()
        assert np.all(se >= 0.0)

    def test_mean_approximately_zero(self):
        """E[IF] should be approximately zero for properly centered IF."""
        rng = np.random.default_rng(42)
        n = 5000
        # Simulate regression setup
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + rng.normal(0, 0.5, n)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta_hat
        bread = np.linalg.inv(X.T @ X)

        loadings = np.array([[1.0, 0.5]])  # Single linear functional
        score = X * residual[:, None]
        if_values = score @ bread.T @ loadings.T

        inf_func = InfluenceFunction(
            unit_ids=tuple(range(n)),
            values=if_values,
            estimand_labels=("theta",),
            n_total=n,
        )
        # Mean of IF should be close to zero
        mean_if = np.mean(inf_func.values, axis=0)
        assert np.all(np.abs(mean_if) < 0.1)

    def test_bootstrap_produces_valid_output(self):
        """Multiplier bootstrap returns properly formatted output."""
        values = np.random.default_rng(42).normal(size=(100, 5))
        inf_func = InfluenceFunction(
            unit_ids=tuple(range(100)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(5)),
            n_total=100,
        )
        result = inf_func.multiplier_bootstrap(
            biters=500, alp=0.05, cband=True, seed=123
        )
        assert "std_error" in result
        assert "critical_value" in result
        assert "confidence_band_kind" in result
        assert result["confidence_band_kind"] == "simultaneous_multiplier"
        assert result["critical_value"] >= result["pointwise_critical_value"]

    def test_bootstrap_no_cband(self):
        """Without cband, critical_value equals pointwise."""
        values = np.random.default_rng(42).normal(size=(100, 5))
        inf_func = InfluenceFunction(
            unit_ids=tuple(range(100)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(5)),
            n_total=100,
        )
        result = inf_func.multiplier_bootstrap(
            biters=500, alp=0.05, cband=False, seed=123
        )
        assert result["critical_value"] == result["pointwise_critical_value"]

    def test_se_consistent_with_sandwich(self):
        """IF-based SE should match sandwich estimator SE."""
        rng = np.random.default_rng(42)
        n = 1000
        X = np.column_stack(
            [np.ones(n), rng.uniform(0.1, 1, n), rng.uniform(0.1, 1, n) ** 2]
        )
        beta_true = np.array([1.0, 2.0, -0.5])
        y = X @ beta_true + rng.normal(0, 0.3, n)
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ beta_hat

        # Sandwich covariance
        bread = np.linalg.pinv(X.T @ X)
        score = X * residual[:, None]
        meat = score.T @ score
        sandwich_cov = bread @ meat @ bread

        # Loadings for dose grid evaluation
        dose_grid = np.linspace(0.2, 0.8, 5)
        loadings = np.column_stack([np.ones(5), dose_grid, dose_grid**2])

        # Sandwich SE
        sandwich_se = np.sqrt(
            np.einsum("ij,jk,ik->i", loadings, sandwich_cov, loadings)
        )

        # IF-based SE
        if_values = n * (score @ bread.T @ loadings.T)  # (n, 5) -- scaled by n
        inf_func = InfluenceFunction(
            unit_ids=tuple(range(n)),
            values=if_values,
            estimand_labels=tuple(f"d_{i}" for i in range(5)),
            n_total=n,
        )
        if_se = inf_func.standard_error()

        np.testing.assert_allclose(if_se, sandwich_se, rtol=1e-10)


class TestComputeDoseInfluenceFunction:
    def test_basic_computation(self):
        """compute_dose_influence_function returns correctly shaped IF."""
        rng = np.random.default_rng(42)
        n_treated = 100
        n_grid = 10
        p = 4

        X = rng.normal(size=(n_treated, p))
        residual = rng.normal(size=n_treated)
        coef = rng.normal(size=p)
        loadings = rng.normal(size=(n_grid, p))
        ids = tuple(range(n_treated))
        dose = rng.uniform(0.1, 1.0, n_treated)

        inf_func = compute_dose_influence_function(
            design=X,
            residual=residual,
            coefficients=coef,
            loadings=loadings,
            treated_unit_ids=ids,
            treated_dose=dose,
            n_total=n_treated + 50,
        )
        assert inf_func.n_units == n_treated
        assert inf_func.n_estimands == n_grid

    def test_with_untreated(self):
        """Include untreated units for level estimation."""
        rng = np.random.default_rng(42)
        n_treated = 80
        n_untreated = 20
        p = 3
        n_grid = 5

        X = rng.normal(size=(n_treated, p))
        residual = rng.normal(size=n_treated)
        coef = rng.normal(size=p)
        loadings = rng.normal(size=(n_grid, p))
        treated_ids = tuple(range(n_treated))
        untreated_ids = tuple(range(n_treated, n_treated + n_untreated))
        dose = rng.uniform(0.1, 1.0, n_treated)
        untreated_delta = rng.normal(size=n_untreated)

        inf_func = compute_dose_influence_function(
            design=X,
            residual=residual,
            coefficients=coef,
            loadings=loadings,
            treated_unit_ids=treated_ids,
            treated_dose=dose,
            n_total=n_treated + n_untreated,
            untreated_delta=untreated_delta,
            untreated_unit_ids=untreated_ids,
            include_untreated=True,
        )
        assert inf_func.n_units == n_treated + n_untreated


class TestAggregateInfluenceFunctions:
    def test_basic_aggregation(self):
        """Aggregate two IFs with equal weights."""
        rng = np.random.default_rng(42)
        n1, n2 = 50, 60
        n_grid = 5

        if1 = InfluenceFunction(
            unit_ids=tuple(range(n1)),
            values=rng.normal(size=(n1, n_grid)),
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n1 + n2,
        )
        if2 = InfluenceFunction(
            unit_ids=tuple(range(n1, n1 + n2)),
            values=rng.normal(size=(n2, n_grid)),
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n1 + n2,
        )

        agg = aggregate_influence_functions([if1, if2], weights=[0.5, 0.5])
        assert agg.n_units == n1 + n2
        assert agg.n_estimands == n_grid

    def test_overlapping_units(self):
        """Units appearing in multiple local IFs get summed contributions."""
        rng = np.random.default_rng(42)
        n_grid = 3

        # Overlapping unit IDs
        if1 = InfluenceFunction(
            unit_ids=(0, 1, 2, 3),
            values=np.ones((4, n_grid)),
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=5,
        )
        if2 = InfluenceFunction(
            unit_ids=(2, 3, 4),
            values=np.ones((3, n_grid)) * 2,
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=5,
        )

        agg = aggregate_influence_functions([if1, if2], weights=[1.0, 1.0])
        assert agg.n_units == 5  # Units 0,1,2,3,4
        assert agg.n_total == 1  # Aggregated IF uses n_total=1

        # Unit 2 appears in both: 0.5*1/5 + 0.5*2/5 = 0.3
        # (scaled by w_k / n_k for correct variance)
        idx_2 = list(agg.unit_ids).index(2)
        np.testing.assert_allclose(agg.values[idx_2], 0.3)

    def test_aggregate_unequal_n_total(self):
        """Aggregated SE should be correct with unequal local sample sizes."""
        rng = np.random.default_rng(42)
        n_grid = 3

        # Local IF 1: small sample (n=50)
        n1 = 50
        values1 = rng.normal(size=(n1, n_grid))
        if1 = InfluenceFunction(
            unit_ids=tuple(range(n1)),
            values=values1,
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n1,
        )

        # Local IF 2: large sample (n=500)
        n2 = 500
        values2 = rng.normal(size=(n2, n_grid))
        if2 = InfluenceFunction(
            unit_ids=tuple(range(n1, n1 + n2)),
            values=values2,
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n2,
        )

        # Weights proportional to sample size
        w1, w2 = n1 / (n1 + n2), n2 / (n1 + n2)

        # Correct variance: w1² * Var(θ̂_1) + w2² * Var(θ̂_2)
        # where Var(θ̂_k) = diag(V_k'V_k) / n_k²
        var1 = np.sum(values1**2, axis=0) / n1**2
        var2 = np.sum(values2**2, axis=0) / n2**2
        correct_var = w1**2 * var1 + w2**2 * var2
        correct_se = np.sqrt(correct_var)

        # Aggregated IF SE
        agg = aggregate_influence_functions(
            [if1, if2], weights=[float(n1), float(n2)]
        )
        agg_se = agg.standard_error()

        np.testing.assert_allclose(agg_se, correct_se, rtol=1e-10)

    def test_aggregate_equal_n_total_matches_simple(self):
        """When all local n_total are equal, result matches simple weighted sum."""
        rng = np.random.default_rng(123)
        n_grid = 4
        n = 100  # Same n_total for both

        values1 = rng.normal(size=(n, n_grid))
        if1 = InfluenceFunction(
            unit_ids=tuple(range(n)),
            values=values1,
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n,
        )

        values2 = rng.normal(size=(n, n_grid))
        if2 = InfluenceFunction(
            unit_ids=tuple(range(n, 2 * n)),
            values=values2,
            estimand_labels=tuple(f"d_{i}" for i in range(n_grid)),
            n_total=n,
        )

        agg = aggregate_influence_functions([if1, if2], weights=[1.0, 1.0])

        # With equal n_total=n, normalized weights w1=w2=0.5, scale=0.5/n
        # Var = (0.5/n)² * V1'V1 + (0.5/n)² * V2'V2
        #     = 0.25/n² * (V1'V1 + V2'V2)
        expected_var = 0.25 / n**2 * (
            np.sum(values1**2, axis=0) + np.sum(values2**2, axis=0)
        )
        expected_se = np.sqrt(expected_var)

        np.testing.assert_allclose(agg.standard_error(), expected_se, rtol=1e-10)
