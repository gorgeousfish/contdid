"""Tests for clustered standard errors."""
from __future__ import annotations

import numpy as np
import pytest
from contdid.influence import InfluenceFunction


class TestClusteredCovariance:
    def test_shape(self):
        """Clustered covariance has correct shape."""
        rng = np.random.default_rng(42)
        n, k = 100, 5
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(10), 10)
        cov = inf.clustered_covariance(cluster_ids)
        assert cov.shape == (k, k)

    def test_psd(self):
        """Clustered covariance should be positive semi-definite."""
        rng = np.random.default_rng(42)
        n, k = 200, 4
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(20), 10)
        cov = inf.clustered_covariance(cluster_ids)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-14)

    def test_reduces_to_unit_when_each_unit_is_cluster(self):
        """With n clusters of size 1, clustered = unit-level."""
        rng = np.random.default_rng(42)
        n, k = 50, 3
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        # Each unit is its own cluster
        cluster_ids = np.arange(n)
        cov_clustered = inf.clustered_covariance(cluster_ids)
        cov_unit = inf.covariance()
        np.testing.assert_allclose(cov_clustered, cov_unit, atol=1e-12)

    def test_larger_with_within_cluster_correlation(self):
        """Clustered SE >= unit SE when there's positive within-cluster correlation."""
        rng = np.random.default_rng(42)
        n, k = 100, 3
        n_clusters = 10

        # Generate data with strong within-cluster correlation
        values = np.zeros((n, k))
        for c in range(n_clusters):
            cluster_effect = rng.normal(size=k) * 3.0
            for i in range(n // n_clusters):
                values[c * (n // n_clusters) + i] = cluster_effect + rng.normal(size=k) * 0.5

        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)

        se_unit = inf.standard_error()
        se_cluster = inf.clustered_standard_error(cluster_ids)

        # With strong positive correlation, clustered SE should be larger
        assert np.all(se_cluster > se_unit * 1.5)

    def test_wrong_length_raises(self):
        """Mismatched cluster_ids length should raise."""
        rng = np.random.default_rng(42)
        values = rng.normal(size=(50, 3))
        inf = InfluenceFunction(
            unit_ids=tuple(range(50)), values=values,
            estimand_labels=("a", "b", "c"), n_total=50,
        )
        with pytest.raises(ValueError):
            inf.clustered_covariance(np.arange(30))  # Wrong length


class TestClusteredStandardError:
    def test_nonnegative(self):
        """Clustered SEs must be non-negative."""
        rng = np.random.default_rng(42)
        n, k = 100, 5
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(20), 5)
        se = inf.clustered_standard_error(cluster_ids)
        assert np.all(se >= 0)
        assert se.shape == (k,)


class TestClusteredBootstrap:
    def test_pointwise_output_format(self):
        """Pointwise clustered bootstrap returns correct format."""
        rng = np.random.default_rng(42)
        n, k = 80, 4
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(8), 10)

        result = inf.clustered_multiplier_bootstrap(
            cluster_ids, biters=100, alp=0.05, cband=False, seed=42
        )
        assert "std_error" in result
        assert "critical_value" in result
        assert "n_clusters" in result
        assert result["n_clusters"] == 8
        assert result["confidence_band_kind"] == "pointwise_clustered_multiplier"

    def test_simultaneous_output_format(self):
        """Simultaneous clustered bootstrap returns correct format."""
        rng = np.random.default_rng(42)
        n, k = 80, 4
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(8), 10)

        result = inf.clustered_multiplier_bootstrap(
            cluster_ids, biters=200, alp=0.05, cband=True, seed=42
        )
        assert result["confidence_band_kind"] == "simultaneous_clustered_multiplier"
        assert result["critical_value"] >= result["pointwise_critical_value"]

    def test_reproducible_with_seed(self):
        """Same seed gives same results."""
        rng = np.random.default_rng(42)
        n, k = 60, 3
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        cluster_ids = np.repeat(np.arange(6), 10)

        r1 = inf.clustered_multiplier_bootstrap(cluster_ids, biters=100, cband=True, seed=123)
        r2 = inf.clustered_multiplier_bootstrap(cluster_ids, biters=100, cband=True, seed=123)
        assert r1["critical_value"] == r2["critical_value"]


class TestClusteredWithSpec:
    def test_spec_cluster_column_field(self):
        """ContDIDSpec should accept cluster_column."""
        from contdid.specs import ContDIDSpec
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
            cluster_column="state"
        )
        assert spec.cluster_column == "state"

    def test_spec_default_none(self):
        """Default cluster_column is None."""
        from contdid.specs import ContDIDSpec
        spec = ContDIDSpec(
            target_parameter="level", aggregation="dose",
            dose_est_method="parametric", control_group="nevertreated",
        )
        assert spec.cluster_column is None


class TestBackwardCompatibility:
    def test_existing_influence_tests_unaffected(self):
        """Adding clustered methods doesn't break existing IF functionality."""
        rng = np.random.default_rng(42)
        n, k = 100, 5
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)), values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)), n_total=n,
        )
        # Original methods still work
        se = inf.standard_error()
        cov = inf.covariance()
        assert se.shape == (k,)
        assert cov.shape == (k, k)
        boot = inf.multiplier_bootstrap(biters=50, alp=0.05, cband=False, seed=42)
        assert "std_error" in boot
