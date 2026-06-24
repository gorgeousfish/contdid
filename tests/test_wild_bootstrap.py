"""Tests for wild bootstrap weight distributions."""
import numpy as np
import pytest
from contdid.influence import InfluenceFunction, _draw_bootstrap_weights


class TestDrawBootstrapWeights:
    def test_gaussian_shape(self):
        rng = np.random.default_rng(42)
        w = _draw_bootstrap_weights(rng, (100, 5), "multiplier")
        assert w.shape == (100, 5)

    def test_rademacher_values(self):
        """Rademacher weights should only be ±1."""
        rng = np.random.default_rng(42)
        w = _draw_bootstrap_weights(rng, (1000,), "rademacher")
        unique = set(np.unique(w))
        assert unique == {-1.0, 1.0}

    def test_rademacher_moments(self):
        """E[ξ]≈0, Var(ξ)≈1 for Rademacher."""
        rng = np.random.default_rng(42)
        w = _draw_bootstrap_weights(rng, (100000,), "rademacher")
        assert abs(np.mean(w)) < 0.02
        assert abs(np.var(w) - 1.0) < 0.02

    def test_mammen_moments(self):
        """E[ξ]≈0, Var(ξ)≈1 for Mammen."""
        rng = np.random.default_rng(42)
        w = _draw_bootstrap_weights(rng, (100000,), "mammen")
        assert abs(np.mean(w)) < 0.02
        assert abs(np.var(w) - 1.0) < 0.02

    def test_mammen_two_values(self):
        """Mammen draws should have exactly 2 unique values."""
        rng = np.random.default_rng(42)
        w = _draw_bootstrap_weights(rng, (10000,), "mammen")
        assert len(np.unique(w)) == 2

    def test_invalid_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            _draw_bootstrap_weights(rng, (10,), "invalid_type")


class TestMultiplierBootstrapTypes:
    @pytest.fixture
    def inf_func(self):
        rng = np.random.default_rng(42)
        n, k = 200, 5
        values = rng.normal(size=(n, k))
        return InfluenceFunction(
            unit_ids=tuple(range(n)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)),
            n_total=n,
        )

    def test_default_is_gaussian(self, inf_func):
        """Default boot_type should be 'multiplier' (Gaussian)."""
        r = inf_func.multiplier_bootstrap(biters=50, cband=True, seed=42)
        # Should work without specifying boot_type
        assert r["critical_value"] > 0

    def test_rademacher_works(self, inf_func):
        r = inf_func.multiplier_bootstrap(
            biters=100, cband=True, seed=42, boot_type="rademacher"
        )
        assert r["critical_value"] > 0

    def test_mammen_works(self, inf_func):
        r = inf_func.multiplier_bootstrap(
            biters=100, cband=True, seed=42, boot_type="mammen"
        )
        assert r["critical_value"] > 0

    def test_invalid_boot_type_raises(self, inf_func):
        with pytest.raises(ValueError):
            inf_func.multiplier_bootstrap(biters=50, boot_type="invalid")

    def test_all_types_give_similar_cv(self, inf_func):
        """All weight types should give similar critical values (large sample)."""
        r_g = inf_func.multiplier_bootstrap(
            biters=500, cband=True, seed=42, boot_type="multiplier"
        )
        r_r = inf_func.multiplier_bootstrap(
            biters=500, cband=True, seed=42, boot_type="rademacher"
        )
        r_m = inf_func.multiplier_bootstrap(
            biters=500, cband=True, seed=42, boot_type="mammen"
        )
        # Should all be in a reasonable range of each other
        cvs = [r_g["critical_value"], r_r["critical_value"], r_m["critical_value"]]
        assert max(cvs) - min(cvs) < 1.5  # Not too different

    def test_se_same_across_types(self, inf_func):
        """SE should be identical regardless of boot_type (analytical from IF)."""
        r_g = inf_func.multiplier_bootstrap(
            biters=50, cband=False, seed=42, boot_type="multiplier"
        )
        r_r = inf_func.multiplier_bootstrap(
            biters=50, cband=False, seed=42, boot_type="rademacher"
        )
        assert r_g["std_error"] == r_r["std_error"]


class TestClusteredBootstrapTypes:
    def test_clustered_rademacher(self):
        rng = np.random.default_rng(42)
        n, k = 100, 3
        values = rng.normal(size=(n, k))
        inf = InfluenceFunction(
            unit_ids=tuple(range(n)),
            values=values,
            estimand_labels=tuple(f"d_{i}" for i in range(k)),
            n_total=n,
        )
        cluster_ids = np.repeat(np.arange(10), 10)
        r = inf.clustered_multiplier_bootstrap(
            cluster_ids, biters=100, cband=True, seed=42, boot_type="rademacher"
        )
        assert r["critical_value"] > 0

    def test_clustered_invalid_raises(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(50, 3))
        inf = InfluenceFunction(
            unit_ids=tuple(range(50)),
            values=values,
            estimand_labels=("a", "b", "c"),
            n_total=50,
        )
        cluster_ids = np.repeat(np.arange(5), 10)
        with pytest.raises(ValueError):
            inf.clustered_multiplier_bootstrap(cluster_ids, biters=50, boot_type="bad")
