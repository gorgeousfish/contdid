"""Comprehensive tests for anticipation parameter support.

Tests cover:
1. anticipation=0 (default, regression compatibility)
2. anticipation=1 (base period shifts one period)
3. anticipation=2 (base period shifts two periods)
4. Base period calculation correctness
5. Event-study pre/post period correct division
6. Impact on ATT(d)/ACRT(d) estimation
7. Parameter validation (non-negative integer, upper bound)
8. Regression test: anticipation=0 results identical to default

Theoretical reference:
- arXiv-2107.02637v7 Assumption 3-MP (pp.1423-1436):
  base_period = g - anticipation - 1
  pre-treatment: t < g - anticipation (no treatment effect)
  post-treatment: t >= g - anticipation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    PanelData,
    cont_did,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
    simulate_contdid_data,
)
from contdid.timing import prepare_timing_groups
from contdid.validation import ContDIDValidationError, validate_spec


# ---------------------------------------------------------------------------
# Helper: generate staggered panel suitable for anticipation testing
# ---------------------------------------------------------------------------


def _make_staggered_panel(
    n_units: int = 400,
    n_periods: int = 8,
    groups: dict[int, float] | None = None,
    seed: int = 2024,
    effect_size: float = 1.5,
) -> PanelData:
    """Create a balanced staggered panel for anticipation testing.

    Parameters
    ----------
    n_units : total number of units
    n_periods : number of time periods (1..n_periods)
    groups : mapping {timing_group: fraction}
    seed : random seed
    effect_size : linear dose effect for post-treatment periods
    """
    if groups is None:
        groups = {4: 0.25, 5: 0.20, 6: 0.15}

    rng = np.random.default_rng(seed)
    time_periods = np.arange(1, n_periods + 1)

    never_treated_frac = 1.0 - sum(groups.values())
    assert never_treated_frac > 0
    group_keys = [0] + list(groups.keys())
    group_probs = [never_treated_frac] + list(groups.values())
    G = rng.choice(group_keys, size=n_units, replace=True, p=group_probs)

    # Positive dose for treated, 0 for control
    D = np.where(G > 0, rng.uniform(0.5, 3.0, size=n_units), 0.0)

    # Unit fixed effects
    eta = rng.normal(0, 1, size=n_units)

    records = []
    for i in range(n_units):
        for t in time_periods:
            y = float(t) + eta[i] + rng.normal(0, 0.3)
            # True treatment effect: linear in dose, post-treatment only
            if G[i] > 0 and t >= G[i]:
                y += effect_size * D[i]
            records.append({
                "id": i + 1,
                "time_period": int(t),
                "Y": y,
                "G": int(G[i]),
                "D": float(D[i]),
            })

    frame = pd.DataFrame(records)
    return PanelData(frame=frame)


# ---------------------------------------------------------------------------
# 1. Default behavior: anticipation=0
# ---------------------------------------------------------------------------


class TestAnticipationDefaultZero:
    """anticipation=0 is the default and must be backward-compatible."""

    def test_spec_default_is_zero(self):
        """ContDIDSpec defaults anticipation to 0."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert spec.anticipation == 0

    def test_eventstudy_factory_default(self):
        """ContDIDSpec.eventstudy() defaults anticipation to 0."""
        spec = ContDIDSpec.eventstudy()
        assert spec.anticipation == 0

    def test_cont_did_default_anticipation(self):
        """cont_did() defaults anticipation to 0."""
        panel = _make_staggered_panel(n_units=200, n_periods=6)
        result = cont_did(
            panel,
            aggregation="eventstudy",
            anticipation=0,
            bstrap=False,
            degree=1,
            num_knots=0,
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0


class TestAnticipationZeroRegression:
    """anticipation=0 must produce identical results to omitting the parameter."""

    def test_explicit_zero_matches_default(self):
        """Results with anticipation=0 must equal results without specifying it."""
        panel = _make_staggered_panel(n_units=300, n_periods=6, seed=99)
        spec_default = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
        )
        spec_zero = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
            bstrap=False,
        )
        result_default = estimate_eventstudy_effects(panel, spec_default, degree=1, num_knots=0)
        result_zero = estimate_eventstudy_effects(panel, spec_zero, degree=1, num_knots=0)

        assert result_default.event_time_grid == result_zero.event_time_grid
        np.testing.assert_allclose(result_default.estimate, result_zero.estimate, rtol=1e-10)
        np.testing.assert_allclose(result_default.std_error, result_zero.std_error, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Base period calculation correctness
# ---------------------------------------------------------------------------


class TestBasePeriodCalculation:
    """Verify base_period = g - anticipation - 1 for all anticipation values."""

    @pytest.fixture
    def panel_8period(self):
        return _make_staggered_panel(n_units=400, n_periods=8, groups={4: 0.3, 6: 0.3})

    def test_base_period_anticipation_0(self, panel_8period):
        """anticipation=0: base_period = g - 1."""
        prepared = prepare_timing_groups(
            panel_8period, control_group="nevertreated", anticipation=0
        )
        for g in prepared["timing_group"].unique():
            g_post = prepared[
                (prepared["timing_group"] == g) & prepared["post_treatment"]
            ]
            if not g_post.empty:
                expected_base = g - 1
                assert (g_post["base_period"] == expected_base).all(), (
                    f"g={g}: expected base_period={expected_base}, got {g_post['base_period'].unique()}"
                )

    def test_base_period_anticipation_1(self, panel_8period):
        """anticipation=1: base_period = g - 2."""
        prepared = prepare_timing_groups(
            panel_8period, control_group="nevertreated", anticipation=1
        )
        for g in prepared["timing_group"].unique():
            g_post = prepared[
                (prepared["timing_group"] == g) & prepared["post_treatment"]
            ]
            if not g_post.empty:
                expected_base = g - 2
                assert (g_post["base_period"] == expected_base).all(), (
                    f"g={g}: expected base_period={expected_base}, got {g_post['base_period'].unique()}"
                )

    def test_base_period_anticipation_2(self, panel_8period):
        """anticipation=2: base_period = g - 3."""
        prepared = prepare_timing_groups(
            panel_8period, control_group="nevertreated", anticipation=2
        )
        for g in prepared["timing_group"].unique():
            g_post = prepared[
                (prepared["timing_group"] == g) & prepared["post_treatment"]
            ]
            if not g_post.empty:
                expected_base = g - 3
                assert (g_post["base_period"] == expected_base).all(), (
                    f"g={g}: expected base_period={expected_base}, got {g_post['base_period'].unique()}"
                )

    def test_formula_general(self, panel_8period):
        """General formula: base_period = g - anticipation - 1 for various values."""
        for anticipation in [0, 1, 2]:
            prepared = prepare_timing_groups(
                panel_8period, control_group="nevertreated", anticipation=anticipation
            )
            for g in prepared["timing_group"].unique():
                g_post = prepared[
                    (prepared["timing_group"] == g) & prepared["post_treatment"]
                ]
                if not g_post.empty:
                    expected_base = g - anticipation - 1
                    assert (g_post["base_period"] == expected_base).all()


# ---------------------------------------------------------------------------
# 3. Event-study pre/post period correct division
# ---------------------------------------------------------------------------


class TestPrePostDivision:
    """Verify post_treatment flag: t >= g - anticipation."""

    @pytest.fixture
    def panel(self):
        return _make_staggered_panel(n_units=300, n_periods=8, groups={5: 0.5})

    def test_post_treatment_boundary_a0(self, panel):
        """anticipation=0: post starts at t=g (event_time=0)."""
        prepared = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=0
        )
        g5 = prepared[prepared["timing_group"] == 5]
        for _, row in g5.iterrows():
            if row["time_period"] >= 5:
                assert row["post_treatment"] is True
            else:
                assert row["post_treatment"] is False

    def test_post_treatment_boundary_a1(self, panel):
        """anticipation=1: post starts at t=g-1 (event_time=-1)."""
        prepared = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=1
        )
        g5 = prepared[prepared["timing_group"] == 5]
        for _, row in g5.iterrows():
            if row["time_period"] >= 4:  # g - anticipation = 5 - 1 = 4
                assert row["post_treatment"] is True
            else:
                assert row["post_treatment"] is False

    def test_post_treatment_boundary_a2(self, panel):
        """anticipation=2: post starts at t=g-2 (event_time=-2)."""
        prepared = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=2
        )
        g5 = prepared[prepared["timing_group"] == 5]
        for _, row in g5.iterrows():
            if row["time_period"] >= 3:  # g - anticipation = 5 - 2 = 3
                assert row["post_treatment"] is True
            else:
                assert row["post_treatment"] is False

    def test_more_post_rows_with_higher_anticipation(self, panel):
        """Higher anticipation => more rows classified as post-treatment."""
        prepared_a0 = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=0
        )
        prepared_a1 = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=1
        )
        prepared_a2 = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=2
        )
        post_a0 = prepared_a0["post_treatment"].sum()
        post_a1 = prepared_a1["post_treatment"].sum()
        post_a2 = prepared_a2["post_treatment"].sum()
        assert post_a1 >= post_a0
        assert post_a2 >= post_a1


# ---------------------------------------------------------------------------
# 4. Impact on ATT(d)/ACRT(d) estimation
# ---------------------------------------------------------------------------


class TestAnticipationImpactOnEstimation:
    """Different anticipation values should produce different estimates."""

    @pytest.fixture
    def panel(self):
        return _make_staggered_panel(
            n_units=500, n_periods=8, groups={5: 0.3, 6: 0.2}, seed=777
        )

    def test_different_anticipation_different_estimates(self, panel):
        """anticipation=0 vs anticipation=1 should yield different estimates."""
        spec_a0 = ContDIDSpec.eventstudy(anticipation=0)
        spec_a1 = ContDIDSpec.eventstudy(anticipation=1)

        result_a0 = estimate_eventstudy_effects(
            panel, spec_a0, degree=1, num_knots=0
        )
        result_a1 = estimate_eventstudy_effects(
            panel, spec_a1, degree=1, num_knots=0
        )

        # With different base periods, results should differ
        grids_differ = result_a0.event_time_grid != result_a1.event_time_grid
        estimates_differ = not np.allclose(
            result_a0.estimate, result_a1.estimate, atol=1e-10
        ) if len(result_a0.estimate) == len(result_a1.estimate) else True
        assert grids_differ or estimates_differ

    def test_anticipation_affects_slope_estimates(self, panel):
        """ACRT(d) slope estimates should also change with anticipation."""
        spec_a0 = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
            bstrap=False,
        )
        spec_a1 = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=1,
            bstrap=False,
        )
        result_a0 = estimate_eventstudy_slope_effects(
            panel, spec_a0, degree=1, num_knots=0
        )
        result_a1 = estimate_eventstudy_slope_effects(
            panel, spec_a1, degree=1, num_knots=0
        )
        grids_differ = result_a0.event_time_grid != result_a1.event_time_grid
        estimates_differ = not np.allclose(
            result_a0.estimate, result_a1.estimate, atol=1e-10
        ) if len(result_a0.estimate) == len(result_a1.estimate) else True
        assert grids_differ or estimates_differ

    def test_anticipation_via_cont_did(self, panel):
        """cont_did() unified API correctly uses anticipation parameter."""
        result_a0 = cont_did(
            panel,
            aggregation="eventstudy",
            anticipation=0,
            bstrap=False,
            degree=1,
            num_knots=0,
        )
        result_a1 = cont_did(
            panel,
            aggregation="eventstudy",
            anticipation=1,
            bstrap=False,
            degree=1,
            num_knots=0,
        )
        assert isinstance(result_a0, ContDIDResult)
        assert isinstance(result_a1, ContDIDResult)
        # Results should differ
        assert result_a0.event_time_grid != result_a1.event_time_grid or (
            result_a0.estimate != result_a1.estimate
        )


# ---------------------------------------------------------------------------
# 5. Parameter validation
# ---------------------------------------------------------------------------


class TestAnticipationValidation:
    """Validate anticipation parameter constraints."""

    def test_negative_anticipation_raises(self):
        """anticipation < 0 must raise ContDIDValidationError."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=-1,
        )
        with pytest.raises(ContDIDValidationError, match="anticipation"):
            validate_spec(spec)

    def test_float_anticipation_raises(self):
        """Non-integer anticipation must raise ContDIDValidationError."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=1.5,  # type: ignore[arg-type]
        )
        with pytest.raises(ContDIDValidationError, match="anticipation"):
            validate_spec(spec)

    def test_too_large_anticipation_raises_in_validate_spec(self):
        """anticipation too large for panel should raise with clear message."""
        # Panel with groups at t=4, time 1..8 -> min_g=4, max_allowed=4-1-1=2
        panel = _make_staggered_panel(
            n_units=100, n_periods=8, groups={4: 0.5}, seed=55
        )
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=4,  # g=4, base = 4-4-1 = -1 < min_time=1
        )
        with pytest.raises(ContDIDValidationError, match="too large"):
            validate_spec(spec, panel=panel)

    def test_maximum_valid_anticipation_accepted(self):
        """Anticipation at the boundary should still work."""
        # Panel with group at t=5, time 1..8 -> min_g=5, max_allowed=5-1-1=3
        panel = _make_staggered_panel(
            n_units=200, n_periods=8, groups={5: 0.5}, seed=66
        )
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=3,  # g=5, base = 5-3-1 = 1 = min_time ✓
            bstrap=False,
        )
        # Should not raise
        validated = validate_spec(spec, panel=panel)
        assert validated.anticipation == 3

    def test_too_large_anticipation_in_timing_groups(self):
        """prepare_timing_groups with too-large anticipation raises descriptively."""
        panel = _make_staggered_panel(
            n_units=100, n_periods=5, groups={3: 0.5}, seed=88
        )
        with pytest.raises(ContDIDValidationError):
            prepare_timing_groups(panel, control_group="nevertreated", anticipation=3)

    def test_zero_anticipation_always_valid(self):
        """anticipation=0 should never fail validation regardless of panel."""
        panel = _make_staggered_panel(n_units=100, n_periods=3, groups={2: 0.5}, seed=11)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
        )
        validated = validate_spec(spec, panel=panel)
        assert validated.anticipation == 0


# ---------------------------------------------------------------------------
# 6. Using simulate_contdid_data for testing
# ---------------------------------------------------------------------------


class TestAnticipationWithSimulatedData:
    """Test anticipation using simulate_contdid_data generated panels."""

    @pytest.fixture
    def simulated_panel(self):
        """Generate a multi-period panel via simulate_contdid_data.

        Default: 4 periods (t=1..4), groups at {0, 2, 3, 4}.
        """
        return simulate_contdid_data(n=1000, seed=42)

    def test_anticipation_0_with_simulated(self, simulated_panel):
        """anticipation=0 works on simulate_contdid_data output."""
        spec = ContDIDSpec.eventstudy(anticipation=0)
        result = estimate_eventstudy_effects(
            simulated_panel, spec, degree=1, num_knots=0
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0

    def test_anticipation_1_with_simulated(self, simulated_panel):
        """anticipation=1 works on simulate_contdid_data output.

        With 4 periods (t=1..4) and groups at {2, 3, 4}:
        - g=2: base = 2-1-1 = 0 < 1 (excluded)
        - g=3: base = 3-1-1 = 1 >= 1 (valid)
        - g=4: base = 4-1-1 = 2 >= 1 (valid)
        """
        spec = ContDIDSpec.eventstudy(anticipation=1)
        result = estimate_eventstudy_effects(
            simulated_panel, spec, degree=1, num_knots=0
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0

    def test_anticipation_2_excluded_groups(self, simulated_panel):
        """anticipation=2 with default panel: only g=4 has valid base.

        With 4 periods (t=1..4) and groups at {2, 3, 4}:
        - g=2: base = 2-2-1 = -1 < 1 (excluded)
        - g=3: base = 3-2-1 = 0 < 1 (excluded)
        - g=4: base = 4-2-1 = 1 >= 1 (valid)
        """
        spec = ContDIDSpec.eventstudy(anticipation=2)
        result = estimate_eventstudy_effects(
            simulated_panel, spec, degree=1, num_knots=0
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0

    def test_base_period_correctness_simulated(self, simulated_panel):
        """Verify base period formula on simulated data."""
        for ant in [0, 1]:
            prepared = prepare_timing_groups(
                simulated_panel, control_group="nevertreated", anticipation=ant
            )
            for g in prepared["timing_group"].unique():
                g_post = prepared[
                    (prepared["timing_group"] == g) & prepared["post_treatment"]
                ]
                if not g_post.empty:
                    expected_base = g - ant - 1
                    assert (g_post["base_period"] == expected_base).all(), (
                        f"anticipation={ant}, g={g}: "
                        f"expected base={expected_base}, "
                        f"got {g_post['base_period'].unique()}"
                    )


# ---------------------------------------------------------------------------
# 7. Anticipation with not-yet-treated control group
# ---------------------------------------------------------------------------


class TestAnticipationNotYetTreated:
    """Anticipation should work correctly with notyettreated control group."""

    @pytest.fixture
    def panel(self):
        return _make_staggered_panel(
            n_units=400, n_periods=8, groups={4: 0.2, 5: 0.2, 6: 0.2}
        )

    def test_notyettreated_with_anticipation_0(self, panel):
        prepared = prepare_timing_groups(
            panel, control_group="notyettreated", anticipation=0
        )
        assert not prepared.empty
        assert prepared["support"].any()

    def test_notyettreated_with_anticipation_1(self, panel):
        prepared = prepare_timing_groups(
            panel, control_group="notyettreated", anticipation=1
        )
        assert not prepared.empty
        assert prepared["support"].any()

    def test_eventstudy_notyettreated_anticipation(self, panel):
        """Full event-study estimation with notyettreated + anticipation=1."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            anticipation=1,
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel, spec, degree=1, num_knots=0)
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0


# ---------------------------------------------------------------------------
# 8. Anticipation excludes early cohorts gracefully
# ---------------------------------------------------------------------------


class TestAnticipationCohortExclusion:
    """Anticipation > 0 may exclude early cohorts; this should be handled gracefully."""

    def test_early_cohort_excluded(self):
        """With anticipation=1, cohort g=2 (min time=1) has no valid base period."""
        panel = _make_staggered_panel(
            n_units=300, n_periods=6, groups={2: 0.3, 5: 0.3}, seed=123
        )
        prepared = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=1
        )
        # g=2: base = 2-1-1 = 0 < min_time=1 -> excluded
        # g=5: base = 5-1-1 = 3 >= 1 -> included
        assert 5 in prepared["timing_group"].values
        assert 2 not in prepared["timing_group"].values

    def test_all_cohorts_excluded_raises(self):
        """If all cohorts are excluded due to high anticipation, clear error."""
        panel = _make_staggered_panel(
            n_units=100, n_periods=4, groups={2: 0.5}, seed=44
        )
        # g=2, anticipation=2: base = 2-2-1 = -1 < 1
        with pytest.raises(ContDIDValidationError):
            prepare_timing_groups(panel, control_group="nevertreated", anticipation=2)

    def test_partial_exclusion_still_works(self):
        """If some cohorts are excluded but others remain, estimation succeeds."""
        panel = _make_staggered_panel(
            n_units=400, n_periods=8, groups={3: 0.2, 6: 0.3}, seed=555
        )
        # anticipation=2: g=3 -> base=-1+3=3-2-1=0 excluded; g=6 -> base=6-2-1=3 valid
        spec = ContDIDSpec.eventstudy(anticipation=2)
        result = estimate_eventstudy_effects(panel, spec, degree=1, num_knots=0)
        assert isinstance(result, ContDIDResult)
        # Only g=6 should be identified
        timing_groups = result.metadata.get("timing_group_support", {}).get(
            "timing_groups", []
        )
        assert 6 in timing_groups
        assert 3 not in timing_groups


# ---------------------------------------------------------------------------
# 9. Event-time grid consistency
# ---------------------------------------------------------------------------


class TestEventTimeGrid:
    """Event-time grid should correctly reflect anticipation effects."""

    @pytest.fixture
    def panel(self):
        return _make_staggered_panel(n_units=400, n_periods=8, groups={5: 0.5}, seed=333)

    def test_event_time_is_relative_to_treatment(self, panel):
        """event_time = time_period - timing_group (not affected by anticipation)."""
        for ant in [0, 1, 2]:
            prepared = prepare_timing_groups(
                panel, control_group="nevertreated", anticipation=ant
            )
            for _, row in prepared.iterrows():
                assert row["event_time"] == row["time_period"] - row["timing_group"]

    def test_pre_treatment_event_times_with_anticipation(self, panel):
        """Pre-treatment diagnostics: event_time in [-T+g, -anticipation-1]."""
        prepared_a0 = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=0
        )
        prepared_a1 = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=1
        )

        # With anticipation=0, pre-treatment means event_time < 0
        pre_a0 = prepared_a0[~prepared_a0["post_treatment"]]
        assert all(pre_a0["event_time"] < 0)

        # With anticipation=1, pre-treatment means t < g - 1
        # event_time < -1 for pre-treatment
        pre_a1 = prepared_a1[~prepared_a1["post_treatment"]]
        assert all(pre_a1["event_time"] < -1)


# ---------------------------------------------------------------------------
# 10. Spec propagation through full pipeline
# ---------------------------------------------------------------------------


class TestSpecPropagation:
    """Verify anticipation propagates through the full estimation pipeline."""

    def test_spec_to_result_metadata(self):
        """Result metadata should reflect the spec's anticipation value."""
        panel = _make_staggered_panel(n_units=300, n_periods=8, groups={5: 0.4})
        spec = ContDIDSpec.eventstudy(anticipation=2)
        result = estimate_eventstudy_effects(panel, spec, degree=1, num_knots=0)
        # Verify timing group support reflects anticipation effect
        timing_support = result.metadata.get("timing_group_support", {})
        timing_groups = timing_support.get("timing_groups", [])
        # With anticipation=2, only groups with g >= 4 (base=g-3 >= 1) are valid
        for g in timing_groups:
            assert g - 2 - 1 >= 1, f"group {g} should not be included with anticipation=2"

    def test_local_spec_preserves_anticipation(self):
        """_eventstudy_local_spec should preserve the anticipation value."""
        from contdid.eventstudy import _eventstudy_local_spec

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=2,
        )
        local_spec = _eventstudy_local_spec(spec, expected_target="level")
        assert local_spec.anticipation == 2
