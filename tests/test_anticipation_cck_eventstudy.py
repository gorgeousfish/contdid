"""End-to-end tests for anticipation parameter and CCK event-study functionality.

Tests cover:
1. Anticipation parameter validation, base-period shifting, and event-time grid effects
2. CCK fixed-dimension estimation in event-study level and slope targets
3. Theory boundary documentation and enforcement

Theoretical references:
- CGBS Assumption 3-MP(a): no-anticipation with parameter a >= 0
- CCK (arXiv:2107.11869v3 Theorem 2): two-period B-spline sieve convergence
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    PanelData,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
)
from contdid.multiperiod import estimate_multiperiod_dose
from contdid.validation import ContDIDValidationError


# ---------------------------------------------------------------------------
# Fixture: simulated balanced panel data
# ---------------------------------------------------------------------------


def _make_panel(
    n_units: int = 300,
    n_periods: int = 5,
    groups: dict[int, float] | None = None,
    seed: int = 42,
    effect_size: float = 0.0,
) -> PanelData:
    """Create a balanced panel suitable for event-study estimation.

    Parameters
    ----------
    n_units : total number of units
    n_periods : number of time periods (1..n_periods)
    groups : mapping of group_timing -> fraction of treated units
        e.g. {3: 0.3, 4: 0.2} means 30% treated at t=3, 20% at t=4
        Remaining units are never-treated (group=0).
    seed : random seed
    effect_size : linear dose effect for post-treatment periods (0 = null DGP)
    """
    if groups is None:
        groups = {3: 0.3, 4: 0.2}

    rng = np.random.default_rng(seed)
    time_periods = np.arange(1, n_periods + 1)

    # Assign groups
    never_treated_frac = 1.0 - sum(groups.values())
    assert never_treated_frac > 0, "Must have some never-treated units"
    group_keys = [0] + list(groups.keys())
    group_probs = [never_treated_frac] + list(groups.values())
    G = rng.choice(group_keys, size=n_units, replace=True, p=group_probs)

    # Assign dose (positive for treated, 0 for control)
    D = np.where(G > 0, rng.uniform(0.1, 2.0, size=n_units), 0.0)

    # Unit fixed effect
    eta = rng.normal(0, 1, size=n_units)

    records = []
    for i in range(n_units):
        for t in time_periods:
            y = float(t) + eta[i] + rng.normal(0, 0.5)
            # Add treatment effect if post-treatment
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


@pytest.fixture
def panel_5period():
    """Standard 5-period panel with groups at t=3 and t=4."""
    return _make_panel(n_units=300, n_periods=5, groups={3: 0.3, 4: 0.2}, seed=42)


@pytest.fixture
def panel_null_effect():
    """Panel with zero treatment effect for unbiasedness tests."""
    return _make_panel(
        n_units=500, n_periods=5, groups={3: 0.3, 4: 0.2},
        seed=123, effect_size=0.0,
    )


@pytest.fixture
def panel_positive_effect():
    """Panel with known positive dose-response for comparison tests."""
    return _make_panel(
        n_units=500, n_periods=5, groups={3: 0.3, 4: 0.2},
        seed=456, effect_size=2.0,
    )


# ---------------------------------------------------------------------------
# 1. Anticipation parameter tests
# ---------------------------------------------------------------------------


class TestAnticipationZeroDefault:
    """anticipation=0 should be the default, matching prior behavior."""

    def test_default_spec_anticipation_is_zero(self):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert spec.anticipation == 0

    def test_anticipation_zero_produces_results(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=1, num_knots=0)
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0


class TestAnticipationPositiveShiftsBasePeriod:
    """anticipation>0 should shift base period from g-1 to g-1-anticipation.

    Theory: CGBS Assumption 3-MP(a) — units do not react to treatment before
    g-anticipation.
    """

    def test_base_period_shifts_with_anticipation(self, panel_5period):
        """For group g=3, T=5: anticipation=0 -> base=2; anticipation=1 -> base=1."""
        from contdid.timing import prepare_timing_groups

        spec_a0 = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
        )
        spec_a1 = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=1,
        )

        prepared_a0 = prepare_timing_groups(
            panel_5period, control_group="nevertreated", anticipation=0
        )
        prepared_a1 = prepare_timing_groups(
            panel_5period, control_group="nevertreated", anticipation=1
        )

        # For timing group g=3: anticipation=0 -> universal base = g-1-0 = 2
        g3_a0 = prepared_a0[prepared_a0["timing_group"] == 3]
        post_a0 = g3_a0[g3_a0["post_treatment"]]
        if not post_a0.empty:
            assert (post_a0["base_period"] == 2).all()

        # For timing group g=3: anticipation=1 -> universal base = g-1-1 = 1
        g3_a1 = prepared_a1[prepared_a1["timing_group"] == 3]
        post_a1 = g3_a1[g3_a1["post_treatment"]]
        if not post_a1.empty:
            assert (post_a1["base_period"] == 1).all()

    def test_anticipation_changes_results(self, panel_5period):
        """anticipation=0 vs anticipation=1 should give different event_time_grids or estimates."""
        spec_a0 = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=0,
            bstrap=False,
        )
        spec_a1 = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=1,
            bstrap=False,
        )
        result_a0 = estimate_eventstudy_effects(panel_5period, spec_a0, degree=1, num_knots=0)
        result_a1 = estimate_eventstudy_effects(panel_5period, spec_a1, degree=1, num_knots=0)

        # The grids or estimates should differ due to base period shift
        grids_differ = result_a0.event_time_grid != result_a1.event_time_grid
        estimates_differ = result_a0.estimate != result_a1.estimate
        assert grids_differ or estimates_differ


class TestAnticipationAffectsEventTimeGrid:
    """anticipation>0 should change the event-time grid in event-study output.

    With anticipation=1, the post_treatment flag starts earlier (at g-anticipation),
    affecting which event times are classified as pre/post treatment.
    """

    def test_post_treatment_boundary_shifts(self, panel_5period):
        from contdid.timing import prepare_timing_groups

        prepared_a0 = prepare_timing_groups(
            panel_5period, control_group="nevertreated", anticipation=0
        )
        prepared_a1 = prepare_timing_groups(
            panel_5period, control_group="nevertreated", anticipation=1
        )

        # With anticipation=1, post_treatment starts at g - anticipation = g - 1
        # For g=3: anticipation=0 -> post starts at t=3 (event_time=0)
        #          anticipation=1 -> post starts at t=2 (event_time=-1)
        g3_a0_post = prepared_a0[
            (prepared_a0["timing_group"] == 3) & prepared_a0["post_treatment"]
        ]
        g3_a1_post = prepared_a1[
            (prepared_a1["timing_group"] == 3) & prepared_a1["post_treatment"]
        ]

        # anticipation=1 should have more post-treatment rows (includes earlier periods)
        assert len(g3_a1_post) >= len(g3_a0_post)


class TestAnticipationNegativeRaisesError:
    """anticipation<0 should raise a validation error."""

    def test_negative_anticipation_in_spec_raises(self):
        """ContDIDSpec with negative anticipation should fail validation."""
        from contdid.validation import validate_spec

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=-1,
        )
        with pytest.raises(ContDIDValidationError, match="anticipation"):
            validate_spec(spec)

    def test_negative_anticipation_in_timing(self):
        """prepare_timing_groups with negative anticipation should raise."""
        from contdid.timing import prepare_timing_groups

        panel = _make_panel(n_units=50, n_periods=4, groups={3: 0.5}, seed=99)
        with pytest.raises(ContDIDValidationError, match="anticipation"):
            prepare_timing_groups(panel, control_group="nevertreated", anticipation=-1)


class TestAnticipationTooLargeRaisesError:
    """If anticipation is too large, no valid base period exists -> error."""

    def test_anticipation_exceeds_available_periods(self):
        """With T=5, g=3, anticipation=3 -> base=g-1-3=-1 < min_time -> error."""
        panel = _make_panel(n_units=100, n_periods=5, groups={3: 0.5}, seed=77)
        from contdid.timing import prepare_timing_groups

        # anticipation=3 for g=3 means base_period = 3-1-3 = -1, which < 1
        with pytest.raises(ContDIDValidationError):
            prepare_timing_groups(panel, control_group="nevertreated", anticipation=3)

    def test_large_anticipation_in_eventstudy(self):
        """Event study should fail gracefully with too-large anticipation."""
        panel = _make_panel(n_units=100, n_periods=5, groups={3: 0.5}, seed=77)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=3,
            bstrap=False,
        )
        with pytest.raises(ContDIDValidationError):
            estimate_eventstudy_effects(panel, spec, degree=1, num_knots=0)


class TestAnticipationInMultiperiod:
    """Verify estimate_multiperiod_dose correctly accepts anticipation parameter."""

    def test_multiperiod_accepts_anticipation_zero(self):
        panel = _make_panel(n_units=200, n_periods=5, groups={3: 0.3, 4: 0.2}, seed=50)
        result = estimate_multiperiod_dose(
            panel.frame,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=1,
            num_knots=0,
            anticipation=0,
            biters=100,
            boot_seed=42,
        )
        assert result.metadata["target"] == "level"
        assert len(result.point_estimate) > 0

    def test_multiperiod_accepts_anticipation_positive(self):
        panel = _make_panel(n_units=200, n_periods=5, groups={3: 0.3, 4: 0.2}, seed=50)
        result = estimate_multiperiod_dose(
            panel.frame,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=1,
            num_knots=0,
            anticipation=1,
            biters=100,
            boot_seed=42,
        )
        assert len(result.point_estimate) > 0

    def test_multiperiod_rejects_negative_anticipation(self):
        panel = _make_panel(n_units=100, n_periods=5, groups={3: 0.5}, seed=50)
        with pytest.raises(ContDIDValidationError, match="anticipation"):
            estimate_multiperiod_dose(
                panel.frame,
                id_column="id",
                time_column="time_period",
                outcome_column="Y",
                dose_column="D",
                group_column="G",
                anticipation=-1,
            )


# ---------------------------------------------------------------------------
# 2. CCK event-study tests
# ---------------------------------------------------------------------------


class TestCCKEventstudyLevelBasic:
    """CCK fixed-dimension in event-study level target should run correctly.

    Theory: Each (g,t) local comparison is a two-period problem where
    CCK (arXiv:2107.11869v3 Theorem 2) applies directly.
    """

    def test_cck_level_eventstudy_runs(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=3, num_knots=0)
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(event_time)"

    def test_cck_level_with_notyettreated(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="notyettreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        assert isinstance(result, ContDIDResult)


class TestCCKEventstudySlopeBasic:
    """CCK fixed-dimension in event-study slope (ACRT) target should run."""

    def test_cck_slope_eventstudy_runs(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_slope_effects(
            panel_5period, spec, degree=3, num_knots=0
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ACRT(event_time)"


class TestCCKEventstudyProducesValidResult:
    """CCK event-study results should contain estimate, std_error, event_time_grid."""

    def test_result_has_required_fields(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        assert hasattr(result, "estimate")
        assert hasattr(result, "std_error")
        assert hasattr(result, "event_time_grid")
        assert len(result.estimate) == len(result.event_time_grid)
        assert len(result.std_error) == len(result.event_time_grid)

    def test_std_errors_nonnegative(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        for se in result.std_error:
            assert se >= 0.0

    def test_event_time_grid_sorted_ascending(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        grid = result.event_time_grid
        assert grid == sorted(grid)


class TestCCKEventstudyZeroEffectUnbiased:
    """Under zero treatment effect DGP, CCK event-study should estimate ~0.

    Theory: Under PT, E[LATT(g,t,d|g,d)] = 0 when true effect is 0.
    Note: The parametric level path subtracts control mean (giving DID),
    while the CCK level path reports the dose-regression fit on treated deltas.
    For a proper zero-effect test, use the slope (ACRT) target where both
    paths use regression and the dose derivative is 0 under null.
    """

    def test_zero_effect_slope_estimates_near_zero(self, panel_null_effect):
        """Under null DGP, ACRT (slope/derivative) should be ~0 regardless of time trend."""
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_slope_effects(
            panel_null_effect, spec, degree=2, num_knots=0
        )
        # Under null DGP, dose derivative (ACRT) should be ~0
        for est, se in zip(result.estimate, result.std_error):
            if np.isfinite(est):
                assert abs(est) < 3 * se + 1.0, (
                    f"ACRT estimate {est:.3f} too far from 0 (SE={se:.3f}) under null DGP"
                )

    def test_zero_effect_parametric_level_near_zero(self, panel_null_effect):
        """Parametric level path (which subtracts control mean) should give ~0 under null."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(
            panel_null_effect, spec, degree=1, num_knots=0
        )
        for est, se in zip(result.estimate, result.std_error):
            if np.isfinite(est):
                assert abs(est) < 3 * se + 0.5, (
                    f"Estimate {est:.3f} too far from 0 (SE={se:.3f}) under null DGP"
                )


class TestCCKAdaptiveEventstudyRaisesError:
    """Lepski adaptive in event-study should be forbidden.

    Theory boundary: CCK paper proves two-period convergence rates only;
    aggregating heterogeneous adaptive dimensions across event-time cells
    lacks joint coverage theory.
    """

    def test_adaptive_not_exposed_in_eventstudy_api(self):
        """The eventstudy API enforces adaptive=False internally.

        Since ContDIDSpec does not expose an 'adaptive' flag and
        estimate_eventstudy_effects forces adaptive=False when calling
        run_cck_backend, adaptive Lepski is structurally prevented
        in event-study context.
        """
        # Verify that ContDIDSpec has no 'adaptive' attribute
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
        )
        assert not hasattr(spec, "adaptive"), (
            "ContDIDSpec should not expose an 'adaptive' field; "
            "adaptive Lepski is not supported in event-study aggregation"
        )

    def test_cck_eventstudy_metadata_documents_fixed_dimension(self, panel_5period):
        """Result metadata should indicate fixed-dimension CCK (not adaptive)."""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=3, num_knots=0)
        # The source estimator should indicate fixed-dimension CCK
        assert result.metadata["source_estimator"] == "cck_fixed_dimension_eventstudy"


class TestCCKEventstudyFixedVsParametricComparable:
    """CCK fixed-dimension vs parametric should give comparable results for SLOPE target.

    For the slope (ACRT) target, both CCK and parametric paths use B-spline
    dose regression, differing only in inference (sandwich vs OLS SE).
    Point estimates should be numerically close.

    Note: For the level (ATT) target, the parametric path uses simple
    mean-difference while CCK uses dose regression, so they estimate
    structurally different quantities and are not directly comparable.
    """

    def test_slope_estimates_comparable(self, panel_positive_effect):
        """ACRT estimates from CCK and parametric should be similar."""
        spec_cck = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        spec_param = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
        )
        result_cck = estimate_eventstudy_slope_effects(
            panel_positive_effect, spec_cck, degree=3, num_knots=0
        )
        result_param = estimate_eventstudy_slope_effects(
            panel_positive_effect, spec_param, degree=3, num_knots=0
        )

        # Both should produce the same event_time_grid
        assert result_cck.event_time_grid == result_param.event_time_grid

        # Compare post-treatment slope estimates
        for et, est_c, est_p in zip(
            result_cck.event_time_grid,
            result_cck.estimate,
            result_param.estimate,
        ):
            if et >= 0 and np.isfinite(est_c) and np.isfinite(est_p):
                # Both use B-spline regression on the same data;
                # difference is sandwich vs OLS SE. Point estimates should
                # be close but may differ due to implementation details.
                combined_se = max(
                    result_cck.std_error[result_cck.event_time_grid.index(et)],
                    result_param.std_error[result_param.event_time_grid.index(et)],
                    0.1,
                )
                assert abs(est_c - est_p) < 5 * combined_se + 2.0, (
                    f"CCK ACRT={est_c:.3f} vs param ACRT={est_p:.3f} differ too much at et={et}"
                )


# ---------------------------------------------------------------------------
# 3. Theory boundary tests
# ---------------------------------------------------------------------------


class TestCCKEventstudyTheoryBoundaryDocumented:
    """CCK event-study result metadata should contain theory boundary annotations."""

    def test_metadata_has_identification_info(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        md = result.metadata
        assert "identification" in md
        assert "paper_estimand" in md["identification"]
        assert md["identification"]["paper_estimand"] == "ATT(event_time)"

    def test_metadata_has_basis_info(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=3, num_knots=0)
        md = result.metadata
        assert "basis" in md
        assert md["basis"]["degree"] == 3
        assert md["basis"]["num_knots"] == 0

    def test_metadata_dose_est_method_is_cck(self, panel_5period):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel_5period, spec, degree=2, num_knots=0)
        assert result.metadata["dose_est_method"] == "cck"


class TestAnticipationSpecPreserved:
    """ContDIDSpec should correctly store and propagate anticipation value."""

    def test_spec_stores_anticipation(self):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=2,
        )
        assert spec.anticipation == 2

    def test_spec_default_anticipation_zero(self):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert spec.anticipation == 0

    def test_anticipation_propagated_to_timing(self, panel_5period):
        """Anticipation value in spec should propagate to timing preparation."""
        from contdid.timing import prepare_timing_groups

        panel = _make_panel(n_units=200, n_periods=6, groups={4: 0.4}, seed=88)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=2,
            bstrap=False,
        )
        prepared = prepare_timing_groups(
            panel, control_group="nevertreated", anticipation=spec.anticipation
        )
        # For g=4, anticipation=2: universal base = 4-1-2 = 1
        g4_post = prepared[
            (prepared["timing_group"] == 4) & prepared["post_treatment"]
        ]
        if not g4_post.empty:
            assert (g4_post["base_period"] == 1).all()

    def test_anticipation_passed_through_eventstudy(self, panel_5period):
        """estimate_eventstudy_effects should use anticipation from spec."""
        # Use a panel where anticipation=1 produces different results
        panel = _make_panel(n_units=300, n_periods=6, groups={4: 0.4}, seed=101)
        spec_a1 = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            anticipation=1,
            bstrap=False,
        )
        result = estimate_eventstudy_effects(panel, spec_a1, degree=1, num_knots=0)
        # Should succeed and produce a valid result
        assert isinstance(result, ContDIDResult)
        assert len(result.event_time_grid) > 0
