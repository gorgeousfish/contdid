"""Boundary tests for not-yet-treated controls in dose estimation.

The public manuscript release keeps two-period dose routes on
``control_group='nevertreated'``.  Event-study internals may reuse dose helpers
with route-specific control logic, but public dose estimators must not silently
broaden the manuscript claim without a matching source-backed evidence update.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import ContDIDSpec, PanelData
from contdid.estimation import (
    _require_supported_dose_control_group,
    estimate_dose_effects,
    estimate_dose_slope_effects,
)
from contdid.validation import ContDIDValidationError


def _make_two_period_panel_with_future_cohort(
    n_never=50, n_treated=80, n_future=30, seed=42
):
    """Create a 2-period panel where some units are 'not yet treated'.

    - group=0: never treated (n_never units, dose=0)
    - group=2: treated in period 2 (n_treated units, with positive dose)
    - group=3: treated in period 3 (n_future units, dose=0 — not yet treated at period 2)

    Periods: 1 (pre) and 2 (post)
    True DGP: treatment effect = 1 + 2*dose for group=2
    """
    rng = np.random.default_rng(seed)
    records = []
    uid = 0

    # Never-treated
    for _ in range(n_never):
        y_pre = rng.normal(0, 0.5)
        y_post = y_pre + rng.normal(0, 0.3)
        records.append({"id": uid, "time": 1, "outcome": y_pre, "dose": 0.0, "group": 0})
        records.append({"id": uid, "time": 2, "outcome": y_post, "dose": 0.0, "group": 0})
        uid += 1

    # Treated at time 2
    for _ in range(n_treated):
        dose = rng.uniform(0.1, 1.0)
        y_pre = rng.normal(0, 0.5)
        effect = 1.0 + 2.0 * dose
        y_post = y_pre + effect + rng.normal(0, 0.3)
        records.append({"id": uid, "time": 1, "outcome": y_pre, "dose": dose, "group": 2})
        records.append({"id": uid, "time": 2, "outcome": y_post, "dose": dose, "group": 2})
        uid += 1

    # Future cohort (treated at time 3, not yet treated at time 2)
    for _ in range(n_future):
        y_pre = rng.normal(0, 0.5)
        y_post = y_pre + rng.normal(0, 0.3)  # No treatment effect yet
        records.append({"id": uid, "time": 1, "outcome": y_pre, "dose": 0.0, "group": 3})
        records.append({"id": uid, "time": 2, "outcome": y_post, "dose": 0.0, "group": 3})
        uid += 1

    return pd.DataFrame(records)


def _make_spec(control_group="notyettreated", target="level"):
    return ContDIDSpec(
        target_parameter=target,
        aggregation="dose",
        dose_est_method="parametric",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
        alp=0.05,
        bstrap=True,
        cband=False,
        boot_type="multiplier",
        biters=199,
    )


class TestRequireSupportedDoseControlGroup:
    def test_notyettreated_accepted_on_public_dose_route(self):
        """notyettreated is now supported on the public two-period dose route."""
        spec = _make_spec("notyettreated")
        # Should not raise — notyettreated is now supported
        _require_supported_dose_control_group(spec)

    def test_nevertreated_still_works(self):
        """nevertreated still works after the change."""
        spec = _make_spec("nevertreated")
        _require_supported_dose_control_group(spec)

    def test_invalid_control_group_raises(self):
        """Invalid control group should raise."""
        spec = _make_spec("eventuallytreated")
        with pytest.raises(ContDIDValidationError, match="dose estimation supports"):
            _require_supported_dose_control_group(spec)


class TestPrepareDoseSamplePublicBoundary:
    def test_nevertreated_excludes_future_cohort(self):
        """With nevertreated, only group=0 are untreated — future cohort excluded."""
        # Build a panel without future cohort (pure nevertreated case)
        df = _make_two_period_panel_with_future_cohort(n_future=0)
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        spec = _make_spec("nevertreated")
        result = estimate_dose_effects(
            panel,
            spec,
            dvals=[0.3, 0.5, 0.7],
            degree=1,
            num_knots=0,
        )
        assert result.metadata["control_group"] == "nevertreated"
        assert result.grid == [0.3, 0.5, 0.7]
        assert len(result.estimate) == 3

    def test_notyettreated_accepted_before_sample_preparation(self):
        """A future-cohort panel should now produce a valid dose result."""
        df = _make_two_period_panel_with_future_cohort()
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        spec_nyt = _make_spec("notyettreated")
        result = estimate_dose_effects(panel, spec_nyt, dvals=[0.3, 0.5, 0.7], degree=1, num_knots=0)
        assert result.metadata["control_group"] == "notyettreated"
        assert len(result.estimate) == 3


class TestEstimateDoseEffectsNotYetTreated:
    def test_level_estimation_with_notyettreated(self):
        """Public estimate_dose_effects accepts not-yet-treated controls."""
        df = _make_two_period_panel_with_future_cohort(
            n_never=80, n_treated=120, n_future=50, seed=123
        )
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        spec = _make_spec("notyettreated", target="level")
        dose_grid = np.array([0.2, 0.4, 0.6, 0.8])
        result = estimate_dose_effects(
            panel, spec, dvals=dose_grid, degree=1, num_knots=0
        )
        assert result.metadata["control_group"] == "notyettreated"
        assert len(result.estimate) == 4

    def test_slope_estimation_with_notyettreated(self):
        """Public estimate_dose_slope_effects accepts not-yet-treated controls."""
        df = _make_two_period_panel_with_future_cohort(
            n_never=80, n_treated=120, n_future=50, seed=123
        )
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        spec = _make_spec("notyettreated", target="slope")
        dose_grid = np.array([0.2, 0.4, 0.6, 0.8])
        result = estimate_dose_slope_effects(
            panel, spec, dvals=dose_grid, degree=1, num_knots=0
        )
        assert result.metadata["control_group"] == "notyettreated"
        assert len(result.estimate) == 4

    def test_validation_passes_for_future_group_panel(self):
        """Panel with group > post_period passes validation when using notyettreated."""
        df = _make_two_period_panel_with_future_cohort()
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        from contdid.validation import validate_panel_data

        # Should not raise — future groups with dose=0 are allowed
        validate_panel_data(panel)

    def test_nevertreated_rejects_panel_with_future_groups(self):
        """nevertreated should reject panel with groups beyond post period."""
        df = _make_two_period_panel_with_future_cohort()
        panel = PanelData(
            frame=df,
            id_column="id",
            time_column="time",
            outcome_column="outcome",
            group_column="group",
            dose_column="dose",
        )
        spec = _make_spec("nevertreated")
        with pytest.raises(ContDIDValidationError, match="two-period dose aggregation"):
            estimate_dose_effects(panel, spec, dvals=[0.3, 0.5, 0.7])
