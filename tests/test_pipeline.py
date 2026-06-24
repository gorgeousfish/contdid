"""Tests for the pipeline (chained-call) API.

Verifies:
- panel.with_spec(spec).fit() produces the same result as cont_did(panel, spec)
- ContDIDSpec convenience classmethods produce correct specs
- FittablePanel property access works
- FittablePanel repr is informative
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import ContDIDResult, ContDIDSpec, FittablePanel, PanelData, cont_did


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_period_panel() -> PanelData:
    """Minimal two-period panel for testing (50 units, 2 periods).

    Dose is unit-level (constant within unit) per contdid validation rules.
    """
    rng = np.random.default_rng(42)
    n_units = 50

    # Generate unit-level attributes
    unit_groups = np.where(np.arange(1, n_units + 1) <= 30, 2, 0)  # 30 treated
    unit_doses = np.where(unit_groups > 0, rng.uniform(0.5, 3.0, size=n_units), 0.0)

    # Expand to panel (repeat each unit for 2 periods)
    ids = np.repeat(np.arange(1, n_units + 1), 2)
    times = np.tile([1, 2], n_units)
    groups = np.repeat(unit_groups, 2)
    doses = np.repeat(unit_doses, 2)

    # Outcome with treatment effect in post-period
    post = (times >= groups) & (groups > 0)
    outcomes = 1.0 + 0.5 * doses * post + rng.normal(0, 0.3, size=len(ids))

    df = pd.DataFrame({
        "id": ids,
        "time_period": times,
        "Y": outcomes,
        "G": groups,
        "D": doses,
    })
    return PanelData(frame=df)


# ---------------------------------------------------------------------------
# ContDIDSpec convenience classmethods
# ---------------------------------------------------------------------------


class TestContDIDSpecConvenience:
    """Test convenience class methods on ContDIDSpec."""

    def test_dose_response_defaults(self):
        spec = ContDIDSpec.dose_response()
        assert spec.target_parameter == "level"
        assert spec.aggregation == "dose"
        assert spec.dose_est_method == "parametric"
        assert spec.control_group == "notyettreated"

    def test_dose_response_cck(self):
        spec = ContDIDSpec.dose_response(method="cck", cband=True)
        assert spec.dose_est_method == "cck"
        assert spec.cband is True

    def test_dose_response_custom_control(self):
        spec = ContDIDSpec.dose_response(control_group="notyettreated")
        assert spec.control_group == "notyettreated"

    def test_eventstudy_defaults(self):
        spec = ContDIDSpec.eventstudy()
        assert spec.target_parameter == "level"
        assert spec.aggregation == "eventstudy"
        assert spec.dose_est_method == "parametric"
        assert spec.control_group == "notyettreated"

    def test_eventstudy_with_anticipation(self):
        spec = ContDIDSpec.eventstudy(anticipation=2)
        assert spec.anticipation == 2

    def test_marginal_response_defaults(self):
        spec = ContDIDSpec.marginal_response()
        assert spec.target_parameter == "slope"
        assert spec.aggregation == "dose"
        assert spec.dose_est_method == "parametric"
        assert spec.control_group == "notyettreated"

    def test_marginal_response_cck(self):
        spec = ContDIDSpec.marginal_response(method="cck")
        assert spec.dose_est_method == "cck"


# ---------------------------------------------------------------------------
# FittablePanel
# ---------------------------------------------------------------------------


class TestFittablePanel:
    """Test FittablePanel intermediate object."""

    def test_with_spec_returns_fittable_panel(self, two_period_panel):
        spec = ContDIDSpec.dose_response()
        fp = two_period_panel.with_spec(spec)
        assert isinstance(fp, FittablePanel)

    def test_panel_property(self, two_period_panel):
        spec = ContDIDSpec.dose_response()
        fp = two_period_panel.with_spec(spec)
        assert fp.panel is two_period_panel

    def test_spec_property(self, two_period_panel):
        spec = ContDIDSpec.dose_response()
        fp = two_period_panel.with_spec(spec)
        assert fp.spec is spec

    def test_repr(self, two_period_panel):
        spec = ContDIDSpec.dose_response()
        fp = two_period_panel.with_spec(spec)
        r = repr(fp)
        assert "FittablePanel" in r
        assert "level" in r
        assert "dose" in r
        assert "parametric" in r

    def test_fit_returns_result(self, two_period_panel):
        spec = ContDIDSpec.dose_response(bstrap=False)
        result = two_period_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)

    def test_fit_with_kwargs(self, two_period_panel):
        spec = ContDIDSpec.dose_response(bstrap=False)
        result = two_period_panel.with_spec(spec).fit(degree=2)
        assert isinstance(result, ContDIDResult)


# ---------------------------------------------------------------------------
# Pipeline equivalence with traditional API
# ---------------------------------------------------------------------------


class TestPipelineEquivalence:
    """Verify pipeline results match direct cont_did() calls."""

    def test_dose_response_equivalence(self, two_period_panel):
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
        )
        # Traditional call
        result_traditional = cont_did(two_period_panel, spec)
        # Pipeline call
        result_pipeline = two_period_panel.with_spec(spec).fit()

        # Results should be numerically identical
        np.testing.assert_array_equal(result_traditional.grid, result_pipeline.grid)
        np.testing.assert_array_equal(result_traditional.estimate, result_pipeline.estimate)
        np.testing.assert_array_equal(result_traditional.std_error, result_pipeline.std_error)

    def test_slope_equivalence(self, two_period_panel):
        spec = ContDIDSpec.marginal_response(bstrap=False)
        result_traditional = cont_did(two_period_panel, spec)
        result_pipeline = two_period_panel.with_spec(spec).fit()

        np.testing.assert_array_equal(result_traditional.grid, result_pipeline.grid)
        np.testing.assert_array_equal(result_traditional.estimate, result_pipeline.estimate)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify public API exports."""

    def test_fittable_panel_importable(self):
        from contdid import FittablePanel  # noqa: F401

    def test_pipeline_module_importable(self):
        from contdid.pipeline import FittablePanel  # noqa: F401
