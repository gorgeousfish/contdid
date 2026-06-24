"""End-to-end integration tests covering estimation → export → plotting pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    ContDIDValidationError,
    PanelData,
    estimate_dose_effects,
    estimate_eventstudy_effects,
    simulate_contdid_data,
    summary,
    to_csv,
    to_dataframe,
    to_latex,
)
from contdid.plotting import plot, plot_dose_response, plot_eventstudy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dose_spec(*, boot_type: str = "multiplier", bstrap: bool = True) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        bstrap=bstrap,
        boot_type=boot_type,
        biters=199,
    )


def _make_eventstudy_spec(*, boot_type: str = "multiplier") -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        bstrap=True,
        boot_type=boot_type,
        biters=199,
    )


def _small_two_period_panel(n: int = 200, seed: int = 42) -> PanelData:
    """Generate a small two-period panel for fast integration tests."""
    return simulate_contdid_data(
        n=n,
        num_time_periods=2,
        num_groups=2,
        pg=[0.75],
        pu=0.25,
        dgp_id="SIM-002-linear-dose",
        seed=seed,
    )


def _staggered_panel(n: int = 300, seed: int = 99) -> PanelData:
    """Generate a staggered panel for event-study tests."""
    return simulate_contdid_data(
        n=n,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. Dose-response full pipeline test
# ---------------------------------------------------------------------------


class TestDoseResponseFullPipeline:
    """Verify estimate → summary → export → plot for dose-response."""

    @pytest.fixture()
    def dose_result(self) -> ContDIDResult:
        panel = _small_two_period_panel()
        spec = _make_dose_spec()
        return estimate_dose_effects(panel, spec)

    def test_result_structure(self, dose_result: ContDIDResult):
        """Result has expected estimand, grid, estimate, std_error."""
        assert dose_result.estimand == "ATT(d)"
        assert len(dose_result.grid) > 0
        assert len(dose_result.estimate) == len(dose_result.grid)
        assert len(dose_result.std_error) == len(dose_result.grid)

    def test_summary_output(self, dose_result: ContDIDResult):
        """summary() returns a non-empty string with key sections."""
        text = summary(dose_result)
        assert isinstance(text, str)
        assert len(text) > 100
        assert "ContDID Estimation Results" in text
        assert "ATT(d)" in text

    def test_to_dataframe(self, dose_result: ContDIDResult):
        """to_dataframe() returns DataFrame with dose, estimate, std_error."""
        df = to_dataframe(dose_result)
        assert isinstance(df, pd.DataFrame)
        assert "dose" in df.columns
        assert "estimate" in df.columns
        assert "std_error" in df.columns
        assert len(df) == len(dose_result.grid)

    def test_to_csv_string(self, dose_result: ContDIDResult):
        """to_csv() returns valid CSV string when no path given."""
        csv_str = to_csv(dose_result)
        assert isinstance(csv_str, str)
        assert "dose" in csv_str
        assert "estimate" in csv_str
        lines = csv_str.strip().split("\n")
        # header + data rows
        assert len(lines) == len(dose_result.grid) + 1

    def test_to_csv_file(self, dose_result: ContDIDResult, tmp_path: Path):
        """to_csv() writes to file correctly."""
        csv_file = tmp_path / "result.csv"
        to_csv(dose_result, csv_file)
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "dose" in content

    def test_to_latex(self, dose_result: ContDIDResult):
        """to_latex() returns valid LaTeX table string."""
        latex_str = to_latex(dose_result)
        assert isinstance(latex_str, str)
        assert "\\begin{table}" in latex_str
        assert "\\end{table}" in latex_str
        assert "ATT(d)" in latex_str

    def test_plot_dose_response(self, dose_result: ContDIDResult):
        """plot_dose_response() returns Axes object."""
        plt = pytest.importorskip("matplotlib.pyplot")
        ax = plot_dose_response(dose_result)
        assert ax is not None
        # Check it has title set
        assert "Dose-Response" in ax.get_title()
        plt.close("all")

    def test_plot_auto_routes_to_dose(self, dose_result: ContDIDResult):
        """plot(result, kind='auto') routes to dose-response plot."""
        plt = pytest.importorskip("matplotlib.pyplot")
        ax = plot(dose_result, kind="auto")
        assert ax is not None
        assert "Dose-Response" in ax.get_title()
        plt.close("all")

    def test_confidence_interval_present(self, dose_result: ContDIDResult):
        """Bootstrap result should have confidence_interval."""
        assert dose_result.confidence_interval is not None
        assert len(dose_result.confidence_interval) == len(dose_result.grid)
        # Each interval is [lower, upper]
        for ci in dose_result.confidence_interval:
            assert len(ci) == 2
            assert ci[0] <= ci[1]


# ---------------------------------------------------------------------------
# 2. Event-study full pipeline test
# ---------------------------------------------------------------------------


class TestEventStudyFullPipeline:
    """Verify estimate → summary → export → plot for event-study."""

    @pytest.fixture()
    def eventstudy_result(self) -> ContDIDResult:
        panel = _staggered_panel()
        spec = _make_eventstudy_spec()
        return estimate_eventstudy_effects(panel, spec)

    def test_result_structure(self, eventstudy_result: ContDIDResult):
        """Result has expected estimand and event_time_grid."""
        assert eventstudy_result.estimand == "ATT(event_time)"
        assert eventstudy_result.event_time_grid is not None
        assert len(eventstudy_result.event_time_grid) > 0
        assert len(eventstudy_result.estimate) == len(eventstudy_result.event_time_grid)

    def test_metadata_aggregation(self, eventstudy_result: ContDIDResult):
        """Metadata has aggregation='eventstudy'."""
        assert eventstudy_result.metadata["aggregation"] == "eventstudy"

    def test_to_dataframe_has_event_time(self, eventstudy_result: ContDIDResult):
        """DataFrame has event_time column for event-study results."""
        df = to_dataframe(eventstudy_result)
        assert "event_time" in df.columns
        assert "estimate" in df.columns
        assert "std_error" in df.columns

    def test_summary_output(self, eventstudy_result: ContDIDResult):
        """summary() returns string mentioning ATT(event_time)."""
        text = summary(eventstudy_result)
        assert isinstance(text, str)
        assert "ATT(event_time)" in text

    def test_to_csv_string(self, eventstudy_result: ContDIDResult):
        """CSV export includes event_time column."""
        csv_str = to_csv(eventstudy_result)
        assert "event_time" in csv_str

    def test_to_latex(self, eventstudy_result: ContDIDResult):
        """LaTeX export works for event-study results."""
        latex_str = to_latex(eventstudy_result)
        assert "\\begin{table}" in latex_str

    def test_plot_eventstudy(self, eventstudy_result: ContDIDResult):
        """plot_eventstudy() returns Axes for event-study results."""
        plt = pytest.importorskip("matplotlib.pyplot")
        ax = plot_eventstudy(eventstudy_result)
        assert ax is not None
        assert "Event Study" in ax.get_title()
        plt.close("all")

    def test_plot_auto_routes_to_eventstudy(self, eventstudy_result: ContDIDResult):
        """plot(result, kind='auto') routes to event-study plot."""
        plt = pytest.importorskip("matplotlib.pyplot")
        ax = plot(eventstudy_result, kind="auto")
        assert ax is not None
        assert "Event Study" in ax.get_title()
        plt.close("all")


# ---------------------------------------------------------------------------
# 3. Wild bootstrap integration tests
# ---------------------------------------------------------------------------


class TestWildBootstrapIntegration:
    """Verify bootstrap types work in estimation pipeline."""

    def test_rademacher_bootstrap(self):
        """boot_type='rademacher' produces valid result with CI."""
        panel = _small_two_period_panel(n=150, seed=10)
        spec = _make_dose_spec(boot_type="rademacher")
        result = estimate_dose_effects(panel, spec)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == len(result.grid)
        # Verify standard errors are positive
        assert all(se >= 0.0 for se in result.std_error)

    def test_mammen_bootstrap(self):
        """boot_type='mammen' produces valid result with CI."""
        panel = _small_two_period_panel(n=150, seed=11)
        spec = _make_dose_spec(boot_type="mammen")
        result = estimate_dose_effects(panel, spec)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == len(result.grid)
        assert all(se >= 0.0 for se in result.std_error)

    def test_multiplier_bootstrap(self):
        """boot_type='multiplier' (default) produces valid result."""
        panel = _small_two_period_panel(n=150, seed=12)
        spec = _make_dose_spec(boot_type="multiplier")
        result = estimate_dose_effects(panel, spec)
        assert result.confidence_interval is not None
        assert result.critical_value is not None
        assert result.critical_value > 0.0


# ---------------------------------------------------------------------------
# 4. Boundary cases and error handling
# ---------------------------------------------------------------------------


class TestBoundaryCasesAndErrors:
    """Verify error handling for invalid inputs."""

    def test_plot_eventstudy_on_dose_result_raises(self):
        """plot_eventstudy on dose result raises ValueError."""
        pytest.importorskip("matplotlib.pyplot")
        panel = _small_two_period_panel(n=100, seed=20)
        spec = _make_dose_spec()
        result = estimate_dose_effects(panel, spec)
        with pytest.raises(ValueError, match="aggregation"):
            plot_eventstudy(result)

    def test_plot_eventstudy_no_aggregation_metadata_raises(self):
        """plot_eventstudy on result without aggregation metadata raises ValueError."""
        pytest.importorskip("matplotlib.pyplot")
        # Construct a minimal ContDIDResult with no aggregation in metadata
        result = ContDIDResult(
            estimand="ATT(d)",
            grid=[0.1, 0.2, 0.3],
            estimate=[1.0, 2.0, 3.0],
            std_error=[0.1, 0.1, 0.1],
            metadata={},
        )
        with pytest.raises(ValueError, match="aggregation"):
            plot_eventstudy(result)

    def test_invalid_boot_type_raises(self):
        """Invalid boot_type raises ContDIDValidationError."""
        panel = _small_two_period_panel(n=100, seed=30)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            bstrap=True,
            boot_type="invalid_boot",
            biters=199,
        )
        with pytest.raises(ContDIDValidationError, match="boot_type"):
            estimate_dose_effects(panel, spec)

    def test_dose_result_plot_auto_not_eventstudy(self):
        """plot(dose_result, kind='auto') does not call eventstudy plot."""
        plt = pytest.importorskip("matplotlib.pyplot")
        panel = _small_two_period_panel(n=100, seed=40)
        spec = _make_dose_spec()
        result = estimate_dose_effects(panel, spec)
        ax = plot(result, kind="auto")
        # Should route to dose, not event study
        assert "Dose-Response" in ax.get_title()
        plt.close("all")

    def test_eventstudy_result_plot_dose_response_works(self):
        """plot_dose_response on event-study result doesn't crash (but is unusual)."""
        plt = pytest.importorskip("matplotlib.pyplot")
        # Construct event-study-like result manually
        result = ContDIDResult(
            estimand="ATT(event_time)",
            grid=[-2, -1, 0, 1],
            estimate=[0.0, 0.1, 0.5, 0.8],
            std_error=[0.1, 0.1, 0.1, 0.1],
            event_time_grid=[-2, -1, 0, 1],
            metadata={"aggregation": "eventstudy"},
        )
        # plot_dose_response should still work (it doesn't gate on aggregation)
        ax = plot_dose_response(result)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# 5. Export format validation
# ---------------------------------------------------------------------------


class TestExportFormatValidation:
    """Detailed validation of export outputs."""

    @pytest.fixture()
    def dose_result_with_band(self) -> ContDIDResult:
        """Produce a result with confidence band for export tests."""
        panel = _small_two_period_panel(n=200, seed=55)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        )
        return estimate_dose_effects(panel, spec)

    def test_dataframe_has_band_columns(self, dose_result_with_band: ContDIDResult):
        """DataFrame includes band_lower and band_upper when cband=True."""
        df = to_dataframe(dose_result_with_band)
        assert "band_lower" in df.columns
        assert "band_upper" in df.columns
        # Band should be wider than or equal to CI
        if "ci_lower" in df.columns and "ci_upper" in df.columns:
            assert (df["band_lower"] <= df["ci_lower"] + 1e-10).all()
            assert (df["band_upper"] >= df["ci_upper"] - 1e-10).all()

    def test_csv_roundtrip(self, dose_result_with_band: ContDIDResult, tmp_path: Path):
        """CSV export can be read back into a valid DataFrame."""
        csv_file = tmp_path / "roundtrip.csv"
        to_csv(dose_result_with_band, csv_file)
        df = pd.read_csv(csv_file)
        assert "dose" in df.columns
        assert "estimate" in df.columns
        assert len(df) == len(dose_result_with_band.grid)

    def test_latex_custom_caption_label(self, dose_result_with_band: ContDIDResult):
        """to_latex with custom caption and label."""
        latex_str = to_latex(
            dose_result_with_band,
            caption="Custom Caption",
            label="tab:custom",
        )
        assert "Custom Caption" in latex_str
        assert "tab:custom" in latex_str

    def test_summary_max_rows(self, dose_result_with_band: ContDIDResult):
        """summary(max_rows=10) truncates output."""
        text = summary(dose_result_with_band, max_rows=10)
        assert "..." in text
