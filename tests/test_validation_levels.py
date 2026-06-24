"""Tests for hierarchical validation severity mechanism.

Verifies that validation checks are correctly classified into ERROR/WARNING/INFO
levels, and that ValidationStrictness controls which levels block execution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from contdid import PanelData
from contdid.validation import (
    ContDIDValidationError,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationStrictness,
    validate_panel_data,
    validate_panel_data_report,
)


# ---------------------------------------------------------------------------
# Fixtures: valid and invalid panels
# ---------------------------------------------------------------------------


def _make_valid_panel() -> PanelData:
    """Create a minimal valid two-period balanced panel."""
    n_treated = 50
    n_control = 50
    n = n_treated + n_control
    ids = list(range(1, n + 1))
    # Two periods: t=1, t=2; treatment at t=2
    rows = []
    for i in ids:
        is_treated = i <= n_treated
        group = 2 if is_treated else 0
        dose = float(i) * 0.1 if is_treated else 0.0
        for t in [1, 2]:
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "outcome": np.random.default_rng(i + t).normal(),
                    "group": group,
                    "dose": dose,
                }
            )
    df = pd.DataFrame(rows)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",
    )


def _make_unbalanced_panel() -> PanelData:
    """Panel missing observations for some units (ERROR: balanced_panel)."""
    rows = [
        {"id": 1, "time": 1, "outcome": 1.0, "group": 2, "dose": 1.0},
        {"id": 1, "time": 2, "outcome": 2.0, "group": 2, "dose": 1.0},
        {"id": 2, "time": 1, "outcome": 1.0, "group": 0, "dose": 0.0},
        # id=2, time=2 is missing -> unbalanced
    ]
    df = pd.DataFrame(rows)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",
    )


def _make_nan_outcome_panel() -> PanelData:
    """Panel with NaN outcomes (WARNING: finite_outcomes)."""
    rows = []
    for i in range(1, 11):
        is_treated = i <= 5
        group = 2 if is_treated else 0
        dose = 1.5 if is_treated else 0.0
        for t in [1, 2]:
            outcome = float("nan") if (i == 3 and t == 2) else float(i + t)
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "outcome": outcome,
                    "group": group,
                    "dose": dose,
                }
            )
    df = pd.DataFrame(rows)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",
    )


def _make_negative_dose_panel() -> PanelData:
    """Panel with negative dose (WARNING: nonnegative_dose)."""
    rows = []
    for i in range(1, 11):
        is_treated = i <= 5
        group = 2 if is_treated else 0
        dose = -0.5 if (is_treated and i == 1) else (1.0 if is_treated else 0.0)
        for t in [1, 2]:
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "outcome": float(i + t),
                    "group": group,
                    "dose": dose,
                }
            )
    df = pd.DataFrame(rows)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",
    )


def _make_never_treated_violation_panel() -> PanelData:
    """Panel violating never-treated rule: G=0 but D>0 (ERROR)."""
    rows = []
    for i in range(1, 11):
        is_treated = i <= 5
        group = 2 if is_treated else 0
        # Unit 6 has group=0 but dose=1.0 -> violation
        dose = 1.0 if (i == 6) else (1.5 if is_treated else 0.0)
        for t in [1, 2]:
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "outcome": float(i + t),
                    "group": group,
                    "dose": dose,
                }
            )
    df = pd.DataFrame(rows)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",
    )


def _make_missing_column_panel() -> PanelData:
    """Panel with missing required column (ERROR: required_columns)."""
    df = pd.DataFrame(
        {"id": [1, 1], "time": [1, 2], "outcome": [1.0, 2.0], "group": [0, 0]}
    )
    # dose column is missing from the dataframe
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time",
        outcome_column="outcome",
        group_column="group",
        dose_column="dose",  # column does not exist in df
    )


# ---------------------------------------------------------------------------
# Tests: ValidationSeverity classification
# ---------------------------------------------------------------------------


class TestSeverityClassification:
    """Verify that each check is classified at the correct severity level."""

    def test_required_columns_is_error(self):
        panel = _make_missing_column_panel()
        report = validate_panel_data_report(panel)
        assert len(report.errors) >= 1
        assert report.errors[0].check_name == "required_columns"
        assert report.errors[0].severity == ValidationSeverity.ERROR

    def test_balanced_panel_is_error(self):
        panel = _make_unbalanced_panel()
        report = validate_panel_data_report(panel)
        assert len(report.errors) >= 1
        error_names = [e.check_name for e in report.errors]
        assert "balanced_panel" in error_names

    def test_never_treated_rule_is_error(self):
        panel = _make_never_treated_violation_panel()
        report = validate_panel_data_report(panel)
        assert len(report.errors) >= 1
        error_names = [e.check_name for e in report.errors]
        assert "never_treated_rule" in error_names

    def test_finite_outcomes_is_warning(self):
        panel = _make_nan_outcome_panel()
        report = validate_panel_data_report(panel)
        warning_names = [w.check_name for w in report.warnings]
        assert "finite_outcomes" in warning_names

    def test_nonnegative_dose_is_warning(self):
        panel = _make_negative_dose_panel()
        report = validate_panel_data_report(panel)
        warning_names = [w.check_name for w in report.warnings]
        assert "nonnegative_dose" in warning_names

    def test_sample_diagnostics_is_info(self):
        panel = _make_valid_panel()
        report = validate_panel_data_report(panel)
        info_names = [i.check_name for i in report.infos]
        assert "sample_diagnostics" in info_names

    def test_valid_panel_has_no_errors_or_warnings(self):
        panel = _make_valid_panel()
        report = validate_panel_data_report(panel)
        assert report.is_valid
        assert len(report.errors) == 0
        assert len(report.warnings) == 0


# ---------------------------------------------------------------------------
# Tests: ValidationStrictness behavior
# ---------------------------------------------------------------------------


class TestStrictnessBehavior:
    """Verify that strictness controls blocking behavior correctly."""

    def test_strict_mode_blocks_on_warning(self):
        """STRICT mode: Warning-level issues should raise."""
        panel = _make_nan_outcome_panel()
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(panel, strictness=ValidationStrictness.STRICT)

    def test_normal_mode_warns_on_warning(self):
        """NORMAL mode: Warning-level issues emit warnings but don't raise."""
        panel = _make_nan_outcome_panel()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_panel_data(panel, strictness=ValidationStrictness.NORMAL)
            assert result is panel
            assert len(w) >= 1
            assert any("[contdid]" in str(warning.message) for warning in w)

    def test_lenient_mode_silent_on_warning(self):
        """LENIENT mode: Warning-level issues are silent."""
        panel = _make_nan_outcome_panel()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_panel_data(panel, strictness=ValidationStrictness.LENIENT)
            assert result is panel
            # No warnings should be emitted
            contdid_warnings = [x for x in w if "[contdid]" in str(x.message)]
            assert len(contdid_warnings) == 0

    def test_error_always_blocks_regardless_of_strictness(self):
        """ERROR-level issues always raise, even in LENIENT mode."""
        panel = _make_unbalanced_panel()
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(panel, strictness=ValidationStrictness.LENIENT)

    def test_error_blocks_in_normal_mode(self):
        panel = _make_never_treated_violation_panel()
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(panel, strictness=ValidationStrictness.NORMAL)


# ---------------------------------------------------------------------------
# Tests: ValidationReport API
# ---------------------------------------------------------------------------


class TestValidationReport:
    """Test ValidationReport properties and methods."""

    def test_empty_report_is_valid(self):
        report = ValidationReport(issues=[])
        assert report.is_valid
        assert report.errors == []
        assert report.warnings == []
        assert report.infos == []

    def test_report_with_error_not_valid(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="test",
                    message="test error",
                )
            ]
        )
        assert not report.is_valid

    def test_report_with_only_warnings_is_valid(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="test",
                    message="test warning",
                )
            ]
        )
        assert report.is_valid

    def test_report_raise_if_invalid_strict_with_warning(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="test_w",
                    message="a warning",
                )
            ]
        )
        with pytest.raises(ContDIDValidationError, match="a warning"):
            report.raise_if_invalid(ValidationStrictness.STRICT)

    def test_report_raise_if_invalid_normal_with_warning(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="test_w",
                    message="a warning",
                )
            ]
        )
        # Should NOT raise, but emit warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report.raise_if_invalid(ValidationStrictness.NORMAL)
            assert len(w) >= 1

    def test_report_raise_if_invalid_with_error(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="test_e",
                    message="an error",
                )
            ]
        )
        with pytest.raises(ContDIDValidationError, match="an error"):
            report.raise_if_invalid(ValidationStrictness.LENIENT)

    def test_report_str_empty(self):
        report = ValidationReport(issues=[])
        assert "all checks passed" in str(report)

    def test_report_str_with_issues(self):
        report = ValidationReport(
            issues=[
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="x",
                    message="msg",
                )
            ]
        )
        output = str(report)
        assert "[ERROR]" in output
        assert "msg" in output

    def test_sample_diagnostics_details(self):
        """Sample diagnostics INFO should include details dict."""
        panel = _make_valid_panel()
        report = validate_panel_data_report(panel)
        diag = [i for i in report.infos if i.check_name == "sample_diagnostics"]
        assert len(diag) == 1
        details = diag[0].details
        assert details is not None
        assert "n_units" in details
        assert "n_treated" in details
        assert "n_control" in details
        assert details["n_units"] == 100
        assert details["n_treated"] == 50
        assert details["n_control"] == 50


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure default behavior matches pre-refactor behavior."""

    def test_default_strictness_is_strict(self):
        """Default call without strictness arg should behave like before."""
        panel = _make_valid_panel()
        # Should return panel without raising
        result = validate_panel_data(panel)
        assert result is panel

    def test_default_raises_on_formerly_raising_checks(self):
        """Checks that previously raised should still raise by default."""
        # Unbalanced panel
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(_make_unbalanced_panel())

        # NaN outcomes (was previously raise, now WARNING but STRICT still raises)
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(_make_nan_outcome_panel())

        # Negative dose (was previously raise, now WARNING but STRICT still raises)
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(_make_negative_dose_panel())

        # Never-treated violation
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(_make_never_treated_violation_panel())

    def test_valid_panel_passes_default(self):
        """Valid panel should pass without raising."""
        panel = _make_valid_panel()
        result = validate_panel_data(panel)
        assert result is panel

    def test_spec_parameter_is_optional(self):
        """spec parameter should be optional (backward compat)."""
        panel = _make_valid_panel()
        # These should all work
        validate_panel_data(panel)
        validate_panel_data(panel, spec=None)
        validate_panel_data(panel, strictness=ValidationStrictness.STRICT)


# ---------------------------------------------------------------------------
# Tests: ContDIDSpec.validation_strictness integration
# ---------------------------------------------------------------------------


class TestSpecStrictnessIntegration:
    """Test that ContDIDSpec.validation_strictness propagates correctly."""

    def test_spec_default_strictness_is_strict(self):
        from contdid import ContDIDSpec

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert spec.validation_strictness == "strict"

    def test_spec_accepts_all_strictness_values(self):
        from contdid import ContDIDSpec

        for val in ("strict", "normal", "lenient"):
            spec = ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                validation_strictness=val,
            )
            assert spec.validation_strictness == val
