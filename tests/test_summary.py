"""Tests for result summary and export utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from contdid.summary import summary, to_dataframe, to_latex, to_csv


class _MockResult:
    grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    estimate = [0.5, 1.0, 1.5, 2.0, 2.5]
    std_error = [0.2, 0.2, 0.2, 0.2, 0.2]
    estimand = "ATT(d)"
    critical_value = 1.96
    confidence_band = {"lower": [0.1, 0.6, 1.1, 1.6, 2.1],
                      "upper": [0.9, 1.4, 1.9, 2.4, 2.9]}
    confidence_interval = [[0.1, 0.9], [0.6, 1.4], [1.1, 1.9], [1.6, 2.4], [2.1, 2.9]]
    metadata = {"inference": "bootstrap", "alp": 0.05, "treated_count": 100,
               "untreated_count": 50, "basis": {"type": "bspline", "degree": 3, "num_knots": 0},
               "critical_value": 1.96, "confidence_band_kind": "simultaneous_multiplier",
               "aggregation": "dose"}


class _MockEventStudyResult:
    grid = [-2, -1, 0, 1, 2]
    event_time_grid = [-2, -1, 0, 1, 2]
    estimate = [0.1, -0.05, 0.5, 1.0, 1.2]
    std_error = [0.1, 0.1, 0.15, 0.2, 0.25]
    estimand = "ATT"
    critical_value = 1.96
    confidence_band = None
    confidence_interval = [[-0.1, 0.3], [-0.25, 0.15], [0.2, 0.8], [0.6, 1.4], [0.7, 1.7]]
    metadata = {"inference": "bootstrap", "alp": 0.05, "aggregation": "eventstudy"}


class TestSummary:
    def test_returns_string(self):
        s = summary(_MockResult())
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_estimand(self):
        s = summary(_MockResult())
        assert "ATT(d)" in s

    def test_contains_inference(self):
        s = summary(_MockResult())
        assert "bootstrap" in s

    def test_contains_grid_values(self):
        s = summary(_MockResult())
        assert "0.1" in s or "0.10" in s

    def test_contains_estimates(self):
        s = summary(_MockResult())
        assert "0.5" in s or "0.50" in s

    def test_max_rows(self):
        s_full = summary(_MockResult())
        s_short = summary(_MockResult(), max_rows=3)
        assert "..." in s_short
        assert len(s_short) < len(s_full)

    def test_event_study_result(self):
        s = summary(_MockEventStudyResult())
        assert "ATT" in s


class TestToDataframe:
    def test_returns_dataframe(self):
        df = to_dataframe(_MockResult())
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = to_dataframe(_MockResult())
        assert df.shape[0] == 5
        assert "dose" in df.columns
        assert "estimate" in df.columns
        assert "std_error" in df.columns

    def test_has_ci_columns(self):
        df = to_dataframe(_MockResult())
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns

    def test_has_band_columns(self):
        df = to_dataframe(_MockResult())
        assert "band_lower" in df.columns
        assert "band_upper" in df.columns

    def test_event_study_uses_event_time_column(self):
        df = to_dataframe(_MockEventStudyResult())
        assert "event_time" in df.columns
        assert "dose" not in df.columns

    def test_values_match(self):
        df = to_dataframe(_MockResult())
        np.testing.assert_allclose(df["estimate"].values, [0.5, 1.0, 1.5, 2.0, 2.5])


class TestToLatex:
    def test_returns_string(self):
        latex = to_latex(_MockResult())
        assert isinstance(latex, str)

    def test_contains_tabular(self):
        latex = to_latex(_MockResult())
        assert "tabular" in latex

    def test_custom_caption(self):
        latex = to_latex(_MockResult(), caption="My Table")
        assert "My Table" in latex

    def test_custom_label(self):
        latex = to_latex(_MockResult(), label="tab:my_results")
        assert "tab:my_results" in latex


class TestToCsv:
    def test_returns_string(self):
        csv_str = to_csv(_MockResult())
        assert isinstance(csv_str, str)

    def test_parseable(self):
        """CSV output should be parseable back by pandas."""
        from io import StringIO
        csv_str = to_csv(_MockResult())
        df = pd.read_csv(StringIO(csv_str))
        assert df.shape[0] == 5
        assert "estimate" in df.columns

    def test_to_file(self, tmp_path):
        """Writing to file should work."""
        path = tmp_path / "results.csv"
        to_csv(_MockResult(), path)
        df = pd.read_csv(path)
        assert df.shape[0] == 5


class TestEdgeCases:
    def test_no_ci(self):
        """Result without confidence interval should still work."""
        class NoCIResult:
            grid = [0.1, 0.2, 0.3]
            estimate = [1.0, 2.0, 3.0]
            std_error = [0.1, 0.1, 0.1]
            estimand = "ATT(d)"
            confidence_band = None
            confidence_interval = None
            metadata = {"aggregation": "dose"}

        s = summary(NoCIResult())
        assert "ATT(d)" in s
        df = to_dataframe(NoCIResult())
        assert df.shape[0] == 3
        assert "ci_lower" not in df.columns

    def test_single_point(self):
        class SinglePoint:
            grid = [0.5]
            estimate = [1.0]
            std_error = [0.1]
            estimand = "ATT(d)"
            confidence_band = None
            confidence_interval = [[0.8, 1.2]]
            metadata = {}

        s = summary(SinglePoint())
        assert "1.0" in s or "1.00" in s
        df = to_dataframe(SinglePoint())
        assert df.shape[0] == 1
