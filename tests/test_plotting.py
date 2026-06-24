"""Tests for plotting module."""
from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")


class _MockDoseResult:
    """Mock ContDIDResult for dose-response."""
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    estimate = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    std_error = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    estimand = "ATT(d)"
    critical_value = 1.96
    confidence_band = {
        "lower": [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1],
        "upper": [0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9, 4.4, 4.9],
    }
    confidence_interval = [[0.1, 0.9], [0.6, 1.4], [1.1, 1.9], [1.6, 2.4],
                          [2.1, 2.9], [2.6, 3.4], [3.1, 3.9], [3.6, 4.4], [4.1, 4.9]]
    metadata = {"aggregation": "dose"}


class _MockEventStudyResult:
    """Mock ContDIDResult for event study."""
    grid = [-3, -2, -1, 0, 1, 2]
    event_time_grid = [-3, -2, -1, 0, 1, 2]
    estimate = [0.1, -0.05, 0.02, 0.5, 1.0, 1.2]
    std_error = [0.1, 0.1, 0.1, 0.15, 0.2, 0.25]
    estimand = "ATT"
    critical_value = 1.96
    confidence_band = None
    confidence_interval = None
    metadata = {"aggregation": "eventstudy"}


class TestPlotDoseResponse:
    def test_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_dose_response

        ax = plot_dose_response(_MockDoseResult())
        assert ax is not None
        assert hasattr(ax, "plot")  # It's a matplotlib Axes
        plt.close("all")

    def test_custom_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_dose_response

        fig, ax = plt.subplots()
        returned_ax = plot_dose_response(_MockDoseResult(), ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_no_band(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_dose_response

        result = _MockDoseResult()
        result.confidence_band = None
        ax = plot_dose_response(result, show_confidence_band=False)
        assert ax is not None
        plt.close("all")

    def test_custom_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_dose_response

        ax = plot_dose_response(
            _MockDoseResult(), title="Custom Title", xlabel="Treatment", ylabel="Effect"
        )
        assert ax.get_title() == "Custom Title"
        assert ax.get_xlabel() == "Treatment"
        assert ax.get_ylabel() == "Effect"
        plt.close("all")


class TestPlotEventStudy:
    def test_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_eventstudy

        ax = plot_eventstudy(_MockEventStudyResult())
        assert ax is not None
        plt.close("all")

    def test_with_confidence_band(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_eventstudy

        result = _MockEventStudyResult()
        result.confidence_band = {
            "lower": [-0.1, -0.2, -0.1, 0.2, 0.6, 0.7],
            "upper": [0.3, 0.1, 0.15, 0.8, 1.4, 1.7],
        }
        ax = plot_eventstudy(result)
        assert ax is not None
        plt.close("all")


class TestAutoDispatch:
    def test_auto_dose(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot

        ax = plot(_MockDoseResult())
        assert ax is not None
        plt.close("all")

    def test_auto_eventstudy(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot

        ax = plot(_MockEventStudyResult())
        assert ax is not None
        plt.close("all")

    def test_explicit_kind(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot

        ax = plot(_MockDoseResult(), kind="dose")
        assert ax is not None
        plt.close("all")


class TestEdgeCases:
    def test_single_point(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from contdid.plotting import plot_dose_response

        class SinglePoint:
            grid = [0.5]
            estimate = [1.0]
            std_error = [0.1]
            estimand = "ATT(d)"
            critical_value = 1.96
            confidence_band = {"lower": [0.8], "upper": [1.2]}
            confidence_interval = [[0.8, 1.2]]
            metadata = {}

        ax = plot_dose_response(SinglePoint())
        assert ax is not None
        plt.close("all")
