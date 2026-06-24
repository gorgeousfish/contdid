"""Visualization utilities for contdid estimation results.

Provides matplotlib-based plotting for dose-response curves and event-study
effects. Matplotlib is an optional dependency — import errors are raised with
a clear message when the plotting module is used without matplotlib installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .results import ContDIDResult


def _require_matplotlib():
    """Lazy import matplotlib; raise clear error if missing."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from None


def plot_dose_response(
    result: ContDIDResult,
    *,
    ax: Any | None = None,
    show_pointwise_ci: bool = True,
    show_confidence_band: bool = True,
    show_zero_line: bool = True,
    color: str = "#2563EB",
    band_color: str = "#BFDBFE",
    ci_color: str = "#93C5FD",
    zero_color: str = "#64748B",
    title: str | None = None,
    xlabel: str = "Dose",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Any:
    """Plot dose-response curve with confidence bands.

    Parameters
    ----------
    result : ContDIDResult
        Estimation result containing grid, estimate, std_error,
        confidence_band, and confidence_interval fields.
    ax : matplotlib Axes or None
        If None, creates a new figure and axes.
    show_pointwise_ci : bool
        Whether to show pointwise confidence intervals (lighter band).
    show_confidence_band : bool
        Whether to show simultaneous confidence band (if available).
    show_zero_line : bool
        Whether to draw a horizontal line at y=0.
    color : str
        Color for the estimate line.
    band_color : str
        Fill color for confidence band.
    ci_color : str
        Fill color for pointwise confidence intervals.
    zero_color : str
        Color for the zero reference line.
    title : str or None
        Plot title. If None, auto-generated from estimand.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib Axes
    """
    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    grid = np.asarray(result.grid, dtype=float)
    estimate = np.asarray(result.estimate, dtype=float)

    # Draw confidence band (simultaneous) if available
    if (
        show_confidence_band
        and hasattr(result, "confidence_band")
        and result.confidence_band is not None
    ):
        band = result.confidence_band
        if isinstance(band, dict) and "lower" in band and "upper" in band:
            lower = np.asarray(band["lower"], dtype=float)
            upper = np.asarray(band["upper"], dtype=float)
            ax.fill_between(
                grid,
                lower,
                upper,
                alpha=0.3,
                color=band_color,
                label="Simultaneous confidence band",
            )

    # Draw pointwise CI if available
    if (
        show_pointwise_ci
        and hasattr(result, "confidence_interval")
        and result.confidence_interval is not None
    ):
        ci = result.confidence_interval
        if isinstance(ci, list) and len(ci) > 0 and isinstance(ci[0], (list, tuple)):
            ci_lower = np.array([interval[0] for interval in ci])
            ci_upper = np.array([interval[1] for interval in ci])
            ax.fill_between(
                grid,
                ci_lower,
                ci_upper,
                alpha=0.2,
                color=ci_color,
                label="Pointwise CI",
            )

    # Draw zero line
    if show_zero_line:
        ax.axhline(y=0, color=zero_color, linestyle="--", linewidth=0.8, alpha=0.7)

    # Draw estimate line
    ax.plot(grid, estimate, color=color, linewidth=2, label=result.estimand or "Estimate")

    # Mark some points along the curve
    stride = max(1, len(grid) // 10)
    marker_idx = list(range(0, len(grid), stride))
    if len(grid) - 1 not in marker_idx:
        marker_idx.append(len(grid) - 1)
    ax.scatter(grid[marker_idx], estimate[marker_idx], color=color, s=20, zorder=5)

    # Labels
    if title is None:
        estimand = getattr(result, "estimand", "ATT(d)")
        title = f"Dose-Response: {estimand}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel or getattr(result, "estimand", "Effect"), fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_eventstudy(
    result: ContDIDResult,
    *,
    ax: Any | None = None,
    show_pointwise_ci: bool = True,
    show_confidence_band: bool = True,
    show_zero_line: bool = True,
    color: str = "#2563EB",
    band_color: str = "#BFDBFE",
    zero_color: str = "#64748B",
    title: str | None = None,
    xlabel: str = "Event Time",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Any:
    """Plot event-study effects over event time.

    Parameters
    ----------
    result : ContDIDResult
        Estimation result with event_time_grid field.
    ax : matplotlib Axes or None
        If None, creates a new figure and axes.
    show_pointwise_ci : bool
        Whether to show pointwise confidence intervals as error bars.
    show_confidence_band : bool
        Whether to show simultaneous confidence band (if available).
    show_zero_line : bool
        Whether to draw a horizontal line at y=0.
    color : str
        Color for point estimates and error bars.
    band_color : str
        Fill color for confidence band.
    zero_color : str
        Color for zero and event-onset reference lines.
    title : str or None
        Plot title. If None, auto-generated from estimand.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib Axes
    """
    # Validate this is actually event-study data (before any figure creation)
    # Gate requires BOTH explicit metadata AND structural evidence to guard
    # against unchecked mutable metadata being set on arbitrary results.
    meta = getattr(result, "metadata", {}) or {}
    aggregation = meta.get("aggregation")
    if aggregation != "eventstudy":
        if aggregation == "dose":
            raise ValueError(
                "plot_eventstudy called on a result with aggregation='dose'. "
                "Use plot_dose_response for dose results, "
                "or pass kind='dose' to plot()."
            )
        raise ValueError(
            "plot_eventstudy requires metadata['aggregation'] == 'eventstudy' "
            "to confirm the result is event-study data. "
            f"Got aggregation={aggregation!r}. "
            "Set result.metadata['aggregation'] = 'eventstudy' before plotting."
        )

    # Cross-validate: metadata claim must be substantiated by structure.
    # ContDIDResult allows event-study results to carry event_time_grid
    # OR event_time — accept either as structural evidence.
    event_times = getattr(result, "event_time_grid", None)
    if event_times is None:
        event_times = getattr(result, "event_time", None)
    if event_times is None:
        raise ValueError(
            "metadata['aggregation'] == 'eventstudy' but result has neither "
            "event_time_grid nor event_time attribute. The metadata tag is not "
            "substantiated by the result structure \u2014 cannot plot as event study."
        )
    event_times = np.asarray(event_times, dtype=float)
    estimate = np.asarray(result.estimate, dtype=float)
    std_error = np.asarray(result.std_error, dtype=float)

    n = len(event_times)
    if len(estimate) != n:
        raise ValueError(
            f"event_time_grid length ({n}) does not match estimate length "
            f"({len(estimate)}). The result may have mismatched fields."
        )
    if len(std_error) != n:
        raise ValueError(
            f"event_time_grid length ({n}) does not match std_error length "
            f"({len(std_error)}). The result may have mismatched fields."
        )

    plt = _require_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Zero line
    if show_zero_line:
        ax.axhline(y=0, color=zero_color, linestyle="--", linewidth=0.8, alpha=0.7)

    # Vertical line at event_time = -0.5 (treatment onset marker)
    ax.axvline(x=-0.5, color=zero_color, linestyle=":", linewidth=0.8, alpha=0.5)

    # Confidence band
    if (
        show_confidence_band
        and hasattr(result, "confidence_band")
        and result.confidence_band is not None
    ):
        band = result.confidence_band
        if isinstance(band, dict) and "lower" in band and "upper" in band:
            lower = np.asarray(band["lower"], dtype=float)
            upper = np.asarray(band["upper"], dtype=float)
            ax.fill_between(event_times, lower, upper, alpha=0.25, color=band_color)

    # Pointwise CI as error bars
    if show_pointwise_ci and np.any(std_error > 0):
        ci_half_pw = 1.96 * std_error
        ax.errorbar(
            event_times,
            estimate,
            yerr=ci_half_pw,
            fmt="o",
            color=color,
            capsize=3,
            markersize=5,
            label=result.estimand or "Effect",
        )
    else:
        ax.scatter(
            event_times,
            estimate,
            color=color,
            s=30,
            zorder=5,
            label=result.estimand or "Effect",
        )

    # Connect with line
    ax.plot(event_times, estimate, color=color, linewidth=1.5, alpha=0.7)

    # Labels
    if title is None:
        title = f"Event Study: {getattr(result, 'estimand', 'Effect')}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel or getattr(result, "estimand", "Effect"), fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Integer x-ticks for event time
    ax.set_xticks(event_times.astype(int))

    return ax


def plot(result: ContDIDResult, kind: str = "auto", **kwargs: Any) -> Any:
    """Dispatch to appropriate plot type based on result content.

    Parameters
    ----------
    result : ContDIDResult
        Estimation result.
    kind : str
        "dose" for dose-response, "eventstudy" for event study,
        "auto" to detect from result metadata.
    **kwargs
        Additional keyword arguments passed to the specific plot function.

    Returns
    -------
    matplotlib Axes
    """
    if kind == "auto":
        # Detect from metadata; only explicit aggregation field is trusted
        meta = getattr(result, "metadata", {}) or {}
        aggregation = meta.get("aggregation")
        if aggregation == "eventstudy":
            kind = "eventstudy"
        else:
            kind = "dose"

    if kind == "eventstudy":
        return plot_eventstudy(result, **kwargs)
    else:
        return plot_dose_response(result, **kwargs)
