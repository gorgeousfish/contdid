"""Type stubs for contdid.plotting public API."""

from typing import Any

def plot_dose_response(
    result,
    *,
    ax: Any | None = ...,
    show_pointwise_ci: bool = ...,
    show_confidence_band: bool = ...,
    show_zero_line: bool = ...,
    color: str = ...,
    band_color: str = ...,
    ci_color: str = ...,
    zero_color: str = ...,
    title: str | None = ...,
    xlabel: str = ...,
    ylabel: str | None = ...,
    figsize: tuple[float, float] = ...,
) -> Any:
    """Plot dose-response curve with confidence bands.

    Parameters
    ----------
    result : ContDIDResult
        Estimation result.
    ax : matplotlib Axes or None
        If None, creates a new figure and axes.
    show_pointwise_ci : bool
        Whether to show pointwise confidence intervals.
    show_confidence_band : bool
        Whether to show simultaneous confidence band.
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
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib Axes
    """
    ...

def plot_eventstudy(
    result,
    *,
    ax: Any | None = ...,
    show_pointwise_ci: bool = ...,
    show_confidence_band: bool = ...,
    show_zero_line: bool = ...,
    color: str = ...,
    band_color: str = ...,
    zero_color: str = ...,
    title: str | None = ...,
    xlabel: str = ...,
    ylabel: str | None = ...,
    figsize: tuple[float, float] = ...,
) -> Any:
    """Plot event-study effects over event time.

    Parameters
    ----------
    result : ContDIDResult
        Estimation result with event_time_grid field.
    ax : matplotlib Axes or None
        If None, creates a new figure and axes.
    show_pointwise_ci : bool
        Whether to show pointwise confidence intervals.
    show_confidence_band : bool
        Whether to show simultaneous confidence band.
    show_zero_line : bool
        Whether to draw a horizontal line at y=0.
    color : str
        Color for point estimates.
    band_color : str
        Fill color for confidence band.
    zero_color : str
        Color for zero reference lines.
    title : str or None
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple
        Figure size if creating new figure.

    Returns
    -------
    matplotlib Axes
    """
    ...

def plot(result, kind: str = ..., **kwargs) -> Any:
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
    ...
