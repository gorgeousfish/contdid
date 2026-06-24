"""Type stubs for contdid.estimation public API."""

from typing import Iterable

from .data import PanelData
from .results import ContDIDResult
from .specs import ContDIDSpec

def build_dose_grid(
    panel: PanelData, *, dvals: Iterable[float] | float | None = ...
) -> list[float]:
    """Build the evaluation grid from observed positive doses.

    Parameters
    ----------
    panel : PanelData
        Validated panel data.
    dvals : array-like or None
        Explicit grid values; None uses default quantile grid.

    Returns
    -------
    list[float]
        Dose grid within observed support.
    """
    ...

def estimate_dose_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = ...,
    degree: int = ...,
    num_knots: int = ...,
) -> ContDIDResult:
    """Estimate dose-level ATT(d) effects.

    Parameters
    ----------
    panel : PanelData
        Panel data with dose column.
    spec : ContDIDSpec
        Estimation specification (target_parameter='level').
    dvals : array-like or None
        Evaluation grid.
    degree : int
        B-spline degree (default 3).
    num_knots : int
        Number of interior knots (default 0).

    Returns
    -------
    ContDIDResult
    """
    ...

def estimate_dose_level_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = ...,
    degree: int = ...,
    num_knots: int = ...,
) -> ContDIDResult:
    """Alias for estimate_dose_effects."""
    ...

def estimate_dose_slope_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = ...,
    degree: int = ...,
    num_knots: int = ...,
) -> ContDIDResult:
    """Estimate dose-slope ACRT(d) effects.

    Parameters
    ----------
    panel : PanelData
        Panel data with dose column.
    spec : ContDIDSpec
        Estimation specification (target_parameter='slope').
    dvals : array-like or None
        Evaluation grid.
    degree : int
        B-spline degree (default 3).
    num_knots : int
        Number of interior knots (default 0).

    Returns
    -------
    ContDIDResult
    """
    ...

def estimate_dose_effects_multiperiod(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = ...,
    degree: int = ...,
    num_knots: int = ...,
    anticipation: int = ...,
    seed: int | None = ...,
) -> ContDIDResult:
    """Multi-period staggered design dose estimation.

    Parameters
    ----------
    panel : PanelData
        Balanced panel data (supports >2 time periods).
    spec : ContDIDSpec
        Estimation specification.
    dvals : array-like or None
        Evaluation grid.
    degree : int
        B-spline degree (default 3).
    num_knots : int
        Number of interior knots (default 0).
    anticipation : int
        Anticipation periods (default 0).
    seed : int or None
        Bootstrap random seed.

    Returns
    -------
    ContDIDResult
    """
    ...
