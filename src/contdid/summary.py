"""Result summary and export utilities for contdid estimation.

Provides formatted text summaries (similar to statsmodels), DataFrame
conversion, LaTeX table generation, and CSV export.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .results import ContDIDResult


def summary(result: ContDIDResult, *, max_rows: int | None = None) -> str:
    """Generate a formatted text summary of estimation results.

    Similar to statsmodels' summary() output, shows:
    - Header: estimand type, sample info, inference method
    - Body: grid points, estimates, SEs, confidence intervals
    - Footer: metadata highlights

    Parameters
    ----------
    result : ContDIDResult
        Estimation result.
    max_rows : int or None
        Maximum rows to display. None shows all.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines: list[str] = []
    meta = getattr(result, "metadata", {})

    # Header
    lines.append("=" * 72)
    lines.append(f"{'ContDID Estimation Results':^72}")
    lines.append("=" * 72)
    lines.append(f"Estimand:           {getattr(result, 'estimand', 'N/A')}")
    lines.append(f"Inference:          {meta.get('inference', 'N/A')}")
    lines.append(f"Confidence level:   {1 - meta.get('alp', 0.05):.0%}")

    # Sample info from metadata
    if "treated_count" in meta:
        lines.append(f"N (treated):        {meta['treated_count']}")
    if "untreated_count" in meta:
        lines.append(f"N (untreated):      {meta['untreated_count']}")
    if "n_clusters" in meta and meta["n_clusters"] is not None:
        lines.append(f"N (clusters):       {meta['n_clusters']}")

    # Basis info
    basis = meta.get("basis", {})
    if basis:
        lines.append(
            f"Basis:              {basis.get('type', 'N/A')} "
            f"(degree={basis.get('degree', '?')}, "
            f"knots={basis.get('num_knots', '?')})"
        )

    lines.append("-" * 72)

    # Body: table of results
    grid = np.asarray(getattr(result, "grid", []), dtype=float)
    estimate = np.asarray(getattr(result, "estimate", []), dtype=float)
    std_error = np.asarray(getattr(result, "std_error", []), dtype=float)

    # Determine CI availability
    ci = getattr(result, "confidence_interval", None)
    has_ci = ci is not None and isinstance(ci, list) and len(ci) > 0

    # Column header
    header = f"{'Grid':>10}  {'Estimate':>10}  {'Std.Err.':>10}"
    if has_ci:
        header += f"  {'CI Lower':>10}  {'CI Upper':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    n_rows = len(grid)
    truncated = max_rows is not None and n_rows > max_rows

    if truncated:
        # 等间距取样，覆盖整个范围（参考 Julia ContDiD.jl）
        indices = np.round(np.linspace(0, n_rows - 1, max_rows)).astype(int)
    else:
        indices = np.arange(n_rows)

    def _format_row(i: int) -> str:
        row = f"{grid[i]:>10.4f}  {estimate[i]:>10.4f}  {std_error[i]:>10.4f}"
        if has_ci and i < len(ci):
            row += f"  {ci[i][0]:>10.4f}  {ci[i][1]:>10.4f}"
        return row

    for i in indices:
        lines.append(_format_row(i))

    if truncated:
        lines.append(f"  ... ({max_rows} of {n_rows} points shown, equidistant sampling)")

    lines.append("-" * 72)

    # Footer
    cv = getattr(result, "critical_value", None)
    if cv is not None:
        lines.append(f"Critical value:     {cv:.4f}")
    elif meta.get("critical_value") is not None:
        lines.append(f"Critical value:     {meta['critical_value']:.4f}")

    band_kind = meta.get("confidence_band_kind", "")
    if band_kind:
        lines.append(f"Band type:          {band_kind}")
    if meta.get("cluster_column"):
        lines.append(f"Cluster variable:   {meta['cluster_column']}")
    lines.append("=" * 72)

    return "\n".join(lines)


def to_dataframe(result: ContDIDResult) -> pd.DataFrame:
    """Convert estimation results to a pandas DataFrame.

    Columns: grid (or event_time), estimate, std_error, ci_lower, ci_upper,
             band_lower, band_upper (if available).

    Parameters
    ----------
    result : ContDIDResult

    Returns
    -------
    pd.DataFrame
    """
    # Determine grid column name
    has_event_time = getattr(result, "event_time_grid", None) is not None
    meta = getattr(result, "metadata", {})
    aggregation = meta.get("aggregation")

    if aggregation == "eventstudy" or (has_event_time and aggregation != "dose"):
        grid_col = "event_time"
        grid_vals = list(result.event_time_grid if has_event_time else result.grid)
    else:
        grid_col = "dose"
        grid_vals = list(result.grid)

    data: dict[str, list[Any]] = {
        grid_col: grid_vals,
        "estimate": list(result.estimate),
        "std_error": list(result.std_error),
    }

    # Confidence interval
    ci = getattr(result, "confidence_interval", None)
    if ci is not None and isinstance(ci, list) and len(ci) > 0:
        data["ci_lower"] = [interval[0] for interval in ci]
        data["ci_upper"] = [interval[1] for interval in ci]

    # Confidence band
    band = getattr(result, "confidence_band", None)
    if band is not None and isinstance(band, dict):
        if "lower" in band and "upper" in band:
            data["band_lower"] = list(band["lower"])
            data["band_upper"] = list(band["upper"])

    # Support state (which grid points have valid local support)
    # Contract: metadata["support"] must be a list of bool values matching grid length
    support = meta.get("support")
    if support is not None:
        n_grid = len(grid_vals)
        # Handle scalar bool (broadcast to all grid points)
        if isinstance(support, (bool, np.bool_)):
            data["supported"] = [bool(support)] * n_grid
        else:
            if not isinstance(support, (bool, np.bool_)):
                # Non-bool scalar: contract violation
                try:
                    support_arr = list(support)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"metadata['support'] must be a list of bool values or a "
                        f"scalar bool; got non-iterable {type(support).__name__}"
                    )
            else:
                support_arr = list(support)

            if len(support_arr) != n_grid:
                raise ValueError(
                    f"metadata['support'] length ({len(support_arr)}) must match "
                    f"grid length ({n_grid})"
                )
            # Contract requires only bool/np.bool_ values
            invalid = [s for s in support_arr if not isinstance(s, (bool, np.bool_))]
            if invalid:
                raise ValueError(
                    f"metadata['support'] must contain only boolean values; "
                    f"found non-boolean: {invalid[:3]}{'...' if len(invalid) > 3 else ''}"
                )
            data["supported"] = [bool(s) for s in support_arr]

    return pd.DataFrame(data)


def to_latex(
    result: ContDIDResult,
    *,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "%.4f",
) -> str:
    """Export results as a LaTeX table.

    Parameters
    ----------
    result : ContDIDResult
    caption : str or None
        Table caption.
    label : str or None
        LaTeX label for referencing.
    float_format : str
        Format string for numeric values.

    Returns
    -------
    str
        LaTeX table source.
    """
    df = to_dataframe(result)

    if caption is None:
        estimand = getattr(result, "estimand", "Effect")
        caption = f"Estimation Results: {estimand}"
    if label is None:
        label = "tab:contdid_results"

    latex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
    )
    return latex


def to_csv(
    result: ContDIDResult, path_or_buf: str | Path | IO[str] | None = None, **kwargs: Any
) -> str | None:
    """Export results to CSV.

    Parameters
    ----------
    result : ContDIDResult
    path_or_buf : str, Path, or file-like, or None
        If None, returns CSV string. Otherwise writes to file.
    **kwargs
        Additional arguments passed to DataFrame.to_csv().

    Returns
    -------
    str or None
        CSV string if path_or_buf is None, else None.
    """
    df = to_dataframe(result)
    return df.to_csv(path_or_buf, index=False, **kwargs)
