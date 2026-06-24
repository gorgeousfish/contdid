"""Type stubs for contdid.summary public API."""

import pandas as pd

def summary(result, *, max_rows: int | None = ...) -> str:
    """Generate a formatted text summary of estimation results.

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
    ...

def to_dataframe(result) -> pd.DataFrame:
    """Convert estimation results to a pandas DataFrame.

    Parameters
    ----------
    result : ContDIDResult

    Returns
    -------
    pd.DataFrame
    """
    ...

def to_latex(
    result,
    *,
    caption: str | None = ...,
    label: str | None = ...,
    float_format: str = ...,
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
    ...

def to_csv(result, path_or_buf=..., **kwargs) -> str | None:
    """Export results to CSV.

    Parameters
    ----------
    result : ContDIDResult
    path_or_buf : str, Path, or file-like, or None
        If None, returns CSV string. Otherwise writes to file.

    Returns
    -------
    str or None
        CSV string if path_or_buf is None, else None.
    """
    ...
