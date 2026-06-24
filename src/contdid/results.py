"""Result containers shared by the contdid runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


_PLOT_WIDTH = 1260
_PLOT_HEIGHT = 810
_PLOT_MARGIN_LEFT = 112
_PLOT_MARGIN_RIGHT = 56
_PLOT_MARGIN_TOP = 104
_PLOT_MARGIN_BOTTOM = 116
_PLOT_BACKGROUND = "#FFFFFF"
_PLOT_PANEL = "#F8FAFC"
_PLOT_GRID = "#E2E8F0"
_PLOT_AXIS = "#475569"
_PLOT_TEXT = "#0F172A"
_PLOT_MUTED_TEXT = "#475569"
_PLOT_LINE = "#2563EB"
_PLOT_POINT = "#1D4ED8"
_PLOT_UNSUPPORTED_POINT = "#94A3B8"
_PLOT_INTERVAL = "#1E40AF"
_PLOT_BAND = "#BFDBFE"
_PLOT_BAND_OUTLINE = "#93C5FD"
_PLOT_ZERO = "#334155"
_PLOT_EVENT_ZERO = "#64748B"
_MARKDOWN_MAX_ROWS_ERROR = "max_rows must be None or an integer greater than or equal to 2"
_TIMING_GROUP_REPORTING_SCALE = "length of exposure to treatment"
_TIMING_GROUP_BASE_PERIOD_STRATEGIES = frozenset({"fixed", "universal", "varying_pre_period"})
_IDENTIFICATION_PAYLOADS = {
    "ATT(d)": {
        "paper_estimand": "ATT(d)",
        "identifying_assumption": "SPT",
        "ordinary_pt_interpretation": "LATT(d|d)",
        "identification_note": (
            "The same dose-specific contrast identifies LATT(d|d) under "
            "ordinary PT; interpreting it as ATT(d) requires SPT."
        ),
    },
    "ACRT(d)": {
        "paper_estimand": "ACRT(d)",
        "identifying_assumption": "SPT + continuous dose support",
        "ordinary_pt_interpretation": (
            "derivative of LATT(d|d) with local selection-bias contamination"
        ),
        "identification_note": (
            "Ordinary PT is not enough for a causal ACRT(d) interpretation; "
            "the public slope route reports the SPT-based causal-response label."
        ),
    },
    "ATT(event_time)": {
        "paper_estimand": "ATT(event_time)",
        "identifying_assumption": "PT-MP",
        "ordinary_pt_interpretation": (
            "post-treatment ATT(event_time); negative event-time cells are pre-trend diagnostics"
        ),
        "identification_note": (
            "Post-treatment ATT(event_time) cells are identified by "
            "PT-MP/local binary event-study comparisons; negative event-time "
            "cells diagnose pre-treatment parallel-trends plausibility rather "
            "than treatment effects."
        ),
    },
    "ACRT(event_time)": {
        "paper_estimand": "ACRT(event_time)",
        "identifying_assumption": "SPT-MP + continuous dose support",
        "ordinary_pt_interpretation": (
            "derivative of event-time LATT path with local selection-bias "
            "contamination under PT-MP alone"
        ),
        "identification_note": (
            "The public slope event-study route reports the SPT-MP "
            "causal-response label; under PT-MP alone, differentiating "
            "event-time paths can retain selection-bias terms."
        ),
    },
}


def _normalize_confidence_interval(
    confidence_interval: Any,
    estimate: np.ndarray,
) -> list[list[float]] | None:
    if confidence_interval is None:
        return None
    if _contains_bool_values(confidence_interval):
        raise ValueError("confidence_interval must contain only finite non-boolean values")
    try:
        interval = np.asarray(confidence_interval, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "confidence_interval must contain only finite non-boolean values"
        ) from exc
    if interval.ndim != 2 or interval.shape != (estimate.shape[0], 2):
        raise ValueError("confidence_interval must have one lower/upper pair per estimate")
    if not np.isfinite(interval).all():
        raise ValueError("confidence_interval must contain only finite non-boolean values")
    if np.any(interval[:, 0] > interval[:, 1]):
        raise ValueError("confidence_interval lower bounds must not exceed upper bounds")
    if np.any(interval[:, 0] > estimate) or np.any(estimate > interval[:, 1]):
        raise ValueError("confidence_interval must contain the point estimate")
    return [[float(lower), float(upper)] for lower, upper in interval.tolist()]  # type: ignore[arg-type]


def _normalize_confidence_band(
    confidence_band: Any,
    estimate: np.ndarray,
    result_critical_value: float | None,
) -> dict[str, Any] | None:
    if confidence_band is None:
        return None
    if not isinstance(confidence_band, dict):
        raise ValueError("confidence_band must be a mapping with lower, upper, and critical_value")
    required_keys = {"lower", "upper", "critical_value"}
    if not required_keys.issubset(confidence_band):
        raise ValueError("confidence_band must contain lower, upper, and critical_value")
    if _contains_bool_values(confidence_band["lower"]) or _contains_bool_values(
        confidence_band["upper"]
    ):
        raise ValueError(
            "confidence_band lower and upper must contain only finite non-boolean values"
        )
    try:
        lower = np.asarray(confidence_band["lower"], dtype=float)
        upper = np.asarray(confidence_band["upper"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "confidence_band lower and upper must contain only finite non-boolean values"
        ) from exc
    if (
        lower.ndim != 1
        or upper.ndim != 1
        or lower.shape != estimate.shape
        or upper.shape != estimate.shape
    ):
        raise ValueError("confidence_band lower and upper must match estimate shape")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(
            "confidence_band lower and upper must contain only finite non-boolean values"
        )
    if np.any(lower > upper):
        raise ValueError("confidence_band lower bounds must not exceed upper bounds")
    if np.any(lower > estimate) or np.any(estimate > upper):
        raise ValueError("confidence_band must contain the point estimate")
    band_critical_value = confidence_band["critical_value"]
    if isinstance(band_critical_value, (bool, np.bool_)):
        raise ValueError("confidence_band critical_value must be finite and nonnegative")
    try:
        critical = float(band_critical_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_band critical_value must be finite and nonnegative") from exc
    if not np.isfinite(critical) or critical < 0.0:
        raise ValueError("confidence_band critical_value must be finite and nonnegative")
    if result_critical_value is not None and not np.isclose(
        critical,
        result_critical_value,
        rtol=0.0,
        atol=np.finfo(float).eps * max(1.0, abs(critical), abs(result_critical_value)),
    ):
        raise ValueError("critical_value must match confidence_band critical_value")
    normalized = dict(confidence_band)
    normalized["lower"] = [float(value) for value in lower.tolist()]  # type: ignore[arg-type]
    normalized["upper"] = [float(value) for value in upper.tolist()]  # type: ignore[arg-type]
    normalized["critical_value"] = critical
    return normalized


def _contains_bool_values(values: Any) -> bool:
    array = np.asarray(values, dtype=object)
    return any(isinstance(value, (bool, np.bool_)) for value in array.ravel())


def _validate_table_digits(digits: int) -> int:
    if isinstance(digits, (bool, np.bool_)) or not isinstance(digits, Integral):
        raise ValueError("digits must be an integer between 0 and 12")
    normalized = int(digits)
    if normalized < 0 or normalized > 12:
        raise ValueError("digits must be an integer between 0 and 12")
    return normalized


def _validate_include_caption(include_caption: bool) -> bool:
    if not isinstance(include_caption, (bool, np.bool_)):
        raise ValueError("include_caption must be a boolean")
    return bool(include_caption)


def _validate_markdown_max_rows(max_rows: int | None) -> int | None:
    if max_rows is None:
        return None
    if isinstance(max_rows, (bool, np.bool_)) or not isinstance(max_rows, Integral):
        raise ValueError(_MARKDOWN_MAX_ROWS_ERROR)
    normalized = int(max_rows)
    if normalized < 2:
        raise ValueError(_MARKDOWN_MAX_ROWS_ERROR)
    return normalized


def _format_table_float(value: object, *, digits: int = 6) -> str:
    formatted = f"{float(value):.{digits}f}"  # type: ignore[arg-type]
    if not formatted.startswith("-"):
        return formatted
    magnitude = formatted[1:]
    if magnitude == "0" or (magnitude.startswith("0.") and set(magnitude[2:]) <= {"0"}):
        return magnitude
    return formatted


def _format_table_interval(lower: object, upper: object, *, digits: int = 6) -> str:
    return (
        f"[{_format_table_float(lower, digits=digits)}, "
        f"{_format_table_float(upper, digits=digits)}]"
    )


def _confidence_band_display_label(metadata: dict[str, Any]) -> tuple[str, str]:
    kind = metadata.get("confidence_band_kind")
    if kind == "simultaneous_multiplier" or metadata.get("cband") is True:
        return "Uniform band", "uniform band"
    if kind in {"pointwise_multiplier", "pointwise_analytic"} or (metadata.get("cband") is False):
        return "Pointwise band", "pointwise band"
    return "Confidence band", "confidence band"


def _missing_table_cell(label: str) -> str:
    return f"not estimated ({label})"


def _checked_display_support_values(values: Iterable[Any]) -> list[bool]:
    support_values: list[bool] = []
    for value in values:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("display support must contain only boolean values")
        support_values.append(bool(value))
    return support_values


def _markdown_display_rows(
    frame: pd.DataFrame,
    *,
    max_rows: int | None,
) -> list[pd.Series | dict[str, Any]]:
    if max_rows is None or len(frame) <= max_rows:
        return [row for _, row in frame.iterrows()]
    head_count = max_rows // 2
    tail_count = max_rows - head_count
    omitted_count = len(frame) - max_rows
    omitted_rows = frame.iloc[head_count:-tail_count]
    axis_name = str(frame.columns[0])
    omitted_support: dict[str, int] | None = None
    if "support" in omitted_rows.columns:
        support_values = _checked_display_support_values(omitted_rows["support"].tolist())
        omitted_support = {
            "supported": sum(support_values),
            "total": int(len(support_values)),
        }
    head_rows = [row for _, row in frame.iloc[:head_count].iterrows()]
    tail_rows = [row for _, row in frame.iloc[-tail_count:].iterrows()]
    omitted_marker = {
        "kind": "omitted",
        "count": omitted_count,
        "axis_name": axis_name,
        "axis_first": omitted_rows.iloc[0][axis_name],
        "axis_last": omitted_rows.iloc[-1][axis_name],
        "support": omitted_support,
    }
    return [*head_rows, omitted_marker, *tail_rows]


def _format_markdown_omitted_axis_range(
    marker: dict[str, Any],
    *,
    digits: int,
) -> str:
    axis_name = marker["axis_name"]
    axis_first = marker["axis_first"]
    axis_last = marker["axis_last"]
    if axis_name == "event_time":
        first = str(int(axis_first))
        last = str(int(axis_last))
        axis_label = "event times"
    else:
        first = _format_table_float(axis_first, digits=digits)
        last = _format_table_float(axis_last, digits=digits)
        axis_label = "dose values"
    if first == last:
        return f"{axis_label} {first}"
    return f"{axis_label} {first} to {last}"


def _format_markdown_omitted_marker(
    marker: dict[str, Any],
    *,
    digits: int,
) -> str:
    omitted_range = _format_markdown_omitted_axis_range(marker, digits=digits)
    details = [omitted_range]
    support = marker.get("support")
    if isinstance(support, dict):
        details.append(f"support {support['supported']}/{support['total']}")
    return f"... {marker['count']} rows omitted ({'; '.join(details)}) ..."


def _pointwise_band_duplicates_interval(
    row: pd.Series,
    *,
    metadata: dict[str, Any],
) -> bool:
    kind = metadata.get("confidence_band_kind")
    if kind not in {"pointwise_multiplier", "pointwise_analytic"} and (
        metadata.get("cband") is not False
    ):
        return False
    required_columns = {"ci_lower", "ci_upper", "band_lower", "band_upper"}
    if not required_columns.issubset(row.index):
        return False
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    band_lower = float(row["band_lower"])
    band_upper = float(row["band_upper"])
    tolerance = np.finfo(float).eps * max(
        1.0,
        abs(lower),
        abs(upper),
        abs(band_lower),
        abs(band_upper),
    )
    return bool(
        np.isclose(lower, band_lower, rtol=0.0, atol=tolerance)
        and np.isclose(upper, band_upper, rtol=0.0, atol=tolerance)
    )


def _plot_band_duplicates_intervals(result: ContDIDResult) -> bool:
    if result.confidence_interval is None or result.confidence_band is None:
        return False
    kind = result.metadata.get("confidence_band_kind")
    if kind not in {"pointwise_multiplier", "pointwise_analytic"} and (
        result.metadata.get("cband") is not False
    ):
        return False
    rendered_indices = _plot_supported_indices(result)
    if not rendered_indices:
        return False
    interval = np.asarray(result.confidence_interval, dtype=float)
    band_lower = np.asarray(result.confidence_band["lower"], dtype=float)
    band_upper = np.asarray(result.confidence_band["upper"], dtype=float)
    interval_lower = interval[rendered_indices, 0]
    interval_upper = interval[rendered_indices, 1]
    rendered_band_lower = band_lower[rendered_indices]
    rendered_band_upper = band_upper[rendered_indices]
    tolerance = np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(interval_lower))) if interval_lower.size else 0.0,
        float(np.max(np.abs(interval_upper))) if interval_upper.size else 0.0,
        float(np.max(np.abs(rendered_band_lower))) if rendered_band_lower.size else 0.0,
        float(np.max(np.abs(rendered_band_upper))) if rendered_band_upper.size else 0.0,
    )
    return bool(
        np.allclose(interval_lower, rendered_band_lower, rtol=0.0, atol=tolerance)
        and np.allclose(interval_upper, rendered_band_upper, rtol=0.0, atol=tolerance)
    )


def _normalize_optional_integer_vector(
    name: str,
    values: Any,
    *,
    expected_length: int | None = None,
) -> list[int] | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    if expected_length is not None and array.shape[0] != expected_length:
        raise ValueError(f"{name} must match result estimate length")
    normalized: list[int] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain only integer values")
        if isinstance(value, Integral):
            checked = int(value)
            try:
                finite_probe = float(checked)
            except OverflowError as exc:
                raise ValueError(f"{name} must contain only integer values") from exc
            if not np.isfinite(finite_probe):
                raise ValueError(f"{name} must contain only integer values")
            normalized.append(checked)
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only integer values") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must contain only integer values")
        normalized.append(int(numeric))
    return normalized


def _normalize_positive_integer_vector(
    name: str,
    values: Any,
    *,
    expected_length: int | None = None,
) -> list[int] | None:
    normalized = _normalize_optional_integer_vector(
        name,
        values,
        expected_length=expected_length,
    )
    if normalized is None:
        return None
    if any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain only positive integer values")
    return normalized


def _normalize_finite_positive_integer_vector(
    name: str,
    values: Any,
    *,
    expected_length: int | None = None,
) -> list[int]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    if expected_length is not None and array.shape[0] != expected_length:
        raise ValueError(f"{name} must match result estimate length")
    normalized: list[int] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain only finite positive integer values")
        if isinstance(value, Integral):
            checked = int(value)
            try:
                finite_probe = float(checked)
            except OverflowError as exc:
                raise ValueError(
                    f"{name} must contain only finite positive integer values"
                ) from exc
            if checked <= 0 or not np.isfinite(finite_probe):
                raise ValueError(f"{name} must contain only finite positive integer values")
            normalized.append(checked)
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite positive integer values") from exc
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
            raise ValueError(f"{name} must contain only finite positive integer values")
        normalized.append(int(numeric))
    return normalized


def _normalize_optional_bool_vector(
    name: str,
    values: Any,
    *,
    expected_length: int | None = None,
) -> list[bool] | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional boolean vector")
    if expected_length is not None and array.shape[0] != expected_length:
        raise ValueError(f"{name} must match result estimate length")
    normalized: list[bool] = []
    for value in array.tolist():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain only boolean values")
        normalized.append(bool(value))
    return normalized


def _validate_event_time_grid_alignment(
    *,
    grid: np.ndarray,
    event_time: list[int] | None,
    event_time_grid: list[int] | None,
) -> None:
    reference = event_time_grid if event_time_grid is not None else event_time
    if reference is None:
        return
    grid_as_event_time = _normalize_optional_integer_vector(
        "grid",
        grid,
        expected_length=len(reference),
    )
    if grid_as_event_time != reference:
        raise ValueError("event_time and event_time_grid must match result grid")
    if event_time is not None and event_time_grid is not None and event_time != event_time_grid:
        raise ValueError("event_time and event_time_grid must match each other")


def _validate_cohort_summary(
    cohort_summary: Any,
    event_time: list[int] | None,
    estimate: np.ndarray,
    std_error: np.ndarray,
) -> list[dict[str, Any]] | None:
    if cohort_summary is None:
        return None
    if not isinstance(cohort_summary, list) or not all(
        isinstance(entry, dict) for entry in cohort_summary
    ):
        raise ValueError("cohort_summary must be a list of mappings")
    normalized_summary = [dict(entry) for entry in cohort_summary]
    if event_time is not None:
        if len(normalized_summary) != len(event_time):
            raise ValueError("cohort_summary must match event_time length")
        for row_index, (expected_event_time, entry) in enumerate(
            zip(event_time, normalized_summary)
        ):
            if "event_time" not in entry:
                raise ValueError("cohort_summary rows must include event_time")
            row_event_time = _normalize_optional_integer_vector(
                "cohort_summary event_time",
                [entry["event_time"]],
                expected_length=1,
            )
            if row_event_time != [expected_event_time]:
                raise ValueError("cohort_summary event_time must match event_time grid")
            entry["event_time"] = row_event_time[0]
            if "timing_groups" in entry:
                row_timing_groups = _normalize_positive_integer_vector(
                    "cohort_summary timing_groups",
                    entry["timing_groups"],
                )
                if row_timing_groups == []:
                    raise ValueError(
                        "cohort_summary timing_groups must contain at least one timing group"
                    )
                entry["timing_groups"] = row_timing_groups
            if "cohort_estimates" in entry:
                cohort_estimates = entry["cohort_estimates"]
                if not isinstance(cohort_estimates, list) or not all(
                    isinstance(cohort, dict) for cohort in cohort_estimates
                ):
                    raise ValueError("cohort_estimates must be a list of mappings")
                normalized_cohorts: list[dict[str, Any]] = []
                for cohort in cohort_estimates:
                    if "timing_group" not in cohort:
                        raise ValueError("cohort_estimates rows must include timing_group")
                    normalized_cohort = dict(cohort)
                    cohort_timing_group = _normalize_positive_integer_vector(
                        "cohort_estimates timing_group",
                        [normalized_cohort["timing_group"]],
                        expected_length=1,
                    )
                    assert cohort_timing_group is not None
                    if (
                        entry.get("timing_groups") is not None
                        and cohort_timing_group[0] not in entry["timing_groups"]
                    ):
                        raise ValueError(
                            "cohort_estimates timing_group must be listed in cohort_summary timing_groups"
                        )
                    normalized_cohort["timing_group"] = cohort_timing_group[0]
                    if "time_period" in normalized_cohort:
                        time_period = _normalize_optional_integer_vector(
                            "cohort_estimates time_period",
                            [normalized_cohort["time_period"]],
                            expected_length=1,
                        )
                        assert time_period is not None
                        normalized_cohort["time_period"] = time_period[0]
                    if "base_period" in normalized_cohort:
                        base_period = _normalize_optional_integer_vector(
                            "cohort_estimates base_period",
                            [normalized_cohort["base_period"]],
                            expected_length=1,
                        )
                        assert base_period is not None
                        normalized_cohort["base_period"] = base_period[0]
                    for count_field in ("comparison_count", "treated_count"):
                        if count_field in normalized_cohort:
                            count = _normalize_finite_positive_integer_vector(
                                f"cohort_estimates {count_field}",
                                [normalized_cohort[count_field]],
                                expected_length=1,
                            )
                            normalized_cohort[count_field] = count[0]
                    if "estimate" in normalized_cohort:
                        cohort_estimate = float(normalized_cohort["estimate"])
                        if not np.isfinite(cohort_estimate):
                            raise ValueError("cohort_estimates estimate must be finite")
                        normalized_cohort["estimate"] = cohort_estimate
                    if "std_error" in normalized_cohort:
                        cohort_std_error = float(normalized_cohort["std_error"])
                        if not np.isfinite(cohort_std_error) or cohort_std_error < 0.0:
                            raise ValueError(
                                "cohort_estimates std_error must be finite and nonnegative"
                            )
                        normalized_cohort["std_error"] = cohort_std_error
                    normalized_cohorts.append(normalized_cohort)
                if any("aggregation_weight" in cohort for cohort in normalized_cohorts):
                    if not all("aggregation_weight" in cohort for cohort in normalized_cohorts):
                        raise ValueError(
                            "cohort_estimates aggregation_weight must be present for every cohort in an event-time cell"
                        )
                    weight_values: list[float] = []
                    for cohort in normalized_cohorts:
                        raw_weight = cohort["aggregation_weight"]
                        if isinstance(raw_weight, (bool, np.bool_)):
                            raise ValueError(
                                "cohort_estimates aggregation_weight must be finite and nonnegative"
                            )
                        aggregation_weight = float(raw_weight)
                        if not np.isfinite(aggregation_weight) or aggregation_weight < 0.0:
                            raise ValueError(
                                "cohort_estimates aggregation_weight must be finite and nonnegative"
                            )
                        cohort["aggregation_weight"] = aggregation_weight
                        weight_values.append(aggregation_weight)
                    if not all("treated_count" in cohort for cohort in normalized_cohorts):
                        raise ValueError(
                            "cohort_estimates aggregation_weight requires treated_count for every cohort in an event-time cell"
                        )
                    weight_total = float(sum(weight_values))
                    if not np.isclose(
                        weight_total,
                        1.0,
                        rtol=0.0,
                        atol=np.finfo(float).eps * max(1.0, len(weight_values), abs(weight_total)),
                    ):
                        raise ValueError(
                            "cohort_estimates aggregation_weight must sum to one within each event-time cell"
                        )
                    total_treated_count = sum(
                        cohort["treated_count"] for cohort in normalized_cohorts
                    )
                    for cohort in normalized_cohorts:
                        treated_share = cohort["treated_count"] / total_treated_count
                        if not np.isclose(
                            cohort["aggregation_weight"],
                            treated_share,
                            rtol=0.0,
                            atol=np.finfo(float).eps
                            * max(
                                1.0,
                                len(normalized_cohorts),
                                abs(float(cohort["aggregation_weight"])),
                                abs(float(treated_share)),
                            ),
                        ):
                            raise ValueError(
                                "cohort_estimates aggregation_weight must equal treated_count share within each event-time cell"
                            )
                    if all("estimate" in cohort for cohort in normalized_cohorts):
                        weighted_estimate = float(
                            sum(
                                cohort["estimate"] * cohort["aggregation_weight"]
                                for cohort in normalized_cohorts
                            )
                        )
                        if not np.isclose(
                            weighted_estimate,
                            estimate[row_index],
                            rtol=0.0,
                            atol=np.finfo(float).eps
                            * max(
                                1.0,
                                abs(weighted_estimate),
                                abs(float(estimate[row_index])),
                            ),
                        ):
                            raise ValueError(
                                "cohort_estimates aggregation_weight values must reconstruct result estimate"
                            )
                entry["cohort_estimates"] = normalized_cohorts
            if "mean_estimate" in entry:
                row_mean = float(entry["mean_estimate"])
                if not np.isfinite(row_mean):
                    raise ValueError("cohort_summary mean_estimate must be finite")
                if not np.isclose(
                    row_mean,
                    estimate[row_index],
                    rtol=0.0,
                    atol=np.finfo(float).eps
                    * max(1.0, abs(row_mean), abs(float(estimate[row_index]))),
                ):
                    raise ValueError("cohort_summary mean_estimate must match result estimate")
                entry["mean_estimate"] = row_mean
            if "std_error" in entry:
                row_std_error = float(entry["std_error"])
                if not np.isfinite(row_std_error) or row_std_error < 0.0:
                    raise ValueError("cohort_summary std_error must be finite and nonnegative")
                if not np.isclose(
                    row_std_error,
                    std_error[row_index],
                    rtol=0.0,
                    atol=np.finfo(float).eps
                    * max(1.0, abs(row_std_error), abs(float(std_error[row_index]))),
                ):
                    raise ValueError("cohort_summary std_error must match result std_error")
                entry["std_error"] = row_std_error
            if "support" in entry:
                if not isinstance(entry["support"], (bool, np.bool_)):
                    raise ValueError("cohort_summary support must be a boolean")
                row_support = bool(entry["support"])
                if "cohort_estimates" in entry and row_support != bool(entry["cohort_estimates"]):
                    raise ValueError("cohort_summary support must match cohort_estimates presence")
                entry["support"] = row_support
    return normalized_summary


def _timing_groups_from_cohort_summary(
    cohort_summary: list[dict[str, Any]] | None,
) -> list[int] | None:
    if cohort_summary is None:
        return None
    timing_groups: set[int] = set()
    for entry in cohort_summary:
        if "timing_groups" in entry:
            timing_groups.update(int(value) for value in entry["timing_groups"])
    if not timing_groups:
        return None
    return sorted(timing_groups)


def _normalize_timing_group_support_metadata(
    timing_group_support: Any,
    *,
    timing_group: list[int] | None,
    cohort_summary: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if timing_group_support is None:
        return None
    if not isinstance(timing_group_support, dict):
        raise ValueError("timing_group_support must be a mapping")
    if "timing_groups" not in timing_group_support:
        raise ValueError("timing_group_support must include timing_groups")
    normalized = dict(timing_group_support)
    timing_groups = _normalize_positive_integer_vector(
        "timing_group_support timing_groups",
        normalized["timing_groups"],
    )
    if timing_groups == []:
        raise ValueError(
            "timing_group_support timing_groups must contain at least one timing group"
        )
    expected_sources = [
        source
        for source in (timing_group, _timing_groups_from_cohort_summary(cohort_summary))
        if source is not None
    ]
    if any(timing_groups != sorted(source) for source in expected_sources):
        raise ValueError("timing_group_support timing_groups must match public timing groups")
    normalized["timing_groups"] = timing_groups
    if "never_treated_group" not in normalized:
        raise ValueError("timing_group_support must include never_treated_group")
    never_treated_group = _normalize_optional_integer_vector(
        "timing_group_support never_treated_group",
        [normalized["never_treated_group"]],
        expected_length=1,
    )
    if never_treated_group != [0]:
        raise ValueError("timing_group_support never_treated_group must be 0")
    normalized["never_treated_group"] = 0
    if "reporting_scale" not in normalized:
        raise ValueError("timing_group_support must include reporting_scale")
    if (
        not isinstance(normalized["reporting_scale"], str)
        or normalized["reporting_scale"] != _TIMING_GROUP_REPORTING_SCALE
    ):
        raise ValueError(
            "timing_group_support reporting_scale must be length of exposure to treatment"
        )
    if "base_period_strategy" not in normalized:
        raise ValueError("timing_group_support must include base_period_strategy")
    if (
        not isinstance(normalized["base_period_strategy"], str)
        or normalized["base_period_strategy"] not in _TIMING_GROUP_BASE_PERIOD_STRATEGIES
    ):
        raise ValueError(
            "timing_group_support base_period_strategy must be fixed, universal, or varying_pre_period"
        )
    return normalized


def _normalize_identification_metadata(
    identification: Any,
    *,
    estimand: str,
) -> dict[str, str] | None:
    if identification is None:
        return None
    if not isinstance(identification, dict):
        raise ValueError("identification metadata must be a mapping")
    expected = _IDENTIFICATION_PAYLOADS.get(estimand)
    if expected is None:
        raise ValueError("identification metadata requires a public contdid estimand")
    normalized = dict(identification)
    for key in expected:
        value = normalized.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"identification metadata {key} must be a non-empty string")
    if normalized != expected:
        raise ValueError(
            "identification metadata must match the checked public estimand interpretation"
        )
    return normalized


def _plot_axis(result: ContDIDResult) -> tuple[str, list[float]]:
    if result.event_time_grid is not None:
        return "event_time", [float(value) for value in result.event_time_grid]
    if result.event_time is not None:
        return "event_time", [float(value) for value in result.event_time]
    return "dose", list(result.grid)


def _plot_support_values(result: ContDIDResult) -> list[bool] | None:
    support = result.metadata.get("support")
    if support is None:
        return None
    if not isinstance(support, list) or len(support) != len(result.estimate):
        raise ValueError("display support must match result estimate length")
    return _checked_display_support_values(support)


def _plot_supported_indices(result: ContDIDResult) -> list[int]:
    support = _plot_support_values(result)
    if support is None:
        return list(range(len(result.estimate)))
    return [index for index, supported in enumerate(support) if supported]


def _plot_y_values(result: ContDIDResult) -> list[float]:
    support = _plot_support_values(result)
    indices = _plot_supported_indices(result)
    if support is not None and not indices:
        return [float(value) for value in result.estimate]
    values = [float(value) for value in result.estimate]
    if result.confidence_interval is not None:
        for index in indices:
            lower, upper = result.confidence_interval[index]
            values.extend([float(lower), float(upper)])
    if result.confidence_band is not None:
        values.extend(float(result.confidence_band["lower"][index]) for index in indices)
        values.extend(float(result.confidence_band["upper"][index]) for index in indices)
    return values


def _plot_bounds(
    result: ContDIDResult,
) -> tuple[tuple[float, float], tuple[float, float]]:
    _, axis_values = _plot_axis(result)
    x_min = min(axis_values)
    x_max = max(axis_values)
    y_values = _plot_y_values(result)
    y_min = min(y_values)
    y_max = max(y_values)

    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    y_span = y_max - y_min
    y_min -= 0.10 * y_span
    y_max += 0.10 * y_span
    if y_min > 0.0:
        y_min = 0.0
    if y_max < 0.0:
        y_max = 0.0
    return (x_min, x_max), (y_min, y_max)


def _plot_scale(
    x_values: list[float],
    y_values: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[tuple[float, float]]:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    plot_width = _PLOT_WIDTH - _PLOT_MARGIN_LEFT - _PLOT_MARGIN_RIGHT
    plot_height = _PLOT_HEIGHT - _PLOT_MARGIN_TOP - _PLOT_MARGIN_BOTTOM

    points: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        x_norm = (float(x_value) - x_min) / (x_max - x_min)
        y_norm = (float(y_value) - y_min) / (y_max - y_min)
        points.append(
            (
                _PLOT_MARGIN_LEFT + x_norm * plot_width,
                _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM - y_norm * plot_height,
            )
        )
    return points


def _supported_line_segments(
    points: list[tuple[float, float]],
    support: list[bool] | None,
) -> list[list[tuple[float, float]]]:
    if support is None:
        return [points] if len(points) > 1 else []
    return [
        [points[index] for index in segment]
        for segment in _supported_index_segments(support, len(points))
        if len(segment) > 1
    ]


def _supported_index_segments(
    support: list[bool] | None,
    length: int,
) -> list[list[int]]:
    if support is None:
        return [list(range(length))] if length > 0 else []
    segments: list[list[int]] = []
    current: list[int] = []
    for index, supported in enumerate(support[:length]):
        if supported:
            current.append(index)
            continue
        if current:
            segments.append(current)
        current = []
    if current:
        segments.append(current)
    return segments


def _plot_font_stack() -> tuple[Any, Any, Any, Any]:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return (
                ImageFont.truetype(candidate, 28),
                ImageFont.truetype(candidate, 18),
                ImageFont.truetype(candidate, 15),
                ImageFont.truetype(candidate, 13),
            )
        except OSError:
            continue
    fallback = ImageFont.load_default()
    return fallback, fallback, fallback, fallback


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _fit_plot_text(
    draw: Any,
    text: str,
    font: Any,
    *,
    max_width: int,
) -> str:
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    ellipsis = "..."
    if _text_size(draw, ellipsis, font)[0] > max_width:
        return ""

    low = 0
    high = len(text)
    best = ellipsis
    while low <= high:
        midpoint = (low + high) // 2
        candidate = text[:midpoint].rstrip() + ellipsis
        if _text_size(draw, candidate, font)[0] <= max_width:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _draw_centered_text(
    draw: Any,
    *,
    x_left: float,
    x_right: float,
    y: float,
    text: str,
    font: Any,
    fill: str,
) -> None:
    width, _ = _text_size(draw, text, font)
    draw.text((x_left + (x_right - x_left - width) / 2, y), text, font=font, fill=fill)


def _format_tick(value: float, *, axis_name: str) -> str:
    if axis_name == "event_time" and float(value).is_integer():
        return str(int(value))
    return _format_table_float(value, digits=2)


def _plot_x_ticks(
    *,
    axis_name: str,
    axis_values: list[float],
    x_bounds: tuple[float, float],
) -> list[float]:
    if axis_name == "event_time":
        event_ticks = sorted({int(value) for value in axis_values})
        if len(event_ticks) <= 9:
            return [float(value) for value in event_ticks]
        x_min, x_max = x_bounds
        integer_ticks = list(range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1))
        if len(integer_ticks) <= 9:
            return [float(value) for value in integer_ticks]
        step = int(np.ceil((integer_ticks[-1] - integer_ticks[0]) / 8.0))
        sparse_ticks = [value for value in integer_ticks if (value - integer_ticks[0]) % step == 0]
        for required in (integer_ticks[0], 0, integer_ticks[-1]):
            if integer_ticks[0] <= required <= integer_ticks[-1]:
                sparse_ticks.append(required)
        return [float(value) for value in sorted(set(sparse_ticks))]
    unique_values = sorted({float(value) for value in axis_values})
    if len(unique_values) <= 6:
        return unique_values
    x_min, x_max = x_bounds
    return [x_min + (x_max - x_min) * idx / 5 for idx in range(6)]


def _format_y_tick(value: float) -> str:
    if np.isclose(value, 0.0, rtol=0.0, atol=1e-12):
        value = 0.0
    absolute = abs(value)
    if absolute >= 100.0:
        formatted = f"{value:.0f}"
    elif absolute >= 10.0:
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    elif absolute >= 1.0:
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if formatted == "-0" else formatted


def _plot_y_ticks(y_bounds: tuple[float, float]) -> list[float]:
    y_min, y_max = y_bounds
    span = y_max - y_min
    if span <= 0.0:
        return [y_min]
    exponent = int(np.floor(np.log10(span / 5.0)))
    candidates: list[tuple[int, float, list[float]]] = []
    for power in range(exponent - 1, exponent + 3):
        scale = 10.0**power
        for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
            step = multiplier * scale
            first = int(np.ceil((y_min - np.finfo(float).eps) / step))
            last = int(np.floor((y_max + np.finfo(float).eps) / step))
            ticks = [index * step for index in range(first, last + 1)]
            if 3 <= len(ticks) <= 7:
                score = abs(5 - len(ticks))
                candidates.append((score, step, ticks))
    if not candidates:
        return [y_min + span * idx / 5 for idx in range(6)]
    _, _, ticks = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    return [0.0 if np.isclose(value, 0.0, rtol=0.0, atol=1e-12) else value for value in ticks]


def _draw_plot_grid(
    draw: Any,
    *,
    axis_values: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    axis_name: str,
    tick_font: Any,
) -> None:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    plot_width = _PLOT_WIDTH - _PLOT_MARGIN_LEFT - _PLOT_MARGIN_RIGHT
    plot_height = _PLOT_HEIGHT - _PLOT_MARGIN_TOP - _PLOT_MARGIN_BOTTOM
    x_ticks = _plot_x_ticks(
        axis_name=axis_name,
        axis_values=axis_values,
        x_bounds=x_bounds,
    )
    y_ticks = _plot_y_ticks(y_bounds)

    for y_tick in y_ticks:
        y_px = (
            _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM - (y_tick - y_min) / (y_max - y_min) * plot_height
        )
        draw.line(
            [(_PLOT_MARGIN_LEFT, y_px), (_PLOT_WIDTH - _PLOT_MARGIN_RIGHT, y_px)],
            fill=_PLOT_GRID,
            width=1,
        )

        y_label = _format_y_tick(y_tick)
        y_width, y_height = _text_size(draw, y_label, tick_font)
        draw.text(
            (
                max(12, _PLOT_MARGIN_LEFT - y_width - 18),
                min(
                    max(_PLOT_MARGIN_TOP, y_px - y_height / 2),
                    _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM - y_height,
                ),
            ),
            y_label,
            fill=_PLOT_MUTED_TEXT,
            font=tick_font,
        )
    for x_tick in x_ticks:
        x_px = _PLOT_MARGIN_LEFT + (x_tick - x_min) / (x_max - x_min) * plot_width
        draw.line(
            [(x_px, _PLOT_MARGIN_TOP), (x_px, _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM)],
            fill=_PLOT_GRID,
            width=1,
        )
        x_label = _format_tick(x_tick, axis_name=axis_name)
        x_width, _ = _text_size(draw, x_label, tick_font)
        draw.text(
            (
                min(
                    max(_PLOT_MARGIN_LEFT, x_px - x_width / 2),
                    _PLOT_WIDTH - _PLOT_MARGIN_RIGHT - x_width,
                ),
                _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM + 14,
            ),
            x_label,
            fill=_PLOT_MUTED_TEXT,
            font=tick_font,
        )


def _draw_plot_band(
    draw: Any,
    result: ContDIDResult,
    axis_values: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    support: list[bool] | None,
) -> bool:
    if result.confidence_band is None:
        return False
    if _plot_band_duplicates_intervals(result):
        return False
    upper_points = _plot_scale(
        axis_values,
        [float(value) for value in result.confidence_band["upper"]],
        x_bounds,
        y_bounds,
    )
    lower_points = _plot_scale(
        axis_values,
        [float(value) for value in result.confidence_band["lower"]],
        x_bounds,
        y_bounds,
    )
    drew_band = False
    for segment in _supported_index_segments(support, len(axis_values)):
        if len(segment) == 1:
            index = segment[0]
            x_px = upper_points[index][0]
            y_upper = upper_points[index][1]
            y_lower = lower_points[index][1]
            draw.rectangle(
                [
                    (x_px - 10, min(y_upper, y_lower)),
                    (x_px + 10, max(y_upper, y_lower)),
                ],
                fill=_PLOT_BAND,
                outline=_PLOT_BAND_OUTLINE,
                width=2,
            )
            drew_band = True
            continue
        if len(segment) < 2:
            continue
        segment_upper = [upper_points[index] for index in segment]
        segment_lower = [lower_points[index] for index in segment]
        draw.polygon(segment_upper + list(reversed(segment_lower)), fill=_PLOT_BAND)
        draw.line(segment_upper, fill=_PLOT_BAND_OUTLINE, width=2)
        draw.line(segment_lower, fill=_PLOT_BAND_OUTLINE, width=2)
        drew_band = True
    return drew_band


def _draw_plot_intervals(
    draw: Any,
    result: ContDIDResult,
    axis_values: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    support: list[bool] | None,
) -> bool:
    if result.confidence_interval is None:
        return False
    interval = np.asarray(result.confidence_interval, dtype=float)
    lower_points = _plot_scale(
        axis_values,
        interval[:, 0].tolist(),  # type: ignore[arg-type]
        x_bounds,
        y_bounds,
    )
    upper_points = _plot_scale(
        axis_values,
        interval[:, 1].tolist(),  # type: ignore[arg-type]
        x_bounds,
        y_bounds,
    )
    drew_interval = False
    for index, (lower, upper) in enumerate(zip(lower_points, upper_points)):
        if support is not None and not support[index]:
            continue
        x_px = lower[0]
        draw.line([(x_px, lower[1]), (x_px, upper[1])], fill=_PLOT_INTERVAL, width=2)
        draw.line([(x_px - 5, lower[1]), (x_px + 5, lower[1])], fill=_PLOT_INTERVAL, width=2)
        draw.line([(x_px - 5, upper[1]), (x_px + 5, upper[1])], fill=_PLOT_INTERVAL, width=2)
        drew_interval = True
    return drew_interval


def _plot_legend_size(
    *,
    has_band: bool,
    has_interval: bool,
    support: list[bool] | None,
) -> tuple[int, int]:
    row_gap = 27
    row_count = 1 + int(has_band) + int(has_interval) + int(support is not None)
    return 272, 22 + row_gap * row_count


def _plot_unsupported_legend_label(support: list[bool]) -> str:
    unsupported = sum(1 for supported in support if not supported)
    return f"No local support ({unsupported}/{len(support)})"


def _legend_collision_count(
    bounds: tuple[float, float, float, float],
    data_points: list[tuple[float, float]],
) -> int:
    left, top, right, bottom = bounds
    padding = 10
    return sum(
        left - padding <= x <= right + padding and top - padding <= y <= bottom + padding
        for x, y in data_points
    )


def _plot_legend_bounds(
    *,
    has_band: bool,
    has_interval: bool,
    support: list[bool] | None,
    data_points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    width, height = _plot_legend_size(
        has_band=has_band,
        has_interval=has_interval,
        support=support,
    )
    plot_left = _PLOT_MARGIN_LEFT
    plot_top = _PLOT_MARGIN_TOP
    plot_right = _PLOT_WIDTH - _PLOT_MARGIN_RIGHT
    plot_bottom = _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM
    inset = 18
    candidates = [
        (plot_left + inset, plot_top + inset),
        (plot_right - width - inset, plot_top + inset),
        (plot_left + inset, plot_bottom - height - inset),
        (plot_right - width - inset, plot_bottom - height - inset),
    ]
    rectangles = [(left, top, left + width, top + height) for left, top in candidates]
    return min(
        rectangles,
        key=lambda bounds: _legend_collision_count(bounds, data_points),
    )


def _plot_legend_data_points(
    result: ContDIDResult,
    axis_values: list[float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    support: list[bool] | None,
) -> list[tuple[float, float]]:
    data_points = _plot_scale(axis_values, result.estimate, x_bounds, y_bounds)
    supported_indices = (
        list(range(len(axis_values)))
        if support is None
        else [index for index, supported in enumerate(support) if supported]
    )
    if result.confidence_interval is not None:
        interval = np.asarray(result.confidence_interval, dtype=float)
        if supported_indices:
            supported_axis = [axis_values[index] for index in supported_indices]
            data_points.extend(
                _plot_scale(
                    supported_axis,
                    [float(interval[index, 0]) for index in supported_indices],
                    x_bounds,
                    y_bounds,
                )
            )
            data_points.extend(
                _plot_scale(
                    supported_axis,
                    [float(interval[index, 1]) for index in supported_indices],
                    x_bounds,
                    y_bounds,
                )
            )
    if result.confidence_band is not None:
        if supported_indices:
            supported_axis = [axis_values[index] for index in supported_indices]
            data_points.extend(
                _plot_scale(
                    supported_axis,
                    [float(result.confidence_band["lower"][index]) for index in supported_indices],
                    x_bounds,
                    y_bounds,
                )
            )
            data_points.extend(
                _plot_scale(
                    supported_axis,
                    [float(result.confidence_band["upper"][index]) for index in supported_indices],
                    x_bounds,
                    y_bounds,
                )
            )
    return data_points


def _draw_reference_lines(
    draw: Any,
    *,
    axis_name: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    y_min, y_max = y_bounds
    if y_min <= 0.0 <= y_max:
        zero_y = _plot_scale([x_bounds[0]], [0.0], x_bounds, y_bounds)[0][1]
        draw.line(
            [(_PLOT_MARGIN_LEFT, zero_y), (_PLOT_WIDTH - _PLOT_MARGIN_RIGHT, zero_y)],
            fill=_PLOT_ZERO,
            width=2,
        )
    if axis_name == "event_time" and x_bounds[0] <= 0.0 <= x_bounds[1]:
        zero_x = _plot_scale([0.0], [y_bounds[0]], x_bounds, y_bounds)[0][0]
        draw.line(
            [(zero_x, _PLOT_MARGIN_TOP), (zero_x, _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM)],
            fill=_PLOT_EVENT_ZERO,
            width=2,
        )


def _draw_plot_legend(
    draw: Any,
    *,
    legend_font: Any,
    band_label: str,
    has_band: bool,
    has_interval: bool,
    support: list[bool] | None,
    data_points: list[tuple[float, float]],
) -> None:
    left, top, right, bottom = _plot_legend_bounds(
        has_band=has_band,
        has_interval=has_interval,
        support=support,
        data_points=data_points,
    )
    row_gap = 27
    label_x = left + 58
    label_max_width = max(0, int(right - label_x - 14))
    draw.rectangle(
        [(left, top), (right, bottom)],
        fill="#FFFFFF",
        outline="#CBD5E1",
        width=1,
    )
    y = top + 15
    if has_band:
        draw.rectangle([(left + 16, y + 5), (left + 46, y + 17)], fill=_PLOT_BAND)
        draw.text(
            (label_x, y),
            _fit_plot_text(draw, band_label, legend_font, max_width=label_max_width),
            fill=_PLOT_TEXT,
            font=legend_font,
        )
        y += row_gap
    if has_interval:
        draw.line([(left + 31, y + 2), (left + 31, y + 21)], fill=_PLOT_INTERVAL, width=2)
        draw.line([(left + 25, y + 2), (left + 37, y + 2)], fill=_PLOT_INTERVAL, width=2)
        draw.line([(left + 25, y + 21), (left + 37, y + 21)], fill=_PLOT_INTERVAL, width=2)
        draw.text(
            (label_x, y),
            _fit_plot_text(
                draw,
                "Pointwise CI",
                legend_font,
                max_width=label_max_width,
            ),
            fill=_PLOT_TEXT,
            font=legend_font,
        )
        y += row_gap
    draw.line([(left + 16, y + 12), (left + 46, y + 12)], fill=_PLOT_LINE, width=4)
    draw.ellipse([(left + 27, y + 7), (left + 35, y + 15)], fill=_PLOT_POINT)
    draw.text(
        (label_x, y),
        _fit_plot_text(draw, "Estimate", legend_font, max_width=label_max_width),
        fill=_PLOT_TEXT,
        font=legend_font,
    )
    y += row_gap
    if support is not None:
        draw.ellipse(
            [(left + 25, y + 6), (left + 37, y + 18)],
            fill="#FFFFFF",
            outline=_PLOT_UNSUPPORTED_POINT,
            width=2,
        )
        draw.text(
            (label_x, y),
            _fit_plot_text(
                draw,
                _plot_unsupported_legend_label(support),
                legend_font,
                max_width=label_max_width,
            ),
            fill=_PLOT_TEXT,
            font=legend_font,
        )


def _draw_rotated_y_label(image: Any, *, font: Any, label: str) -> None:
    from PIL import Image, ImageDraw

    text_box = font.getbbox(label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_image = Image.new(
        "RGBA",
        (text_width + 12, text_height + 12),
        (0, 0, 0, 0),
    )
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text(
        (6 - text_box[0], 6 - text_box[1]),
        label,
        fill=_PLOT_TEXT,
        font=font,
    )
    rotated = label_image.rotate(90, expand=True)
    y_position = int(
        _PLOT_MARGIN_TOP
        + (_PLOT_HEIGHT - _PLOT_MARGIN_TOP - _PLOT_MARGIN_BOTTOM - rotated.height) / 2
    )
    image.paste(rotated, (24, y_position), rotated)


@dataclass(slots=True)
class ContDIDResult:
    """Result container for continuous-dose DiD estimators.

    Holds point estimates, standard errors, confidence intervals/bands,
    and metadata from dose-response or event-study estimation.

    Attributes:
        estimand: Name of the estimated quantity (e.g., "ATT(d)", "ACRT(d)").
        grid: Evaluation grid points (dose values or event-time indices).
        estimate: Point estimates at each grid point.
        std_error: Standard errors at each grid point.
        critical_value: Critical value used for confidence bands.
        confidence_interval: Pointwise confidence intervals [[lower, upper], ...].
        confidence_band: Simultaneous confidence band with lower/upper/critical_value.
        timing_group: Timing groups for event-study cells.
        event_time: Event-time indices (alternative to grid for event studies).
        event_time_grid: Event-time grid for event-study results.
        cohort_summary: Per-cohort summary statistics.
        metadata: Additional estimation metadata and diagnostics.
    """

    estimand: str
    grid: list[float]
    estimate: list[float]
    std_error: list[float]
    critical_value: float | None = None
    confidence_interval: list[list[float]] | None = None
    confidence_band: dict[str, Any] | None = None
    timing_group: list[int] | None = None
    event_time: list[int] | None = None
    event_time_grid: list[int] | None = None
    cohort_summary: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _hook_outputs: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def hook_outputs(self) -> dict[str, Any]:
        """获取后处理钩子的输出结果。

        Returns
        -------
        dict[str, Any]
            各钩子的输出副本，键为钩子名称。
        """
        return dict(self._hook_outputs)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        if _contains_bool_values(self.estimate):
            raise ValueError("estimate must contain only finite non-boolean values")
        if _contains_bool_values(self.std_error):
            raise ValueError("std_error must contain only finite non-boolean nonnegative values")
        if _contains_bool_values(self.grid):
            raise ValueError("grid must contain only finite non-boolean values")
        try:
            estimate = np.asarray(self.estimate, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("estimate must contain only finite non-boolean values") from exc
        try:
            std_error = np.asarray(self.std_error, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "std_error must contain only finite non-boolean nonnegative values"
            ) from exc
        try:
            grid = np.asarray(self.grid, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("grid must contain only finite non-boolean values") from exc
        if estimate.ndim != 1 or std_error.ndim != 1 or grid.ndim != 1:
            raise ValueError("grid, estimate, and std_error must be one-dimensional")
        if estimate.size == 0:
            raise ValueError("estimate and std_error must contain at least one value")
        if estimate.shape != std_error.shape:
            raise ValueError("estimate and std_error must have the same shape")
        if grid.shape != estimate.shape:
            raise ValueError("grid must have the same shape as estimate")
        if not np.isfinite(estimate).all():
            raise ValueError("estimate must contain only finite non-boolean values")
        if not np.isfinite(std_error).all() or np.any(std_error < 0.0):
            raise ValueError("std_error must contain only finite non-boolean nonnegative values")
        if not np.isfinite(grid).all():
            raise ValueError("grid must contain only finite non-boolean values")

        self.grid = [float(value) for value in grid.tolist()]  # type: ignore[arg-type]
        self.estimate = [float(value) for value in estimate.tolist()]  # type: ignore[arg-type]
        self.std_error = [float(value) for value in std_error.tolist()]  # type: ignore[arg-type]
        self.event_time = _normalize_optional_integer_vector(
            "event_time",
            self.event_time,
            expected_length=estimate.shape[0] if self.event_time is not None else None,
        )
        self.event_time_grid = _normalize_optional_integer_vector(
            "event_time_grid",
            self.event_time_grid,
            expected_length=(estimate.shape[0] if self.event_time_grid is not None else None),
        )
        has_eventstudy_estimand = "event_time" in self.estimand
        has_eventstudy_axis = self.event_time is not None or self.event_time_grid is not None
        if has_eventstudy_estimand and not has_eventstudy_axis:
            raise ValueError("event-study result requires event_time or event_time_grid")
        if not has_eventstudy_estimand and has_eventstudy_axis:
            raise ValueError("event-time fields require event-study estimand")
        eventstudy_metadata_keys = {
            "event_time",
            "event_time_grid",
            "timing_group",
            "cohort_summary",
            "support",
            "timing_group_support",
        }
        if not has_eventstudy_estimand:
            carried_keys = sorted(
                key
                for key in eventstudy_metadata_keys
                if key in self.metadata and self.metadata[key] is not None
            )
            if carried_keys:
                joined = ", ".join(carried_keys)
                raise ValueError(
                    f"event-study metadata fields require event-study estimand: {joined}"
                )
        _validate_event_time_grid_alignment(
            grid=grid,
            event_time=self.event_time,
            event_time_grid=self.event_time_grid,
        )
        self.timing_group = _normalize_positive_integer_vector(
            "timing_group",
            self.timing_group,
        )
        if self.timing_group is not None and not has_eventstudy_estimand:
            raise ValueError("timing_group requires event-study estimand")
        if (
            self.cohort_summary is not None
            and self.event_time_grid is None
            and self.event_time is None
        ):
            raise ValueError("cohort_summary requires event_time or event_time_grid")
        self.cohort_summary = _validate_cohort_summary(
            self.cohort_summary,
            self.event_time_grid if self.event_time_grid is not None else self.event_time,
            estimate,
            std_error,
        )
        if not has_eventstudy_estimand and self.metadata.get("support") is not None:
            raise ValueError("support metadata requires event-study estimand")
        row_support: list[bool] | None = None
        if self.cohort_summary is not None:
            if not all("support" in entry for entry in self.cohort_summary):
                raise ValueError("cohort_summary support must be present for every event-time row")
            row_support = [bool(entry["support"]) for entry in self.cohort_summary]
        metadata_support = _normalize_optional_bool_vector(
            "metadata support",
            self.metadata.get("support"),
            expected_length=estimate.shape[0]
            if self.metadata.get("support") is not None
            else None,
        )
        if (
            metadata_support is not None
            and row_support is not None
            and metadata_support != row_support
        ):
            raise ValueError("metadata support must match cohort_summary support")
        if metadata_support is None and row_support is not None:
            metadata_support = row_support
        if not has_eventstudy_estimand and self.metadata.get("timing_group_support") is not None:
            raise ValueError("timing_group_support metadata requires event-study estimand")
        timing_group_support = _normalize_timing_group_support_metadata(
            self.metadata.get("timing_group_support"),
            timing_group=self.timing_group,
            cohort_summary=self.cohort_summary,
        )
        identification = _normalize_identification_metadata(
            self.metadata.get("identification"),
            estimand=self.estimand,
        )

        if self.critical_value is None and self.metadata.get("critical_value") is not None:
            metadata_critical_value = self.metadata["critical_value"]
            if isinstance(metadata_critical_value, (bool, np.bool_)):
                raise ValueError("critical_value must be finite and nonnegative")
            try:
                self.critical_value = float(metadata_critical_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("critical_value must be finite and nonnegative") from exc
        if self.critical_value is not None:
            if isinstance(self.critical_value, (bool, np.bool_)):
                raise ValueError("critical_value must be finite and nonnegative")
            try:
                self.critical_value = float(self.critical_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("critical_value must be finite and nonnegative") from exc
            if not np.isfinite(self.critical_value) or self.critical_value < 0.0:
                raise ValueError("critical_value must be finite and nonnegative")
        if (
            self.confidence_interval is None
            and self.metadata.get("confidence_interval") is not None
        ):
            self.confidence_interval = self.metadata["confidence_interval"]
        if self.confidence_band is None and self.metadata.get("confidence_band") is not None:
            self.confidence_band = self.metadata["confidence_band"]

        self.confidence_interval = _normalize_confidence_interval(
            self.confidence_interval,
            estimate,
        )
        self.confidence_band = _normalize_confidence_band(
            self.confidence_band,
            estimate,
            self.critical_value,
        )
        if self.critical_value is None and self.confidence_band is not None:
            self.critical_value = float(self.confidence_band["critical_value"])

        metadata_defaults = {
            "estimand": self.estimand,
            "grid": self.grid,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "bootstrap_type": self.metadata.get("bootstrap_type"),
            "critical_value": self.critical_value,
            "confidence_interval": self.confidence_interval,
            "confidence_band": self.confidence_band,
            "identification": identification,
        }
        if has_eventstudy_estimand:
            metadata_defaults.update(
                {
                    "event_time": self.event_time,
                    "event_time_grid": self.event_time_grid,
                    "timing_group": self.timing_group,
                    "cohort_summary": self.cohort_summary,
                    "support": metadata_support,
                    "timing_group_support": timing_group_support,
                }
            )
        else:
            for key in eventstudy_metadata_keys:
                self.metadata.pop(key, None)
        for key, value in metadata_defaults.items():
            self.metadata[key] = value

    @property
    def has_inference(self) -> bool:
        """Whether this result has inference (confidence intervals) attached."""
        return bool(self.std_error) and any(se > 0 for se in self.std_error)

    def is_significant(self, index: int | None = None) -> bool | list[bool]:
        """Check if estimates are significantly different from zero.

        Uses the attached confidence intervals: significant if CI excludes 0.
        If no confidence interval is available, falls back to the Wald test
        using estimate / std_error against the normal critical value at the
        stored significance level.

        Parameters
        ----------
        index : int or None
            If provided, check significance at a single grid index.
            If None, return a list of booleans for all grid points.

        Returns
        -------
        bool or list of bool
            Significance indicator(s).
        """
        if self.confidence_interval is not None:
            # CI-based: significant if interval does not contain zero
            if index is not None:
                lower, upper = self.confidence_interval[index]
                return lower > 0.0 or upper < 0.0
            return [lower > 0.0 or upper < 0.0 for lower, upper in self.confidence_interval]
        # Fallback: Wald-type test
        import scipy.stats as stats

        alp = self.metadata.get("alp", 0.05)
        z_crit = stats.norm.ppf(1.0 - alp / 2.0)

        def _wald_significant(est: float, se: float) -> bool:
            if se <= 0.0:
                return False
            return abs(est / se) > z_crit

        if index is not None:
            return _wald_significant(self.estimate[index], self.std_error[index])
        return [_wald_significant(est, se) for est, se in zip(self.estimate, self.std_error)]

    def __repr__(self) -> str:
        return f"ContDIDResult(estimand={self.estimand!r}, grid_size={len(self.grid)})"

    def __str__(self) -> str:
        return self.summary()

    @staticmethod
    def _significance_code(p_value: float) -> str:
        """Return significance star code for a p-value."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "."
        return ""

    def summary(self, alpha: float = 0.05, max_rows: int | None = None) -> str:
        """Generate a formatted summary table of estimation results.

        Mimics R's summary() output with estimates, standard errors,
        confidence intervals, and significance codes.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for confidence intervals and stars.
        max_rows : int or None
            Maximum number of grid rows to display. None shows all.

        Returns
        -------
        str
            Formatted summary table ready for printing.
        """
        from scipy.stats import norm

        lines: list[str] = []
        meta = self.metadata

        # === Header ===
        lines.append("Continuous DiD Estimation Results")
        lines.append("=" * 50)

        # Estimand
        lines.append(f"Estimand: {self.estimand}")

        # Method / basis info
        basis = meta.get("basis", {})
        if basis:
            basis_type = basis.get("type", "parametric")
            degree = basis.get("degree", "?")
            knots = basis.get("num_knots", 0)
            lines.append(f"Method: {basis_type} (degree={degree}, knots={knots})")
        else:
            inference = meta.get("inference", "parametric")
            lines.append(f"Method: {inference}")

        # Control group
        control = meta.get("control_group", meta.get("comparison_group"))
        if control:
            lines.append(f"Control group: {control}")

        # Observations
        n_treated = meta.get("treated_count")
        n_untreated = meta.get("untreated_count")
        if n_treated is not None and n_untreated is not None:
            n_total = n_treated + n_untreated
            lines.append(f"Observations: {n_total} ({n_treated} treated, {n_untreated} control)")
        elif n_treated is not None:
            lines.append(f"N (treated): {n_treated}")

        # Clusters
        n_clusters = meta.get("n_clusters")
        if n_clusters is not None:
            lines.append(f"Clusters: {n_clusters}")

        lines.append("")

        # === Body: Dose-Response or Event-Study Table ===
        is_eventstudy = self.event_time_grid is not None or self.event_time is not None
        if is_eventstudy:
            axis_label = "event_time"
            axis_values: list[float] | list[int] = (
                self.event_time_grid if self.event_time_grid is not None else self.event_time  # type: ignore[assignment]
            )
            lines.append("Event-Study Estimates:")
        else:
            axis_label = "dose"
            axis_values = self.grid
            lines.append("Dose-Response Curve:")

        assert axis_values is not None  # guaranteed by branch logic above

        # z critical value for CI
        z_crit = norm.ppf(1.0 - alpha / 2.0)

        # Determine which rows to show
        n_rows = len(axis_values)
        if max_rows is not None and n_rows > max_rows:
            indices = np.round(np.linspace(0, n_rows - 1, max_rows)).astype(int).tolist()
            truncated = True
        else:
            indices = list(range(n_rows))
            truncated = False

        # Table header
        header = f"  {axis_label:>10}  {'estimate':>10}  {'std.error':>10}  {'[' + str(int((1 - alpha) * 100)) + '% CI]':>20}  {'sig':>4}"
        lines.append(header)

        # Table rows
        for i in indices:
            est = self.estimate[i]
            se = self.std_error[i]

            # Confidence interval
            if self.confidence_interval is not None and i < len(self.confidence_interval):
                ci_lo, ci_hi = self.confidence_interval[i]
            else:
                ci_lo = est - z_crit * se
                ci_hi = est + z_crit * se

            # p-value (two-sided Wald)
            if se > 0:
                z_stat = abs(est / se)
                p_value = 2.0 * (1.0 - norm.cdf(z_stat))
            else:
                p_value = 1.0

            sig = self._significance_code(p_value)

            # Format axis value
            if is_eventstudy:
                axis_str = f"{int(axis_values[i]):>10d}"
            else:
                axis_str = f"{axis_values[i]:>10.3f}"

            ci_str = f"[{ci_lo:>7.3f}, {ci_hi:>7.3f}]"
            row_str = f"  {axis_str}  {est:>10.3f}  {se:>10.3f}    {ci_str:>20}  {sig:>4}"
            lines.append(row_str)

        if truncated:
            lines.append(f"  ... ({max_rows} of {n_rows} points shown)")

        lines.append("---")
        lines.append("Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1")

        # === Optional: overall summary statistics ===
        overall_att = meta.get("overall_att")
        overall_acrt = meta.get("overall_acrt")
        if overall_att is not None:
            lines.append("")
            lines.append(f"Overall ATT: {overall_att:.4f}")
        if overall_acrt is not None:
            lines.append(f"Overall ACRT: {overall_acrt:.4f}")

        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return a row-oriented DataFrame with the complete checked payload."""

        axis_name = "event_time" if self.event_time_grid is not None else "dose"
        axis_values: list[int] | list[float]
        if self.event_time_grid is not None:
            axis_values = self.event_time_grid
        elif self.event_time is not None:
            axis_values = self.event_time
            axis_name = "event_time"
        else:
            axis_values = self.grid

        columns: dict[str, Any] = {
            axis_name: axis_values,
            "estimate": self.estimate,
            "std_error": self.std_error,
        }
        if self.confidence_interval is not None:
            interval = np.asarray(self.confidence_interval, dtype=float)
            columns["ci_lower"] = interval[:, 0].tolist()
            columns["ci_upper"] = interval[:, 1].tolist()
        if self.confidence_band is not None:
            columns["band_lower"] = list(self.confidence_band["lower"])
            columns["band_upper"] = list(self.confidence_band["upper"])
        support = self.metadata.get("support")
        if support is not None:
            if not isinstance(support, list) or len(support) != len(axis_values):
                raise ValueError("display support must match result estimate length")
            columns["support"] = _checked_display_support_values(support)
        return pd.DataFrame(columns)

    def to_markdown(
        self,
        *,
        include_caption: bool = False,
        digits: int = 6,
        max_rows: int | None = None,
    ) -> str:
        """Return a compact Markdown table for release-facing reports."""

        include_caption = _validate_include_caption(include_caption)
        digits = _validate_table_digits(digits)
        max_rows = _validate_markdown_max_rows(max_rows)
        frame = self.to_frame()
        axis_name = "event_time" if "event_time" in frame.columns else "dose"
        axis_header = "Event time" if axis_name == "event_time" else "Dose"
        band_header, missing_band_label = _confidence_band_display_label(self.metadata)
        headers = [axis_header, "Estimate", "Std. error", "Pointwise CI", band_header]
        alignments = ["---:", "---:", "---:", "---", "---"]
        support_values: list[bool] | None = None
        if "support" in frame.columns:
            support_values = _checked_display_support_values(frame["support"].tolist())
            headers.append("Support")
            alignments.append("---")

        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(alignments) + " |",
        ]
        for row in _markdown_display_rows(frame, max_rows=max_rows):
            if isinstance(row, dict) and row.get("kind") == "omitted":
                values = [_format_markdown_omitted_marker(row, digits=digits)]
                values.extend("..." for _ in headers[1:])
                rows.append("| " + " | ".join(values) + " |")
                continue
            if axis_name == "event_time":
                axis_value = str(int(row[axis_name]))
            else:
                axis_value = _format_table_float(row[axis_name], digits=digits)
            supported_row = (
                True
                if support_values is None
                else _checked_display_support_values([row["support"]])[0]
            )
            interval = (
                _missing_table_cell("local support")
                if not supported_row
                else _format_table_interval(row["ci_lower"], row["ci_upper"], digits=digits)
                if "ci_lower" in frame.columns and "ci_upper" in frame.columns
                else _missing_table_cell("pointwise CI")
            )
            band = (
                _missing_table_cell("local support")
                if not supported_row
                else "same as pointwise CI"
                if _pointwise_band_duplicates_interval(row, metadata=self.metadata)
                else _format_table_interval(
                    row["band_lower"],
                    row["band_upper"],
                    digits=digits,
                )
                if "band_lower" in frame.columns and "band_upper" in frame.columns
                else _missing_table_cell(missing_band_label)
            )
            values = [
                axis_value,
                _format_table_float(row["estimate"], digits=digits),
                _format_table_float(row["std_error"], digits=digits),
                interval,
                band,
            ]
            if support_values is not None:
                values.append("yes" if supported_row else "no")
            rows.append("| " + " | ".join(values) + " |")
        table = "\n".join(rows)
        if not include_caption:
            return table

        caption = f"ContDIDResult: {self.estimand}, {len(self.estimate)} rows, {axis_name} axis"
        if max_rows is not None and len(frame) > max_rows:
            caption += f", showing {max_rows}/{len(frame)} rows"
        if self.critical_value is not None:
            caption += (
                f", critical value {_format_table_float(self.critical_value, digits=digits)}"
            )
        if (
            axis_name == "event_time"
            and self.confidence_band is not None
            and self.metadata.get("inference_covariance") == "full_event_time_covariance"
        ):
            caption += ", full event-time covariance band"
        if support_values is not None:
            supported = sum(support_values)
            caption += f", support {supported}/{len(frame)} rows"
        return f"{caption}.\n\n{table}"

    def save_plot(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> Path:
        """Render the result curve and exported uncertainty payload to a PNG file."""

        from PIL import Image, ImageDraw

        if not isinstance(path, (str, Path)):
            raise ValueError("path must be a string or pathlib.Path")
        output = Path(path)
        if output.suffix.lower() != ".png":
            raise ValueError("path must end with .png")

        axis_name, axis_values = _plot_axis(self)
        x_bounds, y_bounds = _plot_bounds(self)
        band_label, _ = _confidence_band_display_label(self.metadata)
        support_values = _plot_support_values(self)
        unsupported_legend_values = (
            support_values if support_values is not None and not all(support_values) else None
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (_PLOT_WIDTH, _PLOT_HEIGHT), _PLOT_BACKGROUND)
        draw = ImageDraw.Draw(image)
        title_font, label_font, tick_font, legend_font = _plot_font_stack()

        plot_left = _PLOT_MARGIN_LEFT
        plot_top = _PLOT_MARGIN_TOP
        plot_right = _PLOT_WIDTH - _PLOT_MARGIN_RIGHT
        plot_bottom = _PLOT_HEIGHT - _PLOT_MARGIN_BOTTOM
        draw.rectangle(
            [(plot_left, plot_top), (plot_right, plot_bottom)],
            fill=_PLOT_PANEL,
            outline="#CBD5E1",
            width=2,
        )

        axis_title = "event time" if axis_name == "event_time" else "dose"
        plot_title = title or f"{self.estimand} by {axis_title}"
        if subtitle is None:
            subtitle_parts = [f"{len(self.estimate)} rows"]
            if self.critical_value is not None:
                subtitle_parts.append(f"critical value {self.critical_value:.3f}")
            if support_values is not None:
                subtitle_parts.append(f"support {sum(support_values)}/{len(support_values)}")
            subtitle = "; ".join(subtitle_parts)
        header_width = int(plot_right - plot_left)
        plot_title = _fit_plot_text(
            draw,
            plot_title,
            title_font,
            max_width=header_width,
        )
        subtitle = _fit_plot_text(
            draw,
            subtitle,
            legend_font,
            max_width=header_width,
        )
        draw.text((plot_left, 24), plot_title, fill=_PLOT_TEXT, font=title_font)
        draw.text((plot_left, 60), subtitle, fill=_PLOT_MUTED_TEXT, font=legend_font)

        _draw_plot_grid(
            draw,
            axis_values=axis_values,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            axis_name=axis_name,
            tick_font=tick_font,
        )
        has_band = _draw_plot_band(
            draw,
            self,
            axis_values,
            x_bounds,
            y_bounds,
            support_values,
        )
        has_interval = _draw_plot_intervals(
            draw,
            self,
            axis_values,
            x_bounds,
            y_bounds,
            support_values,
        )
        _draw_reference_lines(
            draw,
            axis_name=axis_name,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=_PLOT_AXIS, width=2)
        draw.line(
            [(plot_left, plot_bottom), (plot_right, plot_bottom)],
            fill=_PLOT_AXIS,
            width=2,
        )

        points = _plot_scale(axis_values, self.estimate, x_bounds, y_bounds)
        for segment in _supported_line_segments(points, support_values):
            draw.line(segment, fill=_PLOT_LINE, width=5)
        marker_stride = max(1, len(points) // 24)
        for idx, (x_px, y_px) in enumerate(points):
            supported = support_values is None or support_values[idx]
            always_mark = idx == 0 or idx == len(points) - 1
            if supported and not always_mark and idx % marker_stride != 0:
                continue
            if supported:
                draw.ellipse(
                    [(x_px - 5, y_px - 5), (x_px + 5, y_px + 5)],
                    fill=_PLOT_POINT,
                    outline="#FFFFFF",
                    width=2,
                )
            else:
                draw.ellipse(
                    [(x_px - 6, y_px - 6), (x_px + 6, y_px + 6)],
                    fill=_PLOT_PANEL,
                    outline=_PLOT_UNSUPPORTED_POINT,
                    width=2,
                )

        x_axis_label = "Event time" if axis_name == "event_time" else "Dose"
        _draw_centered_text(
            draw,
            x_left=plot_left,
            x_right=plot_right,
            y=_PLOT_HEIGHT - 44,
            text=x_axis_label,
            font=label_font,
            fill=_PLOT_TEXT,
        )
        y_axis_label = _fit_plot_text(
            draw,
            self.estimand,
            label_font,
            max_width=int(plot_bottom - plot_top),
        )
        _draw_rotated_y_label(image, font=label_font, label=y_axis_label)
        _draw_plot_legend(
            draw,
            legend_font=legend_font,
            band_label=band_label,
            has_band=has_band,
            has_interval=has_interval,
            support=unsupported_legend_values,
            data_points=_plot_legend_data_points(
                self,
                axis_values,
                x_bounds,
                y_bounds,
                support_values,
            ),
        )

        image.save(output, format="PNG")
        return output
