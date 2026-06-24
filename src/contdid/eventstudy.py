"""Phase 5/6 event-study aggregation on top of the shared dose estimator stack.

Theoretical boundaries for CCK in event study
---------------------------------------------
- CCK theory (arXiv:2107.11869v3 Theorem 2) provides convergence rate and
  confidence band guarantees for individual two-period B-spline sieve
  regression problems.
- In event-study aggregation, each local (g,t) comparison IS a two-period
  problem, so fixed-dimension CCK applies directly to each cell.
- Aggregation across event-times uses standard sample-size-weighted averages
  with the same structure as the parametric path.
- Adaptive/Lepski dimension selection across event-time cells is NOT supported:
  aggregating estimates with heterogeneous adaptive dimensions lacks joint
  coverage theory. Use fixed-dimension CCK (num_knots parameter) instead.

Anticipation support
--------------------
- CGBS Assumption 3-MP(a) defines "no anticipation" with parameter a >= 0.
  When anticipation=a > 0, the base period shifts from g-1 to g-1-a.
  This is handled by timing.py's prepare_timing_groups().
"""

from __future__ import annotations

from math import nan
from numbers import Integral
from typing import Iterable

import numpy as np

from .bspline import (
    build_bspline_design,
    build_bspline_derivative_design,
    quantile_knots,
)
from .data import PanelData
from .cck import run_cck_backend
from .estimation import (  # type: ignore[attr-defined]
    _build_derivative_matrix,
    _build_design_matrix,
    _collapse_to_unit_differences,
    _coerce_basis_integer,
    _coerce_grid,
    _coerce_knots,
    _fit_shared_dose_design,
    _is_positive_dose_support_error,
    _DoseRegressionFit,
)
from .inference import attach_inference_payload
from .results import ContDIDResult
from .specs import ContDIDSpec
from .timing import _comparison_mask, prepare_timing_groups
from .validation import ContDIDValidationError, validate_panel_data, validate_spec


_GENERIC_SHAPE_CONSTRAINT = "no flat-zero shape restriction across event time"


def _eventstudy_identification_payload(expected_target: str) -> dict[str, str]:
    if expected_target == "level":
        return {
            "paper_estimand": "ATT(event_time)",
            "identifying_assumption": "PT-MP",
            "ordinary_pt_interpretation": (
                "post-treatment ATT(event_time); negative event-time cells are "
                "pre-trend diagnostics"
            ),
            "identification_note": (
                "Post-treatment ATT(event_time) cells are identified by "
                "PT-MP/local binary event-study comparisons; negative event-time "
                "cells diagnose pre-treatment parallel-trends plausibility rather "
                "than treatment effects."
            ),
        }
    return {
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
    }


def _scalar_variance_zero_tolerance(
    *,
    covariance_scale: float,
    loading_scale: float,
) -> float:
    return np.finfo(float).eps * max(1.0, covariance_scale) * max(1.0, loading_scale**2)


def _coerce_scalar_variance(
    value: float,
    *,
    covariance_scale: float,
    loading_scale: float,
) -> float:
    zero_tolerance = _scalar_variance_zero_tolerance(
        covariance_scale=covariance_scale,
        loading_scale=loading_scale,
    )
    if abs(value) <= zero_tolerance:
        return 0.0
    return value


def _local_fit_has_inference_df(
    fit: _DoseRegressionFit,
    *,
    expected_target: str,
) -> bool:
    basis_columns = len(fit.knots) + fit.degree + 1
    if fit.treated_count <= basis_columns:
        return False
    if expected_target == "level" and fit.untreated_count <= 1:
        return False
    return True


def _validate_eventstudy_request(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    expected_target: str,
) -> tuple[PanelData, ContDIDSpec]:
    validated_panel = validate_panel_data(panel)
    if spec.aggregation != "eventstudy":
        raise ContDIDValidationError("event-study estimators require aggregation='eventstudy'")
    validated_spec = validate_spec(spec, panel=validated_panel)

    if validated_spec.target_parameter != expected_target:
        raise ContDIDValidationError(
            f"expected target_parameter={expected_target!r}, got {validated_spec.target_parameter!r}"
        )
    if validated_spec.dose_est_method == "cck":
        # CCK with fixed dimension is supported for event study: each local
        # (g,t) comparison is a two-period problem where CCK theory
        # (arXiv:2107.11869v3 Theorem 2) applies directly.
        # Adaptive/Lepski dimension selection is NOT supported in event-study
        # aggregation due to lack of joint coverage theory across event-times.
        pass  # Allow fixed-dimension CCK to proceed
    if validated_spec.control_group not in {"notyettreated", "nevertreated"}:
        raise ContDIDValidationError(
            "event-study aggregation supports control_group values "
            "'notyettreated' and 'nevertreated' only"
        )
    return validated_panel, validated_spec


def _build_local_eventstudy_panel(
    panel: PanelData,
    *,
    timing_group: int,
    time_period: int,
    base_period: int,
    control_group: str,
    covariates: tuple[str, ...] | None = None,
) -> PanelData:
    frame = panel.frame
    group_column = panel.group_column
    time_column = panel.time_column
    id_column = panel.id_column
    dose_column = panel.dose_column

    comparison_cutoff_period = max(int(time_period), int(base_period))
    comparison_mask = _comparison_mask(
        frame,
        group_column=group_column,
        time_period=comparison_cutoff_period,
        control_group=control_group,
        exclude_group=timing_group,
    )

    # Build column list, including covariates if specified
    columns_to_keep = [
        id_column,
        time_column,
        panel.outcome_column,
        group_column,
        dose_column,
    ]
    if covariates:
        for cov_col in covariates:
            if cov_col in frame.columns and cov_col not in columns_to_keep:
                columns_to_keep.append(cov_col)

    subset = frame.loc[
        ((frame[group_column] == timing_group) | comparison_mask)
        & (frame[time_column].isin([base_period, time_period])),
        columns_to_keep,
    ].copy()
    if subset.empty:
        raise ContDIDValidationError(
            f"event-study subset is empty for timing_group={timing_group}, time_period={time_period}"
        )
    observed_periods = {int(value) for value in subset[time_column].unique().tolist()}
    if {int(base_period), int(time_period)} - observed_periods:
        raise ContDIDValidationError(
            f"event-study subset must contain base_period={base_period} and time_period={time_period}"
        )

    subset["local_treated"] = subset[group_column] == timing_group
    subset[group_column] = subset["local_treated"].map({True: 2, False: 0}).astype(int)
    subset.loc[~subset["local_treated"], dose_column] = 0.0

    if base_period == time_period:
        raise ContDIDValidationError(
            "event-study local comparison requires distinct base and target periods"
        )
    time_map = {base_period: 1, time_period: 2}
    subset[time_column] = subset[time_column].map(time_map).astype(int)

    return PanelData(
        frame=subset.drop(columns=["local_treated"]).reset_index(drop=True),
        id_column=id_column,
        time_column=time_column,
        outcome_column=panel.outcome_column,
        group_column=group_column,
        dose_column=dose_column,
    )


def _eventstudy_local_spec(
    validated_spec: ContDIDSpec,
    *,
    expected_target: str,
) -> ContDIDSpec:
    # Local event-study panels already recode the requested comparison set into
    # zero-dose controls, so the shared two-period dose fit should use the
    # normalized internal control-group token rather than the public route label.
    #
    # THEORY NOTE (CGBS Appendix SD.3):
    # Covariate adjustment in event-study uses conditional parallel trends
    # (SPT-X) applied independently to each local (g,t) two-period comparison.
    # Joint inference across event-times with covariates relies on the standard
    # influence function aggregation — the theoretical coverage guarantee for
    # the full event-study curve with covariates is an extension of the
    # unconditional case and has not been separately proven in the paper.
    # Users should interpret covariate-adjusted event-study results with
    # appropriate caution regarding the uniformity of the confidence band.
    return ContDIDSpec(
        target_parameter=expected_target,
        aggregation="dose",
        dose_est_method=validated_spec.dose_est_method,
        control_group="nevertreated",
        treatment_type=validated_spec.treatment_type,
        anticipation=validated_spec.anticipation,
        alp=validated_spec.alp,
        bstrap=validated_spec.bstrap,
        cband=validated_spec.cband,
        boot_type=validated_spec.boot_type,
        biters=validated_spec.biters,
        covariates=validated_spec.covariates,
        cluster_column=validated_spec.cluster_column,
    )


def _coerce_eventstudy_basis_controls(
    *,
    degree: int,
    num_knots: int,
) -> tuple[int, int]:
    checked_degree = _coerce_basis_integer(degree, name="degree")
    checked_num_knots = _coerce_basis_integer(num_knots, name="num_knots")
    if checked_degree < 1:
        raise ContDIDValidationError("degree must be at least 1 for dose estimation")
    if checked_num_knots < 0:
        raise ContDIDValidationError("num_knots must be nonnegative")
    return checked_degree, checked_num_knots


def _eventstudy_positive_dose_support(
    validated_panel: PanelData,
    *,
    timing_groups: Iterable[int] | None = None,
) -> np.ndarray:
    collapsed = _collapse_to_unit_differences(validated_panel, assume_valid_panel=True)
    max_time = int(
        np.max(validated_panel.frame[validated_panel.time_column].to_numpy(dtype=float))
    )
    dose_column = validated_panel.dose_column
    group_column = validated_panel.group_column
    if timing_groups is None:
        supported_groups: set[int] | None = None
    else:
        supported_groups = {int(value) for value in timing_groups}
    realized_treated = (
        (collapsed[group_column] > 0.0)
        & (collapsed[group_column] <= float(max_time))
        & (collapsed[dose_column] > 0.0)
    )
    if supported_groups is not None:
        realized_treated &= collapsed[group_column].astype(int).isin(supported_groups)
    return collapsed.loc[realized_treated, dose_column].to_numpy(dtype=float)


def _eventstudy_base_period_metadata(base_period: int | str | None) -> object:
    if base_period in {None, "varying"}:
        return None
    if base_period == "universal":
        return "universal"
    return int(base_period)  # type: ignore[arg-type]


def _eventstudy_base_period_strategy(base_period: int | str | None) -> str:
    if base_period == "universal":
        return "universal"
    if base_period in {None, "varying"}:
        return "varying_pre_period"
    return "fixed"


def _local_summary_from_fit(
    fit: _DoseRegressionFit,
    *,
    expected_target: str,
) -> tuple[float, float]:
    if expected_target == "level":
        treated_support = _build_design_matrix(
            fit.treated_dose, degree=fit.degree, knots=fit.knots
        )
    else:
        treated_support = _build_derivative_matrix(
            fit.treated_dose, degree=fit.degree, knots=fit.knots
        )

    # Zero-pad for covariate columns: covariates do not contribute to dose curve
    n_basis = treated_support.shape[1]
    n_coef = fit.coefficients.shape[0]
    if n_coef > n_basis:
        treated_support = np.column_stack(
            [
                treated_support,
                np.zeros((treated_support.shape[0], n_coef - n_basis)),
            ]
        )

    curve = treated_support @ fit.coefficients
    scalar_estimate = float(np.mean(curve))
    summary_loading = treated_support.mean(axis=0)
    treated_variance = float(summary_loading @ fit.covariance @ summary_loading)

    variance = treated_variance
    if expected_target == "level" and fit.untreated_delta.size > 1:
        variance += float(np.var(fit.untreated_delta, ddof=1) / fit.untreated_delta.size)
    if expected_target == "slope":
        variance = _mean_variance(_slope_summary_influence_by_id(fit).values())

    covariance_scale = float(np.max(np.abs(fit.covariance))) if fit.covariance.size else 0.0
    loading_scale = float(np.sum(np.abs(summary_loading)))
    variance = _coerce_scalar_variance(
        variance,
        covariance_scale=covariance_scale,
        loading_scale=loading_scale,
    )

    scalar_se = float(np.sqrt(max(variance, 0.0)))
    return scalar_estimate, scalar_se


def _slope_summary_influence_by_id(fit: _DoseRegressionFit) -> dict[object, float]:
    treated_support = _build_derivative_matrix(
        fit.treated_dose, degree=fit.degree, knots=fit.knots
    )

    # Zero-pad for covariate columns: covariates do not contribute to dose curve
    n_basis = treated_support.shape[1]
    n_coef = fit.coefficients.shape[0]
    if n_coef > n_basis:
        treated_support = np.column_stack(
            [
                treated_support,
                np.zeros((treated_support.shape[0], n_coef - n_basis)),
            ]
        )

    curve = treated_support @ fit.coefficients
    centered_curve = curve - float(np.mean(curve))
    summary_loading = treated_support.mean(axis=0)
    score = fit.design * fit.residual[:, None]
    bread = np.linalg.pinv((fit.design.T @ fit.design) / fit.treated_count)
    coefficient_influence = score @ bread @ summary_loading
    influence = centered_curve + coefficient_influence
    return {
        unit_id: float(value) for unit_id, value in zip(fit.treated_unit_ids, influence.tolist())
    }


def _local_untreated_delta_by_id(local_panel: PanelData) -> dict[object, float]:
    _, comparison = _local_level_delta_maps_by_id(local_panel)
    return comparison


def _local_treated_delta_by_id(local_panel: PanelData) -> dict[object, float]:
    treated, _ = _local_level_delta_maps_by_id(local_panel)
    return treated


def _local_level_delta_maps_by_id(
    local_panel: PanelData,
) -> tuple[dict[object, float], dict[object, float]]:
    collapsed = _collapse_to_unit_differences(local_panel, assume_valid_panel=True)
    id_values = collapsed[local_panel.id_column].to_numpy()
    group_values = collapsed[local_panel.group_column].to_numpy()
    delta_values = collapsed["delta_outcome"].to_numpy(dtype=float)
    # Vectorized split instead of Python loop
    treated_mask = group_values != 0
    treated = dict(
        zip(
            id_values[treated_mask].tolist(),
            delta_values[treated_mask].tolist(),
        )
    )
    comparison = dict(
        zip(
            id_values[~treated_mask].tolist(),
            delta_values[~treated_mask].tolist(),
        )
    )
    return treated, comparison


def _mean_variance(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size <= 1:
        return 0.0
    return float(np.var(array, ddof=1) / array.size)


def _local_level_has_inference_df(
    treated_delta_by_id: dict[object, float],
    comparison_delta_by_id: dict[object, float],
) -> bool:
    return len(treated_delta_by_id) > 1 and len(comparison_delta_by_id) > 1


def _local_level_summary(
    treated_delta_by_id: dict[object, float],
    comparison_delta_by_id: dict[object, float],
) -> tuple[float, float]:
    treated_values = list(treated_delta_by_id.values())
    comparison_values = list(comparison_delta_by_id.values())
    treated_mean = float(np.mean(np.asarray(treated_values, dtype=float)))
    comparison_mean = float(np.mean(np.asarray(comparison_values, dtype=float)))
    treated_variance = _mean_variance(treated_values)
    comparison_variance = _mean_variance(comparison_values)
    return (
        treated_mean - comparison_mean,
        float(np.sqrt(max(treated_variance + comparison_variance, 0.0))),
    )


def _centered_mean_influence_by_id(values: dict[object, float]) -> dict[object, float]:
    sample_size = len(values)
    if sample_size <= 1:
        return {unit_id: 0.0 for unit_id in values}

    keys = list(values.keys())
    vals = np.fromiter(values.values(), dtype=float, count=sample_size)
    mean_value = vals.mean()
    scale = float(np.sqrt(sample_size / (sample_size - 1)) / sample_size)
    centered = (vals - mean_value) * scale
    return dict(zip(keys, centered.tolist()))


def _add_scaled_influence(
    target: dict[object, float],
    source: dict[object, float],
    *,
    scale: float,
) -> None:
    # Hot loop: avoid repeated attribute lookup and redundant float() casts
    target_get = target.get
    for unit_id, value in source.items():
        target[unit_id] = target_get(unit_id, 0.0) + scale * value


def _level_entry_influence_by_id(entry: dict[str, object]) -> dict[object, float]:
    treated_values = entry.get("_treated_delta_by_id")
    comparison_values = entry.get("_comparison_delta_by_id")
    influence: dict[object, float] = {}
    if isinstance(treated_values, dict):
        _add_scaled_influence(
            influence,
            _centered_mean_influence_by_id(treated_values),
            scale=1.0,
        )
    if isinstance(comparison_values, dict):
        _add_scaled_influence(
            influence,
            _centered_mean_influence_by_id(comparison_values),
            scale=-1.0,
        )
    return influence


def _slope_entry_influence_by_id(entry: dict[str, object]) -> dict[object, float]:
    influence_values = entry.get("_slope_influence_by_id")
    if not isinstance(influence_values, dict):
        return {}
    return _centered_mean_influence_by_id(influence_values)


def _entry_influence_by_id(
    entry: dict[str, object],
    *,
    expected_target: str,
) -> dict[object, float]:
    if expected_target == "level":
        return _level_entry_influence_by_id(entry)
    return _slope_entry_influence_by_id(entry)


# DESIGN NOTE (Theory: CGBS Appendix B.2):
# Event-study influence aggregation accumulates IF within event-time cells
# (weighted by treated sample size per CGBS Corollary 2). Cross-event-time
# covariance is computed explicitly in _cross_eventstudy_covariance() rather
# than through IF matrix aggregation. This reflects the panel structure where
# different event-times correspond to different (g,t) pairs that may share
# units in the comparison group — the multiplier bootstrap correctly handles
# this via shared unit-level weights in attach_inference_payload().
#
# Assumption: within each event-time cell, (g,t) contributions are independent
# conditional on treatment timing. This is standard in staggered DiD literature.
def _aggregate_eventstudy_influence_by_id(
    entries: list[dict[str, object]],
    *,
    expected_target: str,
) -> dict[object, float]:
    influence: dict[object, float] = {}
    for weight, entry in _weighted_entries(entries):
        _add_scaled_influence(
            influence,
            _entry_influence_by_id(entry, expected_target=expected_target),
            scale=weight,
        )
    return influence


def _influence_covariance(
    first: dict[object, float],
    second: dict[object, float],
) -> float:
    if not first or not second:
        return 0.0
    if len(first) > len(second):
        first, second = second, first
    return float(sum(value * second.get(unit_id, 0.0) for unit_id, value in first.items()))


def _weighted_entries(
    entries: Iterable[dict[str, object]],
) -> list[tuple[float, dict[str, object]]]:
    materialized = list(entries)
    if not materialized:
        return []
    treated_count_values: list[int] = []
    for entry in materialized:
        treated_count = entry.get("treated_count")
        if isinstance(treated_count, (bool, np.bool_)) or not isinstance(treated_count, Integral):
            raise ContDIDValidationError(
                "event-study cohort treated_count must be a positive integer"
            )
        checked_count = int(treated_count)
        if checked_count <= 0:
            raise ContDIDValidationError(
                "event-study cohort treated_count must be a positive integer"
            )
        treated_count_values.append(checked_count)
    try:
        treated_counts = np.asarray(treated_count_values, dtype=float)
    except OverflowError as exc:
        raise ContDIDValidationError(
            "event-study cohort treated_count must be a finite positive integer"
        ) from exc
    if not np.isfinite(treated_counts).all():
        raise ContDIDValidationError(
            "event-study cohort treated_count must be a finite positive integer"
        )
    total_treated = float(np.sum(treated_counts))
    if not np.isfinite(total_treated) or total_treated <= 0.0:
        raise ContDIDValidationError(
            "event-study cohort treated_count must sum to a finite positive integer"
        )
    return [
        (float(weight), entry)
        for weight, entry in zip(treated_counts / total_treated, materialized)
    ]


def _cross_eventstudy_covariance(
    left_entries: list[dict[str, object]],
    right_entries: list[dict[str, object]],
    *,
    expected_target: str,
) -> float:
    left_influence = _aggregate_eventstudy_influence_by_id(
        left_entries,
        expected_target=expected_target,
    )
    right_influence = _aggregate_eventstudy_influence_by_id(
        right_entries,
        expected_target=expected_target,
    )
    return _influence_covariance(left_influence, right_influence)


def _aggregate_eventstudy_estimate(entries: list[dict[str, object]]) -> float:
    weighted = _weighted_entries(entries)
    if not weighted:
        return nan
    return float(
        sum(weight * float(entry["estimate"]) for weight, entry in weighted)  # type: ignore[arg-type, misc]
    )


def _aggregate_eventstudy_variance(
    entries: list[dict[str, object]],
    *,
    expected_target: str,
) -> float:
    weighted = _weighted_entries(entries)
    if not weighted:
        return 0.0

    influence = _aggregate_eventstudy_influence_by_id(
        entries,
        expected_target=expected_target,
    )
    variance = _influence_covariance(influence, influence)
    return max(
        _coerce_scalar_variance(
            variance,
            covariance_scale=abs(variance),
            loading_scale=float(sum(abs(weight) for weight, _ in weighted)),
        ),
        0.0,
    )


def _aggregate_eventstudy(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    expected_target: str,
    dvals: Iterable[float] | float | None,
    degree: int,
    num_knots: int,
    base_period: int | str | None,
) -> ContDIDResult:
    validated_panel, validated_spec = _validate_eventstudy_request(
        panel,
        spec,
        expected_target=expected_target,
    )
    degree, num_knots = _coerce_eventstudy_basis_controls(
        degree=degree,
        num_knots=num_knots,
    )
    prepared = prepare_timing_groups(
        validated_panel,
        control_group=validated_spec.control_group,
        anticipation=validated_spec.anticipation,
        base_period=base_period,
        assume_valid_panel=True,
    )
    if not prepared.loc[prepared["support"] & prepared["post_treatment"], :].shape[0]:
        raise ContDIDValidationError(
            "event-study aggregation requires at least one observed post-treatment event time"
        )
    supported_timing_groups = sorted(
        int(value)
        for value in prepared.loc[prepared["support"], "timing_group"].drop_duplicates().tolist()
    )
    positive_dose_support = _eventstudy_positive_dose_support(
        validated_panel,
        timing_groups=supported_timing_groups,
    )
    dose_grid = _coerce_grid(
        dvals=dvals,
        positive_dose=positive_dose_support,
        enforce_observed_support=True,
        require_strict_explicit_grid=True,
    )
    basis_knots = _coerce_knots(
        positive_dose_support,
        num_knots,
    )

    estimand = "ATT(event_time)" if expected_target == "level" else "ACRT(event_time)"
    summary_label = "level" if expected_target == "level" else "slope"

    identified_by_event_time: dict[int, list[dict[str, object]]] = {}
    identified_timing_groups: set[int] = set()
    has_positive_dose_supported_row = False
    has_post_treatment_positive_dose_support = False
    has_post_treatment_inference_df = False
    for row in prepared.loc[prepared["support"], :].itertuples(index=False):
        local_panel = _build_local_eventstudy_panel(
            validated_panel,
            timing_group=int(row.timing_group),
            time_period=int(row.time_period),
            base_period=int(row.base_period),
            control_group=str(row.comparison_type),
            covariates=validated_spec.covariates,
        )
        local_spec = _eventstudy_local_spec(
            validated_spec,
            expected_target=expected_target,
        )
        if expected_target == "level" and validated_spec.dose_est_method != "cck":
            treated_delta_by_id, comparison_delta_by_id = _local_level_delta_maps_by_id(
                local_panel
            )
            if not treated_delta_by_id or not comparison_delta_by_id:
                continue
            has_positive_dose_supported_row = True
            if row.post_treatment:
                has_post_treatment_positive_dose_support = True
            local_has_inference_df = _local_level_has_inference_df(
                treated_delta_by_id,
                comparison_delta_by_id,
            )
            event_time = int(row.event_time)
            if row.post_treatment and local_has_inference_df:
                has_post_treatment_inference_df = True
            if not local_has_inference_df:
                continue
            scalar_estimate, scalar_se = _local_level_summary(
                treated_delta_by_id,
                comparison_delta_by_id,
            )
            identified_timing_groups.add(int(row.timing_group))
            identified_by_event_time.setdefault(event_time, []).append(
                {
                    "timing_group": int(row.timing_group),
                    "time_period": int(row.time_period),
                    "base_period": int(row.base_period),
                    "comparison_count": int(row.comparison_count),
                    "treated_count": int(row.treated_count),
                    "estimate": scalar_estimate,
                    "std_error": scalar_se,
                    "_post_treatment": bool(row.post_treatment),
                    "_treated_delta_by_id": treated_delta_by_id,
                    "_comparison_delta_by_id": comparison_delta_by_id,
                }
            )
            continue

        # --- CCK fixed-dimension path ---
        # Each local (g,t) comparison is a two-period problem where CCK theory
        # (arXiv:2107.11869v3 Theorem 2) applies directly.
        if validated_spec.dose_est_method == "cck":
            collapsed = _collapse_to_unit_differences(local_panel, assume_valid_panel=True)
            treated_mask = collapsed[local_panel.group_column] != 0
            dose_treated = collapsed.loc[treated_mask, local_panel.dose_column].to_numpy(
                dtype=float
            )
            delta_treated = collapsed.loc[treated_mask, "delta_outcome"].to_numpy(dtype=float)
            delta_untreated = collapsed.loc[~treated_mask, "delta_outcome"].to_numpy(dtype=float)

            # Require positive dose support and minimum sample sizes
            if dose_treated.size < 2 or delta_untreated.size < 1:
                continue
            if not np.any(dose_treated > 0):
                continue
            # Filter to positive-dose treated units only
            pos_mask = dose_treated > 0
            # Save unit IDs for positive-dose subset before overwriting arrays
            treated_ids_all = collapsed.loc[treated_mask, local_panel.id_column].to_numpy()
            pos_treated_ids = treated_ids_all[pos_mask].tolist()
            dose_treated = dose_treated[pos_mask]
            delta_treated = delta_treated[pos_mask]
            if dose_treated.size < 2:
                continue

            has_positive_dose_supported_row = True
            if row.post_treatment:
                has_post_treatment_positive_dose_support = True

            try:
                cck_result = run_cck_backend(
                    delta_outcome=delta_treated,
                    dose=dose_treated,
                    dvals=list(dose_grid),
                    untreated_delta=delta_untreated,
                    require_untreated_variance_df=(expected_target == "level"),
                    bstrap=validated_spec.bstrap,
                    cband=validated_spec.cband,
                    alp=validated_spec.alp,
                    biters=validated_spec.biters,
                    boot_type=validated_spec.boot_type,
                    degree=degree,
                    num_knots=num_knots,
                    adaptive=False,  # Critical: no adaptive in event study
                )
            except ContDIDValidationError as error:
                if _is_positive_dose_support_error(error):
                    continue
                raise

            # Extract scalar estimate and SE
            if expected_target == "level":
                scalar_estimate = float(cck_result["overall_att"])
                scalar_se = float(cck_result["overall_att_se"])
            else:
                scalar_estimate = float(cck_result["overall_acrt"])
                scalar_se = float(cck_result["overall_acrt_se"])

            # Compute CCK slope influence function for event-study inference.
            # This mirrors _slope_summary_influence_by_id() but uses the CCK
            # B-spline basis from the sieve regression.  Without this, the
            # event-study variance/covariance path returns zero for CCK slope.
            # Compute actual interior knots (quantile_knots may dedup)
            _ik = quantile_knots(dose_treated, num_knots) if num_knots > 0 else []
            if expected_target == "slope":
                _cck_design = build_bspline_design(dose_treated, degree, _ik)
                _cck_coef, *_ = np.linalg.lstsq(_cck_design, delta_treated, rcond=None)
                _cck_resid = delta_treated - _cck_design @ _cck_coef
                _cck_deriv = build_bspline_derivative_design(dose_treated, degree, _ik)
                _cck_curve = _cck_deriv @ _cck_coef
                _cck_centered = _cck_curve - float(np.mean(_cck_curve))
                _cck_summary_ld = _cck_deriv.mean(axis=0)
                _cck_score = _cck_design * _cck_resid[:, None]
                _n_t = dose_treated.size
                _cck_bread = np.linalg.pinv((_cck_design.T @ _cck_design) / _n_t)
                _cck_coef_if = _cck_score @ _cck_bread @ _cck_summary_ld
                _cck_influence = _cck_centered + _cck_coef_if
                _slope_if_by_id: dict[object, float] | None = dict(
                    zip(pos_treated_ids, _cck_influence.tolist())
                )
            else:
                _slope_if_by_id = None

            # Check inference df: need enough treated obs for the basis
            # Use actual knot count (quantile_knots may return fewer due to dedup)
            basis_columns = len(_ik) + degree + 1
            local_has_inference_df = dose_treated.size > basis_columns
            if expected_target == "level" and delta_untreated.size <= 1:
                local_has_inference_df = False

            event_time = int(row.event_time)
            if row.post_treatment and local_has_inference_df:
                has_post_treatment_inference_df = True
            if not local_has_inference_df:
                continue

            identified_timing_groups.add(int(row.timing_group))
            entry_dict: dict[str, object] = {
                "timing_group": int(row.timing_group),
                "time_period": int(row.time_period),
                "base_period": int(row.base_period),
                "comparison_count": int(row.comparison_count),
                "treated_count": int(row.treated_count),
                "estimate": scalar_estimate,
                "std_error": scalar_se,
                "_post_treatment": bool(row.post_treatment),
                "_treated_delta_by_id": dict(zip(pos_treated_ids, delta_treated.tolist())),
                "_comparison_delta_by_id": dict(
                    zip(
                        collapsed.loc[~treated_mask, local_panel.id_column].tolist(),
                        delta_untreated.tolist(),
                    )
                ),
            }
            if _slope_if_by_id is not None:
                entry_dict["_slope_influence_by_id"] = _slope_if_by_id
            identified_by_event_time.setdefault(event_time, []).append(entry_dict)
            continue

        # --- Parametric path (slope target) ---
        try:
            local_fit = _fit_shared_dose_design(
                local_panel,
                local_spec,
                expected_target=expected_target,
                dvals=dose_grid,
                degree=degree,
                num_knots=num_knots,
                enforce_observed_support=False,
                require_inference_df=False,
                require_untreated_variance_df=False,
                require_public_dose_control_group=False,
                require_strict_explicit_grid=False,
                assume_valid_panel=True,
                knots=basis_knots,
            )
        except ContDIDValidationError as error:
            if _is_positive_dose_support_error(error):
                continue
            raise
        has_positive_dose_supported_row = True
        if row.post_treatment:
            has_post_treatment_positive_dose_support = True
        local_has_inference_df = _local_fit_has_inference_df(
            local_fit,
            expected_target=expected_target,
        )
        event_time = int(row.event_time)
        if row.post_treatment and local_has_inference_df:
            has_post_treatment_inference_df = True
        if not local_has_inference_df:
            continue
        scalar_estimate, scalar_se = _local_summary_from_fit(
            local_fit,
            expected_target=expected_target,
        )
        identified_timing_groups.add(int(row.timing_group))
        identified_by_event_time.setdefault(event_time, []).append(
            {
                "timing_group": int(row.timing_group),
                "time_period": int(row.time_period),
                "base_period": int(row.base_period),
                "comparison_count": int(row.comparison_count),
                "treated_count": int(row.treated_count),
                "estimate": scalar_estimate,
                "std_error": scalar_se,
                "_post_treatment": bool(row.post_treatment),
                "_slope_influence_by_id": _slope_summary_influence_by_id(local_fit),
                "_local_fit": local_fit,
            }
        )

    if not has_positive_dose_supported_row:
        raise ContDIDValidationError(
            "event-study aggregation found no locally identified positive-dose support for the requested parametric basis"
        )
    if not has_post_treatment_positive_dose_support:
        raise ContDIDValidationError(
            "event-study aggregation requires at least one locally identified post-treatment event time with positive-dose support"
        )
    if not has_post_treatment_inference_df:
        raise ContDIDValidationError(
            "event-study aggregation requires at least one locally identified post-treatment event time with positive-dose support and inference degrees of freedom"
        )

    identified_positive_dose_support = (
        validated_panel.frame.loc[
            validated_panel.frame[validated_panel.group_column]
            .astype(int)
            .isin(identified_timing_groups)
            & (validated_panel.frame[validated_panel.dose_column] > 0.0),
            [validated_panel.id_column, validated_panel.dose_column],
        ]
        .drop_duplicates(subset=[validated_panel.id_column])[validated_panel.dose_column]
        .to_numpy(dtype=float)
    )
    dose_grid = _coerce_grid(
        dvals=dvals,
        positive_dose=identified_positive_dose_support,
        enforce_observed_support=True,
        require_strict_explicit_grid=True,
    )
    identified_basis_knots = _coerce_knots(
        identified_positive_dose_support,
        num_knots,
    )
    if identified_basis_knots != basis_knots:
        raise ContDIDValidationError(
            "event-study spline basis support changed after local inference filtering; "
            "the requested num_knots is not identified on the reported timing groups"
        )
    basis_knots = identified_basis_knots

    event_time_grid = sorted(identified_by_event_time)
    timing_groups = sorted(identified_timing_groups)
    cohort_summary: list[dict[str, object]] = []
    eventstudy_curve: list[float] = []
    eventstudy_se: list[float] = []
    support: list[bool] = []

    for event_time in event_time_grid:
        cohort_estimates = identified_by_event_time[event_time]
        weighted = _weighted_entries(cohort_estimates)

        supported = bool(weighted)
        if weighted:
            mean_estimate = float(
                sum(weight * float(entry["estimate"]) for weight, entry in weighted)  # type: ignore[arg-type, misc]
            )
            mean_se = float(
                np.sqrt(
                    _aggregate_eventstudy_variance(
                        cohort_estimates,
                        expected_target=expected_target,
                    )
                )
            )
        else:
            mean_estimate = nan
            mean_se = 0.0
        public_cohort_estimates = [
            {
                **{key: value for key, value in entry.items() if not key.startswith("_")},
                "aggregation_weight": float(weight),
            }
            for weight, entry in weighted
        ]
        support.append(supported)
        eventstudy_curve.append(mean_estimate)
        eventstudy_se.append(mean_se)
        cohort_summary.append(
            {
                "event_time": int(event_time),
                "timing_groups": [int(entry["timing_group"]) for entry in cohort_estimates],  # type: ignore[call-overload]
                "cohort_estimates": public_cohort_estimates,
                "mean_estimate": mean_estimate,
                "std_error": mean_se,
                "support": supported,
            }
        )

    post_treatment = [value for time, value in zip(event_time_grid, eventstudy_curve) if time >= 0]
    post_treatment_entries = [
        entry
        for event_time in event_time_grid
        for entry in identified_by_event_time[event_time]
        if bool(entry.get("_post_treatment"))
    ]
    summary_aggregates = {
        f"overall_{summary_label}": _aggregate_eventstudy_estimate(post_treatment_entries),
        f"post_treatment_mean_{summary_label}": float(sum(post_treatment) / len(post_treatment)),
    }
    shape_constraints = {
        "level_curve": _GENERIC_SHAPE_CONSTRAINT,
        "slope_curve": _GENERIC_SHAPE_CONSTRAINT,
        "event_time_order": "ascending integers",
    }
    metadata = {
        "target_parameter": expected_target,
        "dose_est_method": validated_spec.dose_est_method,
        "aggregation": "eventstudy",
        "event_time_grid": event_time_grid,
        "timing_group_support": {
            "timing_groups": timing_groups,
            "never_treated_group": 0,
            "reporting_scale": "length of exposure to treatment",
            "base_period_strategy": _eventstudy_base_period_strategy(base_period),
        },
        "base_period": _eventstudy_base_period_metadata(base_period),
        "support": support,
        "control_group": validated_spec.control_group,
        "dose_grid": dose_grid,
        "basis": {
            "type": "bspline" if basis_knots else "global_polynomial",
            "degree": degree,
            "num_knots": len(basis_knots),
            "interior_knots": basis_knots,
        },
        "cohort_summary": cohort_summary,
        "summary": summary_aggregates,
        "summary_aggregates": summary_aggregates,
        "identification": _eventstudy_identification_payload(expected_target),
        "shape_constraints": shape_constraints,
        "source_estimator": (
            "cck_fixed_dimension_eventstudy"
            if validated_spec.dose_est_method == "cck"
            else (
                "binary_eventstudy_mean"
                if expected_target == "level"
                else "phase4_shared_dose_stack"
            )
        ),
        "inference": "bootstrap",
        "inference_covariance": "full_event_time_covariance",
    }

    result = ContDIDResult(
        estimand=estimand,
        grid=[float(t) for t in event_time_grid],
        estimate=eventstudy_curve,
        std_error=eventstudy_se,
        timing_group=timing_groups,
        event_time=event_time_grid,
        event_time_grid=event_time_grid,
        cohort_summary=cohort_summary,
        metadata=metadata,
    )
    covariance = np.diag(
        np.square(np.clip(np.asarray(eventstudy_se, dtype=float), a_min=0.0, a_max=None))
    )
    for left_index, left_event_time in enumerate(event_time_grid):
        for right_index in range(left_index + 1, len(event_time_grid)):
            right_event_time = event_time_grid[right_index]
            cross_covariance = _cross_eventstudy_covariance(
                identified_by_event_time[left_event_time],
                identified_by_event_time[right_event_time],
                expected_target=expected_target,
            )
            covariance[left_index, right_index] = cross_covariance
            covariance[right_index, left_index] = cross_covariance
    result.metadata["event_time_covariance"] = covariance.tolist()
    loadings = np.eye(len(event_time_grid), dtype=float)
    return attach_inference_payload(
        result, loadings=loadings, covariance=covariance, spec=validated_spec
    )


def estimate_eventstudy_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    base_period: int | str | None = None,
) -> ContDIDResult:
    return _aggregate_eventstudy(
        panel,
        spec,
        expected_target="level",
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        base_period=base_period,
    )


def estimate_eventstudy_slope_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    base_period: int | str | None = None,
) -> ContDIDResult:
    return _aggregate_eventstudy(
        panel,
        spec,
        expected_target="slope",
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        base_period=base_period,
    )
