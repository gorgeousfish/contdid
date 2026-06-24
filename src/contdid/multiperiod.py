"""Multi-period dose estimation via 2x2 decomposition.

Implements the staggered treatment extension from Callaway, Goodman-Bacon,
& Sant'Anna (2024, arXiv:2107.02637v7, Theorem 3.1) where each (group g,
time t) pair with t >= g provides a local LATT(g,t,d|g,d) or LACRT(g,t,d|g,d)
estimate, and these are aggregated using sample-size-weighted influence functions.

Theoretical basis
-----------------
- Identification: CGBS Theorem 3.1 — under parallel trends (PT), LATT(g,t,d|g,d)
  is identified for all t >= g via long differences from base period to t.
- Aggregation: CGBS Corollary 1 — dose-specific aggregation uses treated sample
  size weights n_g / sum(n_g).
- Inference: Multiplier bootstrap with shared unit-level multipliers (CGBS
  Appendix C) produces valid uniform confidence bands.

Theoretical boundaries
---------------------
- Parametric B-spline basis is applied locally per (g,t) pair; the paper does NOT
  provide complete asymptotic distribution theory for multi-period continuous D.
  Bootstrap inference is empirically validated through simulation.
- CCK nonparametric / Lepski adaptive methods are NOT supported in multi-period
  settings (paper only proves two-period convergence rates).
- Covariate adjustment is NOT supported (paper does not provide conditional
  parallel trends theory for multi-period decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .bspline import (
    build_bspline_design,
    build_bspline_derivative_design,
    quantile_knots,
)
from .influence import (
    InfluenceFunction,
    compute_dose_influence_function,
    aggregate_influence_functions,
)
from .validation import ContDIDValidationError


@dataclass(slots=True)
class MultiPeriodDoseResult:
    """Result container for multi-period dose estimation.

    Attributes
    ----------
    dose_grid : tuple of evaluation dose points
    point_estimate : tuple of estimated effects at each dose grid point
    standard_error : tuple of standard errors
    confidence_band_lower : tuple of lower band bounds
    confidence_band_upper : tuple of upper band bounds
    aggregated_influence : InfluenceFunction — aggregated IF for further use
    local_results : list of per-(g,t) result dicts
    metadata : dict of estimation metadata
    """

    dose_grid: tuple[float, ...]
    point_estimate: tuple[float, ...]
    standard_error: tuple[float, ...]
    confidence_band_lower: tuple[float, ...]
    confidence_band_upper: tuple[float, ...]
    aggregated_influence: InfluenceFunction
    local_results: list[dict[str, Any]]
    metadata: dict[str, Any]


def _identify_dose_timing_groups(
    panel_df: pd.DataFrame,
    *,
    group_column: str,
    time_column: str,
    id_column: str,
    dose_column: str,
    control_group: str = "nevertreated",
    anticipation: int = 0,
) -> list[dict[str, Any]]:
    """Identify valid (group, time) pairs for dose estimation.

    Per CGBS Theorem 3.1, LATT(g,t,d|g,d) is identified for ALL t >= g under
    parallel trends.  For each timing group g, we enumerate all post-treatment
    periods t in {g, g+1, ..., T} and construct a 2x2 comparison using long
    differences from the base period to t.

    A valid pair (g, t) has:
    - t >= g (post-treatment period for cohort g)
    - At least some treated units with positive dose observable at times pre_t and t
    - Available control units (nevertreated or not-yet-treated at time t)

    Base period selection (varying strategy):
    - pre_t = max{tt in all_times : tt < g}  (the period immediately before
      treatment adoption), providing long differences from g-1 to t.

    Returns list of dicts with keys: group, time, pre_time, event_time,
    treated_ids, control_ids, n_treated, n_control.
    """
    all_times = sorted(panel_df[time_column].unique())
    groups = sorted(panel_df[panel_df[group_column] > 0][group_column].unique())
    timing_groups: list[dict[str, Any]] = []

    for g in groups:
        # Base period: g - 1 - anticipation (CGBS Assumption 3-MP(a))
        pre_times = [tt for tt in all_times if tt < g - anticipation]
        if len(pre_times) == 0:
            continue
        pre_t = pre_times[-1]

        # Treated units: those in group g
        treated_mask = panel_df[group_column] == g
        treated_ids = panel_df.loc[treated_mask, id_column].unique()

        if len(treated_ids) < 2:
            continue

        # Enumerate all post-treatment periods t >= g (CGBS Theorem 3.1)
        post_periods = [tt for tt in all_times if tt >= g]

        for t in post_periods:
            # Control units based on control_group strategy at time t
            if control_group == "nevertreated":
                control_mask = panel_df[group_column] == 0
            elif control_group == "notyettreated":
                # Units not yet treated by time t (never-treated + groups treated after t)
                control_mask = (panel_df[group_column] == 0) | (panel_df[group_column] > t)
            else:
                raise ContDIDValidationError(f"Unsupported control_group: {control_group}")

            control_ids = panel_df.loc[control_mask, id_column].unique()

            if len(control_ids) < 1:
                continue

            timing_groups.append(
                {
                    "group": g,
                    "time": t,
                    "pre_time": pre_t,
                    "event_time": t - g,
                    "treated_ids": treated_ids,
                    "control_ids": control_ids,
                    "n_treated": len(treated_ids),
                    "n_control": len(control_ids),
                }
            )

    return timing_groups


def _run_local_dose_estimation(
    panel_df: pd.DataFrame,
    *,
    group_info: dict[str, Any],
    id_column: str,
    time_column: str,
    outcome_column: str,
    dose_column: str,
    group_column: str,
    dose_grid: np.ndarray,
    degree: int,
    num_knots: int,
    target: str,
    covariate_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Run local 2x2 dose estimation for a single (g,t) pair.

    Steps:
    1. Extract pre/post data for treated and control units
    2. Compute delta_outcome = Y_post - Y_pre for each unit
    3. Subtract mean control change (DID adjustment)
    4. Run dose regression on treated units' adjusted delta_outcome
    5. Return local estimates and influence function
    """
    g = group_info["group"]
    t = group_info["time"]
    pre_t = group_info["pre_time"]
    treated_ids = group_info["treated_ids"]
    control_ids = group_info["control_ids"]

    # Extract pre and post observations
    pre_data = panel_df[panel_df[time_column] == pre_t].set_index(id_column)
    post_data = panel_df[panel_df[time_column] == t].set_index(id_column)

    # Compute outcome changes for treated
    treated_pre = pre_data.loc[pre_data.index.isin(treated_ids)]
    treated_post = post_data.loc[post_data.index.isin(treated_ids)]
    common_treated = treated_pre.index.intersection(treated_post.index)

    if len(common_treated) < 2:
        return {"status": "skipped", "reason": "insufficient treated observations"}

    delta_treated = treated_post.loc[common_treated, outcome_column].values.astype(
        float
    ) - treated_pre.loc[common_treated, outcome_column].values.astype(float)
    dose_treated = treated_post.loc[common_treated, dose_column].values.astype(float)
    treated_unit_ids = tuple(common_treated.tolist())

    # Compute outcome changes for control
    control_pre = pre_data.loc[pre_data.index.isin(control_ids)]
    control_post = post_data.loc[post_data.index.isin(control_ids)]
    common_control = control_pre.index.intersection(control_post.index)

    if len(common_control) < 1:
        return {"status": "skipped", "reason": "insufficient control observations"}

    delta_control = control_post.loc[common_control, outcome_column].values.astype(
        float
    ) - control_pre.loc[common_control, outcome_column].values.astype(float)
    control_unit_ids = tuple(common_control.tolist())

    # DID adjustment: subtract mean control change
    control_mean = float(np.mean(delta_control))
    delta_adjusted = delta_treated - control_mean

    # Select knots from the local treated dose distribution
    if num_knots > 0:
        knots = quantile_knots(dose_treated, num_knots)
    else:
        knots = []

    # Build design matrix with explicit boundaries
    xmin = float(dose_treated.min())
    xmax = float(dose_treated.max())
    if xmax <= xmin:
        xmax = xmin + 1.0

    design = build_bspline_design(dose_treated, degree, knots, xmin=xmin, xmax=xmax)

    # Covariate adjustment not supported in multi-period local estimation
    if covariate_columns is not None and len(covariate_columns) > 0:
        raise NotImplementedError(
            "Covariate adjustment is not supported in multi-period local estimation."
        )
    p_cov = 0

    # Check rank
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        return {"status": "skipped", "reason": "rank-deficient design"}

    # Fit OLS
    coefficients, _, _, _ = np.linalg.lstsq(design, delta_adjusted, rcond=None)
    residual = delta_adjusted - design @ coefficients

    # Build loadings for dose grid evaluation (clip grid to local observed range)
    grid_clipped = np.clip(dose_grid, xmin, xmax)

    if target == "level":
        eval_loadings = build_bspline_design(grid_clipped, degree, knots, xmin=xmin, xmax=xmax)
    else:  # slope
        eval_loadings = build_bspline_derivative_design(
            grid_clipped, degree, knots, xmin=xmin, xmax=xmax
        )

    # Zero-pad loadings for covariate columns so point estimates reflect dose only
    if p_cov > 0:
        loadings = np.column_stack([eval_loadings, np.zeros((len(dose_grid), p_cov))])
    else:
        loadings = eval_loadings

    # Point estimates at grid (dose part only)
    point_est = loadings @ coefficients

    # Compute influence function for this local estimate
    n_total = len(common_treated) + len(common_control)

    inf_func = compute_dose_influence_function(
        design=design,
        residual=residual,
        coefficients=coefficients,
        loadings=loadings,
        treated_unit_ids=treated_unit_ids,
        treated_dose=dose_treated,
        n_total=n_total,
        untreated_delta=delta_control if target == "level" else None,
        untreated_unit_ids=control_unit_ids if target == "level" else None,
        include_untreated=(target == "level"),
        estimand_labels=tuple(f"d_{d:.4f}" for d in dose_grid),
    )

    return {
        "status": "success",
        "group": g,
        "time": t,
        "event_time": group_info.get("event_time", t - g),
        "point_estimate": point_est,
        "influence_function": inf_func,
        "n_treated": len(common_treated),
        "n_control": len(common_control),
        "coefficients": coefficients,
    }


def estimate_multiperiod_dose(
    panel_df: pd.DataFrame,
    *,
    id_column: str,
    time_column: str,
    outcome_column: str,
    dose_column: str,
    group_column: str,
    dose_grid: np.ndarray | list[float] | None = None,
    degree: int = 3,
    num_knots: int = 0,
    control_group: str = "nevertreated",
    target: str = "level",
    anticipation: int = 0,
    biters: int = 1000,
    alp: float = 0.05,
    cband: bool = True,
    boot_seed: int | None = None,
    covariates: list[str] | None = None,
) -> MultiPeriodDoseResult:
    """Estimate multi-period dose effects via 2x2 decomposition.

    Decomposes a staggered panel into (group, time) pairs, estimates local
    ATT(d) or ACRT(d) for each, and aggregates using sample-size-weighted
    influence functions for proper inference.

    Parameters
    ----------
    panel_df : DataFrame with panel data
    id_column : unit identifier column
    time_column : time period column
    outcome_column : outcome variable column
    dose_column : dose/treatment intensity column
    group_column : treatment group/cohort column (0=never-treated, g>0 treated at time g)
    dose_grid : evaluation points for dose-response (default: 20 quantiles of treated dose)
    degree : B-spline degree (default 3 = cubic)
    num_knots : number of interior knots (default 0 = polynomial)
    control_group : "nevertreated" or "notyettreated"
    target : "level" for ATT(d) or "slope" for ACRT(d)
    biters : bootstrap iterations
    alp : significance level
    cband : whether to compute simultaneous confidence band
    boot_seed : random seed for bootstrap

    Returns
    -------
    MultiPeriodDoseResult
    """
    # Validate inputs
    if not isinstance(anticipation, int) or anticipation < 0:
        raise ContDIDValidationError("anticipation must be a non-negative integer")
    if covariates is not None and len(covariates) > 0:
        raise NotImplementedError(
            "Covariate adjustment in multi-period dose estimation is not yet "
            "supported. The theoretical framework (CGBS Theorem 3.1) identifies "
            "LATT(g,t,d|g,d) under unconditional parallel trends; conditional "
            "parallel trends with covariates requires additional theoretical "
            "development not fully covered in the paper for multi-period "
            "decomposition. Use two-period panel estimation with covariates instead."
        )
    if target not in ("level", "slope"):
        raise ContDIDValidationError(f"target must be 'level' or 'slope', got {target!r}")
    if control_group not in ("nevertreated", "notyettreated"):
        raise ContDIDValidationError(
            f"control_group must be 'nevertreated' or 'notyettreated', got {control_group!r}"
        )

    # Coerce dose_grid
    if dose_grid is not None:
        dose_grid_arr = np.asarray(dose_grid, dtype=float)
    else:
        dose_grid_arr = None

    # Identify timing groups
    timing_groups = _identify_dose_timing_groups(
        panel_df,
        group_column=group_column,
        time_column=time_column,
        id_column=id_column,
        dose_column=dose_column,
        control_group=control_group,
        anticipation=anticipation,
    )

    if not timing_groups:
        raise ContDIDValidationError(
            "No valid (group, time) pairs found for multi-period dose estimation. "
            "Check that your panel has treated groups with positive dose and available control units."
        )

    # Default dose grid: quantiles of all treated dose values
    if dose_grid_arr is None:
        all_treated_mask = (panel_df[group_column] > 0) & (panel_df[dose_column] > 0)
        all_treated_dose = panel_df.loc[all_treated_mask, dose_column].values.astype(float)
        if all_treated_dose.size == 0:
            raise ContDIDValidationError("No positive dose values found among treated units.")
        dose_grid_arr = np.quantile(all_treated_dose, np.linspace(0.05, 0.95, 20))

    # Run local estimations for each (g, t) pair
    local_results: list[dict[str, Any]] = []
    local_ifs: list[InfluenceFunction] = []
    local_weights: list[float] = []
    local_point_estimates: list[np.ndarray] = []

    for group_info in timing_groups:
        result = _run_local_dose_estimation(
            panel_df,
            group_info=group_info,
            id_column=id_column,
            time_column=time_column,
            outcome_column=outcome_column,
            dose_column=dose_column,
            group_column=group_column,
            dose_grid=dose_grid_arr,
            degree=degree,
            num_knots=num_knots,
            target=target,
            covariate_columns=covariates,
        )

        if result["status"] == "success":
            local_results.append(result)
            local_ifs.append(result["influence_function"])
            local_weights.append(float(result["n_treated"]))
            local_point_estimates.append(result["point_estimate"])

    if not local_results:
        raise ContDIDValidationError(
            "All (group, time) pairs were skipped due to insufficient data. "
            "Ensure each cohort has at least 2 treated units and 1 control unit "
            "with both pre and post period observations."
        )

    # Aggregate point estimates (sample-size weighted)
    total_weight = sum(local_weights)
    normalized_weights = [w / total_weight for w in local_weights]

    aggregated_point = np.zeros(len(dose_grid_arr))
    for w, pe in zip(normalized_weights, local_point_estimates):
        aggregated_point += w * pe

    # Aggregate influence functions across (g, t) pairs
    aggregated_if = aggregate_influence_functions(local_ifs, local_weights)

    # Inference via multiplier bootstrap on aggregated IF
    boot_result = aggregated_if.multiplier_bootstrap(
        biters=biters, alp=alp, cband=cband, seed=boot_seed
    )

    se = np.array(boot_result["std_error"])
    crit = boot_result["critical_value"]

    lower = aggregated_point - crit * se
    upper = aggregated_point + crit * se

    # Build metadata
    metadata: dict[str, Any] = {
        "estimator": "multiperiod_dose_2x2",
        "target": target,
        "control_group": control_group,
        "degree": degree,
        "num_knots": num_knots,
        "n_groups": len(local_results),
        "total_treated": sum(r["n_treated"] for r in local_results),
        "total_control": sum(r["n_control"] for r in local_results),
        "biters": biters,
        "alp": alp,
        "cband": cband,
        "boot_seed": boot_seed,
        "confidence_band_kind": boot_result["confidence_band_kind"],
        "critical_value": crit,
        "basis_type": "bspline",
        "covariates": covariates,
        "n_covariates": len(covariates) if covariates else 0,
        "theoretical_basis": "CGBS_Theorem_3.1_parametric_bspline",
        "theoretical_boundaries": [
            "multi-period asymptotic theory not fully proven for continuous D",
            "CCK/Lepski not supported (two-period only)",
            "covariate adjustment not supported",
        ],
    }

    # Summarize local results for output (without IF to keep it lean)
    local_summaries = [
        {
            "group": r["group"],
            "time": r["time"],
            "event_time": r.get("event_time", r["time"] - r["group"]),
            "n_treated": r["n_treated"],
            "n_control": r["n_control"],
            "weight": w,
        }
        for r, w in zip(local_results, normalized_weights)
    ]

    return MultiPeriodDoseResult(
        dose_grid=tuple(float(d) for d in dose_grid_arr),
        point_estimate=tuple(float(p) for p in aggregated_point),
        standard_error=tuple(float(s) for s in se),
        confidence_band_lower=tuple(float(lb) for lb in lower),
        confidence_band_upper=tuple(float(ub) for ub in upper),
        aggregated_influence=aggregated_if,
        local_results=local_summaries,
        metadata=metadata,
    )
