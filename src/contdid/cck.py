"""CCK backend wrapper.

This backend serves the supported dose-only predicate through B-spline sieve
estimation. It supports both fixed-dimension and adaptive (Lepski) dimension
selection.

Theoretical scope (arXiv:2107.11869v3 Theorem 2)
-------------------------------------------------
- Provides convergence rate and uniform confidence band guarantees for a SINGLE
  two-period B-spline sieve regression problem.
- When used inside event-study aggregation, each local (g,t) comparison is a
  separate two-period problem where this theory applies directly.
- Adaptive/Lepski dimension selection is valid only for individual two-period
  problems; aggregating estimates with heterogeneous adaptive dimensions across
  event-time cells lacks joint coverage theory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bspline import build_bspline_design, build_bspline_derivative_design
from .inference import (
    append_independent_mean_variance,
    compute_multiplier_bootstrap,
    estimate_mean_variance,
)
from .validation import ContDIDValidationError


def _build_cck_design(
    dose: np.ndarray,
    *,
    degree: int = 2,
    interior_knots: list[float] | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build CCK design matrix using B-spline basis."""
    if interior_knots is None:
        interior_knots = []
    return build_bspline_design(dose, degree, interior_knots, xmin=xmin, xmax=xmax)


def _build_cck_derivative_design(
    dose: np.ndarray,
    *,
    degree: int = 2,
    interior_knots: list[float] | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build CCK derivative design matrix using B-spline basis."""
    if interior_knots is None:
        interior_knots = []
    return build_bspline_derivative_design(dose, degree, interior_knots, xmin=xmin, xmax=xmax)


def _estimate_cck_sandwich_covariance(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.pinv(design.T @ design)
    score = design * residual[:, None]
    meat = score.T @ score
    covariance = xtx_inv @ meat @ xtx_inv
    return (covariance + covariance.T) / 2.0


def _require_identified_cck_design(design: np.ndarray, dose: np.ndarray) -> None:
    basis_columns = int(design.shape[1])
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank >= basis_columns:
        return

    treated_count = int(design.shape[0])
    unique_support = int(np.unique(np.asarray(dose, dtype=float)).size)
    raise ContDIDValidationError(
        "underidentified positive-dose support for the requested cck basis; "
        f"need full column rank but observed rank={design_rank} < columns={basis_columns} "
        f"with treated_count={treated_count} and unique_positive_doses={unique_support}"
    )


def _require_cck_inference_df(design: np.ndarray) -> None:
    treated_count = int(design.shape[0])
    basis_columns = int(design.shape[1])
    residual_df = treated_count - basis_columns
    if residual_df > 0:
        return

    raise ContDIDValidationError(
        "positive-dose treated support leaves no residual degrees of freedom for inference "
        f"on the requested cck basis; need treated_count > columns but observed "
        f"treated_count={treated_count} and columns={basis_columns}"
    )


def _require_untreated_benchmark_variance_df(untreated_delta: np.ndarray) -> None:
    untreated_count = int(np.asarray(untreated_delta, dtype=float).size)
    if untreated_count > 1:
        return

    raise ContDIDValidationError(
        "untreated benchmark variance requires at least two untreated units; "
        f"observed untreated_count={untreated_count}"
    )


def _interval_payload(
    curve: np.ndarray, std_error: np.ndarray, critical_value: float
) -> list[list[float]]:
    return [
        [float(point - critical_value * se), float(point + critical_value * se)]
        for point, se in zip(curve, std_error)
    ]


def _band_payload(
    curve: np.ndarray, std_error: np.ndarray, critical_value: float
) -> dict[str, Any]:
    return {
        "lower": [float(value) for value in (curve - critical_value * std_error).tolist()],  # type: ignore[arg-type, union-attr]
        "upper": [float(value) for value in (curve + critical_value * std_error).tolist()],  # type: ignore[arg-type, union-attr]
        "critical_value": float(critical_value),
    }


def run_cck_backend(
    delta_outcome: np.ndarray,
    dose: np.ndarray,
    dvals: list[float],
    *,
    untreated_delta: np.ndarray,
    require_untreated_variance_df: bool,
    bstrap: bool,
    cband: bool,
    alp: float,
    biters: int = 1000,
    boot_type: str = "multiplier",
    degree: int = 2,
    num_knots: int = 0,
    adaptive: bool = False,
    adaptive_k_min: int | None = None,
    adaptive_k_max: int | None = None,
    adaptive_seed: int | None = None,
    covariates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return ATT/ACRT curves, standard errors, and critical values for CCK.

    Parameters
    ----------
    covariates : (n, p_cov) array or None
        Optional covariate matrix for treated units.  When provided the design
        matrix is extended as [B-spline(dose), covariates] and evaluation
        loadings are zero-padded so that point estimates reflect the dose curve
        only.
    """

    dose = np.asarray(dose, dtype=float)
    delta_outcome = np.asarray(delta_outcome, dtype=float)
    dose_grid = np.asarray(dvals, dtype=float)

    # --- Adaptive dimension selection via Lepski algorithm ---
    # Lepski selects B-spline complexity using dose-only data; covariates are
    # added afterward.
    if adaptive:
        from .lepski import select_lepski_dimension

        lepski_result = select_lepski_dimension(
            delta_outcome=delta_outcome,
            dose=dose,
            degree=degree,
            k_min=adaptive_k_min,
            k_max=adaptive_k_max,
            eval_grid=dose_grid,
            bootstrap_reps=biters,
            seed=adaptive_seed,
        )
        # Use the Lepski-selected dimension
        num_knots = max(0, lepski_result.selected_dimension - degree - 1)

        # Store Lepski info for metadata
        _lepski_metadata: dict[str, Any] | None = {
            "adaptive": True,
            "selected_dimension": lepski_result.selected_dimension,
            "candidate_dimensions": lepski_result.candidate_dimensions,
            "bootstrap_critical_value": lepski_result.bootstrap_critical_value,
            "alpha_hat": lepski_result.alpha_hat,
        }
    else:
        _lepski_metadata = None

    # Select interior knots if requested
    if num_knots > 0:
        from .bspline import quantile_knots

        interior_knots = quantile_knots(dose, num_knots)
    else:
        interior_knots = []

    design = _build_cck_design(dose, degree=degree, interior_knots=interior_knots)

    # Extend design with covariates if provided
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        p_cov = covariates.shape[1]
        design = np.column_stack([design, covariates])
    else:
        p_cov = 0

    _require_identified_cck_design(design, dose)
    _require_cck_inference_df(design)
    if require_untreated_variance_df:
        _require_untreated_benchmark_variance_df(untreated_delta)
    coefficients, *_ = np.linalg.lstsq(design, delta_outcome, rcond=None)
    fitted = design @ coefficients
    residual = delta_outcome - fitted
    covariance = _estimate_cck_sandwich_covariance(design, residual)

    # Use fitting dose boundaries for grid evaluation
    _dose_xmin = float(np.min(dose))
    _dose_xmax = float(np.max(dose))

    # Build dose-only loadings then zero-pad for covariates
    att_loadings_dose = _build_cck_design(
        dose_grid,
        degree=degree,
        interior_knots=interior_knots,
        xmin=_dose_xmin,
        xmax=_dose_xmax,
    )
    acrt_loadings_dose = _build_cck_derivative_design(
        dose_grid,
        degree=degree,
        interior_knots=interior_knots,
        xmin=_dose_xmin,
        xmax=_dose_xmax,
    )
    if p_cov > 0:
        _zero_pad = np.zeros((len(dose_grid), p_cov))
        att_loadings = np.column_stack([att_loadings_dose, _zero_pad])
        acrt_loadings = np.column_stack([acrt_loadings_dose, _zero_pad])
    else:
        att_loadings = att_loadings_dose
        acrt_loadings = acrt_loadings_dose

    att_curve = att_loadings @ coefficients
    acrt_curve = acrt_loadings @ coefficients

    # Overall ATT/ACRT: average over treated units (dose part only)
    dose_coef = coefficients[: design.shape[1] - p_cov] if p_cov > 0 else coefficients
    design_dose_only = _build_cck_design(dose, degree=degree, interior_knots=interior_knots)
    overall_att = float(np.mean(design_dose_only @ dose_coef))
    deriv_dose_only = _build_cck_derivative_design(
        dose, degree=degree, interior_knots=interior_knots
    )
    overall_acrt = float(np.mean(deriv_dose_only @ dose_coef))

    # Scalar SEs for overall ATT/ACRT (used by event-study aggregation).
    # These are the SE of the mean prediction over the treated dose distribution.
    att_summary_loading = np.mean(design_dose_only, axis=0)
    if p_cov > 0:
        att_summary_loading_full = np.concatenate([att_summary_loading, np.zeros(p_cov)])
    else:
        att_summary_loading_full = att_summary_loading
    overall_att_var = float(att_summary_loading_full @ covariance @ att_summary_loading_full)
    # Add untreated variance component for level estimation
    if untreated_delta.size > 1:
        overall_att_var += float(np.var(untreated_delta, ddof=1) / untreated_delta.size)
    overall_att_se = float(np.sqrt(max(overall_att_var, 0.0)))

    acrt_summary_loading = np.mean(deriv_dose_only, axis=0)
    if p_cov > 0:
        acrt_summary_loading_full = np.concatenate([acrt_summary_loading, np.zeros(p_cov)])
    else:
        acrt_summary_loading_full = acrt_summary_loading
    overall_acrt_var = float(acrt_summary_loading_full @ covariance @ acrt_summary_loading_full)
    overall_acrt_se = float(np.sqrt(max(overall_acrt_var, 0.0)))
    att_inference_loadings, att_covariance = append_independent_mean_variance(
        att_loadings,
        covariance,
        mean_variance=estimate_mean_variance(untreated_delta),
        loading_value=-1.0,
    )

    att_bootstrap = compute_multiplier_bootstrap(
        att_inference_loadings,
        att_covariance,
        alp=alp,
        bstrap=bstrap,
        cband=cband,
        boot_type=boot_type,
        biters=biters,
    )
    acrt_bootstrap = compute_multiplier_bootstrap(
        acrt_loadings,
        covariance,
        alp=alp,
        bstrap=bstrap,
        cband=cband,
        boot_type=boot_type,
        biters=biters,
    )

    att_se = np.asarray(att_bootstrap["std_error"], dtype=float)
    acrt_se = np.asarray(acrt_bootstrap["std_error"], dtype=float)

    result: dict[str, Any] = {
        "att_curve": [float(value) for value in att_curve.tolist()],  # type: ignore[arg-type]
        "att_se": [float(value) for value in att_se.tolist()],  # type: ignore[arg-type]
        "att_crit": float(att_bootstrap["critical_value"]),
        "att_interval": _interval_payload(
            att_curve, att_se, float(att_bootstrap["pointwise_critical_value"])
        ),
        "att_band": _band_payload(att_curve, att_se, float(att_bootstrap["critical_value"])),
        "acrt_curve": [float(value) for value in acrt_curve.tolist()],  # type: ignore[arg-type]
        "acrt_se": [float(value) for value in acrt_se.tolist()],  # type: ignore[arg-type]
        "acrt_crit": float(acrt_bootstrap["critical_value"]),
        "acrt_interval": _interval_payload(
            acrt_curve, acrt_se, float(acrt_bootstrap["pointwise_critical_value"])
        ),
        "acrt_band": _band_payload(acrt_curve, acrt_se, float(acrt_bootstrap["critical_value"])),
        "overall_att": overall_att,
        "overall_acrt": overall_acrt,
        "overall_att_se": overall_att_se,
        "overall_acrt_se": overall_acrt_se,
        "bootstrap_type": att_bootstrap["bootstrap_type"],
        "att_bootstrap_seed": att_bootstrap["bootstrap_seed"],
        "acrt_bootstrap_seed": acrt_bootstrap["bootstrap_seed"],
        "confidence_band_kind": att_bootstrap["confidence_band_kind"],
    }
    if _lepski_metadata is not None:
        result["lepski"] = _lepski_metadata
    return result
