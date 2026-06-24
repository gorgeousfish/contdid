"""Lepski adaptive sieve dimension selection for CCK estimation.

.. note::
    This module is internal and not part of the public API.
    Import directly from ``contdid.lepski`` for advanced use.
    The adaptive parameter in ``run_cck_backend`` remains available
    but is not publicly documented.

Implements the data-driven dimension selection from Chen, Christensen &
Kankanala (2024) "Adaptive Estimation and Uniform Confidence Bands for
Nonparametric Structural Functions and Elasticities". The algorithm selects
the optimal B-spline basis dimension by comparing estimates across candidate
dimensions using multiplier bootstrap critical values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bspline import (
    build_bspline_design,
    build_bspline_derivative_design,
    quantile_knots,
)
from .validation import ContDIDValidationError


@dataclass(slots=True)
class LepskiResult:
    """Result of Lepski adaptive dimension selection.

    Attributes
    ----------
    selected_dimension : int
        The data-driven optimal sieve dimension K_hat.
    candidate_dimensions : tuple of int
        The candidate dimensions that were searched.
    fitted_values : ndarray of shape (grid_size,)
        Estimated function values at eval_grid using selected dimension.
    derivative_values : ndarray of shape (grid_size,) or None
        Estimated first derivative at eval_grid (if computed).
    standard_errors : ndarray of shape (grid_size,)
        Standard errors at eval_grid for the selected dimension.
    eval_grid : ndarray of shape (grid_size,)
        The evaluation grid points.
    bootstrap_critical_value : float
        The multiplier bootstrap critical value theta*_{1-alpha_hat}.
    alpha_hat : float
        The adaptive significance level.
    contrast_statistics : dict
        Diagnostic: max contrast statistic for each candidate K.
    metadata : dict
        Additional diagnostic information.
    """

    selected_dimension: int
    candidate_dimensions: tuple[int, ...]
    fitted_values: np.ndarray
    derivative_values: np.ndarray | None
    standard_errors: np.ndarray
    eval_grid: np.ndarray
    bootstrap_critical_value: float
    alpha_hat: float
    contrast_statistics: dict[int, float]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_candidate_grid(
    n: int, degree: int, k_min: int | None = None, k_max: int | None = None
) -> list[int]:
    """Build dyadic candidate dimension grid.

    Grid: T = {2^l + degree : l = 0, 1, 2, ...}

    Determine K_max_hat using the rule:
        K_max_hat = min{K in T : K * sqrt(log K) <= 10 * sqrt(n)}
    (simplified from the full formula; v_n is set to 1 for the basic version)

    Search range: {K in T : max(k_min, 0.1*(log K_max_hat)^2) <= K <= K_max_hat}

    Parameters
    ----------
    n : int
        Sample size.
    degree : int
        B-spline degree (e.g., 3 for cubic).
    k_min : int or None
        Minimum dimension (default: degree + 1).
    k_max : int or None
        Maximum dimension override (default: determined by formula).

    Returns
    -------
    list of int
        Candidate dimensions in ascending order.
    """
    if k_min is None:
        k_min = degree + 1

    # Validate: k_min must not exceed k_max
    if k_max is not None and k_min > k_max:
        raise ValueError(f"k_min ({k_min}) must be <= k_max ({k_max})")

    # Short-circuit: when k_min == k_max, return that single dimension directly
    if k_max is not None and k_min == k_max:
        k = k_min
        # Validate the requested dimension
        if k < 1:
            raise ContDIDValidationError(
                f"Requested dimension k_min=k_max={k} must be a positive integer."
            )
        if k < degree + 1:
            raise ContDIDValidationError(
                f"Requested dimension {k} is less than minimum "
                f"(degree + 1 = {degree + 1}). "
                f"Dimension must be >= {degree + 1} for degree-{degree} B-splines."
            )
        # Verify the single dimension is feasible for this sample size
        if 2 * k > n:
            raise ContDIDValidationError(
                f"Requested dimension k_min=k_max={k} requires at least "
                f"2*{k}={2 * k} observations, but n={n}. "
                f"Reduce k_min/k_max or increase sample size."
            )
        return [k]

    # Build dyadic grid: 2^level + degree for level = 0, 1, 2, ...
    max_l = max(1, int(math.ceil(math.log2(max(n, 2)))))
    full_grid = [2**level + degree for level in range(0, max_l + 1)]

    # Filter out dimensions that require more parameters than we can fit
    # Need n >= 2*K for reasonable estimation
    full_grid = [k for k in full_grid if 2 * k <= n]
    if not full_grid:
        # No feasible dimensions at all for this sample size
        raise ContDIDValidationError(
            f"No feasible candidate dimensions for n={n}, degree={degree}. "
            f"The minimum dimension (degree+1={degree + 1}) requires at least "
            f"{2 * (degree + 1)} observations. Increase sample size."
        )

    # Determine K_max_hat: largest K such that K * sqrt(log K) <= 10 * sqrt(n)
    threshold = 10.0 * math.sqrt(n)
    k_max_hat = full_grid[0]
    for k in full_grid:
        if k * math.sqrt(math.log(max(k, 2))) <= threshold:
            k_max_hat = k
        else:
            break

    # Apply user override for k_max
    if k_max is not None:
        k_max_hat = min(k_max_hat, k_max)

    # Apply user k_min: k_max_hat must be at least k_min for a valid range
    k_max_hat = max(k_max_hat, k_min)

    # Determine lower bound of search
    lower_bound = max(k_min, int(math.ceil(0.1 * math.log(max(k_max_hat, 2)) ** 2)))

    # Filter grid to [lower_bound, k_max_hat], also respecting user k_max
    effective_upper = k_max_hat if k_max is None else min(k_max_hat, k_max)
    candidates = [k for k in full_grid if lower_bound <= k <= effective_upper]

    # Must have at least 2 candidates for the algorithm to work
    if len(candidates) < 2:
        # Try to add more from full_grid, respecting user bounds
        effective_k_max = k_max if k_max is not None else k_max_hat
        valid = [k for k in full_grid if k >= k_min and k <= effective_k_max]
        if len(valid) >= 2:
            candidates = valid
        elif len(valid) == 1:
            # Try to add a neighbor that still respects k_min and effective_k_max
            idx = full_grid.index(valid[0])
            if idx > 0 and full_grid[idx - 1] >= k_min:
                candidates = [full_grid[idx - 1], valid[0]]
            elif idx < len(full_grid) - 1 and full_grid[idx + 1] <= effective_k_max:
                candidates = [valid[0], full_grid[idx + 1]]
            else:
                # Cannot expand without violating bounds; return single candidate
                candidates = valid
        else:
            # No dyadic grid point falls within [k_min, k_max]
            raise ContDIDValidationError(
                f"No feasible candidate dimensions in [{k_min}, {effective_k_max}] "
                f"for n={n}, degree={degree}. "
                f"The dyadic grid points available are {full_grid}. "
                f"Try relaxing k_min/k_max bounds or increasing sample size."
            )

    candidates = sorted(candidates)

    # Final safety check: all candidates must respect user bounds
    if k_max is not None:
        candidates = [k for k in candidates if k <= k_max]
    candidates = [k for k in candidates if k >= k_min]

    if not candidates:
        effective_k_max = k_max if k_max is not None else k_max_hat
        raise ContDIDValidationError(
            f"No feasible candidate dimensions in [{k_min}, {effective_k_max}] "
            f"for n={n}, degree={degree}. "
            f"Try relaxing k_min/k_max bounds or increasing sample size."
        )

    return candidates


def _fit_sieve_at_dimension(
    delta_outcome: np.ndarray,
    dose: np.ndarray,
    dimension: int,
    degree: int,
    eval_grid: np.ndarray,
) -> dict[str, Any]:
    """Fit sieve regression at a given dimension K.

    Uses B-spline basis with (dimension - degree - 1) interior knots
    selected at quantiles of dose.

    Parameters
    ----------
    delta_outcome : ndarray of shape (n,)
        Outcome differences.
    dose : ndarray of shape (n,)
        Treatment dose values.
    dimension : int
        Total basis dimension K.
    degree : int
        B-spline degree.
    eval_grid : ndarray of shape (grid_size,)
        Grid points for evaluation.

    Returns
    -------
    dict with keys:
        'coefficients', 'fitted_at_grid', 'derivative_at_grid',
        'residuals', 'design', 'eval_design', 'eval_deriv_design',
        'bread', 'hat_values'
    """
    num_interior_knots = dimension - degree - 1
    if num_interior_knots < 0:
        num_interior_knots = 0

    interior_knots = quantile_knots(dose, num_interior_knots)

    # Build design matrix
    design = build_bspline_design(dose, degree, interior_knots)

    # OLS coefficients via pinv for numerical stability
    xtx = design.T @ design
    bread = np.linalg.pinv(xtx)
    coefficients = bread @ (design.T @ delta_outcome)

    # Residuals
    residuals = delta_outcome - design @ coefficients

    # Evaluation at grid
    dose_min = float(np.min(dose))
    dose_max = float(np.max(dose))
    eval_design = build_bspline_design(
        eval_grid, degree, interior_knots, xmin=dose_min, xmax=dose_max
    )
    eval_deriv_design = build_bspline_derivative_design(
        eval_grid, degree, interior_knots, xmin=dose_min, xmax=dose_max
    )

    fitted_at_grid = eval_design @ coefficients
    derivative_at_grid = eval_deriv_design @ coefficients

    # Hat values: diag(X @ bread @ X') computed efficiently
    xb = design @ bread  # (n, K)
    hat_values = np.sum(xb * design, axis=1)  # (n,)

    return {
        "coefficients": coefficients,
        "fitted_at_grid": fitted_at_grid,
        "derivative_at_grid": derivative_at_grid,
        "residuals": residuals,
        "design": design,
        "eval_design": eval_design,
        "eval_deriv_design": eval_deriv_design,
        "bread": bread,
        "hat_values": hat_values,
    }


def _compute_pointwise_se(fit: dict[str, Any]) -> np.ndarray:
    """Compute pointwise standard errors at eval_grid for a single fit.

    Uses heteroskedasticity-robust (HC0) sandwich variance:
    Var(h_hat(x)) = psi(x)' (X'X)^{-1} (X' diag(e^2) X) (X'X)^{-1} psi(x)

    Parameters
    ----------
    fit : dict
        Output from ``_fit_sieve_at_dimension``.

    Returns
    -------
    ndarray of shape (grid_size,)
        Pointwise standard errors.
    """
    design = fit["design"]  # (n, K)
    residuals = fit["residuals"]  # (n,)
    bread = fit["bread"]  # (K, K)
    eval_design = fit["eval_design"]  # (G, K)

    # Meat: X' diag(e^2) X
    score = design * residuals[:, None]  # (n, K)
    meat = score.T @ score  # (K, K)

    # Sandwich covariance of coefficients
    sandwich = bread @ meat @ bread  # (K, K)

    # Pointwise variance: psi(x)' @ sandwich @ psi(x) for each grid point
    # = diag(eval_design @ sandwich @ eval_design.T)
    tmp = eval_design @ sandwich  # (G, K)
    pointwise_var = np.sum(tmp * eval_design, axis=1)  # (G,)

    # Clip to non-negative and sqrt
    pointwise_var = np.maximum(pointwise_var, 0.0)
    return np.sqrt(pointwise_var)


def _compute_pairwise_variance(
    fit_k: dict[str, Any], fit_k2: dict[str, Any], n: int
) -> np.ndarray:
    """Compute variance of the contrast h_K(x) - h_K2(x).

    sigma^2_{K,K2}(x) = Var(h_K(x)) + Var(h_K2(x)) - 2*Cov(h_K(x), h_K2(x))

    The cross-covariance is computed exactly using:
    Cov(h_K(x), h_K2(x)) = psi_K(x)' bread_K X_K' diag(e_K * e_K2) X_K2 bread_K2 psi_K2(x)

    Parameters
    ----------
    fit_k : dict
        Fit result for dimension K.
    fit_k2 : dict
        Fit result for dimension K2.
    n : int
        Sample size.

    Returns
    -------
    ndarray of shape (grid_size,)
        Standard deviations of the contrast at each grid point.
    """
    # Variance terms for each fit
    var_k = _compute_pointwise_se(fit_k) ** 2
    var_k2 = _compute_pointwise_se(fit_k2) ** 2

    # Cross-covariance term
    design_k = fit_k["design"]  # (n, K1)
    design_k2 = fit_k2["design"]  # (n, K2)
    bread_k = fit_k["bread"]  # (K1, K1)
    bread_k2 = fit_k2["bread"]  # (K2, K2)
    eval_design_k = fit_k["eval_design"]  # (G, K1)
    eval_design_k2 = fit_k2["eval_design"]  # (G, K2)
    resid_k = fit_k["residuals"]  # (n,)
    resid_k2 = fit_k2["residuals"]  # (n,)

    # cross_product = X_K' diag(e_K * e_K2) X_K2, shape (K1, K2)
    cross_resid = resid_k * resid_k2  # (n,)
    weighted_k = design_k * cross_resid[:, None]  # (n, K1)
    cross_product = weighted_k.T @ design_k2  # (K1, K2)

    # Full cross-covariance at each grid point:
    # cov(x) = eval_k(x) @ bread_k @ cross_product @ bread_k2 @ eval_k2(x)'
    # We need diag of: (eval_k @ bread_k @ cross_product @ bread_k2 @ eval_k2.T)
    left = eval_design_k @ bread_k @ cross_product @ bread_k2  # (G, K2)
    cov_term = np.sum(left * eval_design_k2, axis=1)  # (G,)

    # Contrast variance
    contrast_var = var_k + var_k2 - 2.0 * cov_term

    # Clip to avoid numerical negatives
    var_floor = 1e-10 * max(float(np.max(var_k + var_k2)), 1e-30)
    contrast_var = np.maximum(contrast_var, var_floor)

    return np.sqrt(contrast_var)


def _multiplier_bootstrap_critical_value(
    fits: dict[int, dict[str, Any]],
    candidate_dims: list[int],
    n: int,
    bootstrap_reps: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute multiplier bootstrap critical value for Lepski selection.

    Algorithm:
    1. Compute alpha_hat = min(0.5, sqrt(log(K_max) / K_max))
    2. For each bootstrap replicate b = 1..B:
       a. Draw xi ~ N(0, 1) of shape (n,)
       b. For each pair (K, K2) with K < K2, for each grid point x:
          Compute Z*_n(x, K, K2) = [psi_K(x)' bread_K X_K' diag(xi * resid_K)
                                    - psi_K2(x)' bread_K2 X_K2' diag(xi * resid_K2)]
                                   / sigma_{K,K2}(x)
       c. Take max_stat_b = sup over all (x, K, K2) of |Z*_n|
    3. Critical value = quantile of max_stats at level (1 - alpha_hat)

    Parameters
    ----------
    fits : dict mapping dimension -> fit result dict
    candidate_dims : list of candidate dimensions (sorted)
    n : int
        Sample size.
    bootstrap_reps : int
        Number of multiplier bootstrap replications.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple of (critical_value, alpha_hat)
    """
    k_max = max(candidate_dims)
    alpha_hat = min(0.5, math.sqrt(math.log(max(k_max, 2)) / max(k_max, 2)))

    rng = np.random.default_rng(seed)

    # Pre-compute score matrices and loadings for each K
    # score_K = X_K * resid_K[:, None], shape (n, K_dim)
    # loading_K = eval_design_K @ bread_K, shape (G, K_dim)
    scores: dict[int, np.ndarray] = {}
    loadings: dict[int, np.ndarray] = {}
    for k in candidate_dims:
        fit = fits[k]
        scores[k] = fit["design"] * fit["residuals"][:, None]  # (n, K_dim)
        loadings[k] = fit["eval_design"] @ fit["bread"]  # (G, K_dim)

    # Pre-compute pairwise sigma for normalization
    pair_sigmas: dict[tuple[int, int], np.ndarray] = {}
    for i, k in enumerate(candidate_dims):
        for k2 in candidate_dims[i + 1 :]:
            sigma = _compute_pairwise_variance(fits[k], fits[k2], n)
            pair_sigmas[(k, k2)] = sigma

    # Bootstrap loop
    max_stats = np.empty(bootstrap_reps)

    for b in range(bootstrap_reps):
        xi = rng.standard_normal(n)

        # For each K, compute bootstrap perturbation at grid:
        # boot_K(x) = loading_K @ (score_K.T @ xi)
        boot_values: dict[int, np.ndarray] = {}
        for k in candidate_dims:
            # score_K.T @ xi has shape (K_dim,)
            perturb_coef = scores[k].T @ xi  # (K_dim,)
            boot_values[k] = loadings[k] @ perturb_coef  # (G,)

        # Compute max over all pairs
        current_max = 0.0
        for i, k in enumerate(candidate_dims):
            for k2 in candidate_dims[i + 1 :]:
                contrast = boot_values[k] - boot_values[k2]  # (G,)
                sigma = pair_sigmas[(k, k2)]
                # Standardize
                standardized = np.abs(contrast) / sigma
                pair_max = float(np.max(standardized))
                if pair_max > current_max:
                    current_max = pair_max

        max_stats[b] = current_max

    # Critical value at (1 - alpha_hat) quantile
    critical_value = float(np.quantile(max_stats, 1.0 - alpha_hat))

    return critical_value, alpha_hat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_lepski_dimension(
    delta_outcome: np.ndarray,
    dose: np.ndarray,
    *,
    degree: int = 3,
    k_min: int | None = None,
    k_max: int | None = None,
    eval_grid: np.ndarray | None = None,
    grid_size: int = 50,
    bootstrap_reps: int = 1000,
    tuning_constant: float = 1.1,
    seed: int | None = None,
    n_jobs: int = 1,
) -> LepskiResult:
    """Select optimal sieve dimension using Lepski's adaptive method.

    Implements the data-driven dimension selection from Chen, Christensen &
    Kankanala (2024). The algorithm selects the smallest sieve dimension K
    such that increasing to any larger K2 does not produce a statistically
    significant change in the fitted curve (judged by a multiplier bootstrap
    critical value).

    Parameters
    ----------
    delta_outcome : ndarray of shape (n,)
        First-differenced outcome for treated units.
    dose : ndarray of shape (n,)
        Continuous treatment dose.
    degree : int, default 3
        B-spline polynomial degree.
    k_min : int or None
        Minimum basis dimension to consider.
    k_max : int or None
        Maximum basis dimension to consider.
    eval_grid : ndarray or None
        Grid points for evaluation. If None, uses quantile grid of dose.
    grid_size : int, default 50
        Number of grid points if eval_grid is None.
    bootstrap_reps : int, default 1000
        Number of multiplier bootstrap replications.
    tuning_constant : float, default 1.1
        Multiplicative tuning constant for the selection rule.
    seed : int or None
        Random seed for bootstrap reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs for bootstrap computation. Currently only
        sequential execution (n_jobs=1) is supported. The BLAS-level parallelism
        in NumPy matrix operations already provides implicit multi-threading.
        Explicit parallelization is reserved for future optimization when
        bootstrap computation becomes the identified bottleneck.

    Returns
    -------
    LepskiResult
        Contains selected dimension, fitted values, standard errors, and
        diagnostic information.

    Raises
    ------
    ContDIDValidationError
        If inputs fail validation (too few observations, no dose variation, etc.)
    """
    # --- Validate inputs ---
    delta_outcome = np.asarray(delta_outcome, dtype=float).ravel()
    dose = np.asarray(dose, dtype=float).ravel()
    n = len(delta_outcome)

    if n != len(dose):
        raise ContDIDValidationError(
            f"delta_outcome and dose must have the same length; got {n} and {len(dose)}"
        )
    if n < degree + 2:
        raise ContDIDValidationError(
            f"sample size n={n} is too small for degree={degree}; "
            f"need at least {degree + 2} observations"
        )
    if np.ptp(dose) < 1e-12:
        raise ContDIDValidationError("dose has no variation; cannot fit sieve regression")

    # --- Validate dimension bounds early ---
    effective_k_min = k_min if k_min is not None else degree + 1
    if k_max is not None and effective_k_min > k_max:
        raise ValueError(f"k_min ({effective_k_min}) must be <= k_max ({k_max})")

    # --- Build candidate grid ---
    candidates = _build_candidate_grid(n, degree, k_min=k_min, k_max=k_max)

    # --- Create evaluation grid if needed ---
    if eval_grid is None:
        probs = np.linspace(0.0, 1.0, grid_size)
        eval_grid = np.quantile(dose, probs)
    else:
        eval_grid = np.asarray(eval_grid, dtype=float).ravel()

    # --- Fit sieve at each candidate dimension ---
    fits: dict[int, dict[str, Any]] = {}
    for k in candidates:
        fits[k] = _fit_sieve_at_dimension(delta_outcome, dose, k, degree, eval_grid)

    # --- Handle single candidate case ---
    if len(candidates) == 1:
        selected = candidates[0]
        se = _compute_pointwise_se(fits[selected])
        return LepskiResult(
            selected_dimension=selected,
            candidate_dimensions=tuple(candidates),
            fitted_values=fits[selected]["fitted_at_grid"],
            derivative_values=fits[selected]["derivative_at_grid"],
            standard_errors=se,
            eval_grid=eval_grid,
            bootstrap_critical_value=0.0,
            alpha_hat=0.5,
            contrast_statistics={selected: 0.0},
            metadata={"n": n, "degree": degree, "single_candidate": True},
        )

    # --- Compute bootstrap critical value ---
    critical_value, alpha_hat = _multiplier_bootstrap_critical_value(
        fits, candidates, n, bootstrap_reps=bootstrap_reps, seed=seed
    )

    # --- Apply Lepski selection rule ---
    contrast_statistics: dict[int, float] = {}
    selected_k: int | None = None

    for i, k in enumerate(candidates):
        max_contrast = 0.0
        for k2 in candidates[i + 1 :]:
            sigma = _compute_pairwise_variance(fits[k], fits[k2], n)
            diff = np.abs(fits[k]["fitted_at_grid"] - fits[k2]["fitted_at_grid"])
            contrast = diff / sigma
            sup_contrast = float(np.max(contrast))
            if sup_contrast > max_contrast:
                max_contrast = sup_contrast
        contrast_statistics[k] = max_contrast

        if selected_k is None and max_contrast <= tuning_constant * critical_value:
            selected_k = k

    # Fallback to largest if none selected
    if selected_k is None:
        selected_k = candidates[-1]

    # Conservative bound: don't select the very largest dimension
    if len(candidates) > 1 and selected_k == candidates[-1]:
        selected_k = candidates[-2]

    # --- Final bounds enforcement ---
    # The selected dimension MUST be within user-specified bounds.
    # This is a safety net: if the algorithm or grid construction has a bug,
    # we raise rather than silently returning an out-of-bounds result.
    effective_k_min_final = k_min if k_min is not None else degree + 1
    effective_k_max_final = k_max
    if selected_k < effective_k_min_final:
        raise ContDIDValidationError(
            f"Internal error: Lepski selected dimension {selected_k} is below "
            f"k_min={effective_k_min_final}. This should not happen. "
            f"Candidates were: {candidates}"
        )
    if effective_k_max_final is not None and selected_k > effective_k_max_final:
        raise ContDIDValidationError(
            f"Internal error: Lepski selected dimension {selected_k} exceeds "
            f"k_max={effective_k_max_final}. This should not happen. "
            f"Candidates were: {candidates}"
        )

    # --- Compute standard errors for selected dimension ---
    se = _compute_pointwise_se(fits[selected_k])

    return LepskiResult(
        selected_dimension=selected_k,
        candidate_dimensions=tuple(candidates),
        fitted_values=fits[selected_k]["fitted_at_grid"],
        derivative_values=fits[selected_k]["derivative_at_grid"],
        standard_errors=se,
        eval_grid=eval_grid,
        bootstrap_critical_value=critical_value,
        alpha_hat=alpha_hat,
        contrast_statistics=contrast_statistics,
        metadata={
            "n": n,
            "degree": degree,
            "bootstrap_reps": bootstrap_reps,
            "tuning_constant": tuning_constant,
            "seed": seed,
        },
    )
