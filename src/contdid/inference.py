"""Shared bootstrap and confidence-band inference helpers.

Provides multiplier bootstrap for simultaneous confidence bands and
pointwise intervals, used by both dose-response and event-study estimators.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from .results import ContDIDResult
from .specs import ContDIDSpec
from .validation import ContDIDValidationError


# Seed semantics: uses numpy.random.SeedSequence.spawn() to derive per-chunk
# child seeds. This ensures identical results regardless of thread scheduling
# order. Note: results differ from pre-parallelization sequential RNG draws.
_BOOTSTRAP_SEED = 20260407
_BOOTSTRAP_MAX_DRAW_CELLS = 1_000_000
_NORMAL = NormalDist()
_PARALLEL_BITERS_THRESHOLD = 200  # biters below this use serial execution


def _inference_mode(bootstrap_type: str) -> str:
    return "analytic" if bootstrap_type == "analytic" else "bootstrap"


def _confidence_band_kind(*, bootstrap_type: str, cband: bool) -> str:
    if bootstrap_type == "analytic":
        return "pointwise_analytic"
    return "simultaneous_multiplier" if cband else "pointwise_multiplier"


def _checked_covariance_eigendecomposition(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_covariance = np.asarray(covariance, dtype=object)
    if raw_covariance.ndim != 2 or raw_covariance.shape[0] != raw_covariance.shape[1]:
        raise ValueError("covariance matrix must be square")
    try:
        covariance = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("covariance matrix must contain only finite values") from None
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix must be square")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance matrix must contain only finite values")

    symmetric = (covariance + covariance.T) / 2.0
    covariance_scale = float(np.max(np.abs(symmetric))) if symmetric.size else 0.0
    rank_tolerance = np.finfo(float).eps * max(1.0, covariance_scale) * max(1, symmetric.shape[0])
    if not np.allclose(covariance, symmetric, rtol=0.0, atol=rank_tolerance):
        raise ValueError("covariance matrix must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if min_eigenvalue < -rank_tolerance:
        raise ValueError("covariance matrix must be positive semidefinite")

    clipped = np.where(eigenvalues <= rank_tolerance, 0.0, eigenvalues)
    return symmetric, eigenvectors, clipped


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    _, eigenvectors, clipped = _checked_covariance_eigendecomposition(covariance)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def _variance_zero_tolerance(loadings: np.ndarray, covariance: np.ndarray) -> float:
    covariance_scale = float(np.max(np.abs(covariance))) if covariance.size else 0.0
    loading_scale = float(np.max(np.sum(np.abs(loadings), axis=1))) if loadings.size else 0.0
    return np.finfo(float).eps * max(1.0, covariance_scale) * max(1.0, loading_scale**2)


def _compute_standard_error(loadings: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    raw = np.einsum("ij,jk,ik->i", loadings, covariance, loadings)
    raw = np.where(np.abs(raw) <= _variance_zero_tolerance(loadings, covariance), 0.0, raw)
    return np.sqrt(np.clip(raw, a_min=0.0, a_max=None))


def _bootstrap_draw_chunk_size(
    *,
    biters: int,
    factor_columns: int,
    estimand_count: int,
) -> int:
    row_cells = max(1, 2 * int(factor_columns) + int(estimand_count))
    return max(1, min(int(biters), _BOOTSTRAP_MAX_DRAW_CELLS // row_cells))


def _bootstrap_chunk_worker(
    *,
    seed_child: np.random.SeedSequence,
    chunk_biters: int,
    factor: np.ndarray,
    loadings: np.ndarray,
    std_error: np.ndarray,
    boot_type: str,
) -> np.ndarray:
    """Compute max-statistics for one chunk of bootstrap draws.

    Each chunk uses an independent child seed derived from SeedSequence.spawn(),
    ensuring deterministic results regardless of execution order.
    """
    rng = np.random.default_rng(seed_child)
    draw_shape = (chunk_biters, factor.shape[1])

    if boot_type == "rademacher":
        shocks = rng.choice([-1.0, 1.0], size=draw_shape)
    elif boot_type == "mammen":
        _sqrt5 = np.sqrt(5.0)
        _a = (1.0 - _sqrt5) / 2.0
        _b = (1.0 + _sqrt5) / 2.0
        _prob_a = (_sqrt5 + 1.0) / (2.0 * _sqrt5)
        shocks = np.where(rng.random(draw_shape) < _prob_a, _a, _b)
    else:
        shocks = rng.standard_normal(size=draw_shape)

    coefficient_draws = shocks @ factor.T
    statistic_draws = coefficient_draws @ loadings.T
    standardized = np.divide(
        np.abs(statistic_draws),
        std_error,
        out=np.zeros_like(statistic_draws),
        where=std_error > 0.0,
    )
    return standardized.max(axis=1)


def _run_bootstrap_draws(
    *,
    biters: int,
    factor: np.ndarray,
    loadings: np.ndarray,
    std_error: np.ndarray,
    boot_type: str,
    seed: int,
) -> np.ndarray:
    """Run bootstrap draws with optional thread-level parallelism.

    Uses SeedSequence.spawn() to derive independent per-chunk seeds, ensuring
    reproducibility regardless of whether execution is serial or parallel.
    When biters >= _PARALLEL_BITERS_THRESHOLD and multiple chunks exist,
    uses ThreadPoolExecutor (numpy releases GIL during matrix operations).
    Falls back to serial execution for small problems or on failure.
    """
    chunk_size = _bootstrap_draw_chunk_size(
        biters=biters,
        factor_columns=factor.shape[1],
        estimand_count=loadings.shape[0],
    )

    # Compute chunk boundaries
    num_chunks = (biters + chunk_size - 1) // chunk_size
    chunk_sizes: list[int] = []
    for i in range(num_chunks):
        start = i * chunk_size
        stop = min(start + chunk_size, biters)
        chunk_sizes.append(stop - start)

    # Derive deterministic per-chunk seeds via SeedSequence
    parent_seq = np.random.SeedSequence(seed)
    child_seeds = parent_seq.spawn(num_chunks)

    # Decide whether to use parallel execution
    use_parallel = biters >= _PARALLEL_BITERS_THRESHOLD and num_chunks > 1

    if use_parallel:
        max_workers = min(num_chunks, os.cpu_count() or 1)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _bootstrap_chunk_worker,
                        seed_child=child_seeds[i],
                        chunk_biters=chunk_sizes[i],
                        factor=factor,
                        loadings=loadings,
                        std_error=std_error,
                        boot_type=boot_type,
                    )
                    for i in range(num_chunks)
                ]
                chunks = [f.result() for f in futures]
        except Exception:
            # Graceful fallback to serial on any threading failure
            chunks = [
                _bootstrap_chunk_worker(
                    seed_child=child_seeds[i],
                    chunk_biters=chunk_sizes[i],
                    factor=factor,
                    loadings=loadings,
                    std_error=std_error,
                    boot_type=boot_type,
                )
                for i in range(num_chunks)
            ]
    else:
        chunks = [
            _bootstrap_chunk_worker(
                seed_child=child_seeds[i],
                chunk_biters=chunk_sizes[i],
                factor=factor,
                loadings=loadings,
                std_error=std_error,
                boot_type=boot_type,
            )
            for i in range(num_chunks)
        ]

    return np.concatenate(chunks)


def estimate_mean_variance(values: Iterable[float]) -> float:
    """Estimate the variance of a sample mean from independent unit-level values.

    Computes Var(mean) = s^2 / n using the unbiased sample variance s^2.

    Parameters
    ----------
    values : iterable of float
        One-dimensional numeric sample.

    Returns
    -------
    float
        Estimated variance of the sample mean.

    Raises
    ------
    ValueError
        If values are empty, non-numeric, or contain booleans/NaN.
    """

    try:
        raw_values = list(values)
    except TypeError:
        raise ValueError("values must be a one-dimensional sample") from None
    if _contains_bool_values(raw_values):
        raise ValueError("values must contain only finite non-boolean values")
    try:
        array = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("values must contain only finite non-boolean values") from None
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sample")
    if array.size == 0:
        raise ValueError("values must contain at least one observation")
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite non-boolean values")
    if array.size <= 1:
        return 0.0
    return float(np.var(array, ddof=1) / array.size)


def append_independent_mean_variance(
    loadings: np.ndarray,
    covariance: np.ndarray,
    *,
    mean_variance: float,
    loading_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Append one independent mean component to a linear inference problem.

    Augments the loading matrix and covariance to account for an independent
    mean-estimation variance (e.g., the untreated benchmark for ATT level).

    Parameters
    ----------
    loadings : numpy.ndarray
        Shape (n_estimands, p) loading matrix.
    covariance : numpy.ndarray
        Shape (p, p) covariance matrix of coefficients.
    mean_variance : float
        Scalar variance of the independent mean.
    loading_value : float
        The loading coefficient for the new component.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple of (augmented_loadings, augmented_covariance).

    Raises
    ------
    ValueError
        If inputs have incompatible dimensions or invalid values.
    """

    if (
        isinstance(mean_variance, (bool, np.bool_))
        or not isinstance(mean_variance, Real)
        or not np.isfinite(float(mean_variance))
        or float(mean_variance) < 0.0
    ):
        raise ValueError("mean_variance must be a finite nonnegative scalar")
    if (
        isinstance(loading_value, (bool, np.bool_))
        or not isinstance(loading_value, Real)
        or not np.isfinite(float(loading_value))
    ):
        raise ValueError("loading_value must be a finite scalar")

    raw_loadings = np.asarray(loadings, dtype=object)
    if raw_loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    try:
        checked_loadings = np.asarray(loadings, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("loadings must contain only finite values") from None
    checked_covariance, _, _ = _checked_covariance_eigendecomposition(covariance)
    if checked_loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    if checked_loadings.shape[1] != checked_covariance.shape[0]:
        raise ValueError("loadings and covariance dimensions must align")
    if not np.isfinite(checked_loadings).all():
        raise ValueError("loadings must contain only finite values")

    if float(mean_variance) == 0.0:
        return loadings, covariance

    augmented_loadings = np.column_stack(
        [checked_loadings, np.full(checked_loadings.shape[0], loading_value)]
    )
    augmented_covariance = np.zeros(
        (checked_covariance.shape[0] + 1, checked_covariance.shape[1] + 1), dtype=float
    )
    augmented_covariance[:-1, :-1] = checked_covariance
    augmented_covariance[-1, -1] = float(mean_variance)
    return augmented_loadings, augmented_covariance


def _build_interval_payload(
    estimate: np.ndarray,
    std_error: np.ndarray,
    critical_value: float,
) -> list[list[float]]:
    return [
        [float(point - critical_value * se), float(point + critical_value * se)]
        for point, se in zip(estimate, std_error)
    ]


def _contains_bool_values(values: Any) -> bool:
    array = np.asarray(values, dtype=object)
    return any(isinstance(value, (bool, np.bool_)) for value in array.ravel())


def _coerce_curve_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    try:
        raw_values: object = list(values)
    except TypeError:
        raw_values = values
    if _contains_bool_values(raw_values):
        if name == "std_error":
            raise ValueError("std_error must contain only finite non-boolean nonnegative values")
        raise ValueError("estimate must contain only finite non-boolean values")
    try:
        return np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError):
        if name == "std_error":
            raise ValueError(
                "std_error must contain only finite non-boolean nonnegative values"
            ) from None
        raise ValueError("estimate must contain only finite non-boolean values") from None


def build_confidence_band(
    curve_estimate: Iterable[float],
    std_error: Iterable[float],
    *,
    critical_value: float,
) -> dict[str, list[float] | float]:
    """Build a lower/upper envelope for a curve under a scalar critical value.

    Constructs confidence band boundaries as:
        lower = estimate - critical_value * std_error
        upper = estimate + critical_value * std_error

    Parameters
    ----------
    curve_estimate : iterable of float
        Point estimates at grid points.
    std_error : iterable of float
        Standard errors at grid points.
    critical_value : float
        Scalar critical value (e.g., from bootstrap).

    Returns
    -------
    dict
        Dict with keys 'lower', 'upper' (lists of float), and 'critical_value'.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or invalid values.
    """

    estimate = _coerce_curve_vector(curve_estimate, name="estimate")
    se = _coerce_curve_vector(std_error, name="std_error")
    if estimate.ndim != 1 or se.ndim != 1:
        raise ValueError("estimate and std_error must be one-dimensional")
    if estimate.shape != se.shape:
        raise ValueError("estimate and std_error must have the same shape")
    if estimate.size == 0:
        raise ValueError("estimate and std_error must contain at least one value")
    if not np.isfinite(estimate).all():
        raise ValueError("estimate must contain only finite non-boolean values")
    if not np.isfinite(se).all() or np.any(se < 0.0):
        raise ValueError("std_error must contain only finite non-boolean nonnegative values")
    if (
        isinstance(critical_value, bool)
        or not isinstance(critical_value, Real)
        or not np.isfinite(float(critical_value))
        or float(critical_value) < 0.0
    ):
        raise ValueError("critical_value must be a finite non-boolean nonnegative scalar")
    return {
        "lower": [float(value) for value in (estimate - critical_value * se).tolist()],
        "upper": [float(value) for value in (estimate + critical_value * se).tolist()],
        "critical_value": float(critical_value),
    }


def _validate_inference_controls(
    *,
    alp: float,
    bstrap: bool,
    cband: bool,
    boot_type: str,
    biters: int,
) -> tuple[float, bool, bool, str, int]:
    if isinstance(alp, bool) or not isinstance(alp, Real):
        raise ContDIDValidationError("alp must lie strictly between 0 and 1")
    checked_alp = float(alp)
    if not np.isfinite(checked_alp) or not 0.0 < checked_alp < 1.0:
        raise ContDIDValidationError("alp must lie strictly between 0 and 1")

    if not isinstance(bstrap, bool):
        raise ContDIDValidationError("bstrap must be a boolean")
    if not isinstance(cband, bool):
        raise ContDIDValidationError("cband must be a boolean")
    if boot_type not in ("multiplier", "rademacher", "mammen"):
        raise ContDIDValidationError(
            f"boot_type must be 'multiplier', 'rademacher', or 'mammen'; got {boot_type!r}"
        )
    if isinstance(biters, bool) or not isinstance(biters, Integral):
        raise ContDIDValidationError("biters must be a positive integer")
    checked_biters = int(biters)
    if checked_biters <= 0:
        raise ContDIDValidationError("biters must be a positive integer")

    return checked_alp, bstrap, cband, boot_type, checked_biters


def _validate_bootstrap_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ContDIDValidationError("seed must be a nonnegative integer")
    checked_seed = int(seed)
    if checked_seed < 0:
        raise ContDIDValidationError("seed must be a nonnegative integer")
    return checked_seed


def compute_multiplier_bootstrap(
    loadings: np.ndarray,
    covariance: np.ndarray,
    *,
    alp: float,
    bstrap: bool,
    cband: bool,
    boot_type: str,
    biters: int,
    seed: int = _BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute standard errors and critical values via multiplier bootstrap.

    Implements the Gaussian (or Rademacher/Mammen) multiplier bootstrap for
    simultaneous confidence band construction. Uses thread-level parallelism
    for large bootstrap iterations (numpy releases GIL during matrix ops).

    Reproducibility is guaranteed by SeedSequence.spawn(): each chunk receives
    a deterministic child seed independent of execution order.

    Parameters
    ----------
    loadings : numpy.ndarray
        Shape (n_estimands, p) loading matrix mapping coefficients to estimands.
    covariance : numpy.ndarray
        Shape (p, p) coefficient covariance matrix.
    alp : float
        Significance level (0 < alp < 1).
    bstrap : bool
        Whether to use bootstrap (False returns analytic z-critical).
    cband : bool
        Whether to compute simultaneous band (False returns pointwise).
    boot_type : str
        Weight distribution: "multiplier", "rademacher", or "mammen".
    biters : int
        Number of bootstrap replications.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dict with keys: bootstrap_type, bootstrap_seed, std_error,
        pointwise_critical_value, critical_value, confidence_band_kind.

    Raises
    ------
    ContDIDValidationError
        If inference parameters are invalid.
    ValueError
        If loadings/covariance have incompatible dimensions.

    Notes
    -----
    For biters >= 200 with multiple chunks, computation is parallelized
    using ThreadPoolExecutor. Reproducibility is guaranteed via
    numpy.random.SeedSequence.spawn() -- same seed always produces
    identical results regardless of thread scheduling.
    """

    seed = _validate_bootstrap_seed(seed)
    alp, bstrap, cband, boot_type, biters = _validate_inference_controls(
        alp=alp,
        bstrap=bstrap,
        cband=cband,
        boot_type=boot_type,
        biters=biters,
    )
    raw_loadings = np.asarray(loadings, dtype=object)
    if raw_loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    try:
        loadings = np.asarray(loadings, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("loadings must contain only finite values") from None
    covariance, _, _ = _checked_covariance_eigendecomposition(covariance)
    if loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    if loadings.shape[0] == 0:
        raise ValueError("loadings must contain at least one estimand row")
    if loadings.shape[1] != covariance.shape[0]:
        raise ValueError("loadings and covariance dimensions must align")
    if not np.isfinite(loadings).all():
        raise ValueError("loadings must contain only finite values")

    std_error = _compute_standard_error(loadings, covariance)
    pointwise_critical_value = float(_NORMAL.inv_cdf(1.0 - alp / 2.0))

    if not bstrap:
        return {
            "bootstrap_type": "analytic",
            "bootstrap_seed": None,
            "std_error": std_error.tolist(),
            "pointwise_critical_value": pointwise_critical_value,
            "critical_value": pointwise_critical_value,
            "confidence_band_kind": _confidence_band_kind(
                bootstrap_type="analytic",
                cband=cband,
            ),
        }
    if not cband:
        return {
            "bootstrap_type": boot_type,
            "bootstrap_seed": None,
            "std_error": std_error.tolist(),
            "pointwise_critical_value": pointwise_critical_value,
            "critical_value": pointwise_critical_value,
            "confidence_band_kind": _confidence_band_kind(
                bootstrap_type=boot_type,
                cband=cband,
            ),
        }
    if np.all(std_error == 0.0):
        return {
            "bootstrap_type": boot_type,
            "bootstrap_seed": seed,
            "std_error": std_error.tolist(),
            "pointwise_critical_value": pointwise_critical_value,
            "critical_value": 0.0,
            "confidence_band_kind": _confidence_band_kind(
                bootstrap_type=boot_type,
                cband=cband,
            ),
        }

    factor = _covariance_factor(covariance)
    max_stat = _run_bootstrap_draws(
        biters=biters,
        factor=factor,
        loadings=loadings,
        std_error=std_error,
        boot_type=boot_type,
        seed=seed,
    )

    critical_value = float(np.quantile(max_stat, 1.0 - alp))

    return {
        "bootstrap_type": boot_type,
        "bootstrap_seed": seed,
        "std_error": std_error.tolist(),
        "pointwise_critical_value": pointwise_critical_value,
        "critical_value": critical_value,
        "confidence_band_kind": _confidence_band_kind(
            bootstrap_type=boot_type,
            cband=cband,
        ),
    }


def attach_inference_payload(
    result: ContDIDResult,
    *,
    loadings: np.ndarray,
    covariance: np.ndarray,
    spec: ContDIDSpec,
) -> ContDIDResult:
    """Attach standard errors and confidence interval/band payloads to a result.

    Runs the multiplier bootstrap using the spec's inference parameters and
    mutates the result in-place with SE, CI, and confidence band fields.

    Parameters
    ----------
    result : ContDIDResult
        The ContDIDResult to augment (modified in place).
    loadings : numpy.ndarray
        Shape (n_estimands, p) loading matrix.
    covariance : numpy.ndarray
        Shape (p, p) coefficient covariance matrix.
    spec : ContDIDSpec
        ContDIDSpec with inference parameters (alp, bstrap, cband, etc.).

    Returns
    -------
    ContDIDResult
        The same result object with inference fields populated.

    Raises
    ------
    ValueError
        If loadings/covariance dimensions are incompatible with result.
    """

    try:
        estimate = np.asarray(result.estimate, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("result estimates must be a finite one-dimensional vector") from None
    try:
        grid = np.asarray(result.grid, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("result grid must be a finite one-dimensional vector") from None
    raw_loadings = np.asarray(loadings, dtype=object)
    if raw_loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    try:
        checked_loadings = np.asarray(loadings, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("loadings must contain only finite values") from None
    if estimate.ndim != 1 or not np.isfinite(estimate).all():
        raise ValueError("result estimates must be a finite one-dimensional vector")
    if estimate.size == 0:
        raise ValueError("result estimates must contain at least one value")
    if grid.ndim != 1 or not np.isfinite(grid).all():
        raise ValueError("result grid must be a finite one-dimensional vector")
    if grid.shape != estimate.shape:
        raise ValueError(
            "result grid must match result estimate length before attaching inference"
        )
    if checked_loadings.ndim != 2:
        raise ValueError("loadings must be a two-dimensional matrix")
    if checked_loadings.shape[0] != estimate.shape[0]:
        raise ValueError(
            "loadings row count must match result estimate length before attaching inference"
        )
    bootstrap = compute_multiplier_bootstrap(
        checked_loadings,
        covariance,
        alp=spec.alp,
        bstrap=spec.bstrap,
        cband=spec.cband,
        boot_type=spec.boot_type,
        biters=spec.biters,
    )
    std_error = np.asarray(bootstrap["std_error"], dtype=float)
    if std_error.shape != estimate.shape:
        raise ValueError(
            "loadings row count must match result estimate length before attaching inference"
        )
    normalized_grid = [float(value) for value in grid.tolist()]  # type: ignore[arg-type]
    normalized_estimate = [float(value) for value in estimate.tolist()]  # type: ignore[arg-type]
    pointwise_interval = _build_interval_payload(
        estimate,
        std_error,
        bootstrap["pointwise_critical_value"],
    )
    confidence_band = build_confidence_band(
        estimate,
        std_error,
        critical_value=float(bootstrap["critical_value"]),
    )

    result.grid = normalized_grid
    result.estimate = normalized_estimate
    result.std_error = [float(value) for value in std_error.tolist()]  # type: ignore[arg-type]
    result.critical_value = float(bootstrap["critical_value"])
    result.confidence_interval = pointwise_interval
    result.confidence_band = confidence_band
    result.metadata.update(
        {
            "estimand": result.estimand,
            "grid": result.grid,
            "estimate": result.estimate,
            "std_error": result.std_error,
            "inference": _inference_mode(str(bootstrap["bootstrap_type"])),
            "alp": spec.alp,
            "bstrap": spec.bstrap,
            "cband": spec.cband,
            "biters": spec.biters,
            "bootstrap_seed": bootstrap["bootstrap_seed"],
            "bootstrap_type": bootstrap["bootstrap_type"],
            "confidence_band_kind": bootstrap["confidence_band_kind"],
            "critical_value": result.critical_value,
            "confidence_interval": pointwise_interval,
            "confidence_band": confidence_band,
        }
    )
    return result
