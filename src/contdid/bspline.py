"""B-spline basis function construction for dose-response estimation.

Numerical properties
--------------------
This module uses scipy.interpolate.BSpline which implements the numerically
stable Cox-de Boor recursion (NOT truncated power series). Key guarantees:

- Partition of unity: sum of basis values = 1 at any point (error < 1e-12)
- Non-negativity: all basis values >= 0
- Local support: each basis function is nonzero on at most degree+1 intervals
- Condition number: typical cond(X'X) ~ 30 (vs 1000+ for truncated power)
- Derivative accuracy: analytic derivatives via BSpline.derivative() match
  numerical differentiation to within 1e-8 (double precision limit)

The quantile_knots() function deduplicates output to prevent rank-deficient
design matrices when the dose distribution has mass points.

Performance note — sparse matrix evaluation (2026-06 benchmark)
--------------------------------------------------------------
B-spline design matrices are structurally sparse (each row has at most
degree+1 nonzero entries), but for contdid's typical num_knots=3-10 the
matrix dimension K=7-14 means 29-57% of entries are nonzero. Benchmark
results on Apple M-series (scipy 1.17, numpy 2.2):

- Sparse matrix operations (B^T @ B, least-squares) are 5-25x SLOWER than
  dense due to index management overhead dominating at small K.
- BSpline.design_matrix (scipy C implementation) constructs the matrix 3-4x
  faster than a Python loop, so we use it and immediately convert to dense.
- Only at num_knots >= 20 would keeping sparse format benefit matrix-vector
  multiply, but Gram matrix and solve still need dense. Since contdid never
  uses num_knots > 10 in practice, we always produce dense output.

See tests/benchmarks/bench_sparse.py for full evaluation data.

Decision: Sparse representation is definitively rejected for contdid's
operating range. This conclusion is based on measured benchmarks, not
theoretical projection. See tests/benchmarks/bench_sparse.py.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def _make_knot_vector(
    dose: np.ndarray,
    degree: int,
    interior_knots: list[float],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build clamped knot vector for B-spline construction.

    The clamped knot vector repeats the boundary knots (degree+1) times so that
    the spline interpolates at the endpoints.  The full knot vector is:
        [xmin]*(degree+1) + interior_knots + [xmax]*(degree+1)

    Parameters
    ----------
    dose : array of dose values (used only for min/max when xmin/xmax not given)
    degree : polynomial degree
    interior_knots : sorted interior knot positions
    xmin : optional explicit left boundary
    xmax : optional explicit right boundary

    Returns
    -------
    1-D array of knot positions for scipy.interpolate.BSpline
    """
    if xmin is None:
        xmin = float(np.min(dose))
    if xmax is None:
        xmax = float(np.max(dose))
    # Handle degenerate case where all dose values are identical
    if xmax <= xmin:
        xmax = xmin + 1.0
    left = [xmin] * (degree + 1)
    right = [xmax] * (degree + 1)
    return np.asarray(left + list(interior_knots) + right, dtype=float)


# Performance note — caching evaluation (2026-06 analysis)
# -----------------------------------------------------------
# Caching the design matrix was evaluated but NOT implemented because:
# - Each cont_did() call invokes build_bspline_design 2-3 times max
# - Construction time is ~1-2ms (0.5-2% of total execution)
# - Cache key computation (array hashing) costs ≈ construction itself
# - Event-study uses different dose arrays per (g,t), preventing cross-call reuse
# Net benefit: <5%. Complexity cost outweighs gain.


def build_bspline_design(
    dose: np.ndarray,
    degree: int,
    interior_knots: list[float],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build B-spline design matrix (n x K) where K = len(interior_knots) + degree + 1.

    Uses clamped (repeated) boundary knots at min/max of dose.
    Includes intercept column (constant B-spline basis function).

    Uses scipy's optimized C implementation (BSpline.design_matrix) which
    evaluates all basis functions simultaneously — ~3-4x faster than the
    naive Python loop over individual basis functions.

    Parameters
    ----------
    dose : array of dose values
    degree : polynomial degree (typically 3 for cubic)
    interior_knots : sorted interior knot positions
    xmin : optional explicit left boundary (defaults to min(dose))
    xmax : optional explicit right boundary (defaults to max(dose))

    Returns
    -------
    Design matrix of shape (len(dose), K) where K = len(interior_knots) + degree + 1
    """
    dose = np.asarray(dose, dtype=float)
    knots = _make_knot_vector(dose, degree, interior_knots, xmin=xmin, xmax=xmax)
    num_basis = len(interior_knots) + degree + 1

    # Clamp dose to knot range to avoid NaN from boundary evaluation.
    # BSpline.design_matrix requires x within [knots[degree], knots[-degree-1]].
    dose_clamped = np.clip(dose, float(knots[0]), float(knots[-1]))

    # Use scipy's optimized C routine (available since scipy 1.8) which
    # constructs the sparse design matrix in one pass. Convert to dense
    # immediately because downstream operations (B^T @ B, least-squares)
    # are faster in dense format for typical K=7-14.
    design = BSpline.design_matrix(dose_clamped, knots, degree).toarray()

    # Partition-of-unity check (debug mode)
    if __debug__ and num_basis > 0:
        row_sums = design.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10), (
            f"B-spline partition of unity violated: max|sum-1| = "
            f"{np.max(np.abs(row_sums - 1.0)):.2e}"
        )

    return design


def build_bspline_derivative_design(
    dose: np.ndarray,
    degree: int,
    interior_knots: list[float],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build B-spline first-derivative design matrix.

    Uses BSpline.derivative() method for exact analytic derivatives.

    Parameters
    ----------
    dose : array of dose values
    degree : polynomial degree (typically 3 for cubic)
    interior_knots : sorted interior knot positions
    xmin : optional explicit left boundary (defaults to min(dose))
    xmax : optional explicit right boundary (defaults to max(dose))

    Returns
    -------
    Derivative design matrix of shape (len(dose), K) where K = len(interior_knots) + degree + 1
    """
    dose = np.asarray(dose, dtype=float)
    knots = _make_knot_vector(dose, degree, interior_knots, xmin=xmin, xmax=xmax)
    num_basis = len(interior_knots) + degree + 1
    n = len(dose)
    deriv_design = np.empty((n, num_basis), dtype=float)

    for j in range(num_basis):
        coeffs = np.zeros(num_basis, dtype=float)
        coeffs[j] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        dspline = spline.derivative(nu=1)
        values = dspline(dose)
        # Handle boundary NaN the same way
        nan_mask = np.isnan(values)
        if np.any(nan_mask):
            dose_clamped = np.clip(dose[nan_mask], float(knots[0]), float(knots[-1]))
            spline_ext = BSpline(knots, coeffs, degree, extrapolate=True)
            dspline_ext = spline_ext.derivative(nu=1)
            values[nan_mask] = dspline_ext(dose_clamped)
        deriv_design[:, j] = values

    # Derivative row sums should be ~0 (derivative of partition-of-unity)
    if __debug__ and num_basis > 0:
        deriv_row_sums = deriv_design.sum(axis=1)
        assert np.allclose(deriv_row_sums, 0.0, atol=1e-8), (
            f"B-spline derivative partition check violated: max|sum| = "
            f"{np.max(np.abs(deriv_row_sums)):.2e}"
        )

    return deriv_design


def quantile_knots(dose: np.ndarray, num_knots: int) -> list[float]:
    """Select interior knots at quantiles of the dose distribution.

    Handles duplicate values by deduplicating knots and ensuring minimum
    separation to maintain numerical stability of the B-spline design matrix.

    Parameters
    ----------
    dose : array of positive dose values
    num_knots : number of interior knots requested

    Returns
    -------
    List of unique interior knot positions (may be fewer than num_knots
    if data has insufficient unique values).
    """
    if num_knots <= 0:
        return []
    dose = np.asarray(dose, dtype=float)
    if dose.size == 0:
        return []

    # Compute quantile-based knots
    probs = np.linspace(0.0, 1.0, num_knots + 2)[1:-1]  # Interior probs
    knots = np.quantile(dose, probs)

    # Deduplicate: remove knots that are too close together
    # Use relative tolerance based on data range
    data_range = float(np.ptp(dose))
    if data_range <= 0:
        return []

    min_separation = data_range * np.finfo(float).eps ** 0.5  # ~1e-8 * range

    # Also exclude knots too close to boundaries (they create near-singular
    # basis when the clamped knot vector repeats boundary values)
    xmin = float(np.min(dose))
    xmax = float(np.max(dose))
    boundary_margin = data_range * 1e-6  # Small margin from boundaries

    unique_knots: list[float] = []
    for knot in sorted(knots):
        # Skip knots at or very near boundaries
        if knot <= xmin + boundary_margin or knot >= xmax - boundary_margin:
            continue
        if not unique_knots or abs(knot - unique_knots[-1]) > min_separation:
            unique_knots.append(float(knot))

    return unique_knots


def even_knots(dose: np.ndarray, num_knots: int) -> list[float]:
    """Select evenly-spaced interior knots between min and max dose.

    Matches R package's choose_knots_even:
        seq(min,max,length=num_knots+2)[-c(1,n+2)]

    Parameters
    ----------
    dose : array of dose values
    num_knots : number of interior knots to place

    Returns
    -------
    Sorted list of evenly-spaced interior knot positions
    """
    if num_knots <= 0:
        return []
    dose = np.asarray(dose, dtype=float)
    xmin = float(np.min(dose))
    xmax = float(np.max(dose))
    grid = np.linspace(xmin, xmax, num_knots + 2)
    # Remove first and last (boundary)
    return [float(k) for k in grid[1:-1]]
