"""Influence function infrastructure for contdid estimation.

Provides unit-level influence functions for efficient multiplier bootstrap
and cross-(g,t) covariance computation in multi-period aggregation.

The influence function for a linear functional θ = l'β of OLS coefficients is:
  IF_i = l' (X'X)^{-1} X_i ε_i

For the overall ATT = E[ATT(D)|D>0]:
  Treated unit i: IF_i = [l'(X'X)^{-1} X_i ε_i] + [f̂(D_i) - θ̂] / n_treated
  Untreated unit j: IF_j = -(ΔY_j - mean(ΔY_untreated)) / n_untreated  (level only)
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np


_NORMAL = NormalDist()


_VALID_BOOT_TYPES = ("multiplier", "rademacher", "mammen")


def _get_progress_bar(iterable, total, desc="Bootstrap"):
    """如果 tqdm 可用则使用进度条，否则返回原始迭代器。"""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except ImportError:
        return iterable


def _validate_boot_type(boot_type: str) -> None:
    """Validate boot_type parameter."""
    if boot_type not in _VALID_BOOT_TYPES:
        raise ValueError(
            f"boot_type must be 'multiplier', 'rademacher', or 'mammen'; got {boot_type!r}"
        )


def _draw_bootstrap_weights(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    boot_type: str = "multiplier",
) -> np.ndarray:
    """Draw bootstrap weights with E[ξ]=0, Var(ξ)=1.

    Parameters
    ----------
    rng : numpy Generator
    shape : shape of the weight array
    boot_type : "multiplier" (Gaussian), "rademacher" (±1), or "mammen"
    """
    if boot_type == "multiplier":
        return rng.standard_normal(shape)
    elif boot_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=shape)
    elif boot_type == "mammen":
        # Two-point distribution matching first 3 moments of N(0,1)
        # P(ξ = a) = (√5+1)/(2√5), P(ξ = b) = (√5-1)/(2√5)
        # where a = (1-√5)/2, b = (1+√5)/2
        sqrt5 = np.sqrt(5.0)
        a = (1.0 - sqrt5) / 2.0  # ≈ -0.618
        b = (1.0 + sqrt5) / 2.0  # ≈ 1.618
        prob_a = (sqrt5 + 1.0) / (2.0 * sqrt5)  # ≈ 0.724
        draws = rng.random(shape)
        return np.where(draws < prob_a, a, b)
    else:
        raise ValueError(
            f"boot_type must be 'multiplier', 'rademacher', or 'mammen'; got {boot_type!r}"
        )


@dataclass(slots=True)
class InfluenceFunction:
    """Unit-level influence function storage for efficient inference.

    Attributes
    ----------
    unit_ids : tuple of unit identifiers (matches panel id_column values)
    values : ndarray of shape (n_units, n_estimands)
        The influence function value for each unit and each estimand point.
        For dose-level estimation, n_estimands = len(dose_grid).
    estimand_labels : tuple of strings describing each estimand column
    n_total : int
        Total sample size used for variance normalization.
    """

    unit_ids: tuple[object, ...]
    values: np.ndarray
    estimand_labels: tuple[str, ...]
    n_total: int

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("influence function values must be 2-dimensional")
        if values.shape[0] != len(self.unit_ids):
            raise ValueError("influence function rows must match unit_ids length")
        if values.shape[1] != len(self.estimand_labels):
            raise ValueError("influence function columns must match estimand_labels length")
        object.__setattr__(self, "values", values)

    @property
    def n_units(self) -> int:
        return self.values.shape[0]

    @property
    def n_estimands(self) -> int:
        return self.values.shape[1]

    def covariance(self) -> np.ndarray:
        """Compute covariance matrix from influence functions.

        Var(θ̂) = (1/n²) * Σ IF_i IF_i'
        """
        cov = (self.values.T @ self.values) / (self.n_total**2)
        return (cov + cov.T) / 2.0  # Symmetrize

    def standard_error(self) -> np.ndarray:
        """Compute standard errors from influence functions.

        SE = sqrt(diag(Cov))
        """
        cov = self.covariance()
        raw_var = np.diag(cov)
        raw_var = np.clip(raw_var, a_min=0.0, a_max=None)
        return np.sqrt(raw_var)

    def clustered_covariance(self, cluster_ids: np.ndarray | tuple) -> np.ndarray:
        """Compute cluster-robust covariance matrix.

        Aggregates influence functions within clusters before computing
        the outer product:

        .. code-block:: text

            Cov_c = (1/n^2) * sum_c (IF_c)(IF_c)'
            where IF_c = sum_{i in cluster c} IF_i

        Parameters
        ----------
        cluster_ids : array-like of shape (n_units,)
            Cluster identifier for each unit. Units with the same cluster_id
            have their influence functions summed before the outer product.

        Returns
        -------
        Covariance matrix of shape (n_estimands, n_estimands).
        """
        cluster_ids = np.asarray(cluster_ids)
        if len(cluster_ids) != self.n_units:
            raise ValueError(
                f"cluster_ids length ({len(cluster_ids)}) must match n_units ({self.n_units})"
            )

        # Get unique clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Sum IFs within each cluster
        cluster_sums = np.zeros((n_clusters, self.n_estimands), dtype=float)
        for c_idx, c_id in enumerate(unique_clusters):
            mask = cluster_ids == c_id
            cluster_sums[c_idx] = self.values[mask].sum(axis=0)

        # Cluster-robust covariance: (1/n^2) * sum_c IF_c IF_c'
        cov = (cluster_sums.T @ cluster_sums) / (self.n_total**2)
        return (cov + cov.T) / 2.0  # Symmetrize

    def clustered_standard_error(self, cluster_ids: np.ndarray | tuple) -> np.ndarray:
        """Compute cluster-robust standard errors."""
        cov = self.clustered_covariance(cluster_ids)
        raw_var = np.diag(cov)
        raw_var = np.clip(raw_var, a_min=0.0, a_max=None)
        return np.sqrt(raw_var)

    def clustered_multiplier_bootstrap(
        self,
        cluster_ids: np.ndarray | tuple,
        *,
        biters: int = 1000,
        alp: float = 0.05,
        cband: bool = False,
        seed: int | None = None,
        boot_type: str = "multiplier",
    ) -> dict[str, Any]:
        """Cluster-robust multiplier bootstrap.

        Draws one multiplier per CLUSTER (not per unit), then applies
        to cluster-summed IFs.
        """
        _validate_boot_type(boot_type)
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Sum IFs within clusters
        cluster_sums = np.zeros((n_clusters, self.n_estimands), dtype=float)
        for c_idx, c_id in enumerate(unique_clusters):
            mask = cluster_ids == c_id
            cluster_sums[c_idx] = self.values[mask].sum(axis=0)

        se = self.clustered_standard_error(cluster_ids)
        pointwise_crit = float(_NORMAL.inv_cdf(1.0 - alp / 2.0))

        if not cband:
            return {
                "bootstrap_type": "clustered_multiplier",
                "bootstrap_seed": seed,
                "std_error": se.tolist(),
                "pointwise_critical_value": pointwise_crit,
                "critical_value": pointwise_crit,
                "confidence_band_kind": "pointwise_clustered_multiplier",
                "n_clusters": n_clusters,
            }

        rng = np.random.default_rng(seed)
        max_stats = np.empty(biters, dtype=float)

        for b in _get_progress_bar(range(biters), biters, "Clustered Bootstrap"):
            # One draw per cluster
            xi = _draw_bootstrap_weights(rng, (n_clusters,), boot_type)
            # Bootstrap statistic: (1/n) * xi @ cluster_sums
            boot_stats = (xi @ cluster_sums) / self.n_total
            with np.errstate(divide="ignore", invalid="ignore"):
                standardized = np.where(se > 0, np.abs(boot_stats) / se, 0.0)
            max_stats[b] = standardized.max()

        critical_value = float(np.quantile(max_stats, 1.0 - alp))

        return {
            "bootstrap_type": "clustered_multiplier",
            "bootstrap_seed": seed,
            "std_error": se.tolist(),
            "pointwise_critical_value": pointwise_crit,
            "critical_value": critical_value,
            "confidence_band_kind": "simultaneous_clustered_multiplier",
            "n_clusters": n_clusters,
        }

    def multiplier_bootstrap(
        self,
        *,
        biters: int = 1000,
        alp: float = 0.05,
        cband: bool = False,
        seed: int | None = None,
        boot_type: str = "multiplier",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Efficient multiplier bootstrap using influence functions.

        Algorithm:
        1. Draw ξ ~ N(0, 1) of shape (biters, n_units)
        2. Compute bootstrap statistics: T_b = (1/n_total) * ξ_b @ IF
        3. Standardize: max_j \|T_b,j / SE_j\|
        4. Critical value: quantile of max_stat at 1-alpha

        Parameters
        ----------
        biters : int, default 1000
            Number of bootstrap replications.
        alp : float, default 0.05
            Significance level.
        cband : bool, default False
            If True, compute simultaneous confidence band critical value.
        seed : int or None
            Random seed for reproducibility.
        boot_type : str, default "multiplier"
            Bootstrap weight distribution: "multiplier", "rademacher", or "mammen".
        n_jobs : int, default 1
            Number of parallel jobs for bootstrap computation. Currently only
            sequential execution (n_jobs=1) is supported. The BLAS-level
            parallelism in NumPy matrix operations already provides implicit
            multi-threading. Explicit parallelization is reserved for future
            optimization when bootstrap computation becomes the identified
            bottleneck.

        Returns dict compatible with compute_multiplier_bootstrap output format.
        """
        _validate_boot_type(boot_type)
        se = self.standard_error()
        pointwise_crit = float(_NORMAL.inv_cdf(1.0 - alp / 2.0))

        if not cband:
            return {
                "bootstrap_type": "multiplier",
                "bootstrap_seed": seed,
                "std_error": se.tolist(),
                "pointwise_critical_value": pointwise_crit,
                "critical_value": pointwise_crit,
                "confidence_band_kind": "pointwise_multiplier",
            }

        rng = np.random.default_rng(seed)

        # Draw multiplier weights and compute bootstrap statistics
        max_stats = np.empty(biters, dtype=float)
        chunk_size = max(1, min(biters, 500_000 // max(1, self.n_units)))

        for start in _get_progress_bar(
            range(0, biters, chunk_size),
            (biters + chunk_size - 1) // chunk_size,
            "Multiplier Bootstrap",
        ):
            stop = min(start + chunk_size, biters)
            xi = _draw_bootstrap_weights(rng, (stop - start, self.n_units), boot_type)
            # Bootstrap estimate perturbation: (1/n) * xi @ IF
            boot_stats = (xi @ self.values) / self.n_total
            # Standardize by SE
            with np.errstate(divide="ignore", invalid="ignore"):
                standardized = np.where(
                    se > 0.0,
                    np.abs(boot_stats) / se,
                    0.0,
                )
            max_stats[start:stop] = standardized.max(axis=1)

        critical_value = float(np.quantile(max_stats, 1.0 - alp))

        return {
            "bootstrap_type": "multiplier",
            "bootstrap_seed": seed,
            "std_error": se.tolist(),
            "pointwise_critical_value": pointwise_crit,
            "critical_value": critical_value,
            "confidence_band_kind": "simultaneous_multiplier",
        }


def compute_dose_influence_function(
    *,
    design: np.ndarray,
    residual: np.ndarray,
    coefficients: np.ndarray,
    loadings: np.ndarray,
    treated_unit_ids: tuple[object, ...],
    treated_dose: np.ndarray,
    n_total: int,
    untreated_delta: np.ndarray | None = None,
    untreated_unit_ids: tuple[object, ...] | None = None,
    include_untreated: bool = False,
    estimand_labels: tuple[str, ...] | None = None,
) -> InfluenceFunction:
    """Compute unit-level influence function from dose regression.

    For the linear functional θ_j = l_j' β (where l_j is row j of loadings):

    Treated unit i contribution:
      IF_i,j = l_j' @ bread @ (X_i * ε_i)
      where bread = (X'X)^{-1}

    If include_untreated and untreated_delta provided (for ATT level):
      Untreated unit k contribution:
        IF_k,j = -(ΔY_k - mean(ΔY_untreated)) / n_untreated

    Parameters
    ----------
    design : (n_treated, p) design matrix X
    residual : (n_treated,) residual vector
    coefficients : (p,) coefficient vector
    loadings : (n_grid, p) loading matrix for dose grid evaluation
    treated_unit_ids : unit IDs for treated observations
    treated_dose : dose values for treated observations
    n_total : total sample size (treated + untreated)
    untreated_delta : outcome changes for untreated units (for level)
    untreated_unit_ids : unit IDs for untreated observations
    include_untreated : whether to include untreated IF components
    estimand_labels : labels for each column of the IF matrix
    """
    n_grid = loadings.shape[0]

    # Bread matrix: (X'X)^{-1}
    bread = np.linalg.pinv(design.T @ design)

    # Score for each treated unit: X_i * ε_i -> (n_treated, p)
    score = design * residual[:, None]

    # IF for treated: n_total * loadings @ bread @ score_i' for each i
    # Convention: θ̂ - θ ≈ (1/n_total) Σ IF_i, so Var(θ̂) = (1/n²) Σ IF_i²
    treated_if = n_total * (score @ bread.T @ loadings.T)  # (n_treated, n_grid)

    if include_untreated and untreated_delta is not None and untreated_unit_ids is not None:
        n_untreated = len(untreated_unit_ids)
        untreated_mean = float(np.mean(untreated_delta))
        # Each untreated unit contributes: -n_total*(delta_j - mean) / n_untreated
        # This contributes equally to all grid points (shifts the benchmark)
        untreated_if = np.outer(
            -n_total * (untreated_delta - untreated_mean) / n_untreated,
            np.ones(n_grid),
        )  # (n_untreated, n_grid)

        all_ids = tuple(list(treated_unit_ids) + list(untreated_unit_ids))
        all_values = np.vstack([treated_if, untreated_if])
    else:
        all_ids = treated_unit_ids
        all_values = treated_if

    if estimand_labels is None:
        estimand_labels = tuple(f"grid_{j}" for j in range(n_grid))

    return InfluenceFunction(
        unit_ids=all_ids,
        values=all_values,
        estimand_labels=estimand_labels,
        n_total=n_total,
    )


def aggregate_influence_functions(
    local_ifs: list[InfluenceFunction],
    weights: list[float],
) -> InfluenceFunction:
    """Aggregate influence functions across (g,t) pairs with given weights.

    Used for multi-period dose aggregation where each (g,t) contributes
    a local IF weighted by the relative treated sample size.

    Each local IF_k has values V_{k,i} satisfying:
    
    .. code-block:: text
    
        Var(\u0302\u03b8_k) = V_k'V_k / n_k\u00b2
    
    For the aggregate \u0302\u03b8 = \u03a3 w_k \u0302\u03b8_k (with normalized weights), assuming
    independence across (g,t) pairs:
    
    .. code-block:: text
    
        Var(\u0302\u03b8) = \u03a3_k w_k\u00b2 * Var(\u0302\u03b8_k) = \u03a3_k w_k\u00b2 * V_k'V_k / n_k\u00b2
    
    We store aggregated IF_agg_i = w_k * V_{k,i} / n_k and set n_total = 1,
    so that covariance() = IF_agg'IF_agg / 1\u00b2 gives the correct variance.
    """
    # Theory (CGBS Appendix B.2): Aggregated IF for unit i across K local
    # estimates is: IF_agg,i = sum_k (w_k / n_k) * IF_k,i
    # where w_k = n_treated_k / sum(n_treated) and IF_k,i is the local IF.
    # Units appearing in multiple (g,t) comparisons (e.g., never-treated controls)
    # accumulate contributions correctly via the unit_id index mapping.
    # The returned InfluenceFunction has n_total=1 so that covariance() = IF'IF
    # directly gives the asymptotic variance of the aggregated estimator.
    if not local_ifs:
        raise ValueError("must provide at least one influence function to aggregate")
    if len(local_ifs) != len(weights):
        raise ValueError("weights must match number of influence functions")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized_weights = [w / total_weight for w in weights]

    # Collect all unique unit IDs
    all_ids_set: dict[object, int] = {}
    for lif in local_ifs:
        for uid in lif.unit_ids:
            if uid not in all_ids_set:
                all_ids_set[uid] = len(all_ids_set)

    all_ids = tuple(all_ids_set.keys())
    n_estimands = local_ifs[0].n_estimands

    # Aggregate with proper scaling: w_k * V_{k,i} / n_k
    aggregated = np.zeros((len(all_ids), n_estimands), dtype=float)
    for lif, w in zip(local_ifs, normalized_weights):
        scale = w / lif.n_total
        for local_idx, uid in enumerate(lif.unit_ids):
            global_idx = all_ids_set[uid]
            aggregated[global_idx] += scale * lif.values[local_idx]

    return InfluenceFunction(
        unit_ids=all_ids,
        values=aggregated,
        estimand_labels=local_ifs[0].estimand_labels,
        n_total=1,
    )
