"""Pre-trend testing for continuous treatment DID (internal).

Tests the null hypothesis that all pre-treatment event-study effects
are jointly zero, following Section 4.4 of Chen, Christensen &
Kankanala (2024). Uses a Wald test on the pre-treatment event-study
estimates with covariance from influence functions.

The test leverages the existing event-study estimation infrastructure
to compute effects at each event-time, then extracts the pre-treatment
subset for joint hypothesis testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .data import PanelData
from .eventstudy import estimate_eventstudy_effects
from .specs import ContDIDSpec
from .validation import ContDIDValidationError


@dataclass(slots=True)
class PreTrendTestResult:
    """Result of pre-trend test.

    Attributes
    ----------
    test_statistic : float
        Wald test statistic (chi-squared distributed under H0).
    p_value : float
        p-value from chi-squared distribution with df degrees of freedom.
    degrees_of_freedom : int
        Number of pre-treatment periods tested (= number of constraints).
    pre_period_estimates : tuple of float
        Point estimates for each pre-treatment event-time.
    pre_period_se : tuple of float
        Standard errors for each pre-treatment event-time.
    pre_period_event_times : tuple of int
        The event-time indices tested (e.g., -3, -2, -1).
    reject_at_05 : bool
        Whether H0 is rejected at the caller's significance level ``alp``
        (default 0.05).
    reject_at_10 : bool
        Whether H0 is rejected at the 10% significance level.
    covariance_matrix : np.ndarray
        The (p x p) estimated covariance matrix of pre-period estimates.
    metadata : dict
        Additional diagnostic information.
    """

    test_statistic: float
    p_value: float
    degrees_of_freedom: int
    pre_period_estimates: tuple[float, ...]
    pre_period_se: tuple[float, ...]
    pre_period_event_times: tuple[int, ...]
    reject_at_05: bool
    reject_at_10: bool
    covariance_matrix: np.ndarray
    metadata: dict[str, Any]


def _compute_wald_statistic(
    estimates: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float, int]:
    """Compute Wald test statistic W = theta' Sigma^{-1} theta ~ chi2_p.

    Parameters
    ----------
    estimates : (p,) array of pre-period point estimates
    covariance : (p, p) covariance matrix

    Returns
    -------
    (test_statistic, p_value, degrees_of_freedom)
    """
    p = len(estimates)
    if p == 0:
        raise ContDIDValidationError("No pre-treatment periods available for testing")

    # Use pseudo-inverse for numerical stability
    cov_inv = np.linalg.pinv(covariance)
    W = float(estimates @ cov_inv @ estimates)
    W = max(W, 0.0)  # Clip to non-negative (numerical safety)
    p_value = float(1.0 - stats.chi2.cdf(W, df=p))
    return W, p_value, p


def pre_trend_test(
    panel_df: pd.DataFrame,
    *,
    id_column: str,
    time_column: str,
    outcome_column: str,
    group_column: str,
    dose_column: str,
    control_group: str = "nevertreated",
    base_period: int | str | None = None,
    covariates: list[str] | None = None,
    alp: float = 0.05,
    biters: int = 1000,
) -> PreTrendTestResult:
    """Pre-trend test for parallel trends plausibility.

    Performs a joint Wald test of the null hypothesis that all pre-treatment
    event-study coefficients are zero, diagnosing the plausibility of the
    parallel trends assumption.

    Theoretical basis (arXiv-2107.02637v7, Section 4.4, lines 807-815):
        H0: \u03b8_e = 0 for all e < 0  (no pre-treatment differential trends)
        Test statistic: W = \u03b8\u0302'_pre @ \u03a3\u0302\u207b\u00b9_pre @ \u03b8\u0302_pre ~ \u03c7\u00b2_p under H0
        where p = number of pre-treatment event times

    Interpretation
    --------------
    This test DIAGNOSES parallel trends plausibility but does NOT PROVE that
    parallel trends hold. Important caveats:

    1. Failure to reject H0 does not confirm the parallel trends assumption;
       it only indicates insufficient evidence against it at the tested scale.
    2. The test cannot distinguish between PT (parallel trends) and SPT
       (strong parallel trends) \u2014 it diagnoses the weaker condition only.
    3. The test has limited power in small samples or when pre-treatment
       trends diverge only slightly.
    4. Post-treatment violations of parallel trends are not detectable
       by this pre-trend test.

    Parameters
    ----------
    panel_df : DataFrame
        Panel data with at least 3 time periods. Must be balanced, with
        constant dose and group within each unit.
    id_column, time_column, outcome_column, group_column, dose_column :
        Column names matching the panel structure.
    control_group : "nevertreated" or "notyettreated"
        Which units form the comparison group.
    base_period : Reference period (default: period just before treatment)
    covariates : Optional covariate column names for covariate-adjusted
        estimation (passed to event-study estimator via ContDIDSpec).
    alp : Significance level for the primary reject decision and bootstrap
        inference (default 0.05).
    biters : Bootstrap iterations for covariance estimation.

    Returns
    -------
    PreTrendTestResult
        Contains test_statistic (Wald W), p_value, degrees_of_freedom,
        pre_period_estimates, pre_period_se, and reject decisions at
        standard significance levels.

    Raises
    ------
    ContDIDValidationError
        If panel has fewer than 3 periods or no pre-treatment effects.

    References
    ----------
    Chen, Christensen, Kankanala & Sant'Anna (2024), Section 4.4.
    "Pre-testing for parallel trends via event-study-type diagnostics."
    """
    # --- Validate minimum period count ---
    time_values = pd.to_numeric(panel_df[time_column], errors="coerce")
    n_periods = int(time_values.nunique())
    if n_periods < 3:
        raise ContDIDValidationError(
            "Pre-trend test requires at least 3 time periods to have "
            "both pre-treatment diagnostics and post-treatment effects"
        )

    # --- Build PanelData wrapper ---
    panel = PanelData(
        frame=panel_df.copy(),
        id_column=id_column,
        time_column=time_column,
        outcome_column=outcome_column,
        group_column=group_column,
        dose_column=dose_column,
    )

    # --- Build ContDIDSpec for event-study estimation ---
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group=control_group,
        treatment_type="continuous",
        anticipation=0,
        alp=alp,
        bstrap=True,
        cband=False,
        boot_type="multiplier",
        biters=biters,
        covariates=tuple(covariates) if covariates else None,
    )

    # --- Run the event-study estimator ---
    result = estimate_eventstudy_effects(panel, spec, base_period=base_period)

    # --- Extract event-time grid and estimates ---
    event_time_grid = list(result.event_time_grid) if result.event_time_grid is not None else []
    estimates_all = np.asarray(result.estimate, dtype=float)
    se_all = np.asarray(result.std_error, dtype=float)

    # --- Extract the full covariance matrix ---
    raw_cov = result.metadata.get("event_time_covariance")
    if raw_cov is not None:
        full_covariance = np.asarray(raw_cov, dtype=float)
    else:
        # Fallback: diagonal covariance from standard errors
        full_covariance = np.diag(se_all**2)

    # --- Identify pre-treatment indices (event_time < 0) ---
    pre_indices = [i for i, et in enumerate(event_time_grid) if et < 0]

    if not pre_indices:
        raise ContDIDValidationError(
            "No pre-treatment event-time periods found in event-study results; "
            "the panel may have too few periods before treatment onset"
        )

    pre_event_times = tuple(int(event_time_grid[i]) for i in pre_indices)
    pre_estimates = estimates_all[pre_indices]
    pre_se = se_all[pre_indices]

    # --- Extract the pre-treatment sub-covariance matrix ---
    idx = np.array(pre_indices, dtype=int)
    pre_covariance = full_covariance[np.ix_(idx, idx)]

    # --- Compute Wald test ---
    W, p_value, df = _compute_wald_statistic(pre_estimates, pre_covariance)

    return PreTrendTestResult(
        test_statistic=W,
        p_value=p_value,
        degrees_of_freedom=df,
        pre_period_estimates=tuple(float(x) for x in pre_estimates),
        pre_period_se=tuple(float(x) for x in pre_se),
        pre_period_event_times=pre_event_times,
        reject_at_05=(p_value < alp),
        reject_at_10=(p_value < 0.10),
        covariance_matrix=pre_covariance,
        metadata={
            "full_event_time_grid": event_time_grid,
            "n_total_event_times": len(event_time_grid),
            "n_pre_periods": len(pre_indices),
            "n_post_periods": sum(1 for et in event_time_grid if et > 0),
            "control_group": control_group,
            "alp": alp,
            "biters": biters,
            "covariates": list(covariates) if covariates else None,
        },
    )
