"""Unified high-level API for continuous treatment difference-in-differences.

This module provides the ``cont_did()`` function as a single entry point that
mirrors the R package's ``cont_did()`` interface. It automatically routes to
the appropriate estimator based on the specification parameters and returns
a complete ContDIDResult with inference already attached.

Theoretical basis
-----------------
The unified API dispatches to estimators implementing:
- ATT(d): arXiv-2107.02637v7, Theorem 3.1 (identification)
- ACRT(d): arXiv-2107.02637v7, Section 4 (derivative estimation)
- Event-study: arXiv-2107.02637v7, Section A3 (multi-period aggregation)
- CCK inference: arXiv-2107.11869v3, Theorem 2 (uniform confidence bands)

All fine-grained functions remain available for advanced users who need
custom estimation or inference pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from .data import PanelData
from .registry import get_default_registry
from .results import ContDIDResult
from .specs import ContDIDSpec
from .validation import ContDIDValidationError, ValidationStrictness, validate_panel_data

if TYPE_CHECKING:
    from .pretest import PreTrendTestResult


def cont_did(
    panel: PanelData,
    spec: ContDIDSpec | None = None,
    *,
    # Core specification (used if spec is None)
    target_parameter: str = "level",
    aggregation: str = "dose",
    dose_est_method: str = "parametric",
    control_group: str = "notyettreated",
    treatment_type: str = "continuous",
    # Estimation parameters
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    knot_method: str = "quantile",
    # Inference parameters
    anticipation: int = 0,
    alp: float = 0.05,
    bstrap: bool = True,
    cband: bool = False,
    boot_type: str = "multiplier",
    biters: int = 1000,
    # Event-study parameters
    base_period: int | str | None = None,
    # CCK adaptive parameters
    adaptive: bool = False,
    adaptive_k_min: int | None = None,
    adaptive_k_max: int | None = None,
    adaptive_seed: int | None = None,
    # Covariates and clustering
    covariates: tuple[str, ...] | None = None,
    cluster_column: str | None = None,
) -> ContDIDResult:
    """Unified entry point for continuous treatment DiD estimation.

    This function mirrors the R package's ``cont_did()`` interface, automatically
    routing to the appropriate estimator and returning a complete result with
    inference (confidence intervals/bands) already attached.

    Parameters
    ----------
    panel : PanelData
        Validated panel data object.
    spec : ContDIDSpec or None
        If provided, overrides all specification parameters below.
    target_parameter : str
        "level" for ATT(d) or "slope" for ACRT(d).
    aggregation : str
        "dose" for dose-response aggregation or "eventstudy" for event-study.
    dose_est_method : str
        "parametric" for B-spline OLS or "cck" for CCK sieve estimation.
    control_group : str
        Which units serve as the comparison group (default: "notyettreated").

        - "notyettreated": Uses never-treated (G=0) AND not-yet-treated (G>t)
          units as controls. Recommended for staggered adoption designs;
          maximizes control-group sample size and improves statistical power.
          This is the default, consistent with the R package ``contdid``.
        - "nevertreated": Uses only never-treated (G=0) units as controls.
          Appropriate when a sufficient pool of never-treated units exists
          and the researcher prefers a fixed comparison group throughout.

        Both strategies are supported by the multi-period identification
        theory in arXiv-2107.02637v7, Section A3 (Assumption 3-MP).
    treatment_type : str
        Treatment type. Currently only "continuous" is supported.
        The paper (arXiv-2107.02637v7, Assumption 4) also covers multi-valued
        discrete treatment theoretically, but only the continuous-dose
        estimators (B-spline / CCK sieve) are implemented.
    dvals : array-like or None
        Dose grid evaluation points. Default: quantiles of treated dose.
    degree : int
        B-spline degree (default 3 = cubic).
    num_knots : int
        Number of interior knots (default 0 = global polynomial).
    knot_method : str, default "quantile"
        Interior knot placement strategy.

        - "quantile": Place knots at quantiles of the positive dose distribution.
          Adapts to the data distribution; recommended when dose values cluster.
          Matches R package's choose_knots_quantile().
        - "even": Place knots at evenly-spaced positions between min and max dose.
          Uniform spacing; matches R package's choose_knots_even().
    anticipation : int
        Number of anticipation periods (default 0).
    alp : float
        Significance level (default 0.05).
    bstrap : bool
        Whether to use bootstrap inference (default True).
    cband : bool
        Whether to compute simultaneous confidence band (default False).
    boot_type : str
        Bootstrap type: "multiplier", "rademacher", or "mammen".
    biters : int
        Number of bootstrap iterations (default 1000).
    base_period : int, str, or None
        Base period strategy for event study: "varying" (default), "universal",
        or fixed int.
    adaptive : bool
        Whether to use Lepski adaptive dimension selection (CCK only,
        two-period only).
    adaptive_k_min : int or None
        Minimum dimension for adaptive selection.
    adaptive_k_max : int or None
        Maximum dimension for adaptive selection.
    adaptive_seed : int or None
        Random seed for adaptive bootstrap.
    covariates : tuple of str or None
        Covariate column names for adjustment.
    cluster_column : str or None
        Column name for cluster-robust standard errors.

    Returns
    -------
    ContDIDResult
        Complete result with point estimates, standard errors, confidence
        intervals/bands, and metadata including identification assumptions.

    Raises
    ------
    NotImplementedError
        If ``covariates`` is not None or ``treatment_type="discrete"`` is
        specified. These features lack complete estimation theory in the
        current paper framework (arXiv:2107.02637v7).
    ContDIDValidationError
        If panel data fails balance/consistency checks, or if spec
        parameters (e.g., control_group, aggregation) are invalid.
    ValueError
        If incompatible parameter combinations are passed.

    Examples
    --------
    >>> from contdid import PanelData, cont_did
    >>> result = cont_did(panel, target_parameter="level", aggregation="dose")
    >>> print(result.summary())

    >>> # Event study with anticipation
    >>> result = cont_did(panel, aggregation="eventstudy", anticipation=1)

    >>> # CCK nonparametric with adaptive dimension
    >>> result = cont_did(panel, dose_est_method="cck", adaptive=True)

    Notes
    -----
    This is a convenience wrapper. For fine-grained control, use the
    individual estimation functions directly:

    - ``estimate_dose_effects()`` / ``estimate_dose_level_effects()`` /
      ``estimate_dose_slope_effects()``
    - ``estimate_eventstudy_effects()`` /
      ``estimate_eventstudy_slope_effects()``
    - ``estimate_dose_effects_multiperiod()``
    """
    # Build spec if not provided
    if spec is None:
        spec = ContDIDSpec(
            target_parameter=target_parameter,
            aggregation=aggregation,
            dose_est_method=dose_est_method,
            control_group=control_group,
            treatment_type=treatment_type,
            anticipation=anticipation,
            alp=alp,
            bstrap=bstrap,
            cband=cband,
            boot_type=boot_type,
            biters=biters,
            covariates=covariates,
            cluster_column=cluster_column,
        )

    # Validate panel
    _strictness_map = {
        "strict": ValidationStrictness.STRICT,
        "normal": ValidationStrictness.NORMAL,
        "lenient": ValidationStrictness.LENIENT,
    }
    strictness = _strictness_map.get(
        spec.validation_strictness, ValidationStrictness.STRICT
    )
    validated_panel = validate_panel_data(panel, spec=spec, strictness=strictness)

    # Validate aggregation
    if spec.aggregation not in ("dose", "eventstudy"):
        raise ContDIDValidationError(
            f"Unsupported aggregation: {spec.aggregation!r}. Use 'dose' or 'eventstudy'."
        )

    # Route via estimator registry
    registry = get_default_registry()
    estimator = registry.get(spec.dose_est_method)

    # Validate estimator constraints against spec and panel
    errors = estimator.validate_spec(spec, validated_panel)
    if errors:
        raise ContDIDValidationError(
            f"估计器 {spec.dose_est_method!r} 约束验证失败: " + "; ".join(errors)
        )

    # Dispatch to estimator
    result = estimator.estimate(
        validated_panel,
        spec,
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        knot_method=knot_method,
        base_period=base_period,
        adaptive=adaptive,
        adaptive_k_min=adaptive_k_min,
        adaptive_k_max=adaptive_k_max,
        adaptive_seed=adaptive_seed,
    )

    # Execute post-processing hooks (does not modify result itself)
    from .hooks import _default_hook_registry, HookStage

    hook_outputs = _default_hook_registry.execute(result, HookStage.POST_INFERENCE)
    if hook_outputs:
        result._hook_outputs = hook_outputs

    return result


def _route_eventstudy(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals,
    degree: int,
    num_knots: int,
    base_period,
) -> ContDIDResult:
    """Route to event-study estimator based on target_parameter."""
    from .eventstudy import (
        estimate_eventstudy_effects,
        estimate_eventstudy_slope_effects,
    )

    if spec.target_parameter == "level":
        return estimate_eventstudy_effects(
            panel,
            spec,
            dvals=dvals,
            degree=degree,
            num_knots=num_knots,
            base_period=base_period,
        )
    elif spec.target_parameter == "slope":
        return estimate_eventstudy_slope_effects(
            panel,
            spec,
            dvals=dvals,
            degree=degree,
            num_knots=num_knots,
            base_period=base_period,
        )
    else:
        raise ContDIDValidationError(
            f"Unsupported target_parameter for event study: {spec.target_parameter!r}"
        )


def _route_multiperiod_dose(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals,
    degree: int,
    num_knots: int,
) -> ContDIDResult:
    """Route to multi-period dose estimator."""
    import numpy as np

    from .multiperiod import estimate_multiperiod_dose

    # Convert dvals to array for multiperiod API
    dose_grid = None
    if dvals is not None:
        if hasattr(dvals, "__iter__"):
            dose_grid = np.asarray(list(dvals), dtype=float)
        else:
            dose_grid = np.asarray([dvals], dtype=float)

    mp_result = estimate_multiperiod_dose(
        panel_df=panel.frame,
        id_column=panel.id_column,
        time_column=panel.time_column,
        outcome_column=panel.outcome_column,
        dose_column=panel.dose_column,
        group_column=panel.group_column,
        dose_grid=dose_grid,
        degree=degree,
        num_knots=num_knots,
        control_group=spec.control_group,
        target=spec.target_parameter,
        biters=spec.biters,
        alp=spec.alp,
        cband=spec.cband,
        anticipation=spec.anticipation,
    )

    # Wrap MultiPeriodDoseResult into ContDIDResult for unified interface
    estimand = "ATT(d)" if spec.target_parameter == "level" else "ACRT(d)"
    return ContDIDResult(
        estimand=estimand,
        grid=list(mp_result.dose_grid),
        estimate=list(mp_result.point_estimate),
        std_error=list(mp_result.standard_error),
        metadata={
            "target_parameter": spec.target_parameter,
            "aggregation": "dose",
            "dose_est_method": spec.dose_est_method,
            "panel_type": "multiperiod",
            "confidence_band_lower": list(mp_result.confidence_band_lower),
            "confidence_band_upper": list(mp_result.confidence_band_upper),
            "local_results": mp_result.local_results,
            **mp_result.metadata,
        },
    )


def _route_two_period_dose(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals,
    degree: int,
    num_knots: int,
    knot_method: str = "quantile",
    adaptive: bool,
    adaptive_k_min: int | None,
    adaptive_k_max: int | None,
    adaptive_seed: int | None,
) -> ContDIDResult:
    """Route to two-period dose estimator."""
    from .estimation import (
        estimate_dose_level_effects,
        estimate_dose_slope_effects,
    )

    # Build kwargs for the estimation function
    kwargs: dict = dict(dvals=dvals, degree=degree, num_knots=num_knots, knot_method=knot_method)

    if spec.dose_est_method == "cck" and adaptive:
        kwargs["adaptive"] = True
        if adaptive_k_min is not None:
            kwargs["adaptive_k_min"] = adaptive_k_min
        if adaptive_k_max is not None:
            kwargs["adaptive_k_max"] = adaptive_k_max
        if adaptive_seed is not None:
            kwargs["adaptive_seed"] = adaptive_seed

    if spec.target_parameter == "level":
        return estimate_dose_level_effects(panel, spec, **kwargs)
    elif spec.target_parameter == "slope":
        return estimate_dose_slope_effects(panel, spec, **kwargs)
    else:
        raise ContDIDValidationError(f"Unsupported target_parameter: {spec.target_parameter!r}")


def pre_trend_test_from_result(result: ContDIDResult) -> "PreTrendTestResult":
    """Run pre-trend test on an existing event-study result.

    This is a convenience wrapper that extracts pre-treatment event-time
    estimates and their covariance from a ContDIDResult produced by
    estimate_eventstudy_effects() or cont_did(aggregation='eventstudy').

    Parameters
    ----------
    result : ContDIDResult
        Must be an event-study result with event_time_covariance in metadata.

    Returns
    -------
    PreTrendTestResult
    """
    from .pretest import _compute_wald_statistic, PreTrendTestResult
    import numpy as np

    event_time_grid = result.metadata.get("event_time_grid")
    covariance = result.metadata.get("event_time_covariance")

    if event_time_grid is None or covariance is None:
        raise ContDIDValidationError(
            "pre_trend_test_from_result requires an event-study result "
            "with event_time_grid and event_time_covariance in metadata"
        )

    # Extract pre-treatment indices
    pre_indices = [i for i, et in enumerate(event_time_grid) if et < 0]
    if not pre_indices:
        raise ContDIDValidationError(
            "No pre-treatment event times found; cannot perform pre-trend test"
        )

    estimates = np.array([result.estimate[i] for i in pre_indices])
    se = np.array([result.std_error[i] for i in pre_indices])
    cov_matrix = np.array(covariance)[np.ix_(pre_indices, pre_indices)]
    event_times = [event_time_grid[i] for i in pre_indices]

    # Compute Wald statistic
    W, p_value, df = _compute_wald_statistic(estimates, cov_matrix)

    return PreTrendTestResult(
        test_statistic=W,
        p_value=p_value,
        degrees_of_freedom=df,
        pre_period_estimates=tuple(estimates.tolist()),  # type: ignore[arg-type]
        pre_period_se=tuple(se.tolist()),  # type: ignore[arg-type]
        pre_period_event_times=tuple(int(et) for et in event_times),
        reject_at_05=p_value < 0.05,
        reject_at_10=p_value < 0.10,
        covariance_matrix=cov_matrix,
        metadata={"source": "from_result", "n_pre_periods": len(pre_indices)},
    )
