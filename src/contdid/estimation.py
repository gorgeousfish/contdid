"""Phase 4/6 dose-level and slope estimators for ATT(d) and ACRT(d).

# Performance note — JIT acceleration (2026-06 analysis)
# -------------------------------------------------------
# Numba/Cython acceleration was evaluated but NOT adopted because:
# - B-spline evaluation already uses scipy's C implementation (BSpline.design_matrix)
# - Matrix operations use numpy/BLAS (Fortran-level, near-optimal)
# - Bootstrap inner loop is vectorized numpy (GIL-free)
# - Numba adds ~200MB dependency, inappropriate for academic package
# - Measured potential gain: <1% beyond current compiled backends
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd

from .cck import run_cck_backend
from .data import PanelData
from .influence import InfluenceFunction, compute_dose_influence_function
from .inference import (
    _build_interval_payload,
    append_independent_mean_variance,
    attach_inference_payload,
    build_confidence_band,
    estimate_mean_variance,
)
from .results import ContDIDResult
from .specs import ContDIDSpec
from .validation import (
    ContDIDValidationError,
    _require_public_two_period_dose_timing,
    validate_panel_data,
    validate_spec,
)


def _get_progress_bar(iterable, total, desc="Bootstrap"):
    """如果 tqdm 可用则使用进度条，否则返回原始迭代器。"""
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, leave=False)
    except ImportError:
        return iterable


_DEFAULT_GRID_PROBS = tuple(np.arange(0.10, 1.0, 0.01).tolist())
_DEFAULT_CCK_GRID_SIZE = 50
_SUPPORTED_DOSE_CONTROL_GROUPS = ("nevertreated", "notyettreated")
_DOSE_GRID_NUMERIC_ERROR = "dose grid must contain only finite non-boolean numeric values"
_INTERIOR_KNOT_NUMERIC_ERROR = "interior knots must contain only finite non-boolean numeric values"
_POSITIVE_DOSE_SUPPORT_ERROR_PREFIXES = (
    "positive-dose treated sample is empty",
    "underidentified positive-dose support",
)


@dataclass(slots=True)
class _DoseRegressionFit:
    """Shared parametric dose design used by ATT(d) and ACRT(d)."""

    dose_grid: list[float]
    treated_dose: np.ndarray
    untreated_delta: np.ndarray
    coefficients: np.ndarray
    covariance: np.ndarray
    design: np.ndarray
    residual: np.ndarray
    treated_unit_ids: tuple[object, ...]
    residual_df: int
    spec: ContDIDSpec
    degree: int
    num_knots: int
    knots: list[float]
    untreated_benchmark: float
    treated_dose_mean: float
    treated_count: int
    untreated_count: int
    influence_function: InfluenceFunction | None = None
    p_cov: int = 0  # Number of covariate columns in extended design
    cluster_ids: tuple | None = None  # Cluster IDs for all units (treated + untreated)


@dataclass(slots=True)
class _DosePreparedSample:
    validated_panel: PanelData
    spec: ContDIDSpec
    collapsed: pd.DataFrame
    treated_unit_ids: tuple[object, ...]
    treated_dose: np.ndarray
    treated_outcome: np.ndarray
    untreated_benchmark: float
    treated_count: int
    untreated_count: int
    treated_covariates: np.ndarray | None = None  # (n_treated, p_cov) or None
    untreated_covariates: np.ndarray | None = None  # (n_untreated, p_cov) or None
    treated_cluster_ids: tuple | None = None  # Cluster IDs for treated units
    untreated_cluster_ids: tuple | None = None  # Cluster IDs for untreated units


def _collapse_to_unit_differences(
    panel: PanelData, *, assume_valid_panel: bool = False
) -> pd.DataFrame:
    validated_panel = panel if assume_valid_panel else validate_panel_data(panel)
    frame = validated_panel.frame.sort_values(
        [validated_panel.id_column, validated_panel.time_column]
    )
    grouped = frame.groupby(validated_panel.id_column, sort=False)
    collapsed = grouped.agg(
        {
            validated_panel.time_column: ["first", "last"],
            validated_panel.outcome_column: ["first", "last"],
            validated_panel.group_column: "first",
            validated_panel.dose_column: "first",
        }
    )
    collapsed.columns = [
        "time_first",
        "time_last",
        "outcome_first",
        "outcome_last",
        validated_panel.group_column,
        validated_panel.dose_column,
    ]
    collapsed = collapsed.reset_index()
    collapsed["delta_outcome"] = collapsed["outcome_last"] - collapsed["outcome_first"]
    return collapsed


def _coerce_grid(
    dvals: Iterable[float] | float | None,
    positive_dose: np.ndarray,
    *,
    enforce_observed_support: bool,
    require_strict_explicit_grid: bool = True,
) -> list[float]:
    if positive_dose.size == 0:
        raise ContDIDValidationError("positive-dose treated sample is empty")

    if dvals is None:
        grid = np.quantile(positive_dose, _DEFAULT_GRID_PROBS)
    else:
        if isinstance(dvals, (str, bytes, bool, np.bool_)):
            raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR)
        if isinstance(dvals, np.ndarray):
            if dvals.dtype.kind in {"b", "S", "U"}:
                raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR)
            if dvals.dtype.kind == "O" and any(
                isinstance(value, (str, bytes, bool, np.bool_)) for value in dvals.ravel()
            ):
                raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR)
            try:
                grid = np.asarray(dvals, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR) from exc
        else:
            try:
                raw_values = list(dvals)
            except TypeError:
                raw_values = None
            else:
                if any(isinstance(value, (str, bytes, bool, np.bool_)) for value in raw_values):
                    raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR)
                dvals = raw_values
            try:
                grid = np.asarray(dvals, dtype=float)
            except (TypeError, ValueError):
                try:
                    grid = np.asarray(list(dvals), dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR) from exc
        grid = np.atleast_1d(grid)

    if grid.ndim != 1 or grid.size == 0:
        raise ContDIDValidationError("dose grid must contain at least one value")
    if not np.isfinite(grid).all():
        raise ContDIDValidationError(_DOSE_GRID_NUMERIC_ERROR)
    if (
        require_strict_explicit_grid
        and dvals is not None
        and grid.size > 1
        and np.any(np.diff(grid) <= 0.0)
    ):
        raise ContDIDValidationError(
            "explicit dvals must be strictly increasing with no duplicate dose values"
        )
    if enforce_observed_support:
        support_min = float(np.min(positive_dose))
        support_max = float(np.max(positive_dose))
        if np.any(grid < support_min) or np.any(grid > support_max):
            raise ContDIDValidationError(
                "dose grid must stay within the observed positive-dose treated support "
                f"[{support_min}, {support_max}]"
            )
    return [float(value) for value in grid.tolist()]


def _default_cck_grid(positive_dose: np.ndarray) -> list[float]:
    if positive_dose.size == 0:
        raise ContDIDValidationError("positive-dose treated sample is empty")

    grid = np.linspace(
        float(np.min(positive_dose)),
        float(np.max(positive_dose)),
        _DEFAULT_CCK_GRID_SIZE,
    )
    return [float(value) for value in grid.tolist()]


def _build_dose_grid_from_validated_panel(
    validated_panel: PanelData,
    *,
    dvals: Iterable[float] | float | None = None,
    require_public_two_period_dose_panel: bool,
) -> list[float]:
    if require_public_two_period_dose_panel:
        _require_public_two_period_dose_timing(validated_panel)
    collapsed = _collapse_to_unit_differences(validated_panel, assume_valid_panel=True)
    positive_dose = collapsed.loc[
        collapsed[validated_panel.dose_column] > 0.0, validated_panel.dose_column
    ].to_numpy(dtype=float)
    return _coerce_grid(dvals, positive_dose, enforce_observed_support=True)


def build_dose_grid(
    panel: PanelData, *, dvals: Iterable[float] | float | None = None
) -> list[float]:
    validated_panel = validate_panel_data(panel)
    return _build_dose_grid_from_validated_panel(
        validated_panel,
        dvals=dvals,
        require_public_two_period_dose_panel=True,
    )


def _coerce_knots(
    positive_dose: np.ndarray, num_knots: int, knot_method: str = "quantile"
) -> list[float]:
    """Select interior knots using the specified placement strategy.

    Delegates to bspline.quantile_knots or bspline.even_knots based on
    knot_method. The returned list may contain fewer than num_knots entries
    when using quantile placement and the dose distribution has mass points.

    Parameters
    ----------
    positive_dose : array of positive dose values
    num_knots : number of interior knots requested
    knot_method : str, default "quantile"
        Interior knot placement strategy.
        - "quantile": Place knots at quantiles of the positive dose distribution.
          Adapts to the data distribution; recommended when dose values cluster.
          Matches R package's choose_knots_quantile().
        - "even": Place knots at evenly-spaced positions between min and max dose.
          Uniform spacing; matches R package's choose_knots_even().
    """
    if num_knots < 0:
        raise ContDIDValidationError("num_knots must be nonnegative")
    if num_knots == 0:
        return []

    _VALID_KNOT_METHODS = ("quantile", "even")
    if knot_method not in _VALID_KNOT_METHODS:
        raise ContDIDValidationError(
            f"knot_method must be 'quantile' or 'even'; got {knot_method!r}"
        )

    from .bspline import even_knots, quantile_knots

    if knot_method == "quantile":
        return quantile_knots(positive_dose, num_knots)
    else:
        return even_knots(positive_dose, num_knots)


def _coerce_explicit_knots(knots: Iterable[float]) -> list[float]:
    try:
        raw_values = list(knots)
    except TypeError as exc:
        raise ContDIDValidationError(_INTERIOR_KNOT_NUMERIC_ERROR) from exc

    if not raw_values:
        return []
    if any(isinstance(value, (str, bytes, bool, np.bool_)) for value in raw_values):
        raise ContDIDValidationError(_INTERIOR_KNOT_NUMERIC_ERROR)
    try:
        knot_grid = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ContDIDValidationError(_INTERIOR_KNOT_NUMERIC_ERROR) from exc
    knot_grid = np.atleast_1d(knot_grid)
    if knot_grid.ndim != 1:
        raise ContDIDValidationError(_INTERIOR_KNOT_NUMERIC_ERROR)
    if not np.isfinite(knot_grid).all():
        raise ContDIDValidationError(_INTERIOR_KNOT_NUMERIC_ERROR)
    if knot_grid.size > 1 and np.any(np.diff(knot_grid) <= 0.0):
        raise ContDIDValidationError(
            "explicit interior knots must be strictly increasing with no duplicate values"
        )
    return [float(value) for value in knot_grid.tolist()]


def _coerce_basis_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ContDIDValidationError(f"{name} must be an integer")
    return int(value)


def _legacy_truncated_power_design(
    dose: np.ndarray, degree: int, knots: list[float]
) -> np.ndarray:
    """Legacy truncated power spline design matrix (kept for backward compatibility testing)."""
    if degree < 1:
        raise ContDIDValidationError("degree must be at least 1 for dose estimation")

    columns = [np.ones_like(dose)]
    for power in range(1, degree + 1):
        columns.append(dose**power)
    for knot in knots:
        columns.append(np.clip(dose - knot, a_min=0.0, a_max=None) ** degree)
    return np.column_stack(columns)


def _legacy_truncated_power_derivative(
    dose: np.ndarray, degree: int, knots: list[float]
) -> np.ndarray:
    """Legacy truncated power spline derivative matrix."""
    columns = [np.zeros_like(dose)]
    for power in range(1, degree + 1):
        columns.append(power * (dose ** (power - 1)))
    for knot in knots:
        positive_part = np.clip(dose - knot, a_min=0.0, a_max=None)
        if degree == 1:
            columns.append((dose >= knot).astype(float))
        else:
            columns.append(degree * (positive_part ** (degree - 1)))
    return np.column_stack(columns)


def _build_design_matrix(
    dose: np.ndarray,
    degree: int,
    knots: list[float],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build dose design matrix using B-spline basis functions."""
    from .bspline import build_bspline_design

    if degree < 1:
        raise ContDIDValidationError("degree must be at least 1 for dose estimation")
    return build_bspline_design(dose, degree, knots, xmin=xmin, xmax=xmax)


def _build_derivative_matrix(
    dose: np.ndarray,
    degree: int,
    knots: list[float],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    """Build dose derivative design matrix using B-spline basis derivatives."""
    from .bspline import build_bspline_derivative_design

    return build_bspline_derivative_design(dose, degree, knots, xmin=xmin, xmax=xmax)


def _require_identified_parametric_design(design: np.ndarray, treated_dose: np.ndarray) -> None:
    basis_columns = int(design.shape[1])
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank >= basis_columns:
        return

    treated_count = int(design.shape[0])
    unique_support = int(np.unique(np.asarray(treated_dose, dtype=float)).size)
    raise ContDIDValidationError(
        "underidentified positive-dose support for the requested parametric basis; "
        f"need full column rank but observed rank={design_rank} < columns={basis_columns} "
        f"with treated_count={treated_count} and unique_positive_doses={unique_support}"
    )


def _require_parametric_inference_df(design: np.ndarray) -> None:
    treated_count = int(design.shape[0])
    basis_columns = int(design.shape[1])
    residual_df = treated_count - basis_columns
    if residual_df > 0:
        return

    raise ContDIDValidationError(
        "positive-dose treated support leaves no residual degrees of freedom for inference "
        f"on the requested parametric basis; need treated_count > columns but observed "
        f"treated_count={treated_count} and columns={basis_columns}"
    )


def _is_positive_dose_support_error(error: ContDIDValidationError) -> bool:
    message = str(error)
    return any(message.startswith(prefix) for prefix in _POSITIVE_DOSE_SUPPORT_ERROR_PREFIXES)


def _require_untreated_benchmark_variance_df(untreated_count: int) -> None:
    if untreated_count > 1:
        return

    raise ContDIDValidationError(
        "untreated benchmark variance requires at least two untreated units; "
        f"observed untreated_count={untreated_count}"
    )


def _estimate_treated_sandwich_covariance(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    xtx_inv = np.linalg.pinv(design.T @ design)
    score = design * residual[:, None]
    meat = score.T @ score
    covariance = xtx_inv @ meat @ xtx_inv
    return (covariance + covariance.T) / 2.0


def _require_supported_dose_control_group(spec: ContDIDSpec) -> None:
    if spec.control_group in _SUPPORTED_DOSE_CONTROL_GROUPS:
        return

    raise ContDIDValidationError(
        "dose estimation supports control_group values "
        "'nevertreated' and 'notyettreated' only, "
        f"got {spec.control_group!r}"
    )


def _prepare_dose_sample(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    expected_target: str,
    require_public_dose_control_group: bool = True,
    assume_valid_panel: bool = False,
) -> _DosePreparedSample:
    validated_panel = panel if assume_valid_panel else validate_panel_data(panel)
    validated_spec = (
        validate_spec(spec, panel=validated_panel, assume_valid_panel=True)
        if assume_valid_panel
        else validate_spec(spec, panel=validated_panel)
    )

    if validated_spec.target_parameter != expected_target:
        raise ContDIDValidationError(
            f"expected target_parameter={expected_target!r}, got {validated_spec.target_parameter!r}"
        )
    if validated_spec.aggregation != "dose":
        raise ContDIDValidationError("Phase 4 only supports aggregation='dose'")
    if require_public_dose_control_group:
        _require_supported_dose_control_group(validated_spec)
    _require_public_two_period_dose_timing(
        validated_panel,
        allow_future_groups=(validated_spec.control_group == "notyettreated"),
    )

    collapsed = _collapse_to_unit_differences(validated_panel, assume_valid_panel=True)
    dose_column = validated_panel.dose_column
    group_column = validated_panel.group_column

    # Determine untreated/treated masks based on control_group strategy
    if validated_spec.control_group == "notyettreated":
        # For notyettreated: determine post period to identify future cohorts
        time_support = sorted(validated_panel.frame[validated_panel.time_column].unique().tolist())
        post_period = float(time_support[-1])
        # Untreated = never-treated (group==0) OR not-yet-treated (group > post_period)
        untreated_mask = (collapsed[group_column] == 0) | (collapsed[group_column] > post_period)
        # Treated = units treated at the post period (group == post_period)
        treated_mask = collapsed[group_column] == post_period
    else:
        # nevertreated: original logic — untreated by dose==0, treated by dose>0
        untreated_mask = collapsed[dose_column] == 0.0
        treated_mask = collapsed[dose_column] > 0.0

    untreated = collapsed.loc[untreated_mask, "delta_outcome"].to_numpy(dtype=float)
    if untreated.size == 0:
        raise ContDIDValidationError("dose estimation requires an untreated benchmark")
    treated = collapsed.loc[
        treated_mask,
        [validated_panel.id_column, dose_column, "delta_outcome"],
    ]
    if treated.empty:
        raise ContDIDValidationError("dose estimation requires positive-dose treated units")

    treated_unit_ids = tuple(treated[validated_panel.id_column].tolist())
    treated_dose = treated[dose_column].to_numpy(dtype=float)
    untreated_benchmark = float(untreated.mean())
    treated_outcome = treated["delta_outcome"].to_numpy(dtype=float) - untreated_benchmark

    # 理论边界防护：协变量应在 validate_spec() 阶段被拦截，不应到达此处。
    # 论文(arXiv:2107.02637v7)未提供协变量情形的影响函数修正公式，
    # 此前的零填充实现在 X 与 D 不独立时引入偏差。
    treated_covariates = None
    untreated_covariates = None
    assert spec.covariates is None or len(spec.covariates) == 0, (
        "Internal error: covariates should have been blocked at validation stage. "
        "The paper (arXiv:2107.02637v7) does not provide the influence function "
        "correction formula for covariate conditioning."
    )

    # Extract cluster IDs if specified
    treated_cluster_ids = None
    untreated_cluster_ids = None
    if spec.cluster_column is not None:
        col = spec.cluster_column
        if col not in collapsed.columns:
            # Try original frame
            frame = validated_panel.frame
            if col not in frame.columns:
                raise ContDIDValidationError(f"Cluster column '{col}' not found in panel data")
            # Merge cluster data from original frame (first observation per unit)
            clu_data = (
                frame.groupby(validated_panel.id_column, sort=False)[[col]].first().reset_index()
            )
            collapsed = collapsed.merge(clu_data, on=validated_panel.id_column, how="left")
        treated_cluster_ids = tuple(collapsed.loc[treated_mask, col].tolist())
        untreated_cluster_ids = tuple(collapsed.loc[untreated_mask, col].tolist())

    return _DosePreparedSample(
        validated_panel=validated_panel,
        spec=validated_spec,
        collapsed=collapsed,
        treated_unit_ids=treated_unit_ids,
        treated_dose=treated_dose,
        treated_outcome=treated_outcome,
        untreated_benchmark=untreated_benchmark,
        treated_count=int(treated.shape[0]),
        untreated_count=int(untreated.size),
        treated_covariates=treated_covariates,
        untreated_covariates=untreated_covariates,
        treated_cluster_ids=treated_cluster_ids,
        untreated_cluster_ids=untreated_cluster_ids,
    )


def _fit_shared_dose_design(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    expected_target: str,
    dvals: Iterable[float] | float | None,
    degree: int,
    num_knots: int,
    enforce_observed_support: bool = True,
    require_inference_df: bool = True,
    require_untreated_variance_df: bool = True,
    require_public_dose_control_group: bool = True,
    require_strict_explicit_grid: bool = True,
    assume_valid_panel: bool = False,
    knots: Iterable[float] | None = None,
    knot_method: str = "quantile",
) -> _DoseRegressionFit:
    degree = _coerce_basis_integer(degree, name="degree")
    num_knots = _coerce_basis_integer(num_knots, name="num_knots")
    prepared = _prepare_dose_sample(
        panel,
        spec,
        expected_target=expected_target,
        require_public_dose_control_group=require_public_dose_control_group,
        assume_valid_panel=assume_valid_panel,
    )
    if prepared.spec.dose_est_method != "parametric":
        raise ContDIDValidationError("Phase 4 only supports dose_est_method='parametric'")

    dose_grid = _coerce_grid(
        dvals,
        prepared.treated_dose,
        enforce_observed_support=enforce_observed_support,
        require_strict_explicit_grid=require_strict_explicit_grid,
    )
    if knots is not None:
        knots = _coerce_explicit_knots(knots)
    else:
        knots = _coerce_knots(prepared.treated_dose, num_knots, knot_method)
    # Compute treated dose boundaries BEFORE building any design matrix
    # to ensure training and evaluation use identical knot vectors
    _treated_xmin = float(np.min(prepared.treated_dose))
    _treated_xmax = float(np.max(prepared.treated_dose))
    design = _build_design_matrix(
        prepared.treated_dose,
        degree=degree,
        knots=knots,
        xmin=_treated_xmin,
        xmax=_treated_xmax,
    )

    # 理论边界防护：协变量不应到达估计逻辑
    # 此前的实现将协变量列拼接到设计矩阵并零填充影响函数载荷，
    # 这在 X 与 D 不独立时引入偏差（论文未提供修正公式）。
    assert prepared.treated_covariates is None, (
        "Internal error: covariates reached estimation logic. "
        "They should be blocked at validate_spec() stage."
    )
    p_cov = 0

    _require_identified_parametric_design(design, prepared.treated_dose)
    if require_inference_df:
        _require_parametric_inference_df(design)
    if expected_target == "level" and require_untreated_variance_df:
        _require_untreated_benchmark_variance_df(prepared.untreated_count)
    coefficients, *_ = np.linalg.lstsq(design, prepared.treated_outcome, rcond=None)
    fitted = design @ coefficients
    residual = prepared.treated_outcome - fitted
    residual_df = prepared.treated_count - design.shape[1]
    covariance = _estimate_treated_sandwich_covariance(design, residual)

    dose_column = prepared.validated_panel.dose_column
    group_column = prepared.validated_panel.group_column

    # Derive untreated mask consistent with control_group strategy
    if prepared.spec.control_group == "notyettreated":
        time_support = sorted(
            prepared.validated_panel.frame[prepared.validated_panel.time_column].unique().tolist()
        )
        post_period = float(time_support[-1])
        _untreated_mask = (prepared.collapsed[group_column] == 0) | (
            prepared.collapsed[group_column] > post_period
        )
    else:
        _untreated_mask = prepared.collapsed[dose_column] == 0.0

    untreated_delta = prepared.collapsed.loc[_untreated_mask, "delta_outcome"].to_numpy(
        dtype=float
    )

    # Compute influence function - use same treated dose boundaries for grid evaluation
    grid_for_if = np.asarray(dose_grid, dtype=float)
    if expected_target == "level":
        if_loadings = _build_design_matrix(
            grid_for_if,
            degree=degree,
            knots=knots,
            xmin=_treated_xmin,
            xmax=_treated_xmax,
        )
    else:
        if_loadings = _build_derivative_matrix(
            grid_for_if,
            degree=degree,
            knots=knots,
            xmin=_treated_xmin,
            xmax=_treated_xmax,
        )

    # 理论边界防护：p_cov 应始终为 0（协变量已在验证阶段被拦截）
    assert p_cov == 0, "Internal error: p_cov > 0 should be impossible after validation guard"

    untreated_ids = tuple(
        prepared.collapsed.loc[
            _untreated_mask,
            prepared.validated_panel.id_column,
        ].tolist()
    )

    influence_func = compute_dose_influence_function(
        design=design,
        residual=residual,
        coefficients=coefficients,
        loadings=if_loadings,
        treated_unit_ids=prepared.treated_unit_ids,
        treated_dose=prepared.treated_dose,
        n_total=prepared.treated_count + prepared.untreated_count,
        untreated_delta=untreated_delta if expected_target == "level" else None,
        untreated_unit_ids=untreated_ids if expected_target == "level" else None,
        include_untreated=(expected_target == "level"),
    )

    # Compute cluster IDs for all units if available
    all_cluster_ids = None
    if prepared.treated_cluster_ids is not None:
        all_cluster_ids = tuple(
            list(prepared.treated_cluster_ids)
            + (list(prepared.untreated_cluster_ids) if prepared.untreated_cluster_ids else [])
        )

    return _DoseRegressionFit(
        dose_grid=dose_grid,
        treated_dose=prepared.treated_dose,
        untreated_delta=untreated_delta,
        coefficients=coefficients,
        covariance=covariance,
        design=design,
        residual=residual,
        treated_unit_ids=prepared.treated_unit_ids,
        residual_df=residual_df,
        spec=prepared.spec,
        degree=degree,
        num_knots=len(knots),
        knots=knots,
        untreated_benchmark=prepared.untreated_benchmark,
        treated_dose_mean=float(prepared.treated_dose.mean()),
        treated_count=prepared.treated_count,
        untreated_count=prepared.untreated_count,
        influence_function=influence_func,
        p_cov=p_cov,
        cluster_ids=all_cluster_ids,
    )


def _summary_payload(
    curve: np.ndarray,
    dose_grid: list[float],
    label: str,
    *,
    overall_treated_support: float | None = None,
) -> dict[str, float]:
    uniform_support = float(np.mean(curve))
    return {
        f"overall_{label}": (
            float(overall_treated_support)
            if overall_treated_support is not None
            else uniform_support
        ),
        f"overall_{label}_uniform_support": uniform_support,
        f"dose_grid_mean_{label}": uniform_support,
        "dose_grid_min": float(min(dose_grid)),
        "dose_grid_max": float(max(dose_grid)),
    }


def _identification_payload(target_parameter: str) -> dict[str, str]:
    if target_parameter == "level":
        return {
            "paper_estimand": "ATT(d)",
            "identifying_assumption": "SPT",
            "ordinary_pt_interpretation": "LATT(d|d)",
            "identification_note": (
                "The same dose-specific contrast identifies LATT(d|d) under "
                "ordinary PT; interpreting it as ATT(d) requires SPT."
            ),
        }

    return {
        "paper_estimand": "ACRT(d)",
        "identifying_assumption": "SPT + continuous dose support",
        "ordinary_pt_interpretation": (
            "derivative of LATT(d|d) with local selection-bias contamination"
        ),
        "identification_note": (
            "Ordinary PT is not enough for a causal ACRT(d) interpretation; "
            "the public slope route reports the SPT-based causal-response label."
        ),
    }


def _common_metadata(
    *,
    dose_grid: list[float],
    untreated_benchmark: float,
    treated_count: int,
    untreated_count: int,
    treated_dose_mean: float,
    target_parameter: str,
    basis: dict[str, object],
    source_estimator: str,
    dose_est_method: str,
    control_group: str = "nevertreated",
) -> dict[str, object]:
    return {
        "target_parameter": target_parameter,
        "dose_grid": dose_grid,
        "untreated_benchmark": untreated_benchmark,
        "control_group": control_group,
        "basis": basis,
        "treated_sample": {
            "positive_dose_mean": treated_dose_mean,
            "treated_count": treated_count,
            "untreated_count": untreated_count,
        },
        "delta_outcome_construction": "last observed period minus first observed period",
        "derivative_construction": "shared derivative basis on the ATT(d) dose grid",
        "identification": _identification_payload(target_parameter),
        "source_estimator": source_estimator,
        "dose_est_method": dose_est_method,
        "inference": "bootstrap",
    }


def _build_cck_result(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    expected_target: str,
    dvals: Iterable[float] | float | None,
    assume_valid_panel: bool = False,
    adaptive: bool = False,
) -> ContDIDResult:
    prepared = _prepare_dose_sample(
        panel,
        spec,
        expected_target=expected_target,
        assume_valid_panel=assume_valid_panel,
    )
    if prepared.spec.dose_est_method != "cck":
        raise ContDIDValidationError("expected dose_est_method='cck'")

    dose_grid = (
        _default_cck_grid(prepared.treated_dose)
        if dvals is None
        else _coerce_grid(
            dvals,
            prepared.treated_dose,
            enforce_observed_support=True,
        )
    )
    backend = run_cck_backend(
        prepared.treated_outcome,
        prepared.treated_dose,
        dose_grid,
        untreated_delta=prepared.collapsed.loc[
            prepared.collapsed[prepared.validated_panel.dose_column] == 0.0,
            "delta_outcome",
        ].to_numpy(dtype=float),
        require_untreated_variance_df=expected_target == "level",
        bstrap=prepared.spec.bstrap,
        cband=prepared.spec.cband,
        alp=prepared.spec.alp,
        biters=prepared.spec.biters,
        boot_type=prepared.spec.boot_type,
        adaptive=adaptive,
    )

    if expected_target == "level":
        estimand = "ATT(d)"
        estimate = backend["att_curve"]
        std_error = backend["att_se"]
        critical_value = backend["att_crit"]
        confidence_interval = backend["att_interval"]
        confidence_band = backend["att_band"]
        summary = _summary_payload(
            np.asarray(estimate, dtype=float),
            dose_grid,
            "att",
            overall_treated_support=float(backend["overall_att"]),
        )
    else:
        estimand = "ACRT(d)"
        estimate = backend["acrt_curve"]
        std_error = backend["acrt_se"]
        critical_value = backend["acrt_crit"]
        confidence_interval = backend["acrt_interval"]
        confidence_band = backend["acrt_band"]
        summary = _summary_payload(
            np.asarray(estimate, dtype=float),
            dose_grid,
            "acrt",
            overall_treated_support=float(backend["overall_acrt"]),
        )

    metadata = _common_metadata(
        dose_grid=dose_grid,
        untreated_benchmark=prepared.untreated_benchmark,
        treated_count=prepared.treated_count,
        untreated_count=prepared.untreated_count,
        treated_dose_mean=float(prepared.treated_dose.mean()),
        target_parameter=expected_target,
        basis={
            "type": "cck_polynomial_backend",
            "degree": 2,
            "num_knots": 0,
            "interior_knots": [],
        },
        source_estimator="phase6_cck_backend",
        dose_est_method="cck",
        control_group=prepared.spec.control_group,
    )
    metadata.update(
        {
            "estimand": estimand,
            "summary": summary,
            "summary_aggregates": summary,
            "inference": ("analytic" if backend["bootstrap_type"] == "analytic" else "bootstrap"),
            "bootstrap_type": backend["bootstrap_type"],
            "bootstrap_seed": (
                backend["att_bootstrap_seed"]
                if expected_target == "level"
                else backend["acrt_bootstrap_seed"]
            ),
            "confidence_band_kind": backend["confidence_band_kind"],
            "critical_value": critical_value,
            "confidence_interval": confidence_interval,
            "confidence_band": confidence_band,
            "alp": prepared.spec.alp,
            "bstrap": prepared.spec.bstrap,
            "cband": prepared.spec.cband,
            "biters": prepared.spec.biters,
        }
    )

    return ContDIDResult(
        estimand=estimand,
        grid=dose_grid,
        estimate=estimate,
        std_error=std_error,
        critical_value=critical_value,
        confidence_interval=confidence_interval,
        confidence_band=confidence_band,
        metadata=metadata,
    )


def _build_parametric_result_from_fit(
    fit: _DoseRegressionFit,
    *,
    expected_target: str,
) -> ContDIDResult:
    grid = np.asarray(fit.dose_grid, dtype=float)
    _fit_xmin = float(np.min(fit.treated_dose))
    _fit_xmax = float(np.max(fit.treated_dose))
    if expected_target == "level":
        estimand = "ATT(d)"
        summary_label = "att"
        loadings = _build_design_matrix(
            grid, degree=fit.degree, knots=fit.knots, xmin=_fit_xmin, xmax=_fit_xmax
        )
        treated_loadings = _build_design_matrix(
            fit.treated_dose,
            degree=fit.degree,
            knots=fit.knots,
            xmin=_fit_xmin,
            xmax=_fit_xmax,
        )
    else:
        estimand = "ACRT(d)"
        summary_label = "acrt"
        loadings = _build_derivative_matrix(
            grid, degree=fit.degree, knots=fit.knots, xmin=_fit_xmin, xmax=_fit_xmax
        )
        treated_loadings = _build_derivative_matrix(
            fit.treated_dose,
            degree=fit.degree,
            knots=fit.knots,
            xmin=_fit_xmin,
            xmax=_fit_xmax,
        )

    # 理论边界防护：fit.p_cov 应始终为 0（协变量已在验证阶段被拦截）
    assert fit.p_cov == 0, (
        "Internal error: fit.p_cov > 0 should be impossible after validation guard"
    )

    curve = loadings @ fit.coefficients
    overall_treated_support = float(np.mean(treated_loadings @ fit.coefficients))
    metadata = _common_metadata(
        dose_grid=fit.dose_grid,
        untreated_benchmark=fit.untreated_benchmark,
        treated_count=fit.treated_count,
        untreated_count=fit.untreated_count,
        treated_dose_mean=fit.treated_dose_mean,
        target_parameter=expected_target,
        basis={
            "type": "bspline" if fit.knots else "global_polynomial",
            "degree": fit.degree,
            "num_knots": fit.num_knots,
            "interior_knots": fit.knots,
        },
        source_estimator="phase4_shared_dose_stack",
        dose_est_method="parametric",
        control_group=fit.spec.control_group,
    )
    metadata["estimand"] = estimand
    metadata["summary"] = _summary_payload(
        curve,
        fit.dose_grid,
        summary_label,
        overall_treated_support=overall_treated_support,
    )
    metadata["summary_aggregates"] = metadata["summary"]

    result = ContDIDResult(
        estimand=estimand,
        grid=fit.dose_grid,
        estimate=[float(value) for value in curve.tolist()],
        std_error=[0.0 for _ in fit.dose_grid],
        metadata=metadata,
    )
    covariance = fit.covariance
    if expected_target == "level":
        loadings, covariance = append_independent_mean_variance(
            loadings,
            covariance,
            mean_variance=estimate_mean_variance(fit.untreated_delta),
            loading_value=-1.0,
        )
    # Use clustered inference if cluster IDs are available
    if fit.cluster_ids is not None and fit.influence_function is not None:
        from statistics import NormalDist

        _CLUSTERED_BOOT_SEED = 20260407  # Match inference.py default seed
        cluster_ids_arr = np.array(fit.cluster_ids)

        if fit.spec.bstrap:
            boot_seed = _CLUSTERED_BOOT_SEED if fit.spec.cband else None
            boot_result = fit.influence_function.clustered_multiplier_bootstrap(
                cluster_ids_arr,
                biters=fit.spec.biters,
                alp=fit.spec.alp,
                cband=fit.spec.cband,
                seed=boot_seed,
                boot_type=fit.spec.boot_type,
            )
            std_error = np.asarray(boot_result["std_error"], dtype=float)
            critical_value = float(boot_result["critical_value"])
            pointwise_crit = float(boot_result["pointwise_critical_value"])
            boot_type = boot_result["bootstrap_type"]
            band_kind = boot_result["confidence_band_kind"]
            n_clusters = boot_result.get("n_clusters")
        else:
            # Analytical clustered SE without bootstrap
            boot_seed = None
            std_error = fit.influence_function.clustered_standard_error(cluster_ids_arr)
            pointwise_crit = float(NormalDist().inv_cdf(1.0 - fit.spec.alp / 2.0))
            critical_value = pointwise_crit
            boot_type = "none"
            band_kind = "pointwise_clustered_analytical"
            n_clusters = int(len(np.unique(cluster_ids_arr)))

        estimate = np.asarray(result.estimate, dtype=float)
        pointwise_interval = _build_interval_payload(estimate, std_error, pointwise_crit)
        confidence_band = build_confidence_band(estimate, std_error, critical_value=critical_value)

        normalized_estimate = [float(v) for v in estimate.tolist()]
        normalized_se = [float(s) for s in std_error.tolist()]
        result.estimate = normalized_estimate
        result.std_error = normalized_se
        result.critical_value = critical_value
        result.confidence_interval = pointwise_interval
        result.confidence_band = confidence_band
        result.metadata.update(
            {
                "estimand": result.estimand,
                "grid": result.grid,
                "estimate": normalized_estimate,
                "std_error": normalized_se,
                "inference": "bootstrap" if fit.spec.bstrap else "analytic",
                "alp": fit.spec.alp,
                "bstrap": fit.spec.bstrap,
                "cband": fit.spec.cband,
                "biters": fit.spec.biters,
                "bootstrap_seed": boot_seed,
                "bootstrap_type": boot_type,
                "confidence_band_kind": band_kind,
                "critical_value": critical_value,
                "confidence_interval": pointwise_interval,
                "confidence_band": confidence_band,
                "n_clusters": n_clusters,
                "cluster_column": fit.spec.cluster_column,
            }
        )
        return result

    return attach_inference_payload(
        result, loadings=loadings, covariance=covariance, spec=fit.spec
    )


def estimate_dose_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    knot_method: str = "quantile",
    adaptive: bool = False,
) -> ContDIDResult:
    """Estimate dose-response effects ATT(d) under continuous treatment.

    Fits a B-spline regression of outcome changes on continuous dose to recover
    the average treatment effect on the treated at each dose level d.

    Parameters
    ----------
    panel : PanelData
        Validated panel data object containing unit, time, outcome, dose, and
        group columns.
    spec : ContDIDSpec
        Estimation specification including control group strategy, inference
        parameters, and estimation method.
    dvals : array-like or float or None, optional
        Evaluation grid for dose values. If None, uses quantiles of the
        positive dose distribution (10th--99th percentile, 1% step, 89 points).
    degree : int, default 3
        Polynomial degree of B-spline basis (3 = cubic).
    num_knots : int, default 0
        Number of interior knots. If 0, uses degree+1 basis functions.
    knot_method : str, default "quantile"
        Interior knot placement strategy:

        - "quantile": knots at quantiles of positive dose distribution.
          Adapts to data density; recommended when doses cluster.
        - "even": evenly-spaced knots between min and max dose.
          Uniform spacing; matches R package's choose_knots_even().
    adaptive : bool, default False
        If True, use Lepski adaptive dimension selection (CCK backend).
        Overrides num_knots with data-driven choice. Only valid when
        spec.dose_est_method == "cck".

    Returns
    -------
    ContDIDResult
        Result object containing:

        - estimate : dose-response curve ATT(d) estimates
        - grid : evaluation points (dose values)
        - std_error : standard errors (if inference enabled)
        - confidence_band : simultaneous confidence band (if cband=True)

    Raises
    ------
    ContDIDValidationError
        If spec parameters are invalid or panel data fails validation.
    NotImplementedError
        If covariates are specified (not yet supported).

    See Also
    --------
    estimate_dose_level_effects : Alias focusing on ATT(d) level.
    estimate_dose_slope_effects : Estimates ACRT(d) marginal effects.
    cont_did : Unified entry point with routing logic.

    Examples
    --------
    >>> from contdid import PanelData, ContDIDSpec, estimate_dose_effects
    >>> spec = ContDIDSpec(target_parameter="level")
    >>> result = estimate_dose_effects(panel, spec)
    >>> result.estimate  # ATT(d) estimates at default grid
    """
    validated_panel = validate_panel_data(panel)
    validated_spec = validate_spec(
        spec,
        panel=validated_panel,
        assume_valid_panel=True,
    )
    if validated_spec.dose_est_method == "cck":
        return _build_cck_result(
            validated_panel,
            validated_spec,
            expected_target="level",
            dvals=dvals,
            assume_valid_panel=True,
            adaptive=adaptive,
        )

    fit = _fit_shared_dose_design(
        validated_panel,
        validated_spec,
        expected_target="level",
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        knot_method=knot_method,
        assume_valid_panel=True,
    )
    return _build_parametric_result_from_fit(fit, expected_target="level")


def estimate_dose_level_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    knot_method: str = "quantile",
) -> ContDIDResult:
    """Estimate ATT(d) level effects (alias for estimate_dose_effects).

    Equivalent to ``estimate_dose_effects`` with target_parameter="level".
    Provided for semantic clarity when calling code explicitly targets
    dose-level effects.

    Parameters
    ----------
    panel : PanelData
        Validated panel data object.
    spec : ContDIDSpec
        Estimation specification.
    dvals : array-like or float or None, optional
        Evaluation grid for dose values.
    degree : int, default 3
        B-spline polynomial degree.
    num_knots : int, default 0
        Number of interior knots.
    knot_method : str, default "quantile"
        Knot placement strategy ("quantile" or "even").

    Returns
    -------
    ContDIDResult
        Result with ATT(d) point estimates and inference.

    See Also
    --------
    estimate_dose_effects : Full documentation of shared parameters.
    estimate_dose_slope_effects : ACRT(d) marginal effects.
    """
    return estimate_dose_effects(
        panel,
        spec,
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        knot_method=knot_method,
    )


def estimate_dose_slope_effects(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    knot_method: str = "quantile",
) -> ContDIDResult:
    """Estimate ACRT(d) marginal effects (dose-response curve slope).

    Estimates the average causal response to treatment ACRT(d) — the derivative
    of ATT(d) with respect to dose. Measures how the treatment effect changes
    at the margin for each dose level d.

    Parameters
    ----------
    panel : PanelData
        Validated panel data object.
    spec : ContDIDSpec
        Estimation specification. When spec.dose_est_method == "cck",
        uses CCK sieve estimation for the slope.
    dvals : array-like or float or None, optional
        Evaluation grid for dose values.
    degree : int, default 3
        B-spline polynomial degree.
    num_knots : int, default 0
        Number of interior knots.
    knot_method : str, default "quantile"
        Knot placement strategy ("quantile" or "even").

    Returns
    -------
    ContDIDResult
        Result with ACRT(d) point estimates and inference.

    See Also
    --------
    estimate_dose_effects : ATT(d) level effects.
    estimate_dose_level_effects : Alias for level effects.
    cont_did : Unified entry point with routing logic.
    """
    validated_panel = validate_panel_data(panel)
    validated_spec = validate_spec(
        spec,
        panel=validated_panel,
        assume_valid_panel=True,
    )
    if validated_spec.dose_est_method == "cck":
        return _build_cck_result(
            validated_panel,
            validated_spec,
            expected_target="slope",
            dvals=dvals,
            assume_valid_panel=True,
        )

    fit = _fit_shared_dose_design(
        validated_panel,
        validated_spec,
        expected_target="slope",
        dvals=dvals,
        degree=degree,
        num_knots=num_knots,
        knot_method=knot_method,
        assume_valid_panel=True,
    )
    return _build_parametric_result_from_fit(fit, expected_target="slope")


# ---------------------------------------------------------------------------
# Multi-period staggered design: g×t loop + shared-multiplier bootstrap
# ---------------------------------------------------------------------------


def _estimate_gt_cell(
    dy_treated: np.ndarray,
    dose_treated: np.ndarray,
    dy_control: np.ndarray,
    dvals: np.ndarray,
    degree: int,
    knots: list[float],
    xmin: float,
    xmax: float,
) -> dict | None:
    """单个 (g,t) 格子的 2×2 DiD 估计。

    对应 Julia ContDiD.jl est_gt() 函数。
    返回 None 如果样本量不足。
    """
    n_t = len(dy_treated)
    n_c = len(dy_control)

    B_treat = _build_design_matrix(dose_treated, degree=degree, knots=knots, xmin=xmin, xmax=xmax)
    n_basis = B_treat.shape[1]
    if n_t < n_basis or n_c == 0:
        return None

    # OLS: beta = (B'B)^{-1} B' dy_treated
    BtB = B_treat.T @ B_treat
    if np.linalg.matrix_rank(BtB) < n_basis:
        return None
    BtB_inv = np.linalg.inv(BtB)
    beta = BtB_inv @ (B_treat.T @ dy_treated)

    # 对照组均值和残差
    ctrl_mean = float(np.mean(dy_control))
    residuals = dy_treated - B_treat @ beta
    ctrl_devs = dy_control - ctrl_mean

    # 评估网格上的 ATT(d) 和 ACRT(d)
    B_grid = _build_design_matrix(dvals, degree=degree, knots=knots, xmin=xmin, xmax=xmax)
    dB_grid = _build_derivative_matrix(dvals, degree=degree, knots=knots, xmin=xmin, xmax=xmax)

    att_d = B_grid @ beta - ctrl_mean
    acrt_d = dB_grid @ beta

    # Overall 标量（在观测剂量上的均值）
    att_overall = float(np.mean(B_treat @ beta)) - ctrl_mean
    dB_treat = _build_derivative_matrix(
        dose_treated, degree=degree, knots=knots, xmin=xmin, xmax=xmax
    )
    acrt_overall = float(np.mean(dB_treat @ beta))

    # HC0 sandwich variance for beta
    Sigma_beta = BtB_inv @ (B_treat.T @ np.diag(residuals**2) @ B_treat) @ BtB_inv
    var_ctrl = float(np.var(dy_control, ddof=1)) / n_c if n_c > 1 else 0.0

    att_var = np.sum((B_grid @ Sigma_beta) * B_grid, axis=1) + var_ctrl
    att_d_se = np.sqrt(np.maximum(att_var, 0.0))
    acrt_var = np.sum((dB_grid @ Sigma_beta) * dB_grid, axis=1)
    acrt_d_se = np.sqrt(np.maximum(acrt_var, 0.0))

    return {
        "att_d": att_d,
        "acrt_d": acrt_d,
        "att_d_se": att_d_se,
        "acrt_d_se": acrt_d_se,
        "att_overall": att_overall,
        "acrt_overall": acrt_overall,
        "beta": beta,
        "n_treated": n_t,
        "n_control": n_c,
        "residuals": residuals,
        "ctrl_devs": ctrl_devs,
        "BtB_inv": BtB_inv,
        "B_treat": B_treat,
        "B_grid": B_grid,
        "dB_grid": dB_grid,
        "treated_ids_idx": None,  # filled later for bootstrap
        "control_ids_idx": None,
    }


def _multiperiod_bootstrap(
    gt_results: list[dict],
    weights: np.ndarray,
    all_unit_ids: np.ndarray,
    unit_id_to_idx: dict,
    gt_unit_maps: list[dict],
    target: str,
    biters: int,
    alp: float,
    seed: int | None,
) -> dict:
    """共享乘子 bootstrap（对应 Julia bootstrap_se）。

    所有 (g,t) 格子共享同一组单位级乘子 V_i，确保跨格子相关性被正确捕获。
    """
    rng = np.random.default_rng(seed)
    W = weights.sum()
    n_grid = len(gt_results[0]["att_d"])
    n_cells = len(gt_results)
    n_units = len(all_unit_ids)

    boot_curves = np.zeros((biters, n_grid))

    for b_iter in _get_progress_bar(range(biters), biters, "Multiperiod Bootstrap"):
        # 共享乘子：所有单位抽一次
        V = rng.standard_normal(n_units)

        agg_curve = np.zeros(n_grid)
        for cell_idx in range(n_cells):
            r = gt_results[cell_idx]
            um = gt_unit_maps[cell_idx]
            w = weights[cell_idx]

            treated_global_idx = um["treated_idx"]
            control_global_idx = um["control_idx"]

            V_treated = V[treated_global_idx]
            V_control = V[control_global_idx]

            n_t = r["n_treated"]
            n_c = r["n_control"]

            # 处理组的 influence: BtB_inv @ B_i * resid_i * V_i
            resid = r["residuals"]
            BtB_inv = r["BtB_inv"]
            B_treat = r["B_treat"]

            # beta 的 bootstrap 扰动
            perturbed_Bty = B_treat.T @ (resid * V_treated)
            delta_beta = BtB_inv @ perturbed_Bty / np.sqrt(n_t)

            # 对照组均值的 bootstrap 扰动
            ctrl_devs = r["ctrl_devs"]
            delta_ctrl = float(np.sum(ctrl_devs * V_control)) / np.sqrt(n_c) if n_c > 0 else 0.0

            if target == "level":
                B_grid = r["B_grid"]
                cell_boot = B_grid @ delta_beta - delta_ctrl
            else:
                dB_grid = r["dB_grid"]
                cell_boot = dB_grid @ delta_beta

            agg_curve += cell_boot * (w / W)

        boot_curves[b_iter] = agg_curve

    # 计算 SE 和 sup-t 临界值
    std_error = np.std(boot_curves, axis=0, ddof=0)
    # 避免除零
    std_error_safe = np.where(std_error > 0, std_error, 1.0)
    t_stats = np.abs(boot_curves) / std_error_safe[None, :]
    sup_t = np.max(t_stats, axis=1)
    critical_value = float(np.quantile(sup_t, 1.0 - alp))

    # pointwise critical value
    pointwise_crit = float(
        np.quantile(np.abs(boot_curves / std_error_safe[None, :]), 1.0 - alp / 2.0, axis=0).max()
    )
    # 用更标准的方式：逐点分位数
    from statistics import NormalDist

    pointwise_crit = float(NormalDist().inv_cdf(1.0 - alp / 2.0))

    return {
        "std_error": std_error,
        "critical_value": critical_value,
        "pointwise_critical_value": pointwise_crit,
        "boot_curves": boot_curves,
    }


def estimate_dose_effects_multiperiod(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: Iterable[float] | float | None = None,
    degree: int = 3,
    num_knots: int = 0,
    anticipation: int = 0,
    seed: int | None = None,
) -> ContDIDResult:
    """多期错开设计的连续剂量 ATT(d) 估计。

    实现 Chen-Christensen-Kankanala (2025) Section 4 的多期扩展：
    ATT(d) = Σ_{g,t: t≥g} [n_{g,t}/N] × ATT_{g,t}(d)

    Parameters
    ----------
    panel : PanelData
        平衡面板数据（支持 >2 个时间周期）
    spec : ContDIDSpec
        估计规格（target_parameter, control_group 等）
    dvals : array-like or None
        评估网格；None 使用正剂量分位数
    degree : int
        B-spline 阶数（默认 3）
    num_knots : int
        内部节点数（默认 0 = 全局多项式）
    anticipation : int
        预期效应期数（默认 0）
    seed : int or None
        Bootstrap 随机种子
    """
    validated_panel = validate_panel_data(panel)
    # 多期路径支持 nevertreated 和 notyettreated，跳过 two-period 和 control_group 限制
    _supported_controls = ("nevertreated", "notyettreated")
    if spec.control_group not in _supported_controls:
        raise ContDIDValidationError(
            f"multiperiod dose estimation supports control_group in "
            f"{_supported_controls}; got {spec.control_group!r}"
        )
    # 仅做基本 spec 字段检查，不触发 panel 级别的 two-period 限制
    validated_spec = spec

    frame = validated_panel.frame
    id_col = validated_panel.id_column
    time_col = validated_panel.time_column
    outcome_col = validated_panel.outcome_column
    group_col = validated_panel.group_column
    dose_col = validated_panel.dose_column

    # 1. 确定处理组和时间周期
    treated_mask = frame[group_col] != 0
    all_positive_doses = frame.loc[treated_mask & (frame[dose_col] > 0), dose_col].to_numpy(
        dtype=float
    )
    if all_positive_doses.size == 0:
        raise ContDIDValidationError("No treated units with positive dose found")

    groups = sorted(int(g) for g in frame.loc[treated_mask, group_col].unique() if g != 0)
    time_periods = sorted(int(t) for t in frame[time_col].unique())
    time_period_set = set(time_periods)

    # 2. 构建共享 knot vector
    knots = _coerce_knots(all_positive_doses, num_knots)
    dmin = float(np.min(all_positive_doses))
    dmax = float(np.max(all_positive_doses))

    # 3. 构建评估网格
    if dvals is None:
        probs = np.linspace(0.01, 0.99, 90)
        dvals_arr = np.quantile(all_positive_doses, probs)
    else:
        dvals_arr = np.atleast_1d(np.asarray(dvals, dtype=float))
    dvals_arr = np.clip(dvals_arr, dmin, dmax)
    # 去重并排序
    dvals_arr = np.unique(dvals_arr)

    # 4. 为 bootstrap 收集所有单位 ID
    all_unit_ids = np.array(sorted(frame[id_col].unique()))
    unit_id_to_idx = {uid: i for i, uid in enumerate(all_unit_ids)}

    # 5. g×t 循环
    gt_results: list[dict] = []
    gt_unit_maps: list[dict] = []

    for g in groups:
        for tp in time_periods:
            if tp < g:
                continue
            base_period = g - anticipation - 1
            if base_period not in time_period_set:
                continue

            # 确定对照组
            if validated_spec.control_group == "notyettreated":
                comp_mask = (frame[group_col] > tp) | (frame[group_col] == 0)
            else:  # nevertreated
                comp_mask = frame[group_col] == 0
            grp_mask = frame[group_col] == g

            # 获取相关单位 ID
            treated_unit_ids = frame.loc[grp_mask, id_col].unique()
            control_unit_ids = frame.loc[comp_mask, id_col].unique()

            # 提取 base_period 和 tp 的数据
            time_mask = frame[time_col].isin([base_period, tp])

            # 构建 ΔY
            dy_treated_list = []
            dose_treated_list = []
            treated_uid_list = []
            dy_control_list = []
            control_uid_list = []

            # 处理组
            for uid in treated_unit_ids:
                u = frame[(frame[id_col] == uid) & time_mask]
                if len(u) != 2:
                    continue
                pre = u[u[time_col] == base_period]
                post = u[u[time_col] == tp]
                if pre.empty or post.empty:
                    continue
                dy = float(post[outcome_col].iloc[0]) - float(pre[outcome_col].iloc[0])
                dose = float(post[dose_col].iloc[0])
                if dose > 0:
                    dy_treated_list.append(dy)
                    dose_treated_list.append(dose)
                    treated_uid_list.append(uid)
                else:
                    dy_control_list.append(dy)
                    control_uid_list.append(uid)

            # 对照组
            for uid in control_unit_ids:
                u = frame[(frame[id_col] == uid) & time_mask]
                if len(u) != 2:
                    continue
                pre = u[u[time_col] == base_period]
                post = u[u[time_col] == tp]
                if pre.empty or post.empty:
                    continue
                dy = float(post[outcome_col].iloc[0]) - float(pre[outcome_col].iloc[0])
                dy_control_list.append(dy)
                control_uid_list.append(uid)

            if not dy_treated_list or not dy_control_list:
                continue

            dy_treated = np.array(dy_treated_list)
            dose_treated = np.array(dose_treated_list)
            dy_control = np.array(dy_control_list)

            result = _estimate_gt_cell(
                dy_treated,
                dose_treated,
                dy_control,
                dvals_arr,
                degree,
                knots,
                dmin,
                dmax,
            )
            if result is not None:
                result["g"] = g
                result["t"] = tp
                gt_results.append(result)
                # 记录单位在全局数组中的索引
                gt_unit_maps.append(
                    {
                        "treated_idx": np.array([unit_id_to_idx[uid] for uid in treated_uid_list]),
                        "control_idx": np.array([unit_id_to_idx[uid] for uid in control_uid_list]),
                    }
                )

    if not gt_results:
        raise ContDIDValidationError(
            "No (g,t) pairs produced results — check your data has "
            "sufficient treated and comparison units in each cell"
        )

    # 6. n_treated 加权聚合
    weights = np.array([r["n_treated"] for r in gt_results], dtype=float)
    W = weights.sum()

    target = validated_spec.target_parameter
    if target == "level":
        att_d = sum(r["att_d"] * w for r, w in zip(gt_results, weights)) / W
        att_d_se = np.sqrt(sum((r["att_d_se"] * w) ** 2 for r, w in zip(gt_results, weights))) / W
        estimate = att_d
        std_error = att_d_se
        estimand = "ATT(d)"
        summary_label = "att"
    else:
        acrt_d = sum(r["acrt_d"] * w for r, w in zip(gt_results, weights)) / W
        acrt_d_se = (
            np.sqrt(sum((r["acrt_d_se"] * w) ** 2 for r, w in zip(gt_results, weights))) / W
        )
        estimate = acrt_d
        std_error = acrt_d_se
        estimand = "ACRT(d)"
        summary_label = "acrt"

    # 7. Bootstrap SE（覆盖 analytical SE）
    confidence_interval = None
    confidence_band = None
    critical_value = None

    if validated_spec.bstrap and validated_spec.biters > 0:
        boot = _multiperiod_bootstrap(
            gt_results=gt_results,
            weights=weights,
            all_unit_ids=all_unit_ids,
            unit_id_to_idx=unit_id_to_idx,
            gt_unit_maps=gt_unit_maps,
            target=target,
            biters=validated_spec.biters,
            alp=validated_spec.alp,
            seed=seed,
        )
        std_error = boot["std_error"]
        critical_value = boot["critical_value"]
        pointwise_crit = boot["pointwise_critical_value"]

        # Pointwise CI
        est_arr = np.asarray(estimate, dtype=float)
        se_arr = np.asarray(std_error, dtype=float)
        ci_lower = est_arr - pointwise_crit * se_arr
        ci_upper = est_arr + pointwise_crit * se_arr
        confidence_interval = [[float(lo), float(hi)] for lo, hi in zip(ci_lower, ci_upper)]

        # Uniform band
        band_lower = est_arr - critical_value * se_arr
        band_upper = est_arr + critical_value * se_arr
        confidence_band = {
            "lower": [float(v) for v in band_lower.tolist()],
            "upper": [float(v) for v in band_upper.tolist()],
            "critical_value": critical_value,
        }

    # 8. Overall aggregates
    if target == "level":
        overall = sum(r["att_overall"] * w for r, w in zip(gt_results, weights)) / W
    else:
        overall = sum(r["acrt_overall"] * w for r, w in zip(gt_results, weights)) / W

    # 9. Metadata
    total_treated = int(sum(r["n_treated"] for r in gt_results))
    total_control = int(sum(r["n_control"] for r in gt_results))
    dose_grid_list = [float(v) for v in dvals_arr.tolist()]

    metadata = _common_metadata(
        dose_grid=dose_grid_list,
        untreated_benchmark=0.0,
        treated_count=total_treated,
        untreated_count=total_control,
        treated_dose_mean=float(np.mean(all_positive_doses)),
        target_parameter=target,
        basis={
            "type": "bspline" if knots else "global_polynomial",
            "degree": degree,
            "num_knots": len(knots),
            "interior_knots": knots,
        },
        source_estimator="multiperiod_staggered_gt_loop",
        dose_est_method="parametric",
        control_group=validated_spec.control_group,
    )
    metadata["estimand"] = estimand
    metadata["multiperiod"] = {
        "n_gt_cells": len(gt_results),
        "groups": groups,
        "gt_pairs": [(r["g"], r["t"]) for r in gt_results],
        "cell_weights": [float(w) for w in weights.tolist()],
        "anticipation": anticipation,
    }
    metadata["summary"] = _summary_payload(
        np.asarray(estimate, dtype=float),
        dose_grid_list,
        summary_label,
        overall_treated_support=overall,
    )
    metadata["summary_aggregates"] = metadata["summary"]
    if validated_spec.bstrap:
        metadata["inference"] = "bootstrap"
        metadata["alp"] = validated_spec.alp
        metadata["bstrap"] = True
        metadata["biters"] = validated_spec.biters
        metadata["bootstrap_seed"] = seed
        metadata["bootstrap_type"] = "shared_multiplier"
        metadata["confidence_band_kind"] = "sup_t_shared_multiplier"
        metadata["critical_value"] = critical_value
        metadata["confidence_interval"] = confidence_interval
        metadata["confidence_band"] = confidence_band

    # 10. 构建 ContDIDResult
    return ContDIDResult(
        estimand=estimand,
        grid=dose_grid_list,
        estimate=[float(v) for v in np.asarray(estimate).tolist()],
        std_error=[float(v) for v in np.asarray(std_error).tolist()],
        critical_value=critical_value,
        confidence_interval=confidence_interval,
        confidence_band=confidence_band,
        metadata=metadata,
    )
    return ContDIDResult(
        estimand=estimand,
        grid=dose_grid_list,
        estimate=[float(v) for v in np.asarray(estimate).tolist()],
        std_error=[float(v) for v in np.asarray(std_error).tolist()],
        critical_value=critical_value,
        confidence_interval=confidence_interval,
        confidence_band=confidence_band,
        metadata=metadata,
    )
