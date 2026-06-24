"""Validation helpers for panel/spec inputs.

Provides runtime validation of :class:`PanelData` and :class:`ContDIDSpec`
objects against the theoretical requirements of continuous-dose DiD estimation.

Validation checks are organized into three severity levels:

- **ERROR**: Violations of core identification assumptions (arXiv-2107.02637v7).
  These always block execution.
- **WARNING**: Issues that may affect result quality but do not break identification.
  Blocked in STRICT mode, printed as warnings in NORMAL mode.
- **INFO**: Diagnostic information about data characteristics.
  Never blocks execution.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._asset_paths import resolve_runtime_asset
from .data import PanelData
from .specs import ContDIDSpec, load_runtime_enum_map

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Validation severity and strictness enums
# ---------------------------------------------------------------------------


class ValidationSeverity(Enum):
    """验证检查的严重级别。"""

    ERROR = auto()  # 违反论文算法原理，必须修正
    WARNING = auto()  # 可能影响结果质量，建议修正
    INFO = auto()  # 诊断信息，不影响执行


class ValidationStrictness(Enum):
    """验证严格度 - 控制哪些级别的检查会阻止执行。"""

    STRICT = auto()  # Error + Warning 都阻止执行（默认，适合正式分析）
    NORMAL = auto()  # 仅 Error 阻止执行，Warning 打印警告
    LENIENT = auto()  # 仅 Error 阻止执行，Warning 和 Info 静默


# ---------------------------------------------------------------------------
# Validation result containers
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """单个验证问题。"""

    severity: ValidationSeverity
    check_name: str
    message: str
    details: dict | None = None


@dataclass
class ValidationReport:
    """完整验证报告。"""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """返回所有 ERROR 级别的问题。"""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """返回所有 WARNING 级别的问题。"""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        """返回所有 INFO 级别的问题。"""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    @property
    def is_valid(self) -> bool:
        """无 Error 级问题则视为有效。"""
        return len(self.errors) == 0

    def raise_if_invalid(
        self, strictness: ValidationStrictness = ValidationStrictness.STRICT
    ) -> None:
        """根据严格度决定是否抛出异常。

        Parameters
        ----------
        strictness : ValidationStrictness
            STRICT: Error 或 Warning 都抛出异常（按原始顺序第一个阻止性问题）
            NORMAL: 仅 Error 抛出异常，Warning 打印 warnings
            LENIENT: 仅 Error 抛出异常，Warning/Info 静默
        """
        # Determine which severity levels block execution
        if strictness == ValidationStrictness.STRICT:
            blocking_severities = {ValidationSeverity.ERROR, ValidationSeverity.WARNING}
        else:
            blocking_severities = {ValidationSeverity.ERROR}

        # Find the first blocking issue in original order
        first_blocking = None
        for issue in self.issues:
            if issue.severity in blocking_severities:
                first_blocking = issue
                break

        if first_blocking is not None:
            raise ContDIDValidationError(first_blocking.message)

        # Emit warnings in NORMAL mode
        if strictness == ValidationStrictness.NORMAL and self.warnings:
            for w in self.warnings:
                warnings.warn(
                    f"[contdid] {w.check_name}: {w.message}",
                    UserWarning,
                    stacklevel=3,
                )

    def __str__(self) -> str:
        lines = []
        for issue in self.issues:
            prefix = issue.severity.name
            lines.append(f"  [{prefix}] {issue.check_name}: {issue.message}")
        if not lines:
            return "ValidationReport: all checks passed"
        return "ValidationReport:\n" + "\n".join(lines)


_MANIFEST_PATH = resolve_runtime_asset(
    package_relative="reproduction/simulate_contdid/manifest.json",
    repo_relative="reproduction/simulate_contdid/manifest.json",
)
_TIME_PERIOD_GRID_ERROR = (
    "time-period values must be finite numeric values on a consecutive integer grid"
)
_ID_NONMISSING_ERROR = (
    "id values must be nonmissing before balanced-panel sorting and first/last "
    "differences are formed"
)
_OUTCOME_NUMERIC_ERROR = (
    "outcome values must be finite numeric values before first/last differences are formed"
)
_DOSE_NUMERIC_ERROR = (
    "dose values must be nonnegative and finite numeric values so D = 0 denotes "
    "untreated units and D > 0 defines positive-dose treated support"
)
_GROUP_TIMING_NUMERIC_ERROR = (
    "group timing values must be nonnegative and finite numeric values so G = 0 "
    "denotes never-treated units and G > 0 defines treatment timing"
)
_DOSE_CONTROL_GROUP_ERROR = (
    "dose aggregation supports control_group values 'nevertreated' and 'notyettreated' only"
)
_EVENTSTUDY_CONTROL_GROUP_ERROR = (
    "event-study aggregation supports control_group values 'notyettreated' and 'nevertreated' only"
)


class ContDIDValidationError(ValueError):
    """Raised when contdid package inputs violate runtime contracts.

    Inherits from ValueError so users can catch it with either
    ``ContDIDValidationError`` or the more generic ``ValueError``.
    """


@lru_cache(maxsize=1)
def _load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _required_columns() -> tuple[str, ...]:
    return tuple(_load_manifest()["panel_contract"]["columns"])


def _panel_required_columns(panel: PanelData) -> tuple[str, ...]:
    return (
        panel.id_column,
        panel.time_column,
        panel.outcome_column,
        panel.group_column,
        panel.dose_column,
    )


@lru_cache(maxsize=1)
def _allowed_enum_map() -> dict[str, tuple[str, ...]]:
    return load_runtime_enum_map()


def _check_required_columns(panel: PanelData) -> None:
    missing = [
        column for column in _panel_required_columns(panel) if column not in panel.frame.columns
    ]
    if missing:
        joined = ", ".join(missing)
        raise ContDIDValidationError(f"panel is missing required columns: {joined}")


def _contains_boolean_values(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return True
    # Fast path: non-object dtypes (int, float, datetime, etc.) cannot
    # hold Python bool instances — skip expensive element-level iteration.
    if not pd.api.types.is_object_dtype(values):
        return False
    # Object-dtype column: must check elements, but use early-exit sampling.
    raw_values = values.to_numpy(dtype=object, copy=False)
    return any(isinstance(value, (bool, np.bool_)) for value in raw_values)


def _check_nonmissing_ids(panel: PanelData) -> None:
    if panel.frame[panel.id_column].isna().any():
        raise ContDIDValidationError(_ID_NONMISSING_ERROR)


def _check_balanced_panel(panel: PanelData) -> None:
    """Verify that the panel is balanced (every unit observed in every period).

    Theoretical requirement (arXiv-2107.02637v7):
    The identification of ATT(d) and ACRT(d) relies on computing first
    differences \u0394Y_i = Y_{i,post} - Y_{i,pre} for every unit i. This
    requires observing each unit in both the base period and the target
    period. The paper's Random Sampling assumption (Assumption 1, p.365)
    implicitly requires a balanced panel structure:

        "The observed data consist of {Y_{i,t=2}, Y_{i,t=1}, D_i}_{i=1}^n"

    In the multi-period extension (Assumption 1-MP, p.1412):

        "The observed data consists of {Y_{i1}, ..., Y_{iT}, D_i, G_i}_{i=1}^n"

    This is a SUBSTANTIVE requirement (not merely technical): without
    balanced observations, the difference \u0394Y cannot be computed for
    missing (id, time) pairs, breaking the fundamental identification
    strategy.

    Practical workarounds for unbalanced raw data:
    - Subset to the largest balanced sub-panel before calling contdid
    - Use pandas: df.groupby(id_col).filter(lambda g: len(g) == T)
    - Impute missing periods if missingness is ignorable (external to contdid)

    References:
    - Assumption 1 (Random Sampling): arXiv-2107.02637v7, p.365
    - Assumption 1-MP (Multiple Periods): arXiv-2107.02637v7, p.1412
    - R package implementation: contdid-r/R/cont_did.R, lines 157-165
    """
    frame = panel.frame
    id_column = panel.id_column
    time_column = panel.time_column
    time_values = frame[time_column]
    if _contains_boolean_values(time_values):
        raise ContDIDValidationError(_TIME_PERIOD_GRID_ERROR)
    if not pd.api.types.is_numeric_dtype(time_values):
        raise ContDIDValidationError(_TIME_PERIOD_GRID_ERROR)

    numeric_time = pd.to_numeric(time_values, errors="coerce")
    time_array = numeric_time.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(time_array).all() or not np.equal(time_array, np.rint(time_array)).all():
        raise ContDIDValidationError(_TIME_PERIOD_GRID_ERROR)

    if frame.duplicated([id_column, time_column]).any():
        raise ContDIDValidationError("panel must be balanced with unique id/time pairs")

    time_support = sorted({int(value) for value in time_array.tolist()})
    if not time_support:
        raise ContDIDValidationError("panel must contain at least one observed time period")
    expected_support = list(range(min(time_support), max(time_support) + 1))
    if time_support != expected_support:
        raise ContDIDValidationError("panel time support must lie on a consecutive integer grid")

    counts = frame.groupby(id_column)[time_column].nunique()
    if not counts.eq(len(time_support)).all():
        raise ContDIDValidationError("panel must be balanced across all observed time periods")


def _require_numeric_dtype(values: pd.Series, *, message: str) -> None:
    if pd.api.types.is_numeric_dtype(values):
        return

    raise ContDIDValidationError(message)


def _check_unit_constancy(panel: PanelData) -> None:
    frame = panel.frame
    id_column = panel.id_column
    group_nunique = frame.groupby(id_column)[panel.group_column].nunique()
    if not group_nunique.le(1).all():
        raise ContDIDValidationError("panel violates within-unit G constancy")

    dose_nunique = frame.groupby(id_column)[panel.dose_column].nunique()
    if not dose_nunique.le(1).all():
        raise ContDIDValidationError("panel violates within-unit D constancy")


def _check_finite_outcomes(panel: PanelData) -> None:
    raw_outcome = panel.frame[panel.outcome_column]
    if _contains_boolean_values(raw_outcome):
        raise ContDIDValidationError(_OUTCOME_NUMERIC_ERROR)
    _require_numeric_dtype(raw_outcome, message=_OUTCOME_NUMERIC_ERROR)
    outcome_values = pd.to_numeric(raw_outcome, errors="coerce")
    if np.isfinite(outcome_values.to_numpy(dtype=float, copy=False)).all():
        return

    raise ContDIDValidationError(_OUTCOME_NUMERIC_ERROR)


def _check_nonnegative_dose(panel: PanelData) -> None:
    raw_dose = panel.frame[panel.dose_column]
    if _contains_boolean_values(raw_dose):
        raise ContDIDValidationError(_DOSE_NUMERIC_ERROR)
    _require_numeric_dtype(raw_dose, message=_DOSE_NUMERIC_ERROR)
    dose_values = pd.to_numeric(raw_dose, errors="coerce")
    if not np.isfinite(dose_values.to_numpy(dtype=float, copy=False)).all():
        raise ContDIDValidationError(_DOSE_NUMERIC_ERROR)

    if (dose_values < 0).any():
        raise ContDIDValidationError(_DOSE_NUMERIC_ERROR)


def _check_nonnegative_group_timing(panel: PanelData) -> None:
    raw_group = panel.frame[panel.group_column]
    if _contains_boolean_values(raw_group):
        raise ContDIDValidationError(_GROUP_TIMING_NUMERIC_ERROR)
    _require_numeric_dtype(raw_group, message=_GROUP_TIMING_NUMERIC_ERROR)
    group_values = pd.to_numeric(raw_group, errors="coerce")
    if not np.isfinite(group_values.to_numpy(dtype=float, copy=False)).all():
        raise ContDIDValidationError(_GROUP_TIMING_NUMERIC_ERROR)

    if (group_values < 0).any():
        raise ContDIDValidationError(_GROUP_TIMING_NUMERIC_ERROR)


def _check_group_timing_observed_integer_grid(panel: PanelData) -> None:
    time_values = pd.to_numeric(panel.frame[panel.time_column], errors="coerce")
    observed_support = sorted({int(value) for value in time_values.drop_duplicates().tolist()})
    first_observed = observed_support[0]
    last_observed = observed_support[-1]
    group_values = pd.to_numeric(panel.frame[panel.group_column], errors="coerce")
    positive_groups = group_values[group_values > 0].drop_duplicates().tolist()

    invalid_groups = []
    for value in positive_groups:
        if not float(value).is_integer():
            invalid_groups.append(value)
            continue
        group = int(value)
        if group < first_observed:
            invalid_groups.append(value)
            continue
        if group <= last_observed and group not in observed_support:
            invalid_groups.append(value)

    if not invalid_groups:
        return

    raise ContDIDValidationError(
        "group timing values must align with or follow the observed integer time-period grid so G = 0 denotes never-treated units, in-window G > 0 denotes realized treatment start periods, and G after the last observed period denotes comparison-only not-yet-treated timing"
    )


def _check_never_treated_rule(panel: PanelData) -> None:
    frame = panel.frame
    group_values = pd.to_numeric(frame[panel.group_column], errors="coerce")
    dose_values = pd.to_numeric(frame[panel.dose_column], errors="coerce")
    invalid_rows = frame[(group_values == 0) & (dose_values > 0)]
    if not invalid_rows.empty:
        raise ContDIDValidationError("never-treated units must have zero dose in every row")


def _check_treated_timing_positive_dose(panel: PanelData) -> None:
    frame = panel.frame
    group_values = pd.to_numeric(frame[panel.group_column], errors="coerce")
    dose_values = pd.to_numeric(frame[panel.dose_column], errors="coerce")
    # Groups beyond the last observed period are not-yet-treated and may have dose=0
    time_values = pd.to_numeric(frame[panel.time_column], errors="coerce")
    last_period = float(time_values.max())
    invalid_rows = frame[(group_values > 0) & (group_values <= last_period) & (dose_values <= 0)]
    if not invalid_rows.empty:
        raise ContDIDValidationError(
            "positive treatment timing must have positive dose so G > 0 cannot "
            "be silently pooled into the untreated benchmark"
        )


def _require_public_two_period_dose_panel(validated_panel: PanelData) -> None:
    time_count = validated_panel.frame[validated_panel.time_column].nunique()
    if time_count == 2:
        return

    raise ContDIDValidationError(
        "dose aggregation currently supports exactly two observed time periods only "
        "until checked multi-period dose timing semantics land on the public dose routes"
    )


def _require_public_two_period_dose_timing(
    validated_panel: PanelData, *, allow_future_groups: bool = False
) -> None:
    _require_public_two_period_dose_panel(validated_panel)
    time_support = sorted(
        pd.to_numeric(validated_panel.frame[validated_panel.time_column], errors="raise")
        .drop_duplicates()
        .to_numpy(dtype=float)
        .tolist()
    )
    post_period = float(time_support[-1])
    group_values = pd.to_numeric(
        validated_panel.frame[validated_panel.group_column], errors="raise"
    ).to_numpy(dtype=float, copy=False)
    invalid_groups = sorted(
        {
            float(value)
            for value in group_values
            if value > 0.0
            and value != post_period
            and not (allow_future_groups and value > post_period)
        }
    )
    if not invalid_groups:
        return

    raise ContDIDValidationError(
        "two-period dose aggregation requires positive treatment timing to start "
        "in the post period so first/last differences retain an untreated baseline; "
        f"expected G > 0 to equal the last observed time period {post_period:g}, "
        f"got {invalid_groups}"
    )


def _require_cck_two_period_post_timing(validated_panel: PanelData) -> None:
    time_support = sorted(
        pd.to_numeric(validated_panel.frame[validated_panel.time_column], errors="raise")
        .drop_duplicates()
        .to_numpy(dtype=float)
        .tolist()
    )
    post_period = float(time_support[-1])
    group_values = pd.to_numeric(
        validated_panel.frame[validated_panel.group_column], errors="raise"
    ).to_numpy(dtype=float, copy=False)
    invalid_groups = sorted(
        {float(value) for value in group_values if value > 0.0 and value != post_period}
    )
    if not invalid_groups:
        return

    raise ContDIDValidationError(
        "cck estimator requires positive treatment timing to start in the post period; "
        f"expected G > 0 to equal the last observed "
        f"time period {post_period:g}, got {invalid_groups}"
    )


def _check_inference_knobs(spec: ContDIDSpec) -> None:
    if isinstance(spec.alp, bool) or not isinstance(spec.alp, Real):
        raise ContDIDValidationError("alp must lie strictly between 0 and 1")
    if not np.isfinite(float(spec.alp)) or not 0.0 < float(spec.alp) < 1.0:
        raise ContDIDValidationError("alp must lie strictly between 0 and 1")

    if not isinstance(spec.bstrap, bool):
        raise ContDIDValidationError("bstrap must be a boolean")
    if not isinstance(spec.cband, bool):
        raise ContDIDValidationError("cband must be a boolean")

    if (
        isinstance(spec.biters, bool)
        or not isinstance(spec.biters, Integral)
        or int(spec.biters) <= 0
    ):
        raise ContDIDValidationError("biters must be a positive integer")


def _check_public_route_control_group(spec: ContDIDSpec) -> None:
    # Explicit rejection of eventuallytreated with theoretical justification
    if spec.control_group == "eventuallytreated":
        raise ContDIDValidationError(
            "control_group='eventuallytreated' is not supported. "
            "The paper (arXiv-2107.02637v7) does not provide theoretical "
            "justification for this control group choice. "
            "Use 'nevertreated' or 'notyettreated' instead."
        )
    if spec.aggregation == "dose" and spec.control_group not in {
        "nevertreated",
        "notyettreated",
    }:
        raise ContDIDValidationError(_DOSE_CONTROL_GROUP_ERROR)
    if spec.aggregation == "eventstudy" and spec.control_group not in {
        "notyettreated",
        "nevertreated",
    }:
        raise ContDIDValidationError(_EVENTSTUDY_CONTROL_GROUP_ERROR)


def _collect_check(
    fn,
    panel: PanelData,
    severity: ValidationSeverity,
    check_name: str,
) -> ValidationIssue | None:
    """Run a check function and capture any raised error as a ValidationIssue."""
    try:
        fn(panel)
        return None
    except ContDIDValidationError as exc:
        return ValidationIssue(
            severity=severity, check_name=check_name, message=str(exc)
        )
    except (ValueError, TypeError, KeyError) as exc:
        # Catch downstream data-access errors that are not wrapped yet
        return ValidationIssue(
            severity=severity, check_name=check_name, message=str(exc)
        )


def _collect_sample_diagnostics(panel: PanelData) -> ValidationIssue:
    """Collect sample size diagnostics as an INFO-level issue."""
    frame = panel.frame
    group_values = pd.to_numeric(frame[panel.group_column], errors="coerce")
    n_units = frame[panel.id_column].nunique()
    n_periods = frame[panel.time_column].nunique()
    n_treated = int((group_values > 0).groupby(frame[panel.id_column]).first().sum())
    n_control = n_units - n_treated
    dose_values = pd.to_numeric(frame[panel.dose_column], errors="coerce")
    treated_dose = dose_values[group_values > 0]

    details = {
        "n_units": n_units,
        "n_periods": n_periods,
        "n_treated": n_treated,
        "n_control": n_control,
        "treated_control_ratio": round(n_treated / max(n_control, 1), 3),
        "dose_mean": round(float(treated_dose.mean()), 4) if len(treated_dose) > 0 else None,
        "dose_std": round(float(treated_dose.std()), 4) if len(treated_dose) > 1 else None,
        "dose_min": round(float(treated_dose.min()), 4) if len(treated_dose) > 0 else None,
        "dose_max": round(float(treated_dose.max()), 4) if len(treated_dose) > 0 else None,
    }
    
    message = (
        f"{n_units} units ({n_treated} treated, {n_control} control), "
        f"{n_periods} periods"
    )
    
    # Dose cardinality diagnostic: emit INFO if treatment looks binary or very
    # low-cardinality, which may indicate the user should use a standard
    # binary DiD package instead.
    # NOTE: 论文 Assumption 4(a) 要求处理变量具有 Lebesgue 密度，二元/极低基数
    # 处理变量不满足此条件，连续处理估计器可能产生不稳定结果。
    if len(treated_dose) > 0:
        dose_nunique = int(treated_dose.nunique())
        details["dose_n_unique"] = dose_nunique
        if dose_nunique <= 2:
            message += (
                f"; dose has only {dose_nunique} unique value(s) among treated "
                "units \u2014 this resembles a binary treatment; consider using a "
                "standard binary DiD package (e.g. did/pydid) instead"
            )
            return ValidationIssue(
                severity=ValidationSeverity.INFO,
                check_name="dose_cardinality",
                message=message,
                details=details,
            )
    
    return ValidationIssue(
        severity=ValidationSeverity.INFO,
        check_name="sample_diagnostics",
        message=message,
        details=details,
    )


def validate_panel_data(
    panel: PanelData,
    spec: ContDIDSpec | None = None,
    strictness: ValidationStrictness = ValidationStrictness.STRICT,
) -> PanelData:
    """Validate balanced-panel and invariant assumptions before estimation.

    Checks are organized into three severity levels:

    - ERROR: Core identification requirements (balanced panel, unit constancy,
      never-treated rule, positive-dose rule). Always block execution.
    - WARNING: Quality-related checks (finite outcomes, nonnegative dose).
      Blocked in STRICT mode, warned in NORMAL mode.
    - INFO: Diagnostic information (time grid alignment, sample statistics).
      Never blocks execution.

    Parameters
    ----------
    panel : PanelData
        The PanelData object to validate.
    spec : ContDIDSpec, optional
        Estimation specification (reserved for future spec-related validation).
    strictness : ValidationStrictness
        Controls which severity levels block execution.

    Returns
    -------
    PanelData
        The same PanelData object (pass-through for chaining).

    Raises
    ------
    ContDIDValidationError
        When blocking issues are found (based on strictness).
    """
    report = validate_panel_data_report(panel, spec=spec)
    report.raise_if_invalid(strictness)
    return panel


def validate_panel_data_report(
    panel: PanelData,
    spec: ContDIDSpec | None = None,
) -> ValidationReport:
    """Run all panel validation checks and return a complete report.

    Unlike :func:`validate_panel_data`, this function never raises. It collects
    all issues into a :class:`ValidationReport` that the caller can inspect
    or act upon.

    Parameters
    ----------
    panel : PanelData
        The PanelData object to validate.
    spec : ContDIDSpec, optional
        Estimation specification (reserved for future spec-related validation).

    Returns
    -------
    ValidationReport
        Complete report with all issues found.
    """
    issues: list[ValidationIssue] = []

    # -----------------------------------------------------------------------
    # Phase 1: Structural prerequisites (ERROR) - cannot proceed if these fail
    # -----------------------------------------------------------------------
    issue = _collect_check(
        _check_required_columns, panel, ValidationSeverity.ERROR, "required_columns"
    )
    if issue is not None:
        issues.append(issue)
        return ValidationReport(issues=issues)

    issue = _collect_check(
        _check_nonmissing_ids, panel, ValidationSeverity.ERROR, "nonmissing_ids"
    )
    if issue is not None:
        issues.append(issue)
        return ValidationReport(issues=issues)

    # -----------------------------------------------------------------------
    # Phase 2: Remaining checks in ORIGINAL order for backward compat.
    # Each check is assigned its severity level but collected in the order
    # the old validate_panel_data() ran them. This ensures that when
    # raise_if_invalid is called with STRICT mode, the first blocking issue
    # matches the pre-refactor error message.
    # -----------------------------------------------------------------------
    ordered_checks: list[
        tuple[object, ValidationSeverity, str]
    ] = [
        (_check_balanced_panel, ValidationSeverity.ERROR, "balanced_panel"),
        (_check_unit_constancy, ValidationSeverity.ERROR, "unit_constancy"),
        (_check_finite_outcomes, ValidationSeverity.WARNING, "finite_outcomes"),
        (_check_nonnegative_dose, ValidationSeverity.WARNING, "nonnegative_dose"),
        (_check_nonnegative_group_timing, ValidationSeverity.ERROR, "nonnegative_group_timing"),
        (_check_group_timing_observed_integer_grid, ValidationSeverity.WARNING, "group_timing_grid_alignment"),
        (_check_never_treated_rule, ValidationSeverity.ERROR, "never_treated_rule"),
        (_check_treated_timing_positive_dose, ValidationSeverity.ERROR, "positive_dose_rule"),
    ]

    for fn, severity, name in ordered_checks:
        issue = _collect_check(fn, panel, severity, name)
        if issue is not None:
            issues.append(issue)

    # -----------------------------------------------------------------------
    # Phase 3: Sample diagnostics (INFO) - only if no ERROR-level issues
    # -----------------------------------------------------------------------
    if not any(i.severity == ValidationSeverity.ERROR for i in issues):
        try:
            diag = _collect_sample_diagnostics(panel)
            issues.append(diag)
        except Exception:
            pass  # diagnostics are best-effort

    return ValidationReport(issues=issues)


def validate_spec(
    spec: ContDIDSpec,
    *,
    panel: PanelData | None = None,
    assume_valid_panel: bool = False,
) -> ContDIDSpec:
    """Validate runtime enums, inference knobs, and unsupported combinations.

    Checks that all enum fields hold allowed values, inference parameters
    are in valid ranges, and that the spec is compatible with the panel
    (if provided).

    Args:
        spec: The ContDIDSpec object to validate.
        panel: Optional panel for cross-validation of spec/panel compatibility.
        assume_valid_panel: If True, skip panel validation (assumes already validated).

    Returns:
        The same ContDIDSpec object (pass-through for chaining).

    Raises:
        ContDIDValidationError: If any validation check fails.
    """

    # ------------------------------------------------------------------
    # 理论边界防护：协变量调整功能不可用
    # 论文(arXiv:2107.02637v7, §Extensions, line 781)仅给出条件平行趋势的
    # 概念框架，未提供完整的估计理论（影响函数修正公式、Bootstrap覆盖率保证）。
    # Robinson(1988)两步法在该论文中完全未提及。在完成必要的理论推导前禁用。
    # ------------------------------------------------------------------
    if spec.covariates is not None and len(spec.covariates) > 0:
        raise NotImplementedError(
            "Covariate conditioning is not available. "
            "The paper (arXiv:2107.02637v7, §Extensions) provides only a conceptual "
            "framework for conditional parallel trends without the full estimation "
            "theory: the influence function correction formula and bootstrap coverage "
            "guarantees are missing. This feature will not be enabled until the "
            "required theoretical derivations are completed. "
            "Please use covariates=None (the default)."
        )

    allowed = _allowed_enum_map()
    for field_name in (
        "target_parameter",
        "aggregation",
        "dose_est_method",
        "control_group",
    ):
        value = getattr(spec, field_name)
        if value not in allowed[field_name]:
            if field_name == "aggregation" and value == "none":
                raise ContDIDValidationError(
                    'aggregation == "none" is not supported; use "dose" or "eventstudy"'
                )
            raise ContDIDValidationError(
                f"unsupported {field_name}: {value!r}; allowed values are {allowed[field_name]}"
            )

    # NOTE: 论文(arXiv-2107.02637v7) Assumption 4 理论上覆盖了连续处理(a)和多值离散处理(b)两种情形，
    # 但当前 Python 实现仅完成了连续处理路径的估计器（参数化B-spline / CCK sieve）。
    # 多值离散处理的饱和回归估计器(论文 Eq.13)尚未实现，故在此拒绝。
    if spec.treatment_type == "discrete":
        raise ContDIDValidationError(
            "discrete treatment is not supported in the current implementation. "
            "The paper (arXiv-2107.02637v7, Assumption 4b) covers multi-valued "
            "discrete treatment theoretically, but the saturated-regression "
            "estimator (Eq. 13) has not been implemented yet."
        )
    if spec.treatment_type != "continuous":
        raise ContDIDValidationError("unsupported treatment_type; expected 'continuous'")
    if not isinstance(spec.anticipation, int) or spec.anticipation < 0:
        raise ContDIDValidationError("anticipation must be a non-negative integer")

    # Anticipation upper-bound check: ensure at least one treated cohort retains
    # an admissible base period (g - anticipation - 1 >= min observed time).
    # This is CGBS Assumption 3-MP(a) feasibility: without a valid pre-treatment
    # baseline, identification fails for all cohorts.
    # Only check for eventstudy aggregation — local 2x2 dose panels have recoded
    # groups (treated=2) and anticipation is already resolved at the outer level.
    if (
        panel is not None
        and spec.anticipation > 0
        and spec.aggregation == "eventstudy"
    ):
        _time_vals = pd.to_numeric(panel.frame[panel.time_column], errors="coerce")
        _group_vals = pd.to_numeric(panel.frame[panel.group_column], errors="coerce")
        _min_time = int(_time_vals.min())
        _max_time = int(_time_vals.max())
        _treated_groups = sorted(
            int(v) for v in _group_vals.unique() if 0 < int(v) <= _max_time
        )
        if _treated_groups:
            # Check that at least one treated group has a valid base period
            _any_valid = any(
                (g - spec.anticipation - 1) >= _min_time for g in _treated_groups
            )
            if not _any_valid:
                _min_g = min(_treated_groups)
                _max_allowed = _min_g - _min_time - 1
                raise ContDIDValidationError(
                    f"anticipation={spec.anticipation} is too large: no treated "
                    f"cohort has an admissible base period "
                    f"(need g - anticipation - 1 >= {_min_time}). "
                    f"Maximum anticipation for this panel is {_max_allowed} "
                    f"(min treated group={_min_g}, min time={_min_time})"
                )

    if spec.dose_est_method != "cck":
        _check_public_route_control_group(spec)
    validated_panel: PanelData | None = None
    if panel is not None and (
        spec.dose_est_method == "cck"
        or (spec.aggregation == "dose" and spec.dose_est_method == "parametric")
    ):
        validated_panel = panel if assume_valid_panel else validate_panel_data(panel)
    if (
        spec.aggregation == "dose"
        and spec.dose_est_method == "parametric"
        and validated_panel is not None
    ):
        _require_public_two_period_dose_timing(
            validated_panel,
            allow_future_groups=(spec.control_group == "notyettreated"),
        )
    _check_inference_knobs(spec)
    if spec.boot_type not in ("multiplier", "rademacher", "mammen"):
        raise ContDIDValidationError(
            f"boot_type must be 'multiplier', 'rademacher', or 'mammen'; got {spec.boot_type!r}"
        )

    if spec.dose_est_method == "cck":
        if validated_panel is not None:
            time_count = validated_panel.frame[validated_panel.time_column].nunique()
            group_values = pd.to_numeric(
                validated_panel.frame[validated_panel.group_column], errors="raise"
            )
            positive_groups = {
                float(value)
                for value in group_values.drop_duplicates().tolist()
                if float(value) > 0.0
            }
            if not positive_groups:
                raise ContDIDValidationError(
                    "cck estimator requires exactly one positive treatment timing cohort"
                )
            # For eventstudy aggregation, staggered adoption IS supported
            # (each local (g,t) comparison is a separate two-period problem).
            if spec.aggregation == "dose":
                if len(positive_groups) > 1:
                    raise ContDIDValidationError(
                        "cck estimator not supported with staggered adoption yet"
                    )
                if time_count != 2:
                    raise ContDIDValidationError(
                        "cck estimator not supported with more than two time periods. consider averaging across pre and post treatment periods"
                    )
                _require_cck_two_period_post_timing(validated_panel)
        if spec.aggregation not in ("dose", "eventstudy"):
            raise ContDIDValidationError(
                "cck estimator requires aggregation='dose' or aggregation='eventstudy'"
            )
        _check_public_route_control_group(spec)

    return spec
