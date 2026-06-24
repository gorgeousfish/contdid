"""Phase 5 timing-group and event-time preparation helpers."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import pandas as pd

from .data import PanelData
from .validation import ContDIDValidationError, validate_panel_data


_SUPPORTED_CONTROL_GROUPS = {"notyettreated", "nevertreated"}
_SUPPORTED_BASE_PERIOD_STRINGS = {"varying", "universal"}


def _coerce_base_period_mode(
    base_period: int | str | None,
) -> tuple[Literal["varying", "universal", "fixed"], int | None]:
    if base_period is None:
        return "varying", None  # type: ignore[return-value]
    if isinstance(base_period, str):
        if base_period not in _SUPPORTED_BASE_PERIOD_STRINGS:
            raise ContDIDValidationError(
                "event-study base_period must be None, 'varying', 'universal', "
                "or an observed integer time period"
            )
        return base_period, None  # type: ignore[return-value]
    if isinstance(base_period, bool) or not isinstance(base_period, Integral):
        raise ContDIDValidationError(
            "event-study base_period must be None, 'varying', 'universal', "
            "or an observed integer time period"
        )
    return "fixed", int(base_period)


def _comparison_mask(
    frame: pd.DataFrame,
    *,
    group_column: str,
    time_period: int,
    control_group: str,
    exclude_group: int | None = None,
) -> pd.Series:
    if control_group == "notyettreated":
        mask = (frame[group_column] == 0) | (frame[group_column] > time_period)
        if exclude_group is not None:
            mask &= frame[group_column] != exclude_group
        return mask
    if control_group == "nevertreated":
        return frame[group_column] == 0
    raise ContDIDValidationError(
        "timing-group preparation supports control_group values "
        "'notyettreated' and 'nevertreated' only"
    )


def prepare_timing_groups(
    panel: PanelData,
    *,
    control_group: str = "notyettreated",
    anticipation: int = 0,
    base_period: int | str | None = None,
    assume_valid_panel: bool = False,
) -> pd.DataFrame:
    """Normalize staggered timing groups into an explicit event-study support table.

    The returned table is one row per timing-group / event-time pair with machine-readable
    support semantics.  Pre-treatment rows use a varying base period (the immediately prior
    observed period), while post-treatment rows compare against the universal base period
    ``g - anticipation - 1``.  Passing ``base_period="universal"`` fixes every
    row for a cohort to that cohort-specific universal base period and omits the
    normalized reference cell.  When an integer ``base_period`` is supplied, all
    non-reference rows compare against that observed pre-treatment period; this
    covers paper reproductions such as Medicare PPS that report event studies
    relative to a named baseline year.  The ``support`` flag is explicit so
    Phase 5 parity and later inference phases can consume the same timing
    metadata instead of reconstructing it.
    """

    if control_group not in _SUPPORTED_CONTROL_GROUPS:
        raise ContDIDValidationError(
            "timing-group preparation supports control_group values "
            "'notyettreated' and 'nevertreated' only"
        )
    if not isinstance(anticipation, int) or anticipation < 0:
        raise ContDIDValidationError("anticipation must be a non-negative integer")

    validated_panel = panel if assume_valid_panel else validate_panel_data(panel)
    frame = validated_panel.frame
    time_support = sorted(frame[validated_panel.time_column].unique().tolist())
    min_time = int(time_support[0])
    max_time = int(time_support[-1])
    treated_groups = sorted(
        int(value)
        for value in frame[validated_panel.group_column].unique().tolist()
        if 0 < int(value) <= max_time
    )
    if not treated_groups:
        raise ContDIDValidationError(
            "timing-group preparation requires at least one treated cohort observed during the panel"
        )

    if control_group == "nevertreated":
        never_treated_count = int(
            frame.loc[
                frame[validated_panel.group_column] == 0, validated_panel.id_column
            ].nunique()
        )
        if never_treated_count == 0:
            raise ContDIDValidationError(
                "timing-group preparation requires never-treated comparison units"
            )

    base_period_mode, checked_base_period = _coerce_base_period_mode(base_period)
    if base_period_mode == "fixed":
        assert checked_base_period is not None
        if checked_base_period not in time_support:
            raise ContDIDValidationError(
                "event-study base_period must be an observed integer time period"
            )

    rows: list[dict[str, object]] = []
    omitted_no_base_groups: list[int] = []
    for timing_group in treated_groups:
        universal_base_period = timing_group - anticipation - 1
        if universal_base_period < min_time:
            omitted_no_base_groups.append(timing_group)
            continue
        if (
            base_period_mode == "fixed"
            and checked_base_period is not None
            and checked_base_period >= timing_group - anticipation
        ):
            omitted_no_base_groups.append(timing_group)
            continue

        treated_count = int(
            frame.loc[
                frame[validated_panel.group_column] == timing_group,
                validated_panel.id_column,
            ].nunique()
        )
        for time_period in time_support:
            if base_period_mode == "fixed":
                assert checked_base_period is not None
                if time_period == checked_base_period:
                    continue
                base_period = checked_base_period
            elif base_period_mode == "universal":
                if time_period == universal_base_period:
                    continue
                base_period = int(universal_base_period)
            elif time_period < (timing_group - anticipation):
                base_period = int(time_period - 1)
            else:
                base_period = int(universal_base_period)

            if base_period < min_time or base_period not in time_support:
                continue

            comparison_cutoff_period = max(int(time_period), int(base_period))
            comparison_count = int(
                frame.loc[
                    _comparison_mask(
                        frame,
                        group_column=validated_panel.group_column,
                        time_period=comparison_cutoff_period,
                        control_group=control_group,
                        exclude_group=int(timing_group),
                    ),
                    validated_panel.id_column,
                ].nunique()
            )
            rows.append(
                {
                    "timing_group": int(timing_group),
                    "time_period": int(time_period),
                    "base_period": base_period,
                    "event_time": int(time_period - timing_group),
                    "post_treatment": bool(time_period >= (timing_group - anticipation)),
                    "treated_count": treated_count,
                    "comparison_count": comparison_count,
                    "comparison_type": control_group,
                    "support": bool(treated_count > 0 and comparison_count > 0),
                }
            )

    prepared = pd.DataFrame(rows)
    if prepared.empty:
        if omitted_no_base_groups:
            raise ContDIDValidationError(
                "timing-group preparation found no treated cohorts with an admissible base period"
            )
        raise ContDIDValidationError(
            "timing-group preparation found no supported event-study rows"
        )

    return prepared.sort_values(["timing_group", "time_period", "event_time"]).reset_index(
        drop=True
    )


def build_event_time_index(prepared_timing_groups: pd.DataFrame) -> list[int]:
    """Return the sorted event-time support implied by ``prepare_timing_groups``."""

    if (
        "support" not in prepared_timing_groups.columns
        or "event_time" not in prepared_timing_groups.columns
    ):
        raise ContDIDValidationError(
            "timing-group table must contain 'event_time' and 'support' columns"
        )

    supported = prepared_timing_groups.loc[prepared_timing_groups["support"], "event_time"]
    if supported.empty:
        raise ContDIDValidationError("timing-group table has no supported event-time rows")
    return sorted(int(value) for value in supported.drop_duplicates().tolist())
