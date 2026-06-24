from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MANIFEST_PATH = REPO_ROOT / "reproduction" / "phase3_validation" / "manifest.json"


def _load_manifest() -> dict:
    return json.loads(VALIDATION_MANIFEST_PATH.read_text(encoding="utf-8"))


def _make_frame(
    *,
    periods: tuple[int, ...] = (1, 2),
    groups: tuple[int, ...] = (2, 0, 2),
    doses: tuple[float, ...] = (0.3, 0.0, 0.8),
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for unit_id, (group, dose) in enumerate(zip(groups, doses, strict=True), start=1):
        for time_period in periods:
            rows.append(
                {
                    "id": unit_id,
                    "time_period": time_period,
                    "Y": float(unit_id * 10 + time_period),
                    "G": group,
                    "D": dose,
                }
            )
    return pd.DataFrame(rows)


def test_validation_manifest_spec_rules_stay_in_sync_with_runtime_enums() -> None:
    from contdid import ContDIDSpec, validate_spec

    manifest = _load_manifest()
    spec_rules = {entry["name"]: tuple(entry["allowed_values"]) for entry in manifest["spec_rules"]}

    assert spec_rules == {
        "target_parameter": ("level", "slope"),
        "aggregation": ("dose", "eventstudy"),
        "dose_est_method": ("parametric", "cck"),
        "control_group": ("notyettreated", "nevertreated", "eventuallytreated"),
    }

    dose_spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
    )
    assert validate_spec(dose_spec) is dose_spec

    for control_group in ("notyettreated", "nevertreated"):
        eventstudy_spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group=control_group,
        )
        assert validate_spec(eventstudy_spec) is eventstudy_spec


def test_validate_panel_data_enforces_phase3_validation_panel_invariants() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    manifest = _load_manifest()
    panel_rules = manifest["panel_rules"]
    assert panel_rules["required_columns"] == ["id", "time_period", "Y", "G", "D"]

    valid_panel = PanelData(frame=_make_frame())
    assert validate_panel_data(valid_panel) is valid_panel

    missing_column = _make_frame().drop(columns=["Y"])
    with pytest.raises(ContDIDValidationError, match="required columns"):
        validate_panel_data(PanelData(frame=missing_column))

    missing_id = _make_frame()
    missing_id.loc[missing_id["id"] == 1, "id"] = pd.NA
    with pytest.raises(ContDIDValidationError, match="id values must be nonmissing"):
        validate_panel_data(PanelData(frame=missing_id))

    group_drift = _make_frame()
    group_drift.loc[group_drift["id"] == 1, "G"] = [2, 3]
    with pytest.raises(ContDIDValidationError, match="G constancy"):
        validate_panel_data(PanelData(frame=group_drift))

    dose_drift = _make_frame()
    dose_drift.loc[dose_drift["id"] == 1, "D"] = [0.3, 0.6]
    with pytest.raises(ContDIDValidationError, match="D constancy"):
        validate_panel_data(PanelData(frame=dose_drift))

    off_grid = _make_frame(periods=(1, 3))
    with pytest.raises(ContDIDValidationError, match="integer grid"):
        validate_panel_data(PanelData(frame=off_grid))


def test_validate_panel_data_rejects_nonnumeric_time_periods_without_leaking_typeerror() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    invalid_time = _make_frame()
    invalid_time["time_period"] = invalid_time["time_period"].astype(object)
    invalid_time.loc[invalid_time["time_period"] == 2, "time_period"] = "bad"

    with pytest.raises(
        ContDIDValidationError,
        match="time-period values must be finite numeric values on a consecutive integer grid",
    ):
        validate_panel_data(PanelData(frame=invalid_time))


def test_validate_panel_data_accepts_paneldata_column_overrides() -> None:
    from contdid import PanelData, validate_panel_data

    frame = _make_frame().rename(
        columns={
            "id": "unit_id",
            "time_period": "period",
            "Y": "outcome",
            "G": "cohort",
            "D": "dose",
        }
    )
    panel = PanelData(
        frame=frame,
        id_column="unit_id",
        time_column="period",
        outcome_column="outcome",
        group_column="cohort",
        dose_column="dose",
    )

    assert validate_panel_data(panel) is panel


def test_validate_panel_data_rejects_negative_dose_support() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    negative_dose = _make_frame()
    negative_dose.loc[negative_dose["id"] == 1, "D"] = -0.3

    with pytest.raises(ContDIDValidationError, match="dose values must be nonnegative"):
        validate_panel_data(PanelData(frame=negative_dose))


@pytest.mark.parametrize("dose_value", [float("nan"), float("inf")])
def test_validate_panel_data_rejects_non_finite_dose_support(dose_value: float) -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    non_finite_dose = _make_frame()
    non_finite_dose.loc[non_finite_dose["id"] == 1, "D"] = dose_value

    with pytest.raises(
        ContDIDValidationError, match="dose values must be nonnegative and finite"
    ):
        validate_panel_data(PanelData(frame=non_finite_dose))


@pytest.mark.parametrize("group_value", [float("nan"), float("inf"), -1.0])
def test_validate_panel_data_rejects_invalid_group_timing(group_value: float) -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    invalid_group = _make_frame()
    invalid_group["G"] = invalid_group["G"].astype(float)
    invalid_group.loc[invalid_group["id"] == 1, "G"] = group_value

    with pytest.raises(
        ContDIDValidationError,
        match="group timing values must be nonnegative and finite",
    ):
        validate_panel_data(PanelData(frame=invalid_group))


@pytest.mark.parametrize("outcome_value", [float("nan"), float("inf")])
def test_validate_panel_data_rejects_non_finite_outcomes(outcome_value: float) -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    non_finite_outcome = _make_frame()
    non_finite_outcome.loc[
        (non_finite_outcome["id"] == 1) & (non_finite_outcome["time_period"] == 2),
        "Y",
    ] = outcome_value

    with pytest.raises(ContDIDValidationError, match="outcome values must be finite"):
        validate_panel_data(PanelData(frame=non_finite_outcome))


@pytest.mark.parametrize(
    ("column_name", "message"),
    [
        ("Y", "outcome values must be finite numeric values"),
        ("D", "dose values must be nonnegative and finite numeric values"),
        ("G", "group timing values must be nonnegative and finite numeric values"),
    ],
)
def test_validate_panel_data_rejects_numeric_string_core_columns(
    column_name: str, message: str
) -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    numeric_string_frame = _make_frame()
    numeric_string_frame[column_name] = numeric_string_frame[column_name].astype(str)

    with pytest.raises(ContDIDValidationError, match=message):
        validate_panel_data(PanelData(frame=numeric_string_frame))


@pytest.mark.parametrize(
    ("column_name", "message"),
    [
        (
            "time_period",
            "time-period values must be finite numeric values on a consecutive integer grid",
        ),
        ("Y", "outcome values must be finite numeric values"),
        ("D", "dose values must be nonnegative and finite numeric values"),
        ("G", "group timing values must be nonnegative and finite numeric values"),
    ],
)
def test_validate_panel_data_rejects_boolean_identification_columns(
    column_name: str, message: str
) -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    boolean_frame = _make_frame()
    boolean_frame[column_name] = boolean_frame[column_name].astype(bool)

    with pytest.raises(ContDIDValidationError, match=message):
        validate_panel_data(PanelData(frame=boolean_frame))


def test_validate_panel_data_rejects_treated_timing_without_positive_dose() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    treated_timing_zero_dose = _make_frame()
    treated_timing_zero_dose.loc[
        treated_timing_zero_dose["id"] == 1, "D"
    ] = 0.0

    with pytest.raises(
        ContDIDValidationError, match="positive treatment timing must have positive dose"
    ):
        validate_panel_data(PanelData(frame=treated_timing_zero_dose))


def test_validate_panel_data_rejects_group_timing_off_the_observed_integer_grid() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    invalid_group = _make_frame()
    invalid_group["G"] = invalid_group["G"].astype(float)
    invalid_group.loc[invalid_group["id"] == 1, "G"] = 1.5

    with pytest.raises(
        ContDIDValidationError,
        match="group timing values must align with or follow the observed integer time-period grid",
    ):
        validate_panel_data(PanelData(frame=invalid_group))


def test_validate_spec_rejects_negative_anticipation() -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    with pytest.raises(
        ContDIDValidationError, match="anticipation must be a non-negative integer"
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                anticipation=-1,
            )
        )


@pytest.mark.parametrize("biters", [True, 1.5, "199"])
def test_validate_spec_rejects_noninteger_bootstrap_iterations(biters: object) -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    with pytest.raises(
        ContDIDValidationError, match="biters must be a positive integer"
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                biters=biters,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("bstrap", 1, "bstrap must be a boolean"),
        ("cband", "yes", "cband must be a boolean"),
        ("alp", "0.1", "alp must lie strictly between 0 and 1"),
    ],
)
def test_validate_spec_rejects_mistyped_inference_knobs(
    field_name: str, value: object, message: str
) -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    kwargs = {
        "target_parameter": "level",
        "aggregation": "dose",
        "dose_est_method": "parametric",
        "control_group": "nevertreated",
        field_name: value,
    }

    with pytest.raises(ContDIDValidationError, match=message):
        validate_spec(ContDIDSpec(**kwargs))


@pytest.mark.parametrize("dose_est_method", ["parametric", "cck"])
def test_validate_spec_rejects_two_period_dose_timing_without_untreated_baseline(
    dose_est_method: str,
) -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, PanelData, validate_spec

    match = (
        "cck estimator requires positive treatment timing to start in the post period"
        if dose_est_method == "cck"
        else "two-period dose aggregation requires positive treatment timing to start in the post period"
    )
    with pytest.raises(
        ContDIDValidationError,
        match=match,
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method=dose_est_method,
                control_group="nevertreated",
            ),
            panel=PanelData(frame=_make_frame(groups=(1, 0, 1))),
        )


def test_validate_spec_rejects_cck_post_timing_drift_with_cck_error() -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, PanelData, validate_spec

    with pytest.raises(
        ContDIDValidationError,
        match="cck estimator requires positive treatment timing to start in the post period",
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="cck",
                control_group="nevertreated",
            ),
            panel=PanelData(frame=_make_frame(groups=(3, 0, 3))),
        )


def test_validate_spec_rejects_unchecked_dose_control_groups_before_estimation() -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="eventuallytreated",
            )
        )


def test_validate_spec_rejects_unchecked_eventstudy_control_group_before_estimation() -> (
    None
):
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "control_group='eventuallytreated' is not supported"
        ),
    ):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="eventstudy",
                dose_est_method="parametric",
                control_group="eventuallytreated",
            )
        )


def test_validate_spec_rejects_manifest_backed_unsupported_combinations() -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, PanelData, validate_spec

    unsupported_cases = {
        entry["id"]: entry for entry in _load_manifest()["unsupported_cases"]
    }
    assert set(unsupported_cases) == {
        "aggregation_none",
        "dose_anticipation_nonzero",
        "unbalanced_panel",
        "repeated_cross_sections",
        "time_varying_dose",
        "covariate_aware_identification",
        "discrete_treatment",
        "cck_staggered_adoption",
        "cck_requires_two_periods",
        "cck_eventstudy",
    }

    with pytest.raises(ContDIDValidationError, match="dose.*eventstudy"):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="none",
                dose_est_method="parametric",
                control_group="notyettreated",
            )
        )

    with pytest.raises(ContDIDValidationError, match="discrete treatment"):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="notyettreated",
                treatment_type="discrete",
            )
        )

    with pytest.raises(ContDIDValidationError, match="staggered adoption"):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="cck",
                control_group="nevertreated",
            ),
            panel=PanelData(frame=_make_frame(groups=(1, 0, 2))),
        )

    # CCK + eventstudy is now supported with fixed dimension
    validated = validate_spec(
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="notyettreated",
        )
    )
    assert validated.dose_est_method == "cck"
    assert validated.aggregation == "eventstudy"


@pytest.mark.parametrize("periods", [(1,), (1, 2, 3)])
def test_validate_spec_rejects_cck_when_panel_is_not_exactly_two_periods(
    periods: tuple[int, ...],
) -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, PanelData, validate_spec

    manifest = _load_manifest()
    cck_case = {
        entry["id"]: entry for entry in manifest["unsupported_cases"]
    }["cck_requires_two_periods"]
    assert (
        cck_case["predicate"]
        == 'dose_est_method == "cck" and the panel does not have exactly two unique time_period values'
    )

    with pytest.raises(ContDIDValidationError, match="two time periods"):
        validate_spec(
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="cck",
                control_group="nevertreated",
            ),
            panel=PanelData(
                frame=_make_frame(
                    periods=periods,
                    groups=(periods[-1], 0, periods[-1]),
                )
            ),
        )
