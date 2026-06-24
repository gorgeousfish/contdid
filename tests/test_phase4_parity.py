from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from contdid import (
    ContDIDSpec,
    estimate_dose_effects,
    estimate_dose_slope_effects,
    simulate_contdid_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_FIXTURE = (
    REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase4_parity_expected.json"
)
_POINT_ESTIMATE_ATOL = 0.05
_SUMMARY_ATOL = 0.04
_BASIS_BY_SCENARIO = {
    "SIM-001-null-dose": 1,
    "SIM-002-linear-dose": 1,
    "SIM-003-quadratic-dose": 2,
}


def _load_fixture() -> dict:
    return json.loads(PACKAGE_FIXTURE.read_text(encoding="utf-8"))


def _simulate_supported_public_dose_surface(
    scenario_id: str,
    *,
    fixture: dict,
    default_seed: int,
):
    execution = fixture["public_dose_surface_execution"]
    return simulate_contdid_data(
        n=int(execution["sample_size"]),
        num_time_periods=int(execution["num_time_periods"]),
        num_groups=int(execution["num_groups"]),
        pg=list(execution["pg"]),
        pu=float(execution["pu"]),
        dgp_id=scenario_id,
        seed=default_seed,
    )


def _strictly_increasing(values: list[float]) -> bool:
    return all(next_value > value for value, next_value in zip(values, values[1:]))


def _constant_step(values: list[float]) -> bool:
    steps = [next_value - value for value, next_value in zip(values, values[1:])]
    return all(math.isclose(step, steps[0], abs_tol=1e-12) for step in steps[1:])


def _flat(values: list[float], *, tol: float = 1e-12) -> bool:
    return all(math.isclose(value, values[0], abs_tol=tol) for value in values[1:])


def _shape_diagnostics(values: list[float], *, estimand: str) -> dict[str, str]:
    if max(abs(value) for value in values) <= _POINT_ESTIMATE_ATOL:
        return {f"{estimand}_sign": "zero", f"{estimand}_monotonicity": "flat"}

    if _flat(values, tol=_POINT_ESTIMATE_ATOL):
        sign = "positive" if values[0] > 0 else "negative"
        monotonicity = "flat" if sign == "zero" else f"flat {sign}"
        return {f"{estimand}_sign": sign, f"{estimand}_monotonicity": monotonicity}

    if estimand == "att":
        second_steps = [
            next_value - value
            for value, next_value in zip(
                [next_value - value for value, next_value in zip(values, values[1:])],
                [next_value - value for value, next_value in zip(values, values[1:])][
                    1:
                ],
            )
        ]
        if _strictly_increasing(values) and _constant_step(values):
            return {
                "att_sign": "nonnegative",
                "att_monotonicity": "strictly increasing linear",
            }
        if (
            _strictly_increasing(values)
            and second_steps
            and all(step > 0.0 for step in second_steps)
        ):
            return {
                "att_sign": "positive",
                "att_monotonicity": "strictly increasing convex",
            }
    if estimand == "acrt":
        if _flat(values) and all(value > 0.0 for value in values):
            return {"acrt_sign": "positive", "acrt_monotonicity": "flat positive"}
        if _strictly_increasing(values) and _constant_step(values):
            return {
                "acrt_sign": "positive",
                "acrt_monotonicity": "strictly increasing linear",
            }

    raise AssertionError(f"unexpected {estimand} shape: {values}")


@pytest.mark.parametrize(
    ("scenario_id", "expected_diagnostics"),
    [
        (
            "SIM-001-null-dose",
            {
                "att_sign": "zero",
                "att_monotonicity": "flat",
                "acrt_sign": "zero",
                "acrt_monotonicity": "flat",
            },
        ),
        (
            "SIM-002-linear-dose",
            {
                "att_sign": "nonnegative",
                "att_monotonicity": "strictly increasing linear",
                "acrt_sign": "positive",
                "acrt_monotonicity": "flat positive",
            },
        ),
        (
            "SIM-003-quadratic-dose",
            {
                "att_sign": "positive",
                "att_monotonicity": "strictly increasing convex",
                "acrt_sign": "positive",
                "acrt_monotonicity": "strictly increasing linear",
            },
        ),
    ],
)
def test_phase4_package_parity_matches_fixture_targets(
    scenario_id: str,
    expected_diagnostics: dict[str, str],
) -> None:
    fixture = _load_fixture()
    scenario = fixture["scenarios"][scenario_id]
    grid = fixture["evaluation_grid"]
    degree = _BASIS_BY_SCENARIO[scenario_id]
    panel = _simulate_supported_public_dose_surface(
        scenario_id,
        fixture=fixture,
        default_seed=int(scenario["default_seed"]),
    )

    level_result = estimate_dose_effects(
        panel,
        ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        dvals=grid,
        degree=degree,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
        ),
        dvals=grid,
        degree=degree,
    )

    assert level_result.grid == scenario["dose_grid"]
    assert slope_result.grid == scenario["dose_grid"]
    assert level_result.estimate == pytest.approx(
        scenario["att_curve"], abs=_POINT_ESTIMATE_ATOL
    )
    assert slope_result.estimate == pytest.approx(
        scenario["acrt_curve"], abs=_POINT_ESTIMATE_ATOL
    )
    assert level_result.metadata["summary"]["overall_att"] == pytest.approx(
        scenario["summary_aggregates"]["overall_att"], abs=_SUMMARY_ATOL
    )
    assert slope_result.metadata["summary"]["overall_acrt"] == pytest.approx(
        scenario["summary_aggregates"]["overall_acrt"], abs=_SUMMARY_ATOL
    )
    assert level_result.metadata["summary"][
        "overall_att_uniform_support"
    ] == pytest.approx(
        scenario["summary_aggregates"]["overall_att_uniform_support"], abs=_SUMMARY_ATOL
    )
    assert slope_result.metadata["summary"][
        "overall_acrt_uniform_support"
    ] == pytest.approx(
        scenario["summary_aggregates"]["overall_acrt_uniform_support"],
        abs=_SUMMARY_ATOL,
    )
    for result in (level_result, slope_result):
        assert result.metadata["inference"] == "bootstrap"
        assert result.metadata["bootstrap_type"] == "multiplier"
        assert result.critical_value is not None and result.critical_value > 0.0
        assert result.confidence_interval is not None
        assert result.confidence_band is not None

    diagnostics = {}
    diagnostics.update(_shape_diagnostics(level_result.estimate, estimand="att"))
    diagnostics.update(_shape_diagnostics(slope_result.estimate, estimand="acrt"))
    assert diagnostics == expected_diagnostics
