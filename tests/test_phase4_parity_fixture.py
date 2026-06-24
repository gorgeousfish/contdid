from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_FIXTURE = (
    REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase4_parity_expected.json"
)
ROOT_ORACLE_MANIFEST = (
    REPO_ROOT / "reproduction" / "phase4_parametric_reference" / "manifest.json"
)
ROOT_ORACLE_FIXTURE = (
    REPO_ROOT / "reproduction" / "phase4_parametric_reference" / "phase4_parametric_reference.json"
)
SIMULATE_MANIFEST = REPO_ROOT / "reproduction" / "simulate_contdid" / "manifest.json"
NUMERICAL_TRUTH_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase2"
    / "numerical_truth_contract_v1.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_differences(values: list[float]) -> list[float]:
    return [next_value - value for value, next_value in zip(values, values[1:])]


def _second_differences(values: list[float]) -> list[float]:
    return _first_differences(_first_differences(values))


def _is_flat(values: list[float]) -> bool:
    return all(math.isclose(value, values[0], abs_tol=1e-12) for value in values[1:])


def _strictly_increasing(values: list[float]) -> bool:
    return all(next_value > value for value, next_value in zip(values, values[1:]))


def _constant_step(values: list[float]) -> bool:
    increments = _first_differences(values)
    return all(math.isclose(step, increments[0], abs_tol=1e-12) for step in increments[1:])


def _strictly_positive(values: list[float]) -> bool:
    return all(value > 0.0 for value in values)


def test_phase4_package_parity_fixture_tracks_phase2_truth_and_root_oracle() -> None:
    assert PACKAGE_FIXTURE.exists(), f"missing Phase 4 package fixture: {PACKAGE_FIXTURE}"
    assert ROOT_ORACLE_MANIFEST.exists(), f"missing Phase 4 oracle manifest: {ROOT_ORACLE_MANIFEST}"
    assert ROOT_ORACLE_FIXTURE.exists(), f"missing Phase 4 oracle fixture: {ROOT_ORACLE_FIXTURE}"
    assert SIMULATE_MANIFEST.exists(), f"missing simulate manifest: {SIMULATE_MANIFEST}"
    assert NUMERICAL_TRUTH_PATH.exists(), f"missing numerical truth contract: {NUMERICAL_TRUTH_PATH}"

    package_fixture = _load_json(PACKAGE_FIXTURE)
    oracle_manifest = _load_json(ROOT_ORACLE_MANIFEST)
    oracle_fixture = _load_json(ROOT_ORACLE_FIXTURE)
    simulate_manifest = _load_json(SIMULATE_MANIFEST)
    numerical_truth = _load_json(NUMERICAL_TRUTH_PATH)["numerical_truth"]

    assert package_fixture["schema_version"] == "0.1"
    assert package_fixture["phase"] == 4
    assert package_fixture["application_id"] == "phase4-package-parity-expected"
    assert package_fixture["evaluation_grid"] == oracle_fixture["evaluation_grid"]

    source_registry = package_fixture["source_registry"]
    assert source_registry == {
        "numerical_truth_contract": "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
        "root_oracle_manifest": "reproduction/phase4_parametric_reference/manifest.json",
        "root_oracle_fixture": "reproduction/phase4_parametric_reference/phase4_parametric_reference.json",
        "simulate_manifest": "reproduction/simulate_contdid/manifest.json",
        "phase4_plan": ".planning/phases/04-dose-level-slope-parametric-b-spline/04-03-PLAN.md",
    }

    public_execution = package_fixture["public_dose_surface_execution"]
    assert public_execution == {
        "sample_size": 64000,
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "seed_rule": "use scenario default_seed",
        "rationale": (
            "package-side parity must evaluate the closed-form dose oracle on the "
            "supported public two-period dose surface instead of the simulator's "
            "default four-period staggered panel"
        ),
    }
    assert public_execution == oracle_manifest["shared_panel_contract"][
        "package_parity_public_dose_surface"
    ]
    assert public_execution["num_time_periods"] != simulate_manifest["default_parameters"][
        "num_time_periods"
    ]
    assert public_execution["num_groups"] != simulate_manifest["default_parameters"][
        "num_groups"
    ]

    tolerance_families = {
        entry["comparison_target"]: entry for entry in numerical_truth["tolerance_families"]
    }
    comparison_target_contract = package_fixture["comparison_target_contract"]
    required_targets = comparison_target_contract["required_targets"]
    deferred_targets = comparison_target_contract["deferred_targets"]

    assert [entry["name"] for entry in required_targets] == oracle_manifest["phase4_handoff"][
        "phase4_required_targets"
    ]
    assert [entry["name"] for entry in deferred_targets] == oracle_manifest["phase4_handoff"][
        "deferred_targets"
    ]

    for entry in required_targets + deferred_targets:
        contract_entry = tolerance_families[entry["name"]]
        assert entry["tolerance_mode"] == contract_entry["tolerance_mode"]
        assert entry["required_from_phase"] == contract_entry["required_from_phase"]

    assert comparison_target_contract["required_targets"][0]["calibration_status"] == (
        "pending_estimator_rerun"
    )
    assert "fresh evidence" in comparison_target_contract["note"]
    assert "numerical-truth-contract.md" in comparison_target_contract["note"]


def test_phase4_package_parity_fixture_preserves_expected_curve_shapes() -> None:
    package_fixture = _load_json(PACKAGE_FIXTURE)
    oracle_manifest = _load_json(ROOT_ORACLE_MANIFEST)
    oracle_fixture = _load_json(ROOT_ORACLE_FIXTURE)

    package_scenarios = package_fixture["scenarios"]
    assert list(package_scenarios) == [
        "SIM-001-null-dose",
        "SIM-002-linear-dose",
        "SIM-003-quadratic-dose",
    ]

    oracle_curves = {entry["scenario_id"]: entry for entry in oracle_fixture["oracle_curves"]}
    oracle_shapes = {
        entry["id"]: entry["expected_shape"] for entry in oracle_manifest["oracle_scenarios"]
    }

    for scenario_id, payload in package_scenarios.items():
        oracle_payload = oracle_curves[scenario_id]
        assert payload["default_seed"] == oracle_payload["default_seed"]
        assert payload["phase4_required_targets"] == oracle_payload["phase4_required_targets"]
        assert payload["deferred_targets"] == oracle_payload["deferred_targets"]
        assert payload["expected_shape"] == oracle_shapes[scenario_id]
        assert payload["dose_grid"] == package_fixture["evaluation_grid"]
        assert payload["att_curve"] == oracle_payload["att_curve"]
        assert payload["acrt_curve"] == oracle_payload["acrt_curve"]
        assert payload["summary_aggregates"] == {
            "overall_att": oracle_payload["overall_att"],
            "overall_acrt": oracle_payload["overall_acrt"],
            "overall_att_uniform_support": oracle_payload["overall_att_uniform_support"],
            "overall_acrt_uniform_support": oracle_payload["overall_acrt_uniform_support"],
        }

    null_payload = package_scenarios["SIM-001-null-dose"]
    assert _is_flat(null_payload["att_curve"])
    assert _is_flat(null_payload["acrt_curve"])
    assert null_payload["shape_diagnostics"] == {
        "att_sign": "zero",
        "att_monotonicity": "flat",
        "acrt_sign": "zero",
        "acrt_monotonicity": "flat",
    }

    linear_payload = package_scenarios["SIM-002-linear-dose"]
    assert _strictly_increasing(linear_payload["att_curve"])
    assert _constant_step(linear_payload["att_curve"])
    assert _is_flat(linear_payload["acrt_curve"])
    assert _strictly_positive(linear_payload["acrt_curve"])
    assert linear_payload["shape_diagnostics"] == {
        "att_sign": "nonnegative",
        "att_monotonicity": "strictly increasing linear",
        "acrt_sign": "positive",
        "acrt_monotonicity": "flat positive",
    }

    quadratic_payload = package_scenarios["SIM-003-quadratic-dose"]
    assert _strictly_increasing(quadratic_payload["att_curve"])
    assert _strictly_increasing(quadratic_payload["acrt_curve"])
    assert _constant_step(quadratic_payload["acrt_curve"])
    second_differences = _second_differences(quadratic_payload["att_curve"])
    assert all(step > 0.0 for step in second_differences)
    assert _is_flat(second_differences)
    assert quadratic_payload["shape_diagnostics"] == {
        "att_sign": "positive",
        "att_monotonicity": "strictly increasing convex",
        "acrt_sign": "positive",
        "acrt_monotonicity": "strictly increasing linear",
    }
