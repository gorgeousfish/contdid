from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOSE_ESTIMATION_MANIFEST_PATH = (
    REPO_ROOT / "reproduction" / "phase4_dose_estimation" / "manifest.json"
)
REFERENCE_MANIFEST_PATH = (
    REPO_ROOT / "reproduction" / "phase4_parametric_reference" / "manifest.json"
)
REFERENCE_FIXTURE_PATH = (
    REPO_ROOT / "reproduction" / "phase4_parametric_reference" / "phase4_parametric_reference.json"
)
NUMERICAL_TRUTH_PATH = (
    REPO_ROOT / "contdid-py" / "contracts" / "phase2" / "numerical_truth_contract_v1.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_registry() -> dict[str, int]:
    numerical_truth = _load_json(NUMERICAL_TRUTH_PATH)["numerical_truth"]
    return {entry["dgp_id"]: entry["default_seed"] for entry in numerical_truth["seed_registry"]}


def test_phase4_reference_fixture_uses_phase2_seed_registry_and_scenario_order() -> None:
    reference_manifest = _load_json(REFERENCE_MANIFEST_PATH)
    reference_fixture = _load_json(REFERENCE_FIXTURE_PATH)

    expected_ids = [entry["id"] for entry in reference_manifest["oracle_scenarios"]]
    curves = {entry["scenario_id"]: entry for entry in reference_fixture["oracle_curves"]}
    handoff = reference_manifest["phase4_handoff"]
    seed_registry = _seed_registry()

    assert expected_ids == [
        "SIM-001-null-dose",
        "SIM-002-linear-dose",
        "SIM-003-quadratic-dose",
    ]
    assert list(curves) == expected_ids

    for scenario_id in expected_ids:
        curve = curves[scenario_id]
        assert curve["default_seed"] == seed_registry[scenario_id]
        assert curve["phase4_required_targets"] == handoff["phase4_required_targets"]
        assert curve["deferred_targets"] == handoff["deferred_targets"]


def test_phase4_oracle_curves_match_closed_form_att_and_acrt_formulae() -> None:
    reference_manifest = _load_json(REFERENCE_MANIFEST_PATH)
    reference_fixture = _load_json(REFERENCE_FIXTURE_PATH)

    scenario_effects = {
        entry["id"]: entry["dose_effects"] for entry in reference_manifest["oracle_scenarios"]
    }
    evaluation_grid = reference_fixture["evaluation_grid"]
    curves = {entry["scenario_id"]: entry for entry in reference_fixture["oracle_curves"]}

    for scenario_id, effects in scenario_effects.items():
        linear = effects["dose_linear_effect"]
        quadratic = effects["dose_quadratic_effect"]
        curve = curves[scenario_id]

        expected_att = [linear * dose + quadratic * (dose**2) for dose in evaluation_grid]
        expected_acrt = [linear + 2 * quadratic * dose for dose in evaluation_grid]
        expected_overall_att = linear / 2 + quadratic / 3
        expected_overall_acrt = linear + quadratic

        assert curve["att_curve"] == pytest.approx(expected_att)
        assert curve["acrt_curve"] == pytest.approx(expected_acrt)
        assert curve["overall_att"] == pytest.approx(expected_overall_att)
        assert curve["overall_acrt"] == pytest.approx(expected_overall_acrt)
        assert curve["overall_att_uniform_support"] == pytest.approx(expected_overall_att)
        assert curve["overall_acrt_uniform_support"] == pytest.approx(expected_overall_acrt)


def test_phase4_dose_estimation_manifest_handoff_stays_in_sync_with_reference_fixture() -> None:
    dose_manifest = _load_json(DOSE_ESTIMATION_MANIFEST_PATH)
    reference_manifest = _load_json(REFERENCE_MANIFEST_PATH)
    reference_fixture = _load_json(REFERENCE_FIXTURE_PATH)

    comparison_contract = dose_manifest["comparison_contract"]
    package_handoff = dose_manifest["package_handoff"]
    reference_handoff = reference_manifest["phase4_handoff"]

    assert package_handoff["owner_plan"] == reference_handoff["owner_plan"]
    assert package_handoff["fixture_path"] == reference_handoff["package_fixture_path"]
    assert package_handoff["test_path"] == reference_handoff["package_test_path"]
    assert comparison_contract["phase4_required_targets"] == reference_handoff[
        "phase4_required_targets"
    ]
    assert comparison_contract["deferred_targets"] == reference_handoff["deferred_targets"]
    assert dose_manifest["source_contracts"]["parametric_reference_fixture"] == (
        "reproduction/phase4_parametric_reference/phase4_parametric_reference.json"
    )
    assert reference_manifest["shared_panel_contract"]["package_parity_public_dose_surface"] == {
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "sample_size": 64000,
        "seed_rule": "use scenario default_seed",
        "rationale": (
            "package-side parity must evaluate the closed-form dose oracle on the "
            "supported public two-period dose surface instead of the simulator's "
            "default four-period staggered panel"
        ),
    }
    assert [entry["scenario_id"] for entry in reference_fixture["oracle_curves"]] == [
        "SIM-001-null-dose",
        "SIM-002-linear-dose",
        "SIM-003-quadratic-dose",
    ]
