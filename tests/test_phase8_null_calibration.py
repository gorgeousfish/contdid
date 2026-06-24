from __future__ import annotations

import json
from pathlib import Path

import pytest

from contdid import ContDIDSpec, estimate_dose_effects, estimate_dose_slope_effects, estimate_eventstudy_effects, simulate_contdid_data


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase8_monte_carlo_expected.json"
PHASE2_CONTRACT_PATH = REPO_ROOT / "contdid-py" / "contracts" / "phase2" / "numerical_truth_contract_v1.json"
PHASE8_MANIFEST_PATH = REPO_ROOT / "reproduction" / "phase8_monte_carlo" / "manifest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixture() -> dict:
    return _load_json(FIXTURE_PATH)


def _phase8_execution_settings(scenario_id: str) -> dict:
    manifest = _load_json(PHASE8_MANIFEST_PATH)
    scenarios = {
        entry["scenario_id"]: entry["execution_settings"]
        for entry in manifest["scenario_matrix"]
    }
    return scenarios[scenario_id]


def _simulation_kwargs(settings: dict) -> dict:
    return {
        key: settings[key]
        for key in ("num_time_periods", "num_groups", "pg", "pu")
        if key in settings
    }


def _make_dose_spec(*, target_parameter: str, dose_est_method: str = "parametric", biters: int = 199) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method=dose_est_method,
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=biters,
    )


def _make_eventstudy_spec(*, target_parameter: str, dose_est_method: str = "parametric", biters: int = 199) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method=dose_est_method,
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=biters,
    )


def _assert_surface_matches(surface: dict, result) -> None:
    assert result.estimand == surface["estimand"]
    assert result.grid == surface["grid"]
    assert result.estimate == pytest.approx(surface["estimate"], abs=1e-10)
    assert result.std_error == pytest.approx(surface["std_error"], abs=1e-10)
    assert result.critical_value == pytest.approx(surface["critical_value"], abs=1e-10)
    assert result.confidence_band["critical_value"] == pytest.approx(
        surface["confidence_band"]["critical_value"], abs=1e-10
    )
    assert result.confidence_band["lower"] == pytest.approx(
        surface["confidence_band"]["lower"], abs=1e-10
    )
    assert result.confidence_band["upper"] == pytest.approx(
        surface["confidence_band"]["upper"], abs=1e-10
    )
    for key, expected in surface["summary"].items():
        assert result.metadata["summary"][key] == pytest.approx(expected, abs=1e-10)
    for key, expected in surface.get("metadata_expectations", {}).items():
        assert result.metadata[key] == expected


def test_phase8_fixture_covers_null_scenarios_and_phase2_tolerance_lineage() -> None:
    fixture = _load_fixture()
    numerical_truth = _load_json(PHASE2_CONTRACT_PATH)["numerical_truth"]
    manifest = _load_json(PHASE8_MANIFEST_PATH)

    assert fixture["schema_version"] == "0.1"
    assert fixture["application_id"] == "phase8-package-calibration-expected"
    assert fixture["source_registry"]["phase8_manifest"] == "reproduction/phase8_monte_carlo/manifest.json"
    assert fixture["source_registry"]["phase6_reference_fixture"] == "reproduction/phase6_inference/phase6_inference_reference.json"

    tolerance_registry = fixture["tolerance_registry"]
    phase2_families = {
        item["comparison_target"]: {
            "tolerance_mode": item["tolerance_mode"],
            "required_from_phase": item["required_from_phase"],
        }
        for item in numerical_truth["tolerance_families"]
    }
    assert tolerance_registry == phase2_families
    assert list(fixture["scenarios"]) == [
        "SIM-001-null-dose",
        "SIM-004-staggered-eventstudy-null",
        "SIM-005-cck-two-period",
    ]

    manifest_execution = {
        entry["scenario_id"]: entry["execution_settings"] for entry in manifest["scenario_matrix"]
    }
    sim001 = fixture["scenarios"]["SIM-001-null-dose"]
    sim004 = fixture["scenarios"]["SIM-004-staggered-eventstudy-null"]

    assert sim001["sample_size"] == manifest_execution["SIM-001-null-dose"]["sample_size"]
    assert manifest_execution["SIM-001-null-dose"]["num_time_periods"] == 2
    assert manifest_execution["SIM-001-null-dose"]["num_groups"] == 2
    assert sim001["level"]["grid"] == manifest_execution["SIM-001-null-dose"]["dvals"]
    assert sim004["sample_size"] == manifest_execution["SIM-004-staggered-eventstudy-null"]["sample_size"]
    assert sim004["level"]["grid"] == sim004["exact_rule_targets"]["event_time_grid"]


def test_sim001_null_dose_level_and_slope_match_phase8_fixture() -> None:
    fixture = _load_fixture()["scenarios"]["SIM-001-null-dose"]
    settings = _phase8_execution_settings("SIM-001-null-dose")
    panel = simulate_contdid_data(
        n=fixture["sample_size"],
        dgp_id="SIM-001-null-dose",
        seed=fixture["default_seed"],
        **_simulation_kwargs(settings),
    )

    level_result = estimate_dose_effects(
        panel,
        _make_dose_spec(target_parameter="level", biters=fixture["biters"]),
        dvals=fixture["level"]["grid"],
        degree=fixture["degree"],
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_dose_spec(target_parameter="slope", biters=fixture["biters"]),
        dvals=fixture["slope"]["grid"],
        degree=fixture["degree"],
    )

    _assert_surface_matches(fixture["level"], level_result)
    _assert_surface_matches(fixture["slope"], slope_result)


def test_sim004_null_eventstudy_reuses_phase6_oracle_chain() -> None:
    fixture = _load_fixture()["scenarios"]["SIM-004-staggered-eventstudy-null"]
    panel = simulate_contdid_data(
        n=fixture["sample_size"],
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=fixture["default_seed"],
    )

    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level", biters=fixture["biters"]),
        degree=fixture["degree"],
    )

    _assert_surface_matches(fixture["level"], result)
    assert result.metadata["support"] == fixture["exact_rule_targets"]["support"]
    assert result.metadata["event_time_grid"] == fixture["exact_rule_targets"]["event_time_grid"]
