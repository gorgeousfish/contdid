from __future__ import annotations

import json
from pathlib import Path

import pytest

from contdid import ContDIDSpec, estimate_dose_effects, estimate_dose_slope_effects, estimate_eventstudy_effects, simulate_contdid_data
from contdid.validation import ContDIDValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase8_monte_carlo_expected.json"
PHASE7_SMOKE_PATH = REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke_manifest.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixture() -> dict:
    return _load_json(FIXTURE_PATH)


def _make_dose_spec(*, target_parameter: str, biters: int) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method="cck",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=biters,
    )


def _make_eventstudy_spec(*, target_parameter: str) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
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
    for key, expected in surface["metadata_expectations"].items():
        assert result.metadata[key] == expected


def test_sim005_cck_stress_surfaces_match_phase8_fixture() -> None:
    fixture = _load_fixture()["scenarios"]["SIM-005-cck-two-period"]
    panel = simulate_contdid_data(
        n=fixture["sample_size"],
        dgp_id="SIM-005-cck-two-period",
        seed=fixture["default_seed"],
    )

    level_result = estimate_dose_effects(
        panel,
        _make_dose_spec(target_parameter="level", biters=fixture["biters"]),
        dvals=fixture["level"]["grid"],
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_dose_spec(target_parameter="slope", biters=fixture["biters"]),
        dvals=fixture["slope"]["grid"],
    )

    _assert_surface_matches(fixture["level"], level_result)
    _assert_surface_matches(fixture["slope"], slope_result)


def test_phase8_stress_fixture_keeps_scaffold_labels_honest_and_cck_eventstudy_succeeds() -> None:
    """CCK eventstudy with fixed dimension should now succeed."""
    fixture = _load_fixture()
    smoke_manifest = _load_json(PHASE7_SMOKE_PATH)
    scenario = fixture["scenarios"]["SIM-005-cck-two-period"]
    panel = simulate_contdid_data(
        n=scenario["sample_size"],
        dgp_id="SIM-005-cck-two-period",
        seed=scenario["default_seed"],
    )

    assert fixture["empirical_reference"] == {
        "mode": "descriptive-or-scaffold-only",
        "source_manifest": "reproduction/medicare_pps/e2e_smoke_manifest.json",
        "parity_claim_allowed": False,
    }
    assert smoke_manifest["execution_modes"]["descriptive-or-scaffold-only"]["parity_claim_allowed"] is False

    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        degree=2,
        num_knots=0,
    )
    assert result.estimand == "ATT(event_time)"
    assert result.metadata["dose_est_method"] == "cck"
