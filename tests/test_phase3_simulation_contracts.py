from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "contdid-py" / "src" / "contdid" / "contracts" / "phase2.py"
BUNDLE_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase2"
    / "phase2_contract_template.json"
)
NUMERICAL_TRUTH_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase2"
    / "numerical_truth_contract_v1.json"
)
MANIFEST_PATH = REPO_ROOT / "reproduction" / "simulate_contdid" / "manifest.json"
PACKAGE_MANIFEST_PATH = (
    REPO_ROOT / "contdid-py" / "reproduction" / "simulate_contdid" / "manifest.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_phase2_module():
    assert MODULE_PATH.exists(), f"missing contract module at {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("contdid.contracts.phase2", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase3_package_side_simulation_inputs_align_with_phase2_truth_bundle() -> None:
    assert BUNDLE_PATH.exists(), f"missing Phase 2 bundle: {BUNDLE_PATH}"
    assert NUMERICAL_TRUTH_PATH.exists(), f"missing numerical truth contract: {NUMERICAL_TRUTH_PATH}"
    assert MANIFEST_PATH.exists(), f"missing simulation manifest: {MANIFEST_PATH}"

    module = _load_phase2_module()
    bundle = module.validate_phase2_contract_bundle(_load_json(BUNDLE_PATH))
    numerical_truth = _load_json(NUMERICAL_TRUTH_PATH)["numerical_truth"]
    manifest = _load_json(MANIFEST_PATH)

    assert bundle["numerical_truth"] == numerical_truth

    source_r_function = manifest["source_r_function"]
    assert source_r_function["paper_truth_bundle"] == (
        "contdid-py/contracts/phase2/phase2_contract_template.json"
    )
    assert source_r_function["numerical_truth_contract"] == (
        "contdid-py/contracts/phase2/numerical_truth_contract_v1.json"
    )

    manifest_dgp_ids = [entry["id"] for entry in manifest["dgp_registry"]]
    assert manifest_dgp_ids == numerical_truth["dgp_ids"][:3]

    seed_registry = {
        record["dgp_id"]: record["default_seed"] for record in numerical_truth["seed_registry"]
    }
    assert {dgp_id: seed_registry[dgp_id] for dgp_id in manifest_dgp_ids} == {
        "SIM-001-null-dose": 1234,
        "SIM-002-linear-dose": 20260407,
        "SIM-003-quadratic-dose": 20261234,
    }


def test_phase3_packaged_simulation_manifest_matches_repo_truth() -> None:
    assert PACKAGE_MANIFEST_PATH.exists(), (
        f"missing packaged simulation manifest: {PACKAGE_MANIFEST_PATH}"
    )

    assert _load_json(PACKAGE_MANIFEST_PATH) == _load_json(MANIFEST_PATH)


def test_phase3_package_side_simulation_contracts_preserve_panel_invariants() -> None:
    manifest = _load_json(MANIFEST_PATH)

    defaults = manifest["default_parameters"]
    panel_contract = manifest["panel_contract"]
    derived_defaults = manifest["derived_defaults"]
    invariants = {record["id"] for record in manifest["invariants"]}

    assert panel_contract["columns"] == ["id", "time_period", "Y", "G", "D"]
    assert panel_contract["shape"] == "balanced_panel_long"
    assert panel_contract["row_count_formula"] == "n * num_time_periods"
    assert panel_contract["never_treated_rule"] == "G = 0 implies D = 0 for every row"
    assert panel_contract["time_period_rule"] == "time_period runs from 1 through num_time_periods"

    assert derived_defaults["row_count"] == defaults["n"] * defaults["num_time_periods"]
    assert derived_defaults["unique_ids"] == defaults["n"]
    assert derived_defaults["time_periods"] == list(
        range(1, defaults["num_time_periods"] + 1)
    )
    assert derived_defaults["default_group_support"] == [
        0,
        *range(2, defaults["num_time_periods"] + 1),
    ]

    assert invariants == {
        "balanced_panel",
        "unit_level_group_constancy",
        "unit_level_dose_constancy",
        "never_treated_zero_dose",
        "dose_support",
        "time_support",
    }

    phase3_route = manifest["phase3_route"]
    assert "Phase 3 simulate-data parity and invariant tests" in phase3_route["consumer"]
    assert "before estimator code lands" in phase3_route["next_step"]
    assert "helper asset only" in phase3_route["blocking_note"]
