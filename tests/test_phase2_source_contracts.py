from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_CONTRACTS_DIR = REPO_ROOT / "contdid-py" / "contracts" / "phase2"
PAPER_TRUTH_PATH = PHASE2_CONTRACTS_DIR / "paper_truth_contract.json"
SYMBOL_MAP_PATH = PHASE2_CONTRACTS_DIR / "symbol_map_contract.json"
NUMERICAL_TRUTH_PATH = PHASE2_CONTRACTS_DIR / "numerical_truth_contract_v1.json"
BUNDLE_PATH = PHASE2_CONTRACTS_DIR / "phase2_contract_template.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase2_paper_truth_source_contract_exists_and_matches_bundle() -> None:
    assert PAPER_TRUTH_PATH.exists(), f"missing paper truth contract: {PAPER_TRUTH_PATH}"

    paper_truth = _load_json(PAPER_TRUTH_PATH)
    bundle = _load_json(BUNDLE_PATH)

    assert paper_truth["phase"] == 2
    assert paper_truth["estimands"] == bundle["estimands"]
    assert paper_truth["assumptions"] == bundle["assumptions"]
    assert paper_truth["algorithm_steps"] == bundle["algorithm_steps"]


def test_phase2_symbol_map_source_contract_exists_and_carries_fidelity_rules() -> None:
    assert SYMBOL_MAP_PATH.exists(), f"missing symbol map contract: {SYMBOL_MAP_PATH}"

    symbol_map_contract = _load_json(SYMBOL_MAP_PATH)
    bundle = _load_json(BUNDLE_PATH)

    assert symbol_map_contract["phase"] == 2
    assert symbol_map_contract["symbol_map"] == bundle["symbol_map"]
    assert {rule["name"] for rule in symbol_map_contract["fidelity_rules"]} >= {
        "target_parameter",
        "aggregation",
        "dose_est_method",
        "control_group",
    }
    assert set(symbol_map_contract["public_api_boundary"]["public_terms"]) >= {
        "dose",
        "event_time",
        "att",
        "acrt",
    }


def test_phase2_numerical_truth_source_contract_exists_and_matches_bundle() -> None:
    assert NUMERICAL_TRUTH_PATH.exists(), (
        f"missing numerical truth contract: {NUMERICAL_TRUTH_PATH}"
    )

    numerical_truth_contract = _load_json(NUMERICAL_TRUTH_PATH)
    bundle = _load_json(BUNDLE_PATH)

    assert numerical_truth_contract["phase"] == 2
    assert numerical_truth_contract["numerical_truth"] == bundle["numerical_truth"]
    tolerance_families = numerical_truth_contract["numerical_truth"]["tolerance_families"]
    assert {
        record["comparison_target"] for record in tolerance_families
    } == set(bundle["numerical_truth"]["comparison_targets"])
    assert {
        record["comparison_target"]
        for record in tolerance_families
        if record["tolerance_mode"] == "exact_rule"
    } == {
        "sign / monotonic segment / support",
        "event-time index semantics",
    }
