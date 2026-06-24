from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "contdid-py" / "src" / "contdid" / "contracts" / "phase2.py"
TEMPLATE_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase2"
    / "phase2_contract_template.json"
)


def load_phase2_module():
    assert MODULE_PATH.exists(), f"missing contract module at {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("contdid.contracts.phase2", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_phase2_template_contract_bundle() -> None:
    module = load_phase2_module()

    template_bundle = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    validated = module.validate_phase2_contract_bundle(template_bundle)

    assert validated["phase"] == 2
    assert validated["schema_version"] == "0.2"
    assert {record["id"] for record in validated["estimands"]} >= {
        "LATT(d|d)",
        "ATT(d)",
        "ACRT(d)",
        "LATT^loc",
        "ATT^glob",
        "ACRT^glob",
    }
    assert {record["id"] for record in validated["assumptions"]} >= {
        "Random Sampling",
        "Treatment Support",
        "No Anticipation",
        "PT",
        "Treatment Type/Support",
        "SPT",
    }
    assert [record["step"] for record in validated["algorithm_steps"]] == [1, 2, 3, 4, 5, 6]
    assert {record["paper_symbol"] for record in validated["symbol_map"]} >= {
        "D",
        "\\Delta Y",
        "LATT(d|d)",
        "ATT(d)",
        "ACRT(d)",
        "\\psi^{\\widehat{K}}(d)",
    }
    assert validated["numerical_truth"]["seed_candidates"] == [1234, 20260407, 20261234]
    assert validated["numerical_truth"]["dgp_ids"] == [
        "SIM-001-null-dose",
        "SIM-002-linear-dose",
        "SIM-003-quadratic-dose",
        "SIM-004-staggered-eventstudy-null",
        "SIM-005-cck-two-period",
    ]
    assert validated["numerical_truth"]["seed_registry"] == [
        {"dgp_id": "SIM-001-null-dose", "default_seed": 1234},
        {"dgp_id": "SIM-002-linear-dose", "default_seed": 20260407},
        {"dgp_id": "SIM-003-quadratic-dose", "default_seed": 20261234},
        {"dgp_id": "SIM-004-staggered-eventstudy-null", "default_seed": 20260407},
        {"dgp_id": "SIM-005-cck-two-period", "default_seed": 20261234},
    ]
    tolerance_families = validated["numerical_truth"]["tolerance_families"]
    assert {record["comparison_target"] for record in tolerance_families} == {
        "point estimate",
        "standard error",
        "band / confidence interval",
        "summary aggregation",
        "sign / monotonic segment / support",
        "event-time index semantics",
    }
    assert {
        record["comparison_target"]
        for record in tolerance_families
        if record["tolerance_mode"] == "exact_rule"
    } == {
        "sign / monotonic segment / support",
        "event-time index semantics",
    }
    assert {
        (record["comparison_target"], record["required_from_phase"])
        for record in tolerance_families
    } >= {
        ("point estimate", 4),
        ("summary aggregation", 4),
        ("event-time index semantics", 5),
        ("standard error", 6),
        ("band / confidence interval", 6),
    }


def test_validate_phase2_contract_bundle_rejects_missing_sections() -> None:
    module = load_phase2_module()

    incomplete_bundle = {
        "schema_version": "0.1",
        "phase": 2,
        "estimands": [],
    }

    with pytest.raises(module.Phase2ContractError, match="missing required sections"):
        module.validate_phase2_contract_bundle(incomplete_bundle)


def test_validate_phase2_contract_bundle_rejects_malformed_algorithm_steps() -> None:
    module = load_phase2_module()

    malformed_bundle = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [{"id": "ATT(d)", "summary": "paper-derived summary", "paper_ref": "main.tex:517-533"}],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": "first", "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "python_name": "dose",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose", "default_seed": 1234}],
            "comparison_targets": ["point estimate"],
            "tolerance_families": [
                {
                    "comparison_target": "point estimate",
                    "tolerance_mode": "fixture_numeric",
                    "required_from_phase": 4,
                    "calibration_rule": "Set fixture-level atol/rtol before parity assertions.",
                }
            ],
        },
    }

    with pytest.raises(module.Phase2ContractError, match="algorithm_steps\\[0\\]\\.step"):
        module.validate_phase2_contract_bundle(malformed_bundle)


def test_validate_phase2_contract_bundle_rejects_symbol_map_entries_without_python_name() -> None:
    module = load_phase2_module()

    malformed_bundle = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [{"id": "ATT(d)", "summary": "paper-derived summary", "paper_ref": "main.tex:517-533"}],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": 1, "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose", "default_seed": 1234}],
            "comparison_targets": ["point estimate"],
            "tolerance_families": [
                {
                    "comparison_target": "point estimate",
                    "tolerance_mode": "fixture_numeric",
                    "required_from_phase": 4,
                    "calibration_rule": "Set fixture-level atol/rtol before parity assertions.",
                }
            ],
        },
    }

    with pytest.raises(module.Phase2ContractError, match="symbol_map\\[0\\]\\.python_name"):
        module.validate_phase2_contract_bundle(malformed_bundle)


def test_load_phase2_contract_bundle_from_disk(tmp_path: Path) -> None:
    module = load_phase2_module()
    payload = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [{"id": "ATT(d)", "summary": "paper-derived summary", "paper_ref": "main.tex:517-533"}],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": 1, "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "python_name": "dose",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose", "default_seed": 1234}],
            "comparison_targets": ["point estimate"],
            "tolerance_families": [
                {
                    "comparison_target": "point estimate",
                    "tolerance_mode": "fixture_numeric",
                    "required_from_phase": 4,
                    "calibration_rule": "Set fixture-level atol/rtol before parity assertions.",
                }
            ],
        },
    }
    bundle_path = tmp_path / "phase2.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = module.load_phase2_contract_bundle(bundle_path)

    assert loaded == payload


def test_load_phase2_contract_bundle_defaults_to_checked_package_asset() -> None:
    module = load_phase2_module()

    loaded = module.load_phase2_contract_bundle()

    assert loaded["phase"] == 2
    assert loaded["schema_version"] == "0.2"
    assert {record["id"] for record in loaded["estimands"]} >= {
        "ATT(d)",
        "ACRT(d)",
        "ATT^glob",
        "ACRT^glob",
    }
    assert loaded["numerical_truth"]["seed_registry"] == [
        {"dgp_id": "SIM-001-null-dose", "default_seed": 1234},
        {"dgp_id": "SIM-002-linear-dose", "default_seed": 20260407},
        {"dgp_id": "SIM-003-quadratic-dose", "default_seed": 20261234},
        {"dgp_id": "SIM-004-staggered-eventstudy-null", "default_seed": 20260407},
        {"dgp_id": "SIM-005-cck-two-period", "default_seed": 20261234},
    ]


def test_validate_phase2_contract_bundle_rejects_placeholder_like_content() -> None:
    module = load_phase2_module()

    malformed_bundle = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [
            {
                "id": "ATT(d)",
                "summary": "Placeholder estimand entry for later extraction.",
                "paper_ref": "pending",
            }
        ],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": 1, "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "python_name": "dose",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose", "default_seed": 1234}],
            "comparison_targets": ["point estimate"],
            "tolerance_families": [
                {
                    "comparison_target": "point estimate",
                    "tolerance_mode": "fixture_numeric",
                    "required_from_phase": 4,
                    "calibration_rule": "Set fixture-level atol/rtol before parity assertions.",
                }
            ],
        },
    }

    with pytest.raises(module.Phase2ContractError, match="placeholder-like content"):
        module.validate_phase2_contract_bundle(malformed_bundle)


def test_validate_phase2_contract_bundle_rejects_seed_registry_entries_without_seed() -> None:
    module = load_phase2_module()

    malformed_bundle = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [{"id": "ATT(d)", "summary": "paper-derived summary", "paper_ref": "main.tex:517-533"}],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": 1, "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "python_name": "dose",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose"}],
            "comparison_targets": ["point estimate"],
            "tolerance_families": [
                {
                    "comparison_target": "point estimate",
                    "tolerance_mode": "fixture_numeric",
                    "required_from_phase": 4,
                    "calibration_rule": "Set fixture-level atol/rtol before parity assertions.",
                }
            ],
        },
    }

    with pytest.raises(module.Phase2ContractError, match="seed_registry\\[0\\]\\.default_seed"):
        module.validate_phase2_contract_bundle(malformed_bundle)


def test_validate_phase2_contract_bundle_rejects_missing_tolerance_family_registry() -> None:
    module = load_phase2_module()

    malformed_bundle = {
        "schema_version": "0.2",
        "phase": 2,
        "estimands": [{"id": "ATT(d)", "summary": "paper-derived summary", "paper_ref": "main.tex:517-533"}],
        "assumptions": [{"id": "PT", "summary": "parallel trends", "paper_ref": "main.tex:385-389"}],
        "algorithm_steps": [{"step": 1, "summary": "paper-derived summary", "paper_ref": "main.tex:683-688"}],
        "symbol_map": [
            {
                "paper_symbol": "D",
                "python_name": "dose",
                "kind": "scalar",
                "description": "dose amount",
                "paper_ref": "main.tex:366-369",
            }
        ],
        "numerical_truth": {
            "dgp_ids": ["SIM-001-null-dose"],
            "seed_candidates": [1234],
            "seed_registry": [{"dgp_id": "SIM-001-null-dose", "default_seed": 1234}],
            "comparison_targets": ["point estimate"],
        },
    }

    with pytest.raises(module.Phase2ContractError, match="tolerance_families"):
        module.validate_phase2_contract_bundle(malformed_bundle)
