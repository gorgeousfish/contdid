from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD_CONTRACT_PATH = (
    REPO_ROOT / "reproduction" / "medicare_pps" / "scaffold_contract.json"
)
SCAFFOLD_MODULE_PATH = REPO_ROOT / "reproduction" / "medicare_pps" / "scaffold.py"


def _make_medicare_panel() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    yearly_values = {
        101: {
            "medicare_share_1983": 0.30,
            "depreciation_share": {
                1980: 4.00,
                1981: 4.10,
                1982: 4.20,
                1983: 4.30,
                1984: 4.80,
                1985: 5.00,
                1986: 5.20,
            },
        },
        202: {
            "medicare_share_1983": 0.00,
            "depreciation_share": {
                1980: 3.20,
                1981: 3.10,
                1982: 3.00,
                1983: 2.90,
                1984: 2.80,
                1985: 2.70,
                1986: 2.60,
            },
        },
    }

    for hospital_id, payload in yearly_values.items():
        for year, outcome in payload["depreciation_share"].items():
            rows.append(
                {
                    "hospital_id": hospital_id,
                    "year": year,
                    "depreciation_share": outcome,
                    "medicare_share_1983": payload["medicare_share_1983"],
                }
            )
    return pd.DataFrame(rows)


def _load_scaffold_module():
    spec = importlib.util.spec_from_file_location(
        "medicare_pps_scaffold", SCAFFOLD_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_medicare_pps_example_panel_feeds_public_scaffold() -> None:
    from contdid import load_medicare_pps_example_panel, prepare_medicare_pps_panel

    annual = load_medicare_pps_example_panel()
    assert annual.shape == (35, 4)
    assert annual["hospital_id"].nunique() == 5
    assert sorted(annual["year"].unique()) == [
        1980,
        1981,
        1982,
        1983,
        1984,
        1985,
        1986,
    ]
    assert annual["medicare_share_1983"].between(0, 1).all()

    prepared = prepare_medicare_pps_panel(
        annual,
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="cms_hcris_hospital_cost_reports",
    )
    assert prepared.panel.frame.shape == (10, 5)
    assert prepared.metadata["allowed_use"] == "descriptive-or-scaffold-only"


def test_prepare_medicare_pps_panel_builds_package_ready_two_period_panel() -> None:
    from contdid import (
        load_medicare_pps_manifest,
        prepare_medicare_pps_panel,
        validate_panel_data,
    )

    result = prepare_medicare_pps_panel(
        _make_medicare_panel(),
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="aha_direct_license",
    )

    validated_panel = validate_panel_data(result.panel)
    frame = validated_panel.frame.sort_values(["id", "time_period"]).reset_index(
        drop=True
    )

    records = frame.to_dict("records")
    assert [
        {key: value for key, value in record.items() if key != "Y"}
        for record in records
    ] == [
        {"id": 101, "time_period": 1, "G": 2, "D": 0.30},
        {"id": 101, "time_period": 2, "G": 2, "D": 0.30},
        {"id": 202, "time_period": 1, "G": 0, "D": 0.00},
        {"id": 202, "time_period": 2, "G": 0, "D": 0.00},
    ]
    assert [record["Y"] for record in records] == pytest.approx(
        [4.15, 5.00, 3.05, 2.70]
    )
    assert result.metadata["application_id"] == "medicare-pps-hospitals"
    assert result.metadata["analysis_mode"] == "paper-source-aligned"
    assert result.metadata["baseline_year"] == 1983
    manifest = load_medicare_pps_manifest()
    assert manifest["published_targets_scope"]["selected_sieve_dimension"] == (
        "paper reported CCK target; not produced by the current "
        "cck_polynomial_backend runtime"
    )
    assert result.metadata["required_years"] == [
        1980,
        1981,
        1982,
        1983,
        1984,
        1985,
        1986,
    ]


def test_resolve_medicare_pps_source_flags_public_substitutes_as_scaffold_only() -> (
    None
):
    from contdid import resolve_medicare_pps_source

    source = resolve_medicare_pps_source("cms_hcris_hospital_cost_reports")

    assert source["provider"] == "Centers for Medicare & Medicaid Services"
    assert source["allowed_use"] == "descriptive-or-scaffold-only"
    assert source["parity_viability"] == "insufficient-for-1980-1986-paper-parity"
    assert source["parity_claim_allowed"] is False


def test_prepare_medicare_pps_panel_exposes_public_substitute_policy_metadata() -> (
    None
):
    from contdid import prepare_medicare_pps_panel

    result = prepare_medicare_pps_panel(
        _make_medicare_panel(),
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="cms_hcris_hospital_cost_reports",
    )

    assert result.metadata["analysis_mode"] == "descriptive-or-scaffold-only"
    assert result.metadata["allowed_use"] == "descriptive-or-scaffold-only"
    assert result.metadata["parity_viability"] == (
        "insufficient-for-1980-1986-paper-parity"
    )
    assert result.metadata["parity_claim_allowed"] is False


def test_medicare_pps_manifest_loader_rejects_published_target_scope_drift() -> None:
    from contdid import ContDIDValidationError, load_medicare_pps_manifest
    from contdid.empirical import _validate_medicare_pps_manifest

    payload = copy.deepcopy(load_medicare_pps_manifest())
    payload["published_targets_scope"]["selected_sieve_dimension"] = (
        "runtime selected K"
    )

    with pytest.raises(
        ContDIDValidationError,
        match="published_targets_scope.*paper-reported CCK targets",
    ):
        _validate_medicare_pps_manifest(payload)


def test_medicare_pps_manifest_loader_returns_mutation_isolated_payload() -> None:
    from contdid import load_medicare_pps_manifest

    payload = load_medicare_pps_manifest()
    payload["application_id"] = "mutated-by-caller"
    payload["published_targets_scope"]["selected_sieve_dimension"] = (
        "mutated-by-caller"
    )

    fresh_payload = load_medicare_pps_manifest()
    assert fresh_payload["application_id"] == "medicare-pps-hospitals"
    assert fresh_payload["published_targets_scope"]["selected_sieve_dimension"] == (
        "paper reported CCK target; not produced by the current "
        "cck_polynomial_backend runtime"
    )


def test_medicare_pps_source_options_loader_rejects_public_substitute_policy_drift() -> (
    None
):
    from contdid import ContDIDValidationError, load_medicare_pps_source_options
    from contdid.empirical import _validate_medicare_pps_source_options

    payload = copy.deepcopy(load_medicare_pps_source_options())
    payload["public_substitutes"][0]["parity_claim_allowed"] = True

    with pytest.raises(
        ContDIDValidationError,
        match="public substitute cms_hcris_hospital_cost_reports must freeze parity_claim_allowed=False",
    ):
        _validate_medicare_pps_source_options(payload)


def test_medicare_pps_source_options_loader_returns_mutation_isolated_payload() -> (
    None
):
    from contdid import load_medicare_pps_source_options

    payload = load_medicare_pps_source_options()
    payload["public_substitutes"][0]["parity_claim_allowed"] = True
    payload["public_substitutes"][0]["allowed_use"] = "mutated-by-caller"

    fresh_payload = load_medicare_pps_source_options()
    public_substitute = fresh_payload["public_substitutes"][0]
    assert public_substitute["parity_claim_allowed"] is False
    assert public_substitute["allowed_use"] == "descriptive-or-scaffold-only"


def test_medicare_pps_source_options_loader_rejects_missing_source_sections() -> None:
    from contdid import ContDIDValidationError, load_medicare_pps_source_options
    from contdid.empirical import _validate_medicare_pps_source_options

    payload = copy.deepcopy(load_medicare_pps_source_options())
    payload.pop("primary_sources")

    with pytest.raises(
        ContDIDValidationError,
        match="primary_sources must be a non-empty list",
    ):
        _validate_medicare_pps_source_options(payload)


def test_medicare_pps_source_options_loader_rejects_missing_source_fields() -> None:
    from contdid import ContDIDValidationError, load_medicare_pps_source_options
    from contdid.empirical import _validate_medicare_pps_source_options

    payload = copy.deepcopy(load_medicare_pps_source_options())
    payload["public_substitutes"][0].pop("provider")

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "public_substitutes entry cms_hcris_hospital_cost_reports "
            "is missing provider"
        ),
    ):
        _validate_medicare_pps_source_options(payload)


def test_prepare_medicare_pps_panel_requires_full_1980_1986_support() -> None:
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    incomplete = _make_medicare_panel().query("year != 1986")

    with pytest.raises(ContDIDValidationError, match="1980-1986"):
        prepare_medicare_pps_panel(
            incomplete,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_reports_incomplete_string_unit_ids_cleanly() -> (
    None
):
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    incomplete = _make_medicare_panel().replace(
        {"hospital_id": {101: "A101", 202: "B202"}}
    )
    incomplete = incomplete.query("not (hospital_id == 'A101' and year == 1986)")

    with pytest.raises(
        ContDIDValidationError,
        match="requires every hospital to cover 1980-1986 exactly once",
    ):
        prepare_medicare_pps_panel(
            incomplete,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_rejects_missing_staged_columns_before_pandas_keyerror() -> (
    None
):
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    missing_dose = _make_medicare_panel().drop(columns=["medicare_share_1983"])

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "medicare PPS scaffold input is missing required columns: "
            "medicare_share_1983"
        ),
    ):
        prepare_medicare_pps_panel(
            missing_dose,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_rejects_fractional_years_before_coercion() -> (
    None
):
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    fractional_year = _make_medicare_panel()
    fractional_year["year"] = fractional_year["year"].astype(object)
    fractional_year.loc[0, "year"] = 1980.5

    with pytest.raises(
        ContDIDValidationError,
        match="year values must be finite integer calendar years",
    ):
        prepare_medicare_pps_panel(
            fractional_year,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_rejects_non_numeric_outcomes_cleanly() -> None:
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    bad_outcome = _make_medicare_panel()
    bad_outcome["depreciation_share"] = bad_outcome["depreciation_share"].astype(
        object
    )
    bad_outcome.loc[0, "depreciation_share"] = "bad"

    with pytest.raises(
        ContDIDValidationError,
        match="outcome values must be finite numeric depreciation-share values",
    ):
        prepare_medicare_pps_panel(
            bad_outcome,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_rejects_non_numeric_or_out_of_scale_dose() -> (
    None
):
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    string_dose = _make_medicare_panel()
    string_dose["medicare_share_1983"] = string_dose[
        "medicare_share_1983"
    ].astype(object)
    string_dose.loc[0, "medicare_share_1983"] = "0.30"

    with pytest.raises(
        ContDIDValidationError,
        match="dose values must be finite numeric 1983 Medicare inpatient shares",
    ):
        prepare_medicare_pps_panel(
            string_dose,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )

    out_of_scale_dose = _make_medicare_panel()
    out_of_scale_dose.loc[0, "medicare_share_1983"] = 1.10

    with pytest.raises(
        ContDIDValidationError,
        match=r"dose values must be finite numeric 1983 Medicare inpatient shares on the \[0, 1\] scale",
    ):
        prepare_medicare_pps_panel(
            out_of_scale_dose,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_prepare_medicare_pps_panel_rejects_within_unit_dose_drift() -> None:
    from contdid import ContDIDValidationError, prepare_medicare_pps_panel

    drifting_dose = _make_medicare_panel()
    drifting_dose.loc[
        (drifting_dose["hospital_id"] == 101) & (drifting_dose["year"] == 1984),
        "medicare_share_1983",
    ] = 0.80

    with pytest.raises(
        ContDIDValidationError,
        match="dose column must be the unit-constant 1983 Medicare inpatient share",
    ):
        prepare_medicare_pps_panel(
            drifting_dose,
            unit_column="hospital_id",
            year_column="year",
            outcome_column="depreciation_share",
            dose_column="medicare_share_1983",
            source_id="aha_direct_license",
        )


def test_phase7_scaffold_contract_preserves_published_targets_and_phase6_oracle_links() -> (
    None
):
    assert SCAFFOLD_CONTRACT_PATH.exists(), (
        f"missing scaffold contract: {SCAFFOLD_CONTRACT_PATH}"
    )

    contract = json.loads(SCAFFOLD_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "0.1"
    assert contract["application_id"] == "medicare-pps-hospitals"
    assert contract["baseline_year"] == 1983
    assert contract["source_labels"] == [
        "licensed",
        "restricted",
        "descriptive-or-scaffold-only",
    ]
    assert contract["published_targets"]["twfe_beta"] == 1.14
    assert contract["published_targets"]["latt_loc"] == 0.80
    assert contract["published_targets"]["acrt_glob"] == -0.08
    assert contract["published_targets_scope"] == {
        "source": (
            "paper reported targets from "
            "arXiv-2107.02637v7/main.tex:864-880,899-899,934-934"
        ),
        "status": (
            "unmet parity targets until licensed AHA inputs and "
            "paper-supported adaptive CCK estimation are available"
        ),
        "selected_sieve_dimension": (
            "paper reported CCK target; not produced by the current "
            "cck_polynomial_backend runtime"
        ),
    }
    assert (
        contract["phase6_inference_reference"]
        == "reproduction/phase6_inference/phase6_inference_reference.json"
    )


def test_build_medicare_pps_scaffold_preserves_source_labels_and_phase6_metadata() -> (
    None
):
    scaffold = _load_scaffold_module()

    result = scaffold.build_medicare_pps_scaffold(
        _make_medicare_panel(),
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="cms_hcris_hospital_cost_reports",
    )

    two_period = result.two_period_panel.frame.sort_values(["id", "time_period"]).reset_index(
        drop=True
    )
    yearly = result.eventstudy_panel.frame.sort_values(["id", "time_period"]).reset_index(
        drop=True
    )

    assert two_period["time_period"].tolist() == [1, 2, 1, 2]
    assert sorted(yearly["time_period"].unique().tolist()) == [
        1980,
        1981,
        1982,
        1983,
        1984,
        1985,
        1986,
    ]
    assert result.metadata["analysis_mode"] == "descriptive-or-scaffold-only"
    assert result.metadata["source_label"] == "descriptive-or-scaffold-only"
    assert result.metadata["baseline_year"] == 1983
    assert result.metadata["allowed_use"] == "descriptive-or-scaffold-only"
    assert result.metadata["parity_viability"] == (
        "insufficient-for-1980-1986-paper-parity"
    )
    assert result.metadata["parity_claim_allowed"] is False
    assert result.metadata["two_period_metadata"]["parity_claim_allowed"] is False
    assert (
        result.metadata["two_period_metadata"]["parity_viability"]
        == result.metadata["parity_viability"]
    )
    assert result.metadata["published_targets"]["selected_sieve_dimension"] == 4
    assert result.metadata["published_targets_scope"]["selected_sieve_dimension"] == (
        "paper reported CCK target; not produced by the current "
        "cck_polynomial_backend runtime"
    )
    assert result.metadata["phase6_inference_reference"]["application_id"] == (
        "phase6-inference-reference-fixture"
    )


def test_root_medicare_pps_scaffold_loaders_return_mutation_isolated_payloads() -> (
    None
):
    scaffold = _load_scaffold_module()

    manifest = scaffold.load_manifest()
    manifest["application_id"] = "mutated-by-caller"
    registry = scaffold.load_source_registry()
    registry["routes"][0]["access_route"] = "mutated-by-caller"
    contract = scaffold.load_scaffold_contract()
    contract["baseline_year"] = 9999
    phase6_reference = scaffold.load_phase6_reference()
    phase6_reference["application_id"] = "mutated-by-caller"

    assert scaffold.load_manifest()["application_id"] == "medicare-pps-hospitals"
    assert scaffold.load_source_registry()["routes"][0]["access_route"] == "licensed"
    assert scaffold.load_scaffold_contract()["baseline_year"] == 1983
    assert scaffold.load_phase6_reference()["application_id"] == (
        "phase6-inference-reference-fixture"
    )


def test_root_medicare_pps_scaffold_loaders_reject_malformed_payloads(
    tmp_path: Path,
) -> None:
    from contdid import ContDIDValidationError

    scaffold = _load_scaffold_module()

    original_contract_path = scaffold._SCAFFOLD_CONTRACT_PATH
    bad_contract_path = tmp_path / "scaffold_contract.json"
    bad_contract_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "application_id": "medicare-pps-hospitals",
            }
        ),
        encoding="utf-8",
    )
    try:
        scaffold._SCAFFOLD_CONTRACT_PATH = bad_contract_path
        scaffold._load_scaffold_contract_cached.cache_clear()
        with pytest.raises(
            ContDIDValidationError,
            match="scaffold contract missing baseline_year",
        ):
            scaffold.load_scaffold_contract()
    finally:
        scaffold._SCAFFOLD_CONTRACT_PATH = original_contract_path
        scaffold._load_scaffold_contract_cached.cache_clear()

    original_registry_path = scaffold._SOURCE_REGISTRY_PATH
    bad_registry_path = tmp_path / "source_registry.json"
    bad_registry_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "application_id": "medicare-pps-hospitals",
            }
        ),
        encoding="utf-8",
    )
    try:
        scaffold._SOURCE_REGISTRY_PATH = bad_registry_path
        scaffold._load_source_registry_cached.cache_clear()
        with pytest.raises(
            ContDIDValidationError,
            match="source registry routes must be a non-empty list",
        ):
            scaffold.load_source_registry()
    finally:
        scaffold._SOURCE_REGISTRY_PATH = original_registry_path
        scaffold._load_source_registry_cached.cache_clear()

    original_phase6_path = scaffold._PHASE6_REFERENCE_PATH
    bad_phase6_path = tmp_path / "phase6_inference_reference.json"
    bad_phase6_path.write_text(
        json.dumps({"application_id": "phase6-inference-reference-fixture"}),
        encoding="utf-8",
    )
    try:
        scaffold._PHASE6_REFERENCE_PATH = bad_phase6_path
        scaffold._load_phase6_reference_cached.cache_clear()
        with pytest.raises(
            ContDIDValidationError,
            match="phase6 reference oracle_targets must be a non-empty list",
        ):
            scaffold.load_phase6_reference()
    finally:
        scaffold._PHASE6_REFERENCE_PATH = original_phase6_path
        scaffold._load_phase6_reference_cached.cache_clear()


def test_medicare_pps_e2e_smoke_exposes_public_substitute_policy() -> None:
    spec = importlib.util.spec_from_file_location(
        "medicare_pps_e2e_smoke",
        REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    payload = smoke.run_medicare_pps_e2e_smoke(
        smoke._build_default_smoke_panel(),
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="cms_hcris_hospital_cost_reports",
    )

    assert payload["run_mode"] == "descriptive-or-scaffold-only"
    assert payload["allowed_use"] == "descriptive-or-scaffold-only"
    assert payload["parity_viability"] == "insufficient-for-1980-1986-paper-parity"
    assert payload["parity_claim_allowed"] is False
    assert payload["outputs"]["att"]["required_metadata"] == {
        "confidence_interval": payload["outputs"]["att"]["confidence_interval"],
        "confidence_band": payload["outputs"]["att"]["confidence_band"],
    }
    assert payload["outputs"]["acrt"]["required_metadata"] == {
        "confidence_interval": payload["outputs"]["acrt"]["confidence_interval"],
        "confidence_band": payload["outputs"]["acrt"]["confidence_band"],
    }
    assert payload["outputs"]["eventstudy"]["event_time_grid"] == payload["outputs"][
        "eventstudy"
    ]["grid"]
    assert set(payload["outputs"]["eventstudy"]["required_metadata"]) == {
        "event_time_grid",
        "confidence_interval",
        "confidence_band",
    }


def test_medicare_pps_e2e_smoke_manifest_loader_is_mutation_isolated() -> None:
    spec = importlib.util.spec_from_file_location(
        "medicare_pps_e2e_smoke",
        REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    manifest = smoke.load_e2e_smoke_manifest()
    manifest["expected_outputs"]["att"]["package_surface"] = "mutated-by-caller"

    assert smoke.load_e2e_smoke_manifest()["expected_outputs"]["att"][
        "package_surface"
    ] == "estimate_dose_effects"


def test_medicare_pps_e2e_smoke_manifest_loader_rejects_malformed_payload(
    tmp_path: Path,
) -> None:
    from contdid import ContDIDValidationError

    spec = importlib.util.spec_from_file_location(
        "medicare_pps_e2e_smoke",
        REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    original_path = smoke._SMOKE_MANIFEST_PATH
    bad_path = tmp_path / "e2e_smoke_manifest.json"
    bad_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "application_id": "medicare-pps-hospitals",
            }
        ),
        encoding="utf-8",
    )
    try:
        smoke._SMOKE_MANIFEST_PATH = bad_path
        smoke._load_e2e_smoke_manifest_cached.cache_clear()
        with pytest.raises(
            ContDIDValidationError,
            match="e2e smoke manifest expected_outputs must be a JSON object",
        ):
            smoke.load_e2e_smoke_manifest()
    finally:
        smoke._SMOKE_MANIFEST_PATH = original_path
        smoke._load_e2e_smoke_manifest_cached.cache_clear()


def test_medicare_pps_e2e_smoke_manifest_requires_metadata_contract(
    tmp_path: Path,
) -> None:
    from contdid import ContDIDValidationError

    spec = importlib.util.spec_from_file_location(
        "medicare_pps_e2e_smoke",
        REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke.py",
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    original_path = smoke._SMOKE_MANIFEST_PATH
    bad_path = tmp_path / "e2e_smoke_manifest.json"
    payload = smoke.load_e2e_smoke_manifest()
    payload["expected_outputs"]["eventstudy"].pop("required_metadata")
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        smoke._SMOKE_MANIFEST_PATH = bad_path
        smoke._load_e2e_smoke_manifest_cached.cache_clear()
        with pytest.raises(
            ContDIDValidationError,
            match="required_metadata for eventstudy must be a list of strings",
        ):
            smoke.load_e2e_smoke_manifest()
    finally:
        smoke._SMOKE_MANIFEST_PATH = original_path
        smoke._load_e2e_smoke_manifest_cached.cache_clear()
