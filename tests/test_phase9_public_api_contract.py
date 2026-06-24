from __future__ import annotations

import copy
import inspect
import json
import tomllib
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT / "contdid-py" / "contracts" / "phase9" / "public_api_contract_v1.json"
)
PHASE2_SYMBOL_MAP_PATH = (
    REPO_ROOT / "contdid-py" / "contracts" / "phase2" / "symbol_map_contract.json"
)
PHASE8_BASELINE_PATH = (
    REPO_ROOT / "reproduction" / "phase8_monte_carlo" / "baseline_registry.json"
)
PHASE9_RELEASE_EXAMPLES_PATH = (
    REPO_ROOT / "reproduction" / "phase9_release_examples" / "manifest.json"
)
MEDICARE_RELEASE_PACKET_PATH = (
    REPO_ROOT / "reproduction" / "medicare_pps" / "release_example_manifest.json"
)
PYPROJECT_PATH = REPO_ROOT / "contdid-py" / "pyproject.toml"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _package_version() -> str:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]["version"]


def test_phase9_public_api_contract_exists_and_points_to_release_truth_inputs() -> None:
    assert CONTRACT_PATH.exists(), (
        f"missing Phase 9 public API contract: {CONTRACT_PATH}"
    )
    assert PHASE2_SYMBOL_MAP_PATH.exists(), (
        f"missing symbol map contract: {PHASE2_SYMBOL_MAP_PATH}"
    )
    assert PHASE8_BASELINE_PATH.exists(), (
        f"missing Phase 8 baseline registry: {PHASE8_BASELINE_PATH}"
    )
    assert PHASE9_RELEASE_EXAMPLES_PATH.exists(), (
        f"missing Phase 9 release example manifest: {PHASE9_RELEASE_EXAMPLES_PATH}"
    )
    assert MEDICARE_RELEASE_PACKET_PATH.exists(), (
        f"missing Medicare release packet manifest: {MEDICARE_RELEASE_PACKET_PATH}"
    )

    contract = _load_json(CONTRACT_PATH)

    assert contract["schema_version"] == "0.1"
    assert contract["phase"] == 9
    assert contract["source_contracts"] == {
        "phase2_symbol_map": "contdid-py/contracts/phase2/symbol_map_contract.json",
        "phase2_contract_template": "contdid-py/contracts/phase2/phase2_contract_template.json",
        "phase11_cck_boundary_contract": (
            "contdid-py/contracts/phase11/cck_unsupported_boundary_contract.json"
        ),
        "phase12_data_shape_boundary_contract": (
            "contdid-py/contracts/phase12/data_shape_boundary_contract.json"
        ),
        "phase8_baseline_registry": "reproduction/phase8_monte_carlo/baseline_registry.json",
        "phase9_release_examples": "reproduction/phase9_release_examples/manifest.json",
        "medicare_release_example": "reproduction/medicare_pps/release_example_manifest.json",
        "package_init_surface": "contdid-py/src/contdid/__init__.py",
        "result_container": "contdid-py/src/contdid/results.py",
        "spec_container": "contdid-py/src/contdid/specs.py",
    }
    required_stable_exports = {
        "__version__",
        "ContDIDResult",
        "ContDIDSpec",
        "ContDIDValidationError",
        "EmpiricalScaffoldResult",
        "PanelData",
        "Phase2ContractError",
        "Phase11ContractError",
        "Phase12ContractError",
        "PublicAPIContractError",
        "attach_inference_payload",
        "build_dose_grid",
        "build_confidence_band",
        "compute_multiplier_bootstrap",
        "load_medicare_pps_example_panel",
        "load_medicare_pps_manifest",
        "load_medicare_pps_source_options",
        "prepare_medicare_pps_panel",
        "resolve_medicare_pps_source",
        "estimate_dose_effects",
        "estimate_dose_level_effects",
        "estimate_dose_slope_effects",
        "estimate_eventstudy_effects",
        "estimate_eventstudy_slope_effects",
        "load_phase2_contract_bundle",
        "load_phase11_cck_boundary_contract_bundle",
        "load_phase12_data_shape_contract_bundle",
        "load_public_api_contract_bundle",
        "simulate_contdid_data",
        "validate_panel_data",
        "validate_public_api_contract_bundle",
        "validate_phase2_contract_bundle",
        "validate_phase11_cck_boundary_contract_bundle",
        "validate_phase12_data_shape_contract_bundle",
        "validate_spec",
    }
    assert set(contract["stable_top_level_exports"]) >= required_stable_exports


def test_phase9_runtime_contract_validator_rejects_missing_source_truth_inputs() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["source_contracts"] = {}

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract source_contracts missing required fields: "
            "phase2_symbol_map"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_blank_source_truth_paths() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["source_contracts"]["phase2_symbol_map"] = " "

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract source_contracts.phase2_symbol_map "
            "must be a non-empty string"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value"),
    [
        ("phase2_symbol_map", "contdid-py/contracts/phase2/symbol_map_v2.json"),
        ("phase2_contract_template", "contdid-py/contracts/phase2/template_v2.json"),
        (
            "phase11_cck_boundary_contract",
            "contdid-py/contracts/phase11/cck_boundary_v2.json",
        ),
        (
            "phase12_data_shape_boundary_contract",
            "contdid-py/contracts/phase12/data_shape_boundary_v2.json",
        ),
        ("phase8_baseline_registry", "reproduction/phase8/baseline.json"),
        ("phase9_release_examples", "reproduction/phase9/examples.json"),
        ("medicare_release_example", "reproduction/medicare_pps/manifest.json"),
        ("package_init_surface", "contdid-py/src/contdid/api.py"),
        ("result_container", "contdid-py/src/contdid/result.py"),
        ("spec_container", "contdid-py/src/contdid/spec.py"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_source_contract_path_drift(
    contract_key: str,
    corrupted_value: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["source_contracts"][contract_key] = corrupted_value

    with pytest.raises(
        PublicAPIContractError,
        match="source_contracts must freeze the checked truth-input paths",
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_blank_docs_bundle_paths() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["files"]["readme"] = " "

    with pytest.raises(
        PublicAPIContractError,
        match="public API contract docs_bundle.files.readme must be a non-empty string",
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_scalar_dvals_contract_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["public_route_controls"]["shared_dose_basis_kwargs"]["dvals"][
        "accepted_values"
    ] = [None, "finite non-boolean numeric iterable dose grid"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract shared_dose_basis_kwargs.dvals must declare "
            "None, finite non-boolean numeric scalar, and finite "
            "non-boolean numeric iterable dose-grid inputs"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ].__setitem__("surfaces", ["estimate_dose_effects"]),
            "shared_dose_basis_kwargs.surfaces must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["dvals"].__setitem__("kind", "positional"),
            "shared_dose_basis_kwargs.dvals must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["dvals"].__setitem__("default", []),
            "shared_dose_basis_kwargs.dvals must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["dvals"].__setitem__("role", "grid"),
            "shared_dose_basis_kwargs.dvals.role must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["degree"].__setitem__("default", 2),
            "shared_dose_basis_kwargs.degree must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["degree"].__setitem__("minimum", 0),
            "shared_dose_basis_kwargs.degree must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["degree"].__setitem__("integer", False),
            "shared_dose_basis_kwargs.degree must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["degree"].__setitem__("role", "basis degree"),
            "shared_dose_basis_kwargs.degree must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["num_knots"].__setitem__("default", 1),
            "shared_dose_basis_kwargs.num_knots must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["num_knots"].__setitem__("minimum", -1),
            "shared_dose_basis_kwargs.num_knots must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["num_knots"].__setitem__("integer", False),
            "shared_dose_basis_kwargs.num_knots must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "shared_dose_basis_kwargs"
            ]["num_knots"].__setitem__("role", "knot count"),
            "shared_dose_basis_kwargs.num_knots must freeze",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_shared_basis_signature_drift(
    mutation, match: str
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_base_period_contract_drift() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["public_route_controls"]["eventstudy_base_period_kwargs"]["base_period"][
        "accepted_values"
    ] = [None, "varying", "universal"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "eventstudy_base_period_kwargs.base_period must declare "
            "None, 'varying', 'universal', and observed integer time-period inputs"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["public_route_controls"][
                "eventstudy_base_period_kwargs"
            ].__setitem__("surfaces", ["estimate_eventstudy_effects"]),
            "eventstudy_base_period_kwargs.surfaces must freeze",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "eventstudy_base_period_kwargs"
            ]["base_period"].__setitem__("kind", "positional"),
            "eventstudy_base_period_kwargs.base_period must freeze a keyword-only",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "eventstudy_base_period_kwargs"
            ]["base_period"].__setitem__("default", "varying"),
            "eventstudy_base_period_kwargs.base_period must freeze a keyword-only",
        ),
        (
            lambda contract: contract["public_route_controls"][
                "eventstudy_base_period_kwargs"
            ]["base_period"].__setitem__("role", "baseline control"),
            "eventstudy_base_period_kwargs.base_period.role must freeze",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_base_period_signature_drift(
    mutation, match: str
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_blank_checked_example_asset_paths() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["checked_example_assets"]["synthetic-dose-demo"][
        "panel"
    ] = " "

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.checked_example_assets."
            "synthetic-dose-demo.panel must be a non-empty string"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_checked_example_asset_fields() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    del corrupted["docs_bundle"]["checked_example_assets"]["medicare-scaffold-demo"][
        "eventstudy_panel"
    ]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.checked_example_assets."
            "medicare-scaffold-demo missing required fields: eventstudy_panel"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_empty_docs_bundle_lists() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["readme_required_terms"] = []

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.readme_required_terms "
            "must be a non-empty JSON array of non-empty strings"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_user_guide_required_terms() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    del corrupted["docs_bundle"]["user_guide_required_terms"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.user_guide_required_terms "
            "must be a non-empty JSON array of non-empty strings"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("docs_key", "removed_term"),
    [
        ("readme_required_terms", "PanelData"),
        ("readme_required_terms", "Supported routes and boundaries"),
        ("public_api_help_required_terms", "balanced long panel"),
        ("public_api_help_required_terms", "cohort_summary"),
        ("user_guide_required_terms", "time-varying doses"),
        ("medicare_example_help_required_terms", "licensed AHA 1980-1986 hospital panel"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_missing_reader_doc_terms(
    docs_key: str,
    removed_term: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"][docs_key].remove(removed_term)

    with pytest.raises(
        PublicAPIContractError,
        match=(
            f"public API contract docs_bundle.{docs_key} must cover "
            "reader-facing documentation tasks"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_public_doc_forbidden_terms() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["public_doc_forbidden_terms"].remove("parity_claim_allowed")

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.public_doc_forbidden_terms must "
            "block internal release/control-plane vocabulary"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_render_profile_field_list() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    del corrupted["docs_bundle"]["dose_curve_render_profile_fields"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract docs_bundle.dose_curve_render_profile_fields "
            "must be a non-empty JSON array of non-empty strings"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_empty_export_surface() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["stable_top_level_exports"] = []

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract stable_top_level_exports must be a non-empty "
            "JSON array of non-empty strings"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_requires_phase11_audit_exports() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["stable_top_level_exports"] = [
        export_name
        for export_name in corrupted["stable_top_level_exports"]
        if export_name != "load_phase11_cck_boundary_contract_bundle"
    ]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "stable_top_level_exports missing required Phase 11 CCK boundary exports"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_requires_phase12_data_shape_exports() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["stable_top_level_exports"] = [
        export_name
        for export_name in corrupted["stable_top_level_exports"]
        if export_name != "load_phase12_data_shape_contract_bundle"
    ]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "stable_top_level_exports missing required "
            "Phase 12 data-shape boundary exports"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["stable_top_level_exports"].append(
            "estimate_dose_effects_v2"
        ),
        lambda contract: contract["stable_top_level_exports"].__setitem__(
            0, "__version_v2__"
        ),
        lambda contract: contract["stable_top_level_exports"].reverse(),
    ],
)
def test_phase9_runtime_contract_validator_rejects_export_surface_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="stable_top_level_exports must freeze the checked public export surface",
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_estimand_route_export_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["traceability"]["estimand_routes"]["eventstudy"]["package_surface"] = (
        "estimate_eventstudy_effects_v2"
    )

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract traceability.estimand_routes.eventstudy."
            "package_surface must reference a stable top-level export"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["traceability"].__setitem__(
            "public_terms", ["dose", "event_time", "effect"]
        ),
        lambda contract: contract["traceability"].__setitem__(
            "internal_trace_terms", ["target_parameter", "dose_est_method"]
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_traceability_terms_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="traceability\\.(public_terms|internal_trace_terms).*freeze",
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["traceability"]["estimand_routes"]["att"].__setitem__(
            "result_estimand", "ATT"
        ),
        lambda contract: contract["traceability"]["estimand_routes"][
            "acrt"
        ].__setitem__("paper_symbol", "ACR(d)"),
        lambda contract: contract["traceability"]["estimand_routes"][
            "eventstudy"
        ].__setitem__("result_estimand", "ATT(time)"),
        lambda contract: contract["traceability"]["estimand_routes"][
            "eventstudy_slope"
        ].__setitem__("paper_symbol", "ACRT(t)"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_estimand_route_label_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="estimand_routes\\..*must freeze the checked paper/public estimand mapping",
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_eventstudy_control_group_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["traceability"]["eventstudy_input_contract"][
        "supported_control_groups"
    ] = [
        "notyettreated",
        "eventuallytreated",
    ]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract eventstudy_input_contract must freeze "
            "the supported event-study control groups"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_eventstudy_unsupported_route_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["traceability"]["eventstudy_input_contract"][
        "unsupported_dose_est_method_rule"
    ] = "event study silently falls back to parametric estimation"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract eventstudy_input_contract must freeze "
            "the unsupported CCK event-study rule"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_cck_panel_shape_precedence_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["traceability"]["eventstudy_input_contract"][
        "cck_panel_shape_guard_precedence_rule"
    ] = "event-study CCK errors can run before panel-shape guards"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract eventstudy_input_contract must freeze "
            "the CCK panel-shape guard precedence rule"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("result_payload_key", "corrupted_value"),
    [
        ("required_fields", []),
        ("eventstudy_fields", ["event_time", " "]),
        ("metadata_must_carry", []),
        ("metadata_forbidden_fields", ["boot_type", None]),
        ("inference_labels", []),
        ("bootstrap_type_labels", ["multiplier", ""]),
        ("confidence_band_kind_labels", []),
    ],
)
def test_phase9_runtime_contract_validator_rejects_empty_result_payload_lists(
    result_payload_key: str,
    corrupted_value: list[object],
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"][result_payload_key] = corrupted_value

    with pytest.raises(
        PublicAPIContractError,
        match=(
            f"public API contract result_payload.{result_payload_key} "
            "must be a non-empty JSON array of non-empty strings"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value"),
    [
        ("metadata_must_carry", ["target_parameter"]),
        ("metadata_forbidden_fields", ["backend_name"]),
        ("inference_labels", ["analytic"]),
        ("bootstrap_type_labels", ["analytic"]),
        ("confidence_band_kind_labels", ["pointwise_analytic"]),
    ],
)
def test_phase9_runtime_contract_validator_rejects_inference_metadata_label_drift(
    contract_key: str,
    corrupted_value: list[str],
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"][contract_key] = corrupted_value

    with pytest.raises(
        PublicAPIContractError,
        match=(
            f"public API contract result_payload.{contract_key} must freeze "
            "the checked inference metadata contract"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value"),
    [
        ("alpha_field", "alpha"),
        ("bootstrap_flag_field", "bootstrap"),
        ("bootstrap_iteration_field", "iterations"),
        ("confidence_band_flag_field", "uniform_band"),
        ("bootstrap_disabled_label", "none"),
        ("bootstrap_enabled_label", "multiplier"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_inference_echo_drift(
    contract_key: str,
    corrupted_value: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"]["inference_control_echo_contract"][contract_key] = (
        corrupted_value
    )

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract inference_control_echo_contract must freeze "
            "the checked inference-control echo semantics"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["result_payload"][
                "summary_field_families"
            ].__setitem__("dose_level", ["overall_att"]),
            "summary_field_families must freeze the dose summary aggregate fields",
        ),
        (
            lambda contract: contract["result_payload"][
                "summary_field_families"
            ].__setitem__("dose_slope", ["overall_acrt"]),
            "summary_field_families must freeze the dose summary aggregate fields",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_dose_summary_field_drift(
    mutation,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["result_payload"][
                "dose_route_metadata_contract"
            ].__setitem__("basis_field", "basis_info"),
            "dose_route_metadata_contract must freeze the dose basis metadata",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_metadata_contract"
            ].__setitem__("basis_types", ["global_polynomial"]),
            "dose_route_metadata_contract must freeze the dose basis metadata",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_metadata_contract"
            ].__setitem__("lineage_values", ["phase4_shared_dose_stack"]),
            "dose_route_metadata_contract must freeze the dose basis metadata",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_traceability_contract"
            ].__setitem__("treated_sample_keys", ["treated_count"]),
            "dose_route_traceability_contract must freeze the treated sample",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_traceability_contract"
            ].__setitem__("untreated_benchmark_field", "control_mean"),
            "dose_route_traceability_contract must freeze the treated sample",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_traceability_contract"
            ].__setitem__("delta_outcome_value", "post minus pre"),
            "dose_route_traceability_contract must freeze the treated sample",
        ),
        (
            lambda contract: contract["result_payload"][
                "dose_route_traceability_contract"
            ].__setitem__("derivative_value", "finite difference"),
            "dose_route_traceability_contract must freeze the treated sample",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_dose_traceability_drift(
    mutation,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["result_payload"][
                "eventstudy_summary_aggregates"
            ].__setitem__("level", ["overall_level"]),
            "eventstudy_summary_aggregates must freeze event-study weighting",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_summary_aggregates"
            ].__setitem__("slope", ["overall_slope"]),
            "eventstudy_summary_aggregates must freeze event-study weighting",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_summary_aggregates"
            ].__setitem__("overall_weighting", "simple mean"),
            "eventstudy_summary_aggregates must freeze event-study weighting",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_summary_aggregates"
            ].__setitem__("post_treatment_mean_rule", "all event-time cells"),
            "eventstudy_summary_aggregates must freeze event-study weighting",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_summary_aggregates"
            ].__setitem__("explicit_dvals_role", "support filter"),
            "eventstudy_summary_aggregates must freeze event-study weighting",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_eventstudy_summary_drift(
    mutation,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_eventstudy_inference_covariance_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"]["eventstudy_inference_covariance_rule"] = (
        "event-time cells are independent"
    )

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "eventstudy_inference_covariance_rule must freeze event-study "
            "cross-event covariance semantics"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("timing_group_support_fields", ["timing_groups"]),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("base_period_field", "baseline"),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("basis_origin_rule", "local knots per cohort"),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("lineage_values", {"level": "mean", "slope": "derivative"}),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("shape_constraint_honesty", "flat zero outside support"),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__(
                "result_container_event_time_rule",
                "cohort_summary event_time may be numeric",
            ),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"].__setitem__(
                "eventstudy_fields",
                ["event_time", "event_time_grid", "cohort_summary"],
            ),
            "result_payload.eventstudy_fields must freeze",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("cohort_summary_fields", ["event_time"]),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
        (
            lambda contract: contract["result_payload"][
                "eventstudy_metadata_contract"
            ].__setitem__("cohort_estimate_fields", ["estimate"]),
            "eventstudy_metadata_contract must freeze event-study support",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_eventstudy_metadata_drift(
    mutation,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "default_columns", {"id_column": "unit"}
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "runtime_rule", "default columns only"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "id_nonmissing_rule", "ids may be missing"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "time_period_grid_rule", "any sortable time values"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "dose_support_rule", "dose may be negative"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "group_timing_grid_rule", "any group values"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "treated_timing_positive_dose_rule", "G>0 may have D=0"
        ),
        lambda contract: contract["object_model"]["panel_column_contract"].__setitem__(
            "two_period_post_timing_rule", "any positive G"
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_panel_contract_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "panel_column_contract must freeze PanelData column override "
            "and runtime validation semantics"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "treatment_type"
        ].__setitem__("supported_values", ["continuous", "discrete"]),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "treatment_type"
        ].__setitem__("discrete_error", "discrete allowed"),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "boot_type"
        ].__setitem__("supported_values", ["multiplier", "wild"]),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "alp"
        ].__setitem__("unsupported_value_error", "alpha invalid"),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "biters"
        ].__setitem__("rule", "nonnegative integer"),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "bstrap"
        ].__setitem__("rule", "truthy flag"),
        lambda contract: contract["object_model"]["runtime_spec_guard_contract"][
            "cband"
        ].__setitem__("rule", "truthy flag"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_runtime_spec_guard_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "runtime_spec_guard_contract must freeze treatment, bootstrap, "
            "and inference knob hard-fail semantics"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value"),
    [
        ("panel_input", "PanelFrame"),
        ("runtime_spec", "Spec"),
        ("result_object", "Result"),
        ("empirical_scaffold_result", "EmpiricalResult"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_object_model_label_drift(
    contract_key: str,
    corrupted_value: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["object_model"][contract_key] = corrupted_value

    with pytest.raises(
        PublicAPIContractError,
        match=(
            f"object_model.{contract_key} must freeze the checked public object label"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["public_route_controls"][
            "default_dose_grid_reference_sources"
        ]["parametric"].__setitem__("source", "contdid-r/R/cont_did.R"),
        lambda contract: contract["public_route_controls"][
            "default_dose_grid_reference_sources"
        ]["parametric"].__setitem__("rule", "dvals defaults to observed support"),
        lambda contract: contract["public_route_controls"][
            "default_dose_grid_reference_sources"
        ]["cck"].__setitem__("source", "contdid-r/R/setup_pte_cont.R"),
        lambda contract: contract["public_route_controls"][
            "default_dose_grid_reference_sources"
        ]["cck"].__setitem__("rule", "dvals defaults to 99 quantiles"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_default_dose_grid_reference_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "default_dose_grid_reference_sources must freeze the R reference "
            "dvals rules"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["release_examples"][
            "synthetic-dose-demo"
        ].__setitem__("source_manifest_entry", "synthetic-v2"),
        lambda contract: contract["release_examples"][
            "synthetic-dose-demo"
        ].__setitem__("source_surface", "simulate_data"),
        lambda contract: contract["release_examples"][
            "synthetic-dose-demo"
        ].__setitem__("consumer_surface", "simulate_contdid_data"),
        lambda contract: contract["release_examples"]["synthetic-dose-demo"][
            "release_surfaces"
        ].append("private-api"),
        lambda contract: contract["release_examples"][
            "synthetic-dose-demo"
        ].__setitem__("estimand_family", "effect"),
        lambda contract: contract["release_examples"][
            "synthetic-dose-demo"
        ].__setitem__("dose_control_group", "notyettreated"),
        lambda contract: contract["release_examples"][
            "medicare-scaffold-demo"
        ].__setitem__("source_manifest_entry", "medicare-v2"),
        lambda contract: contract["release_examples"][
            "medicare-scaffold-demo"
        ].__setitem__("source_surface", "load_medicare_panel"),
        lambda contract: contract["release_examples"]["medicare-scaffold-demo"][
            "consumer_surfaces"
        ].__setitem__("eventstudy", "prepare_medicare_pps_panel"),
        lambda contract: contract["release_examples"]["medicare-scaffold-demo"][
            "release_surfaces"
        ].append("private-api"),
        lambda contract: contract["release_examples"][
            "medicare-scaffold-demo"
        ].__setitem__("honesty_label", "licensed-parity"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_release_example_route_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="release_examples must freeze the checked release example routing",
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["english_release_docs"].__setitem__(
            "language", "English and Chinese"
        ),
        lambda contract: contract["english_release_docs"]["required_consumers"].append(
            "private-notebook"
        ),
        lambda contract: contract["english_release_docs"][
            "required_honesty_labels"
        ].__setitem__(0, "paper-parity"),
        lambda contract: contract["english_release_docs"][
            "forbidden_claims"
        ].__setitem__(0, "licensed data unavailable"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_english_release_docs_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="english_release_docs must freeze English release consumers",
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract["notes"].append("new empirical truth source allowed"),
        lambda contract: contract["notes"].__setitem__(
            0, "Public terms may ignore the Phase 2 symbol map."
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_release_note_boundary_drift(
    mutation,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    mutation(corrupted)

    with pytest.raises(
        PublicAPIContractError,
        match="notes must freeze the release traceability and truth-source boundaries",
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value"),
    [
        ("dose_summary_aggregates_alias_field", "summaries"),
        ("dose_route_anticipation_rule", "anticipation allowed"),
        ("dose_route_control_group_rule", "all control groups"),
        ("dose_route_time_support_rule", "multi-period supported"),
        ("dose_route_grid_support_rule", "extrapolation allowed"),
        ("dose_route_identification_rule", "ordinary PT identifies ATT"),
        ("dose_route_inference_df_rule", "df optional"),
        ("dose_route_treated_covariance_rule", "diagonal only"),
        ("dose_route_benchmark_variance_rule", "ignore untreated variance"),
        ("dose_route_benchmark_variance_df_rule", "single untreated ok"),
        ("dose_route_guard_precedence_rule", "control-group first"),
        ("linear_spline_hinge_derivative_rule", "right derivative"),
    ],
)
def test_phase9_runtime_contract_validator_rejects_dose_route_rule_drift(
    contract_key: str,
    corrupted_value: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"][contract_key] = corrupted_value

    with pytest.raises(
        PublicAPIContractError,
        match=(
            f"public API contract result_payload.{contract_key} must freeze "
            "the checked dose-route guard and inference semantics"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_release_packet_mirror_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"]["release_packet_metadata_mirror_fields"] = ["estimand"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "release_packet_metadata_mirror_fields must freeze the result fields "
            "mirrored into release packets"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value", "match"),
    [
        (
            "estimate_shape_rule",
            "estimate may be scalar",
            "confidence_band_shape_contract must freeze the estimate shape rule",
        ),
        (
            "estimate_finiteness_rule",
            "estimate may contain missing values",
            "confidence_band_shape_contract must freeze the estimate finiteness rule",
        ),
        (
            "std_error_finiteness_rule",
            "standard errors may be negative",
            "confidence_band_shape_contract must freeze the standard-error finiteness rule",
        ),
        (
            "critical_value_rule",
            "critical value may be missing",
            "confidence_band_shape_contract must freeze the critical-value rule",
        ),
        (
            "result_container_interval_rule",
            "result intervals may have any shape",
            "confidence_band_shape_contract must freeze the result-container interval rule",
        ),
        (
            "result_container_band_rule",
            "result bands may have any shape",
            "confidence_band_shape_contract must freeze the result-container band rule",
        ),
        (
            "result_container_critical_value_echo_rule",
            "result critical values may disagree",
            "confidence_band_shape_contract must freeze the result-container critical-value echo rule",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_confidence_band_rule_drift(
    contract_key: str,
    corrupted_value: str,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"]["confidence_band_shape_contract"][contract_key] = (
        corrupted_value
    )

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


@pytest.mark.parametrize(
    ("contract_key", "corrupted_value", "match"),
    [
        (
            "covariance_shape_rule",
            "covariance may be rectangular",
            "multiplier_bootstrap_covariance_contract must freeze the covariance shape rule",
        ),
        (
            "covariance_finiteness_rule",
            "covariance may contain missing values",
            "multiplier_bootstrap_covariance_contract must freeze the covariance finiteness rule",
        ),
        (
            "covariance_symmetry_rule",
            "covariance may be asymmetric",
            "multiplier_bootstrap_covariance_contract must freeze the covariance symmetry rule",
        ),
        (
            "negative_eigenvalue_tolerance_rule",
            "negative eigenvalues may be clipped",
            "multiplier_bootstrap_covariance_contract must freeze the negative-eigenvalue tolerance rule",
        ),
        (
            "loadings_shape_rule",
            "loadings may be one-dimensional",
            "multiplier_bootstrap_covariance_contract must freeze the loadings shape rule",
        ),
        (
            "loadings_dimension_rule",
            "loadings dimensions may be inferred",
            "multiplier_bootstrap_covariance_contract must freeze the loadings dimension rule",
        ),
        (
            "loadings_finiteness_rule",
            "loadings may contain missing values",
            "multiplier_bootstrap_covariance_contract must freeze the loadings finiteness rule",
        ),
        (
            "result_length_rule",
            "result length may differ from loading rows",
            "multiplier_bootstrap_covariance_contract must freeze the result length rule",
        ),
        (
            "result_grid_length_rule",
            "grid length may differ from estimate length",
            "multiplier_bootstrap_covariance_contract must freeze the result grid length rule",
        ),
        (
            "result_normalization_rule",
            "result vectors may keep caller-provided container types",
            "multiplier_bootstrap_covariance_contract must freeze the result normalization rule",
        ),
    ],
)
def test_phase9_runtime_contract_validator_rejects_multiplier_payload_rule_drift(
    contract_key: str,
    corrupted_value: str,
    match: str,
) -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    corrupted = copy.deepcopy(_load_json(CONTRACT_PATH))
    corrupted["result_payload"]["multiplier_bootstrap_covariance_contract"][
        contract_key
    ] = corrupted_value

    with pytest.raises(PublicAPIContractError, match=match):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_dose_identification_contract() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    del corrupted["result_payload"]["dose_route_identification_contract"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract dose_route_identification_contract must be a "
            "JSON object"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_blank_dose_identification_fields() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"]["dose_route_identification_contract"]["slope"][
        "identifying_assumption"
    ] = " "

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract dose_route_identification_contract.slope."
            "identifying_assumption must be a non-empty string"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_dose_identification_semantic_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"]["dose_route_identification_contract"]["level"][
        "ordinary_pt_interpretation"
    ] = "ATT(d)"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "dose_route_identification_contract.level must freeze "
            "the paper PT/SPT interpretation boundary"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"]["dose_route_identification_contract"]["slope"][
        "ordinary_pt_interpretation"
    ] = "causal ACRT(d) under ordinary PT"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "dose_route_identification_contract.slope must freeze "
            "the paper PT/SPT interpretation boundary"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_missing_eventstudy_identification_contract() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    del corrupted["result_payload"]["eventstudy_identification_contract"]

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "public API contract eventstudy_identification_contract must be a "
            "JSON object"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_eventstudy_identification_semantic_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"]["eventstudy_identification_contract"]["level"][
        "identifying_assumption"
    ] = "SPT"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "eventstudy_identification_contract.level must freeze "
            "the paper PT-MP/SPT-MP interpretation boundary"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_runtime_contract_validator_rejects_result_container_identification_rule_drift() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["result_payload"][
        "result_container_identification_rule"
    ] = "stale identification metadata may pass through"

    with pytest.raises(
        PublicAPIContractError,
        match=(
            "result_container_identification_rule must freeze the "
            "result-container identification metadata boundary"
        ),
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_public_api_contract_freezes_release_exports_traceability_and_result_fields() -> (
    None
):
    import contdid
    from contdid.results import ContDIDResult
    from contdid.specs import ContDIDSpec

    contract = _load_json(CONTRACT_PATH)
    phase2_symbol_map = _load_json(PHASE2_SYMBOL_MAP_PATH)

    for export_name in contract["stable_top_level_exports"]:
        assert hasattr(contdid, export_name), export_name

    assert "__version__" in contract["stable_top_level_exports"]
    assert contract["stable_top_level_exports"] == list(contdid.__all__)
    assert contdid.__version__ == _package_version()

    assert (
        contract["traceability"]["public_terms"]
        == phase2_symbol_map["public_api_boundary"]["public_terms"]
    )
    assert (
        contract["traceability"]["internal_trace_terms"]
        == phase2_symbol_map["public_api_boundary"]["internal_trace_terms"]
    )
    assert contract["traceability"]["estimand_routes"] == {
        "att": {
            "package_surface": "estimate_dose_effects",
            "result_estimand": "ATT(d)",
            "paper_symbol": "ATT(d)",
        },
        "acrt": {
            "package_surface": "estimate_dose_slope_effects",
            "result_estimand": "ACRT(d)",
            "paper_symbol": "ACRT(d)",
        },
        "eventstudy": {
            "package_surface": "estimate_eventstudy_effects",
            "result_estimand": "ATT(event_time)",
            "paper_symbol": "ATT(event_time)",
        },
        "eventstudy_slope": {
            "package_surface": "estimate_eventstudy_slope_effects",
            "result_estimand": "ACRT(event_time)",
            "paper_symbol": "ACRT(event_time)",
        },
    }
    assert contract["traceability"]["eventstudy_input_contract"] == {
        "supported_control_groups": [
            "notyettreated",
            "nevertreated",
        ],
        "unsupported_aggregation_error": (
            "event-study estimators require aggregation='eventstudy'"
        ),
        "anticipation_rule": "anticipation=0 only",
        "unsupported_dose_est_method_rule": (
            "event study must raise ContDIDValidationError when "
            "dose_est_method='cck' because the checked public event-study "
            "surface does not support the CCK estimator yet; CCK panel-shape "
            "unsupported guards take precedence over event-study control-group guards"
        ),
        "cck_panel_shape_guard_precedence_rule": (
            "For CCK requests with panel data, staggered adoption must raise "
            "cck estimator not supported with staggered adoption yet before "
            "the generic two-period CCK guard, post-period timing guard, "
            "event-study CCK hard-fail, or event-study control-group guard; single-cohort multi-period "
            "panels must raise cck estimator not supported with more than two "
            "time periods. consider averaging across pre and post treatment "
            "periods before the event-study CCK hard-fail; two-period "
            "single-cohort CCK panels whose positive treatment timing does "
            "not start in the last observed period must raise cck estimator "
            "requires positive treatment timing to start in the post period."
        ),
        "cck_boundary_minimum_runtime_test_nodes": [
            "contdid-py/tests/test_phase6_cck_guards.py::test_cck_rejects_staggered_adoption_with_exact_manifest_substring",
            "contdid-py/tests/test_phase6_cck_guards.py::test_cck_staggered_adoption_guard_precedes_multi_period_guard",
            "contdid-py/tests/test_phase6_cck_guards.py::test_cck_dose_staggered_guard_precedes_unchecked_control_group",
            "contdid-py/tests/test_phase6_cck_guards.py::test_cck_requires_exactly_two_time_periods",
            "contdid-py/tests/test_phase6_cck_guards.py::test_cck_requires_positive_timing_in_post_period_with_cck_error",
            "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_path_stays_hard_failed",
            "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_post_timing_guard_precedes_eventstudy_guard",
            "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_staggered_guard_precedes_multi_period_and_control_group",
        ],
        "unsupported_control_group_error": (
            "event-study aggregation supports control_group values "
            "'notyettreated' and 'nevertreated' only"
        ),
        "notyettreated_comparison_rule": (
            "notyettreated comparisons exclude the focal timing group and "
            "require comparison units to remain untreated through the later "
            "of the target event-study period and the chosen base period "
            "before support counts are computed"
        ),
        "eventstudy_evaluation_grid_support_rule": (
            "event-study evaluation grids must be drawn from observed-window "
            "positive-dose treated cohorts that survive admissible base-period "
            "and local inference-degree-of-freedom filtering; "
            "first-period/no-baseline cohorts, after-window comparison-only "
            "not-yet-treated doses, and locally unidentified cohorts cannot define "
            "default quantiles or satisfy explicit dvals support"
        ),
        "eventstudy_base_period_rule": (
            "base_period accepts None/'varying' for adjacent pre-period "
            "comparisons, 'universal' for cohort-specific g-anticipation-1 "
            "comparisons with the normalized reference cell omitted, or an "
            "observed integer baseline for fixed-baseline event-study "
            "comparisons; fixed and universal comparisons support "
            "control_group values 'notyettreated' and 'nevertreated'; timing "
            "groups whose treatment timing is not preceded by the selected "
            "baseline are omitted before event-time support is built, and the "
            "route hard-fails if no treated cohort has an admissible base period"
        ),
        "post_treatment_identification_rule": (
            "requires at least one locally identified post-treatment event time "
            "with positive-dose support and inference degrees of freedom"
        ),
    }
    assert contract["object_model"]["panel_column_contract"] == {
        "container": "PanelData",
        "default_columns": {
            "id_column": "id",
            "time_column": "time_period",
            "outcome_column": "Y",
            "group_column": "G",
            "dose_column": "D",
        },
        "runtime_rule": (
            "validate_panel_data, build_dose_grid, and the public estimation entrypoints "
            "must honor PanelData column overrides instead of hard-requiring the "
            "default manifest column names"
        ),
        "id_nonmissing_rule": (
            "id values must be nonmissing before balanced-panel sorting and "
            "first/last differences are formed"
        ),
        "time_period_grid_rule": (
            "time-period values must be finite numeric values on a consecutive integer "
            "grid before balanced-panel sorting and first/last differences run"
        ),
        "dose_support_rule": (
            "dose values must be nonnegative and finite numeric values so D = 0 "
            "denotes untreated units and D > 0 defines positive-dose treated support"
        ),
        "group_timing_finiteness_rule": (
            "group timing values must be nonnegative and finite numeric values so "
            "G = 0 denotes never-treated units and G > 0 defines treatment timing"
        ),
        "group_timing_grid_rule": (
            "positive group timing values must align with or follow the observed "
            "integer time-period grid so in-window G > 0 denotes realized treatment "
            "start periods and G after the last observed period denotes "
            "comparison-only not-yet-treated timing"
        ),
        "outcome_finiteness_rule": (
            "outcome values must be finite numeric values before first/last differences are formed"
        ),
        "treated_timing_positive_dose_rule": (
            "positive treatment timing must have positive dose so G > 0 cannot be silently "
            "pooled into the untreated benchmark"
        ),
        "two_period_post_timing_rule": (
            "two-period dose aggregation requires positive treatment timing to start "
            "in the post period so first/last differences retain an untreated baseline"
        ),
        "phase12_data_shape_boundary_contract": (
            "contdid-py/contracts/phase12/data_shape_boundary_contract.json"
        ),
        "unsupported_data_shapes": {
            "unbalanced_panel": "panel must be balanced",
            "repeated_cross_sections": (
                "repeated cross-sections data is not supported by the PanelData contract"
            ),
            "time_varying_dose": "panel violates within-unit D constancy",
            "covariate_aware_identification": (
                "additive covariate adjustment is supported via ContDIDSpec.covariates; "
                "interaction terms and nonparametric covariate adjustment are not supported"
            ),
        },
    }
    assert contract["object_model"]["runtime_spec_guard_contract"] == {
        "treatment_type": {
            "default": "continuous",
            "supported_values": ["continuous"],
            "discrete_error": (
                "discrete treatment is not supported in the current implementation. "
                "The paper (arXiv-2107.02637v7, Assumption 4b) covers multi-valued "
                "discrete treatment theoretically, but the saturated-regression "
                "estimator (Eq. 13) has not been implemented yet."
            ),
            "unsupported_value_error": (
                "unsupported treatment_type; expected 'continuous'"
            ),
        },
        "boot_type": {
            "default": "multiplier",
            "supported_values": ["multiplier", "rademacher", "mammen"],
            "unsupported_value_error": "boot_type must be 'multiplier', 'rademacher', or 'mammen'",
        },
        "alp": {
            "rule": "finite real value strictly between 0 and 1",
            "unsupported_value_error": "alp must lie strictly between 0 and 1",
        },
        "biters": {
            "rule": "positive integer bootstrap iteration count",
            "unsupported_value_error": "biters must be a positive integer",
        },
        "bstrap": {
            "rule": "boolean bootstrap enable flag",
            "unsupported_value_error": "bstrap must be a boolean",
        },
        "cband": {
            "rule": "boolean confidence-band enable flag",
            "unsupported_value_error": "cband must be a boolean",
        },
    }

    spec_fields = {field.name: field for field in fields(ContDIDSpec)}
    assert spec_fields["treatment_type"].default == "continuous"
    assert spec_fields["boot_type"].default == "multiplier"

    result_fields = {field.name for field in fields(ContDIDResult)}
    assert set(contract["result_payload"]["required_fields"]).issubset(result_fields)
    assert contract["result_payload"]["eventstudy_fields"] == [
        "timing_group",
        "event_time",
        "event_time_grid",
        "cohort_summary",
    ]
    assert set(contract["result_payload"]["eventstudy_fields"]).issubset(result_fields)
    assert contract["result_payload"]["display_methods"] == [
        "to_frame",
        "to_markdown",
        "save_plot",
    ]
    for method_name in contract["result_payload"]["display_methods"]:
        assert hasattr(ContDIDResult, method_name)
    assert set(contract["result_payload"]["required_fields"]) >= {
        "critical_value",
        "confidence_interval",
        "confidence_band",
    }
    assert set(contract["result_payload"]["metadata_must_carry"]) == {
        "target_parameter",
        "dose_est_method",
        "inference",
        "bootstrap_type",
        "alp",
        "biters",
        "bootstrap_seed",
        "cband",
        "bstrap",
        "confidence_band_kind",
    }
    assert contract["result_payload"]["inference_control_echo_contract"] == {
        "alpha_field": "alp",
        "bootstrap_flag_field": "bstrap",
        "bootstrap_iteration_field": "biters",
        "bootstrap_seed_field": "bootstrap_seed",
        "confidence_band_flag_field": "cband",
        "bootstrap_disabled_label": "analytic",
        "bootstrap_enabled_label": "bootstrap",
    }
    assert contract["result_payload"]["multiplier_bootstrap_covariance_contract"] == {
        "covariance_shape_rule": "covariance matrix must be square",
        "covariance_finiteness_rule": "covariance matrix must contain only finite values",
        "covariance_symmetry_rule": "covariance matrix must be symmetric",
        "covariance_psd_rule": "covariance matrix must be positive semidefinite",
        "negative_eigenvalue_tolerance_rule": (
            "machine-precision rank noise may be clipped to zero, but material "
            "negative eigenvalues must raise ValueError"
        ),
        "loadings_shape_rule": "loadings must be a two-dimensional matrix",
        "loadings_nonempty_rule": "loadings must contain at least one estimand row",
        "loadings_dimension_rule": "loadings and covariance dimensions must align",
        "loadings_finiteness_rule": "loadings must contain only finite values",
        "result_length_rule": (
            "loadings row count must match result estimate length before attaching inference"
        ),
        "result_grid_length_rule": (
            "result grid must match result estimate length before attaching inference"
        ),
        "result_nonempty_rule": "result estimates must contain at least one value",
        "result_normalization_rule": (
            "attach_inference_payload must normalize finite result grid and estimate "
            "vectors to JSON-serializable float lists before echoing metadata"
        ),
        "seed_rule": "seed must be a nonnegative integer",
    }
    assert contract["result_payload"]["confidence_band_shape_contract"] == {
        "estimate_shape_rule": "estimate and std_error must be one-dimensional",
        "shape_match_rule": "estimate and std_error must have the same shape",
        "nonempty_rule": "estimate and std_error must contain at least one value",
        "estimate_finiteness_rule": "estimate must contain only finite non-boolean values",
        "std_error_finiteness_rule": (
            "std_error must contain only finite non-boolean nonnegative values"
        ),
        "critical_value_rule": (
            "critical_value must be a finite non-boolean nonnegative scalar"
        ),
        "result_container_interval_rule": (
            "ContDIDResult confidence_interval must have one finite lower/upper pair "
            "per estimate, lower bounds not exceeding upper bounds, and the point "
            "estimate inside each interval"
        ),
        "result_container_band_rule": (
            "ContDIDResult confidence_band lower and upper arrays must match estimate "
            "shape, contain only finite values, contain each point estimate, and carry "
            "a finite nonnegative critical_value"
        ),
        "result_container_critical_value_echo_rule": (
            "ContDIDResult critical_value must match confidence_band critical_value "
            "when both are present, and must be backfilled from confidence_band when "
            "only the band critical value is provided"
        ),
    }
    assert contract["result_payload"]["metadata_forbidden_fields"] == [
        "boot_type",
        "backend_name",
    ]
    assert contract["result_payload"]["summary_field_families"] == {
        "dose_level": [
            "overall_att",
            "overall_att_uniform_support",
            "dose_grid_mean_att",
            "dose_grid_min",
            "dose_grid_max",
        ],
        "dose_slope": [
            "overall_acrt",
            "overall_acrt_uniform_support",
            "dose_grid_mean_acrt",
            "dose_grid_min",
            "dose_grid_max",
        ],
    }
    assert contract["result_payload"]["dose_route_metadata_contract"] == {
        "basis_field": "basis",
        "basis_keys": [
            "type",
            "degree",
            "num_knots",
            "interior_knots",
        ],
        "basis_types": [
            "global_polynomial",
            "bspline",
            "cck_polynomial_backend",
        ],
        "lineage_field": "source_estimator",
        "lineage_values": [
            "phase4_shared_dose_stack",
            "phase6_cck_backend",
        ],
    }
    assert contract["result_payload"]["dose_route_traceability_contract"] == {
        "treated_sample_field": "treated_sample",
        "treated_sample_keys": [
            "positive_dose_mean",
            "treated_count",
            "untreated_count",
        ],
        "untreated_benchmark_field": "untreated_benchmark",
        "delta_outcome_field": "delta_outcome_construction",
        "delta_outcome_value": "last observed period minus first observed period",
        "derivative_field": "derivative_construction",
        "derivative_value": "shared derivative basis on the ATT(d) dose grid",
    }
    assert contract["result_payload"]["dose_route_identification_contract"] == {
        "metadata_field": "identification",
        "required_keys": [
            "paper_estimand",
            "identifying_assumption",
            "ordinary_pt_interpretation",
            "identification_note",
        ],
        "level": {
            "paper_estimand": "ATT(d)",
            "identifying_assumption": "SPT",
            "ordinary_pt_interpretation": "LATT(d|d)",
            "identification_note": (
                "The same dose-specific contrast identifies LATT(d|d) under "
                "ordinary PT; interpreting it as ATT(d) requires SPT."
            ),
        },
        "slope": {
            "paper_estimand": "ACRT(d)",
            "identifying_assumption": "SPT + continuous dose support",
            "ordinary_pt_interpretation": (
                "derivative of LATT(d|d) with local selection-bias contamination"
            ),
            "identification_note": (
                "Ordinary PT is not enough for a causal ACRT(d) interpretation; "
                "the public slope route reports the SPT-based causal-response label."
            ),
        },
    }
    assert contract["result_payload"]["eventstudy_identification_contract"] == {
        "metadata_field": "identification",
        "required_keys": [
            "paper_estimand",
            "identifying_assumption",
            "ordinary_pt_interpretation",
            "identification_note",
        ],
        "level": {
            "paper_estimand": "ATT(event_time)",
            "identifying_assumption": "PT-MP",
            "ordinary_pt_interpretation": (
                "post-treatment ATT(event_time); negative event-time cells are "
                "pre-trend diagnostics"
            ),
            "identification_note": (
                "Post-treatment ATT(event_time) cells are identified by "
                "PT-MP/local binary event-study comparisons; negative event-time "
                "cells diagnose pre-treatment parallel-trends plausibility rather "
                "than treatment effects."
            ),
        },
        "slope": {
            "paper_estimand": "ACRT(event_time)",
            "identifying_assumption": "SPT-MP + continuous dose support",
            "ordinary_pt_interpretation": (
                "derivative of event-time LATT path with local selection-bias "
                "contamination under PT-MP alone"
            ),
            "identification_note": (
                "The public slope event-study route reports the SPT-MP "
                "causal-response label; under PT-MP alone, differentiating "
                "event-time paths can retain selection-bias terms."
            ),
        },
    }
    assert contract["result_payload"]["result_container_identification_rule"] == (
        "ContDIDResult must reject stale identification metadata and only preserve "
        "the checked paper_estimand, identifying_assumption, "
        "ordinary_pt_interpretation, and identification_note payload for ATT(d), "
        "ACRT(d), ATT(event_time), and ACRT(event_time)"
    )
    assert contract["result_payload"]["dose_summary_aggregates_alias_field"] == (
        "summary_aggregates"
    )
    assert contract["result_payload"]["eventstudy_summary_aggregates"] == {
        "level": [
            "overall_level",
            "post_treatment_mean_level",
        ],
        "slope": [
            "overall_slope",
            "post_treatment_mean_slope",
        ],
        "cohort_summary_weighting": "treated-sample-share",
        "cohort_treated_count_weight_rule": (
            "cohort_summary aggregation weights are formed only from "
            "finite positive-integer treated_count values"
        ),
        "overall_weighting": (
            "treated-sample-share over locally identified post-treatment "
            "timing-group comparisons only"
        ),
        "post_treatment_mean_rule": (
            "simple mean over post-treatment event-time cells; negative "
            "event-time pre-test cells are excluded"
        ),
        "local_support_rule": (
            "ATT(event_time) uses binary treated/control mean support; ACRT(event_time) uses positive-dose derivative-basis support with local inference degrees of freedom"
        ),
        "cohort_summary_point_estimate_rule": (
            "cohort_summary aggregation_weight values reconstruct the public "
            "event-time point estimate"
        ),
        "cohort_summary_standard_error_rule": (
            "event-time cells with multiple timing groups do not reconstruct the "
            "public std_error by combining nested cohort standard errors alone; "
            "public event-study std_error comes from the event-study "
            "influence-covariance aggregation path"
        ),
        "explicit_dvals_role": "evaluation-grid-only",
    }
    assert contract["result_payload"]["eventstudy_inference_covariance_rule"] == (
        "event-study simultaneous confidence bands must compute critical values "
        "from the full event-time covariance matrix rather than treating "
        "event-time cells as independent"
    )
    assert contract["result_payload"]["eventstudy_metadata_contract"] == {
        "timing_group_support_fields": [
            "timing_groups",
            "never_treated_group",
            "reporting_scale",
            "base_period_strategy",
        ],
        "base_period_field": "base_period",
        "timing_group_reporting_scale": "length of exposure to treatment",
        "top_level_support_field": "support",
        "summary_alias_field": "summary",
        "summary_aggregates_alias_field": "summary_aggregates",
        "inference_covariance_field": "inference_covariance",
        "inference_covariance_value": "full_event_time_covariance",
        "evaluation_grid_field": "dose_grid",
        "basis_field": "basis",
        "basis_keys": [
            "type",
            "degree",
            "num_knots",
            "interior_knots",
        ],
        "basis_origin_rule": (
            "event-study routes compute positive-num_knots interior knots once "
            "from observed-window positive-dose treated cohorts and require "
            "that shared basis to match the post-local-inference support of "
            "the reported timing groups"
        ),
        "lineage_field": "source_estimator",
        "lineage_values": {
            "level": "binary_eventstudy_mean",
            "slope": "phase4_shared_dose_stack",
        },
        "shape_constraints_field": "shape_constraints",
        "shape_constraint_keys": [
            "level_curve",
            "slope_curve",
            "event_time_order",
        ],
        "shape_constraint_honesty": "no flat-zero shape restriction across event time",
        "result_container_event_time_rule": (
            "ContDIDResult ATT(event_time)/ACRT(event_time) estimands require "
            "event_time or event_time_grid, non-event-study estimands must not carry "
            "event-time or timing_group fields, event_time and event_time_grid must "
            "be integer vectors that match the result grid; cohort_summary requires "
            "one of those event-time vectors, and every cohort_summary row must "
            "carry normalized integer event_time and follow the same event-time order; "
            "timing_group, cohort_summary timing_groups, and cohort_estimates timing_group "
            "must be positive treated timing-group identifiers; cohort_summary "
            "mean_estimate must match the public estimate, std_error must be finite, "
            "nonnegative, and match the public std_error, support must be present "
            "on every row and match cohort_estimates presence, and "
            "cohort_estimates estimate/std_error/count/weight/period fields must be "
            "finite, nonnegative, positive-count, normalized-weight, and integer-valued "
            "as appropriate; aggregation weights require treated_count and must equal "
            "treated-count shares within each event-time cell; when cohort_estimates "
            "carry estimates and aggregation weights, the weighted cohort estimates "
            "must reconstruct the public estimate; "
            "metadata support must be a boolean vector matching every cohort_summary support row; "
            "timing_group_support must carry matching timing_groups, "
            "never_treated_group = 0, reporting_scale = length of exposure to treatment, "
            "and base_period_strategy in fixed/universal/varying_pre_period"
        ),
        "cohort_summary_fields": [
            "event_time",
            "timing_groups",
            "cohort_estimates",
            "mean_estimate",
            "std_error",
            "support",
        ],
        "cohort_estimate_fields": [
            "timing_group",
            "time_period",
            "base_period",
            "comparison_count",
            "treated_count",
            "aggregation_weight",
            "estimate",
            "std_error",
        ],
    }
    assert contract["result_payload"]["dose_route_anticipation_rule"] == (
        "nonzero anticipation must raise ContDIDValidationError because the checked dose aggregation routes support anticipation=0 only"
    )
    assert contract["result_payload"]["dose_route_control_group_rule"] == (
        "dose aggregation must raise ContDIDValidationError unless control_group='nevertreated' because checked notyettreated/eventuallytreated timing semantics have not landed on the public dose routes"
    )
    assert contract["result_payload"]["dose_route_time_support_rule"] == (
        "parametric dose aggregation must raise ContDIDValidationError unless the observed panel has exactly two time periods because checked multi-period dose timing semantics have not landed on the public dose routes"
    )
    assert contract["result_payload"]["dose_route_grid_support_rule"] == (
        "explicit dvals must contain only finite non-boolean numeric values and stay within the "
        "observed positive-dose treated support, and multi-point explicit dvals "
        "must be strictly increasing with no duplicate dose values"
    )
    assert contract["result_payload"]["dose_route_identification_rule"] == (
        "rank-deficient positive-dose treated supports must raise "
        "ContDIDValidationError on both the shared parametric and supported "
        "CCK dose routes instead of returning an underidentified fit"
    )
    assert contract["result_payload"]["dose_route_inference_df_rule"] == (
        "exact-fit positive-dose treated supports must raise "
        "ContDIDValidationError on both the shared parametric and supported "
        "CCK dose routes because inference requires treated_count > basis_columns"
    )
    assert contract["result_payload"]["dose_route_treated_covariance_rule"] == (
        "shared parametric ATT(d)/ACRT(d) standard errors must use the "
        "treated-fit sandwich covariance from positive-dose design scores "
        "rather than a homoskedastic OLS covariance"
    )
    assert contract["result_payload"]["dose_route_benchmark_variance_rule"] == (
        "ATT(d) std_error and interval payloads on the shared parametric and "
        "supported CCK dose routes include the independent variance of the "
        "untreated benchmark mean; ACRT(d) payloads exclude that benchmark "
        "component"
    )
    assert contract["result_payload"]["dose_route_benchmark_variance_df_rule"] == (
        "ATT(d) level dose inference must raise ContDIDValidationError when "
        "the untreated benchmark variance is not estimable because the shared "
        "parametric and supported CCK dose routes require at least two untreated units"
    )
    assert contract["result_payload"]["dose_route_guard_precedence_rule"] == (
        "treated-support hard-fails must run before ATT(d) untreated benchmark "
        "variance guards on the shared parametric and supported CCK dose routes"
    )
    assert contract["result_payload"]["linear_spline_hinge_derivative_rule"] == (
        "degree=1 bspline ACRT(d) hinge derivatives must stay "
        "zero below each interior knot and switch on at and above the knot "
        "to match the paper right-derivative convention"
    )
    assert contract["result_payload"]["inference_labels"] == [
        "bootstrap",
        "analytic",
    ]
    assert contract["result_payload"]["bootstrap_type_labels"] == [
        "multiplier",
        "analytic",
    ]
    assert contract["result_payload"]["confidence_band_kind_labels"] == [
        "simultaneous_multiplier",
        "pointwise_multiplier",
        "pointwise_analytic",
    ]
    assert contract["result_payload"]["release_packet_metadata_mirror_fields"] == [
        "estimand",
        "grid",
        "estimate",
        "std_error",
        "critical_value",
        "confidence_interval",
        "confidence_band",
    ]
    assert contract["public_route_controls"]["shared_dose_basis_kwargs"] == {
        "surfaces": [
            "estimate_dose_effects",
            "estimate_dose_level_effects",
            "estimate_dose_slope_effects",
            "estimate_eventstudy_effects",
            "estimate_eventstudy_slope_effects",
        ],
        "dvals": {
            "kind": "keyword-only",
            "default": None,
            "accepted_values": [
                None,
                "finite non-boolean numeric scalar dose value",
                "finite non-boolean numeric iterable dose grid",
            ],
            "role": "evaluation grid override",
        },
        "degree": {
            "kind": "keyword-only",
            "default": 3,
            "minimum": 1,
            "integer": True,
            "role": "shared dose-basis polynomial degree",
        },
        "num_knots": {
            "kind": "keyword-only",
            "default": 0,
            "minimum": 0,
            "integer": True,
            "role": (
                "shared dose-basis interior knot count placed at positive-dose treated quantiles"
            ),
        },
    }
    assert contract["public_route_controls"]["eventstudy_base_period_kwargs"] == {
        "surfaces": [
            "estimate_eventstudy_effects",
            "estimate_eventstudy_slope_effects",
        ],
        "base_period": {
            "kind": "keyword-only",
            "default": None,
            "accepted_values": [
                None,
                "varying",
                "universal",
                "observed integer time period",
            ],
            "role": (
                "optional event-study baseline control: None/'varying' uses "
                "adjacent pre-period comparisons, 'universal' uses each cohort's "
                "g-anticipation-1 baseline, and an integer fixes a shared "
                "observed pre-treatment reference period"
            ),
        },
    }
    for surface in contract["public_route_controls"]["shared_dose_basis_kwargs"][
        "surfaces"
    ]:
        signature = inspect.signature(getattr(contdid, surface))
        dvals = signature.parameters["dvals"]
        degree = signature.parameters["degree"]
        num_knots = signature.parameters["num_knots"]
        assert dvals.kind is inspect.Parameter.KEYWORD_ONLY
        assert dvals.default is None
        assert str(dvals.annotation) == "Iterable[float] | float | None"
        assert degree.kind is inspect.Parameter.KEYWORD_ONLY
        assert degree.default == 3
        assert num_knots.kind is inspect.Parameter.KEYWORD_ONLY
        assert num_knots.default == 0
    for surface in contract["public_route_controls"]["eventstudy_base_period_kwargs"][
        "surfaces"
    ]:
        signature = inspect.signature(getattr(contdid, surface))
        base_period = signature.parameters["base_period"]
        assert base_period.kind is inspect.Parameter.KEYWORD_ONLY
        assert base_period.default is None

    docs_bundle = contract["docs_bundle"]
    assert set(docs_bundle["public_doc_forbidden_terms"]) >= {
        "descriptive-or-scaffold-only",
        "source_surface",
        "package_surface",
        "package_surfaces",
        "manifest_path",
        "run_mode",
        "parity_claim_allowed",
        "parity_viability",
        "allowed_use",
        "claim_status",
        "release-gate",
        "phase4_shared_dose_stack",
        "phase6_cck_backend",
    }
    expected_reader_terms = {
        "readme_required_terms": {
            "PanelData",
            "ContDIDSpec",
            "ContDIDResult",
            "estimate_dose_effects",
            "estimate_eventstudy_effects",
            "load_medicare_pps_example_panel",
            "Supported routes and boundaries",
            "python3 manuscript/replication/run_all.py",
        },
        "public_api_help_required_terms": {
            "PanelData",
            "ContDIDSpec",
            "ContDIDResult",
            "estimate_dose_effects",
            "estimate_eventstudy_effects",
            "estimate_eventstudy_slope_effects",
            "balanced long panel",
            "cohort_summary",
            "ContDIDValidationError",
        },
        "user_guide_required_terms": {
            "ContDIDResult.to_frame()",
            "ContDIDResult.to_markdown()",
            "balanced long panels",
            "repeated cross-sections",
            "time-varying doses",
            "covariate-aware identification",
            "Medicare PPS",
        },
        "medicare_example_help_required_terms": {
            "constructed hospital-year records",
            "two-period dose panel",
            "annual event-study panel",
            "ContDIDResult",
            "load_medicare_pps_example_panel",
            "prepare_medicare_pps_panel",
            "estimate_eventstudy_effects",
            "licensed AHA 1980-1986 hospital panel",
        },
    }
    for key, terms in expected_reader_terms.items():
        assert set(docs_bundle[key]) >= terms
        assert not set(docs_bundle[key]).intersection(docs_bundle["public_doc_forbidden_terms"])


def test_phase9_public_api_contract_consumes_release_example_manifests_and_english_doc_rules() -> (
    None
):
    contract = _load_json(CONTRACT_PATH)
    release_examples = _load_json(PHASE9_RELEASE_EXAMPLES_PATH)
    medicare_release_packet = _load_json(MEDICARE_RELEASE_PACKET_PATH)
    docs_bundle = contract["docs_bundle"]

    example_catalog = {
        entry["example_id"]: entry for entry in release_examples["example_catalog"]
    }
    release_examples_contract = contract["release_examples"]

    assert release_examples_contract["synthetic-dose-demo"] == {
        "source_manifest_entry": "synthetic-dose-demo",
        "source_surface": "simulate_contdid_data",
        "consumer_surface": "estimate_dose_effects",
        "release_surfaces": example_catalog["synthetic-dose-demo"]["release_surfaces"],
        "estimand_family": "ATT(d)/ACRT(d)",
        "dose_control_group": "nevertreated",
        "seed_rule": (
            "simulate_contdid_data seed inputs must be None or nonnegative integers; "
            "None uses the DGP seed registry and booleans, strings, negative values, "
            "and non-integers raise seed must be a nonnegative integer"
        ),
    }
    assert release_examples_contract["medicare-scaffold-demo"] == {
        "source_manifest_entry": "medicare-scaffold-demo",
        "source_surface": "prepare_medicare_pps_panel",
        "consumer_surfaces": {
            "att": "estimate_dose_effects",
            "acrt": "estimate_dose_slope_effects",
            "eventstudy": "estimate_eventstudy_effects",
        },
        "release_surfaces": example_catalog["medicare-scaffold-demo"][
            "release_surfaces"
        ],
        "honesty_label": "descriptive-or-scaffold-only",
    }

    assert contract["english_release_docs"] == {
        "language": "English",
        "required_consumers": medicare_release_packet["release_consumers"],
        "required_honesty_labels": ["descriptive-or-scaffold-only"],
        "forbidden_claims": [
            "licensed Medicare PPS parity",
            "unchecked Monte Carlo baseline drift",
            "undocumented private metadata as public API",
        ],
    }
    assert (
        "checked_eventstudy_confidence_interval"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_confidence_band"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_alp"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_biters"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_bootstrap_seed"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_cband"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_bstrap"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_summary"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_dose_grid"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert (
        "checked_eventstudy_dose_est_method"
        in docs_bundle["medicare_walkthrough_checked_json_fields"]
    )
    assert docs_bundle["medicare_walkthrough_release_surface_fields"] == [
        "application_id",
        "manifest_path",
        "snapshot_path",
        "public_api_contract",
        "run_mode",
        "source_label",
        "release_consumers",
        "source_surface",
        "package_surfaces",
        "required_years",
        "baseline_year",
        "published_targets",
        "published_targets_scope",
        "eventstudy_input_contract",
        "parity_claim_allowed",
        "parity_viability",
        "allowed_use",
    ]
    assert set(docs_bundle["medicare_example_help_required_terms"]) >= {
        "constructed hospital-year records",
        "two-period dose panel",
        "annual event-study panel",
        "ContDIDResult",
        "prepare_medicare_pps_panel",
        "estimate_eventstudy_effects",
        "licensed AHA 1980-1986 hospital panel",
    }
    assert not set(docs_bundle["medicare_example_help_required_terms"]).intersection(
        docs_bundle["public_doc_forbidden_terms"]
    )
    assert docs_bundle["dose_curve_checked_json_fields"] == [
        "example_id",
        "source_kind",
        "source_surface",
        "package_surface",
        "public_api_contract",
        "manifest_path",
        "release_surfaces",
        "estimand",
        "grid",
        "estimate",
        "std_error",
        "critical_value",
        "confidence_interval",
        "confidence_band",
        "render_profile",
        "metadata",
    ]
    assert docs_bundle["dose_curve_render_profile_fields"] == [
        "width",
        "height",
        "x_axis_label",
        "y_axis_label",
        "confidence_band_label",
        "estimate_label",
        "point_marker_stride",
        "palette",
        "legend_box",
    ]
    assert docs_bundle["dose_curve_palette_fields"] == [
        "background",
        "panel",
        "grid",
        "axis",
        "text",
        "muted_text",
        "estimate_line",
        "estimate_point",
        "confidence_band",
        "confidence_band_outline",
        "zero_line",
    ]
    assert docs_bundle["dose_curve_legend_box_fields"] == [
        "left",
        "top",
        "width",
        "height",
    ]
    assert docs_bundle["dose_curve_release_surface_fields"] == [
        "example_id",
        "source_kind",
        "source_surface",
        "package_surface",
        "public_api_contract",
        "manifest_path",
        "release_surfaces",
    ]
    machine_only_release_fields = [
        field
        for field in docs_bundle["dose_curve_release_surface_fields"]
        if field != "public_api_contract"
    ]
    for field in machine_only_release_fields:
        assert field not in docs_bundle["readme_required_terms"]
        assert field not in docs_bundle["public_api_help_required_terms"]


def test_phase9_runtime_contract_validator_requires_medicare_published_target_scope() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        validate_public_api_contract_bundle,
    )

    contract = _load_json(CONTRACT_PATH)
    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["medicare_walkthrough_release_surface_fields"].remove(
        "published_targets_scope"
    )

    with pytest.raises(
        PublicAPIContractError,
        match="medicare_walkthrough_release_surface_fields.*published-target scope",
    ):
        validate_public_api_contract_bundle(corrupted)

    corrupted = copy.deepcopy(contract)
    corrupted["docs_bundle"]["medicare_example_help_required_terms"].remove(
        "licensed AHA 1980-1986 hospital panel"
    )

    with pytest.raises(
        PublicAPIContractError,
        match="medicare_example_help_required_terms.*reader-facing documentation tasks",
    ):
        validate_public_api_contract_bundle(corrupted)


def test_phase9_public_api_contract_loader_requires_eventstudy_grid_support_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["traceability"]["eventstudy_input_contract"][
        "eventstudy_evaluation_grid_support_rule"
    ]
    assert rule == (
        "event-study evaluation grids must be drawn from observed-window "
        "positive-dose treated cohorts that survive admissible base-period "
        "and local inference-degree-of-freedom filtering; "
        "first-period/no-baseline cohorts, after-window comparison-only "
        "not-yet-treated doses, and locally unidentified cohorts cannot define "
        "default quantiles or satisfy explicit dvals support"
    )

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["traceability"]["eventstudy_input_contract"][
        "eventstudy_evaluation_grid_support_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "eventstudy_evaluation_grid_support_rule" in str(exc)
    else:
        raise AssertionError("missing event-study grid support rule should fail")


def test_phase9_public_api_contract_loader_requires_eventstudy_base_period_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["traceability"]["eventstudy_input_contract"][
        "eventstudy_base_period_rule"
    ]
    assert rule == (
        "base_period accepts None/'varying' for adjacent pre-period comparisons, "
        "'universal' for cohort-specific g-anticipation-1 comparisons with the "
        "normalized reference cell omitted, or an observed integer baseline for "
        "fixed-baseline event-study comparisons; fixed and universal comparisons "
        "support control_group values 'notyettreated' and 'nevertreated'; timing "
        "groups whose treatment timing is not preceded by the selected baseline "
        "are omitted before event-time support is built, and the route hard-fails "
        "if no treated cohort has an admissible base period"
    )

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["traceability"]["eventstudy_input_contract"][
        "eventstudy_base_period_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "eventstudy_base_period_rule" in str(exc)
    else:
        raise AssertionError("missing event-study base-period rule should fail")


def test_phase9_public_api_contract_loader_requires_eventstudy_post_treatment_identification_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["traceability"]["eventstudy_input_contract"][
        "post_treatment_identification_rule"
    ]
    assert rule == (
        "requires at least one locally identified post-treatment event time "
        "with positive-dose support and inference degrees of freedom"
    )

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["traceability"]["eventstudy_input_contract"][
        "post_treatment_identification_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "post_treatment_identification_rule" in str(exc)
    else:
        raise AssertionError(
            "missing event-study post-treatment identification rule should fail"
        )


def test_phase9_public_api_contract_loader_requires_cck_panel_shape_guard_precedence_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["traceability"]["eventstudy_input_contract"][
        "cck_panel_shape_guard_precedence_rule"
    ]
    assert "staggered adoption must raise" in rule
    assert "single-cohort multi-period panels must raise" in rule
    assert "post period" in rule

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["traceability"]["eventstudy_input_contract"][
        "cck_panel_shape_guard_precedence_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "CCK panel-shape guard precedence rule" in str(exc)
    else:
        raise AssertionError(
            "missing CCK panel-shape guard precedence rule should fail"
        )


def test_phase9_public_api_contract_loader_requires_cck_executable_runtime_nodes() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    nodes = bundle["traceability"]["eventstudy_input_contract"][
        "cck_boundary_minimum_runtime_test_nodes"
    ]
    assert nodes == [
        "contdid-py/tests/test_phase6_cck_guards.py::test_cck_rejects_staggered_adoption_with_exact_manifest_substring",
        "contdid-py/tests/test_phase6_cck_guards.py::test_cck_staggered_adoption_guard_precedes_multi_period_guard",
        "contdid-py/tests/test_phase6_cck_guards.py::test_cck_dose_staggered_guard_precedes_unchecked_control_group",
        "contdid-py/tests/test_phase6_cck_guards.py::test_cck_requires_exactly_two_time_periods",
        "contdid-py/tests/test_phase6_cck_guards.py::test_cck_requires_positive_timing_in_post_period_with_cck_error",
        "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_path_stays_hard_failed",
        "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_post_timing_guard_precedes_eventstudy_guard",
        "contdid-py/tests/test_phase6_cck_guards.py::test_eventstudy_cck_staggered_guard_precedes_multi_period_and_control_group",
    ]

    broken_bundle = json.loads(json.dumps(bundle))
    broken_bundle["traceability"]["eventstudy_input_contract"][
        "cck_boundary_minimum_runtime_test_nodes"
    ] = [
        "contdid-py/tests/test_phase5_eventstudy.py::test_eventstudy_cck_path_stays_hard_failed"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "CCK executable runtime test nodes" in str(exc)
    else:
        raise AssertionError("stale CCK runtime test nodes should fail")


def test_phase9_public_api_contract_loader_requires_confidence_band_shape_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["result_payload"]["confidence_band_shape_contract"][
        "shape_match_rule"
    ]
    assert rule == "estimate and std_error must have the same shape"

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["result_payload"]["confidence_band_shape_contract"][
        "shape_match_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "confidence_band_shape_contract" in str(exc)
    else:
        raise AssertionError("missing confidence-band shape rule should fail")


def test_phase9_public_api_contract_loader_requires_confidence_band_nonempty_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["result_payload"]["confidence_band_shape_contract"]["nonempty_rule"]
    assert rule == "estimate and std_error must contain at least one value"

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["result_payload"]["confidence_band_shape_contract"][
        "nonempty_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "confidence_band_shape_contract" in str(exc)
    else:
        raise AssertionError("missing confidence-band nonempty rule should fail")


def test_phase9_public_api_contract_loader_requires_multiplier_covariance_psd_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["result_payload"]["multiplier_bootstrap_covariance_contract"][
        "covariance_psd_rule"
    ]
    assert rule == "covariance matrix must be positive semidefinite"

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["result_payload"]["multiplier_bootstrap_covariance_contract"][
        "covariance_psd_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "multiplier_bootstrap_covariance_contract" in str(exc)
    else:
        raise AssertionError("missing multiplier covariance PSD rule should fail")


def test_phase9_public_api_contract_loader_requires_multiplier_nonempty_rules() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    contract = bundle["result_payload"]["multiplier_bootstrap_covariance_contract"]
    assert (
        contract["loadings_nonempty_rule"]
        == "loadings must contain at least one estimand row"
    )
    assert (
        contract["result_nonempty_rule"]
        == "result estimates must contain at least one value"
    )

    for key in ("loadings_nonempty_rule", "result_nonempty_rule"):
        broken_bundle = json.loads(json.dumps(bundle))
        del broken_bundle["result_payload"]["multiplier_bootstrap_covariance_contract"][
            key
        ]

        try:
            validate_public_api_contract_bundle(broken_bundle)
        except PublicAPIContractError as exc:
            assert "multiplier_bootstrap_covariance_contract" in str(exc)
        else:
            raise AssertionError(f"missing multiplier {key} should fail")


def test_phase9_public_api_contract_loader_requires_multiplier_seed_rule() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["result_payload"]["multiplier_bootstrap_covariance_contract"][
        "seed_rule"
    ]
    assert rule == "seed must be a nonnegative integer"

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["result_payload"]["multiplier_bootstrap_covariance_contract"][
        "seed_rule"
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "multiplier_bootstrap_covariance_contract" in str(exc)
    else:
        raise AssertionError("missing multiplier bootstrap seed rule should fail")


def test_phase9_public_api_contract_loader_requires_simulation_seed_rule() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["release_examples"]["synthetic-dose-demo"]["seed_rule"]
    assert rule == (
        "simulate_contdid_data seed inputs must be None or nonnegative integers; "
        "None uses the DGP seed registry and booleans, strings, negative values, "
        "and non-integers raise seed must be a nonnegative integer"
    )

    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["release_examples"]["synthetic-dose-demo"]["seed_rule"]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "synthetic-dose-demo.seed_rule" in str(exc)
    else:
        raise AssertionError("missing simulate_contdid_data seed rule should fail")


def test_phase9_public_api_contract_loader_requires_eventstudy_local_support_rule() -> (
    None
):
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    rule = bundle["result_payload"]["eventstudy_summary_aggregates"][
        "local_support_rule"
    ]
    assert rule == (
        "ATT(event_time) uses binary treated/control mean support; "
        "ACRT(event_time) uses positive-dose derivative-basis support with local "
        "inference degrees of freedom"
    )

    broken_bundle = json.loads(json.dumps(bundle))
    broken_bundle["result_payload"]["eventstudy_summary_aggregates"][
        "local_support_rule"
    ] = "missing"

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "eventstudy_summary_aggregates" in str(exc)
    else:
        raise AssertionError("missing event-study local support rule should fail")


def test_phase9_public_api_contract_loader_rejects_duplicate_exports() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    broken_bundle = json.loads(json.dumps(bundle))
    broken_bundle["stable_top_level_exports"] = [
        *broken_bundle["stable_top_level_exports"],
        "ContDIDSpec",
    ]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "stable_top_level_exports must not contain duplicates" in str(exc)
    else:
        raise AssertionError("duplicate public API exports should fail")


def test_phase9_public_api_contract_loader_requires_docs_bundle_files_section() -> None:
    from contdid.contracts import (
        PublicAPIContractError,
        load_public_api_contract_bundle,
    )
    from contdid.contracts.phase9 import validate_public_api_contract_bundle

    bundle = load_public_api_contract_bundle()
    broken_bundle = json.loads(json.dumps(bundle))
    del broken_bundle["docs_bundle"]["files"]

    try:
        validate_public_api_contract_bundle(broken_bundle)
    except PublicAPIContractError as exc:
        assert "docs_bundle.files" in str(exc)
    else:
        raise AssertionError("missing docs bundle files section should fail")


def test_phase9_public_api_contract_docs_terms_are_contiguous_for_eventstudy_grid_support() -> (
    None
):
    contract = _load_json(CONTRACT_PATH)
    docs_bundle = contract["docs_bundle"]
    readme_text = (REPO_ROOT / docs_bundle["files"]["readme"]).read_text(
        encoding="utf-8"
    )
    public_api_help_text = (
        REPO_ROOT / docs_bundle["files"]["public_api_help"]
    ).read_text(encoding="utf-8")
    user_guide_text = (REPO_ROOT / docs_bundle["files"]["user_guide"]).read_text(
        encoding="utf-8"
    )

    support_rule = contract["traceability"]["eventstudy_input_contract"][
        "eventstudy_evaluation_grid_support_rule"
    ]
    for marker in [
        "observed-window positive-dose treated cohorts",
        "admissible base-period",
        "local inference-degree-of-freedom filtering",
    ]:
        assert marker in support_rule

    assert "positive treated-dose support" in readme_text
    assert "support flags" in readme_text
    assert "event_time" in readme_text
    assert "support" in docs_bundle["readme_required_terms"]
    assert "event_time" in docs_bundle["public_api_help_required_terms"]
    assert "event_time_covariance" in docs_bundle["public_api_help_required_terms"]
    assert "event_time" in public_api_help_text
    assert "support" in public_api_help_text
    assert "event_time_covariance" in public_api_help_text

    user_guide_terms = [
        "cohort_summary",
        "aggregation_weight",
        "weighted cohort estimates",
    ]
    for term in user_guide_terms:
        assert term in user_guide_text


def test_phase9_eventstudy_default_grid_excludes_after_window_notyettreated_doses() -> (
    None
):
    from contdid import ContDIDSpec, PanelData, estimate_eventstudy_effects
    from contdid.validation import ContDIDValidationError

    rows = []
    for unit_id, group, dose in [
        (1, 0, 0.0),
        (2, 0, 0.0),
        (3, 3, 1.0),
        (4, 3, 1.2),
        (5, 5, 9.0),
        (6, 5, 10.0),
    ]:
        for time_period in [1, 2, 3, 4]:
            rows.append(
                {
                    "id": unit_id,
                    "time_period": time_period,
                    "G": group,
                    "D": dose,
                    "Y": (
                        0.1 * unit_id
                        + 0.05 * time_period
                        + 0.2 * (time_period >= group > 0)
                    ),
                }
            )
    panel = PanelData(frame=pd.DataFrame.from_records(rows))
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group="notyettreated",
        bstrap=False,
        cband=False,
    )

    result = estimate_eventstudy_effects(panel, spec, degree=1)
    assert max(result.metadata["dose_grid"]) <= 1.2

    try:
        estimate_eventstudy_effects(panel, spec, dvals=[9.0], degree=1)
    except ContDIDValidationError as exc:
        assert "observed positive-dose treated support [1.0, 1.2]" in str(exc)
    else:
        raise AssertionError("after-window not-yet-treated dvals should fail")
