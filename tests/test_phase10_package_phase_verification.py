from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import get_type_hints

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase10"
    / "package_phase_verification_manifest.json"
)
RUNNER_PATH = REPO_ROOT / "contdid-py" / "scripts" / "run_package_phase_verification.py"
CHECKED_SUMMARY_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase10"
    / "runs"
    / "package_phase_verification_summary.json"
)
CHECKED_INVENTORY_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase10"
    / "runs"
    / "package_phase_verification_inventory.md"
)
CHECKED_ROLLUP_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase10"
    / "runs"
    / "package_phase_verification_rollup.json"
)
CHECKED_PHASE_VERIFICATION_DOCS = {
    3: REPO_ROOT / ".planning" / "phases" / "03-contdid-py" / "03-VERIFICATION.md",
    7: REPO_ROOT
    / ".planning"
    / "phases"
    / "07-paper-to-python-e2e"
    / "07-VERIFICATION.md",
    8: REPO_ROOT / ".planning" / "phases" / "08-monte-carlo" / "08-VERIFICATION.md",
    9: REPO_ROOT
    / ".planning"
    / "phases"
    / "09-pythonic-public-api-plotting-examples-english-release-docs"
    / "09-VERIFICATION.md",
}
EXPECTED_RELEASE_GATE_TRUTH_REFRESH_COMMAND = (
    "python3 contdid-py/scripts/run_package_phase_verification.py "
    "--refresh-release-gate-truth-snapshot-only "
    "--output contdid-py/contracts/phase10/runs/package_phase_verification_summary.json "
    "--inventory-output contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md "
    "--rollup-output contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json"
)
EXPECTED_LANE49_TIMEOUT_SAFE_VERIFIER_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=contdid-py/src "
    "python3 automation/scripts/run_lane49_packaging_release_verifier.py "
    "--package-bundle-timeout-seconds 300 --pytest-timeout-seconds 240"
)


def _expected_focused_bundle_rerun_command(
    bundle_id: str,
    *,
    bundle_timeout_seconds: float | None = None,
) -> str:
    runs_dir = "contdid-py/contracts/phase10/runs"
    stem = f"package_phase_verification_{bundle_id}"
    timeout_arg = (
        f"--bundle-timeout-seconds {bundle_timeout_seconds} "
        if bundle_timeout_seconds is not None
        else ""
    )
    return (
        "python3 contdid-py/scripts/run_package_phase_verification.py "
        f"--bundle-id {bundle_id} "
        f"{timeout_arg}"
        f"--output {runs_dir}/{stem}_summary.json "
        f"--inventory-output {runs_dir}/{stem}_inventory.md "
        f"--rollup-output {runs_dir}/{stem}_rollup.json "
        "--verification-doc-root ."
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_output_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _assert_iso_utc_timestamp(value: object) -> None:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def _bundle_by_id(bundle_id: str) -> dict:
    manifest = _load_json(MANIFEST_PATH)
    return next(
        bundle
        for bundle in manifest["phase_bundles"]
        if bundle["bundle_id"] == bundle_id
    )


def _load_runner_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase10_package_truth_snapshot_ignores_stale_blocker_when_summary_is_green(
    tmp_path: Path,
) -> None:
    module = _load_runner_module(
        "phase10_package_phase_verification_stale_green_blocker"
    )
    summary_path = tmp_path / "checked_summary.json"
    blocker_path = tmp_path / "checked_blocker_packet.json"
    summary_path.write_text(
        json.dumps(
            {
                "completed_successfully": True,
                "live_frontier": {
                    "source": "checked-summary",
                    "path": "checked_summary.json",
                    "exists": True,
                    "owner_lane": None,
                    "owner_ready_label": "green checked root",
                    "failing_label": None,
                    "next_command": "maintenance refresh only",
                    "exit_criteria": "already green",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    blocker_path.write_text(
        json.dumps(
            {
                "owner_lane": "contdid-gsd-qa-mc-48",
                "failing_gate_id": "numerical-audit",
                "failing_label": "stale numerical blocker",
                "failing_test_nodeid": "tests/test_stale.py::test_stale",
                "next_command": "stale rerun command",
                "exact_next_predicate": "stale predicate",
                "route_status": "blocked",
                "why_not_done_now": "stale packet left on disk",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = module._summary_truth_snapshot(summary_path, blocker_path)

    assert snapshot["summary_completed_successfully"] is True
    assert snapshot["blocker_packet_path"] == str(blocker_path)
    assert snapshot["blocker_packet_exists"] is False
    assert snapshot["blocker_owner_lane"] is None
    assert snapshot["blocker_failing_gate_id"] is None
    assert snapshot["blocker_failing_label"] is None
    assert snapshot["blocker_failing_test_nodeid"] is None
    assert snapshot["blocker_next_command"] is None
    assert snapshot["blocker_exact_next_predicate"] is None
    assert snapshot["blocker_route_status"] is None
    assert snapshot["blocker_why_not_done_now"] is None


def _expected_release_gate_truth_snapshot() -> dict:
    release_gate_summary = _load_json(
        REPO_ROOT / "reproduction" / "phase10_release_gate" / "runs" / "release-gate_summary.json"
    )
    all_gates_summary = _load_json(
        REPO_ROOT / "reproduction" / "phase10_release_gate" / "runs" / "all_gates_summary.json"
    )
    qa_summary = _load_json(
        REPO_ROOT
        / "tests"
        / "contracts"
        / "phase10"
        / "runs"
        / "qa_phase_verification_summary.json"
    )
    release_gate_blocker_path = (
        REPO_ROOT
        / "reproduction"
        / "phase10_release_gate"
        / "runs"
        / "release-gate_blocker_packet.json"
    )
    all_gates_blocker_path = (
        REPO_ROOT
        / "reproduction"
        / "phase10_release_gate"
        / "runs"
        / "all_gates_blocker_packet.json"
    )
    qa_blocker_path = (
        REPO_ROOT
        / "tests"
        / "contracts"
        / "phase10"
        / "runs"
        / "qa_phase_verification_blocker_packet.json"
    )
    release_gate_blocker = (
        _load_json(release_gate_blocker_path)
        if release_gate_blocker_path.exists()
        else None
    )
    all_gates_blocker = (
        _load_json(all_gates_blocker_path) if all_gates_blocker_path.exists() else None
    )
    qa_blocker = _load_json(qa_blocker_path) if qa_blocker_path.exists() else None
    if qa_summary["completed_successfully"] is True:
        qa_blocker = None
    release_gate_live_frontier = release_gate_summary.get("live_frontier", {})
    all_gates_live_frontier = all_gates_summary.get("live_frontier", {})
    return {
        "manifest_path": "reproduction/phase10_release_gate/manifest.json",
        "manifest_exists": True,
        "manifest_sha256": _sha256(
            REPO_ROOT / "reproduction" / "phase10_release_gate" / "manifest.json"
        ),
        "checked_release_gate_summary_path": "reproduction/phase10_release_gate/runs/release-gate_summary.json",
        "checked_release_gate_summary_exists": True,
        "checked_release_gate_summary_completed_successfully": release_gate_summary[
            "completed_successfully"
        ],
        "checked_release_gate_blocker_packet_path": "reproduction/phase10_release_gate/runs/release-gate_blocker_packet.json",
        "checked_release_gate_blocker_packet_exists": release_gate_blocker is not None,
        "checked_release_gate_live_frontier_source": release_gate_live_frontier.get(
            "source"
        ),
        "checked_release_gate_live_frontier_path": release_gate_live_frontier.get(
            "path"
        ),
        "checked_release_gate_live_frontier_exists": release_gate_live_frontier.get(
            "exists"
        ),
        "checked_release_gate_live_frontier_owner_lane": release_gate_live_frontier.get(
            "owner_lane"
        ),
        "checked_release_gate_live_frontier_owner_ready_label": release_gate_live_frontier.get(
            "owner_ready_label"
        ),
        "checked_release_gate_live_frontier_failing_label": release_gate_live_frontier.get(
            "failing_label"
        ),
        "checked_release_gate_live_frontier_next_command": release_gate_live_frontier.get(
            "next_command"
        ),
        "checked_release_gate_live_frontier_exit_criteria": release_gate_live_frontier.get(
            "exit_criteria"
        ),
        "checked_release_gate_blocker_route_status": (
            None if release_gate_blocker is None else release_gate_blocker.get("route_status")
        ),
        "checked_release_gate_blocker_why_not_done_now": (
            None
            if release_gate_blocker is None
            else release_gate_blocker.get("why_not_done_now")
        ),
        "checked_release_gate_blocker_exact_next_predicate": (
            None
            if release_gate_blocker is None
            else release_gate_blocker.get("exact_next_predicate")
        ),
        "checked_all_gates_summary_path": "reproduction/phase10_release_gate/runs/all_gates_summary.json",
        "checked_all_gates_summary_exists": True,
        "checked_all_gates_summary_completed_successfully": all_gates_summary[
            "completed_successfully"
        ],
        "checked_all_gates_summary_generated_at": all_gates_summary.get("generated_at"),
        "checked_all_gates_blocker_packet_path": "reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json",
        "checked_all_gates_blocker_packet_exists": all_gates_blocker is not None,
        "checked_all_gates_blocker_owner_lane": (
            None if all_gates_blocker is None else all_gates_blocker["owner_lane"]
        ),
        "checked_all_gates_blocker_failing_gate_id": (
            None if all_gates_blocker is None else all_gates_blocker["failing_gate_id"]
        ),
        "checked_all_gates_blocker_failing_label": (
            None if all_gates_blocker is None else all_gates_blocker["failing_label"]
        ),
        "checked_all_gates_blocker_failing_test_nodeid": (
            None
            if all_gates_blocker is None
            else all_gates_blocker["failing_test_nodeid"]
        ),
        "checked_all_gates_blocker_next_command": (
            None if all_gates_blocker is None else all_gates_blocker["next_command"]
        ),
        "checked_all_gates_live_frontier_source": all_gates_live_frontier.get(
            "source"
        ),
        "checked_all_gates_live_frontier_path": all_gates_live_frontier.get("path"),
        "checked_all_gates_live_frontier_exists": all_gates_live_frontier.get(
            "exists"
        ),
        "checked_all_gates_live_frontier_owner_lane": all_gates_live_frontier.get(
            "owner_lane"
        ),
        "checked_all_gates_live_frontier_owner_ready_label": all_gates_live_frontier.get(
            "owner_ready_label"
        ),
        "checked_all_gates_live_frontier_failing_label": all_gates_live_frontier.get(
            "failing_label"
        ),
        "checked_all_gates_live_frontier_next_command": all_gates_live_frontier.get(
            "next_command"
        ),
        "checked_all_gates_live_frontier_exit_criteria": all_gates_live_frontier.get(
            "exit_criteria"
        ),
        "checked_all_gates_blocker_route_status": (
            None if all_gates_blocker is None else all_gates_blocker.get("route_status")
        ),
        "checked_all_gates_blocker_why_not_done_now": (
            None if all_gates_blocker is None else all_gates_blocker.get("why_not_done_now")
        ),
        "checked_all_gates_blocker_exact_next_predicate": (
            None
            if all_gates_blocker is None
            else all_gates_blocker.get("exact_next_predicate")
        ),
        "maintenance_refresh_command": all_gates_summary.get(
            "maintenance_refresh_command"
        ),
        "maintenance_refresh_reason": all_gates_summary.get(
            "maintenance_refresh_reason"
        ),
        "archival_next_command": all_gates_summary.get("archival_next_command"),
        "release_evidence_snapshot": all_gates_summary.get("release_evidence_snapshot"),
        "checked_qa_summary_path": "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
        "checked_qa_summary_exists": True,
        "checked_qa_summary_completed_successfully": qa_summary[
            "completed_successfully"
        ],
        "checked_qa_blocker_packet_path": "tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json",
        "checked_qa_blocker_packet_exists": qa_blocker is not None,
        "checked_qa_blocker_owner_lane": (
            None if qa_blocker is None else qa_blocker["owner_lane"]
        ),
        "checked_qa_blocker_owner_ready_label": (
            None if qa_blocker is None else qa_blocker.get("owner_ready_label")
        ),
        "checked_qa_blocker_failing_label": (
            None if qa_blocker is None else qa_blocker["failing_label"]
        ),
        "checked_qa_blocker_failing_test_nodeid": (
            None if qa_blocker is None else qa_blocker.get("failing_test_nodeid")
        ),
        "checked_qa_blocker_next_command": (
            None if qa_blocker is None else qa_blocker["next_command"]
        ),
    }


def _expected_source_truth_snapshot(
    selected_bundle_ids: list[str] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    manifest = _load_json(MANIFEST_PATH)
    bundle_lookup = {
        bundle["bundle_id"]: bundle for bundle in manifest["phase_bundles"]
    }
    bundle_ids = (
        selected_bundle_ids
        if selected_bundle_ids is not None
        else [bundle["bundle_id"] for bundle in manifest["phase_bundles"]]
    )
    return {
        "source_contracts": {
            label: {
                "path": relative_path,
                "exists": (REPO_ROOT / relative_path).exists(),
                "sha256": _sha256(REPO_ROOT / relative_path),
            }
            for label, relative_path in manifest["source_contracts"].items()
        },
        "bundle_evidence": {
            bundle_id: {
                relative_path: {
                    "path": relative_path,
                    "exists": (REPO_ROOT / relative_path).exists(),
                    "sha256": _sha256(REPO_ROOT / relative_path),
                }
                for relative_path in bundle_lookup[bundle_id]["evidence_paths"]
            }
            for bundle_id in bundle_ids
        },
    }


def _expected_maintenance_commands() -> list[dict[str, str]]:
    return [
        {
            "command_id": "refresh-release-gate-truth-snapshot",
            "purpose": (
                "Refresh the checked release-gate truth snapshot, then cascade the "
                "downstream checked QA/theory handoffs, without rerunning package "
                "bundle tests."
            ),
            "command": EXPECTED_RELEASE_GATE_TRUTH_REFRESH_COMMAND,
        },
        {
            "command_id": "lane49-timeout-safe-packaging-release-verifier",
            "purpose": (
                "Run the default self-cleaning lane49 packaging/release verifier "
                "route, so the checked package packet, the Phase 7-10 "
                "verification docs, the package layout/offline build-backend "
                "contract, the release generated-artifact hygiene gate, "
                "the budgeted Phase 9 release-example contract smoke, "
                "the budgeted installed-wheel frontier helper smoke, the installed-"
                "wheel public runtime asset/API smoke, the installed-wheel public "
                "API contract loader smoke, the installed-wheel Phase 2 truth "
                "contract loader smoke, the installed-wheel public event-study "
                "basis validation smoke, the installed-wheel CCK unsupported-boundary "
                "smoke, the installed-sdist CCK unsupported-boundary smoke, the installed-sdist "
                "public runtime asset/API smoke, the installed-sdist public API contract loader "
                "smoke, the pip editable-install CCK unsupported-boundary smoke, the pip "
                "editable-install public runtime asset/API smoke, the pip editable-install "
                "public API contract loader smoke, and the package-local release frontier "
                "docs rerun against live disk facts while downstream QA/theory/v1 "
                "checked packet refreshes remain explicit handoffs."
            ),
            "command": EXPECTED_LANE49_TIMEOUT_SAFE_VERIFIER_COMMAND,
        },
    ]


def _expected_phase10_frontier_runtime_helper_test_paths() -> list[str]:
    return [
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py",
        "tests/test_release_generated_artifact_hygiene.py",
        "tests/test_phase10_frontier_expectations_import.py",
        "tests/test_phase10_qa_frontier_expectations.py",
    ]


def _expected_phase10_frontier_runtime_helper_execution_test_paths() -> list[str]:
    return [
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_phase10_frontier_runtime_helper_supports_lane49_checked_qa_blocker_from_installed_wheel",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_runtime_contract_assets_are_available_from_installed_wheel",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_api_contract_bundle_is_available_from_installed_wheel",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_phase2_contract_bundle_is_available_from_installed_wheel",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_installed_wheel_eventstudy_routes_reject_invalid_basis_controls",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_installed_wheel_cck_routes_keep_unsupported_boundaries",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_installed_sdist_cck_routes_keep_unsupported_boundaries",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_runtime_contract_assets_are_available_from_installed_sdist",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_api_contract_bundle_is_available_from_installed_sdist",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_editable_install_cck_routes_keep_unsupported_boundaries",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_runtime_contract_assets_are_available_from_editable_install",
        "contdid-py/tests/test_phase10_frontier_runtime_helper.py::test_public_api_contract_bundle_is_available_from_editable_install",
        "tests/test_release_generated_artifact_hygiene.py",
        "tests/test_phase10_frontier_expectations_import.py",
        "tests/test_phase10_qa_frontier_expectations.py",
    ]


def _expected_phase10_frontier_runtime_helper_evidence_paths() -> list[str]:
    return [
        ".gitignore",
        "phase10_frontier_expectations.py",
        "contdid-py/src/contdid/_phase10_frontier_root_logic.py",
        "contdid-py/src/phase10_frontier_expectations.py",
        "contdid-py/pyproject.toml",
        "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
        "contdid-py/contracts/phase2/paper_truth_contract.json",
        "contdid-py/contracts/phase2/phase2_contract_template.json",
        "contdid-py/contracts/phase2/symbol_map_contract.json",
        "contdid-py/contracts/phase9/public_api_contract_v1.json",
        "reproduction/simulate_contdid/manifest.json",
        "reproduction/medicare_pps/manifest.json",
        "reproduction/medicare_pps/source_options.json",
    ]


def _expected_external_requirement_results() -> list[dict[str, object]]:
    return [
        {**source, "status": "passed"}
        for source in _load_json(MANIFEST_PATH)["external_requirement_sources"]
    ]


def _pending_truth_inputs(required_truth_inputs: list[str]) -> list[str]:
    pending: list[str] = []
    for relative_path in required_truth_inputs:
        path = REPO_ROOT / relative_path
        if not path.exists():
            pending.append(relative_path)
            continue
        if path.suffix != ".json":
            continue
        payload = _load_json(path)
        if (
            isinstance(payload, dict)
            and "completed_successfully" in payload
            and payload["completed_successfully"] is not True
        ):
            pending.append(relative_path)
    return pending


def _expected_downstream_handoff_statuses() -> list[dict[str, object]]:
    manifest = _load_json(MANIFEST_PATH)
    release_gate_truth_snapshot = _expected_release_gate_truth_snapshot()
    blocker_owner_lane = release_gate_truth_snapshot[
        "checked_all_gates_blocker_owner_lane"
    ]
    handoff_by_owner = {
        handoff["owner_lane"]: handoff["handoff_id"]
        for handoff in manifest["downstream_handoffs"]
    }
    statuses: list[dict[str, object]] = []
    for handoff in manifest["downstream_handoffs"]:
        pending_truth_inputs = _pending_truth_inputs(handoff["required_truth_inputs"])
        if not pending_truth_inputs:
            status = "satisfied"
            blocking_owner = None
            blocking_handoff_id = None
            next_command = None
            status_reason = "All required truth inputs are already present and green."
        elif blocker_owner_lane and blocker_owner_lane != handoff["owner_lane"]:
            status = "blocked"
            blocking_owner = blocker_owner_lane
            blocking_handoff_id = handoff_by_owner.get(blocker_owner_lane)
            next_command = release_gate_truth_snapshot[
                "checked_all_gates_blocker_next_command"
            ]
            status_reason = (
                "The live checked-root blocker is owned by another lane, so this "
                "handoff cannot proceed yet."
            )
        else:
            status = "ready_now"
            blocking_owner = None
            blocking_handoff_id = None
            next_command = handoff["next_command"]
            status_reason = (
                "This handoff owns the remaining package-side frontier and can "
                "consume the refreshed packet immediately."
            )
        statuses.append(
            {
                "handoff_id": handoff["handoff_id"],
                "owner_lane": handoff["owner_lane"],
                "status": status,
                "pending_truth_inputs": pending_truth_inputs,
                "blocking_owner_lane": blocking_owner,
                "blocking_handoff_id": blocking_handoff_id,
                "next_command": next_command,
                "status_reason": status_reason,
            }
        )
    return statuses


def test_phase10_package_phase_verification_manifest_exists_and_freezes_bundle_inputs() -> (
    None
):
    assert MANIFEST_PATH.exists(), (
        f"missing package verification manifest: {MANIFEST_PATH}"
    )
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["schema_version"] == "0.1"
    assert manifest["report_type"] == "package_phase_verification"
    assert manifest["package_root"] == "contdid-py"
    assert manifest["owner_lane"] == "contdid-gsd-audit-repair-49"
    assert manifest["checked_summary"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json"
    )
    assert manifest["checked_inventory"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md"
    )
    assert manifest["checked_rollup"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json"
    )
    assert manifest["source_contracts"] == {
        "numerical_truth_contract": "automation/contracts/numerical-truth-contract.md",
        "phase2_numerical_truth_contract": "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
        "phase9_public_api_contract": "contdid-py/contracts/phase9/public_api_contract_v1.json",
        "phase10_release_gate_manifest": "reproduction/phase10_release_gate/manifest.json",
        "phase10_checked_release_gate_summary": "reproduction/phase10_release_gate/runs/release-gate_summary.json",
        "phase10_checked_all_gates_summary": "reproduction/phase10_release_gate/runs/all_gates_summary.json",
    }
    assert "phase10_checked_qa_summary" not in manifest["source_contracts"]
    assert manifest["downstream_handoffs"] == [
        {
            "handoff_id": "phase10-qa-refresh",
            "owner_lane": "contdid-gsd-qa-mc-48",
            "purpose": (
                "Refresh the checked QA and shared v1 handoff packets whenever "
                "the package verification packet changes, because the QA packet "
                "embeds the package and release-gate truth snapshots."
            ),
            "required_truth_inputs": [
                "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json",
                "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md",
                "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json",
                "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
                "automation/contracts/phase10/runs/v1_audit_blocker_handoff_summary.json",
            ],
            "next_command": (
                "python3 tests/run_phase10_qa_verification.py "
                "--allow-incomplete-write "
                "--output tests/contracts/phase10/runs/qa_phase_verification_summary.json"
            ),
            "exit_criteria": (
                "The checked QA summary embeds the refreshed package snapshot "
                "and the shared v1 handoff packet re-materializes against the "
                "current release-gate frontier."
            ),
            "blocker_packet_paths": [
                "tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json",
                "automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json",
            ],
        },
        {
            "handoff_id": "phase10-theory-refresh",
            "owner_lane": "contdid-gsd-theory-parity-01",
            "purpose": (
                "Refresh the checked theory and v1 audit handoff packets whenever "
                "the package verification packet changes, because the theory packet "
                "embeds the package snapshot hashes."
            ),
            "required_truth_inputs": [
                "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json",
                "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md",
                "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json",
                "automation/contracts/phase10/runs/theory_phase_verification_summary.json",
                "automation/contracts/phase10/runs/v1_audit_blocker_handoff_summary.json",
            ],
            "next_command": (
                "python3 automation/scripts/run_theory_phase_verification.py "
                "--output automation/contracts/phase10/runs/theory_phase_verification_summary.json "
                "--inventory-output automation/contracts/phase10/runs/theory_phase_verification_inventory.md "
                "--rollup-output automation/contracts/phase10/runs/theory_phase_verification_rollup.json "
                "--verification-doc-root . && "
                "python3 automation/scripts/run_v1_audit_blocker_handoff.py "
                "--allow-incomplete-write "
                "--output automation/contracts/phase10/runs/v1_audit_blocker_handoff_summary.json "
                "--inventory-output automation/contracts/phase10/runs/v1_audit_blocker_handoff_inventory.md "
                "--blocker-output automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json"
            ),
            "exit_criteria": (
                "The checked theory and v1 audit handoff summaries return "
                "completed_successfully=true against the refreshed package packet."
            ),
            "blocker_packet_paths": [
                "automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json"
            ],
        },
        {
            "handoff_id": "phase10-release-gate-rerun",
            "owner_lane": "contdid-gsd-audit-repair-49",
            "purpose": (
                "Rerun the checked release-gate/all-gates bundle after the upstream "
                "theory and QA packets are green."
            ),
            "required_truth_inputs": [
                "reproduction/phase10_release_gate/manifest.json",
                "reproduction/phase10_release_gate/runs/release-gate_summary.json",
                "reproduction/phase10_release_gate/runs/all_gates_summary.json",
                "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
            ],
            "next_command": (
                "python3 reproduction/phase10_release_gate/run_release_gate.py "
                "--gate-id all --output-root reproduction/phase10_release_gate/runs"
            ),
            "exit_criteria": (
                "reproduction/phase10_release_gate/runs/{release-gate,all_gates}_summary.json "
                "both report completed_successfully=true and no release-gate blocker "
                "packet remains."
            ),
            "blocker_packet_paths": [
                "reproduction/phase10_release_gate/runs/release-gate_blocker_packet.json",
                "reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json",
            ],
        },
    ]
    assert manifest["maintenance_commands"] == _expected_maintenance_commands()
    assert manifest["external_requirement_sources"] == [
        {
            "phase": 2,
            "bundle_id": "phase2-truth-contracts",
            "label": "Theory verification evidence for the paper-truth contracts",
            "summary_path": "automation/contracts/phase10/runs/theory_phase_verification_summary.json",
            "inventory_path": "automation/contracts/phase10/runs/theory_phase_verification_inventory.md",
            "requirement_ids": ["TRUTH-01", "TRUTH-02"],
            "evidence_paths": [
                "contdid-py/contracts/phase2/paper_truth_contract.json",
                "contdid-py/contracts/phase2/symbol_map_contract.json",
                "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
                "contdid-py/contracts/phase2/phase2_contract_template.json",
            ],
        },
        {
            "phase": 10,
            "bundle_id": "phase10-release-gate-evidence",
            "label": "QA verification evidence for the checked release gate",
            "summary_path": "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
            "inventory_path": None,
            "requirement_ids": ["REL-02"],
            "evidence_paths": [
                "reproduction/phase10_release_gate/manifest.json",
                "reproduction/phase10_release_gate/runs/all_gates_summary.json",
            ],
        },
    ]

    bundle_ids = [bundle["bundle_id"] for bundle in manifest["phase_bundles"]]
    assert bundle_ids == [
        "phase3-package-foundation",
        "phase4-core-estimation",
        "phase5-eventstudy",
        "phase6-inference",
        "phase7-empirical-scaffold",
        "phase8-monte-carlo",
        "phase9-public-api",
        "phase10-frontier-runtime-helper",
    ]

    phase9_bundle = next(
        bundle
        for bundle in manifest["phase_bundles"]
        if bundle["bundle_id"] == "phase9-public-api"
    )
    assert phase9_bundle["phase"] == 9
    assert phase9_bundle["requirement_ids"] == ["REL-01"]
    assert phase9_bundle["test_paths"] == [
        "contdid-py/tests/test_phase9_public_api_contract.py",
        "tests/test_phase9_release_docs.py",
        "tests/test_phase9_release_docs_contract.py",
        "tests/test_phase9_release_examples.py",
        "tests/test_phase9_release_example_packet.py",
    ]
    assert phase9_bundle["evidence_paths"] == [
        "contdid-py/contracts/phase9/public_api_contract_v1.json",
        "README.md",
        "docs/help/public-api.md",
        "docs/help/medicare-pps-example.md",
        "reproduction/phase9_release_examples/manifest.json",
        "reproduction/phase9_release_examples/consumer-outputs/dose_curve.png",
        "reproduction/phase9_release_examples/consumer-outputs/dose_curve_metadata.json",
        "reproduction/phase9_release_examples/consumer-outputs/medicare_release_walkthrough.md",
        "reproduction/phase9_release_examples/consumer-outputs/medicare_release_walkthrough.json",
        "reproduction/medicare_pps/release_example_manifest.json",
    ]
    assert (
        phase9_bundle["verification_doc_target"]
        == ".planning/phases/09-pythonic-public-api-plotting-examples-english-release-docs/09-VERIFICATION.md"
    )
    phase10_bundle = next(
        bundle
        for bundle in manifest["phase_bundles"]
        if bundle["bundle_id"] == "phase10-frontier-runtime-helper"
    )
    assert phase10_bundle["phase"] == 10
    assert phase10_bundle["requirement_ids"] == ["REL-02"]
    assert (
        phase10_bundle["test_paths"]
        == _expected_phase10_frontier_runtime_helper_test_paths()
    )
    assert (
        phase10_bundle["execution_test_paths"]
        == _expected_phase10_frontier_runtime_helper_execution_test_paths()
    )
    assert (
        phase10_bundle["evidence_paths"]
        == _expected_phase10_frontier_runtime_helper_evidence_paths()
    )

    doc_targets = {
        bundle["phase"]: bundle["verification_doc_target"]
        for bundle in manifest["phase_bundles"]
        if "verification_doc_target" in bundle
    }
    assert doc_targets == {
        3: ".planning/phases/03-contdid-py/03-VERIFICATION.md",
        7: ".planning/phases/07-paper-to-python-e2e/07-VERIFICATION.md",
        8: ".planning/phases/08-monte-carlo/08-VERIFICATION.md",
        9: ".planning/phases/09-pythonic-public-api-plotting-examples-english-release-docs/09-VERIFICATION.md",
    }


def test_phase10_package_phase_verification_runner_dry_run_writes_bundle_summary(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "package_phase_verification_summary.json"
    inventory_path = tmp_path / "package_phase_verification_inventory.md"
    rollup_path = tmp_path / "package_phase_verification_rollup.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--dry-run",
            "--output",
            str(output_path),
            "--inventory-output",
            str(inventory_path),
            "--rollup-output",
            str(rollup_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists(), output_path

    summary = _load_json(output_path)
    assert summary["schema_version"] == "0.1"
    assert summary["report_type"] == "package_phase_verification_summary"
    _assert_iso_utc_timestamp(summary["generated_at"])
    assert summary["checked_summary"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json"
    )
    assert summary["checked_inventory"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md"
    )
    assert summary["checked_rollup"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json"
    )
    assert summary["dry_run"] is True
    assert summary["completed_successfully"] is True
    assert summary["bundle_count"] == 8
    assert summary["result_totals"] == {"planned": 8, "passed": 0, "failed": 0}
    assert summary["inventory_output"] == _expected_output_path(inventory_path)
    assert summary["rollup_output"] == _expected_output_path(rollup_path)
    assert summary["covered_requirement_ids"] == [
        "EST-01",
        "EST-02",
        "EST-03",
        "EST-04",
        "PKG-01",
        "PKG-02",
        "REL-01",
        "REL-02",
        "TEST-01",
        "TEST-02",
        "TEST-03",
        "TEST-04",
    ]
    assert summary["requirement_totals"] == {"unique": 12, "bundle_mentions": 13}
    assert summary["milestone_requirement_ids"] == [
        "EST-01",
        "EST-02",
        "EST-03",
        "EST-04",
        "PKG-01",
        "PKG-02",
        "REL-01",
        "REL-02",
        "TEST-01",
        "TEST-02",
        "TEST-03",
        "TEST-04",
        "TRUTH-01",
        "TRUTH-02",
    ]
    assert summary["milestone_requirement_totals"] == {
        "unique": 14,
        "package_bundle_mentions": 13,
        "external_source_mentions": 3,
    }
    assert summary["total_duration_seconds"] == 0
    assert summary["slowest_bundle"] is None
    assert summary["verification_doc_targets"] == [
        {
            "bundle_id": "phase3-package-foundation",
            "phase": 3,
            "requirement_ids": ["PKG-01", "PKG-02", "TEST-01"],
            "target_path": ".planning/phases/03-contdid-py/03-VERIFICATION.md",
        },
        {
            "bundle_id": "phase7-empirical-scaffold",
            "phase": 7,
            "requirement_ids": ["TEST-03"],
            "target_path": ".planning/phases/07-paper-to-python-e2e/07-VERIFICATION.md",
        },
        {
            "bundle_id": "phase8-monte-carlo",
            "phase": 8,
            "requirement_ids": ["TEST-04"],
            "target_path": ".planning/phases/08-monte-carlo/08-VERIFICATION.md",
        },
        {
            "bundle_id": "phase9-public-api",
            "phase": 9,
            "requirement_ids": ["REL-01"],
            "target_path": ".planning/phases/09-pythonic-public-api-plotting-examples-english-release-docs/09-VERIFICATION.md",
        },
    ]
    assert summary["verification_doc_output_root"] is None
    assert summary["verification_doc_outputs"] == []
    assert (
        summary["downstream_handoffs"]
        == _load_json(MANIFEST_PATH)["downstream_handoffs"]
    )
    assert summary["source_contracts"] == {
        "numerical_truth_contract": "automation/contracts/numerical-truth-contract.md",
        "phase2_numerical_truth_contract": "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
        "phase9_public_api_contract": "contdid-py/contracts/phase9/public_api_contract_v1.json",
        "phase10_release_gate_manifest": "reproduction/phase10_release_gate/manifest.json",
        "phase10_checked_release_gate_summary": "reproduction/phase10_release_gate/runs/release-gate_summary.json",
        "phase10_checked_all_gates_summary": "reproduction/phase10_release_gate/runs/all_gates_summary.json",
    }
    assert "phase10_checked_qa_summary" not in summary["source_contracts"]
    assert summary["maintenance_commands"] == _expected_maintenance_commands()
    assert summary["source_truth_snapshot"] == _expected_source_truth_snapshot()
    assert (
        summary["release_gate_truth_snapshot"]
        == _expected_release_gate_truth_snapshot()
    )
    assert [result["status"] for result in summary["bundle_results"]] == ["planned"] * 8
    assert [result["requirement_ids"] for result in summary["bundle_results"]] == [
        ["PKG-01", "PKG-02", "TEST-01"],
        ["EST-01", "TEST-02"],
        ["EST-02", "TEST-02"],
        ["EST-03", "EST-04"],
        ["TEST-03"],
        ["TEST-04"],
        ["REL-01"],
        ["REL-02"],
    ]
    assert summary["external_requirement_results"] == [
        {
            "phase": 2,
            "bundle_id": "phase2-truth-contracts",
            "label": "Theory verification evidence for the paper-truth contracts",
            "status": "passed",
            "summary_path": "automation/contracts/phase10/runs/theory_phase_verification_summary.json",
            "inventory_path": "automation/contracts/phase10/runs/theory_phase_verification_inventory.md",
            "requirement_ids": ["TRUTH-01", "TRUTH-02"],
            "evidence_paths": [
                "contdid-py/contracts/phase2/paper_truth_contract.json",
                "contdid-py/contracts/phase2/symbol_map_contract.json",
                "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
                "contdid-py/contracts/phase2/phase2_contract_template.json",
            ],
        },
        {
            "phase": 10,
            "bundle_id": "phase10-release-gate-evidence",
            "label": "QA verification evidence for the checked release gate",
            "status": "passed",
            "summary_path": "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
            "inventory_path": None,
            "requirement_ids": ["REL-02"],
            "evidence_paths": [
                "reproduction/phase10_release_gate/manifest.json",
                "reproduction/phase10_release_gate/runs/all_gates_summary.json",
            ],
        },
    ]
    assert inventory_path.exists(), inventory_path
    assert rollup_path.exists(), rollup_path
    inventory_text = inventory_path.read_text(encoding="utf-8")
    assert "# contdid-py phase verification inventory" in inventory_text
    assert "## Source truth snapshot" in inventory_text
    assert "## Release-gate truth snapshot" in inventory_text
    assert "## Maintenance commands" in inventory_text
    assert "refresh-release-gate-truth-snapshot" in inventory_text
    assert EXPECTED_RELEASE_GATE_TRUTH_REFRESH_COMMAND in inventory_text
    assert "lane49-timeout-safe-packaging-release-verifier" in inventory_text
    assert EXPECTED_LANE49_TIMEOUT_SAFE_VERIFIER_COMMAND in inventory_text
    assert "phase10_frontier_expectations.py" in inventory_text
    assert "checked_release_gate_summary_path" in inventory_text
    assert "checked_release_gate_blocker_exact_next_predicate" in inventory_text
    assert "maintenance_refresh_command" in inventory_text
    assert "maintenance_refresh_reason" in inventory_text
    assert "archival_next_command" in inventory_text
    assert "release_evidence_snapshot" in inventory_text
    assert "checked_qa_summary_path" in inventory_text
    assert "## Requirement coverage" in inventory_text
    assert "## External requirement sources" in inventory_text
    assert "| 3 | phase3-package-foundation | planned |" in inventory_text
    assert "TRUTH-01, TRUTH-02" in inventory_text
    assert "REL-02" in inventory_text
    assert "package_phase_verification_rollup.json" in inventory_text
    assert "## Downstream handoffs" in inventory_text
    assert "phase10-qa-refresh" in inventory_text
    assert "contdid-gsd-qa-mc-48" in inventory_text
    assert (
        "tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json"
        in inventory_text
    )
    assert "phase10-theory-refresh" in inventory_text
    assert "contdid-gsd-theory-parity-01" in inventory_text
    assert (
        "automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json"
        in inventory_text
    )
    assert "phase10-release-gate-rerun" in inventory_text
    assert (
        "reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json"
        in inventory_text
    )
    assert "tests/test_phase10_frontier_expectations_import.py" in inventory_text
    assert "tests/test_phase10_qa_frontier_expectations.py" in inventory_text
    assert "## Phase verification targets" in inventory_text
    assert (
        "| 3 | phase3-package-foundation | PKG-01, PKG-02, TEST-01 | .planning/phases/03-contdid-py/03-VERIFICATION.md | ready |"
        in inventory_text
    )
    assert "| 10 | phase10-frontier-runtime-helper | planned |" in inventory_text
    assert (
        "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json"
        in inventory_text
    )
    assert "contdid-py/src/phase10_frontier_expectations.py" in inventory_text

    rollup = _load_json(rollup_path)
    assert rollup["report_type"] == "package_phase_verification_rollup"
    assert rollup["generated_at"] == summary["generated_at"]
    assert rollup["summary_path"] == _expected_output_path(output_path)
    assert rollup["covered_requirement_ids"] == summary["covered_requirement_ids"]
    assert rollup["milestone_requirement_ids"] == summary["milestone_requirement_ids"]
    assert rollup["phase_results"][0]["phase"] == 3
    assert rollup["external_phase_results"] == summary["external_requirement_results"]
    assert rollup["downstream_handoffs"] == summary["downstream_handoffs"]
    assert rollup["maintenance_commands"] == summary["maintenance_commands"]
    assert rollup["source_truth_snapshot"] == summary["source_truth_snapshot"]
    assert (
        rollup["release_gate_truth_snapshot"] == summary["release_gate_truth_snapshot"]
    )
    assert f"- Generated at: `{summary['generated_at']}`" in inventory_text
    assert rollup["phase_results"][0][
        "rerun_command"
    ] == _expected_focused_bundle_rerun_command("phase3-package-foundation")


def test_phase10_package_phase_verification_external_requirement_results_fail_on_red_summary(
    tmp_path: Path,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_external_red")
    red_summary = tmp_path / "red_external_summary.json"
    inventory_path = tmp_path / "external_inventory.md"
    evidence_path = tmp_path / "external_evidence.json"
    red_summary.write_text(
        json.dumps({"completed_successfully": False}) + "\n",
        encoding="utf-8",
    )
    inventory_path.write_text("# external inventory\n", encoding="utf-8")
    evidence_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "external_requirement_sources": [
            {
                "phase": 10,
                "bundle_id": "synthetic-red-external-source",
                "label": "Synthetic red external source",
                "summary_path": str(red_summary),
                "inventory_path": str(inventory_path),
                "requirement_ids": ["REL-02"],
                "evidence_paths": [str(evidence_path)],
            }
        ],
        "phase_bundles": [
            {
                "bundle_id": "synthetic-package-bundle",
                "phase": 10,
                "label": "Synthetic package bundle",
                "test_paths": [],
                "evidence_paths": [],
                "requirement_ids": ["REL-02"],
            }
        ],
    }

    [result] = module._resolve_external_requirement_results(
        manifest,
        manifest["phase_bundles"],
    )

    assert result["status"] == "failed"


def test_phase10_package_phase_verification_external_requirement_results_ignore_stale_self_reference(
    tmp_path: Path,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_stale_self_ref")
    stale_summary = tmp_path / "qa_phase_verification_summary.json"
    evidence_path = tmp_path / "all_gates_summary.json"
    stale_summary.write_text(
        json.dumps(
            {
                "completed_successfully": False,
                "blocker_packet": {
                    "owner_lane": "contdid-gsd-correct-course-56",
                    "failing_label": (
                        "Checked release-gate packet, shared frontier expectations, "
                        "and archival-ready control-plane evidence remain aligned"
                    ),
                    "failing_test_nodeid": (
                        "tests/test_phase10_frontier_expectations_import.py::"
                        "test_phase10_packaged_frontier_helper_uses_packaged_fallback_without_root_helper"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "external_requirement_sources": [
            {
                "phase": 10,
                "bundle_id": "phase10-release-gate-evidence",
                "label": "QA verification evidence for the checked release gate",
                "summary_path": "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
                "inventory_path": None,
                "requirement_ids": ["REL-02"],
                "evidence_paths": [str(evidence_path)],
            }
        ],
        "phase_bundles": [
            {
                "bundle_id": "synthetic-package-bundle",
                "phase": 10,
                "label": "Synthetic package bundle",
                "test_paths": [],
                "evidence_paths": [],
                "requirement_ids": ["REL-02"],
            }
        ],
    }

    monkeypatch_path = tmp_path / "repo" / "tests" / "contracts" / "phase10" / "runs"
    monkeypatch_path.mkdir(parents=True)
    (monkeypatch_path / "qa_phase_verification_summary.json").write_text(
        stale_summary.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    original_repo_root = module.REPO_ROOT
    module.REPO_ROOT = tmp_path / "repo"
    try:
        [result] = module._resolve_external_requirement_results(
            manifest,
            manifest["phase_bundles"],
        )
    finally:
        module.REPO_ROOT = original_repo_root

    assert result["status"] == "passed"


def test_phase10_package_phase_verification_external_requirement_results_ignore_package_self_reference(
    tmp_path: Path,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_stale_package_ref")
    evidence_path = tmp_path / "all_gates_summary.json"
    manifest = {
        "external_requirement_sources": [
            {
                "phase": 10,
                "bundle_id": "phase10-release-gate-evidence",
                "label": "QA verification evidence for the checked release gate",
                "summary_path": "tests/contracts/phase10/runs/qa_phase_verification_summary.json",
                "inventory_path": None,
                "requirement_ids": ["REL-02"],
                "evidence_paths": [str(evidence_path)],
            }
        ],
        "phase_bundles": [
            {
                "bundle_id": "synthetic-package-bundle",
                "phase": 10,
                "label": "Synthetic package bundle",
                "test_paths": [],
                "evidence_paths": [],
                "requirement_ids": ["REL-02"],
            }
        ],
    }
    qa_path = tmp_path / "repo" / "tests" / "contracts" / "phase10" / "runs"
    qa_path.mkdir(parents=True)
    (qa_path / "qa_phase_verification_summary.json").write_text(
        json.dumps(
            {
                "completed_successfully": False,
                "blocker_packet": {
                    "owner_lane": "contdid-gsd-main-exec-18",
                    "failing_test_nodeid": (
                        "contdid-py/tests/test_phase10_package_phase_verification.py::"
                        "test_phase10_package_phase_verification_checked_outputs_are_present_and_green"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text("{}\n", encoding="utf-8")
    original_repo_root = module.REPO_ROOT
    module.REPO_ROOT = tmp_path / "repo"
    try:
        [result] = module._resolve_external_requirement_results(
            manifest,
            manifest["phase_bundles"],
        )
    finally:
        module.REPO_ROOT = original_repo_root

    assert result["status"] == "passed"


def test_phase10_package_phase_verification_refresh_only_is_lock_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_runner_lock_free")
    manifest = _load_json(MANIFEST_PATH)
    stale_pid_path = tmp_path / ".tmp-release-gate.pid"
    stale_pid_path.write_text("424242\n", encoding="utf-8")

    refresh_called = False
    downstream_calls: list[dict[str, object]] = []

    def _expected_refresh(**_: object) -> int:
        nonlocal refresh_called
        refresh_called = True
        return 0

    def _record_downstream_refresh(**kwargs: object) -> None:
        downstream_calls.append(kwargs)

    assert not hasattr(module, "RUN_LOCK_PATH")
    assert not hasattr(module, "_acquire_canonical_writer_guard")
    assert not hasattr(module, "_release_canonical_writer_guard")
    monkeypatch.setattr(
        module,
        "_refresh_release_gate_truth_snapshot_outputs",
        _expected_refresh,
    )
    monkeypatch.setattr(
        module,
        "_refresh_downstream_checked_packets_after_canonical_write",
        _record_downstream_refresh,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--refresh-release-gate-truth-snapshot-only",
            "--output",
            manifest["checked_summary"],
            "--inventory-output",
            manifest["checked_inventory"],
            "--rollup-output",
            manifest["checked_rollup"],
        ],
    )

    assert module.main() == 0
    assert refresh_called is True
    assert downstream_calls == [
        {
            "manifest": manifest,
            "output_path": Path(manifest["checked_summary"]),
            "inventory_output": Path(manifest["checked_inventory"]),
            "rollup_output": Path(manifest["checked_rollup"]),
            "selected_bundles": manifest["phase_bundles"],
            "dry_run": False,
            "completed_successfully": True,
        }
    ]
    assert stale_pid_path.read_text(encoding="utf-8") == "424242\n"


def test_phase10_package_phase_verification_refresh_only_refreshes_downstream_checked_packets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_refresh_only_sync")
    manifest = _load_json(MANIFEST_PATH)
    output_path = tmp_path / "package_phase_verification_summary.json"
    inventory_path = tmp_path / "package_phase_verification_inventory.md"
    rollup_path = tmp_path / "package_phase_verification_rollup.json"
    downstream_calls: list[dict] = []

    def _write_refreshed_summary(
        *,
        manifest: dict,
        output_path: Path,
        inventory_output: Path | None,
        rollup_output: Path | None,
    ) -> int:
        output_path.write_text(
            json.dumps(
                {
                    "selected_bundle_ids": [
                        bundle["bundle_id"] for bundle in manifest["phase_bundles"]
                    ],
                    "completed_successfully": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    def _record_downstream_refresh(**kwargs: object) -> None:
        downstream_calls.append(kwargs)

    monkeypatch.setattr(
        module,
        "_refresh_release_gate_truth_snapshot_outputs",
        _write_refreshed_summary,
    )
    monkeypatch.setattr(
        module,
        "_refresh_downstream_checked_packets_after_canonical_write",
        _record_downstream_refresh,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--refresh-release-gate-truth-snapshot-only",
            "--output",
            str(output_path),
            "--inventory-output",
            str(inventory_path),
            "--rollup-output",
            str(rollup_path),
        ],
    )

    assert module.main() == 0
    assert downstream_calls == [
        {
            "manifest": manifest,
            "output_path": output_path,
            "inventory_output": inventory_path,
            "rollup_output": rollup_path,
            "selected_bundles": manifest["phase_bundles"],
            "dry_run": False,
            "completed_successfully": True,
        }
    ]


def test_phase10_package_phase_verification_refreshes_downstream_checked_packets_after_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_downstream_sync")
    manifest = _load_json(MANIFEST_PATH)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        module,
        "_release_gate_truth_snapshot",
        lambda manifest: {
            "checked_all_gates_summary_completed_successfully": False,
            "checked_all_gates_live_frontier_owner_ready_label": "refresh checked theory verification packet",
        },
    )

    def _record_refresh(command: list[str], *, label: str) -> None:
        calls.append((label, command))

    def _record_truth_snapshot_refresh(
        *,
        manifest: dict,
        output_path: Path,
        inventory_output: Path | None,
        rollup_output: Path | None,
    ) -> int:
        calls.append(
            (
                "refresh package release-gate snapshot after QA",
                {
                    "output_path": output_path,
                    "inventory_output": inventory_output,
                    "rollup_output": rollup_output,
                },
            )
        )
        return 0

    monkeypatch.setattr(module, "_run_checked_refresh_command", _record_refresh)
    monkeypatch.setattr(
        module,
        "_refresh_release_gate_truth_snapshot_outputs",
        _record_truth_snapshot_refresh,
    )

    module._refresh_downstream_checked_packets_after_canonical_write(
        manifest=manifest,
        output_path=REPO_ROOT / manifest["checked_summary"],
        inventory_output=REPO_ROOT / manifest["checked_inventory"],
        rollup_output=REPO_ROOT / manifest["checked_rollup"],
        selected_bundles=manifest["phase_bundles"],
        dry_run=False,
        completed_successfully=True,
    )

    assert calls == [
        (
            "post-package checked QA refresh",
            [
                sys.executable,
                str(module.QA_RUNNER_PATH),
                "--skip-release-gate-shared-frontier-sync",
                "--skip-downstream-checked-packet-refresh",
                "--output",
                str(module.QA_SUMMARY_PATH),
            ],
        ),
        (
            "refresh package release-gate snapshot after QA",
            {
                "output_path": REPO_ROOT / manifest["checked_summary"],
                "inventory_output": REPO_ROOT / manifest["checked_inventory"],
                "rollup_output": REPO_ROOT / manifest["checked_rollup"],
            },
        ),
        (
            "post-package checked theory refresh",
            [
                sys.executable,
                str(module.THEORY_RUNNER_PATH),
                "--output",
                str(module.THEORY_SUMMARY_PATH),
                "--inventory-output",
                str(module.THEORY_INVENTORY_PATH),
                "--rollup-output",
                str(module.THEORY_ROLLUP_PATH),
                "--verification-doc-root",
                ".",
            ],
        ),
        (
            "post-package checked v1 handoff refresh",
            [
                sys.executable,
                str(module.V1_RUNNER_PATH),
                "--allow-incomplete-write",
                "--output",
                str(module.V1_SUMMARY_PATH),
                "--inventory-output",
                str(module.V1_INVENTORY_PATH),
                "--blocker-output",
                str(module.V1_BLOCKER_PACKET_PATH),
            ],
        ),
    ]


def test_phase10_package_phase_verification_preserves_strict_qa_summary_when_refresh_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_qa_fallback")
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        module,
        "_release_gate_truth_snapshot",
        lambda manifest: {
            "checked_all_gates_summary_completed_successfully": False,
            "checked_all_gates_live_frontier_owner_ready_label": "refresh checked theory verification packet",
        },
    )

    def _record_refresh(command: list[str], *, label: str) -> None:
        calls.append((label, command))
        if label == "post-package checked QA refresh":
            raise RuntimeError("strict QA refresh failed")

    monkeypatch.setattr(module, "_run_checked_refresh_command", _record_refresh)

    assert module._refresh_checked_qa_after_package_write() is False

    assert calls == [
        (
            "post-package checked QA refresh",
            [
                sys.executable,
                str(module.QA_RUNNER_PATH),
                "--skip-release-gate-shared-frontier-sync",
                "--skip-downstream-checked-packet-refresh",
                "--output",
                str(module.QA_SUMMARY_PATH),
            ],
        ),
    ]


def test_phase10_package_phase_verification_declares_checked_qa_refresh_return_type() -> (
    None
):
    module = _load_runner_module("phase10_package_phase_verification_return_type")

    annotations = get_type_hints(module._refresh_checked_qa_after_package_write)

    assert annotations["return"] is bool


def test_phase10_package_phase_verification_forces_release_gate_bundle_tests_when_self_owned_package_blocker_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module(
        "phase10_package_phase_verification_self_owned_release_gate_blocker"
    )
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        module,
        "_release_gate_truth_snapshot",
        lambda manifest: {
            "checked_all_gates_summary_completed_successfully": False,
            "checked_all_gates_live_frontier_owner_ready_label": "refresh checked package verification packet",
            "checked_all_gates_live_frontier_failing_label": "refresh checked package verification packet",
        },
    )

    def _record_refresh(command: list[str], *, label: str) -> None:
        calls.append((label, command))
        if label == "post-package checked QA refresh":
            raise RuntimeError("strict QA refresh failed")

    monkeypatch.setattr(module, "_run_checked_refresh_command", _record_refresh)

    assert module._refresh_checked_qa_after_package_write() is False

    assert calls
    for _, command in calls:
        assert "--force-release-gate-bundle-tests" in command
        assert "--skip-downstream-checked-packet-refresh" in command


def test_phase10_package_phase_verification_stops_downstream_refresh_when_strict_qa_refresh_is_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_theory_after_red_qa")
    manifest = _load_json(MANIFEST_PATH)
    calls: list[tuple[str, object]] = []

    def _record_refresh(command: list[str], *, label: str) -> None:
        calls.append((label, command))
        if label == "post-package checked QA refresh":
            raise RuntimeError("strict QA refresh failed")

    def _record_truth_snapshot_refresh(
        *,
        manifest: dict,
        output_path: Path,
        inventory_output: Path | None,
        rollup_output: Path | None,
    ) -> int:
        calls.append(
            (
                "refresh package release-gate snapshot after QA",
                {
                    "output_path": output_path,
                    "inventory_output": inventory_output,
                    "rollup_output": rollup_output,
                },
            )
        )
        return 0

    monkeypatch.setattr(module, "_run_checked_refresh_command", _record_refresh)
    monkeypatch.setattr(
        module,
        "_refresh_release_gate_truth_snapshot_outputs",
        _record_truth_snapshot_refresh,
    )

    module._refresh_downstream_checked_packets_after_canonical_write(
        manifest=manifest,
        output_path=REPO_ROOT / manifest["checked_summary"],
        inventory_output=REPO_ROOT / manifest["checked_inventory"],
        rollup_output=REPO_ROOT / manifest["checked_rollup"],
        selected_bundles=manifest["phase_bundles"],
        dry_run=False,
        completed_successfully=True,
    )

    assert [label for label, _ in calls] == [
        "post-package checked QA refresh",
        "refresh package release-gate snapshot after QA",
    ]


def test_phase10_package_phase_verification_keeps_release_gate_snapshot_for_selected_canonical_output(
    tmp_path: Path,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_selected_snapshot")
    manifest = _load_json(MANIFEST_PATH)

    assert module._should_include_release_gate_truth_snapshot(
        manifest=manifest,
        output_path=REPO_ROOT / manifest["checked_summary"],
        bundle_ids=["phase10-frontier-runtime-helper"],
    )
    assert not module._should_include_release_gate_truth_snapshot(
        manifest=manifest,
        output_path=tmp_path / "selected_summary.json",
        bundle_ids=["phase10-frontier-runtime-helper"],
    )
    assert module._should_include_release_gate_truth_snapshot(
        manifest=manifest,
        output_path=tmp_path / "full_summary.json",
        bundle_ids=[],
    )


def test_phase10_package_phase_verification_refuses_partial_canonical_bundle_write() -> (
    None
):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--bundle-id",
            "phase3-package-foundation",
            "--output",
            "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "refusing to overwrite the canonical checked package summary with a "
        "partial --bundle-id selection"
    ) in completed.stderr


def test_phase10_package_phase_verification_skip_downstream_refresh_covers_full_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_skip_downstream")
    checked_summary = tmp_path / "checked_summary.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "report_type": "package_phase_verification_manifest",
                "owner_lane": "contdid-gsd-audit-repair-49",
                "package_root": "contdid-py",
                "checked_summary": str(checked_summary),
                "checked_inventory": str(tmp_path / "checked_inventory.md"),
                "checked_rollup": str(tmp_path / "checked_rollup.json"),
                "source_contracts": {},
                "external_requirement_sources": [],
                "phase_bundles": [
                    {
                        "bundle_id": "phase-package-smoke",
                        "phase": 10,
                        "label": "Package smoke",
                        "test_paths": ["tests/test_smoke.py"],
                        "evidence_paths": ["tests/test_smoke.py"],
                        "requirement_ids": ["REL-02"],
                    }
                ],
                "downstream_handoffs": [],
                "maintenance_commands": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    downstream_calls: list[dict] = []

    def _fake_run_bundle(bundle: dict, *, bundle_timeout_seconds=None) -> dict:
        return {
            "bundle_id": bundle["bundle_id"],
            "phase": bundle["phase"],
            "label": bundle["label"],
            "status": "passed",
            "returncode": 0,
            "duration_seconds": 0.123,
            "command": ["python3", "-m", "pytest", "-q"],
            "test_paths": bundle["test_paths"],
            "execution_test_paths": bundle["test_paths"],
            "evidence_paths": bundle["evidence_paths"],
            "requirement_ids": bundle["requirement_ids"],
            "verification_doc_target": None,
            "stdout": ".",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_run_bundle", _fake_run_bundle)
    monkeypatch.setattr(
        module,
        "_source_truth_snapshot",
        lambda manifest, selected_bundles: {
            "source_contracts": {},
            "bundle_evidence": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_release_gate_truth_snapshot",
        lambda manifest: {"release_snapshot": "stubbed"},
    )
    monkeypatch.setattr(
        module,
        "_refresh_downstream_checked_packets_after_canonical_write",
        lambda **kwargs: downstream_calls.append(kwargs),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--manifest",
            str(manifest_path),
            "--skip-downstream-refresh",
            "--output",
            str(checked_summary),
        ],
    )

    assert module.main() == 0
    assert checked_summary.exists()
    assert downstream_calls == []


def test_phase10_package_phase_verification_runner_executes_selected_bundle(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "phase3_package_verification_summary.json"
    inventory_path = tmp_path / "phase3_package_verification_inventory.md"
    rollup_path = tmp_path / "phase3_package_verification_rollup.json"
    verification_doc_root = tmp_path / "verification-docs"
    expected_doc_path = (
        verification_doc_root
        / ".planning"
        / "phases"
        / "03-contdid-py"
        / "03-VERIFICATION.md"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--bundle-id",
            "phase3-package-foundation",
            "--output",
            str(output_path),
            "--inventory-output",
            str(inventory_path),
            "--rollup-output",
            str(rollup_path),
            "--verification-doc-root",
            str(verification_doc_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = _load_json(output_path)

    assert summary["dry_run"] is False
    assert summary["completed_successfully"] is True
    assert summary["selected_bundle_ids"] == ["phase3-package-foundation"]
    assert summary["result_totals"] == {"planned": 0, "passed": 1, "failed": 0}
    assert summary["inventory_output"] == _expected_output_path(inventory_path)
    assert summary["rollup_output"] == _expected_output_path(rollup_path)
    assert summary["source_truth_snapshot"] == _expected_source_truth_snapshot(
        ["phase3-package-foundation"]
    )
    [bundle_result] = summary["bundle_results"]
    assert summary["total_duration_seconds"] == bundle_result["duration_seconds"]
    assert summary["slowest_bundle"] == {
        "bundle_id": "phase3-package-foundation",
        "phase": 3,
        "label": bundle_result["label"],
        "duration_seconds": bundle_result["duration_seconds"],
    }
    assert summary["milestone_requirement_ids"] == ["PKG-01", "PKG-02", "TEST-01"]
    assert summary["milestone_requirement_totals"] == {
        "unique": 3,
        "package_bundle_mentions": 3,
        "external_source_mentions": 0,
    }
    assert summary["external_requirement_results"] == []
    assert summary["verification_doc_output_root"] == _expected_output_path(
        verification_doc_root
    )
    assert summary["verification_doc_outputs"] == [str(expected_doc_path.resolve())]
    assert summary["verification_doc_targets"] == [
        {
            "bundle_id": "phase3-package-foundation",
            "phase": 3,
            "requirement_ids": ["PKG-01", "PKG-02", "TEST-01"],
            "target_path": ".planning/phases/03-contdid-py/03-VERIFICATION.md",
        }
    ]

    assert bundle_result["bundle_id"] == "phase3-package-foundation"
    assert bundle_result["phase"] == 3
    assert bundle_result["status"] == "passed"
    assert bundle_result["requirement_ids"] == ["PKG-01", "PKG-02", "TEST-01"]
    assert bundle_result["command"] == [
        "python3",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "contdid-py/tests/test_package_layout.py",
        "contdid-py/tests/test_phase3_objects.py",
        "contdid-py/tests/test_phase3_simulate_data.py",
        "contdid-py/tests/test_phase3_simulation_contracts.py",
        "contdid-py/tests/test_phase3_validation_contracts.py",
        "-q",
    ]
    assert bundle_result["evidence_paths"] == [
        "contdid-py/contracts/phase2/numerical_truth_contract_v1.json",
        "reproduction/simulate_contdid/manifest.json",
        "reproduction/phase3_validation/manifest.json",
    ]
    assert bundle_result[
        "rerun_command"
    ] == _expected_focused_bundle_rerun_command("phase3-package-foundation")
    assert bundle_result["returncode"] == 0
    assert inventory_path.exists(), inventory_path
    assert rollup_path.exists(), rollup_path
    assert expected_doc_path.exists(), expected_doc_path
    doc_text = expected_doc_path.read_text(encoding="utf-8")
    bundle = _bundle_by_id("phase3-package-foundation")
    for needle in (
        "# Phase 3 Verification",
        "Owner lane: `contdid-gsd-audit-repair-49`",
        "Checked package summary: `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json`",
        "Requirement ids: `PKG-01`, `PKG-02`, `TEST-01`",
        "Package bundle id: `phase3-package-foundation`",
        _expected_focused_bundle_rerun_command("phase3-package-foundation"),
        f"- Scope: {bundle['verification_scope']}",
        f"- Non-claim: {bundle['verification_non_claims']}",
    ):
        assert needle in doc_text


def test_phase10_package_phase_verification_runner_executes_with_current_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module("phase10_package_phase_verification_interpreter")
    observed_commands: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ".\n"
        stderr = ""

    def _fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        observed_commands.append(list(command))
        assert kwargs["cwd"] == module.REPO_ROOT
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return _Completed()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module._run_bundle(
        {
            "bundle_id": "phase3-package-foundation",
            "phase": 3,
            "label": "Package layout",
            "test_paths": ["contdid-py/tests/test_package_layout.py"],
            "evidence_paths": ["contdid-py/pyproject.toml"],
            "requirement_ids": ["PKG-01"],
        }
    )

    assert observed_commands == [
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "contdid-py/tests/test_package_layout.py",
            "-q",
        ]
    ]
    assert result["command"] == [
        "python3",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "contdid-py/tests/test_package_layout.py",
        "-q",
    ]
    assert result["status"] == "passed"


def test_phase10_package_phase_verification_runner_records_bundle_timeout(
    tmp_path: Path,
) -> None:
    slow_test = tmp_path / "test_slow_package_bundle.py"
    slow_test.write_text(
        "import time\n\n\ndef test_slow_package_bundle() -> None:\n    time.sleep(0.2)\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "package_phase_verification_summary.json"
    manifest_path = tmp_path / "package_phase_verification_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "report_type": "package_phase_verification",
                "owner_lane": "contdid-gsd-audit-repair-49",
                "package_root": "contdid-py",
                "checked_summary": str(tmp_path / "checked_summary.json"),
                "checked_inventory": str(tmp_path / "checked_inventory.md"),
                "checked_rollup": str(tmp_path / "checked_rollup.json"),
                "source_contracts": {
                    "synthetic_truth": str(slow_test),
                },
                "external_requirement_sources": [],
                "phase_bundles": [
                    {
                        "bundle_id": "phase-slow-package",
                        "phase": 10,
                        "label": "Synthetic slow package bundle",
                        "requirement_ids": ["REL-02"],
                        "test_paths": [str(slow_test)],
                        "evidence_paths": [str(slow_test)],
                    }
                ],
                "downstream_handoffs": [],
                "maintenance_commands": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest",
            str(manifest_path),
            "--bundle-id",
            "phase-slow-package",
            "--bundle-timeout-seconds",
            "0.01",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, completed.stderr
    summary = _load_json(output_path)
    assert summary["bundle_timeout_seconds"] == 0.01
    assert summary["completed_successfully"] is False
    assert summary["result_totals"] == {"planned": 0, "passed": 0, "failed": 1}
    [result] = summary["bundle_results"]
    assert result["status"] == "failed"
    assert result["returncode"] == 124
    assert result["timeout_seconds"] == 0.01
    assert "timed out after 0.01 seconds" in result["failure_reason"]
    assert "--bundle-timeout-seconds 0.01" in result["rerun_command"]


def test_phase10_package_phase_verification_checked_outputs_are_present_and_green() -> (
    None
):
    assert CHECKED_SUMMARY_PATH.exists(), (
        f"missing checked summary: {CHECKED_SUMMARY_PATH}"
    )
    summary = _load_json(CHECKED_SUMMARY_PATH)

    assert summary["report_type"] == "package_phase_verification_summary"
    assert summary["completed_successfully"] is True
    assert summary["selected_bundle_ids"] == [
        "phase3-package-foundation",
        "phase4-core-estimation",
        "phase5-eventstudy",
        "phase6-inference",
        "phase7-empirical-scaffold",
        "phase8-monte-carlo",
        "phase9-public-api",
        "phase10-frontier-runtime-helper",
    ]
    assert summary["result_totals"] == {"planned": 0, "passed": 8, "failed": 0}
    assert summary["checked_inventory"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md"
    )
    assert summary["checked_rollup"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json"
    )
    assert summary["inventory_output"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md"
    )
    assert summary["rollup_output"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_rollup.json"
    )
    assert summary["covered_requirement_ids"] == [
        "EST-01",
        "EST-02",
        "EST-03",
        "EST-04",
        "PKG-01",
        "PKG-02",
        "REL-01",
        "REL-02",
        "TEST-01",
        "TEST-02",
        "TEST-03",
        "TEST-04",
    ]
    assert summary["milestone_requirement_ids"] == [
        "EST-01",
        "EST-02",
        "EST-03",
        "EST-04",
        "PKG-01",
        "PKG-02",
        "REL-01",
        "REL-02",
        "TEST-01",
        "TEST-02",
        "TEST-03",
        "TEST-04",
        "TRUTH-01",
        "TRUTH-02",
    ]
    assert summary["milestone_requirement_totals"] == {
        "unique": 14,
        "package_bundle_mentions": 13,
        "external_source_mentions": 3,
    }
    assert summary["maintenance_commands"] == _expected_maintenance_commands()
    assert summary["source_truth_snapshot"] == _expected_source_truth_snapshot()
    expected_release_gate_truth_snapshot = _expected_release_gate_truth_snapshot()
    observed_release_gate_truth_snapshot = dict(summary["release_gate_truth_snapshot"])
    expected_qa_summary_completed = expected_release_gate_truth_snapshot.pop(
        "checked_qa_summary_completed_successfully"
    )
    observed_qa_summary_completed = observed_release_gate_truth_snapshot.pop(
        "checked_qa_summary_completed_successfully"
    )
    assert observed_release_gate_truth_snapshot == expected_release_gate_truth_snapshot
    assert isinstance(observed_qa_summary_completed, bool)
    assert isinstance(expected_qa_summary_completed, bool)
    assert (
        summary["release_gate_truth_snapshot"]["release_evidence_snapshot"]
        == _expected_release_gate_truth_snapshot()["release_evidence_snapshot"]
    )
    assert (
        summary["downstream_handoffs"]
        == _load_json(MANIFEST_PATH)["downstream_handoffs"]
    )
    assert (
        summary["external_requirement_results"]
        == _expected_external_requirement_results()
    )
    phase10_result = next(
        result
        for result in summary["bundle_results"]
        if result["bundle_id"] == "phase10-frontier-runtime-helper"
    )
    assert (
        phase10_result["evidence_paths"]
        == _expected_phase10_frontier_runtime_helper_evidence_paths()
    )
    assert summary["total_duration_seconds"] == round(
        sum(result["duration_seconds"] for result in summary["bundle_results"]),
        3,
    )
    assert all(result["command"][0] == "python3" for result in summary["bundle_results"])
    slowest_result = max(
        summary["bundle_results"],
        key=lambda result: result["duration_seconds"],
    )
    assert summary["slowest_bundle"] == {
        "bundle_id": slowest_result["bundle_id"],
        "phase": slowest_result["phase"],
        "label": slowest_result["label"],
        "duration_seconds": slowest_result["duration_seconds"],
    }

    assert CHECKED_INVENTORY_PATH.exists(), (
        f"missing checked inventory: {CHECKED_INVENTORY_PATH}"
    )
    inventory_text = CHECKED_INVENTORY_PATH.read_text(encoding="utf-8")
    assert "# contdid-py phase verification inventory" in inventory_text
    assert "## Source truth snapshot" in inventory_text
    assert "## Release-gate truth snapshot" in inventory_text
    assert "## Maintenance commands" in inventory_text
    assert "refresh-release-gate-truth-snapshot" in inventory_text
    assert EXPECTED_RELEASE_GATE_TRUTH_REFRESH_COMMAND in inventory_text
    assert "lane49-timeout-safe-packaging-release-verifier" in inventory_text
    assert EXPECTED_LANE49_TIMEOUT_SAFE_VERIFIER_COMMAND in inventory_text
    assert "phase10_frontier_expectations.py" in inventory_text
    assert "checked_release_gate_summary_path" in inventory_text
    assert "maintenance_refresh_command" in inventory_text
    assert "maintenance_refresh_reason" in inventory_text
    assert "archival_next_command" in inventory_text
    assert "release_evidence_snapshot" in inventory_text
    assert "checked_qa_summary_path" in inventory_text
    assert (
        f"- Total duration: `{summary['total_duration_seconds']}` seconds"
        in inventory_text
    )
    assert (
        f"- Slowest bundle: `{summary['slowest_bundle']['bundle_id']}` "
        f"(Phase {summary['slowest_bundle']['phase']}, "
        f"{summary['slowest_bundle']['duration_seconds']}s)"
    ) in inventory_text
    assert "## External requirement sources" in inventory_text
    assert "## Downstream handoffs" in inventory_text
    assert "phase10-qa-refresh" in inventory_text
    assert (
        "tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json"
        in inventory_text
    )
    assert "phase10-theory-refresh" in inventory_text
    assert (
        "automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json"
        in inventory_text
    )
    assert "phase10-release-gate-rerun" in inventory_text
    assert (
        "reproduction/phase10_release_gate/runs/release-gate_blocker_packet.json"
        in inventory_text
    )
    assert (
        "reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json"
        in inventory_text
    )
    assert "## Phase verification targets" in inventory_text
    assert "| 9 | phase9-public-api | passed |" in inventory_text
    assert "| 10 | phase10-frontier-runtime-helper | passed |" in inventory_text
    assert "| TEST-02 | 2 | 4, 5 |" in inventory_text
    assert "TRUTH-01, TRUTH-02" in inventory_text
    assert "REL-02" in inventory_text
    assert (
        "| 3 | phase3-package-foundation | PKG-01, PKG-02, TEST-01 | .planning/phases/03-contdid-py/03-VERIFICATION.md | ready |"
        in inventory_text
    )
    assert "reproduction/medicare_pps/release_example_manifest.json" in inventory_text
    assert "contdid-py/src/phase10_frontier_expectations.py" in inventory_text
    assert "tests/test_phase10_frontier_expectations_import.py" in inventory_text
    assert "tests/test_phase10_qa_frontier_expectations.py" in inventory_text
    assert _expected_focused_bundle_rerun_command(
        "phase3-package-foundation",
    ) in inventory_text

    assert CHECKED_ROLLUP_PATH.exists(), (
        f"missing checked rollup: {CHECKED_ROLLUP_PATH}"
    )
    rollup = _load_json(CHECKED_ROLLUP_PATH)
    assert rollup["report_type"] == "package_phase_verification_rollup"
    assert rollup["checked_summary"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_summary.json"
    )
    assert rollup["checked_inventory"] == (
        "contdid-py/contracts/phase10/runs/package_phase_verification_inventory.md"
    )
    assert rollup["total_duration_seconds"] == summary["total_duration_seconds"]
    assert rollup["slowest_bundle"] == summary["slowest_bundle"]
    assert rollup["maintenance_commands"] == summary["maintenance_commands"]
    assert rollup["source_truth_snapshot"] == summary["source_truth_snapshot"]
    assert (
        rollup["release_gate_truth_snapshot"] == summary["release_gate_truth_snapshot"]
    )
    assert (
        rollup["release_gate_truth_snapshot"][
            "checked_all_gates_blocker_exact_next_predicate"
        ]
        == summary["release_gate_truth_snapshot"][
            "checked_all_gates_blocker_exact_next_predicate"
        ]
    )
    assert rollup["milestone_requirement_ids"] == summary["milestone_requirement_ids"]
    assert rollup["downstream_handoffs"] == summary["downstream_handoffs"]
    assert rollup["phase_results"][-1]["bundle_id"] == "phase10-frontier-runtime-helper"
    assert rollup["phase_results"][0]["bundle_id"] == "phase3-package-foundation"
    assert rollup["phase_results"][0]["requirement_ids"] == [
        "PKG-01",
        "PKG-02",
        "TEST-01",
    ]
    assert rollup["phase_results"][0][
        "rerun_command"
    ] == _expected_focused_bundle_rerun_command(
        "phase3-package-foundation",
    )
    assert rollup["external_phase_results"] == summary["external_requirement_results"]

    doc_expectations = {
        bundle["phase"]: [
            f"# Phase {bundle['phase']} Verification",
            "Requirement ids: "
            + ", ".join(f"`{item}`" for item in bundle["requirement_ids"]),
            f"Package bundle id: `{bundle['bundle_id']}`",
            f"- Scope: {bundle['verification_scope']}",
            f"- Non-claim: {bundle['verification_non_claims']}",
        ]
        for bundle in _load_json(MANIFEST_PATH)["phase_bundles"]
        if bundle.get("verification_doc_target")
    }
    for phase, path in CHECKED_PHASE_VERIFICATION_DOCS.items():
        assert path.exists(), f"missing package-owned phase verification doc: {path}"
        text = path.read_text(encoding="utf-8")
        for needle in doc_expectations[phase]:
            assert needle in text, f"missing {needle!r} in {path}"


def test_phase10_package_phase_verification_manifest_freezes_frontier_runtime_helper() -> (
    None
):
    manifest = _load_json(MANIFEST_PATH)
    bundle = next(
        (
            item
            for item in manifest["phase_bundles"]
            if item["bundle_id"] == "phase10-frontier-runtime-helper"
        ),
        None,
    )

    assert bundle is not None
    assert bundle["phase"] == 10
    assert bundle["requirement_ids"] == ["REL-02"]
    assert bundle["test_paths"] == _expected_phase10_frontier_runtime_helper_test_paths()
    assert (
        bundle["execution_test_paths"]
        == _expected_phase10_frontier_runtime_helper_execution_test_paths()
    )
    assert (
        bundle["evidence_paths"]
        == _expected_phase10_frontier_runtime_helper_evidence_paths()
    )
