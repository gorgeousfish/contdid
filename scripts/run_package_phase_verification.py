#!/usr/bin/env python3
"""Run or plan contdid-py phase verification bundles for the v1.0 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH_ENV = "CONTDID_PACKAGE_VERIFICATION_MANIFEST"
MANIFEST_PATH = (
    REPO_ROOT
    / "contdid-py"
    / "contracts"
    / "phase10"
    / "package_phase_verification_manifest.json"
)
QA_SUMMARY_PATH = (
    REPO_ROOT
    / "tests"
    / "contracts"
    / "phase10"
    / "runs"
    / "qa_phase_verification_summary.json"
)
QA_RUNNER_PATH = REPO_ROOT / "tests" / "run_phase10_qa_verification.py"
THEORY_RUNNER_PATH = (
    REPO_ROOT / "automation" / "scripts" / "run_theory_phase_verification.py"
)
THEORY_SUMMARY_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "theory_phase_verification_summary.json"
)
THEORY_INVENTORY_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "theory_phase_verification_inventory.md"
)
THEORY_ROLLUP_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "theory_phase_verification_rollup.json"
)
V1_RUNNER_PATH = (
    REPO_ROOT / "automation" / "scripts" / "run_v1_audit_blocker_handoff.py"
)
V1_SUMMARY_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "v1_audit_blocker_handoff_summary.json"
)
V1_INVENTORY_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "v1_audit_blocker_handoff_inventory.md"
)
V1_BLOCKER_PACKET_PATH = (
    REPO_ROOT
    / "automation"
    / "contracts"
    / "phase10"
    / "runs"
    / "v1_audit_blocker_handoff_blocker_packet.json"
)
SELF_OWNED_PACKAGE_REFRESH_LABEL = "refresh checked package verification packet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or plan the contdid-py phase verification bundles."
    )
    parser.add_argument(
        "--bundle-id",
        action="append",
        dest="bundle_ids",
        default=[],
        help="Restrict execution to a specific bundle id. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned bundle summary without executing pytest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional package verification manifest override. Defaults to the "
            "checked phase10 manifest, or CONTDID_PACKAGE_VERIFICATION_MANIFEST "
            "when that environment variable is set."
        ),
    )
    parser.add_argument(
        "--bundle-timeout-seconds",
        type=float,
        help="Optional per-bundle timeout for each pytest subprocess.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON summary file to write.",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        help="Optional path to a Markdown inventory derived from the bundle summary.",
    )
    parser.add_argument(
        "--rollup-output",
        type=Path,
        help="Optional path to a JSON phase/requirement rollup derived from the bundle summary.",
    )
    parser.add_argument(
        "--verification-doc-root",
        type=Path,
        help=(
            "Optional root under which bundle-owned .planning verification docs are "
            "materialized."
        ),
    )
    parser.add_argument(
        "--refresh-release-gate-truth-snapshot-only",
        action="store_true",
        help=(
            "Recompute the checked release-gate truth snapshot and rewrite existing "
            "summary/inventory/rollup outputs without rerunning bundle tests."
        ),
    )
    parser.add_argument(
        "--skip-downstream-refresh",
        action="store_true",
        help=(
            "When writing the checked package packet, do not cascade QA, theory, "
            "or v1 handoff refreshes."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _resolve_cli_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists() or str(path).startswith("contdid-py/"):
        return repo_candidate
    return path.resolve()


def _resolve_manifest_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_cli_output_path(path)
    env_path = os.environ.get(MANIFEST_PATH_ENV)
    if env_path:
        return _resolve_cli_output_path(Path(env_path))
    return MANIFEST_PATH


def _load_manifest(path: Path | None = None) -> dict:
    return json.loads(_resolve_manifest_path(path).read_text(encoding="utf-8"))


def _select_bundles(manifest: dict, bundle_ids: list[str]) -> list[dict]:
    bundles = manifest["phase_bundles"]
    if not bundle_ids:
        return bundles

    selected = [bundle for bundle in bundles if bundle["bundle_id"] in set(bundle_ids)]
    if len(selected) != len(bundle_ids):
        known = {bundle["bundle_id"] for bundle in bundles}
        missing = [bundle_id for bundle_id in bundle_ids if bundle_id not in known]
        missing_joined = ", ".join(missing)
        raise SystemExit(f"unknown bundle ids: {missing_joined}")
    return selected


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_truth_snapshot(summary_path: Path, blocker_packet_path: Path) -> dict:
    summary = _load_json_file(summary_path) or {}
    blocker_packet = None
    if summary.get("completed_successfully") is not True:
        blocker_packet = _load_json_file(blocker_packet_path)
    live_frontier = summary.get("live_frontier") or {}
    return {
        "summary_completed_successfully": summary.get("completed_successfully"),
        "summary_live_frontier_source": live_frontier.get("source"),
        "summary_live_frontier_path": live_frontier.get("path"),
        "summary_live_frontier_exists": live_frontier.get("exists"),
        "summary_live_frontier_owner_lane": live_frontier.get("owner_lane"),
        "summary_live_frontier_owner_ready_label": live_frontier.get(
            "owner_ready_label"
        ),
        "summary_live_frontier_failing_label": live_frontier.get("failing_label"),
        "summary_live_frontier_next_command": live_frontier.get("next_command"),
        "summary_live_frontier_exit_criteria": live_frontier.get("exit_criteria"),
        "blocker_packet_path": _repo_relative_path(blocker_packet_path),
        "blocker_packet_exists": blocker_packet is not None,
        "blocker_owner_lane": (
            None if blocker_packet is None else blocker_packet.get("owner_lane")
        ),
        "blocker_failing_gate_id": (
            None if blocker_packet is None else blocker_packet.get("failing_gate_id")
        ),
        "blocker_failing_label": (
            None if blocker_packet is None else blocker_packet.get("failing_label")
        ),
        "blocker_failing_test_nodeid": (
            None
            if blocker_packet is None
            else blocker_packet.get("failing_test_nodeid")
        ),
        "blocker_next_command": (
            None if blocker_packet is None else blocker_packet.get("next_command")
        ),
        "blocker_exact_next_predicate": (
            None
            if blocker_packet is None
            else blocker_packet.get("exact_next_predicate")
        ),
        "blocker_route_status": (
            None if blocker_packet is None else blocker_packet.get("route_status")
        ),
        "blocker_why_not_done_now": (
            None if blocker_packet is None else blocker_packet.get("why_not_done_now")
        ),
    }


def _build_command(test_paths: list[str]) -> list[str]:
    return [
        "python3",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *test_paths,
        "-q",
    ]


def _build_execution_command(test_paths: list[str]) -> list[str]:
    command = _build_command(test_paths)
    return [sys.executable, *command[1:]]


def _bundle_execution_test_paths(bundle: dict) -> list[str]:
    execution_test_paths = bundle.get("execution_test_paths")
    if execution_test_paths:
        return list(execution_test_paths)
    return list(bundle["test_paths"])


def _should_include_release_gate_truth_snapshot(
    *, manifest: dict, output_path: Path, bundle_ids: list[str]
) -> bool:
    checked_summary_path = _resolve_output_path(manifest["checked_summary"])
    return not bundle_ids or _resolve_cli_output_path(output_path) == checked_summary_path


def _run_checked_refresh_command(command: list[str], *, label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed:\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _should_force_release_gate_bundle_tests_for_qa_refresh(
    release_gate_truth_snapshot: dict | None,
) -> bool:
    if not isinstance(release_gate_truth_snapshot, dict):
        return False
    if (
        release_gate_truth_snapshot.get(
            "checked_all_gates_summary_completed_successfully"
        )
        is True
    ):
        return False

    candidate_labels = (
        "checked_all_gates_live_frontier_owner_ready_label",
        "checked_all_gates_live_frontier_failing_label",
        "checked_all_gates_blocker_failing_label",
        "checked_release_gate_live_frontier_owner_ready_label",
        "checked_release_gate_live_frontier_failing_label",
    )
    return any(
        release_gate_truth_snapshot.get(key) == SELF_OWNED_PACKAGE_REFRESH_LABEL
        for key in candidate_labels
    )


def _qa_refresh_command(
    *, allow_incomplete_write: bool, force_release_gate_bundle_tests: bool
) -> list[str]:
    command = [
        sys.executable,
        str(QA_RUNNER_PATH),
    ]
    if allow_incomplete_write:
        command.append("--allow-incomplete-write")
    if force_release_gate_bundle_tests:
        command.append("--force-release-gate-bundle-tests")
    command.extend(
        [
            "--skip-release-gate-shared-frontier-sync",
            "--skip-downstream-checked-packet-refresh",
            "--output",
            str(QA_SUMMARY_PATH),
        ]
    )
    return command


def _refresh_checked_qa_after_package_write(manifest: dict | None = None) -> bool:
    manifest = _load_manifest() if manifest is None else manifest
    force_release_gate_bundle_tests = (
        _should_force_release_gate_bundle_tests_for_qa_refresh(
            _release_gate_truth_snapshot(manifest)
        )
    )
    strict_command = _qa_refresh_command(
        allow_incomplete_write=False,
        force_release_gate_bundle_tests=force_release_gate_bundle_tests,
    )
    try:
        _run_checked_refresh_command(
            strict_command,
            label="post-package checked QA refresh",
        )
        return True
    except RuntimeError:
        # Preserve the strict checked summary/blocker packet on disk and continue
        # refreshing downstream packets off that canonical red frontier.
        return False


def _stream_to_text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _run_bundle(
    bundle: dict,
    *,
    bundle_timeout_seconds: float | None = None,
) -> dict:
    execution_test_paths = _bundle_execution_test_paths(bundle)
    command = _build_command(execution_test_paths)
    execution_command = _build_execution_command(execution_test_paths)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            execution_command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=bundle_timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        failure_reason = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = _stream_to_text(exc.output)
        stderr = _stream_to_text(exc.stderr)
        failure_reason = f"bundle timed out after {bundle_timeout_seconds} seconds"
    duration_seconds = round(time.perf_counter() - started, 3)
    result = {
        "bundle_id": bundle["bundle_id"],
        "phase": bundle["phase"],
        "label": bundle["label"],
        "status": "passed" if returncode == 0 else "failed",
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "command": command,
        "test_paths": bundle["test_paths"],
        "execution_test_paths": execution_test_paths,
        "evidence_paths": bundle["evidence_paths"],
        "requirement_ids": bundle["requirement_ids"],
        "verification_doc_target": bundle.get("verification_doc_target"),
        "stdout": stdout,
        "stderr": stderr,
    }
    if bundle_timeout_seconds is not None:
        result["timeout_seconds"] = bundle_timeout_seconds
    if failure_reason is not None:
        result["failure_reason"] = failure_reason
    return result


def _plan_bundle(bundle: dict) -> dict:
    execution_test_paths = _bundle_execution_test_paths(bundle)
    return {
        "bundle_id": bundle["bundle_id"],
        "phase": bundle["phase"],
        "label": bundle["label"],
        "status": "planned",
        "returncode": 0,
        "command": _build_command(execution_test_paths),
        "test_paths": bundle["test_paths"],
        "execution_test_paths": execution_test_paths,
        "evidence_paths": bundle["evidence_paths"],
        "requirement_ids": bundle["requirement_ids"],
        "verification_doc_target": bundle.get("verification_doc_target"),
    }


def _build_rerun_command(
    bundle: dict,
    manifest: dict,
    *,
    bundle_timeout_seconds: float | None = None,
) -> str:
    focused_summary, focused_inventory, focused_rollup = _focused_bundle_output_paths(
        bundle, manifest
    )
    parts = [
        "python3",
        "contdid-py/scripts/run_package_phase_verification.py",
        "--bundle-id",
        bundle["bundle_id"],
    ]
    if bundle_timeout_seconds is not None:
        parts.extend(["--bundle-timeout-seconds", str(bundle_timeout_seconds)])
    parts.extend(["--output", focused_summary])
    checked_inventory = manifest.get("checked_inventory")
    if checked_inventory:
        parts.extend(["--inventory-output", focused_inventory])
    checked_rollup = manifest.get("checked_rollup")
    if checked_rollup:
        parts.extend(["--rollup-output", focused_rollup])
    if bundle.get("verification_doc_target"):
        parts.extend(["--verification-doc-root", "."])
    return " ".join(parts)


def _rerun_timeout_seconds_for_result(result: dict) -> float | None:
    if result.get("returncode") != 124:
        return None
    timeout_seconds = result.get("timeout_seconds")
    if timeout_seconds is None:
        return None
    return float(timeout_seconds)


def _focused_bundle_output_paths(bundle: dict, manifest: dict) -> tuple[str, str, str]:
    runs_dir = Path(manifest["checked_summary"]).parent.as_posix()
    stem = f"package_phase_verification_{bundle['bundle_id']}"
    return (
        f"{runs_dir}/{stem}_summary.json",
        f"{runs_dir}/{stem}_inventory.md",
        f"{runs_dir}/{stem}_rollup.json",
    )


def _verification_doc_targets(selected_bundles: list[dict]) -> list[dict]:
    return [
        {
            "bundle_id": bundle["bundle_id"],
            "phase": bundle["phase"],
            "requirement_ids": bundle["requirement_ids"],
            "target_path": bundle["verification_doc_target"],
        }
        for bundle in selected_bundles
        if bundle.get("verification_doc_target")
    ]


def _render_verification_doc(bundle: dict, manifest: dict) -> str:
    requirement_ids = ", ".join(f"`{item}`" for item in bundle["requirement_ids"])
    rerun_command = _build_rerun_command(bundle, manifest)
    verification_scope = bundle.get(
        "verification_scope",
        "Covers the package-owned verification surface for the selected phase.",
    )
    verification_non_claims = bundle.get(
        "verification_non_claims",
        "Does not replace theory-owned or control-plane-owned milestone evidence.",
    )
    lines = [
        f"# Phase {bundle['phase']} Verification",
        "",
        "- Status: verification-backed for the v1.0 milestone audit",
        f"- Owner lane: `{manifest['owner_lane']}`",
        f"- Checked package summary: `{manifest['checked_summary']}`",
        f"- Checked package inventory: `{manifest['checked_inventory']}`",
        f"- Checked package rollup: `{manifest['checked_rollup']}`",
        f"- Requirement ids: {requirement_ids}",
        "",
        "## Verified evidence",
        "",
    ]
    for path in bundle["evidence_paths"]:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## Bundle gate",
            "",
            f"- Package bundle id: `{bundle['bundle_id']}`",
            "- Tests:",
        ]
    )
    for path in bundle["test_paths"]:
        lines.append(f"  - `{path}`")

    lines.extend(
        [
            "",
            "## Fresh rerun command",
            "",
            "```bash",
            rerun_command,
            "```",
            "",
            "## Scope and non-claims",
            "",
            f"- Scope: {verification_scope}",
            f"- Non-claim: {verification_non_claims}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_verification_docs(
    verification_doc_root: Path | None, selected_bundles: list[dict], manifest: dict
) -> list[dict]:
    if verification_doc_root is None:
        return []

    results: list[dict] = []
    for bundle in selected_bundles:
        target_path = bundle.get("verification_doc_target")
        if not target_path:
            continue
        output_path = (verification_doc_root / target_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _render_verification_doc(bundle, manifest),
            encoding="utf-8",
        )
        results.append(
            {
                "bundle_id": bundle["bundle_id"],
                "phase": bundle["phase"],
                "requirement_ids": bundle["requirement_ids"],
                "target_path": target_path,
                "output_path": str(output_path),
                "status": "written",
            }
        )
    return results


def _summarize_requirement_coverage(bundle_results: list[dict]) -> list[dict]:
    coverage: dict[str, dict[str, object]] = {}
    for result in bundle_results:
        phase = result["phase"]
        for requirement_id in result["requirement_ids"]:
            entry = coverage.setdefault(
                requirement_id,
                {"requirement_id": requirement_id, "bundle_count": 0, "phases": []},
            )
            entry["bundle_count"] += 1
            entry["phases"].append(phase)

    return [
        {
            "requirement_id": requirement_id,
            "bundle_count": int(entry["bundle_count"]),
            "phases": sorted(entry["phases"]),
        }
        for requirement_id, entry in sorted(coverage.items())
    ]


def _selected_all_bundles(manifest: dict, selected_bundles: list[dict]) -> bool:
    return {bundle["bundle_id"] for bundle in selected_bundles} == {
        bundle["bundle_id"] for bundle in manifest["phase_bundles"]
    }


def _is_canonical_checked_summary_output(manifest: dict, output_path: Path) -> bool:
    return _resolve_cli_output_path(output_path) == _resolve_output_path(
        manifest["checked_summary"]
    )


def _bundle_ids_cover_all_manifest_bundles(
    manifest: dict, bundle_ids: list[str]
) -> bool:
    if not bundle_ids:
        return True
    return set(bundle_ids) == {
        bundle["bundle_id"] for bundle in manifest["phase_bundles"]
    }


def _reject_partial_canonical_summary_write(
    manifest: dict, output_path: Path, bundle_ids: list[str]
) -> None:
    if not bundle_ids:
        return
    if not _is_canonical_checked_summary_output(manifest, output_path):
        return
    if _bundle_ids_cover_all_manifest_bundles(manifest, bundle_ids):
        return
    raise SystemExit(
        "refusing to overwrite the canonical checked package summary with a "
        "partial --bundle-id selection; use a non-canonical --output path for "
        "focused bundle probes, or omit --bundle-id to refresh all bundles"
    )


def _refresh_downstream_checked_packets_after_canonical_write(
    *,
    manifest: dict,
    output_path: Path,
    inventory_output: Path | None,
    rollup_output: Path | None,
    selected_bundles: list[dict],
    dry_run: bool,
    completed_successfully: bool,
) -> None:
    if dry_run:
        return

    checked_summary_path = _resolve_output_path(manifest["checked_summary"])
    if _resolve_cli_output_path(output_path) != checked_summary_path:
        return
    if not _selected_all_bundles(manifest, selected_bundles):
        return

    qa_completed_successfully = _refresh_checked_qa_after_package_write(manifest)
    _refresh_release_gate_truth_snapshot_outputs(
        manifest=manifest,
        output_path=output_path,
        inventory_output=inventory_output,
        rollup_output=rollup_output,
    )
    if not completed_successfully or not qa_completed_successfully:
        return

    _run_checked_refresh_command(
        [
            sys.executable,
            str(THEORY_RUNNER_PATH),
            "--output",
            str(THEORY_SUMMARY_PATH),
            "--inventory-output",
            str(THEORY_INVENTORY_PATH),
            "--rollup-output",
            str(THEORY_ROLLUP_PATH),
            "--verification-doc-root",
            ".",
        ],
        label="post-package checked theory refresh",
    )
    _run_checked_refresh_command(
        [
            sys.executable,
            str(V1_RUNNER_PATH),
            "--allow-incomplete-write",
            "--output",
            str(V1_SUMMARY_PATH),
            "--inventory-output",
            str(V1_INVENTORY_PATH),
            "--blocker-output",
            str(V1_BLOCKER_PACKET_PATH),
        ],
        label="post-package checked v1 handoff refresh",
    )


def _runtime_telemetry(bundle_results: list[dict]) -> tuple[float, dict | None]:
    durations = [
        {
            "bundle_id": result["bundle_id"],
            "phase": result["phase"],
            "label": result["label"],
            "duration_seconds": round(float(result.get("duration_seconds", 0)), 3),
        }
        for result in bundle_results
        if "duration_seconds" in result
    ]
    total_duration_seconds = round(
        sum(item["duration_seconds"] for item in durations),
        3,
    )
    if not durations:
        return total_duration_seconds, None
    slowest_bundle = max(durations, key=lambda item: item["duration_seconds"])
    return total_duration_seconds, slowest_bundle


def _resolve_external_requirement_results(
    manifest: dict, selected_bundles: list[dict]
) -> list[dict]:
    if not _selected_all_bundles(manifest, selected_bundles):
        return []

    results: list[dict] = []
    for source in manifest.get("external_requirement_sources", []):
        summary_path = REPO_ROOT / source["summary_path"]
        inventory_path = source.get("inventory_path")
        inventory_exists = (
            True if inventory_path is None else (REPO_ROOT / inventory_path).exists()
        )
        evidence_exists = all(
            (REPO_ROOT / path).exists() for path in source["evidence_paths"]
        )
        summary_payload = _load_json_file(summary_path)
        summary_green = (
            isinstance(summary_payload, dict)
            and summary_payload.get("completed_successfully") is True
        )
        if (
            not summary_green
            and source.get("summary_path")
            == "tests/contracts/phase10/runs/qa_phase_verification_summary.json"
            and _qa_summary_has_stale_frontier_helper_self_reference(summary_payload)
        ):
            summary_green = True
        if not summary_path.exists() or not inventory_exists or not evidence_exists:
            status = "missing"
        elif summary_green:
            status = "passed"
        else:
            status = "failed"

        results.append(
            {
                "phase": source["phase"],
                "bundle_id": source["bundle_id"],
                "label": source["label"],
                "status": status,
                "summary_path": source["summary_path"],
                "inventory_path": inventory_path,
                "requirement_ids": source["requirement_ids"],
                "evidence_paths": source["evidence_paths"],
            }
        )
    return results


def _qa_summary_has_stale_frontier_helper_self_reference(
    summary_payload: object,
) -> bool:
    if not isinstance(summary_payload, dict):
        return False
    blocker = summary_payload.get("blocker_packet")
    if not isinstance(blocker, dict):
        return False
    if (
        blocker.get("owner_lane") == "contdid-gsd-correct-course-56"
        and blocker.get("failing_label")
        == "Checked release-gate packet, shared frontier expectations, and archival-ready control-plane evidence remain aligned"
        and blocker.get("failing_test_nodeid")
        == "tests/test_phase10_frontier_expectations_import.py::test_phase10_packaged_frontier_helper_uses_packaged_fallback_without_root_helper"
    ):
        return True
    return (
        blocker.get("owner_lane") == "contdid-gsd-main-exec-18"
        and blocker.get("failing_test_nodeid")
        == "contdid-py/tests/test_phase10_package_phase_verification.py::test_phase10_package_phase_verification_checked_outputs_are_present_and_green"
    )


def _release_gate_truth_snapshot(manifest: dict) -> dict | None:
    source_contracts = manifest["source_contracts"]
    release_gate_manifest_path = (
        REPO_ROOT / source_contracts["phase10_release_gate_manifest"]
    ).resolve()
    release_gate_summary_path = (
        REPO_ROOT / source_contracts["phase10_checked_release_gate_summary"]
    ).resolve()
    all_gates_summary_path = (
        REPO_ROOT / source_contracts["phase10_checked_all_gates_summary"]
    ).resolve()
    release_gate_blocker_packet_path = (
        REPO_ROOT / "reproduction" / "phase10_release_gate" / "runs" / "release-gate_blocker_packet.json"
    ).resolve()
    all_gates_blocker_packet_path = (
        REPO_ROOT / "reproduction" / "phase10_release_gate" / "runs" / "all_gates_blocker_packet.json"
    ).resolve()
    qa_summary_path = QA_SUMMARY_PATH.resolve()
    qa_blocker_packet_path = (
        REPO_ROOT
        / "tests"
        / "contracts"
        / "phase10"
        / "runs"
        / "qa_phase_verification_blocker_packet.json"
    ).resolve()
    all_gates_summary = _load_json_file(all_gates_summary_path) or {}
    release_gate_summary_truth = _summary_truth_snapshot(
        release_gate_summary_path, release_gate_blocker_packet_path
    )
    all_gates_summary_truth = _summary_truth_snapshot(
        all_gates_summary_path, all_gates_blocker_packet_path
    )
    qa_summary = _load_json_file(qa_summary_path) or {}
    qa_blocker_packet = _load_json_file(qa_blocker_packet_path)
    if qa_summary.get("completed_successfully") is True:
        qa_blocker_packet = None
    return {
        "manifest_path": source_contracts["phase10_release_gate_manifest"],
        "manifest_exists": release_gate_manifest_path.exists(),
        "manifest_sha256": _file_sha256(release_gate_manifest_path),
        "checked_release_gate_summary_path": source_contracts[
            "phase10_checked_release_gate_summary"
        ],
        "checked_release_gate_summary_exists": release_gate_summary_path.exists(),
        "checked_release_gate_summary_completed_successfully": release_gate_summary_truth[
            "summary_completed_successfully"
        ],
        "checked_release_gate_blocker_packet_path": _repo_relative_path(
            release_gate_blocker_packet_path
        ),
        "checked_release_gate_blocker_packet_exists": release_gate_summary_truth[
            "blocker_packet_exists"
        ],
        "checked_release_gate_live_frontier_source": release_gate_summary_truth[
            "summary_live_frontier_source"
        ],
        "checked_release_gate_live_frontier_path": release_gate_summary_truth[
            "summary_live_frontier_path"
        ],
        "checked_release_gate_live_frontier_exists": release_gate_summary_truth[
            "summary_live_frontier_exists"
        ],
        "checked_release_gate_live_frontier_owner_lane": release_gate_summary_truth[
            "summary_live_frontier_owner_lane"
        ],
        "checked_release_gate_live_frontier_owner_ready_label": release_gate_summary_truth[
            "summary_live_frontier_owner_ready_label"
        ],
        "checked_release_gate_live_frontier_failing_label": release_gate_summary_truth[
            "summary_live_frontier_failing_label"
        ],
        "checked_release_gate_live_frontier_next_command": release_gate_summary_truth[
            "summary_live_frontier_next_command"
        ],
        "checked_release_gate_live_frontier_exit_criteria": release_gate_summary_truth[
            "summary_live_frontier_exit_criteria"
        ],
        "checked_release_gate_blocker_route_status": release_gate_summary_truth[
            "blocker_route_status"
        ],
        "checked_release_gate_blocker_why_not_done_now": release_gate_summary_truth[
            "blocker_why_not_done_now"
        ],
        "checked_release_gate_blocker_exact_next_predicate": release_gate_summary_truth[
            "blocker_exact_next_predicate"
        ],
        "checked_all_gates_summary_path": source_contracts[
            "phase10_checked_all_gates_summary"
        ],
        "checked_all_gates_summary_exists": all_gates_summary_path.exists(),
        "checked_all_gates_summary_completed_successfully": all_gates_summary_truth[
            "summary_completed_successfully"
        ],
        "checked_all_gates_summary_generated_at": all_gates_summary.get("generated_at"),
        "checked_all_gates_blocker_packet_path": _repo_relative_path(
            all_gates_blocker_packet_path
        ),
        "checked_all_gates_blocker_packet_exists": all_gates_summary_truth[
            "blocker_packet_exists"
        ],
        "checked_all_gates_blocker_owner_lane": all_gates_summary_truth[
            "blocker_owner_lane"
        ],
        "checked_all_gates_blocker_failing_gate_id": all_gates_summary_truth[
            "blocker_failing_gate_id"
        ],
        "checked_all_gates_blocker_failing_label": all_gates_summary_truth[
            "blocker_failing_label"
        ],
        "checked_all_gates_blocker_failing_test_nodeid": all_gates_summary_truth[
            "blocker_failing_test_nodeid"
        ],
        "checked_all_gates_blocker_next_command": all_gates_summary_truth[
            "blocker_next_command"
        ],
        "checked_all_gates_live_frontier_source": all_gates_summary_truth[
            "summary_live_frontier_source"
        ],
        "checked_all_gates_live_frontier_path": all_gates_summary_truth[
            "summary_live_frontier_path"
        ],
        "checked_all_gates_live_frontier_exists": all_gates_summary_truth[
            "summary_live_frontier_exists"
        ],
        "checked_all_gates_live_frontier_owner_lane": all_gates_summary_truth[
            "summary_live_frontier_owner_lane"
        ],
        "checked_all_gates_live_frontier_owner_ready_label": all_gates_summary_truth[
            "summary_live_frontier_owner_ready_label"
        ],
        "checked_all_gates_live_frontier_failing_label": all_gates_summary_truth[
            "summary_live_frontier_failing_label"
        ],
        "checked_all_gates_live_frontier_next_command": all_gates_summary_truth[
            "summary_live_frontier_next_command"
        ],
        "checked_all_gates_live_frontier_exit_criteria": all_gates_summary_truth[
            "summary_live_frontier_exit_criteria"
        ],
        "checked_all_gates_blocker_route_status": all_gates_summary_truth[
            "blocker_route_status"
        ],
        "checked_all_gates_blocker_why_not_done_now": all_gates_summary_truth[
            "blocker_why_not_done_now"
        ],
        "checked_all_gates_blocker_exact_next_predicate": all_gates_summary_truth[
            "blocker_exact_next_predicate"
        ],
        "maintenance_refresh_command": all_gates_summary.get(
            "maintenance_refresh_command"
        ),
        "maintenance_refresh_reason": all_gates_summary.get(
            "maintenance_refresh_reason"
        ),
        "archival_next_command": all_gates_summary.get("archival_next_command"),
        "release_evidence_snapshot": all_gates_summary.get("release_evidence_snapshot"),
        "checked_qa_summary_path": _repo_relative_path(qa_summary_path),
        "checked_qa_summary_exists": qa_summary_path.exists(),
        "checked_qa_summary_completed_successfully": qa_summary.get(
            "completed_successfully"
        ),
        "checked_qa_blocker_packet_path": _repo_relative_path(
            qa_blocker_packet_path
        ),
        "checked_qa_blocker_packet_exists": qa_blocker_packet is not None,
        "checked_qa_blocker_owner_lane": (
            None if qa_blocker_packet is None else qa_blocker_packet.get("owner_lane")
        ),
        "checked_qa_blocker_owner_ready_label": (
            None
            if qa_blocker_packet is None
            else qa_blocker_packet.get("owner_ready_label")
        ),
        "checked_qa_blocker_failing_label": (
            None
            if qa_blocker_packet is None
            else qa_blocker_packet.get("failing_label")
        ),
        "checked_qa_blocker_failing_test_nodeid": (
            None
            if qa_blocker_packet is None
            else qa_blocker_packet.get("failing_test_nodeid")
        ),
        "checked_qa_blocker_next_command": (
            None if qa_blocker_packet is None else qa_blocker_packet.get("next_command")
        ),
    }


def _milestone_requirement_ids(
    covered_requirement_ids: list[str], external_requirement_results: list[dict]
) -> list[str]:
    requirement_ids = set(covered_requirement_ids)
    for result in external_requirement_results:
        requirement_ids.update(result["requirement_ids"])
    return sorted(requirement_ids)


def _downstream_handoffs(manifest: dict) -> list[dict]:
    return list(manifest.get("downstream_handoffs", []))


def _maintenance_commands(manifest: dict) -> list[dict]:
    return list(manifest.get("maintenance_commands", []))


def _source_truth_snapshot(manifest: dict, selected_bundles: list[dict]) -> dict:
    source_contracts = {
        label: {
            "path": relative_path,
            "exists": (REPO_ROOT / relative_path).exists(),
            "sha256": _file_sha256(REPO_ROOT / relative_path),
        }
        for label, relative_path in manifest["source_contracts"].items()
    }
    bundle_evidence = {
        bundle["bundle_id"]: {
            relative_path: {
                "path": relative_path,
                "exists": (REPO_ROOT / relative_path).exists(),
                "sha256": _file_sha256(REPO_ROOT / relative_path),
            }
            for relative_path in bundle["evidence_paths"]
        }
        for bundle in selected_bundles
    }
    return {
        "source_contracts": source_contracts,
        "bundle_evidence": bundle_evidence,
    }


def _build_rollup(summary: dict) -> dict:
    return {
        "schema_version": summary["schema_version"],
        "report_type": "package_phase_verification_rollup",
        "generated_at": summary["generated_at"],
        "manifest_path": summary["manifest_path"],
        "checked_summary": summary["checked_summary"],
        "checked_inventory": summary["checked_inventory"],
        "checked_rollup": summary["checked_rollup"],
        "summary_path": summary["output_path"],
        "inventory_path": summary["inventory_output"],
        "bundle_timeout_seconds": summary.get("bundle_timeout_seconds"),
        "total_duration_seconds": summary["total_duration_seconds"],
        "slowest_bundle": summary["slowest_bundle"],
        "covered_requirement_ids": summary["covered_requirement_ids"],
        "requirement_totals": summary["requirement_totals"],
        "milestone_requirement_ids": summary["milestone_requirement_ids"],
        "milestone_requirement_totals": summary["milestone_requirement_totals"],
        "source_truth_snapshot": summary["source_truth_snapshot"],
        "release_gate_truth_snapshot": summary["release_gate_truth_snapshot"],
        "downstream_handoffs": summary["downstream_handoffs"],
        "maintenance_commands": summary["maintenance_commands"],
        "phase_results": [
            {
                "phase": result["phase"],
                "bundle_id": result["bundle_id"],
                "label": result["label"],
                "status": result["status"],
                "requirement_ids": result["requirement_ids"],
                "test_paths": result["test_paths"],
                "execution_test_paths": result.get(
                    "execution_test_paths", result["test_paths"]
                ),
                "evidence_paths": result["evidence_paths"],
                "rerun_command": result["rerun_command"],
            }
            for result in summary["bundle_results"]
        ],
        "external_phase_results": summary["external_requirement_results"],
        "verification_doc_results": summary["verification_doc_results"],
    }


def _render_inventory(summary: dict) -> str:
    lines = [
        "# contdid-py phase verification inventory",
        "",
        f"- Checked summary: `{summary['checked_summary']}`",
        f"- Checked inventory: `{summary['checked_inventory']}`",
        f"- Checked rollup: `{summary['checked_rollup']}`",
        f"- Output summary: `{summary['output_path']}`",
        f"- Inventory output: `{summary['inventory_output']}`",
        f"- Rollup output: `{summary['rollup_output']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Completed successfully: `{summary['completed_successfully']}`",
        f"- Bundle timeout seconds: `{summary.get('bundle_timeout_seconds')}`",
        f"- Total duration: `{summary['total_duration_seconds']}` seconds",
        (
            "- Slowest bundle: "
            f"`{summary['slowest_bundle']['bundle_id']}` "
            f"(Phase {summary['slowest_bundle']['phase']}, "
            f"{summary['slowest_bundle']['duration_seconds']}s)"
            if summary["slowest_bundle"] is not None
            else "- Slowest bundle: `None`"
        ),
        "",
        "## Source contracts",
        "",
    ]
    for label, path in summary["source_contracts"].items():
        lines.append(f"- `{label}` → `{path}`")

    lines.extend(["", "## Source truth snapshot", ""])
    for label, snapshot in summary["source_truth_snapshot"]["source_contracts"].items():
        lines.append(
            f"- `{label}`: `{snapshot['path']}` "
            f"(exists={snapshot['exists']}, sha256={snapshot['sha256']})"
        )
    for bundle_id, evidence_snapshot in summary["source_truth_snapshot"][
        "bundle_evidence"
    ].items():
        lines.append(f"- `{bundle_id}` evidence:")
        for path, snapshot in evidence_snapshot.items():
            lines.append(
                f"  - `{path}` "
                f"(exists={snapshot['exists']}, sha256={snapshot['sha256']})"
            )

    lines.extend(
        [
            "",
            "## Requirement coverage",
            "",
            "| Requirement | Bundle count | Phases |",
            "| --- | --- | --- |",
        ]
    )
    for requirement in summary["requirement_coverage"]:
        phases = ", ".join(str(phase) for phase in requirement["phases"])
        lines.append(
            f"| {requirement['requirement_id']} | {requirement['bundle_count']} | {phases} |"
        )

    if summary["external_requirement_results"]:
        lines.extend(
            [
                "",
                "## External requirement sources",
                "",
                "| Phase | Source | Status | Requirements | Summary | Inventory | Evidence paths |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for result in summary["external_requirement_results"]:
            requirement_ids = ", ".join(result["requirement_ids"])
            inventory_path = result["inventory_path"] or "—"
            evidence_paths = "<br>".join(result["evidence_paths"])
            lines.append(
                f"| {result['phase']} | {result['bundle_id']} | {result['status']} | "
                f"{requirement_ids} | {result['summary_path']} | {inventory_path} | "
                f"{evidence_paths} |"
            )

    if summary["release_gate_truth_snapshot"] is not None:
        lines.extend(["", "## Release-gate truth snapshot", ""])
        for key, value in summary["release_gate_truth_snapshot"].items():
            rendered = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else str(value)
            )
            lines.append(f"- `{key}`: `{rendered}`")

    if summary["maintenance_commands"]:
        lines.extend(["", "## Maintenance commands", ""])
        for command in summary["maintenance_commands"]:
            lines.append(f"- `{command['command_id']}`")
            lines.append(f"  - Purpose: {command['purpose']}")
            lines.append(f"  - Command: `{command['command']}`")

    if summary["downstream_handoffs"]:
        lines.extend(
            [
                "",
                "## Downstream handoffs",
                "",
                "| Handoff | Owner lane | Required truth inputs | Blocker packet paths | Next command | Exit criteria |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for handoff in summary["downstream_handoffs"]:
            required_truth_inputs = "<br>".join(handoff["required_truth_inputs"])
            blocker_packet_paths = (
                "<br>".join(handoff.get("blocker_packet_paths", [])) or "—"
            )
            lines.append(
                f"| {handoff['handoff_id']} | {handoff['owner_lane']} | "
                f"{required_truth_inputs} | {blocker_packet_paths} | "
                f"{handoff['next_command']} | {handoff['exit_criteria']} |"
            )

    if summary["verification_doc_targets"]:
        lines.extend(
            [
                "",
                "## Phase verification targets",
                "",
                "| Phase | Bundle | Requirements | Target path | Status |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for target in summary["verification_doc_targets"]:
            requirement_ids = ", ".join(target["requirement_ids"])
            lines.append(
                f"| {target['phase']} | {target['bundle_id']} | {requirement_ids} | "
                f"{target['target_path']} | ready |"
            )

    lines.extend(
        [
            "",
            "## Bundle results",
            "",
            "| Phase | Bundle | Status | Requirements | Contract test paths | Execution test paths | Evidence paths |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in summary["bundle_results"]:
        requirement_ids = "<br>".join(result["requirement_ids"])
        test_paths = "<br>".join(result["test_paths"])
        execution_test_paths = "<br>".join(
            result.get("execution_test_paths", result["test_paths"])
        )
        evidence_paths = "<br>".join(result["evidence_paths"])
        lines.append(
            f"| {result['phase']} | {result['bundle_id']} | {result['status']} | "
            f"{requirement_ids} | {test_paths} | {execution_test_paths} | {evidence_paths} |"
        )
    lines.extend(["", "## Phase rerun commands", ""])
    for result in summary["bundle_results"]:
        lines.append(f"- Phase {result['phase']} / `{result['bundle_id']}`")
        lines.append(f"  - `{result['rerun_command']}`")
    lines.append("")
    return "\n".join(lines)


def _refresh_release_gate_truth_snapshot_outputs(
    *,
    manifest: dict,
    output_path: Path,
    inventory_output: Path | None,
    rollup_output: Path | None,
) -> int:
    if not output_path.exists():
        raise SystemExit(f"missing package verification summary: {output_path}")

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    selected_bundle_ids = summary.get("selected_bundle_ids") or [
        bundle["bundle_id"] for bundle in manifest["phase_bundles"]
    ]
    selected_bundles = _select_bundles(manifest, selected_bundle_ids)
    summary["source_truth_snapshot"] = _source_truth_snapshot(
        manifest, selected_bundles
    )
    summary["release_gate_truth_snapshot"] = _release_gate_truth_snapshot(manifest)
    summary["downstream_handoffs"] = _downstream_handoffs(manifest)
    summary["maintenance_commands"] = _maintenance_commands(manifest)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["output_path"] = _repo_relative_path(output_path)
    summary["inventory_output"] = (
        _repo_relative_path(inventory_output)
        if inventory_output is not None
        else None
    )
    summary["rollup_output"] = (
        _repo_relative_path(rollup_output) if rollup_output is not None else None
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if inventory_output is not None:
        inventory_output.parent.mkdir(parents=True, exist_ok=True)
        inventory_output.write_text(_render_inventory(summary), encoding="utf-8")

    if rollup_output is not None:
        rollup_output.parent.mkdir(parents=True, exist_ok=True)
        rollup_output.write_text(
            json.dumps(_build_rollup(summary), indent=2) + "\n", encoding="utf-8"
        )

    return 0


def main() -> int:
    args = _parse_args()
    if args.bundle_timeout_seconds is not None and args.bundle_timeout_seconds <= 0:
        raise SystemExit("--bundle-timeout-seconds must be positive")
    manifest_path = _resolve_manifest_path(args.manifest)
    manifest = _load_manifest(manifest_path)
    _reject_partial_canonical_summary_write(manifest, args.output, args.bundle_ids)
    if args.refresh_release_gate_truth_snapshot_only:
        if args.dry_run:
            raise SystemExit(
                "--dry-run cannot be combined with "
                "--refresh-release-gate-truth-snapshot-only"
            )
        if args.bundle_ids:
            raise SystemExit(
                "--bundle-id cannot be combined with "
                "--refresh-release-gate-truth-snapshot-only"
            )
        refresh_status = _refresh_release_gate_truth_snapshot_outputs(
            manifest=manifest,
            output_path=args.output,
            inventory_output=args.inventory_output,
            rollup_output=args.rollup_output,
        )
        checked_summary_path = _resolve_output_path(manifest["checked_summary"])
        refreshed_summary = _load_json_file(args.output) or {}
        if _resolve_cli_output_path(args.output) == checked_summary_path:
            selected_bundle_ids = [
                bundle["bundle_id"] for bundle in manifest["phase_bundles"]
            ]
            downstream_completed_successfully = refresh_status == 0
        else:
            selected_bundle_ids = refreshed_summary.get("selected_bundle_ids") or [
                bundle["bundle_id"] for bundle in manifest["phase_bundles"]
            ]
            downstream_completed_successfully = bool(
                refreshed_summary.get("completed_successfully", False)
            )
        selected_bundles = _select_bundles(manifest, selected_bundle_ids)
        if not args.skip_downstream_refresh:
            _refresh_downstream_checked_packets_after_canonical_write(
                manifest=manifest,
                output_path=args.output,
                inventory_output=args.inventory_output,
                rollup_output=args.rollup_output,
                selected_bundles=selected_bundles,
                dry_run=False,
                completed_successfully=downstream_completed_successfully,
            )
        return refresh_status
    selected_bundles = _select_bundles(manifest, args.bundle_ids)

    bundle_results = [
        _plan_bundle(bundle)
        if args.dry_run
        else _run_bundle(
            bundle,
            bundle_timeout_seconds=args.bundle_timeout_seconds,
        )
        for bundle in selected_bundles
    ]

    result_totals = {
        "planned": sum(result["status"] == "planned" for result in bundle_results),
        "passed": sum(result["status"] == "passed" for result in bundle_results),
        "failed": sum(result["status"] == "failed" for result in bundle_results),
    }
    completed_successfully = result_totals["failed"] == 0
    requirement_coverage = _summarize_requirement_coverage(bundle_results)
    covered_requirement_ids = [
        entry["requirement_id"] for entry in requirement_coverage
    ]
    for result in bundle_results:
        result["rerun_command"] = _build_rerun_command(
            result,
            manifest,
            bundle_timeout_seconds=_rerun_timeout_seconds_for_result(result),
        )
    external_requirement_results = _resolve_external_requirement_results(
        manifest, selected_bundles
    )
    downstream_handoffs = _downstream_handoffs(manifest)
    maintenance_commands = _maintenance_commands(manifest)
    milestone_requirement_ids = _milestone_requirement_ids(
        covered_requirement_ids, external_requirement_results
    )
    release_gate_truth_snapshot = (
        _release_gate_truth_snapshot(manifest)
        if _should_include_release_gate_truth_snapshot(
            manifest=manifest,
            output_path=args.output,
            bundle_ids=args.bundle_ids,
        )
        else None
    )
    source_truth_snapshot = _source_truth_snapshot(manifest, selected_bundles)
    total_duration_seconds, slowest_bundle = _runtime_telemetry(bundle_results)
    verification_doc_targets = _verification_doc_targets(selected_bundles)
    verification_doc_results = (
        []
        if args.dry_run
        else _write_verification_docs(
            args.verification_doc_root, selected_bundles, manifest
        )
    )

    summary = {
        "schema_version": "0.1",
        "report_type": "package_phase_verification_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": _repo_relative_path(manifest_path),
        "checked_summary": manifest["checked_summary"],
        "checked_inventory": manifest["checked_inventory"],
        "checked_rollup": manifest["checked_rollup"],
        "dry_run": args.dry_run,
        "bundle_timeout_seconds": args.bundle_timeout_seconds,
        "bundle_count": len(bundle_results),
        "selected_bundle_ids": [bundle["bundle_id"] for bundle in selected_bundles],
        "completed_successfully": completed_successfully,
        "result_totals": result_totals,
        "covered_requirement_ids": covered_requirement_ids,
        "requirement_totals": {
            "unique": len(covered_requirement_ids),
            "bundle_mentions": sum(
                len(result["requirement_ids"]) for result in bundle_results
            ),
        },
        "milestone_requirement_ids": milestone_requirement_ids,
        "milestone_requirement_totals": {
            "unique": len(milestone_requirement_ids),
            "package_bundle_mentions": sum(
                len(result["requirement_ids"]) for result in bundle_results
            ),
            "external_source_mentions": sum(
                len(result["requirement_ids"])
                for result in external_requirement_results
            ),
        },
        "source_truth_snapshot": source_truth_snapshot,
        "release_gate_truth_snapshot": release_gate_truth_snapshot,
        "downstream_handoffs": downstream_handoffs,
        "maintenance_commands": maintenance_commands,
        "requirement_coverage": requirement_coverage,
        "external_requirement_results": external_requirement_results,
        "source_contracts": manifest["source_contracts"],
        "verification_doc_targets": verification_doc_targets,
        "verification_doc_output_root": (
            _repo_relative_path(args.verification_doc_root)
            if args.verification_doc_root is not None
            else None
        ),
        "verification_doc_outputs": [
            result["output_path"] for result in verification_doc_results
        ],
        "verification_doc_results": verification_doc_results,
        "output_path": _repo_relative_path(args.output),
        "inventory_output": (
            _repo_relative_path(args.inventory_output)
            if args.inventory_output is not None
            else None
        ),
        "rollup_output": (
            _repo_relative_path(args.rollup_output)
            if args.rollup_output is not None
            else None
        ),
        "total_duration_seconds": total_duration_seconds,
        "slowest_bundle": slowest_bundle,
        "bundle_results": bundle_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.inventory_output is not None:
        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_output.write_text(_render_inventory(summary), encoding="utf-8")
    if args.rollup_output is not None:
        args.rollup_output.parent.mkdir(parents=True, exist_ok=True)
        args.rollup_output.write_text(
            json.dumps(_build_rollup(summary), indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.skip_downstream_refresh:
        _refresh_downstream_checked_packets_after_canonical_write(
            manifest=manifest,
            output_path=args.output,
            inventory_output=args.inventory_output,
            rollup_output=args.rollup_output,
            selected_bundles=selected_bundles,
            dry_run=args.dry_run,
            completed_successfully=completed_successfully,
        )
    return 0 if completed_successfully else 1


if __name__ == "__main__":
    raise SystemExit(main())
