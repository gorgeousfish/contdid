from __future__ import annotations

import builtins
import json
from pathlib import Path


# Resolve the repository root, not the package directory.
REPO = Path(__file__).resolve().parents[3]
EXPORTS_CACHE_KEY = "_phase10_frontier_expectations_exports"


def _load_release_gate_state() -> tuple[dict[str, object], dict[str, object] | None]:
    summary = json.loads(
        (
            REPO / "reproduction" / "phase10_release_gate" / "runs" / "all_gates_summary.json"
        ).read_text(encoding="utf-8")
    )
    blocker_path = (
        REPO / "reproduction" / "phase10_release_gate" / "runs" / "all_gates_blocker_packet.json"
    )
    blocker = None
    if blocker_path.exists():
        blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
    return summary, blocker


def _load_qa_state() -> tuple[dict[str, object], dict[str, object] | None]:
    summary = json.loads(
        (
            REPO
            / "tests"
            / "contracts"
            / "phase10"
            / "runs"
            / "qa_phase_verification_summary.json"
        ).read_text(encoding="utf-8")
    )
    blocker_path = (
        REPO
        / "tests"
        / "contracts"
        / "phase10"
        / "runs"
        / "qa_phase_verification_blocker_packet.json"
    )
    blocker = None
    if blocker_path.exists():
        blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
    return summary, blocker


def _qa_frontier_snapshot_is_stale_for_v2_milestone_design(
    qa_summary: dict[str, object],
) -> bool:
    snapshot = qa_summary.get("phase10_planning_frontier_snapshot")
    if not isinstance(snapshot, dict):
        return False

    if not _v2_milestone_design_frontier_is_live():
        return False

    v2_frontier = _v2_milestone_design_frontier()
    return (
        snapshot.get("project_current_focus") != v2_frontier["project_focus"]
        or snapshot.get("state_current_focus") != v2_frontier["state_focus"]
        or snapshot.get("state_stopped_at") != v2_frontier["state_stopped_at"]
    )


def _qa_frontier_snapshot_is_current_for_v2_milestone_design(
    qa_summary: dict[str, object],
) -> bool:
    snapshot = qa_summary.get("phase10_planning_frontier_snapshot")
    if not isinstance(snapshot, dict):
        return False

    if not _v2_milestone_design_frontier_is_live():
        return False

    v2_frontier = _v2_milestone_design_frontier()
    return (
        snapshot.get("project_current_focus") == v2_frontier["project_focus"]
        and snapshot.get("state_current_focus") == v2_frontier["state_focus"]
        and snapshot.get("state_stopped_at") == v2_frontier["state_stopped_at"]
    )


def _lowercase_last_activity(last_activity: str) -> str:
    return last_activity.replace("Last Activity:", "Last activity:", 1)


def _summary_activity_date(summary: dict[str, object], default: str = "2026-04-13") -> str:
    generated_at = summary.get("generated_at")
    if not isinstance(generated_at, str) or len(generated_at) < 10:
        return default
    return generated_at[:10]


def _v2_milestone_design_frontier_is_live() -> bool:
    project_path = REPO / ".planning" / "PROJECT.md"
    state_path = REPO / ".planning" / "STATE.md"
    if not project_path.exists() or not state_path.exists():
        return False

    v2_frontier = _v2_milestone_design_frontier()
    project_text = project_path.read_text(encoding="utf-8")
    state_text = state_path.read_text(encoding="utf-8")
    return (
        v2_frontier["project_focus"] in project_text and v2_frontier["state_focus"] in state_text
    )


def _blocker_phrase(label: str) -> str:
    return label if label.endswith("blocker") else f"{label} blocker"


def _audit_repair_lane_slug(owner_lane: object) -> str | None:
    if not isinstance(owner_lane, str):
        return None
    prefix = "contdid-gsd-audit-repair-"
    if not owner_lane.startswith(prefix):
        return None
    suffix = owner_lane.removeprefix(prefix)
    if not suffix.isdigit():
        return None
    return f"lane{suffix}"


def _text_or_default(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


CHECKED_QA_RECOVERY_NEXT_COMMAND = (
    "python3 tests/run_phase10_qa_verification.py "
    "--force-release-gate-bundle-tests --output "
    "tests/contracts/phase10/runs/qa_phase_verification_summary.json && "
    "python3 reproduction/phase10_release_gate/run_release_gate.py --gate-id all "
    "--output-root reproduction/phase10_release_gate/runs"
)


def _frontier_next_command_from_qa_blocker(qa_blocker: dict[str, object]) -> str:
    next_command = str(qa_blocker.get("next_command") or CHECKED_QA_RECOVERY_NEXT_COMMAND)
    if "::" in next_command:
        return CHECKED_QA_RECOVERY_NEXT_COMMAND
    return next_command


def _post_v1_paused_lane_rows() -> dict[str, str]:
    paused_focus = "Historical v1 maintenance lane retained as paused evidence"
    return {
        "correct_course_row": (
            "| contdid-gsd-correct-course-56 | 56 | correct-course | "
            ".planning/**, automation/**, repo-wide drift repairs | "
            f"PAUSED | {paused_focus} |"
        ),
        "theory_row": (
            "| contdid-gsd-theory-parity-01 | 01 | theory-parity | "
            "paper specs, contracts, fidelity layer tests | "
            f"PAUSED | {paused_focus} |"
        ),
        "main_exec_row": (
            "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | "
            f"PAUSED | {paused_focus} |"
        ),
        "repro_row": (
            "| contdid-gsd-repro-data-33 | 33 | repro-data | "
            "repro fixtures, data scripts, e2e assets | "
            f"PAUSED | {paused_focus} |"
        ),
        "qa_row": (
            "| contdid-gsd-qa-mc-48 | 48 | qa-mc | "
            "tests, monte carlo harness, numerical contracts | "
            f"PAUSED | {paused_focus} |"
        ),
    }


def expected_post_v1_audit_repair_frontier() -> dict[str, str]:
    return {
        "project_focus": "**Current focus:** Post-v1 global audit & repair — three audit-repair automations are live",
        "project_context": "The v1 release-gate, package-verification, and gitless-archival packets remain frozen on disk as historical completion evidence, while the Phase 9 English release docs/examples still feed the live release-gate audit surface and the control plane now runs three identical audit-repair lanes at :13 / :41 / :49 for repo-wide bug hunting and direct repair.",
        "state_focus": "**Current focus:** Post-v1 global audit & repair — three audit-repair automations are live",
        "state_last_activity": "Last Activity: 2026-04-25 — switched the live automation topology to post-v1 global audit & repair",
        "state_stopped_at": "Stopped At: v1 completion evidence remains frozen on disk; live automation now keeps sweeping repo-wide audit buckets instead of maintenance-only reruns",
        "state_pending_todo": "持续轮转 `core estimation`、`event-study / inference`、`empirical / reproduction`、`packaging / release`、`automation / control-plane` 五个审查桶",
        "active_objective": "- Active objective: 启用 3 条同构 audit-repair lanes 做全链路审查与修复，优先清理 numerical / identification / inference bug",
        "live_frontier_note": "- Live frontier note: the v1 release-gate, package-verification, and gitless-archival packets remain frozen as historical completion evidence on disk, while the live control plane now runs three identical audit-repair lanes at `:13 / :41 / :49`; packet refresh and archival reruns are secondary paths, not the default target.",
        "qa_live_root_note": "- QA/live-root note: the frozen v1 evidence remains available for reference, but live progress now comes from the three audit-repair lanes sweeping repo-wide buckets instead of the retired maintenance-only handoff chain.",
        "next_command": "start the next audit-repair sweep from the lane's assigned bucket offset; repair the highest-priority numerical / identification / inference issue found, and only rerun maintenance packets if that repair truly depends on refreshed evidence",
        "audit_row_13": "| contdid-gsd-audit-repair-13 | 13 | audit-repair | repo-wide audit + repair within /Users/cxy/Desktop/2026project/contdid | ACTIVE | Post-v1 audit lane; start from `core estimation` and keep closing adjacent same-line issues |",
        "audit_row_41": "| contdid-gsd-audit-repair-41 | 41 | audit-repair | repo-wide audit + repair within /Users/cxy/Desktop/2026project/contdid | ACTIVE | Post-v1 audit lane; start from `event-study / inference` and keep closing adjacent same-line issues |",
        "audit_row_49": "| contdid-gsd-audit-repair-49 | 49 | audit-repair | repo-wide audit + repair within /Users/cxy/Desktop/2026project/contdid | ACTIVE | Post-v1 audit lane; current sweep is `packaging / release` on the timeout-safe checked release-evidence verifier |",
        "paused_repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | PAUSED | Historical v1 maintenance lane retained as paused evidence |",
    }


def expected_v2_milestone_design_frontier() -> dict[str, object]:
    return {
        "project_focus": "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones",
        "project_active_item": "- [ ] 设计并验证 v2.0 后续里程碑 ladder：staggered-adoption CCK hard-fail boundary、data-shape generalization、performance backend、reporting/teaching、audit/repair",
        "project_audit_repair_item": "- [ ] 持续轮转 `core estimation`、`event-study / inference`、`empirical / reproduction`、`packaging / release`、`automation / control-plane` 五个审查桶，直到 v2.4 audit/repair milestone 接手",
        "state_focus": "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones",
        "state_last_activity": "Last Activity: 2026-05-06 — started v2.0 milestone design",
        "state_position_last_activity": "Last activity: 2026-05-06 — started v2.0 milestone design",
        "state_stopped_at": "Stopped At: v1 completion evidence remains frozen on disk; v2 milestone design now defines the next feature ladder",
        "state_status_line": "Status: Defining requirements",
        "state_progress_line": "Progress: [░░░░░░░░░░] 0%",
        "state_snapshot": {
            "current_phase": "Not started",
            "current_phase_name": "v2.0 milestone design",
            "total_phases": 15,
            "current_plan": "—",
            "total_plans_in_phase": None,
            "status": "Defining requirements",
        },
    }


def _post_v1_audit_repair_frontier_with_paused_rows() -> dict[str, str]:
    frontier = expected_post_v1_audit_repair_frontier()
    paused_rows = _post_v1_paused_lane_rows()
    return {
        "project_focus": frontier["project_focus"],
        "project_context": frontier["project_context"],
        "state_focus": frontier["state_focus"],
        "state_last_activity": frontier["state_last_activity"],
        "state_stopped_at": frontier["state_stopped_at"],
        "state_pending_todo": frontier["state_pending_todo"],
        "active_objective": frontier["active_objective"],
        "qa_live_root_note": frontier["qa_live_root_note"],
        "correct_course_row": paused_rows["correct_course_row"],
        "theory_row": paused_rows["theory_row"],
        "main_exec_row": paused_rows["main_exec_row"],
        "repro_row": paused_rows["repro_row"],
        "qa_row": paused_rows["qa_row"],
        "next_command": frontier["next_command"],
    }


def _v2_milestone_design_frontier() -> dict[str, str]:
    frontier = expected_post_v1_audit_repair_frontier()
    frontier.update(_post_v1_audit_repair_frontier_with_paused_rows())
    frontier.update(
        {
            "project_focus": "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones",
            "project_context": "在 v1.0 可信发布面之上，把 staggered-adoption 下的 CCK/data-driven 路由收束为 machine-checkable unsupported hard-fail boundary，同时继续推进规模化能力和用户表达层，并把审查/修复作为独立里程碑纳入后续路线。",
            "state_focus": "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones",
            "state_last_activity": "Last Activity: 2026-05-06 — started v2.0 milestone design",
            "state_stopped_at": "Stopped At: v1 completion evidence remains frozen on disk; v2 milestone design now defines the next feature ladder",
            "state_pending_todo": "设计并验证 v2.0 后续里程碑 ladder：staggered-adoption CCK hard-fail boundary、data-shape generalization、performance backend、reporting/teaching、audit/repair",
        }
    )
    return frontier


def expected_v1_archival_frontier() -> dict[str, str]:
    return {
        "project_focus": "**Current focus:** Phase 10 release gate closed; v1.0 gitless milestone snapshot materialized / maintenance-refresh only",
        "project_context": "Phase 10 release gate is closed with checked lane-audit, numerical-audit, release-gate, and all-gates summaries.",
        "state_focus": "**Current focus:** Phase 10 closed — release gate passed; v1.0 gitless milestone snapshot materialized / maintenance-refresh only",
        "state_last_activity": "Last Activity: 2026-04-13 — refreshed gitless milestone snapshot maintenance truth",
        "state_stopped_at": "Stopped At: Gitless milestone snapshot materialized; rerun archival only if source docs drift",
        "state_pending_todo": "仅在 source docs 漂移或需要刷新 snapshots 时重跑 gitless milestone archival runner",
        "active_objective": "- Active objective: 保持 v1.0 gitless milestone snapshot truth，仅在 source docs 漂移时重跑 archival",
        "qa_live_root_note": "- QA/live-root note: the refreshed checked QA packet and the integrated package→theory/v1 refresh route are both on disk; the checked root is green again (`lane-audit`, `numerical-audit`, `release-gate`, and `all-gates` summaries all report `completed_successfully=true`) and no release-gate blocker packet remains on disk.",
        "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 closed; gitless milestone snapshot materialized / maintenance-refresh only |",
        "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
        "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; package handoff blocker paths are frozen and the checked root rerun is green |",
        "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors stay green on the maintenance-only checked root |",
        "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 closed; QA verification bundle now covers phases 8-10 for v1.0 audit backfill |",
        "next_command": "python3 automation/scripts/run_gitless_milestone_archival.py --milestone v1.0 --output-root .planning/milestones",
    }


def expected_v1_archival_frontier_addenda() -> dict[str, str]:
    return {
        "state_position_last_activity": "Last activity: 2026-04-13 — refreshed gitless milestone snapshot maintenance truth",
        "live_frontier_note": "- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is green, no release-gate blocker packet remains on disk, and the checked summary itself now exposes `maintenance_refresh_command` (`python3 reproduction/phase10_release_gate/run_release_gate.py --refresh-shared-frontier-only --output-root reproduction/phase10_release_gate/runs`) plus `maintenance_refresh_reason` (`Refresh shared frontier mirrors without replaying gate command groups while the checked root stays green.`) for mirror-only drift, while `archival_next_command` (`python3 automation/scripts/run_gitless_milestone_archival.py --milestone v1.0 --output-root .planning/milestones`) remains the canonical gitless snapshot path.",
        "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` whenever a blocker reopens, but the current frontier is archival-ready and blocker-free.",
        "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green, and no downstream main-exec rerun owns the live checked root while the archival snapshot remains current.",
        "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: the refreshed checked root stays green, so no live blocker handoff is pending before archival reruns.",
    }


def expected_phase10_frontier_addenda() -> dict[str, str]:
    summary, blocker = _load_release_gate_state()
    qa_summary, qa_blocker = _load_qa_state()
    qa_frontier_snapshot_stale = _qa_frontier_snapshot_is_stale_for_v2_milestone_design(qa_summary)
    qa_frontier_snapshot_v2_current = _qa_frontier_snapshot_is_current_for_v2_milestone_design(
        qa_summary
    )
    frontier = expected_phase10_frontier()
    exact_v1_test = (
        "tests/test_phase10_v1_audit_blocker_handoff.py::"
        "test_v1_audit_blocker_handoff_checked_outputs_match_archival_ready_frontier"
    )

    addenda = {
        "state_position_last_activity": _lowercase_last_activity(frontier["state_last_activity"])
    }

    if (
        summary["completed_successfully"] is True
        and not qa_frontier_snapshot_stale
        and not qa_frontier_snapshot_v2_current
        and qa_summary.get("completed_successfully") is not True
        and qa_blocker is not None
    ):
        addenda.update(
            {
                "live_frontier_note": "- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` remains green, but `tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json` is the canonical live frontier and keeps archival parked behind the checked QA blocker.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` from the canonical checked-root blocker packet, so archival-facing consumers can replay the latest checked QA blocker exactly.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked while the checked QA blocker owns the live control-plane follow-up.",
                "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: even though the checked root stays green, the checked QA blocker packet is now the canonical live frontier; follow its exact `next_command` before archival reruns.",
            }
        )
        return addenda

    if summary["completed_successfully"] is True:
        live_frontier = summary.get("live_frontier", {})
        maintenance_refresh_command = (
            live_frontier.get("maintenance_refresh_command")
            if isinstance(live_frontier, dict)
            else None
        )
        maintenance_refresh_reason = (
            live_frontier.get("maintenance_refresh_reason")
            if isinstance(live_frontier, dict)
            else None
        )
        archival_next_command = (
            live_frontier.get("archival_next_command") if isinstance(live_frontier, dict) else None
        )
        if (
            not isinstance(maintenance_refresh_command, str)
            or not isinstance(maintenance_refresh_reason, str)
            or not isinstance(archival_next_command, str)
        ):
            maintenance_refresh_command = (
                "python3 reproduction/phase10_release_gate/run_release_gate.py "
                "--refresh-shared-frontier-only --output-root "
                "reproduction/phase10_release_gate/runs"
            )
            maintenance_refresh_reason = (
                "Refresh shared frontier mirrors without replaying gate command groups "
                "while the checked root stays green."
            )
            archival_next_command = (
                "python3 automation/scripts/run_gitless_milestone_archival.py "
                "--milestone v1.0 --output-root .planning/milestones"
            )
        addenda.update(
            {
                "live_frontier_note": (
                    "- Live frontier note: "
                    "`reproduction/phase10_release_gate/runs/all_gates_summary.json` "
                    "is green, no release-gate blocker packet remains on disk, and "
                    "the checked summary itself now exposes "
                    f"`maintenance_refresh_command` "
                    f"(`{maintenance_refresh_command}`) plus "
                    f"`maintenance_refresh_reason` "
                    f"(`{maintenance_refresh_reason}`) for mirror-only drift, while "
                    f"`archival_next_command` (`{archival_next_command}`) remains "
                    "the canonical gitless snapshot path."
                ),
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` whenever a blocker reopens, but the current frontier is archival-ready and blocker-free.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green, and no downstream main-exec rerun owns the live checked root while the archival snapshot remains current.",
                "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: the refreshed checked root stays green, so no live blocker handoff is pending before archival reruns.",
            }
        )
        return addenda

    assert blocker is not None
    owner_lane = str(blocker["owner_lane"])
    failing_test = _text_or_default(blocker.get("failing_test_nodeid"), exact_v1_test)

    if owner_lane == "contdid-gsd-correct-course-56":
        addenda.update(
            {
                "live_frontier_note": "- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a correct-course-owned lane-audit/control-plane blocker, `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and downstream checked QA/v1 packets still need the control-plane route to settle behind that live blocker.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory now mirror `failing_gate_id=release-gate` and the exact `failing_test_nodeid` from `reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json`, so archival-facing consumers can follow the current correct-course-owned lane-audit predicate without inferring it from the broader failing command alone.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked while the current correct-course-owned lane-audit blocker owns the live checked root.",
                "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the correct-course-owned lane-audit/control-plane blocker; the owner-ready packet keeps the exact archival handoff predicate so release-evidence consumers can follow the live correct-course route without guessing from older maintenance prose.",
            }
        )
        return addenda

    if owner_lane == "contdid-gsd-theory-parity-01":
        if failing_test == exact_v1_test:
            addenda.update(
                {
                    "live_frontier_note": f"- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a theory-owned v1 handoff blocker (`{failing_test}`), `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and downstream checked QA/v1 packets now need the theory/v1/QA rerun chain to settle behind that live blocker.",
                    "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory now mirror `failing_gate_id=release-gate` and the exact `failing_test_nodeid` from `reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json`, so archival-facing consumers can follow the current theory-owned v1 handoff predicate without inferring it from the broader failing command alone.",
                    "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked while the theory v1 handoff rerun owns the live checked root.",
                    "qa_live_blocker_handoff_note": f"- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the theory-owned v1 handoff blocker; the owner-ready packet keeps the exact `{failing_test}` predicate so release-evidence consumers can follow the live theory route without guessing from older maintenance prose.",
                }
            )
            return addenda

        addenda.update(
            {
                "live_frontier_note": f"- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a theory-owned release-gate blocker (`{failing_test}`), `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and downstream checked QA/v1 packets now need the theory/v1/QA rerun chain to settle behind that live blocker.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory now mirror `failing_gate_id=release-gate` and the exact `failing_test_nodeid` from `reproduction/phase10_release_gate/runs/all_gates_blocker_packet.json`, so archival-facing consumers can follow the current theory-owned release-gate predicate without inferring it from the broader failing command alone.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked while the theory rerun owns the live checked root.",
                "qa_live_blocker_handoff_note": f"- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the theory-owned full-suite blocker; the owner-ready packet keeps the exact `{failing_test}` predicate so release-evidence consumers can follow the live theory route without guessing from older maintenance prose.",
            }
        )
        return addenda

    if owner_lane == "contdid-gsd-qa-mc-48":
        failing_gate_id = str(blocker.get("failing_gate_id", ""))
        failing_label = _text_or_default(blocker.get("failing_label"), "full regression suite")
        if failing_gate_id == "numerical-audit":
            addenda.update(
                {
                    "live_frontier_note": f"- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a QA-owned numerical-audit blocker (`{failing_label}`), `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and the checked v1 handoff packet now mirrors that owner-ready QA blocker route.",
                    "theory_exact_blocker_identity_note": f"- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory now freeze `failing_gate_id=numerical-audit` and the exact `{failing_test}` predicate from the canonical checked-root blocker packet, so archival-facing consumers can replay the current QA-owned numerical blocker exactly.",
                    "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked behind the QA-owned numerical-audit blocker.",
                    "qa_live_blocker_handoff_note": f"- QA live-blocker handoff note: the latest checked root rerun now stops at the QA-owned `{failing_label}` blocker in `numerical-audit`; rerun the QA-owned numerical surface via the blocker packet's `next_command` before archival resumes.",
                }
            )
            return addenda
        addenda.update(
            {
                "live_frontier_note": "- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a QA-owned full regression-suite blocker, `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and the checked QA packet is now the canonical owner-ready route for the live frontier.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` from the canonical checked-root blocker packet, so archival-facing consumers can replay the current QA-owned full-suite predicate exactly.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked behind the QA-owned full regression suite blocker.",
                "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the QA-owned full regression suite blocker; rerun the checked QA packet before archival resumes.",
            }
        )
        return addenda

    if owner_lane == "contdid-gsd-repro-data-33":
        addenda.update(
            {
                "live_frontier_note": "- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a repro-data-owned release-evidence blocker, `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and the focused repro verifier now owns the live frontier handoff.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` from the canonical checked-root blocker packet, so archival-facing consumers can replay the current repro-data-owned predicate exactly.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the checked package packet stays green; main-exec rerun remains parked while the repro-data release-evidence blocker owns the live checked root.",
                "qa_live_blocker_handoff_note": "- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the repro-data release-evidence blocker; rerun the focused repro verifier before archival resumes.",
            }
        )
        return addenda

    if owner_lane == "contdid-gsd-audit-repair-49":
        lane49_label = str(
            blocker.get("failing_label", "refresh checked package verification packet")
        )
        addenda.update(
            {
                "live_frontier_note": f"- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a post-v1 audit-repair lane49 `{lane49_label}` blocker, `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and the checked root now hands back to the live packaging / release audit lane instead of the historical maintenance chain.",
                "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` from the canonical checked-root blocker packet, so archival-facing consumers can replay the current lane49-owned predicate exactly.",
                "main_exec_handoff_note": "- Main-exec handoff addendum: the historical main-exec lane remains paused as evidence only; the live package release blocker now belongs to the post-v1 audit-repair lane49 route.",
                "qa_live_blocker_handoff_note": f"- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the lane49-owned `{lane49_label}` blocker; rerun the checked root directly before archival resumes.",
            }
        )
        return addenda

    assert owner_lane == "contdid-gsd-main-exec-18"
    main_exec_label = str(blocker.get("failing_label", "refresh checked QA verification packet"))
    addenda.update(
        {
            "live_frontier_note": f"- Live frontier note: `reproduction/phase10_release_gate/runs/all_gates_summary.json` is currently red on a main-exec-owned `{main_exec_label}` blocker, `contdid-py/contracts/phase10/runs/package_phase_verification_summary.json` remains green, and the checked root now hands off to the package write surface.",
            "theory_exact_blocker_identity_note": "- Theory exact-blocker-identity addendum: the checked v1 handoff blocker packet plus its inventory still freeze `failing_gate_id` and `failing_test_nodeid` from the canonical checked-root blocker packet, so archival-facing consumers can replay the current main-exec-owned predicate exactly.",
            "main_exec_handoff_note": f"- Main-exec handoff addendum: the checked package packet stays green, but the live checked root is now blocked on the main-exec-owned `{main_exec_label}` rerun.",
            "qa_live_blocker_handoff_note": f"- QA live-blocker handoff note: the latest checked root rerun now clears lane-audit and numerical-audit, then stops at the main-exec-owned `{main_exec_label}` blocker; rerun 10-03 plus the full release gate before archival resumes.",
        }
    )
    return addenda


def expected_phase10_frontier() -> dict[str, str]:
    summary, blocker = _load_release_gate_state()
    qa_summary, qa_blocker = _load_qa_state()
    qa_frontier_snapshot_stale = _qa_frontier_snapshot_is_stale_for_v2_milestone_design(qa_summary)
    qa_frontier_snapshot_v2_current = _qa_frontier_snapshot_is_current_for_v2_milestone_design(
        qa_summary
    )
    if (
        summary["completed_successfully"] is True
        and not qa_frontier_snapshot_stale
        and not qa_frontier_snapshot_v2_current
        and qa_summary.get("completed_successfully") is not True
        and qa_blocker is not None
    ):
        audit_repair_lane_slug = _audit_repair_lane_slug(qa_blocker.get("owner_lane"))
        if audit_repair_lane_slug is not None:
            audit_repair_label = _text_or_default(
                qa_blocker.get("owner_ready_label"),
                _text_or_default(
                    qa_blocker.get("failing_label"),
                    "audit-repair checked QA packet",
                ),
            )
            audit_repair_activity_date = _summary_activity_date(summary)
            paused_rows = _post_v1_paused_lane_rows()
            return {
                "project_focus": f"**Current focus:** Phase 10 checked QA blocker is live; {audit_repair_lane_slug}-owned {audit_repair_label} route is live / rerun checked root",
                "project_context": f"Phase 10 all-gates summary is still green, but the checked QA packet has reopened on a {audit_repair_lane_slug}-owned `{audit_repair_label}` blocker; follow the QA blocker packet's next command and keep archival parked until that blocker disappears.",
                "state_focus": f"**Current focus:** Phase 10 reopened — checked QA blocker is live; {audit_repair_lane_slug}-owned {audit_repair_label} route is live / rerun checked root",
                "state_last_activity": f"Last Activity: {audit_repair_activity_date} — checked QA packet reopened on a {audit_repair_lane_slug}-owned {audit_repair_label} blocker",
                "state_stopped_at": f"Stopped At: Checked QA packet is red while all-gates remains green; the {audit_repair_lane_slug}-owned {audit_repair_label} route is live until the blocker clears",
                "state_pending_todo": f"{audit_repair_lane_slug} 清掉 checked QA blocker 后再决定是否重跑 gitless milestone archival runner",
                "active_objective": f"- Active objective: 对齐 checked QA blocker frontier到 {audit_repair_lane_slug}-owned {audit_repair_label} route，并在 blocker 清空前暂停 archival rerun",
                "qa_live_root_note": f"- QA/live-root note: even though the checked `all-gates` summary is still green, `tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json` is now the canonical live frontier; follow the {audit_repair_lane_slug}-owned checked QA blocker route, replay the checked root, and only resume archival once the QA blocker packet disappears.",
                "correct_course_row": paused_rows["correct_course_row"],
                "theory_row": paused_rows["theory_row"],
                "main_exec_row": paused_rows["main_exec_row"],
                "repro_row": paused_rows["repro_row"],
                "qa_row": paused_rows["qa_row"],
                "next_command": _frontier_next_command_from_qa_blocker(qa_blocker),
            }
        if qa_blocker.get("owner_lane") == "contdid-gsd-repro-data-33":
            repro_label = _text_or_default(
                qa_blocker.get("owner_ready_label"),
                _text_or_default(
                    qa_blocker.get("failing_label"),
                    "release-evidence verifier",
                ),
            )
            repro_activity_date = _summary_activity_date(summary)
            paused_rows = _post_v1_paused_lane_rows()
            return {
                "project_focus": f"**Current focus:** Phase 10 checked QA blocker is live; repro-data-owned {repro_label} route is live / rerun checked root",
                "project_context": f"Phase 10 all-gates summary is still green, but the checked QA packet has reopened on a repro-data-owned `{repro_label}` blocker; follow the QA blocker packet's next command and keep archival parked until that blocker disappears.",
                "state_focus": f"**Current focus:** Phase 10 reopened — checked QA blocker is live; repro-data-owned {repro_label} route is live / rerun checked root",
                "state_last_activity": f"Last Activity: {repro_activity_date} — checked QA packet reopened on a repro-data-owned {repro_label} blocker",
                "state_stopped_at": f"Stopped At: Checked QA packet is red while all-gates remains green; the repro-data-owned {repro_label} route is live until the blocker clears",
                "state_pending_todo": "repro-data 清掉 checked QA blocker 后再决定是否重跑 gitless milestone archival runner",
                "active_objective": f"- Active objective: 对齐 checked QA blocker frontier到 repro-data-owned {repro_label} route，并在 blocker 清空前暂停 archival rerun",
                "qa_live_root_note": "- QA/live-root note: even though the checked `all-gates` summary is still green, `tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json` is now the canonical live frontier; follow the repro-data-owned checked QA blocker route, replay the checked root, and only resume archival once the QA blocker packet disappears.",
                "correct_course_row": paused_rows["correct_course_row"],
                "theory_row": paused_rows["theory_row"],
                "main_exec_row": paused_rows["main_exec_row"],
                "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 checked QA blocker owns release-evidence drift; rerun reproduction release surface + checked root |",
                "qa_row": paused_rows["qa_row"],
                "next_command": _frontier_next_command_from_qa_blocker(qa_blocker),
            }
        if qa_blocker.get("owner_lane") == "contdid-gsd-qa-mc-48":
            qa_label = _text_or_default(
                qa_blocker.get("owner_ready_label"),
                _text_or_default(
                    qa_blocker.get("failing_label"),
                    "checked QA packet",
                ),
            )
            qa_gate = _text_or_default(
                qa_blocker.get("failing_gate_id"),
                "checked QA packet",
            )
            qa_activity_date = _summary_activity_date(summary)
            paused_rows = _post_v1_paused_lane_rows()
            return {
                "project_focus": f"**Current focus:** Phase 10 checked QA blocker is live; QA-owned {qa_label} route is live / rerun checked root",
                "project_context": f"Phase 10 all-gates summary is still green, but the checked QA packet has reopened on a QA-owned `{qa_label}` blocker; follow the QA blocker packet's next command and keep archival parked until that blocker disappears.",
                "state_focus": f"**Current focus:** Phase 10 reopened — checked QA blocker is live; QA-owned {qa_label} route is live / rerun checked root",
                "state_last_activity": f"Last Activity: {qa_activity_date} — checked QA packet reopened on a QA-owned {qa_label} blocker",
                "state_stopped_at": f"Stopped At: Checked QA packet is red while all-gates remains green; the QA-owned {qa_label} route is live until the blocker clears",
                "state_pending_todo": f"QA 清掉 checked QA `{qa_gate}` blocker 后再决定是否重跑 gitless milestone archival runner",
                "active_objective": f"- Active objective: 对齐 checked QA blocker frontier到 QA-owned {qa_label} route，并在 blocker 清空前暂停 archival rerun",
                "qa_live_root_note": f"- QA/live-root note: even though the checked `all-gates` summary is still green, `tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json` is now the canonical live frontier; follow the QA-owned checked QA blocker route for `{qa_label}`, replay the checked root, and only resume archival once the QA blocker packet disappears.",
                "correct_course_row": paused_rows["correct_course_row"],
                "theory_row": paused_rows["theory_row"],
                "main_exec_row": paused_rows["main_exec_row"],
                "repro_row": paused_rows["repro_row"],
                "qa_row": f"| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 checked QA blocker owns `{qa_label}` / {qa_gate} drift; rerun QA packet + checked root |",
                "next_command": _frontier_next_command_from_qa_blocker(qa_blocker),
            }
        return {
            "project_focus": "**Current focus:** Phase 10 checked QA blocker is live; refresh frontier evidence + archival handoff",
            "project_context": "Phase 10 all-gates summary is still green, but the checked QA packet has reopened on a correct-course-owned release-evidence blocker; follow the QA blocker packet's next command and keep archival parked until that blocker disappears.",
            "state_focus": "**Current focus:** Phase 10 reopened — checked QA blocker is live; refresh frontier evidence + archival handoff",
            "state_last_activity": "Last Activity: 2026-04-13 — checked QA packet reopened on correct-course release-evidence drift",
            "state_stopped_at": "Stopped At: Checked QA packet is red while all-gates remains green; refresh frontier evidence + archival handoff before rerunning archival",
            "state_pending_todo": "correct-course 对齐 checked QA blocker frontier、v1 handoff 与 release-evidence docs 后，再决定是否重跑 gitless milestone archival runner",
            "active_objective": "- Active objective: 对齐 checked QA blocker frontier、v1 handoff 与 release-gate README/automation state，并在 blocker 清空前暂停 archival rerun",
            "qa_live_root_note": "- QA/live-root note: even though the checked `all-gates` summary is still green, `tests/contracts/phase10/runs/qa_phase_verification_blocker_packet.json` is now the canonical live frontier; follow its exact `next_command`, refresh the release-evidence/v1 handoff control plane, and only resume archival once the QA blocker packet disappears.",
            "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 checked QA blocker frontier is synced; refresh v1/release-evidence control plane before archival resumes |",
            "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
            "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; main-exec rerun stays green while the checked QA blocker owns the live control-plane follow-up |",
            "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors stay synced to the checked QA blocker while the checked root remains green |",
            "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 reopened on checked release-evidence drift; rerun the QA packet after correct-course realigns the frontier handoff |",
            "next_command": _frontier_next_command_from_qa_blocker(qa_blocker),
        }
    if summary["completed_successfully"] is True:
        return _v2_milestone_design_frontier()

    assert blocker is not None
    if blocker["owner_lane"] == "contdid-gsd-correct-course-56":
        return {
            "project_focus": "**Current focus:** Phase 10 release gate reopened; correct-course lane-audit blocker is live / resync frontier docs + archival handoff",
            "project_context": "Phase 10 release gate is currently reopened on a correct-course-owned checked lane-audit refresh blocker; checked release-gate/all-gates summaries now stop at the lane-audit/control-plane step while the archival handoff control plane catches up to the latest checked root.",
            "state_focus": "**Current focus:** Phase 10 reopened — correct-course lane-audit blocker is live; resync frontier docs + archival handoff",
            "state_last_activity": "Last Activity: 2026-04-13 — checked root rerun now stops at the correct-course lane-audit/control-plane blocker",
            "state_stopped_at": "Stopped At: Checked root replay reopened on the correct-course lane-audit/control-plane blocker; resync frontier docs + archival handoff before archival rerun",
            "state_pending_todo": "correct-course 对齐 checked lane-audit blocker、archival handoff与live docs后，再重跑 checked root",
            "active_objective": "- Active objective: 对齐 checked lane-audit blocker、live docs 与 archival handoff，直到 checked root 恢复绿色",
            "qa_live_root_note": "- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now stop at the correct-course-owned `refresh checked lane-audit packet` blocker on `control-plane regressions`; resync the frontier docs + archival handoff, rerun lane-audit via the blocker packet's `next_command`, and only resume maintenance once the blocker packet disappears.",
            "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 reopened on checked lane-audit/control-plane drift; resync frontier docs + archival handoff |",
            "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
            "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; package handoff blocker paths remain frozen while correct-course owns the current lane-audit blocker |",
            "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors track the correct-course lane-audit blocker while the checked root stays red |",
            "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 checked root now stops at a correct-course-owned lane-audit/control-plane blocker; rerun QA only after that sync lands |",
            "next_command": str(blocker["next_command"]),
        }

    if blocker["owner_lane"] == "contdid-gsd-theory-parity-01":
        failing_test = str(
            blocker.get(
                "failing_test_nodeid",
                "tests/test_phase10_v1_audit_blocker_handoff.py::"
                "test_v1_audit_blocker_handoff_checked_outputs_match_archival_ready_frontier",
            )
        )
        if (
            failing_test == "tests/test_phase10_v1_audit_blocker_handoff.py::"
            "test_v1_audit_blocker_handoff_checked_outputs_match_archival_ready_frontier"
        ):
            return {
                "project_focus": "**Current focus:** Phase 10 release gate reopened; theory-owned v1 handoff blocker is live / rerun theory + v1 handoff + QA + all-gates",
                "project_context": "Phase 10 release gate is currently reopened on a theory-owned v1 handoff blocker; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries point at the theory/v1/QA rerun chain.",
                "state_focus": "**Current focus:** Phase 10 reopened — theory-owned v1 handoff blocker is live; rerun theory + v1 handoff + QA + all-gates",
                "state_last_activity": "Last Activity: 2026-04-15 — archival refresh rerun stayed green while the checked root remained theory-owned",
                "state_stopped_at": "Stopped At: Checked root replay reopened on the theory-owned v1 handoff blocker; rerun theory + v1 handoff + QA + all-gates before archival",
                "state_pending_todo": "theory 清掉 checked v1 handoff blocker 后再重跑 gitless milestone archival runner",
                "active_objective": "- Active objective: 同步 live + archived frontier 到 theory-owned v1 handoff blocker，并在 blocker 清空前暂停 archival rerun",
                "qa_live_root_note": f"- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a theory-owned blocker packet for `{failing_test}`; archival stays parked until the theory/v1/QA rerun chain clears.",
                "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 theory-owned v1 handoff route synced; archival is parked until the checked root goes green again |",
                "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 reopened on v1 handoff/frontier drift; rerun theory + v1 handoff + QA + all-gates |",
                "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; package handoff blocker paths stay frozen while the theory v1 handoff rerun owns the checked root blocker |",
                "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors track the theory-owned v1 handoff blocker while the checked root stays red |",
                "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 QA packet mirrors the live theory-owned v1 handoff blocker |",
                "next_command": str(blocker["next_command"]),
            }
        return {
            "project_focus": "**Current focus:** Phase 10 release gate reopened; theory-owned blocker packet is live / rerun theory + v1 handoff + QA + all-gates",
            "project_context": "Phase 10 release gate is currently reopened on a theory-owned blocker packet; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries point at the theory/v1/QA rerun chain.",
            "state_focus": "**Current focus:** Phase 10 reopened — theory-owned release-gate blocker is live; rerun theory + v1 handoff + QA + all-gates",
            "state_last_activity": "Last Activity: 2026-04-13 — checked root rerun now stops at the theory-owned full-suite blocker",
            "state_stopped_at": "Stopped At: Checked root replay reopened on the theory-owned full-suite blocker; rerun theory + v1 handoff + QA + all-gates before archival",
            "state_pending_todo": "theory 清掉 checked root blocker 后再重跑 gitless milestone archival runner",
            "active_objective": "- Active objective: 同步 live + archived frontier 到 theory-owned release-gate blocker，并在 blocker 清空前暂停 archival rerun",
            "qa_live_root_note": f"- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a theory-owned blocker packet for `{failing_test}`; archival stays parked until the theory/v1/QA rerun chain clears.",
            "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 theory-owned blocker route synced; archival is parked until the checked root goes green again |",
            "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 reopened on theory/v1 handoff drift; rerun theory + v1 handoff + QA + all-gates |",
            "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; package handoff blocker paths stay frozen while theory rerun owns the checked root blocker |",
            "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors track the theory-owned blocker while the checked root stays red |",
            "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 QA packet mirrors the live theory-owned checked-root blocker |",
            "next_command": str(blocker["next_command"]),
        }

    if blocker["owner_lane"] == "contdid-gsd-qa-mc-48":
        failing_gate_id = str(blocker.get("failing_gate_id", ""))
        failing_label = _text_or_default(blocker.get("failing_label"), "full regression suite")
        failing_test = _text_or_default(blocker.get("failing_test_nodeid"), "")
        if failing_gate_id == "numerical-audit":
            return {
                "project_focus": "**Current focus:** Phase 10 release gate reopened; QA-owned numerical blocker is live / rerun numerical-audit + all-gates",
                "project_context": f"Phase 10 release gate is currently reopened on a QA-owned `{failing_label}` blocker in `numerical-audit`; checked lane-audit remains green while the checked all-gates summary now points at the QA numerical rerun chain.",
                "state_focus": "**Current focus:** Phase 10 reopened — QA-owned numerical blocker is live; rerun numerical-audit + all-gates",
                "state_last_activity": "Last Activity: 2026-04-13 — checked root rerun now stops at the QA-owned numerical-audit blocker",
                "state_stopped_at": f"Stopped At: Checked root replay reopened on the QA-owned `{failing_label}` blocker; rerun numerical-audit + all-gates before archival",
                "state_pending_todo": f"QA 清掉 `{failing_label}` blocker后再重跑 gitless milestone archival runner",
                "active_objective": "- Active objective: 同步 live + archived frontier 到 QA-owned numerical-audit blocker，并在 blocker 清空前暂停 archival rerun",
                "qa_live_root_note": f"- QA/live-root note: lane-audit remains green, but the checked `all-gates` summary now carries a QA-owned blocker packet on `{failing_label}` (`{failing_test}`); rerun numerical-audit via the blocker packet's `next_command`, refresh the checked root, and only resume archival once the blocker packet disappears.",
                "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 QA numerical blocker route is synced; rerun only if frontier docs drift behind the checked root |",
                "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
                "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; main-exec rerun remains parked while the QA-owned numerical blocker owns the live checked root |",
                "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors track the QA-owned numerical blocker while the checked root stays red |",
                "qa_row": f"| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 reopened on `{failing_label}` / numerical-audit drift; rerun numerical-audit + all-gates |",
                "next_command": str(blocker["next_command"]),
            }
        return {
            "project_focus": "**Current focus:** Phase 10 release gate reopened; QA-owned release-gate-truth-snapshot blocker is live / rerun the exact QA full-suite predicate, then replay QA verification + all-gates",
            "project_context": "Phase 10 release gate is currently reopened on a QA-owned full-regression-suite blocker; checked lane-audit and numerical-audit remain green while the checked release-gate/all-gates summaries now point at the QA rerun chain.",
            "state_focus": "**Current focus:** Phase 10 reopened — QA-owned release-gate-truth-snapshot blocker is live; rerun the exact QA full-suite predicate, then replay QA verification + all-gates",
            "state_last_activity": "Last Activity: 2026-04-25 — checked root rerun surfaced a QA-owned release-gate-truth-snapshot/full-suite blocker",
            "state_stopped_at": "Stopped At: Checked root replay reopened on the QA release-gate-truth-snapshot blocker; hand off the exact predicate, then rerun checked QA verification + all-gates before archival",
            "state_pending_todo": "QA 清掉 `tests/test_phase10_qa_phase_verification.py::test_phase10_qa_phase_verification_runner_records_release_gate_truth_snapshot` 后，再重跑 checked QA verification、all-gates 与 gitless milestone archival runner",
            "active_objective": "- Active objective: 同步 live + archived frontier 到 QA-owned release-gate-truth-snapshot blocker（`tests/test_phase10_qa_phase_verification.py::test_phase10_qa_phase_verification_runner_records_release_gate_truth_snapshot`），并在 blocker 清空前暂停 archival rerun",
            "qa_live_root_note": "- QA/live-root note: the checked `release-gate` / `all-gates` summaries currently stop at the QA-owned `full regression suite` blocker with exact node `tests/test_phase10_qa_phase_verification.py::test_phase10_qa_phase_verification_runner_records_release_gate_truth_snapshot`; rerun the QA packet, replay the checked root, and only resume archival once the blocker packet disappears.",
            "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 QA release-gate-truth-snapshot blocker is synced; registry/TOML/DB all mirror the current checked-root route |",
            "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff stay green while the checked root is parked on the QA-owned blocker |",
            "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; main-exec rerun is parked behind the QA-owned full regression suite blocker |",
            "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors now track the QA-owned release-gate-truth-snapshot blocker while the checked root stays red |",
            "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 reopened on release-gate-truth-snapshot/full-suite failure; rerun QA packet + all-gates after the snapshot guard is fixed |",
            "next_command": str(blocker["next_command"]),
        }

    if blocker["owner_lane"] == "contdid-gsd-repro-data-33":
        return {
            "project_focus": "**Current focus:** Phase 10 release gate reopened; repro-data blocker packet is live / rerun release-evidence verifier + all-gates",
            "project_context": "Phase 10 release gate is currently reopened on a repro-data-owned full-suite blocker; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries stop at the release-evidence verifier.",
            "state_focus": "**Current focus:** Phase 10 reopened — repro-data-owned release-evidence blocker is live; rerun release-evidence verifier + all-gates",
            "state_last_activity": "Last Activity: 2026-04-13 — reran the checked root and surfaced a repro-data-owned blocker packet",
            "state_stopped_at": "Stopped At: Checked root replay reopened on the repro-data release-evidence blocker; rerun the focused repro verifier + all-gates before archival",
            "state_pending_todo": "repro-data 清掉 checked root blocker 后再重跑 gitless milestone archival runner",
            "active_objective": "- Active objective: 同步 live + archived frontier 到 repro-data-owned release-evidence blocker，并在 blocker 清空前暂停 archival rerun",
            "qa_live_root_note": "- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a repro-data-owned blocker packet on `full regression suite`; archival stays parked until the focused repro verifier and the full release-gate rerun both clear.",
            "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 repro-data blocker route synced; archival is parked until the checked root goes green again |",
            "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
            "main_exec_row": "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 closed; main-exec rerun remains green while the repro-data release-evidence blocker owns the live checked root |",
            "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 reopened on release-evidence drift; rerun the focused repro verifier + all-gates |",
            "qa_row": "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 QA packet now mirrors the repro-data-owned full-suite blocker inside the checked root |",
            "next_command": str(blocker["next_command"]),
        }

    audit_repair_lane_slug = _audit_repair_lane_slug(blocker["owner_lane"])
    if audit_repair_lane_slug is not None:
        audit_repair_label = _text_or_default(
            blocker.get("owner_ready_label"),
            _text_or_default(
                blocker.get("failing_label"),
                "refresh checked package verification packet",
            ),
        )
        audit_repair_blocker_phrase = _blocker_phrase(audit_repair_label)
        audit_repair_nodeid = _text_or_default(
            blocker.get("failing_test_nodeid"),
            (
                "contdid-py/tests/test_phase10_package_phase_verification.py::"
                "test_phase10_package_phase_verification_checked_outputs_are_present_and_green"
            ),
        )
        audit_repair_activity_date = _summary_activity_date(summary)
        paused_rows = _post_v1_paused_lane_rows()
        return {
            "project_focus": f"**Current focus:** Phase 10 release gate reopened; {audit_repair_lane_slug}-owned {audit_repair_blocker_phrase} is live / rerun checked root",
            "project_context": f"Phase 10 release gate is currently reopened on a post-v1 audit-repair {audit_repair_lane_slug} `{audit_repair_label}` blocker; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries stop at `{audit_repair_nodeid}`.",
            "state_focus": f"**Current focus:** Phase 10 reopened — {audit_repair_lane_slug}-owned {audit_repair_blocker_phrase} is live; rerun checked root",
            "state_last_activity": f"Last Activity: {audit_repair_activity_date} — checked root rerun now stops at the {audit_repair_lane_slug}-owned {audit_repair_blocker_phrase}",
            "state_stopped_at": f"Stopped At: Checked root replay reopened on the {audit_repair_lane_slug}-owned {audit_repair_blocker_phrase}; rerun checked root before archival",
            "state_pending_todo": f"{audit_repair_lane_slug} 清掉 checked root blocker 后再重跑 gitless milestone archival runner",
            "active_objective": f"- Active objective: 对齐 live + archived frontier 到 {audit_repair_lane_slug}-owned {audit_repair_blocker_phrase}，并在 blocker 清空前暂停 archival rerun",
            "qa_live_root_note": f"- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a {audit_repair_lane_slug}-owned blocker packet on `{audit_repair_label}`; archival stays parked until the checked root rerun clears.",
            "correct_course_row": paused_rows["correct_course_row"],
            "theory_row": paused_rows["theory_row"],
            "main_exec_row": paused_rows["main_exec_row"],
            "repro_row": paused_rows["repro_row"],
            "qa_row": paused_rows["qa_row"],
            "next_command": str(blocker["next_command"]),
        }

    assert blocker["owner_lane"] == "contdid-gsd-main-exec-18"
    main_exec_label = _text_or_default(
        blocker.get("owner_ready_label"),
        _text_or_default(blocker.get("failing_label"), ""),
    )
    main_exec_blocker_phrase = _blocker_phrase(main_exec_label)
    main_exec_nodeid = _text_or_default(
        blocker.get("failing_test_nodeid"),
        (
            "contdid-py/tests/test_phase10_package_phase_verification.py::"
            "test_phase10_package_phase_verification_checked_outputs_are_present_and_green"
        ),
    )
    main_exec_activity_date = _summary_activity_date(summary)
    if main_exec_label == "full regression suite":
        project_context = f"Phase 10 release gate is currently reopened on a main-exec-owned `full regression suite` blocker; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries stop at `{main_exec_nodeid}`."
        qa_live_root_note = "- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a main-exec-owned blocker packet on `full regression suite`; archival stays parked until the 10-03/full release-gate rerun clears."
        main_exec_row = "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 reopened on downstream full regression-suite drift; execute 10-03 + rerun the full release gate |"
        qa_row = "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 QA packet now snapshots the downstream full regression-suite blocker inside the main-exec route |"
    else:
        project_context = f"Phase 10 release gate is currently reopened on a main-exec-owned `{main_exec_label}` blocker; checked lane-audit and numerical-audit stay green while the checked release-gate/all-gates summaries stop at `{main_exec_nodeid}`."
        qa_live_root_note = f"- QA/live-root note: lane-audit and numerical-audit remain green, but the checked release-gate/all-gates summaries now carry a main-exec-owned blocker packet on `{main_exec_label}`; archival stays parked until the 10-03/full release-gate rerun clears."
        main_exec_row = "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | Phase 10 reopened on checked QA packet refresh drift; execute 10-03 + rerun the full release gate |"
        qa_row = "| contdid-gsd-qa-mc-48 | 48 | qa-mc | tests, monte carlo harness, numerical contracts | ACTIVE | Phase 10 QA packet refresh is the live checked-root touchpoint inside the main-exec blocker route |"
    return {
        "project_focus": f"**Current focus:** Phase 10 release gate reopened; main-exec-owned {main_exec_blocker_phrase} is live / rerun 10-03 + full release gate",
        "project_context": project_context,
        "state_focus": f"**Current focus:** Phase 10 reopened — main-exec-owned {main_exec_blocker_phrase} is live; rerun 10-03 + full release gate",
        "state_last_activity": f"Last Activity: {main_exec_activity_date} — checked root rerun now stops at the main-exec-owned {main_exec_blocker_phrase}",
        "state_stopped_at": f"Stopped At: Checked root replay reopened on the main-exec-owned {main_exec_blocker_phrase}; rerun 10-03 + full release gate before archival",
        "state_pending_todo": "main-exec 清掉 checked root blocker 后再重跑 gitless milestone archival runner",
        "active_objective": f"- Active objective: 同步 live + archived frontier 到 main-exec-owned {main_exec_blocker_phrase}，并在 blocker 清空前暂停 archival rerun",
        "qa_live_root_note": qa_live_root_note,
        "correct_course_row": "| contdid-gsd-correct-course-56 | 56 | correct-course | .planning/**, automation/**, repo-wide drift repairs | ACTIVE | Phase 10 main-exec blocker route synced; archival is parked until the checked root goes green again |",
        "theory_row": "| contdid-gsd-theory-parity-01 | 01 | theory-parity | paper specs, contracts, fidelity layer tests | ACTIVE | Phase 10 closed; theory packet + v1 handoff now freeze package manifest and canonical paths / rerun only if package evidence or source-truth docs drift |",
        "main_exec_row": main_exec_row,
        "repro_row": "| contdid-gsd-repro-data-33 | 33 | repro-data | repro fixtures, data scripts, e2e assets | ACTIVE | Phase 10 closed; release-evidence mirrors track the main-exec blocker while the checked root stays red |",
        "qa_row": qa_row,
        "next_command": str(blocker["next_command"]),
    }


def _frontier_with_post_v1_audit_rows(frontier: dict[str, str]) -> dict[str, str]:
    audit_frontier = expected_post_v1_audit_repair_frontier()
    return {
        **frontier,
        "audit_row_13": audit_frontier["audit_row_13"],
        "audit_row_41": audit_frontier["audit_row_41"],
        "audit_row_49": audit_frontier["audit_row_49"],
        "paused_repro_row": frontier.get(
            "paused_repro_row",
            audit_frontier["paused_repro_row"],
        ),
    }


_expected_phase10_frontier_impl = expected_phase10_frontier


def expected_phase10_frontier() -> dict[str, str]:  # type: ignore[no-redef]
    return _frontier_with_post_v1_audit_rows(_expected_phase10_frontier_impl())


builtins.__dict__[EXPORTS_CACHE_KEY] = {
    "expected_v1_archival_frontier": expected_v1_archival_frontier,
    "expected_v1_archival_frontier_addenda": expected_v1_archival_frontier_addenda,
    "expected_phase10_frontier": expected_phase10_frontier,
    "expected_phase10_frontier_addenda": expected_phase10_frontier_addenda,
    "expected_post_v1_audit_repair_frontier": expected_post_v1_audit_repair_frontier,
    "expected_v2_milestone_design_frontier": expected_v2_milestone_design_frontier,
}
