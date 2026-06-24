from __future__ import annotations

import os
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_HELPER_PATH = (
    REPO_ROOT / "contdid-py" / "src" / "phase10_frontier_expectations.py"
)
BUILD_BACKEND_PATH = (
    REPO_ROOT / "contdid-py" / "build_backend" / "contdid_build_backend.py"
)


def _build_wheel_from_live_source(wheel_dir: Path) -> Path:
    build_result = subprocess.run(
        [
            "python3",
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
            str(REPO_ROOT / "contdid-py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_result.returncode == 0, build_result.stderr
    return next(wheel_dir.glob("contdid_py-*.whl"))


def _build_sdist_from_live_source(sdist_dir: Path) -> Path:
    build_result = subprocess.run(
        [
            "python3",
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(sdist_dir),
            str(REPO_ROOT / "contdid-py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_result.returncode == 0, build_result.stderr
    return next(sdist_dir.glob("contdid_py-*.tar.gz"))


def _create_venv_with_artifact(artifact_path: Path, venv_dir: Path) -> None:
    create_venv_result = subprocess.run(
        ["python3", "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create_venv_result.returncode == 0, create_venv_result.stderr

    pip_path = venv_dir / "bin" / "pip"
    install_result = subprocess.run(
        [
            str(pip_path),
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-compile",
            str(artifact_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install_result.returncode == 0, install_result.stderr


def _create_system_site_venv(
    venv_dir: Path,
    *,
    with_pip: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["python3", "-m", "venv", "--system-site-packages"]
    if not with_pip:
        command.append("--without-pip")
    command.append(str(venv_dir))
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_build_backend_module():
    spec = importlib.util.spec_from_file_location(
        "phase10_runtime_helper_editable_build_backend",
        BUILD_BACKEND_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _venv_site_packages(venv_dir: Path) -> Path:
    python_path = venv_dir / "bin" / "python"
    completed = subprocess.run(
        [
            str(python_path),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return Path(completed.stdout.strip())


def _install_editable_wheel_without_pip(venv_dir: Path) -> None:
    backend = _load_build_backend_module()
    wheel_dir = venv_dir / "_editable-wheelhouse"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = backend.build_editable(str(wheel_dir))
    site_packages = _venv_site_packages(venv_dir)
    site_packages.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel_dir / wheel_name) as editable_wheel:
        editable_wheel.extractall(site_packages)


def _create_editable_venv(venv_dir: Path) -> None:
    create_venv_result = _create_system_site_venv(venv_dir)
    if create_venv_result.returncode != 0:
        shutil.rmtree(venv_dir, ignore_errors=True)
        pipless_result = _create_system_site_venv(venv_dir, with_pip=False)
        assert pipless_result.returncode == 0, (
            create_venv_result.stderr + pipless_result.stderr
        )
        _install_editable_wheel_without_pip(venv_dir)
        return

    pip_path = venv_dir / "bin" / "pip"
    install_result = subprocess.run(
        [
            str(pip_path),
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-compile",
            "-e",
            str(REPO_ROOT / "contdid-py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install_result.returncode == 0, install_result.stderr


def _install_artifact_to_target(artifact_path: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    install_result = subprocess.run(
        [
            "python3",
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "--target",
            str(target_root),
            str(artifact_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert install_result.returncode == 0, install_result.stderr


def _install_editable_wheel_to_target(target_root: Path) -> None:
    backend = _load_build_backend_module()
    wheel_dir = target_root / "_editable-wheelhouse"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = backend.build_editable(str(wheel_dir))
    with zipfile.ZipFile(wheel_dir / wheel_name) as editable_wheel:
        editable_wheel.extractall(target_root)


def _run_python_with_target_path(
    target_root: Path,
    script: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target_root)
    return subprocess.run(
        ["python3", "-c", script],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _installed_distribution_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def test_editable_runtime_smoke_falls_back_when_ensurepip_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []
    fallback_installs: list[Path] = []

    def _fake_create_system_site_venv(
        venv_dir: Path,
        *,
        with_pip: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(with_pip)
        return subprocess.CompletedProcess(
            ["python3", "-m", "venv", str(venv_dir)],
            1 if with_pip else 0,
            stderr="ensurepip unavailable" if with_pip else "",
        )

    monkeypatch.setattr(
        sys.modules[__name__],
        "_create_system_site_venv",
        _fake_create_system_site_venv,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_install_editable_wheel_without_pip",
        lambda venv_dir: fallback_installs.append(venv_dir),
    )

    venv_dir = tmp_path / "editable-venv"
    _create_editable_venv(venv_dir)

    assert calls == [True, False]
    assert fallback_installs == [venv_dir]


def _frontier_snapshot_script() -> str:
    return (
        "import json, os; "
        "os.chdir('/tmp'); "
        "import phase10_frontier_expectations as module; "
        "frontier = module.expected_phase10_frontier(); "
        "post_v1 = module.expected_post_v1_audit_repair_frontier(); "
        "print(json.dumps({"
        "'module_file': module.__file__, "
        "'frontier_project_focus': frontier['project_focus'], "
        "'frontier_project_context': frontier['project_context'], "
        "'frontier_state_focus': frontier['state_focus'], "
        "'frontier_state_last_activity': frontier['state_last_activity'], "
        "'frontier_state_stopped_at': frontier['state_stopped_at'], "
        "'frontier_state_pending_todo': frontier['state_pending_todo'], "
        "'frontier_active_objective': frontier['active_objective'], "
        "'frontier_qa_live_root_note': frontier['qa_live_root_note'], "
        "'frontier_correct_course_row': frontier['correct_course_row'], "
        "'frontier_theory_row': frontier['theory_row'], "
        "'frontier_main_exec_row': frontier['main_exec_row'], "
        "'frontier_repro_row': frontier['repro_row'], "
        "'frontier_qa_row': frontier['qa_row'], "
        "'frontier_next_command': frontier['next_command'], "
        "'post_v1_project_focus': post_v1['project_focus'], "
        "'post_v1_project_context': post_v1['project_context'], "
        "'post_v1_state_focus': post_v1['state_focus'], "
        "'post_v1_state_last_activity': post_v1['state_last_activity'], "
        "'post_v1_state_stopped_at': post_v1['state_stopped_at'], "
        "'post_v1_state_pending_todo': post_v1['state_pending_todo'], "
        "'post_v1_active_objective': post_v1['active_objective'], "
        "'post_v1_qa_live_root_note': post_v1['qa_live_root_note'], "
        "'post_v1_next_command': post_v1['next_command']"
        "}, sort_keys=True))"
    )


def _run_python_in_existing_distribution(
    venv_dir: Path,
    script: str,
) -> subprocess.CompletedProcess[str]:
    python_path = venv_dir / "bin" / "python"
    return subprocess.run(
        [str(python_path), "-c", script],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_wheel_helper_ignores_parent_pythonpath(
    monkeypatch,
    shared_wheel_path: Path,
    tmp_path: Path,
) -> None:
    source_root = REPO_ROOT / "contdid-py" / "src"
    venv_dir = tmp_path / "venv"
    monkeypatch.setenv("PYTHONPATH", str(source_root))

    runtime_result = _run_python_in_installed_distribution(
        shared_wheel_path,
        venv_dir,
        (
            "from pathlib import Path\n"
            "import json\n"
            "import contdid\n"
            "print(json.dumps({\n"
            "'contdid_file': Path(contdid.__file__).as_posix()\n"
            "}, sort_keys=True))\n"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload["contdid_file"] != (
        source_root / "contdid" / "__init__.py"
    ).as_posix()


def _installed_phase10_summary_path(venv_dir: Path) -> Path:
    return next(
        venv_dir.glob(
            "lib/python*/site-packages/contdid/contracts/phase10/runs/package_phase_verification_summary.json"
        )
    )


def _target_root_phase10_summary_path(target_root: Path) -> Path:
    return (
        target_root
        / "contdid"
        / "contracts"
        / "phase10"
        / "runs"
        / "package_phase_verification_summary.json"
    )


def _installed_distribution_frontier_snapshot_from_target_root(
    target_root: Path,
    summary_updates: dict[str, object] | None = None,
) -> dict[str, str]:
    summary_path: Path | None = None
    original_summary_text: str | None = None
    if summary_updates is not None:
        summary_path = _target_root_phase10_summary_path(target_root)
        original_summary_text = summary_path.read_text(encoding="utf-8")
        summary = json.loads(original_summary_text)
        summary["release_gate_truth_snapshot"].update(summary_updates)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import_result = _run_python_with_target_path(
            target_root,
            _frontier_snapshot_script(),
        )
        assert import_result.returncode == 0, import_result.stderr
        return json.loads(import_result.stdout)
    finally:
        if summary_path is not None and original_summary_text is not None:
            summary_path.write_text(original_summary_text, encoding="utf-8")


def _installed_distribution_frontier_snapshot_from_existing_env(
    venv_dir: Path,
    summary_updates: dict[str, object] | None = None,
) -> dict[str, str]:
    summary_path: Path | None = None
    original_summary_text: str | None = None
    if summary_updates is not None:
        summary_path = _installed_phase10_summary_path(venv_dir)
        original_summary_text = summary_path.read_text(encoding="utf-8")
        summary = json.loads(original_summary_text)
        summary["release_gate_truth_snapshot"].update(summary_updates)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import_result = _run_python_in_existing_distribution(
            venv_dir,
            _frontier_snapshot_script(),
        )
        assert import_result.returncode == 0, import_result.stderr
        return json.loads(import_result.stdout)
    finally:
        if summary_path is not None and original_summary_text is not None:
            summary_path.write_text(original_summary_text, encoding="utf-8")


def test_installed_wheel_frontier_helper_routes_lane13_checked_qa_blocker(
    shared_wheel_root: Path,
) -> None:
    checked_qa_recovery_command = (
        "python3 tests/run_phase10_qa_verification.py "
        "--force-release-gate-bundle-tests --output "
        "tests/contracts/phase10/runs/qa_phase_verification_summary.json && "
        "python3 reproduction/phase10_release_gate/run_release_gate.py --gate-id all "
        "--output-root reproduction/phase10_release_gate/runs"
    )
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root,
        {
            "checked_all_gates_summary_completed_successfully": True,
            "checked_all_gates_summary_generated_at": "2026-05-06T07:20:00Z",
            "checked_qa_summary_completed_successfully": False,
            "checked_qa_blocker_packet_exists": True,
            "checked_qa_blocker_owner_lane": "contdid-gsd-audit-repair-13",
            "checked_qa_blocker_owner_ready_label": (
                "core estimation packaging verifier"
            ),
            "checked_qa_blocker_failing_label": "core estimation packaging verifier",
            "checked_qa_blocker_failing_test_nodeid": (
                "contdid-py/tests/test_phase4_level_estimator.py::"
                "test_weighted_average_for_valid_dose"
            ),
            "checked_qa_blocker_next_command": (
                "python3 -m pytest contdid-py/tests/test_phase4_level_estimator.py::"
                "test_weighted_average_for_valid_dose"
            ),
        },
    )

    assert snapshot["frontier_project_focus"] == (
        "**Current focus:** Phase 10 checked QA blocker is live; lane13-owned "
        "core estimation packaging verifier route is live / rerun checked root"
    )
    assert "lane13-owned" in snapshot["frontier_project_context"]
    assert snapshot["frontier_next_command"] == checked_qa_recovery_command


@pytest.fixture(scope="module")
def shared_wheel_path(tmp_path_factory) -> Path:
    wheel_dir = tmp_path_factory.mktemp("phase10-wheelhouse")
    return _build_wheel_from_live_source(wheel_dir)


@pytest.fixture(scope="module")
def shared_sdist_path(tmp_path_factory) -> Path:
    sdist_dir = tmp_path_factory.mktemp("phase10-sdist")
    return _build_sdist_from_live_source(sdist_dir)


@pytest.fixture(scope="module")
def shared_wheel_root(tmp_path_factory, shared_wheel_path: Path) -> Path:
    target_root = tmp_path_factory.mktemp("phase10-wheel-root")
    _install_artifact_to_target(shared_wheel_path, target_root)
    return target_root


@pytest.fixture(scope="module")
def shared_sdist_root(tmp_path_factory, shared_sdist_path: Path) -> Path:
    target_root = tmp_path_factory.mktemp("phase10-sdist-root")
    _install_artifact_to_target(shared_sdist_path, target_root)
    return target_root


@pytest.fixture(scope="module")
def shared_editable_root(tmp_path_factory) -> Path:
    target_root = tmp_path_factory.mktemp("phase10-editable-root")
    _install_editable_wheel_to_target(target_root)
    return target_root


def _load_runtime_helper():
    sys.modules.pop("phase10_frontier_expectations", None)
    spec = importlib.util.spec_from_file_location(
        "phase10_frontier_expectations",
        RUNTIME_HELPER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase10_frontier_expectations"] = module
    spec.loader.exec_module(module)
    return module


def _installed_distribution_frontier_snapshot(
    artifact_path: Path,
    venv_dir: Path,
    summary_updates: dict[str, object] | None = None,
) -> dict[str, str]:
    create_venv_result = subprocess.run(
        ["python3", "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create_venv_result.returncode == 0, create_venv_result.stderr

    pip_path = venv_dir / "bin" / "pip"
    python_path = venv_dir / "bin" / "python"

    install_result = subprocess.run(
        [str(pip_path), "install", "--force-reinstall", "--no-deps", str(artifact_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_installed_distribution_env(),
    )
    assert install_result.returncode == 0, install_result.stderr

    if summary_updates is not None:
        summary_path = next(
            venv_dir.glob(
                "lib/python*/site-packages/contdid/contracts/phase10/runs/package_phase_verification_summary.json"
            )
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["release_gate_truth_snapshot"].update(summary_updates)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import_result = subprocess.run(
        [
            str(python_path),
            "-c",
            (
                "import json, os; "
                "os.chdir('/tmp'); "
                "import phase10_frontier_expectations as module; "
                "frontier = module.expected_phase10_frontier(); "
                "post_v1 = module.expected_post_v1_audit_repair_frontier(); "
                "print(json.dumps({"
                "'module_file': module.__file__, "
                "'frontier_project_focus': frontier['project_focus'], "
                "'frontier_project_context': frontier['project_context'], "
                "'frontier_state_focus': frontier['state_focus'], "
                "'frontier_state_last_activity': frontier['state_last_activity'], "
                "'frontier_state_stopped_at': frontier['state_stopped_at'], "
                "'frontier_state_pending_todo': frontier['state_pending_todo'], "
                "'frontier_active_objective': frontier['active_objective'], "
                "'frontier_qa_live_root_note': frontier['qa_live_root_note'], "
                "'frontier_correct_course_row': frontier['correct_course_row'], "
                "'frontier_theory_row': frontier['theory_row'], "
                "'frontier_main_exec_row': frontier['main_exec_row'], "
                "'frontier_repro_row': frontier['repro_row'], "
                "'frontier_qa_row': frontier['qa_row'], "
                "'frontier_next_command': frontier['next_command'], "
                "'post_v1_project_focus': post_v1['project_focus'], "
                "'post_v1_project_context': post_v1['project_context'], "
                "'post_v1_state_focus': post_v1['state_focus'], "
                "'post_v1_state_last_activity': post_v1['state_last_activity'], "
                "'post_v1_state_stopped_at': post_v1['state_stopped_at'], "
                "'post_v1_state_pending_todo': post_v1['state_pending_todo'], "
                "'post_v1_active_objective': post_v1['active_objective'], "
                "'post_v1_qa_live_root_note': post_v1['qa_live_root_note'], "
                "'post_v1_next_command': post_v1['next_command']"
                "}, sort_keys=True))"
            ),
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
        env=_installed_distribution_env(),
    )
    assert import_result.returncode == 0, import_result.stderr
    return json.loads(import_result.stdout)


def _run_python_in_installed_distribution(
    artifact_path: Path,
    venv_dir: Path,
    script: str,
) -> subprocess.CompletedProcess[str]:
    create_venv_result = subprocess.run(
        ["python3", "-m", "venv", "--system-site-packages", str(venv_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert create_venv_result.returncode == 0, create_venv_result.stderr

    pip_path = venv_dir / "bin" / "pip"
    python_path = venv_dir / "bin" / "python"

    install_result = subprocess.run(
        [str(pip_path), "install", "--force-reinstall", "--no-deps", str(artifact_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_installed_distribution_env(),
    )
    assert install_result.returncode == 0, install_result.stderr

    return subprocess.run(
        [str(python_path), "-c", script],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
        env=_installed_distribution_env(),
    )


@pytest.mark.parametrize(
    ("helper_name", "args"),
    [
        (
            "_installed_distribution_frontier_snapshot",
            lambda artifact_path, venv_dir: (artifact_path, venv_dir),
        ),
        (
            "_run_python_in_installed_distribution",
            lambda artifact_path, venv_dir: (
                artifact_path,
                venv_dir,
                "print('ok')",
            ),
        ),
    ],
)
def test_installed_distribution_helpers_create_system_site_packages_venvs(
    monkeypatch,
    tmp_path: Path,
    helper_name: str,
    args,
) -> None:
    stop = RuntimeError("stop after venv")
    commands: list[list[str]] = []

    def _fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        commands.append(list(command))
        raise stop

    monkeypatch.setattr(subprocess, "run", _fake_run)

    helper = globals()[helper_name]
    artifact_path = tmp_path / "artifact.whl"
    venv_dir = tmp_path / "venv"
    with pytest.raises(RuntimeError, match="stop after venv"):
        helper(*args(artifact_path, venv_dir))

    assert commands == [
        [
            "python3",
            "-m",
            "venv",
            "--system-site-packages",
            str(venv_dir),
        ]
    ]


def _assert_installed_distribution_uses_post_v1_green_frontier(
    snapshot: dict[str, str],
) -> None:
    assert snapshot["frontier_project_focus"] == (
        "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones"
    )
    assert snapshot["frontier_project_context"] == (
        "在 v1.0 可信发布面之上，把 staggered-adoption 下的 CCK/data-driven 路由收束为 machine-checkable unsupported hard-fail boundary，同时继续推进规模化能力和用户表达层，并把审查/修复作为独立里程碑纳入后续路线。"
    )
    assert snapshot["frontier_state_focus"] == (
        "**Current focus:** v2.0 milestone design — extension, scale, reporting, and audit/repair milestones"
    )
    assert snapshot["frontier_state_last_activity"] == (
        "Last Activity: 2026-05-06 — started v2.0 milestone design"
    )
    assert snapshot["frontier_state_stopped_at"] == (
        "Stopped At: v1 completion evidence remains frozen on disk; v2 milestone design now defines the next feature ladder"
    )
    assert (
        snapshot["frontier_state_pending_todo"]
        == "设计并验证 v2.0 后续里程碑 ladder：staggered-adoption CCK hard-fail boundary、data-shape generalization、performance backend、reporting/teaching、audit/repair"
    )
    assert snapshot["frontier_active_objective"] == snapshot["post_v1_active_objective"]
    assert snapshot["frontier_qa_live_root_note"] == snapshot["post_v1_qa_live_root_note"]
    assert snapshot["frontier_correct_course_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )
    assert snapshot["frontier_theory_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )
    assert snapshot["frontier_main_exec_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )
    assert snapshot["frontier_repro_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )
    assert snapshot["frontier_qa_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )
    assert snapshot["frontier_next_command"] == snapshot["post_v1_next_command"]


def test_phase10_frontier_runtime_helper_exports_expected_frontier_route(
    monkeypatch,
) -> None:
    module = _load_runtime_helper()
    next_command = (
        "python3 automation/scripts/run_theory_phase_verification.py --output "
        "automation/contracts/phase10/runs/theory_phase_verification_summary.json "
        "--inventory-output "
        "automation/contracts/phase10/runs/theory_phase_verification_inventory.md "
        "--rollup-output "
        "automation/contracts/phase10/runs/theory_phase_verification_rollup.json "
        "--verification-doc-root . && python3 "
        "automation/scripts/run_v1_audit_blocker_handoff.py --allow-incomplete-write "
        "--output automation/contracts/phase10/runs/v1_audit_blocker_handoff_summary.json "
        "--inventory-output "
        "automation/contracts/phase10/runs/v1_audit_blocker_handoff_inventory.md "
        "--blocker-output "
        "automation/contracts/phase10/runs/v1_audit_blocker_handoff_blocker_packet.json "
        "&& python3 tests/run_phase10_qa_verification.py --allow-incomplete-write "
        "--force-release-gate-bundle-tests --output "
        "tests/contracts/phase10/runs/qa_phase_verification_summary.json "
        "&& python3 reproduction/phase10_release_gate/run_release_gate.py "
        "--gate-id all --output-root reproduction/phase10_release_gate/runs"
    )
    failing_test = (
        "tests/test_phase10_v1_audit_blocker_handoff.py::"
        "test_v1_audit_blocker_handoff_checked_outputs_match_archival_ready_frontier"
    )

    monkeypatch.setattr(
        module,
        "_load_release_gate_state",
        lambda: (
            {"completed_successfully": False},
            {
                "owner_lane": "contdid-gsd-theory-parity-01",
                "failing_label": "full regression suite",
                "failing_test_nodeid": failing_test,
                "next_command": next_command,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_qa_state",
        lambda: ({"completed_successfully": True}, None),
    )

    frontier = module.expected_phase10_frontier()

    assert frontier["project_focus"] == (
        "**Current focus:** Phase 10 release gate reopened; theory-owned v1 handoff "
        "blocker is live / rerun theory + v1 handoff + QA + all-gates"
    )
    assert frontier["main_exec_row"] == (
        "| contdid-gsd-main-exec-18 | 18 | main-exec | contdid-py/** | ACTIVE | "
        "Phase 10 closed; package handoff blocker paths stay frozen while the "
        "theory v1 handoff rerun owns the checked root blocker |"
    )
    assert frontier["next_command"] == next_command


def test_phase10_frontier_runtime_helper_is_shipped_in_wheel(
    shared_wheel_path: Path,
) -> None:
    wheel_path = shared_wheel_path
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert "phase10_frontier_expectations.py" in names
    assert "contdid/__init__.py" in names
    assert "contdid/contracts/phase2/numerical_truth_contract_v1.json" in names
    assert "contdid/contracts/phase2/paper_truth_contract.json" in names
    assert "contdid/contracts/phase2/phase2_contract_template.json" in names
    assert "contdid/contracts/phase2/symbol_map_contract.json" in names
    assert "contdid/contracts/phase10/package_phase_verification_manifest.json" in names
    assert "contdid/contracts/phase10/runs/package_phase_verification_summary.json" in names
    assert "contdid/contracts/phase10/runs/package_phase_verification_inventory.md" in names
    assert "contdid/contracts/phase10/runs/package_phase_verification_rollup.json" in names
    assert "contdid/reproduction/simulate_contdid/manifest.json" in names
    assert "contdid/reproduction/medicare_pps/manifest.json" in names
    assert "contdid/reproduction/medicare_pps/source_options.json" in names


def test_phase10_frontier_runtime_helper_is_importable_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root
    )

    assert snapshot["module_file"].endswith("phase10_frontier_expectations.py")
    _assert_installed_distribution_uses_post_v1_green_frontier(snapshot)


def test_phase10_frontier_runtime_helper_installed_wheel_ignores_stale_green_qa_blocker(
    shared_wheel_root: Path,
) -> None:
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root,
        summary_updates={
            "checked_qa_summary_completed_successfully": True,
            "checked_qa_blocker_packet_exists": True,
            "checked_qa_blocker_owner_lane": "contdid-gsd-qa-mc-48",
            "checked_qa_blocker_owner_ready_label": "stale Monte Carlo blocker",
            "checked_qa_blocker_failing_label": "stale Monte Carlo blocker",
            "checked_qa_blocker_next_command": "stale command",
        },
    )

    _assert_installed_distribution_uses_post_v1_green_frontier(snapshot)


def test_phase10_frontier_runtime_helper_installed_wheel_ignores_cleared_package_checked_output_qa_blocker(
    shared_wheel_root: Path,
) -> None:
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root,
        summary_updates={
            "checked_qa_summary_completed_successfully": False,
            "checked_qa_blocker_packet_exists": True,
            "checked_qa_blocker_owner_lane": "contdid-gsd-main-exec-18",
            "checked_qa_blocker_owner_ready_label": None,
            "checked_qa_blocker_failing_label": (
                "Checked release-gate packet, shared frontier expectations, "
                "and archival-ready control-plane evidence remain aligned"
            ),
            "checked_qa_blocker_failing_test_nodeid": (
                "contdid-py/tests/test_phase10_package_phase_verification.py::"
                "test_phase10_package_phase_verification_checked_outputs_are_present_and_green"
            ),
            "checked_qa_blocker_next_command": (
                "execute .planning/phases/10-bug-sweep-release-gate/"
                "10-03-PLAN.md after lane-audit and numerical-audit are green, "
                "then rerun the full release gate"
            ),
        },
    )

    _assert_installed_distribution_uses_post_v1_green_frontier(snapshot)


def test_phase10_frontier_runtime_helper_supports_lane49_checked_qa_blocker_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root,
        summary_updates={
            "checked_all_gates_summary_generated_at": "2026-04-27T04:12:23+00:00",
            "checked_qa_summary_completed_successfully": False,
            "checked_qa_blocker_packet_exists": True,
            "checked_qa_blocker_owner_lane": "contdid-gsd-audit-repair-49",
            "checked_qa_blocker_owner_ready_label": "refresh checked package verification packet",
            "checked_qa_blocker_failing_label": "refresh checked package verification packet",
            "checked_qa_blocker_failing_test_nodeid": None,
            "checked_qa_blocker_next_command": (
                "python3 reproduction/phase10_release_gate/run_release_gate.py "
                "--gate-id all --output-root reproduction/phase10_release_gate/runs"
            ),
        },
    )

    assert snapshot["frontier_project_focus"] == (
        "**Current focus:** Phase 10 checked QA blocker is live; lane49-owned "
        "refresh checked package verification packet route is live / rerun checked root"
    )
    assert snapshot["frontier_state_last_activity"] == (
        "Last Activity: 2026-04-27 — checked QA packet reopened on a lane49-owned "
        "refresh checked package verification packet blocker"
    )
    assert snapshot["frontier_next_command"] == (
        "python3 reproduction/phase10_release_gate/run_release_gate.py "
        "--gate-id all --output-root reproduction/phase10_release_gate/runs"
    )
    assert snapshot["frontier_correct_course_row"].endswith(
        "PAUSED | Historical v1 maintenance lane retained as paused evidence |"
    )


def test_phase10_frontier_runtime_helper_preserves_qa_numerical_blocker_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    next_command = (
        "python3 tests/run_phase10_qa_verification.py "
        "--force-release-gate-bundle-tests --output "
        "tests/contracts/phase10/runs/qa_phase_verification_summary.json && "
        "python3 reproduction/phase10_release_gate/run_release_gate.py --gate-id all "
        "--output-root reproduction/phase10_release_gate/runs"
    )
    failing_test = (
        "contdid-py/tests/test_phase6_eventstudy_inference.py::"
        "test_eventstudy_uniform_band_contract"
    )
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_wheel_root,
        summary_updates={
            "checked_qa_summary_completed_successfully": False,
            "checked_qa_blocker_packet_exists": True,
            "checked_qa_blocker_owner_lane": "contdid-gsd-qa-mc-48",
            "checked_qa_blocker_owner_ready_label": "event-study inference verifier",
            "checked_qa_blocker_failing_label": "event-study inference verifier",
            "checked_qa_blocker_failing_gate_id": "numerical-audit",
            "checked_qa_blocker_failing_test_nodeid": failing_test,
            "checked_qa_blocker_next_command": next_command,
        },
    )

    assert snapshot["frontier_project_focus"] == (
        "**Current focus:** Phase 10 checked QA blocker is live; QA-owned "
        "event-study inference verifier route is live / rerun checked root"
    )
    assert snapshot["frontier_state_focus"] == (
        "**Current focus:** Phase 10 reopened — checked QA blocker is live; "
        "QA-owned event-study inference verifier route is live / rerun checked root"
    )
    assert "event-study inference verifier" in snapshot["frontier_qa_live_root_note"]
    assert "numerical-audit drift" in snapshot["frontier_qa_row"]
    assert snapshot["frontier_next_command"] == next_command


def test_phase10_frontier_runtime_helper_uses_post_v1_green_frontier_from_installed_sdist(
    shared_sdist_root: Path,
) -> None:
    snapshot = _installed_distribution_frontier_snapshot_from_target_root(
        shared_sdist_root
    )

    assert snapshot["module_file"].endswith("phase10_frontier_expectations.py")
    _assert_installed_distribution_uses_post_v1_green_frontier(snapshot)


def test_public_runtime_contract_assets_are_available_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_wheel_root,
        (
            "import json; "
            "from importlib.metadata import version; "
                "import pandas as pd; "
                "import contdid; "
                "from contdid import ("
                "ContDIDSpec, "
            "load_medicare_pps_manifest, "
            "load_medicare_pps_source_options, "
            "prepare_medicare_pps_panel, "
            "simulate_contdid_data, "
            "validate_spec"
            "); "
            "validate_spec(ContDIDSpec("
            "target_parameter='level', "
            "aggregation='dose', "
            "dose_est_method='parametric', "
            "control_group='nevertreated'"
            ")); "
            "panel = simulate_contdid_data(n=12, dgp_id='SIM-002-linear-dose'); "
            "medicare = pd.DataFrame(["
            "{'hospital_id': 101, 'year': 1980, 'depreciation_share': 4.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1981, 'depreciation_share': 4.10, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1982, 'depreciation_share': 4.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1983, 'depreciation_share': 4.30, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1984, 'depreciation_share': 4.80, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1985, 'depreciation_share': 5.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1986, 'depreciation_share': 5.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 202, 'year': 1980, 'depreciation_share': 3.20, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1981, 'depreciation_share': 3.10, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1982, 'depreciation_share': 3.00, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1983, 'depreciation_share': 2.90, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1984, 'depreciation_share': 2.80, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1985, 'depreciation_share': 2.70, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1986, 'depreciation_share': 2.60, 'medicare_share_1983': 0.00}"
            "]); "
            "scaffold = prepare_medicare_pps_panel("
            "medicare, "
            "unit_column='hospital_id', "
            "year_column='year', "
            "outcome_column='depreciation_share', "
            "dose_column='medicare_share_1983', "
            "source_id='aha_direct_license'"
            "); "
            "print(json.dumps({"
            "'simulate_shape': list(panel.frame.shape), "
            "'medicare_application_id': load_medicare_pps_manifest()['application_id'], "
            "'medicare_primary_source': load_medicare_pps_source_options()['primary_sources'][0]['id'], "
            "'package_version': contdid.__version__, "
            "'metadata_version': version('contdid-py'), "
            "'scaffold_shape': list(scaffold.panel.frame.shape), "
            "'scaffold_mode': scaffold.metadata['analysis_mode']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "medicare_application_id": "medicare-pps-hospitals",
        "medicare_primary_source": "aha_direct_license",
        "metadata_version": "0.1.0",
        "package_version": "0.1.0",
        "scaffold_mode": "paper-source-aligned",
        "scaffold_shape": [4, 5],
        "simulate_shape": [48, 5],
    }


def test_public_api_contract_bundle_is_available_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_wheel_root,
        (
            "import json; "
            "from contdid.contracts import load_public_api_contract_bundle; "
            "payload = load_public_api_contract_bundle(); "
            "print(json.dumps({"
            "'schema_version': payload['schema_version'], "
            "'phase': payload['phase'], "
            "'has_exports': '__version__' in payload['stable_top_level_exports'], "
            "'confidence_band_shape_rule': payload['result_payload']['confidence_band_shape_contract']['shape_match_rule'], "
            "'public_api_contract_path': payload['source_contracts']['package_init_surface']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "confidence_band_shape_rule": "estimate and std_error must have the same shape",
        "has_exports": True,
        "phase": 9,
        "public_api_contract_path": "contdid-py/src/contdid/__init__.py",
        "schema_version": "0.1",
    }


def test_phase2_contract_bundle_is_available_from_installed_wheel(
    shared_wheel_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_wheel_root,
        (
            "import json; "
            "from contdid.contracts import load_phase2_contract_bundle; "
            "payload = load_phase2_contract_bundle(); "
            "print(json.dumps({"
            "'schema_version': payload['schema_version'], "
            "'phase': payload['phase'], "
            "'estimand_count': len(payload['estimands']), "
            "'dgp_count': len(payload['numerical_truth']['dgp_ids']), "
            "'first_seed': payload['numerical_truth']['seed_registry'][0]['default_seed']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "dgp_count": 5,
        "estimand_count": 6,
        "first_seed": 1234,
        "phase": 2,
        "schema_version": "0.2",
    }


def test_installed_wheel_eventstudy_routes_reject_invalid_basis_controls(
    shared_wheel_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_wheel_root,
        """
import json
from contdid import (
    ContDIDSpec,
    ContDIDValidationError,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
    simulate_contdid_data,
)

panel = simulate_contdid_data(n=12, dgp_id="SIM-002-linear-dose")
cases = [
    (
        estimate_eventstudy_effects,
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        ),
        {"degree": 0},
    ),
    (
        estimate_eventstudy_slope_effects,
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        ),
        {"num_knots": -1},
    ),
]
messages = []
for route, spec, kwargs in cases:
    try:
        route(panel, spec, **kwargs)
    except ContDIDValidationError as exc:
        messages.append(str(exc))
    else:
        raise AssertionError(f"{route.__name__} accepted invalid basis controls")

print(json.dumps({"messages": messages, "route_count": len(cases)}, sort_keys=True))
""",
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "messages": [
            "degree must be at least 1 for dose estimation",
            "num_knots must be nonnegative",
        ],
        "route_count": 2,
    }


def _assert_installed_distribution_cck_unsupported_boundaries(
    target_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        target_root,
        """
import json
import pandas as pd
from contdid import (
    ContDIDSpec,
    ContDIDValidationError,
    PanelData,
    estimate_dose_effects,
    estimate_dose_slope_effects,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
    simulate_contdid_data,
)

two_period_panel = simulate_contdid_data(
    n=60,
    dgp_id="SIM-005-cck-two-period",
    seed=20261234,
)
multi_period_panel = simulate_contdid_data(
    n=60,
    dgp_id="SIM-004-staggered-eventstudy-null",
    seed=20260407,
)
staggered_two_period_panel = PanelData(
    frame=pd.DataFrame(
        [
            ("u0", 1, 0.0, 0, 0.0),
            ("u0", 2, 0.0, 0, 0.0),
            ("u1", 1, 0.0, 0, 0.0),
            ("u1", 2, 0.0, 0, 0.0),
            ("g1a", 1, 0.0, 1, 0.2),
            ("g1a", 2, 0.2, 1, 0.2),
            ("g2a", 1, 0.0, 2, 0.8),
            ("g2a", 2, 0.8, 2, 0.8),
        ],
        columns=["id", "time_period", "Y", "G", "D"],
    )
)
single_cohort_three_period_panel = PanelData(
    frame=pd.DataFrame(
        [
            ("u0", 1, 0.0, 0, 0.0),
            ("u0", 2, 0.0, 0, 0.0),
            ("u0", 3, 0.0, 0, 0.0),
            ("u1", 1, 0.0, 0, 0.0),
            ("u1", 2, 0.0, 0, 0.0),
            ("u1", 3, 0.0, 0, 0.0),
            ("t1", 1, 0.0, 2, 0.2),
            ("t1", 2, 0.1, 2, 0.2),
            ("t1", 3, 0.2, 2, 0.2),
            ("t2", 1, 0.0, 2, 0.5),
            ("t2", 2, 0.25, 2, 0.5),
            ("t2", 3, 0.5, 2, 0.5),
        ],
        columns=["id", "time_period", "Y", "G", "D"],
    )
)
single_cohort_after_window_two_period_panel = PanelData(
    frame=pd.DataFrame(
        [
            ("u0", 1, 0.0, 0, 0.0),
            ("u0", 2, 0.0, 0, 0.0),
            ("u1", 1, 0.0, 0, 0.0),
            ("u1", 2, 0.0, 0, 0.0),
            ("t1", 1, 0.0, 3, 0.2),
            ("t1", 2, 0.2, 3, 0.2),
            ("t2", 1, 0.0, 3, 0.5),
            ("t2", 2, 0.5, 3, 0.5),
            ("t3", 1, 0.0, 3, 0.8),
            ("t3", 2, 0.8, 3, 0.8),
            ("t4", 1, 0.0, 3, 0.9),
            ("t4", 2, 0.9, 3, 0.9),
        ],
        columns=["id", "time_period", "Y", "G", "D"],
    )
)
event_level_spec = ContDIDSpec(
    target_parameter="level",
    aggregation="eventstudy",
    dose_est_method="cck",
    control_group="notyettreated",
)
event_slope_spec = ContDIDSpec(
    target_parameter="slope",
    aggregation="eventstudy",
    dose_est_method="cck",
    control_group="notyettreated",
)
dose_spec = ContDIDSpec(
    target_parameter="level",
    aggregation="dose",
    dose_est_method="cck",
    control_group="nevertreated",
)
dose_slope_spec = ContDIDSpec(
    target_parameter="slope",
    aggregation="dose",
    dose_est_method="cck",
    control_group="nevertreated",
)
cases = [
    (estimate_dose_effects, staggered_two_period_panel, dose_spec, "cck estimator not supported with staggered adoption yet"),
    (estimate_dose_effects, multi_period_panel, dose_spec, "cck estimator not supported with staggered adoption yet"),
    (estimate_dose_effects, single_cohort_three_period_panel, dose_spec, "cck estimator not supported with more than two time periods. consider averaging across pre and post treatment periods"),
    (estimate_dose_effects, single_cohort_after_window_two_period_panel, dose_spec, "cck estimator requires positive treatment timing to start in the post period"),
    (estimate_dose_slope_effects, single_cohort_after_window_two_period_panel, dose_slope_spec, "cck estimator requires positive treatment timing to start in the post period"),
]
messages = []
for route, panel, spec, expected_message in cases:
    try:
        route(panel, spec)
    except ContDIDValidationError as exc:
        message = str(exc)
        if expected_message not in message:
            raise AssertionError(f"{route.__name__} raised {message!r}; expected {expected_message!r}")
        messages.append(message)
    else:
        raise AssertionError(f"{route.__name__} accepted unsupported CCK route")

print(json.dumps({"messages": messages, "route_count": len(cases)}, sort_keys=True))
""",
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "messages": [
            "cck estimator not supported with staggered adoption yet",
            "cck estimator not supported with staggered adoption yet",
            "cck estimator not supported with more than two time periods. consider averaging across pre and post treatment periods",
            "cck estimator requires positive treatment timing to start in the post period; expected G > 0 to equal the last observed time period 2, got [3.0]",
            "cck estimator requires positive treatment timing to start in the post period; expected G > 0 to equal the last observed time period 2, got [3.0]",
        ],
        "route_count": 5,
    }


def test_installed_wheel_cck_routes_keep_unsupported_boundaries(
    shared_wheel_root: Path,
) -> None:
    _assert_installed_distribution_cck_unsupported_boundaries(shared_wheel_root)


def test_installed_sdist_cck_routes_keep_unsupported_boundaries(
    shared_sdist_root: Path,
) -> None:
    _assert_installed_distribution_cck_unsupported_boundaries(shared_sdist_root)


def test_editable_install_cck_routes_keep_unsupported_boundaries(
    shared_editable_root: Path,
) -> None:
    _assert_installed_distribution_cck_unsupported_boundaries(shared_editable_root)


def test_public_api_contract_bundle_is_available_from_installed_sdist(
    shared_sdist_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_sdist_root,
        (
            "import json; "
            "from contdid.contracts import load_public_api_contract_bundle; "
            "payload = load_public_api_contract_bundle(); "
            "print(json.dumps({"
            "'schema_version': payload['schema_version'], "
            "'phase': payload['phase'], "
            "'has_loader': 'load_public_api_contract_bundle' in payload['stable_top_level_exports'], "
            "'has_validator': 'validate_public_api_contract_bundle' in payload['stable_top_level_exports'], "
            "'confidence_band_shape_rule': payload['result_payload']['confidence_band_shape_contract']['shape_match_rule']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "confidence_band_shape_rule": "estimate and std_error must have the same shape",
        "has_loader": True,
        "has_validator": True,
        "phase": 9,
        "schema_version": "0.1",
    }


def test_public_api_contract_bundle_is_available_from_editable_install(
    shared_editable_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_editable_root,
        (
            "import json; "
            "from contdid.contracts import load_public_api_contract_bundle; "
            "payload = load_public_api_contract_bundle(); "
            "print(json.dumps({"
            "'schema_version': payload['schema_version'], "
            "'phase': payload['phase'], "
            "'contract_path': payload['source_contracts']['phase9_release_examples'], "
            "'confidence_band_shape_rule': payload['result_payload']['confidence_band_shape_contract']['shape_match_rule']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "confidence_band_shape_rule": "estimate and std_error must have the same shape",
        "contract_path": "reproduction/phase9_release_examples/manifest.json",
        "phase": 9,
        "schema_version": "0.1",
    }


def test_public_runtime_contract_assets_are_available_from_installed_sdist(
    shared_sdist_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_sdist_root,
        (
            "import json; "
            "from importlib.metadata import version; "
            "import pandas as pd; "
            "import contdid; "
            "from contdid import ("
            "ContDIDSpec, "
            "load_phase2_contract_bundle, "
            "load_medicare_pps_manifest, "
            "load_medicare_pps_source_options, "
            "prepare_medicare_pps_panel, "
            "simulate_contdid_data, "
            "validate_spec"
            "); "
            "validate_spec(ContDIDSpec("
            "target_parameter='level', "
            "aggregation='dose', "
            "dose_est_method='parametric', "
            "control_group='nevertreated'"
            ")); "
            "panel = simulate_contdid_data(n=12, dgp_id='SIM-002-linear-dose'); "
            "medicare = pd.DataFrame(["
            "{'hospital_id': 101, 'year': 1980, 'depreciation_share': 4.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1981, 'depreciation_share': 4.10, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1982, 'depreciation_share': 4.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1983, 'depreciation_share': 4.30, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1984, 'depreciation_share': 4.80, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1985, 'depreciation_share': 5.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1986, 'depreciation_share': 5.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 202, 'year': 1980, 'depreciation_share': 3.20, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1981, 'depreciation_share': 3.10, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1982, 'depreciation_share': 3.00, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1983, 'depreciation_share': 2.90, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1984, 'depreciation_share': 2.80, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1985, 'depreciation_share': 2.70, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1986, 'depreciation_share': 2.60, 'medicare_share_1983': 0.00}"
            "]); "
            "scaffold = prepare_medicare_pps_panel("
            "medicare, "
            "unit_column='hospital_id', "
            "year_column='year', "
            "outcome_column='depreciation_share', "
            "dose_column='medicare_share_1983', "
            "source_id='aha_direct_license'"
            "); "
            "phase2 = load_phase2_contract_bundle(); "
            "print(json.dumps({"
            "'simulate_shape': list(panel.frame.shape), "
            "'phase2_schema_version': phase2['schema_version'], "
            "'phase2_dgp_count': len(phase2['numerical_truth']['dgp_ids']), "
            "'medicare_application_id': load_medicare_pps_manifest()['application_id'], "
            "'medicare_primary_source': load_medicare_pps_source_options()['primary_sources'][0]['id'], "
            "'package_version': contdid.__version__, "
            "'metadata_version': version('contdid-py'), "
            "'scaffold_shape': list(scaffold.panel.frame.shape), "
            "'scaffold_mode': scaffold.metadata['analysis_mode']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "medicare_application_id": "medicare-pps-hospitals",
        "medicare_primary_source": "aha_direct_license",
        "metadata_version": "0.1.0",
        "package_version": "0.1.0",
        "phase2_dgp_count": 5,
        "phase2_schema_version": "0.2",
        "scaffold_mode": "paper-source-aligned",
        "scaffold_shape": [4, 5],
        "simulate_shape": [48, 5],
    }


def test_public_runtime_contract_assets_are_available_from_editable_install(
    shared_editable_root: Path,
) -> None:
    runtime_result = _run_python_with_target_path(
        shared_editable_root,
        (
            "import json; "
            "from importlib.metadata import version; "
            "import pandas as pd; "
            "import contdid; "
            "from contdid import ("
            "ContDIDSpec, "
            "load_phase2_contract_bundle, "
            "load_medicare_pps_manifest, "
            "load_medicare_pps_source_options, "
            "prepare_medicare_pps_panel, "
            "simulate_contdid_data, "
            "validate_spec"
            "); "
            "validate_spec(ContDIDSpec("
            "target_parameter='level', "
            "aggregation='dose', "
            "dose_est_method='parametric', "
            "control_group='nevertreated'"
            ")); "
            "panel = simulate_contdid_data(n=12, dgp_id='SIM-002-linear-dose'); "
            "medicare = pd.DataFrame(["
            "{'hospital_id': 101, 'year': 1980, 'depreciation_share': 4.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1981, 'depreciation_share': 4.10, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1982, 'depreciation_share': 4.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1983, 'depreciation_share': 4.30, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1984, 'depreciation_share': 4.80, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1985, 'depreciation_share': 5.00, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 101, 'year': 1986, 'depreciation_share': 5.20, 'medicare_share_1983': 0.30}, "
            "{'hospital_id': 202, 'year': 1980, 'depreciation_share': 3.20, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1981, 'depreciation_share': 3.10, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1982, 'depreciation_share': 3.00, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1983, 'depreciation_share': 2.90, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1984, 'depreciation_share': 2.80, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1985, 'depreciation_share': 2.70, 'medicare_share_1983': 0.00}, "
            "{'hospital_id': 202, 'year': 1986, 'depreciation_share': 2.60, 'medicare_share_1983': 0.00}"
            "]); "
            "scaffold = prepare_medicare_pps_panel("
            "medicare, "
            "unit_column='hospital_id', "
            "year_column='year', "
            "outcome_column='depreciation_share', "
            "dose_column='medicare_share_1983', "
            "source_id='aha_direct_license'"
            "); "
            "phase2 = load_phase2_contract_bundle(); "
            "print(json.dumps({"
            "'editable_module_file': contdid.__file__, "
            "'simulate_shape': list(panel.frame.shape), "
            "'phase2_schema_version': phase2['schema_version'], "
            "'phase2_dgp_count': len(phase2['numerical_truth']['dgp_ids']), "
            "'medicare_application_id': load_medicare_pps_manifest()['application_id'], "
            "'medicare_primary_source': load_medicare_pps_source_options()['primary_sources'][0]['id'], "
            "'package_version': contdid.__version__, "
            "'metadata_version': version('contdid-py'), "
            "'scaffold_shape': list(scaffold.panel.frame.shape), "
            "'scaffold_mode': scaffold.metadata['analysis_mode']"
            "}, sort_keys=True))"
        ),
    )

    assert runtime_result.returncode == 0, runtime_result.stderr
    payload = json.loads(runtime_result.stdout)
    assert payload == {
        "editable_module_file": str(REPO_ROOT / "contdid-py" / "src" / "contdid" / "__init__.py"),
        "medicare_application_id": "medicare-pps-hospitals",
        "medicare_primary_source": "aha_direct_license",
        "metadata_version": "0.1.0",
        "package_version": "0.1.0",
        "phase2_dgp_count": 5,
        "phase2_schema_version": "0.2",
        "scaffold_mode": "paper-source-aligned",
        "scaffold_shape": [4, 5],
        "simulate_shape": [48, 5],
    }
