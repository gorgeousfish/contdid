from __future__ import annotations

import copy
import email
import importlib.util
import os
import ast
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
DOCS_CONF = REPO_ROOT / "docs" / "conf.py"
BUILD_BACKEND_PATH = PACKAGE_ROOT / "build_backend" / "contdid_build_backend.py"
EXPECTED_PHASE2_PACKAGE_ASSETS = {
    "contdid/runtime_assets/method_reference_contract.json",
}
EXPECTED_PHASE9_PACKAGE_ASSETS = {
    "contdid/contracts/phase9.py",
    "contdid/runtime_assets/public_api_contract_v1.json",
}
EXPECTED_PHASE11_PACKAGE_ASSETS = {
    "contdid/contracts/phase11.py",
    "contdid/runtime_assets/cck_boundary_contract.json",
}
EXPECTED_PHASE12_PACKAGE_ASSETS = {
    "contdid/contracts/phase12.py",
    "contdid/runtime_assets/data_shape_boundary_contract.json",
}
EXPECTED_PHASE2_SDIST_ASSETS = {
    "runtime-assets/method_reference_contract.json",
}
EXPECTED_PHASE9_SDIST_ASSETS = {
    "src/contdid/contracts/phase9.py",
    "runtime-assets/public_api_contract_v1.json",
}
EXPECTED_PHASE11_SDIST_ASSETS = {
    "src/contdid/contracts/phase11.py",
    "runtime-assets/cck_boundary_contract.json",
}
EXPECTED_PHASE12_SDIST_ASSETS = {
    "src/contdid/contracts/phase12.py",
    "runtime-assets/data_shape_boundary_contract.json",
}
EXPECTED_RELEASE_SDIST_ASSETS = {
    "LICENSE",
}
EXPECTED_PHASE2_FORCE_INCLUDE = {
    "runtime-assets/method_reference_contract.json": (
        "contdid/runtime_assets/method_reference_contract.json"
    ),
}
EXPECTED_PHASE9_FORCE_INCLUDE = {
    "runtime-assets/public_api_contract_v1.json": (
        "contdid/runtime_assets/public_api_contract_v1.json"
    ),
}
EXPECTED_PHASE11_FORCE_INCLUDE = {
    "runtime-assets/cck_boundary_contract.json": (
        "contdid/runtime_assets/cck_boundary_contract.json"
    ),
}
EXPECTED_PHASE12_FORCE_INCLUDE = {
    "runtime-assets/data_shape_boundary_contract.json": (
        "contdid/runtime_assets/data_shape_boundary_contract.json"
    ),
}
EXPECTED_RELEASE_LICENSE_FILES = ["LICENSE"]
EXPECTED_PROJECT_DEPENDENCIES = ["numpy>=2.1", "pandas>=2.2", "pillow>=10", "scipy>=1.12"]
EXPECTED_PROJECT_CLASSIFIERS_TAIL = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
EXPECTED_RUNTIME_DEPENDENCY_IMPORTS = {"PIL", "numpy", "pandas", "scipy"}
EXPECTED_OPTIONAL_RUNTIME_IMPORTS = {"matplotlib", "tqdm"}
IMPORT_ROOT_TO_DISTRIBUTION = {"PIL": "pillow"}
METHOD_PAPER_AUTHORS = {
    "Brantly Callaway",
    "Andrew Goodman-Bacon",
    "Pedro H. C. Sant'Anna",
}
EXPECTED_PROJECT_DESCRIPTION = (
    "Continuous-dose Difference-in-Differences reporting tools for Python."
)


def _project_metadata() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _project_version() -> str:
    return _project_metadata()["version"]


def _wheel_glob() -> str:
    return f"contdid_py-{_project_version()}-*.whl"


def _isolated_installed_pythonpath(target_dir: Path) -> str:
    dependency_paths = {
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    }
    entries = [str(target_dir)]
    entries.extend(
        str(Path(path))
        for path in sorted(path for path in dependency_paths if path)
    )
    return os.pathsep.join(entries)


def _wheel_artifact_name() -> str:
    return f"contdid-py-{_project_version()}-py3-none-any.whl"


def _pip_subprocess_env() -> dict[str, str]:
    return {
        **os.environ,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


class _RuntimeImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.import_roots: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.import_roots.add(alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self.import_roots.add(node.module.split(".", 1)[0])

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return
        self.generic_visit(node)


def _load_build_backend():
    cache_dir = BUILD_BACKEND_PATH.parent / "__pycache__"
    for pyc_path in cache_dir.glob(f"{BUILD_BACKEND_PATH.stem}.cpython-*.pyc"):
        try:
            pyc_path.unlink()
        except FileNotFoundError:
            pass

    spec = importlib.util.spec_from_file_location(
        "contdid_build_backend_under_test",
        BUILD_BACKEND_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def test_phase3_package_metadata_matches_plan_contract() -> None:
    assert PYPROJECT.exists(), f"missing pyproject metadata: {PYPROJECT}"

    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = payload["project"]
    build_system = payload["build-system"]
    pytest_options = payload["tool"]["pytest"]["ini_options"]

    assert build_system["requires"] == []
    assert build_system["build-backend"] == "contdid_build_backend"
    assert build_system["backend-path"] == ["build_backend"]
    assert project["name"] == "contdid-py"
    assert isinstance(project["version"], str)
    assert project["version"]
    assert project["description"] == EXPECTED_PROJECT_DESCRIPTION
    assert project["readme"]["content-type"] == "text/markdown"
    readme_text = project["readme"]["text"]
    assert "continuous-dose Difference-in-Differences" in readme_text
    assert "panel roles, estimand, comparison rule" in readme_text
    assert "contract assets" not in readme_text
    assert "checked Python release surface" not in readme_text
    assert project["requires-python"] == ">=3.11"
    assert project["license"] == "AGPL-3.0-only"
    assert project["license-files"] == EXPECTED_RELEASE_LICENSE_FILES
    authors = project.get("authors", [])
    assert authors == []
    assert not (METHOD_PAPER_AUTHORS & {author.get("name") for author in authors})
    assert project["keywords"] == [
        "causal-inference",
        "continuous-treatment",
        "difference-in-differences",
        "event-study",
    ]
    classifiers = project["classifiers"]
    assert classifiers[0].startswith("Development Status :: ")
    assert classifiers[1:] == EXPECTED_PROJECT_CLASSIFIERS_TAIL
    assert project["urls"] == {
        "Repository": "https://github.com/gorgeousfish/contdid-py",
        "Method paper": "https://arxiv.org/abs/2107.02637",
        "Reference R package": "https://github.com/bcallaway11/contdid",
        "Reference documentation": "https://bcallaway11.github.io/contdid/",
    }
    assert project["dependencies"] == EXPECTED_PROJECT_DEPENDENCIES
    assert pytest_options["pythonpath"] == ["src"]


def test_sphinx_docs_author_metadata_does_not_copy_method_authors() -> None:
    assert DOCS_CONF.exists(), f"missing Sphinx docs config: {DOCS_CONF}"

    docs_conf_text = DOCS_CONF.read_text(encoding="utf-8")
    assert 'project = "contdid-py"' in docs_conf_text
    assert 'author = "Submission Metadata Pending"' in docs_conf_text
    for method_author in METHOD_PAPER_AUTHORS:
        assert method_author not in docs_conf_text


def test_phase3_package_metadata_declares_all_runtime_import_dependencies() -> None:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared_import_roots = {
        str(dependency).split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .split("[", 1)[0]
        .strip()
        .replace("-", "_")
        for dependency in payload["project"]["dependencies"]
    }
    package_import_roots: set[str] = set()
    standard_library_roots = sys.stdlib_module_names.union(sys.builtin_module_names)

    for source_path in sorted((PACKAGE_ROOT / "src" / "contdid").rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        collector = _RuntimeImportCollector()
        collector.visit(tree)
        package_import_roots.update(collector.import_roots)

    external_import_roots = package_import_roots.difference(
        {"contdid"}, standard_library_roots
    )

    required_external_import_roots = external_import_roots.difference(
        EXPECTED_OPTIONAL_RUNTIME_IMPORTS
    )
    external_distribution_roots = {
        IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root)
        for import_root in required_external_import_roots
    }

    assert external_import_roots == (
        EXPECTED_RUNTIME_DEPENDENCY_IMPORTS | EXPECTED_OPTIONAL_RUNTIME_IMPORTS
    )
    assert external_distribution_roots.issubset(declared_import_roots)


def test_offline_build_backend_supports_pep660_editable_install_contract() -> None:
    module = _load_build_backend()

    assert module.get_requires_for_build_editable() == []
    assert module._editable_pth_name() == "__editable__.contdid_py.pth"
    assert module._editable_pth_payload() == b""
    assert callable(module.build_editable)
    assert callable(module.prepare_metadata_for_build_editable)


def test_offline_build_backend_editable_target_install_imports_package(tmp_path) -> None:
    module = _load_build_backend()
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    wheel_dir.mkdir()
    target_dir.mkdir()

    wheel_name = module.build_editable(str(wheel_dir))
    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel_names = set(wheel.namelist())
        assert module._editable_pth_name() in wheel_names
        assert "contdid/__init__.py" in wheel_names
        wheel.extractall(target_dir)

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(Path(contdid.__file__).as_posix())\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_lines = completed.stdout.splitlines()
    assert stdout_lines == [
        _project_version(),
        (PACKAGE_ROOT / "src" / "contdid" / "__init__.py").as_posix(),
    ]


def test_offline_build_backend_editable_target_install_excludes_frontier_helper(
    tmp_path,
) -> None:
    module = _load_build_backend()
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    wheel_dir.mkdir()
    target_dir.mkdir()

    wheel_name = module.build_editable(str(wheel_dir))
    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel.extractall(target_dir)

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import os; os.chdir('/tmp'); "
                "import importlib.util\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(importlib.util.find_spec('phase10_frontier_expectations'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_lines = completed.stdout.splitlines()
    assert stdout_lines == [
        _project_version(),
        "None",
    ]


def test_offline_build_backend_editable_target_install_reports_missing_source(
    tmp_path,
) -> None:
    module = _load_build_backend()
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    missing_source_root = tmp_path / "missing-source-root"
    wheel_dir.mkdir()
    target_dir.mkdir()
    missing_source_root.mkdir()

    missing_init = missing_source_root / "contdid" / "__init__.py"
    missing_module = missing_source_root / "extra_helper.py"
    missing_init.parent.mkdir()
    missing_init.write_text("__version__ = '0.0.0'\n", encoding="utf-8")
    missing_module.write_text("HELPER = True\n", encoding="utf-8")

    def fake_editable_pth_payload() -> bytes:
        return f"{missing_source_root}\n".encode("utf-8")

    def fake_editable_proxy_sources() -> dict[str, bytes]:
        return {
            "contdid/__init__.py": module._editable_proxy_payload(missing_init.parent),
            "extra_helper.py": module._editable_module_proxy_payload(missing_module),
        }

    assert missing_init.exists()
    assert missing_module.exists()

    original_pth_payload = module._editable_pth_payload
    original_proxy_sources = module._editable_proxy_sources
    module._editable_pth_payload = fake_editable_pth_payload
    module._editable_proxy_sources = fake_editable_proxy_sources
    try:
        wheel_name = module.build_editable(str(wheel_dir))
    finally:
        module._editable_pth_payload = original_pth_payload
        module._editable_proxy_sources = original_proxy_sources

    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel.extractall(target_dir)

    shutil.rmtree(missing_source_root)

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    package_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import contdid",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )
    module_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import extra_helper",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert package_probe.returncode != 0
    assert (
        f"ImportError: Cannot load editable package source: {missing_init}"
        in package_probe.stderr
    )
    assert module_probe.returncode != 0
    assert (
        f"ImportError: Cannot load editable module source: {missing_module}"
        in module_probe.stderr
    )


def test_pep660_pip_editable_target_install_matches_readme_user_path(tmp_path) -> None:
    target_dir = tmp_path / "editable-target"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-compile",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target_dir),
            "-e",
            str(PACKAGE_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert completed.returncode == 0, completed.stderr

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "import importlib.metadata as md\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(md.version('contdid-py'))\n"
                "print(contdid.load_phase2_contract_bundle()['phase'])\n"
                "print(contdid.load_phase11_cck_boundary_contract_bundle()['phase'])\n"
                "print(contdid.load_phase12_data_shape_contract_bundle()['phase'])\n"
                "print(contdid.load_public_api_contract_bundle()['phase'])\n"
                "print(contdid.load_medicare_pps_manifest()['published_targets_scope']['selected_sieve_dimension'])\n"
                "print(contdid.load_medicare_pps_source_options()['public_substitutes'][0]['parity_claim_allowed'])\n"
                "print(Path(contdid.__file__).as_posix())\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        _project_version(),
        _project_version(),
        "2",
        "11",
        "12",
        "9",
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime",
        "False",
        (PACKAGE_ROOT / "src" / "contdid" / "__init__.py").as_posix(),
    ]


def test_offline_build_backend_pip_target_install_loads_public_contract_assets(
    tmp_path,
) -> None:
    target_dir = tmp_path / "target"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-compile",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target_dir),
            str(PACKAGE_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert completed.returncode == 0, completed.stderr

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.metadata as md\n"
                "import importlib.util\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(md.version('contdid-py'))\n"
                "print(contdid.load_public_api_contract_bundle()['phase'])\n"
                "print(contdid.load_phase2_contract_bundle()['phase'])\n"
                "print(contdid.load_phase11_cck_boundary_contract_bundle()['phase'])\n"
                "print(contdid.load_phase12_data_shape_contract_bundle()['phase'])\n"
                "print(contdid.load_medicare_pps_manifest()['published_targets_scope']['selected_sieve_dimension'])\n"
                "print(contdid.load_medicare_pps_source_options()['public_substitutes'][0]['parity_claim_allowed'])\n"
                "print(importlib.util.find_spec('phase10_frontier_expectations'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        _project_version(),
        _project_version(),
        "9",
        "2",
        "11",
        "12",
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime",
        "False",
        "None",
    ]


def test_standard_pep517_frontend_builds_importable_wheel_with_contract_assets(
    tmp_path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    wheel_dir.mkdir()
    target_dir.mkdir()

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(PACKAGE_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert build.returncode == 0, build.stderr
    wheel_paths = sorted(wheel_dir.glob(_wheel_glob()))
    assert len(wheel_paths) == 1, sorted(path.name for path in wheel_dir.iterdir())

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(target_dir),
            str(wheel_paths[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert install.returncode == 0, install.stderr

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": _isolated_installed_pythonpath(target_dir),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.metadata as md\n"
                "import importlib.util\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(md.version('contdid-py'))\n"
                "print(contdid.load_public_api_contract_bundle()['phase'])\n"
                "print(contdid.load_phase2_contract_bundle()['phase'])\n"
                "print(contdid.load_phase11_cck_boundary_contract_bundle()['phase'])\n"
                "print(contdid.load_phase12_data_shape_contract_bundle()['phase'])\n"
                "print(contdid.load_medicare_pps_manifest()['published_targets_scope']['selected_sieve_dimension'])\n"
                "print(contdid.load_medicare_pps_source_options()['public_substitutes'][0]['parity_claim_allowed'])\n"
                "print(importlib.util.find_spec('phase10_frontier_expectations'))\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=tmp_path,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        _project_version(),
        _project_version(),
        "9",
        "2",
        "11",
        "12",
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime",
        "False",
        "None",
    ]


def test_sdist_roundtrip_builds_importable_wheel_with_contract_assets(
    tmp_path,
) -> None:
    module = _load_build_backend()
    sdist_dir = tmp_path / "sdist"
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    sdist_dir.mkdir()
    wheel_dir.mkdir()
    target_dir.mkdir()

    sdist_name = module.build_sdist(str(sdist_dir))
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(sdist_dir / sdist_name),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert build.returncode == 0, build.stderr
    wheel_paths = sorted(wheel_dir.glob(_wheel_glob()))
    assert len(wheel_paths) == 1, sorted(path.name for path in wheel_dir.iterdir())

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(target_dir),
            str(wheel_paths[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert install.returncode == 0, install.stderr

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(target_dir),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as md\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(md.version('contdid-py'))\n"
                "print(contdid.load_public_api_contract_bundle()['phase'])\n"
                "print(contdid.load_phase2_contract_bundle()['phase'])\n"
                "print(contdid.load_phase11_cck_boundary_contract_bundle()['phase'])\n"
                "print(contdid.load_phase12_data_shape_contract_bundle()['phase'])\n"
                "print(contdid.load_medicare_pps_manifest()['published_targets_scope']['selected_sieve_dimension'])\n"
                "print(contdid.load_medicare_pps_source_options()['public_substitutes'][0]['parity_claim_allowed'])\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        _project_version(),
        _project_version(),
        "9",
        "2",
        "11",
        "12",
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime",
        "False",
    ]


def test_sdist_roundtrip_with_isolated_pep517_frontend_builds_importable_wheel(
    tmp_path,
) -> None:
    module = _load_build_backend()
    sdist_dir = tmp_path / "sdist"
    wheel_dir = tmp_path / "wheel"
    target_dir = tmp_path / "target"
    sdist_dir.mkdir()
    wheel_dir.mkdir()
    target_dir.mkdir()

    sdist_name = module.build_sdist(str(sdist_dir))
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(sdist_dir / sdist_name),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert build.returncode == 0, build.stderr
    wheel_paths = sorted(wheel_dir.glob(_wheel_glob()))
    assert len(wheel_paths) == 1, sorted(path.name for path in wheel_dir.iterdir())

    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(target_dir),
            str(wheel_paths[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_pip_subprocess_env(),
    )

    assert install.returncode == 0, install.stderr

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(target_dir),
    }
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as md\n"
                "import contdid\n"
                "print(contdid.__version__)\n"
                "print(md.version('contdid-py'))\n"
                "print(contdid.load_public_api_contract_bundle()['phase'])\n"
                "print(contdid.load_phase2_contract_bundle()['phase'])\n"
                "print(contdid.load_phase11_cck_boundary_contract_bundle()['phase'])\n"
                "print(contdid.load_phase12_data_shape_contract_bundle()['phase'])\n"
                "print(contdid.load_medicare_pps_manifest()['published_targets_scope']['selected_sieve_dimension'])\n"
                "print(contdid.load_medicare_pps_source_options()['public_substitutes'][0]['parity_claim_allowed'])\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [
        _project_version(),
        _project_version(),
        "9",
        "2",
        "11",
        "12",
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime",
        "False",
    ]


def test_offline_build_backend_ignores_common_generated_artifacts() -> None:
    module = _load_build_backend()

    ignored_paths = [
        Path(".DS_Store"),
        Path("Thumbs.db"),
        Path("src/contdid/.coverage"),
        Path("src/contdid/.tmp-release-gate.pid"),
        Path("src/contdid/_phase10_frontier_root_logic.py"),
        Path("src/phase10_frontier_expectations.py"),
        Path(".tmp_lane13_control_group_panel.csv"),
        Path("src/contdid/.tmp_local_probe.csv"),
        Path("src/contdid/.venv/bin/python"),
        Path("src/contdid/.tox/py311/log"),
        Path("src/contdid/.nox/session/log"),
        Path("src/contdid/pip-wheel-metadata/metadata.json"),
        Path("src/contdid/coverage.xml"),
        Path("src/contdid/.pytest_cache/v/cache/nodeids"),
        Path("src/contdid/.ruff_cache/0.13.2/cache"),
        Path("src/contdid/.mypy_cache/3.11/cache.db"),
        Path("src/contdid/.hypothesis/examples/example.db"),
        Path("src/contdid/htmlcov/index.html"),
        Path("src/contdid/build/lib/contdid/__init__.py"),
        Path(f"src/contdid/dist/{_wheel_artifact_name()}"),
        Path("src/contdid/local-release-copy.whl"),
        Path("src/contdid/local-release-copy.tar.gz"),
        Path("src/contdid/local-release-copy.egg"),
        Path("src/contdid/pkg.egg-info/PKG-INFO"),
    ]

    for path in ignored_paths:
        assert module._is_ignored_build_file(path), path


def test_offline_build_backend_keeps_package_files_when_checkout_parent_looks_generated(
    tmp_path,
) -> None:
    module = _load_build_backend()
    package_root = tmp_path / "build" / "checkout" / "src" / "contdid"
    package_file = package_root / "__init__.py"
    cache_file = package_root / "__pycache__" / "stale.cpython-313.pyc"
    generated_file = package_root / "build" / "lib" / "contdid" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    cache_file.parent.mkdir(parents=True)
    generated_file.parent.mkdir(parents=True)
    package_file.write_text("__version__ = '0.0.0'\n", encoding="utf-8")
    cache_file.write_bytes(b"stale bytecode\n")
    generated_file.write_text("# generated build output\n", encoding="utf-8")

    assert list(module._iter_files(package_root)) == [package_file]


def test_offline_build_backend_excludes_transient_release_locks_from_artifacts(
    tmp_path,
) -> None:
    module = _load_build_backend()
    wheel_dir = tmp_path / "wheel"
    sdist_dir = tmp_path / "sdist"
    wheel_dir.mkdir()
    sdist_dir.mkdir()
    transient_lock = PACKAGE_ROOT / "src" / "contdid" / ".tmp-release-gate.pid"

    try:
        transient_lock.write_text("999999\n", encoding="utf-8")
        wheel_name = module.build_wheel(str(wheel_dir))
        sdist_name = module.build_sdist(str(sdist_dir))

        with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
            wheel_names = set(wheel.namelist())
        with tarfile.open(sdist_dir / sdist_name, "r:gz") as sdist:
            sdist_names = set(sdist.getnames())
    finally:
        transient_lock.unlink(missing_ok=True)

    assert not any(name.endswith(".tmp-release-gate.pid") for name in wheel_names)
    assert not any(name.endswith(".tmp-release-gate.pid") for name in sdist_names)


def test_offline_build_backend_rejects_symlinked_package_payloads(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    outside_payload = tmp_path / "outside-package-project-payload.txt"
    package_root = PACKAGE_ROOT / f"_tmp_symlinked_payload_package_{tmp_path.name}"
    symlink_payload = package_root / "_tmp_symlinked_payload.txt"

    outside_payload.write_text("outside package project payload\n", encoding="utf-8")
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    try:
        symlink_payload.symlink_to(outside_payload)
    except OSError as exc:
        shutil.rmtree(package_root, ignore_errors=True)
        pytest.skip(f"symlink creation unavailable: {exc}")

    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = [package_root.name]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)
    try:
        with pytest.raises(ValueError, match="package payload cannot be a symlink"):
            module.build_wheel(str(tmp_path))
        with pytest.raises(ValueError, match="package payload cannot be a symlink"):
            module.build_sdist(str(tmp_path))
    finally:
        shutil.rmtree(package_root, ignore_errors=True)


def test_offline_build_backend_rejects_symlinked_package_roots(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    symlink_package_root = PACKAGE_ROOT / "_tmp_symlinked_package_root"

    try:
        symlink_package_root.symlink_to(PACKAGE_ROOT / "src" / "contdid")
    except OSError as exc:
        symlink_package_root.unlink(missing_ok=True)
        pytest.skip(f"symlink creation unavailable: {exc}")

    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = [
        symlink_package_root.name,
    ]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    try:
        with pytest.raises(ValueError, match="package source cannot be a symlink"):
            module.build_wheel(str(tmp_path))
        with pytest.raises(ValueError, match="package source cannot be a symlink"):
            module.build_sdist(str(tmp_path))
    finally:
        symlink_package_root.unlink(missing_ok=True)


def test_offline_build_backend_rejects_symlinked_force_include_sources(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    symlink_source = PACKAGE_ROOT / "_tmp_force_include_symlink.json"

    try:
        symlink_source.symlink_to(
            PACKAGE_ROOT / "contracts" / "phase2" / "paper_truth_contract.json"
        )
    except OSError as exc:
        symlink_source.unlink(missing_ok=True)
        pytest.skip(f"symlink creation unavailable: {exc}")

    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = []
    project["tool"]["contdid-build"]["force-include"] = {
        symlink_source.name: "contdid/_tmp_force_include_symlink.json",
    }
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    try:
        with pytest.raises(ValueError, match="force-include source cannot be a symlink"):
            module.build_wheel(str(tmp_path))
        with pytest.raises(ValueError, match="force-include source cannot be a symlink"):
            module.build_sdist(str(tmp_path))
    finally:
        symlink_source.unlink(missing_ok=True)


def test_offline_build_backend_rejects_license_file_symlinks_to_outside_project(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    outside_payload = PACKAGE_ROOT.parent / "_tmp_outside_license_payload.txt"
    symlink_payload = PACKAGE_ROOT / "_tmp_license_symlink_payload.txt"
    outside_payload.write_text("outside license payload\n", encoding="utf-8")

    try:
        symlink_payload.symlink_to(outside_payload)
    except OSError as exc:
        outside_payload.unlink(missing_ok=True)
        pytest.skip(f"symlink creation unavailable: {exc}")

    project = copy.deepcopy(module._load_pyproject())
    project["project"]["license-files"] = [symlink_payload.name]
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    try:
        with pytest.raises(ValueError, match="license-file source cannot be a symlink"):
            module.build_wheel(str(tmp_path))
        with pytest.raises(ValueError, match="license-file source cannot be a symlink"):
            module.build_sdist(str(tmp_path))
    finally:
        symlink_payload.unlink(missing_ok=True)
        outside_payload.unlink(missing_ok=True)


def test_offline_build_backend_emits_checked_release_metadata() -> None:
    module = _load_build_backend()
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    metadata = email.message_from_bytes(module._metadata_payload())
    metadata_text = module._metadata_payload().decode("utf-8")

    assert metadata["Metadata-Version"] == "2.4"
    assert metadata["Name"] == "contdid-py"
    assert metadata["Version"] == project["version"]
    assert metadata["Summary"] == EXPECTED_PROJECT_DESCRIPTION
    assert metadata["Requires-Python"] == ">=3.11"
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert metadata["License-Expression"] == "AGPL-3.0-only"
    assert metadata["License"] is None
    assert metadata.get_all("License-File") == EXPECTED_RELEASE_LICENSE_FILES
    assert metadata.get_all("Author-email") is None
    assert metadata.get_all("Requires-Dist") == EXPECTED_PROJECT_DEPENDENCIES
    assert metadata.get_all("Classifier") == project["classifiers"]
    assert metadata.get_all("Project-URL") == [
        "Repository, https://github.com/gorgeousfish/contdid-py",
        "Method paper, https://arxiv.org/abs/2107.02637",
        "Reference R package, https://github.com/bcallaway11/contdid",
        "Reference documentation, https://bcallaway11.github.io/contdid/",
    ]
    assert "Keywords: causal-inference, continuous-treatment" in metadata_text
    assert "`contdid-py` provides Python tools" in metadata_text
    for method_author in METHOD_PAPER_AUTHORS:
        assert method_author not in metadata_text
    assert "checked Python release surface" not in metadata_text


def test_offline_build_backend_prepare_metadata_removes_stale_dist_info_payloads(
    tmp_path,
) -> None:
    module = _load_build_backend()
    stale_dist_info = tmp_path / module._dist_info_name()
    stale_generated_license = stale_dist_info / "licenses" / "stale.whl"
    stale_generated_license.parent.mkdir(parents=True)
    stale_generated_license.write_bytes(b"stale generated metadata payload\n")
    (stale_dist_info / "STALE").write_text("stale metadata file\n", encoding="utf-8")

    dist_info_name = module.prepare_metadata_for_build_wheel(str(tmp_path))

    dist_info = tmp_path / dist_info_name
    assert dist_info == stale_dist_info
    assert (dist_info / "METADATA").is_file()
    assert (dist_info / "WHEEL").is_file()
    assert (dist_info / "licenses" / "LICENSE").is_file()
    assert not stale_generated_license.exists()
    assert not (dist_info / "STALE").exists()


def test_offline_build_backend_wheel_reuses_prepared_metadata_directory(
    tmp_path,
) -> None:
    module = _load_build_backend()
    metadata_root = tmp_path / "metadata"
    wheel_dir = tmp_path / "wheel"
    metadata_root.mkdir()
    wheel_dir.mkdir()

    dist_info_name = module.prepare_metadata_for_build_wheel(str(metadata_root))
    prepared_dist_info = metadata_root / dist_info_name
    prepared_metadata = prepared_dist_info / "METADATA"
    prepared_wheel = prepared_dist_info / "WHEEL"
    prepared_marker = prepared_dist_info / "contdid-prepared-metadata.marker"
    prepared_metadata.write_bytes(prepared_metadata.read_bytes() + b"X-ContDID-Prepared: yes\n")
    prepared_wheel.write_bytes(prepared_wheel.read_bytes() + b"X-ContDID-Wheel: prepared\n")
    prepared_marker.write_text("prepared metadata must be preserved\n", encoding="utf-8")

    wheel_name = module.build_wheel(
        str(wheel_dir),
        metadata_directory=str(prepared_dist_info),
    )

    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel_payloads = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if name.startswith(f"{dist_info_name}/")
        }

    assert wheel_payloads[f"{dist_info_name}/METADATA"] == prepared_metadata.read_bytes()
    assert wheel_payloads[f"{dist_info_name}/WHEEL"] == prepared_wheel.read_bytes()
    assert wheel_payloads[f"{dist_info_name}/contdid-prepared-metadata.marker"] == (
        prepared_marker.read_bytes()
    )
    assert f"{dist_info_name}/RECORD" in wheel_payloads


def test_offline_build_backend_editable_reuses_prepared_metadata_directory(
    tmp_path,
) -> None:
    module = _load_build_backend()
    metadata_root = tmp_path / "metadata"
    wheel_dir = tmp_path / "wheel"
    metadata_root.mkdir()
    wheel_dir.mkdir()

    dist_info_name = module.prepare_metadata_for_build_editable(str(metadata_root))
    prepared_dist_info = metadata_root / dist_info_name
    prepared_metadata = prepared_dist_info / "METADATA"
    prepared_marker = prepared_dist_info / "contdid-editable-metadata.marker"
    prepared_metadata.write_bytes(
        prepared_metadata.read_bytes() + b"X-ContDID-Editable-Prepared: yes\n"
    )
    prepared_marker.write_text("editable metadata must be preserved\n", encoding="utf-8")

    wheel_name = module.build_editable(
        str(wheel_dir),
        metadata_directory=str(prepared_dist_info),
    )

    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel_payloads = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if name.startswith(f"{dist_info_name}/")
        }

    assert wheel_payloads[f"{dist_info_name}/METADATA"] == prepared_metadata.read_bytes()
    assert wheel_payloads[f"{dist_info_name}/contdid-editable-metadata.marker"] == (
        prepared_marker.read_bytes()
    )
    assert f"{dist_info_name}/RECORD" in wheel_payloads


def test_offline_build_backend_rejects_symlinked_prepared_metadata_directory(
    tmp_path,
) -> None:
    module = _load_build_backend()
    real_root = tmp_path / "real"
    linked_root = tmp_path / "linked"
    wheel_dir = tmp_path / "wheel"
    real_root.mkdir()
    linked_root.mkdir()
    wheel_dir.mkdir()

    dist_info_name = module.prepare_metadata_for_build_wheel(str(real_root))
    real_dist_info = real_root / dist_info_name
    linked_dist_info = linked_root / dist_info_name
    linked_dist_info.symlink_to(real_dist_info, target_is_directory=True)

    with pytest.raises(ValueError, match="metadata_directory cannot be a symlink"):
        module.build_wheel(str(wheel_dir), metadata_directory=str(linked_dist_info))
    with pytest.raises(ValueError, match="metadata_directory cannot be a symlink"):
        module.build_editable(str(wheel_dir), metadata_directory=str(linked_dist_info))


def test_phase3_package_metadata_declares_current_interpreter_classifier() -> None:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    classifiers = payload["project"]["classifiers"]
    current_classifier = f"Programming Language :: Python :: {sys.version_info.major}.{sys.version_info.minor}"

    assert current_classifier in classifiers


def test_offline_build_backend_builds_reproducible_sdist(tmp_path) -> None:
    module = _load_build_backend()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_name = module.build_sdist(str(first_dir))
    second_name = module.build_sdist(str(second_dir))

    assert first_name == second_name
    assert (first_dir / first_name).read_bytes() == (second_dir / second_name).read_bytes()


def test_offline_build_backend_ships_checked_contract_and_license_assets(tmp_path) -> None:
    module = _load_build_backend()
    wheel_dir = tmp_path / "wheel"
    sdist_dir = tmp_path / "sdist"
    wheel_dir.mkdir()
    sdist_dir.mkdir()

    wheel_name = module.build_wheel(str(wheel_dir))
    sdist_name = module.build_sdist(str(sdist_dir))

    with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
        wheel_names = set(wheel.namelist())
    with tarfile.open(sdist_dir / sdist_name, "r:gz") as sdist:
        root_name = sdist_name.removesuffix(".tar.gz")
        sdist_names = {
            name.removeprefix(f"{root_name}/")
            for name in sdist.getnames()
            if name.startswith(f"{root_name}/")
        }

    assert EXPECTED_PHASE2_PACKAGE_ASSETS.issubset(wheel_names)
    assert EXPECTED_PHASE9_PACKAGE_ASSETS.issubset(wheel_names)
    assert EXPECTED_PHASE11_PACKAGE_ASSETS.issubset(wheel_names)
    assert EXPECTED_PHASE12_PACKAGE_ASSETS.issubset(wheel_names)
    assert not any("phase10" in name for name in wheel_names)
    assert any(
        name.endswith(".dist-info/licenses/LICENSE") for name in wheel_names
    ), wheel_names
    assert EXPECTED_PHASE2_SDIST_ASSETS.issubset(sdist_names)
    assert EXPECTED_PHASE9_SDIST_ASSETS.issubset(sdist_names)
    assert EXPECTED_PHASE11_SDIST_ASSETS.issubset(sdist_names)
    assert EXPECTED_PHASE12_SDIST_ASSETS.issubset(sdist_names)
    assert EXPECTED_RELEASE_SDIST_ASSETS.issubset(sdist_names)
    assert not any("phase10" in name for name in sdist_names)


def test_pyproject_force_include_maps_complete_phase2_and_phase9_truth_contract_bundles() -> None:
    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    force_include = payload["tool"]["contdid-build"]["force-include"]

    assert {
        source: destination
        for source, destination in force_include.items()
        if source == "runtime-assets/method_reference_contract.json"
    } == EXPECTED_PHASE2_FORCE_INCLUDE
    assert {
        source: destination
        for source, destination in force_include.items()
        if source == "runtime-assets/public_api_contract_v1.json"
    } == EXPECTED_PHASE9_FORCE_INCLUDE
    assert {
        source: destination
        for source, destination in force_include.items()
        if source == "runtime-assets/cck_boundary_contract.json"
    } == EXPECTED_PHASE11_FORCE_INCLUDE
    assert {
        source: destination
        for source, destination in force_include.items()
        if source == "runtime-assets/data_shape_boundary_contract.json"
    } == EXPECTED_PHASE12_FORCE_INCLUDE
    assert not any(source.startswith("contracts/phase") for source in force_include)
    assert payload["project"]["license-files"] == EXPECTED_RELEASE_LICENSE_FILES
    assert "LICENSE" not in force_include

    for source in EXPECTED_PHASE2_FORCE_INCLUDE:
        assert (PACKAGE_ROOT / source).exists(), source
    for source in EXPECTED_PHASE9_FORCE_INCLUDE:
        assert (PACKAGE_ROOT / source).exists(), source
    for source in EXPECTED_PHASE11_FORCE_INCLUDE:
        assert (PACKAGE_ROOT / source).exists(), source
    for source in EXPECTED_PHASE12_FORCE_INCLUDE:
        assert (PACKAGE_ROOT / source).exists(), source
    for source in EXPECTED_RELEASE_LICENSE_FILES:
        assert (PACKAGE_ROOT / source).exists(), source


def test_offline_build_backend_rejects_wheel_force_include_payload_collisions(
    monkeypatch,
) -> None:
    module = _load_build_backend()
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["force-include"] = {
        "LICENSE": "contdid/__init__.py",
    }
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(ValueError, match="build payload collision"):
        module._wheel_sources()


def test_offline_build_backend_rejects_wheel_force_include_path_prefix_collisions(
    monkeypatch,
) -> None:
    module = _load_build_backend()
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["force-include"] = {
        "LICENSE": "contdid",
    }
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(ValueError, match="build payload path collision"):
        module._wheel_sources()


def test_offline_build_backend_rejects_sdist_force_include_payload_collisions(
    monkeypatch,
) -> None:
    module = _load_build_backend()
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["force-include"] = {
        "src/contdid/__init__.py": "contdid/__init__.py",
    }
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(ValueError, match="build payload collision"):
        module._sdist_sources()


def test_offline_build_backend_rejects_absolute_package_roots_outside_project(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    outside_package = tmp_path / "outside_contdid_payload"
    outside_package.mkdir()
    (outside_package / "__init__.py").write_text(
        "ESCAPED_PACKAGE = True\n",
        encoding="utf-8",
    )
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = [str(outside_package)]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(ValueError, match="package source must stay within project root"):
        module.build_wheel(str(tmp_path))
    with pytest.raises(ValueError, match="package source must stay within project root"):
        module.build_sdist(str(tmp_path))


def test_offline_build_backend_rejects_relative_package_roots_outside_project(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    outside_package = PACKAGE_ROOT.parent / ".tmp_outside_contdid_payload"
    outside_package.mkdir()
    (outside_package / "__init__.py").write_text(
        "ESCAPED_PACKAGE = True\n",
        encoding="utf-8",
    )
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = [
        "../.tmp_outside_contdid_payload",
    ]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    try:
        with pytest.raises(ValueError, match="package source must stay within project root"):
            module.build_wheel(str(tmp_path))
        with pytest.raises(ValueError, match="package source must stay within project root"):
            module.build_sdist(str(tmp_path))
    finally:
        shutil.rmtree(outside_package, ignore_errors=True)


@pytest.mark.parametrize("package_root", ["C:/contdid_payload", "C:", "src\\contdid"])
def test_offline_build_backend_rejects_platform_specific_package_roots(
    package_root,
) -> None:
    module = _load_build_backend()

    with pytest.raises(ValueError, match="package source must use POSIX relative paths"):
        module._validated_package_root(package_root)


@pytest.mark.parametrize(
    "source",
    ["C:/force_include_payload.json", "C:", "contracts\\phase2\\paper_truth_contract.json"],
)
def test_offline_build_backend_rejects_platform_specific_force_include_sources(
    source,
) -> None:
    module = _load_build_backend()

    with pytest.raises(
        ValueError,
        match="force-include source must use POSIX relative paths",
    ):
        module._force_include_source(source)


def test_offline_build_backend_rejects_missing_package_roots(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = ["src/missing_contdid_package"]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(FileNotFoundError, match="package source does not exist"):
        module.build_wheel(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="package source does not exist"):
        module.build_sdist(str(tmp_path))


@pytest.mark.parametrize(
    "package_root",
    [
        "build/_tmp_generated_contdid_payload",
        "dist/_tmp_generated_contdid_payload",
    ],
)
def test_offline_build_backend_rejects_generated_package_roots(
    package_root,
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_build_backend()
    project = copy.deepcopy(module._load_pyproject())
    project["tool"]["contdid-build"]["packages"] = [package_root]
    project["tool"]["contdid-build"]["force-include"] = {}
    monkeypatch.setattr(module, "_load_pyproject", lambda: project)

    with pytest.raises(ValueError, match="package source cannot point at generated build output"):
        module.build_wheel(str(tmp_path))
    with pytest.raises(ValueError, match="package source cannot point at generated build output"):
        module.build_sdist(str(tmp_path))


def test_contdid_import_surface_exposes_phase3_shell_symbols() -> None:
    import contdid

    expected_exports = {
        "__version__",
        "PanelData",
        "ContDIDSpec",
        "ContDIDResult",
        "ContDIDValidationError",
        "validate_panel_data",
        "validate_spec",
        "simulate_contdid_data",
        "MethodReferenceContractError",
        "CCKBoundaryContractError",
        "load_method_reference_contract",
        "load_cck_boundary_contract",
        "validate_method_reference_contract",
        "validate_cck_boundary_contract",
        "Phase2ContractError",
        "Phase11ContractError",
        "load_phase2_contract_bundle",
        "load_phase11_cck_boundary_contract_bundle",
        "validate_phase2_contract_bundle",
        "validate_phase11_cck_boundary_contract_bundle",
    }

    for export_name in expected_exports:
        assert hasattr(contdid, export_name), export_name

    payload = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert contdid.__version__ == payload["project"]["version"]
    assert "__version__" in contdid.__all__
