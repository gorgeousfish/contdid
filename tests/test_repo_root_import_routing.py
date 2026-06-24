from __future__ import annotations

from pathlib import Path


def test_pytest_imports_checked_source_tree_from_repo_root() -> None:
    import contdid

    package_file = Path(contdid.__file__).resolve()
    expected_source_root = Path(__file__).resolve().parents[1] / "src" / "contdid"

    assert package_file.is_relative_to(expected_source_root)
