"""Resolve runtime assets from either the source tree or an installed wheel."""

from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[2]


def resolve_runtime_asset(
    *,
    package_relative: str,
    repo_relative: str,
) -> Path:
    """Return the packaged asset when installed, else fall back to repo truth."""

    packaged_path = _PACKAGE_ROOT / package_relative
    if packaged_path.exists():
        return packaged_path

    repo_path = _REPO_ROOT / repo_relative
    if repo_path.exists():
        return repo_path

    return packaged_path
