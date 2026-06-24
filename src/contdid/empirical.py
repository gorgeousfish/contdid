"""Medicare PPS-style scaffold helpers for public examples."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ._asset_paths import resolve_runtime_asset
from .data import PanelData
from .validation import ContDIDValidationError, validate_panel_data

_MEDICARE_PPS_ROOT = resolve_runtime_asset(
    package_relative="reproduction/medicare_pps",
    repo_relative="reproduction/medicare_pps",
)
_MEDICARE_PPS_MANIFEST_PATH = _MEDICARE_PPS_ROOT / "manifest.json"
_MEDICARE_PPS_SOURCE_OPTIONS_PATH = _MEDICARE_PPS_ROOT / "source_options.json"
_EXPECTED_PUBLISHED_TARGETS_SCOPE = {
    "source": ("paper reported targets from arXiv-2107.02637v7/main.tex:864-880,899-899,934-934"),
    "status": (
        "unmet parity targets until licensed AHA inputs and "
        "paper-supported adaptive CCK estimation are available"
    ),
    "selected_sieve_dimension": (
        "paper reported CCK target; not produced by the current cck_polynomial_backend runtime"
    ),
}
_EXPECTED_PUBLIC_SUBSTITUTE_POLICY = {
    "parity_viability": "insufficient-for-1980-1986-paper-parity",
    "parity_claim_allowed": False,
    "allowed_use": "descriptive-or-scaffold-only",
}
_SOURCE_OPTIONS_REQUIRED_FIELDS = ("id", "provider", "url")
_MEDICARE_PPS_YEAR_ERROR = (
    "medicare PPS scaffold year values must be finite integer calendar years"
)
_MEDICARE_PPS_OUTCOME_ERROR = (
    "medicare PPS scaffold outcome values must be finite numeric depreciation-share values"
)
_MEDICARE_PPS_DOSE_ERROR = (
    "medicare PPS scaffold dose values must be finite numeric 1983 Medicare "
    "inpatient shares on the [0, 1] scale"
)
_MEDICARE_PPS_DOSE_CONSTANCY_ERROR = (
    "medicare PPS scaffold dose column must be the unit-constant 1983 Medicare inpatient share"
)


@dataclass(slots=True)
class EmpiricalScaffoldResult:
    """Package-ready panel plus metadata for empirical scaffold workflows."""

    panel: PanelData
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_medicare_pps_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContDIDValidationError("medicare PPS manifest must be a JSON object")
    if payload.get("application_id") != "medicare-pps-hospitals":
        raise ContDIDValidationError(
            "medicare PPS manifest application_id must be medicare-pps-hospitals"
        )
    if payload.get("published_targets_scope") != _EXPECTED_PUBLISHED_TARGETS_SCOPE:
        raise ContDIDValidationError(
            "medicare PPS manifest must freeze published_targets_scope for "
            "paper-reported CCK targets"
        )
    return payload


def _validate_medicare_pps_source_options(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContDIDValidationError("medicare PPS source options must be a JSON object")
    if payload.get("application_id") != "medicare-pps-hospitals":
        raise ContDIDValidationError(
            "medicare PPS source options application_id must be medicare-pps-hospitals"
        )
    for section_name in ("primary_sources", "public_substitutes"):
        entries = payload.get(section_name)
        if not isinstance(entries, list) or not entries:
            raise ContDIDValidationError(
                f"medicare PPS source options {section_name} must be a non-empty list"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                raise ContDIDValidationError(
                    f"medicare PPS source options {section_name} entries must be JSON objects"
                )
            for required_field in _SOURCE_OPTIONS_REQUIRED_FIELDS:
                if required_field not in entry:
                    source_id = entry.get("id", "<missing id>")
                    raise ContDIDValidationError(
                        "medicare PPS source options "
                        f"{section_name} entry {source_id} is missing {required_field}"
                    )
    public_substitutes = payload.get("public_substitutes")
    for entry in public_substitutes:  # type: ignore[union-attr]
        for policy_field, expected_value in _EXPECTED_PUBLIC_SUBSTITUTE_POLICY.items():
            if entry.get(policy_field) != expected_value:
                source_id = entry.get("id", "<missing id>")
                raise ContDIDValidationError(
                    "medicare PPS source options public substitute "
                    f"{source_id} must freeze {policy_field}={expected_value!r}"
                )
    return payload


@lru_cache(maxsize=1)
def _load_medicare_pps_manifest_cached() -> dict[str, Any]:
    payload = json.loads(_MEDICARE_PPS_MANIFEST_PATH.read_text(encoding="utf-8"))
    return _validate_medicare_pps_manifest(payload)


def load_medicare_pps_manifest() -> dict[str, Any]:
    """Load the canonical Medicare PPS reproduction manifest."""

    return copy.deepcopy(_load_medicare_pps_manifest_cached())


@lru_cache(maxsize=1)
def _load_medicare_pps_source_options_cached() -> dict[str, Any]:
    payload = json.loads(_MEDICARE_PPS_SOURCE_OPTIONS_PATH.read_text(encoding="utf-8"))
    return _validate_medicare_pps_source_options(payload)


def load_medicare_pps_source_options() -> dict[str, Any]:
    """Load the machine-checkable Medicare PPS source packet."""

    return copy.deepcopy(_load_medicare_pps_source_options_cached())


@lru_cache(maxsize=1)
def _medicare_pps_required_years() -> tuple[int, ...]:
    manifest = load_medicare_pps_manifest()
    return tuple(manifest["phase7_gate"]["required_years"])


def _check_medicare_pps_input_columns(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    year_column: str,
    outcome_column: str,
    dose_column: str,
) -> None:
    required_columns = (unit_column, year_column, outcome_column, dose_column)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ContDIDValidationError(
            f"medicare PPS scaffold input is missing required columns: {joined}"
        )


def _contains_boolean_values(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return True
    raw_values = values.to_numpy(dtype=object, copy=False)
    return any(isinstance(value, (bool, np.bool_)) for value in raw_values)


def _finite_numeric_values(values: pd.Series, *, message: str) -> pd.Series:
    if _contains_boolean_values(values) or not pd.api.types.is_numeric_dtype(values):
        raise ContDIDValidationError(message)
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all():
        raise ContDIDValidationError(message)
    return numeric


def _check_medicare_pps_input_values(
    frame: pd.DataFrame,
    *,
    year_column: str,
    outcome_column: str,
    dose_column: str,
) -> None:
    year_values = _finite_numeric_values(frame[year_column], message=_MEDICARE_PPS_YEAR_ERROR)
    year_array = year_values.to_numpy(dtype=float, copy=False)
    if not np.equal(year_array, np.rint(year_array)).all():
        raise ContDIDValidationError(_MEDICARE_PPS_YEAR_ERROR)

    _finite_numeric_values(frame[outcome_column], message=_MEDICARE_PPS_OUTCOME_ERROR)
    dose_values = _finite_numeric_values(frame[dose_column], message=_MEDICARE_PPS_DOSE_ERROR)
    if ((dose_values < 0) | (dose_values > 1)).any():
        raise ContDIDValidationError(_MEDICARE_PPS_DOSE_ERROR)


def resolve_medicare_pps_source(source_id: str) -> dict[str, Any]:
    """Resolve a source identifier from the Medicare PPS source packet."""

    packet = load_medicare_pps_source_options()
    for section_name, source_type in (
        ("primary_sources", "paper-source-aligned"),
        ("public_substitutes", "descriptive-or-scaffold-only"),
    ):
        for entry in packet[section_name]:
            if entry["id"] == source_id:
                resolved = dict(entry)
                resolved["source_type"] = source_type
                return resolved
    raise ContDIDValidationError(f"unknown Medicare PPS source_id: {source_id!r}")


def load_medicare_pps_example_panel() -> pd.DataFrame:
    """Return the staged annual hospital-year frame used in public examples.

    The returned data frame is a small software-use example with five hospitals
    and annual rows for 1980--1986. It is not licensed AHA Medicare PPS
    replication data and should not be used for empirical PPS claims.
    """

    yearly_values = {
        101: {
            "medicare_share_1983": 0.25,
            "depreciation_share": {
                1980: 3.90,
                1981: 4.00,
                1982: 4.10,
                1983: 4.20,
                1984: 4.70,
                1985: 4.90,
                1986: 5.10,
            },
        },
        202: {
            "medicare_share_1983": 0.35,
            "depreciation_share": {
                1980: 4.20,
                1981: 4.25,
                1982: 4.30,
                1983: 4.35,
                1984: 4.95,
                1985: 5.10,
                1986: 5.25,
            },
        },
        505: {
            "medicare_share_1983": 0.45,
            "depreciation_share": {
                1980: 4.40,
                1981: 4.45,
                1982: 4.50,
                1983: 4.55,
                1984: 5.30,
                1985: 5.40,
                1986: 5.50,
            },
        },
        303: {
            "medicare_share_1983": 0.00,
            "depreciation_share": {
                1980: 3.10,
                1981: 3.00,
                1982: 2.95,
                1983: 2.90,
                1984: 2.85,
                1985: 2.80,
                1986: 2.75,
            },
        },
        404: {
            "medicare_share_1983": 0.00,
            "depreciation_share": {
                1980: 3.30,
                1981: 3.25,
                1982: 3.20,
                1983: 3.10,
                1984: 3.00,
                1985: 2.95,
                1986: 2.90,
            },
        },
    }
    rows: list[dict[str, float | int]] = []
    for hospital_id, payload in yearly_values.items():
        for year, outcome in payload["depreciation_share"].items():  # type: ignore[attr-defined]
            rows.append(
                {
                    "hospital_id": hospital_id,
                    "year": year,
                    "depreciation_share": outcome,
                    "medicare_share_1983": payload["medicare_share_1983"],  # type: ignore[dict-item]
                }
            )
    return pd.DataFrame(rows)


def _check_required_year_support(frame: pd.DataFrame, year_column: str) -> None:
    required_years = _medicare_pps_required_years()
    observed_years = set(frame[year_column].astype(int).tolist())
    missing_years = [year for year in required_years if year not in observed_years]
    if missing_years:
        raise ContDIDValidationError(
            "medicare PPS scaffold requires full 1980-1986 year support; "
            f"missing years: {missing_years}"
        )


def _check_complete_unit_support(
    frame: pd.DataFrame, *, unit_column: str, year_column: str
) -> None:
    required_years = set(_medicare_pps_required_years())
    support = frame.groupby(unit_column)[year_column].agg(
        lambda values: {int(value) for value in values}
    )
    bad_units = [
        unit_id for unit_id, unit_years in support.items() if unit_years != required_years
    ]
    if bad_units:
        raise ContDIDValidationError(
            "medicare PPS scaffold requires every hospital to cover 1980-1986 exactly once"
        )

    duplicates = frame.duplicated([unit_column, year_column])
    if duplicates.any():
        raise ContDIDValidationError(
            "medicare PPS scaffold requires unique hospital/year observations"
        )


def _check_unit_constant_dose(frame: pd.DataFrame, *, unit_column: str, dose_column: str) -> None:
    dose_nunique = frame.groupby(unit_column)[dose_column].nunique(dropna=False)
    if not dose_nunique.le(1).all():
        raise ContDIDValidationError(_MEDICARE_PPS_DOSE_CONSTANCY_ERROR)


def prepare_medicare_pps_panel(
    frame: pd.DataFrame,
    *,
    unit_column: str,
    year_column: str,
    outcome_column: str,
    dose_column: str,
    source_id: str,
) -> EmpiricalScaffoldResult:
    """Collapse the annual Medicare PPS hospital panel into a package-ready 2-period panel."""

    manifest = load_medicare_pps_manifest()
    source = resolve_medicare_pps_source(source_id)
    required_years = list(_medicare_pps_required_years())
    _check_medicare_pps_input_columns(
        frame,
        unit_column=unit_column,
        year_column=year_column,
        outcome_column=outcome_column,
        dose_column=dose_column,
    )
    _check_medicare_pps_input_values(
        frame,
        year_column=year_column,
        outcome_column=outcome_column,
        dose_column=dose_column,
    )
    working = frame.copy()
    working[year_column] = working[year_column].astype(int)
    working = working.loc[working[year_column].isin(required_years)].copy()

    _check_required_year_support(working, year_column)
    _check_complete_unit_support(working, unit_column=unit_column, year_column=year_column)
    _check_unit_constant_dose(working, unit_column=unit_column, dose_column=dose_column)

    pre_years = manifest["time_aggregation"]["two_period_panel"]["pre_years"]
    post_years = manifest["time_aggregation"]["two_period_panel"]["post_years"]
    baseline_year = int(manifest["time_aggregation"]["event_study"]["baseline_year"])

    baseline_dose = (
        working.loc[working[year_column] == baseline_year, [unit_column, dose_column]]
        .rename(columns={unit_column: "id", dose_column: "D"})
        .copy()
    )
    if baseline_dose["D"].isna().any():
        raise ContDIDValidationError(
            "medicare PPS scaffold requires a non-missing 1983 Medicare inpatient share"
        )

    baseline_dose["G"] = baseline_dose["D"].gt(0).astype(int).replace({1: 2, 0: 0})

    outcome_means = (
        working.assign(
            time_period=working[year_column].map(
                {year: 1 for year in pre_years} | {year: 2 for year in post_years}
            )
        )
        .groupby([unit_column, "time_period"], as_index=False)[outcome_column]
        .mean()
        .rename(columns={unit_column: "id", outcome_column: "Y"})
    )

    package_frame = outcome_means.merge(baseline_dose, on="id", how="left")
    package_frame = package_frame[["id", "time_period", "Y", "G", "D"]]
    panel = validate_panel_data(PanelData(frame=package_frame))

    metadata = {
        "application_id": manifest["application_id"],
        "analysis_mode": source["source_type"],
        "source_id": source_id,
        "source_provider": source["provider"],
        "required_years": required_years,
        "baseline_year": baseline_year,
        "pre_years": pre_years,
        "post_years": post_years,
        "outcome_variable": manifest["outcome_contract"]["variable"],
        "dose_variable": manifest["dose_contract"]["variable"],
    }
    if source["source_type"] == "descriptive-or-scaffold-only":
        metadata.update(
            {
                "parity_claim_allowed": source["parity_claim_allowed"],
                "parity_viability": source["parity_viability"],
                "allowed_use": source["allowed_use"],
            }
        )
    return EmpiricalScaffoldResult(panel=panel, metadata=metadata)
