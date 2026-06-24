from __future__ import annotations

import os
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_MANIFEST_PATH = (
    REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke_manifest.json"
)
SMOKE_MODULE_PATH = REPO_ROOT / "reproduction" / "medicare_pps" / "e2e_smoke.py"


def _make_smoke_panel() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
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


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("medicare_pps_smoke", SMOKE_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase7_smoke_manifest_declares_both_execution_modes() -> None:
    assert SMOKE_MANIFEST_PATH.exists(), f"missing smoke manifest: {SMOKE_MANIFEST_PATH}"

    manifest = json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.1"
    assert manifest["application_id"] == "medicare-pps-hospitals"
    assert manifest["contract_type"] == "paper_to_python"
    assert set(manifest["execution_modes"]) == {
        "licensed-parity",
        "descriptive-or-scaffold-only",
    }
    assert manifest["expected_outputs"]["att"]["package_surface"] == "estimate_dose_effects"
    assert (
        manifest["expected_outputs"]["eventstudy"]["package_surface"]
        == "estimate_eventstudy_effects"
    )
    assert manifest["cli"] == {
        "path": "reproduction/medicare_pps/e2e_smoke.py",
        "default_command": "python3 reproduction/medicare_pps/e2e_smoke.py",
        "default_source_id": "cms_hcris_hospital_cost_reports",
        "default_run_mode": "descriptive-or-scaffold-only",
        "writes_json_payload": True,
        "supports_input_csv": True,
        "supports_output_file": True,
    }


def test_phase7_smoke_runner_reuses_existing_package_surfaces_and_keeps_scaffold_label() -> (
    None
):
    smoke = _load_smoke_module()

    payload = smoke.run_medicare_pps_e2e_smoke(
        _make_smoke_panel(),
        unit_column="hospital_id",
        year_column="year",
        outcome_column="depreciation_share",
        dose_column="medicare_share_1983",
        source_id="cms_hcris_hospital_cost_reports",
    )

    assert payload["run_mode"] == "descriptive-or-scaffold-only"
    assert payload["source_label"] == "descriptive-or-scaffold-only"
    assert payload["package_surfaces"] == {
        "att": "estimate_dose_effects",
        "acrt": "estimate_dose_slope_effects",
        "eventstudy": "estimate_eventstudy_effects",
    }
    assert payload["outputs"]["att"]["estimand"] == "ATT(d)"
    assert "confidence_band" in payload["outputs"]["att"]
    assert payload["outputs"]["acrt"]["estimand"] == "ACRT(d)"
    assert payload["outputs"]["eventstudy"]["estimand"] == "ATT(event_time)"
    assert payload["outputs"]["eventstudy"]["metadata"]["inference"] == "analytic"
    assert payload["outputs"]["eventstudy"]["grid"] == [-4, -3, -2, 0, 1, 2]
    assert payload["outputs"]["eventstudy"]["estimate"] == pytest.approx(
        [
            -0.400000,
            -0.258333,
            -0.141667,
            0.691667,
            0.891667,
            1.091667,
        ],
        abs=1e-6,
    )
    assert {
        cohort["event_time"]: {
            entry["base_period"]
            for entry in cohort["cohort_estimates"]
        }
        for cohort in payload["outputs"]["eventstudy"]["metadata"]["cohort_summary"]
    } == {
        -4: {1983},
        -3: {1983},
        -2: {1983},
        0: {1983},
        1: {1983},
        2: {1983},
    }


def test_phase7_smoke_module_bootstraps_without_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    smoke_run = subprocess.run(
        ["python3", str(SMOKE_MODULE_PATH)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke_run.returncode == 0, smoke_run.stderr
    payload = json.loads(smoke_run.stdout)
    assert payload["run_mode"] == "descriptive-or-scaffold-only"
    assert payload["source_label"] == "descriptive-or-scaffold-only"
    assert payload["package_surfaces"] == {
        "att": "estimate_dose_effects",
        "acrt": "estimate_dose_slope_effects",
        "eventstudy": "estimate_eventstudy_effects",
    }
    assert payload["outputs"]["eventstudy"]["estimand"] == "ATT(event_time)"


def test_phase7_smoke_cli_writes_custom_json_output(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "e2e_smoke.json"
    smoke_run = subprocess.run(
        [
            "python3",
            str(SMOKE_MODULE_PATH),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke_run.returncode == 0, smoke_run.stderr
    assert str(output_path) in smoke_run.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["paper_to_python"] is True
    assert payload["run_mode"] == "descriptive-or-scaffold-only"
    assert payload["outputs"]["att"]["estimand"] == "ATT(d)"


def test_phase7_smoke_cli_accepts_staged_input_csv(tmp_path: Path) -> None:
    input_csv = tmp_path / "staged_medicare_panel.csv"
    output_path = tmp_path / "e2e_smoke.json"
    _make_smoke_panel().to_csv(input_csv, index=False)

    smoke_run = subprocess.run(
        [
            "python3",
            str(SMOKE_MODULE_PATH),
            "--input-csv",
            str(input_csv),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke_run.returncode == 0, smoke_run.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_mode"] == "descriptive-or-scaffold-only"
    assert payload["outputs"]["eventstudy"]["grid"] == [-4, -3, -2, 0, 1, 2]


def test_phase7_smoke_cli_reports_missing_staged_input_columns_as_contract_error(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "missing_dose_panel.csv"
    _make_smoke_panel().drop(columns=["medicare_share_1983"]).to_csv(
        input_csv,
        index=False,
    )

    smoke_run = subprocess.run(
        [
            "python3",
            str(SMOKE_MODULE_PATH),
            "--input-csv",
            str(input_csv),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert smoke_run.returncode != 0
    assert "ContDIDValidationError" in smoke_run.stderr
    assert (
        "medicare PPS scaffold input is missing required columns: "
        "medicare_share_1983"
    ) in smoke_run.stderr
