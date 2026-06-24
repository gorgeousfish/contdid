from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "reproduction" / "simulate_contdid" / "manifest.json"
NUMERICAL_TRUTH_PATH = (
    REPO_ROOT / "contdid-py" / "contracts" / "phase2" / "numerical_truth_contract_v1.json"
)



def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))



def _load_numerical_truth() -> dict:
    return json.loads(NUMERICAL_TRUTH_PATH.read_text(encoding="utf-8"))["numerical_truth"]



def test_simulate_contdid_data_matches_manifest_backed_panel_invariants() -> None:
    from contdid import PanelData, simulate_contdid_data

    manifest = _load_manifest()
    defaults = manifest["default_parameters"]
    derived_defaults = manifest["derived_defaults"]
    panel_contract = manifest["panel_contract"]

    panel = simulate_contdid_data()

    assert isinstance(panel, PanelData)
    frame = panel.frame

    assert list(frame.columns) == panel_contract["columns"]
    assert len(frame) == derived_defaults["row_count"]
    assert frame["id"].nunique() == derived_defaults["unique_ids"]
    assert sorted(frame["time_period"].unique().tolist()) == derived_defaults["time_periods"]
    assert frame.groupby("id")["time_period"].nunique().eq(defaults["num_time_periods"]).all()
    assert not frame.duplicated(["id", "time_period"]).any()
    assert frame.groupby("id")["G"].nunique().eq(1).all()
    assert frame.groupby("id")["D"].nunique().eq(1).all()
    assert (frame.loc[frame["G"] == 0, "D"] == 0).all()
    assert frame["D"].between(0, 1).all()



def test_simulate_contdid_data_uses_phase2_seed_registry_and_dgp_profiles() -> None:
    from contdid import simulate_contdid_data

    manifest_effects = {
        entry["id"]: entry["effects"] for entry in _load_manifest()["dgp_registry"]
    }
    seed_registry = {
        entry["dgp_id"]: entry["default_seed"]
        for entry in _load_numerical_truth()["seed_registry"]
        if entry["dgp_id"] in manifest_effects
    }

    for dgp_id, effects in manifest_effects.items():
        implicit = simulate_contdid_data(dgp_id=dgp_id).frame
        explicit = simulate_contdid_data(
            dgp_id=dgp_id,
            dose_linear_effect=effects["dose_linear_effect"],
            dose_quadratic_effect=effects["dose_quadratic_effect"],
            seed=seed_registry[dgp_id],
        ).frame
        pd.testing.assert_frame_equal(implicit, explicit)


@pytest.mark.parametrize("seed", [True, np.bool_(False), -1, 1.2, "20260407"])
def test_simulate_contdid_data_rejects_non_integer_or_negative_seed(
    seed: object,
) -> None:
    from contdid import ContDIDValidationError, simulate_contdid_data

    with pytest.raises(
        ContDIDValidationError, match="seed must be a nonnegative integer"
    ):
        simulate_contdid_data(
            n=10,
            num_time_periods=2,
            num_groups=2,
            pg=[0.5],
            pu=0.5,
            dgp_id="SIM-002-linear-dose",
            seed=seed,
        )


def test_simulate_contdid_data_accepts_explicit_integer_seed_deterministically() -> None:
    from contdid import simulate_contdid_data

    first = simulate_contdid_data(
        n=10,
        num_time_periods=2,
        num_groups=2,
        pg=[0.5],
        pu=0.5,
        dgp_id="SIM-002-linear-dose",
        seed=np.int64(20260407),
    ).frame
    second = simulate_contdid_data(
        n=10,
        num_time_periods=2,
        num_groups=2,
        pg=[0.5],
        pu=0.5,
        dgp_id="SIM-002-linear-dose",
        seed=20260407,
    ).frame

    pd.testing.assert_frame_equal(first, second)
