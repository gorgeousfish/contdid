"""Manifest-backed simulation scaffold for contdid."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from ._asset_paths import resolve_runtime_asset
from .data import PanelData
from .validation import ContDIDValidationError, validate_panel_data


_NUMERICAL_TRUTH_PATH = resolve_runtime_asset(
    package_relative="runtime_assets/numerical_truth_contract_v1.json",
    repo_relative="contdid-py/runtime-assets/numerical_truth_contract_v1.json",
)
_MANIFEST_PATH = resolve_runtime_asset(
    package_relative="reproduction/simulate_contdid/manifest.json",
    repo_relative="reproduction/simulate_contdid/manifest.json",
)
_SUPPORTED_DGP_IDS = (
    "SIM-001-null-dose",
    "SIM-002-linear-dose",
    "SIM-003-quadratic-dose",
    "SIM-004-staggered-eventstudy-null",
    "SIM-005-cck-two-period",
    "SIM-006-cubic-dose",
    "SIM-007-sine-dose",
    "SIM-008-threshold-dose",
    "SIM-009-staggered-linear-dose",
    "SIM-010-large-panel",
)
_SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "SIM-004-staggered-eventstudy-null": {
        "num_time_periods": 4,
        "num_groups": 4,
        "pg": [0.2, 0.2, 0.2],
        "pu": 0.4,
        "dose_linear_effect": 0.0,
        "dose_quadratic_effect": 0.0,
    },
    "SIM-005-cck-two-period": {
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.6],
        "pu": 0.4,
        "dose_linear_effect": 1.0,
        "dose_quadratic_effect": 0.2,
    },
    "SIM-006-cubic-dose": {
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "dose_linear_effect": 1.0,
        "dose_quadratic_effect": -0.5,
        "dose_cubic_effect": 0.8,
    },
    "SIM-007-sine-dose": {
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "sine_amplitude": 2.0,
        "sine_omega": math.pi,
    },
    "SIM-008-threshold-dose": {
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "threshold_d0": 0.5,
        "threshold_slope": 3.0,
    },
    "SIM-009-staggered-linear-dose": {
        "num_time_periods": 4,
        "num_groups": 4,
        "pg": [0.2, 0.2, 0.2],
        "pu": 0.4,
        "dose_linear_effect": 1.5,
        "dose_quadratic_effect": 0.0,
    },
    "SIM-010-large-panel": {
        "num_time_periods": 10,
        "num_groups": 10,
        "pg": [0.066, 0.066, 0.066, 0.066, 0.066, 0.067, 0.067, 0.067, 0.069],
        "pu": 0.4,
        "dose_linear_effect": 1.0,
        "dose_quadratic_effect": 0.0,
        "n_default": 10000,
    },
}

# Hardcoded seeds for extended DGP scenarios (not in the JSON contract)
_EXTENDED_SEEDS: dict[str, int] = {
    "SIM-006-cubic-dose": 20260601,
    "SIM-007-sine-dose": 20260602,
    "SIM-008-threshold-dose": 20260603,
    "SIM-009-staggered-linear-dose": 20260604,
    "SIM-010-large-panel": 20260605,
}


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_numerical_truth() -> dict[str, Any]:
    return json.loads(_NUMERICAL_TRUTH_PATH.read_text(encoding="utf-8"))["numerical_truth"]


@lru_cache(maxsize=1)
def _manifest_defaults() -> dict[str, Any]:
    return _load_manifest()["default_parameters"]


@lru_cache(maxsize=1)
def _manifest_effect_profiles() -> dict[str, dict[str, float]]:
    return {entry["id"]: entry["effects"] for entry in _load_manifest()["dgp_registry"]}


@lru_cache(maxsize=1)
def _seed_registry() -> dict[str, int]:
    registry = {
        entry["dgp_id"]: entry["default_seed"]
        for entry in _load_numerical_truth()["seed_registry"]
    }
    for k, v in _EXTENDED_SEEDS.items():
        registry.setdefault(k, v)
    return registry


def _validate_simulation_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ContDIDValidationError("seed must be a nonnegative integer")
    checked_seed = int(seed)
    if checked_seed < 0:
        raise ContDIDValidationError("seed must be a nonnegative integer")
    return checked_seed


def _compute_treatment_signal(
    D: np.ndarray,
    dgp_id: str,
    scenario_defaults: dict[str, Any],
    dose_linear_effect: float,
    dose_quadratic_effect: float,
) -> np.ndarray:
    """Compute the true ATT(d) treatment signal for each unit.

    For polynomial DGPs (SIM-001 to SIM-005, SIM-009, SIM-010):
        signal = alpha_1*d + alpha_2*d^2

    For SIM-006 (cubic): alpha_1*d + alpha_2*d^2 + alpha_3*d^3
    For SIM-007 (sine):  A * sin(omega * d)
    For SIM-008 (threshold): 0 for d < d0, beta*(d - d0) for d >= d0
    """
    if dgp_id == "SIM-006-cubic-dose":
        cubic = float(scenario_defaults.get("dose_cubic_effect", 0.0))
        return dose_linear_effect * D + dose_quadratic_effect * (D**2) + cubic * (D**3)
    elif dgp_id == "SIM-007-sine-dose":
        amplitude = float(scenario_defaults.get("sine_amplitude", 2.0))
        omega = float(scenario_defaults.get("sine_omega", math.pi))
        return amplitude * np.sin(omega * D)
    elif dgp_id == "SIM-008-threshold-dose":
        d0 = float(scenario_defaults.get("threshold_d0", 0.5))
        slope = float(scenario_defaults.get("threshold_slope", 3.0))
        return np.where(D >= d0, slope * (D - d0), 0.0)
    else:
        return dose_linear_effect * D + dose_quadratic_effect * (D**2)


def simulate_contdid_data(
    n: int | str = 5000,
    num_time_periods: int = 4,
    num_groups: int = 4,
    pg: list[float] | None = None,
    pu: float | None = None,
    dose_linear_effect: float | None = None,
    dose_quadratic_effect: float | None = None,
    seed: int | None = None,
    dgp_id: str = "SIM-001-null-dose",
) -> PanelData:
    """Generate a balanced long panel for testing and demonstration.

    Produces simulated panel data with known treatment effect parameters,
    suitable for verifying estimation pipelines and numerical accuracy.

    Args:
        n: Number of units (default 5000). Can also pass a dgp_id string
            as the first positional argument for convenience.
        num_time_periods: Number of time periods (default 4).
        num_groups: Number of timing groups including never-treated (default 4).
        pg: Probabilities for each treated group. Must sum to 1-pu with pu.
        pu: Probability of being never-treated.
        dose_linear_effect: Linear dose-response coefficient.
        dose_quadratic_effect: Quadratic dose-response coefficient.
        seed: Random seed for reproducibility.
        dgp_id: DGP scenario identifier from the supported registry.

    Returns:
        A validated PanelData object with columns: id, time_period, Y, G, D.

    Raises:
        ValueError: If dgp_id is unsupported or parameter constraints are violated.
        ContDIDValidationError: If generated data fails panel validation.
    """
    # Allow passing dgp_id as first positional argument for convenience:
    #   simulate_contdid_data('SIM-006-cubic-dose')
    if isinstance(n, str):
        dgp_id = n
        n = 5000

    defaults = _manifest_defaults()
    effect_profiles = _manifest_effect_profiles()
    seed_registry = _seed_registry()

    if dgp_id not in _SUPPORTED_DGP_IDS:
        raise ValueError(f"unsupported dgp_id: {dgp_id!r}; expected one of {_SUPPORTED_DGP_IDS}")

    scenario_defaults = _SCENARIO_DEFAULTS.get(dgp_id, {})
    num_time_periods = int(scenario_defaults.get("num_time_periods", num_time_periods))  # type: ignore[call-overload]
    num_groups = int(scenario_defaults.get("num_groups", num_groups))  # type: ignore[call-overload]

    # SIM-010 has a larger default n
    if dgp_id == "SIM-010-large-panel" and n == 5000:
        n = int(scenario_defaults.get("n_default", 10000))

    if num_groups != num_time_periods:
        raise ValueError("simulate_contdid_data currently requires num_groups == num_time_periods")

    if pg is None:
        pg = list(scenario_defaults.get("pg", defaults["pg"]))  # type: ignore[call-overload]
    if pu is None:
        pu = float(scenario_defaults.get("pu", defaults["pu"]))  # type: ignore[arg-type]
    if seed is None:
        seed = seed_registry.get(dgp_id, 1234)
    seed = _validate_simulation_seed(seed)

    effect_profile = effect_profiles.get(
        dgp_id,
        {
            "dose_linear_effect": scenario_defaults.get("dose_linear_effect", 0.0),
            "dose_quadratic_effect": scenario_defaults.get("dose_quadratic_effect", 0.0),
        },
    )
    if dose_linear_effect is None:
        dose_linear_effect = float(effect_profile.get("dose_linear_effect", 0.0))  # type: ignore[arg-type]
    if dose_quadratic_effect is None:
        dose_quadratic_effect = float(effect_profile.get("dose_quadratic_effect", 0.0))  # type: ignore[arg-type]

    if len(pg) != num_groups - 1:
        raise ValueError("pg must contain one probability for each treated group")
    if not np.isclose(sum(pg) + pu, 1.0):
        raise ValueError("pg plus pu must sum to 1.0")

    rng = np.random.default_rng(seed)
    time_periods = np.arange(1, num_time_periods + 1)
    groups = np.concatenate(([0], time_periods[1:]))
    group_probabilities = np.array([pu, *pg], dtype=float)

    G = rng.choice(groups, size=n, replace=True, p=group_probabilities)
    D = rng.uniform(0.0, 1.0, size=n)
    eta = rng.normal(loc=G, scale=1.0, size=n)

    untreated_outcomes = np.column_stack(
        [period + eta + rng.normal(size=n) for period in time_periods]
    )
    treated_signal = _compute_treatment_signal(
        D, dgp_id, scenario_defaults, dose_linear_effect, dose_quadratic_effect
    )
    treated_outcomes = np.column_stack(
        [treated_signal + period + eta + rng.normal(size=n) for period in time_periods]
    )

    post_matrix = ((G[:, None] != 0) & (G[:, None] <= time_periods[None, :])).astype(int)
    observed_outcomes = post_matrix * treated_outcomes + (1 - post_matrix) * untreated_outcomes

    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, n + 1), num_time_periods),
            "time_period": np.tile(time_periods, n),
            "Y": observed_outcomes.reshape(n * num_time_periods),
            "G": np.repeat(G, num_time_periods),
            "D": np.repeat(D, num_time_periods),
        }
    )
    frame.loc[frame["G"] == 0, "D"] = 0.0

    panel = PanelData(frame=frame)
    return validate_panel_data(panel)
