from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDSpec,
    attach_inference_payload,
    compute_multiplier_bootstrap,
    build_confidence_band,
    estimate_dose_effects,
    estimate_dose_slope_effects,
    simulate_contdid_data,
)
from contdid.data import PanelData
from contdid.inference import append_independent_mean_variance, estimate_mean_variance
from contdid.results import ContDIDResult
from contdid.validation import ContDIDValidationError


_NORMAL = NormalDist()


def _make_inference_spec(
    *, target_parameter: str, cband: bool = True, bstrap: bool = True
) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=bstrap,
        cband=cband,
        boot_type="multiplier",
        biters=199,
    )


def _simulate_two_period_dose_panel(
    *, n: int, dgp_id: str, seed: int | None = None
) -> PanelData:
    kwargs: dict[str, object] = {
        "n": n,
        "num_time_periods": 2,
        "num_groups": 2,
        "pg": [0.75],
        "pu": 0.25,
        "dgp_id": dgp_id,
    }
    if seed is not None:
        kwargs["seed"] = seed
    return simulate_contdid_data(**kwargs)


def _make_perfect_fit_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.04, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.25, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.64, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_untreated_benchmark_variance_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, -0.2, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.2, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.04, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.4),
        ("t2", 2, 0.16, 2, 0.4),
        ("t3", 1, 0.0, 2, 0.6),
        ("t3", 2, 0.36, 2, 0.6),
        ("t4", 1, 0.0, 2, 0.8),
        ("t4", 2, 0.64, 2, 0.8),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_single_untreated_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.1, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.14, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.5),
        ("t2", 2, 0.35, 2, 0.5),
        ("t3", 1, 0.0, 2, 0.8),
        ("t3", 2, 0.74, 2, 0.8),
        ("t4", 1, 0.0, 2, 0.9),
        ("t4", 2, 0.89, 2, 0.9),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_heteroskedastic_treated_fit_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("u2", 1, 0.0, 0, 0.0),
        ("u2", 2, 0.0, 0, 0.0),
        ("u3", 1, 0.0, 0, 0.0),
        ("u3", 2, 0.0, 0, 0.0),
    ]
    for unit, dose, residual in (
        ("t1", 0.2, 0.0),
        ("t2", 0.4, 0.0),
        ("t3", 0.6, -0.6),
        ("t4", 0.8, 0.6),
        ("t5", 1.0, -1.2),
        ("t6", 1.2, 1.2),
    ):
        rows.extend(
            [
                (unit, 1, 0.0, 2, dose),
                (unit, 2, dose + residual, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cubic_bspline_inference_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, -0.20, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.15, 0, 0.0),
        ("u2", 1, 0.0, 0, 0.0),
        ("u2", 2, 0.35, 0, 0.0),
    ]
    for unit, dose, residual in (
        ("tc1", 1.0, 0.00),
        ("tc2", 1.5, 0.18),
        ("tc3", 2.0, -0.24),
        ("tc4", 3.0, 0.31),
        ("tc5", 4.5, -0.16),
        ("tc6", 6.0, 0.22),
        ("tc7", 7.0, -0.08),
    ):
        signal = 0.4 + 0.25 * dose - 0.08 * dose**2 + 0.015 * dose**3
        rows.extend(
            [
                (unit, 1, 0.0, 2, dose),
                (unit, 2, signal + residual, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_linear_bspline_inference_panel() -> PanelData:
    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, -0.10, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.20, 0, 0.0),
        ("u2", 1, 0.0, 0, 0.0),
        ("u2", 2, 0.40, 0, 0.0),
    ]
    for unit, dose, residual in (
        ("tl1", 1.0, 0.00),
        ("tl2", 2.0, 0.35),
        ("tl3", 3.0, -0.25),
        ("tl4", 5.0, 0.40),
        ("tl5", 7.0, -0.15),
    ):
        signal = 0.5 + 0.25 * dose + 0.45 * max(dose - 3.0, 0.0)
        rows.extend(
            [
                (unit, 1, 0.0, 2, dose),
                (unit, 2, signal + residual, 2, dose),
            ]
        )
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _linear_bspline_no_intercept_columns(
    values: np.ndarray,
    *,
    lower: float,
    knot: float,
    upper: float,
) -> np.ndarray:
    left_middle = np.where(
        values < knot,
        (values - lower) / (knot - lower),
        (upper - values) / (upper - knot),
    )
    right = np.where(values < knot, 0.0, (values - knot) / (upper - knot))
    return np.column_stack([left_middle, right])


def _linear_bspline_no_intercept_derivative_columns(
    values: np.ndarray,
    *,
    lower: float,
    knot: float,
    upper: float,
) -> np.ndarray:
    left_middle = np.where(values < knot, 1.0 / (knot - lower), -1.0 / (upper - knot))
    right = np.where(values < knot, 0.0, 1.0 / (upper - knot))
    return np.column_stack([left_middle, right])


def _cubic_bspline_no_intercept_columns(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    scaled = (values - lower) / (upper - lower)
    middle_left = 3.0 * scaled * (1.0 - scaled) ** 2
    middle_right = 3.0 * scaled**2 * (1.0 - scaled)
    right = scaled**3
    return np.column_stack([middle_left, middle_right, right])


def _cubic_bspline_no_intercept_derivative_columns(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    scaled = (values - lower) / (upper - lower)
    scale = 1.0 / (upper - lower)
    middle_left = 3.0 * (1.0 - scaled) * (1.0 - 3.0 * scaled) * scale
    middle_right = 3.0 * scaled * (2.0 - 3.0 * scaled) * scale
    right = 3.0 * scaled**2 * scale
    return np.column_stack([middle_left, middle_right, right])


def test_phase6_public_surface_exports_shared_inference_knobs() -> None:
    spec = _make_inference_spec(target_parameter="level")

    assert callable(attach_inference_payload)
    assert callable(compute_multiplier_bootstrap)
    assert spec.alp == 0.1
    assert spec.bstrap is True
    assert spec.cband is True
    assert spec.boot_type == "multiplier"
    assert spec.biters == 199


def test_parametric_dose_results_attach_phase6_inference_payloads() -> None:
    explicit_grid = [0.2, 0.4, 0.8]
    panel = _simulate_two_period_dose_panel(
        n=12000, dgp_id="SIM-002-linear-dose", seed=20260407
    )

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", cband=True),
        dvals=explicit_grid,
        degree=1,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", cband=False),
        dvals=explicit_grid,
        degree=1,
    )

    for result in (level_result, slope_result):
        assert result.metadata["bootstrap_type"] == "multiplier"
        assert result.metadata["confidence_band_kind"] == (
            "simultaneous_multiplier" if result.metadata["cband"] else "pointwise_multiplier"
        )
        expected_seed = 20260407 if result.metadata["cband"] else None
        assert result.metadata["bootstrap_seed"] == expected_seed
        assert "boot_type" not in result.metadata
        assert result.metadata["summary_aggregates"] == result.metadata["summary"]
        assert result.metadata["critical_value"] == result.critical_value
        assert result.metadata["confidence_interval"] == result.confidence_interval
        assert result.metadata["confidence_band"] == result.confidence_band
        assert result.metadata["inference"] == "bootstrap"
        assert len(result.std_error) == len(explicit_grid)
        assert all(math.isfinite(value) and value > 0.0 for value in result.std_error)
        assert result.confidence_interval is not None
        assert result.confidence_band is not None
        assert result.critical_value is not None and result.critical_value > 0.0

    assert (
        level_result.confidence_band["critical_value"]
        > slope_result.confidence_band["critical_value"]
    )


def test_shared_helper_can_attach_interval_and_band_payloads() -> None:
    base = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.8],
        estimate=[0.5, 0.9],
        std_error=[0.0, 0.0],
    )
    loadings = [[1.0, 0.2], [1.0, 0.8]]
    covariance = [[0.04, 0.01], [0.01, 0.03]]

    enriched = attach_inference_payload(
        base,
        loadings=loadings,
        covariance=covariance,
        spec=_make_inference_spec(target_parameter="level", cband=True),
    )

    assert enriched.metadata["bootstrap_type"] == "multiplier"
    assert enriched.metadata["confidence_band_kind"] == "simultaneous_multiplier"
    assert enriched.metadata["bootstrap_seed"] == 20260407
    assert "boot_type" not in enriched.metadata
    assert enriched.metadata["confidence_interval"] == enriched.confidence_interval
    assert enriched.metadata["confidence_band"] == enriched.confidence_band
    assert enriched.confidence_band["lower"][0] <= enriched.estimate[0]
    assert enriched.confidence_band["upper"][1] >= enriched.estimate[1]


def test_estimate_mean_variance_returns_sample_mean_variance() -> None:
    assert estimate_mean_variance([1.0, 3.0, 5.0]) == pytest.approx(4.0 / 3.0)
    assert estimate_mean_variance((value for value in [2.0, 4.0])) == pytest.approx(
        1.0
    )
    assert estimate_mean_variance([7.0]) == 0.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "values must contain at least one observation"),
        (1.0, "values must be a one-dimensional sample"),
        (np.array(1.0), "values must be a one-dimensional sample"),
        ([[1.0], [2.0]], "values must be a one-dimensional sample"),
        ([1.0, math.nan], "values must contain only finite non-boolean values"),
        ([1.0, math.inf], "values must contain only finite non-boolean values"),
        ([1.0, True], "values must contain only finite non-boolean values"),
        (["stale"], "values must contain only finite non-boolean values"),
        (
            (value for value in [1.0, False]),
            "values must contain only finite non-boolean values",
        ),
    ],
)
def test_estimate_mean_variance_rejects_invalid_samples(
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_mean_variance(values)


def test_append_independent_mean_variance_adds_separate_component() -> None:
    loadings, covariance = append_independent_mean_variance(
        np.asarray([[1.0, 0.5], [0.0, 1.0]], dtype=float),
        np.asarray([[0.04, 0.01], [0.01, 0.09]], dtype=float),
        mean_variance=0.16,
        loading_value=-1.0,
    )

    assert loadings.tolist() == [[1.0, 0.5, -1.0], [0.0, 1.0, -1.0]]
    assert covariance.tolist() == [
        [0.04, 0.01, 0.0],
        [0.01, 0.09, 0.0],
        [0.0, 0.0, 0.16],
    ]


def test_append_independent_mean_variance_preserves_zero_variance_problem() -> None:
    original_loadings = np.asarray([[1.0, 0.5]], dtype=float)
    original_covariance = np.asarray([[0.04, 0.01], [0.01, 0.09]], dtype=float)

    loadings, covariance = append_independent_mean_variance(
        original_loadings,
        original_covariance,
        mean_variance=0.0,
        loading_value=-1.0,
    )

    assert loadings is original_loadings
    assert covariance is original_covariance


@pytest.mark.parametrize("mean_variance", [-1.0, math.nan, math.inf, True])
def test_append_independent_mean_variance_rejects_invalid_mean_variance(
    mean_variance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="mean_variance must be a finite nonnegative scalar",
    ):
        append_independent_mean_variance(
            np.asarray([[1.0]], dtype=float),
            np.asarray([[0.04]], dtype=float),
            mean_variance=mean_variance,
            loading_value=-1.0,
        )


@pytest.mark.parametrize("loading_value", [math.nan, math.inf, True])
def test_append_independent_mean_variance_rejects_invalid_loading_value(
    loading_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="loading_value must be a finite scalar",
    ):
        append_independent_mean_variance(
            np.asarray([[1.0]], dtype=float),
            np.asarray([[0.04]], dtype=float),
            mean_variance=0.0,
            loading_value=loading_value,
        )


@pytest.mark.parametrize(
    ("loadings", "covariance", "message"),
    [
        ([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], "two-dimensional matrix"),
        ([[1.0, 2.0]], [[1.0, 0.0]], "covariance matrix must be square"),
        ([[1.0, 2.0]], [[1.0]], "loadings and covariance dimensions must align"),
        ([[1.0, math.nan]], [[1.0, 0.0], [0.0, 1.0]], "finite values"),
        ([["stale"]], [[1.0]], "loadings must contain only finite values"),
        ([[1.0]], [["stale"]], "covariance matrix must contain only finite values"),
        ([[1.0, 0.0]], [[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ],
)
def test_append_independent_mean_variance_rejects_invalid_linear_problem(
    loadings: object,
    covariance: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        append_independent_mean_variance(
            loadings,
            covariance,
            mean_variance=0.0,
            loading_value=1.0,
        )


@pytest.mark.parametrize(
    ("curve_estimate", "std_error", "critical_value", "message"),
    [
        (
            [[0.5], [0.9]],
            [0.1, 0.2],
            1.96,
            "estimate and std_error must be one-dimensional",
        ),
        (
            0.5,
            [0.1],
            1.96,
            "estimate and std_error must be one-dimensional",
        ),
        (
            np.array(0.5),
            [0.1],
            1.96,
            "estimate and std_error must be one-dimensional",
        ),
        (
            [0.5],
            0.1,
            1.96,
            "estimate and std_error must be one-dimensional",
        ),
        (
            [0.5, 0.9],
            [0.1],
            1.96,
            "estimate and std_error must have the same shape",
        ),
        (
            [],
            [],
            1.96,
            "estimate and std_error must contain at least one value",
        ),
        (
            [0.5, math.nan],
            [0.1, 0.2],
            1.96,
            "estimate must contain only finite non-boolean values",
        ),
        (
            [0.5, 0.9],
            [0.1, -0.2],
            1.96,
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            [0.5, True],
            [0.1, 0.2],
            1.96,
            "estimate must contain only finite non-boolean values",
        ),
        (
            ["stale"],
            [0.1],
            1.96,
            "estimate must contain only finite non-boolean values",
        ),
        (
            (value for value in [0.5, True]),
            [0.1, 0.2],
            1.96,
            "estimate must contain only finite non-boolean values",
        ),
        (
            [0.5, 0.9],
            [0.1, False],
            1.96,
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            [0.5],
            ["stale"],
            1.96,
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            [0.5, 0.9],
            (value for value in [0.1, False]),
            1.96,
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            [0.5, 0.9],
            [0.1, 0.2],
            math.inf,
            "critical_value must be a finite non-boolean nonnegative scalar",
        ),
    ],
)
def test_build_confidence_band_validates_public_shape_contract(
    curve_estimate: object,
    std_error: object,
    critical_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_confidence_band(
            curve_estimate,
            std_error,
            critical_value=critical_value,
        )


def test_shared_helper_rejects_mismatched_loading_rows_before_attaching_payload() -> (
    None
):
    base = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0, 1],
        estimate=[0.5, 0.9],
        std_error=[0.0, 0.0],
        event_time=[0, 1],
    )

    with pytest.raises(
        ValueError,
        match="loadings row count must match result estimate length",
    ):
        attach_inference_payload(
            base,
            loadings=[[1.0]],
            covariance=[[0.04]],
            spec=_make_inference_spec(target_parameter="level", cband=True),
        )


def test_shared_helper_rejects_mismatched_loading_rows_before_bootstrap_draws(
    monkeypatch,
) -> None:
    def unexpected_default_rng(seed: object) -> object:
        raise AssertionError(
            "mismatched result/loadings rows should fail before bootstrap draws"
        )

    monkeypatch.setattr(np.random, "default_rng", unexpected_default_rng)
    base = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0, 1],
        estimate=[0.5, 0.9],
        std_error=[0.0, 0.0],
        event_time=[0, 1],
    )

    with pytest.raises(
        ValueError,
        match="loadings row count must match result estimate length",
    ):
        attach_inference_payload(
            base,
            loadings=[[1.0]],
            covariance=[[0.04]],
            spec=_make_inference_spec(target_parameter="level", cband=True),
        )


def test_shared_helper_rejects_mutated_result_grid_mismatch_before_attaching_payload() -> (
    None
):
    base = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.0, 0.0],
    )
    base.grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ValueError,
        match="result grid must match result estimate length",
    ):
        attach_inference_payload(
            base,
            loadings=[[1.0, 0.0], [0.0, 1.0]],
            covariance=[[0.04, 0.0], [0.0, 0.09]],
            spec=_make_inference_spec(target_parameter="level", cband=False),
        )


def test_shared_helper_normalizes_mutated_result_vectors_before_metadata_echo() -> None:
    base = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.0, 0.0],
    )
    base.grid = np.array([0.2, 0.5])
    base.estimate = ("0.4", "0.8")

    enriched = attach_inference_payload(
        base,
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[0.04, 0.0], [0.0, 0.09]],
        spec=_make_inference_spec(target_parameter="level", cband=False),
    )

    assert enriched.grid == [0.2, 0.5]
    assert enriched.estimate == [0.4, 0.8]
    assert enriched.metadata["grid"] == enriched.grid
    assert enriched.metadata["estimate"] == enriched.estimate
    json.dumps(enriched.metadata)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("estimate", "result estimates must be a finite one-dimensional vector"),
        ("grid", "result grid must be a finite one-dimensional vector"),
        ("loadings", "loadings must contain only finite values"),
    ],
)
def test_shared_helper_rejects_nonnumeric_mutated_payloads_with_stable_errors(
    field: str,
    message: str,
) -> None:
    base = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2],
        estimate=[0.4],
        std_error=[0.0],
    )
    loadings: object = [[1.0]]
    if field == "estimate":
        base.estimate = ["stale"]
    elif field == "grid":
        base.grid = ["stale"]
    else:
        loadings = [["stale"]]

    with pytest.raises(ValueError, match=message):
        attach_inference_payload(
            base,
            loadings=loadings,
            covariance=[[0.04]],
            spec=_make_inference_spec(target_parameter="level", cband=False),
        )


def test_multiplier_bootstrap_ignores_machine_precision_rank_noise() -> None:
    nearly_rank_one = [
        [2.0 - 3.552713678800501e-15, 4.0 - 7.105427357601002e-15],
        [4.0 - 7.105427357601002e-15, 8.0 - 1.4210854715202004e-14],
    ]
    exact_rank_one = [[2.0, 4.0], [4.0, 8.0]]

    nearly = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=nearly_rank_one,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )
    exact = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=exact_rank_one,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert nearly["std_error"] == pytest.approx(exact["std_error"], abs=1e-12)
    assert nearly["critical_value"] == pytest.approx(
        exact["critical_value"],
        abs=1e-12,
    )


def test_multiplier_bootstrap_rejects_materially_indefinite_covariance() -> None:
    with pytest.raises(
        ValueError,
        match="covariance matrix must be positive semidefinite",
    ):
        compute_multiplier_bootstrap(
            loadings=[[1.0, 0.0], [0.0, 1.0]],
            covariance=[[1.0, 2.0], [2.0, 1.0]],
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        )


def test_multiplier_bootstrap_rejects_empty_estimand_loadings() -> None:
    with pytest.raises(
        ValueError,
        match="loadings must contain at least one estimand row",
    ):
        compute_multiplier_bootstrap(
            loadings=np.empty((0, 1)),
            covariance=[[1.0]],
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        )


@pytest.mark.parametrize(
    ("loadings", "covariance", "message"),
    [
        ([1.0], [[1.0]], "loadings must be a two-dimensional matrix"),
        ([["stale"]], [[1.0]], "loadings must contain only finite values"),
        ([[1.0]], [["stale"]], "covariance matrix must contain only finite values"),
    ],
)
def test_multiplier_bootstrap_rejects_nonnumeric_linear_problem_with_stable_errors(
    loadings: object,
    covariance: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_multiplier_bootstrap(
            loadings=loadings,
            covariance=covariance,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        )


def test_shared_helper_rejects_empty_result_estimates_before_attaching_payload() -> (
    None
):
    base = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0],
        estimate=[0.0],
        std_error=[0.0],
        event_time=[0],
    )
    base.estimate = []

    with pytest.raises(
        ValueError,
        match="result estimates must contain at least one value",
    ):
        attach_inference_payload(
            base,
            loadings=[[1.0]],
            covariance=[[1.0]],
            spec=_make_inference_spec(target_parameter="level", cband=True),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"alp": 1.5, "bstrap": True, "cband": True, "biters": 199},
            "alp must lie strictly between 0 and 1",
        ),
        (
            {"alp": 0.1, "bstrap": True, "cband": True, "biters": 0},
            "biters must be a positive integer",
        ),
        (
            {"alp": 0.1, "bstrap": True, "cband": True, "biters": 199.0},
            "biters must be a positive integer",
        ),
        (
            {"alp": 0.1, "bstrap": "yes", "cband": True, "biters": 199},
            "bstrap must be a boolean",
        ),
        (
            {"alp": 0.1, "bstrap": True, "cband": "yes", "biters": 199},
            "cband must be a boolean",
        ),
        (
            {
                "alp": 0.1,
                "bstrap": False,
                "cband": True,
                "boot_type": "wild",
                "biters": 199,
            },
            "boot_type must be",
        ),
    ],
)
def test_multiplier_bootstrap_validates_public_inference_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    checked_kwargs = dict(kwargs)
    boot_type = checked_kwargs.pop("boot_type", "multiplier")
    with pytest.raises(ContDIDValidationError, match=message):
        compute_multiplier_bootstrap(
            loadings=[[1.0]],
            covariance=[[1.0]],
            boot_type=boot_type,
            **checked_kwargs,
        )


@pytest.mark.parametrize("seed", [None, True, np.bool_(False), -1, 1.2, "20260407"])
def test_multiplier_bootstrap_rejects_non_deterministic_or_non_integer_seed(
    seed: object,
) -> None:
    with pytest.raises(
        ContDIDValidationError,
        match="seed must be a nonnegative integer",
    ):
        compute_multiplier_bootstrap(
            loadings=[[1.0]],
            covariance=[[1.0]],
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
            seed=seed,
        )


def test_multiplier_bootstrap_accepts_explicit_integer_seed_deterministically() -> None:
    first = compute_multiplier_bootstrap(
        loadings=[[1.0]],
        covariance=[[1.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
        seed=20260407,
    )
    second = compute_multiplier_bootstrap(
        loadings=[[1.0]],
        covariance=[[1.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
        seed=20260407,
    )

    assert second["critical_value"] == pytest.approx(
        first["critical_value"],
        abs=0.0,
    )
    assert first["bootstrap_seed"] == 20260407
    assert second["bootstrap_seed"] == 20260407


def test_pointwise_multiplier_skips_discarded_simulation_draws(monkeypatch) -> None:
    def unexpected_default_rng(seed: object) -> object:
        raise AssertionError(
            "pointwise multiplier inference should not draw bootstrap shocks"
        )

    monkeypatch.setattr(np.random, "default_rng", unexpected_default_rng)

    result = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[0.04, 0.01], [0.01, 0.09]],
        alp=0.1,
        bstrap=True,
        cband=False,
        boot_type="multiplier",
        biters=10**9,
        seed=20260407,
    )

    expected_critical = _NORMAL.inv_cdf(1.0 - 0.1 / 2.0)
    assert result["bootstrap_type"] == "multiplier"
    assert result["confidence_band_kind"] == "pointwise_multiplier"
    assert result["critical_value"] == pytest.approx(expected_critical, abs=0.0)
    assert result["pointwise_critical_value"] == pytest.approx(
        expected_critical,
        abs=0.0,
    )
    assert result["std_error"] == pytest.approx([0.2, 0.3], abs=1e-12)


def test_simultaneous_multiplier_zero_variance_skips_discarded_draws(monkeypatch) -> None:
    def unexpected_default_rng(seed: object) -> object:
        raise AssertionError(
            "zero-variance simultaneous multiplier inference should not draw shocks"
        )

    monkeypatch.setattr(np.random, "default_rng", unexpected_default_rng)

    result = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[0.0, 0.0], [0.0, 0.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=10**9,
        seed=20260407,
    )

    expected_critical = _NORMAL.inv_cdf(1.0 - 0.1 / 2.0)
    assert result["bootstrap_type"] == "multiplier"
    assert result["bootstrap_seed"] == 20260407
    assert result["confidence_band_kind"] == "simultaneous_multiplier"
    assert result["pointwise_critical_value"] == pytest.approx(
        expected_critical,
        abs=0.0,
    )
    assert result["critical_value"] == pytest.approx(0.0, abs=0.0)
    assert result["std_error"] == pytest.approx([0.0, 0.0], abs=0.0)


def test_simultaneous_multiplier_chunking_preserves_seeded_draw_order(monkeypatch) -> (
    None
):
    """Verify SeedSequence-based chunking is deterministic for a fixed chunk config.

    With SeedSequence.spawn(), each chunk gets an independent child seed.
    Determinism is guaranteed for the same (biters, chunk_size) pair regardless
    of thread scheduling order. Different chunk sizes produce different (but
    equally valid) results by design.
    """
    import contdid.inference as inference_module

    kwargs = {
        "loadings": [[1.0, 0.0], [0.5, 1.0]],
        "covariance": [[0.04, 0.01], [0.01, 0.09]],
        "alp": 0.1,
        "bstrap": True,
        "cband": True,
        "boot_type": "multiplier",
        "biters": 199,
        "seed": 20260407,
    }

    # Same chunk config → identical results (determinism)
    monkeypatch.setattr(inference_module, "_BOOTSTRAP_MAX_DRAW_CELLS", 4)
    chunked_1 = inference_module.compute_multiplier_bootstrap(**kwargs)
    chunked_2 = inference_module.compute_multiplier_bootstrap(**kwargs)

    assert chunked_1["std_error"] == chunked_2["std_error"]
    assert chunked_1["pointwise_critical_value"] == chunked_2["pointwise_critical_value"]
    assert chunked_1["critical_value"] == pytest.approx(
        chunked_2["critical_value"],
        abs=0.0,
    )
    assert chunked_1["confidence_band_kind"] == "simultaneous_multiplier"

    # Different chunk config → still a valid critical value (positive, finite)
    monkeypatch.setattr(inference_module, "_BOOTSTRAP_MAX_DRAW_CELLS", 10**9)
    unchunked = inference_module.compute_multiplier_bootstrap(**kwargs)
    assert unchunked["critical_value"] > 0.0
    assert np.isfinite(unchunked["critical_value"])


def test_parametric_dose_results_report_analytic_inference_when_bootstrap_disabled() -> (
    None
):
    explicit_grid = [0.2, 0.4, 0.8]
    panel = _simulate_two_period_dose_panel(
        n=12000, dgp_id="SIM-002-linear-dose", seed=20260407
    )
    expected_critical = _NORMAL.inv_cdf(1.0 - 0.1 / 2.0)

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", cband=True, bstrap=False),
        dvals=explicit_grid,
        degree=1,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", cband=False, bstrap=False),
        dvals=explicit_grid,
        degree=1,
    )

    for result in (level_result, slope_result):
        assert result.metadata["inference"] == "analytic"
        assert result.metadata["bootstrap_type"] == "analytic"
        assert result.metadata["bootstrap_seed"] is None
        assert result.metadata["confidence_band_kind"] == "pointwise_analytic"
        assert "boot_type" not in result.metadata
        assert result.metadata["summary_aggregates"] == result.metadata["summary"]
        assert result.metadata["bstrap"] is False
        assert result.critical_value == expected_critical
        assert result.confidence_band is not None
        assert result.confidence_band["critical_value"] == expected_critical


def test_parametric_level_inference_includes_untreated_benchmark_variance() -> None:
    panel = _make_untreated_benchmark_variance_panel()
    explicit_grid = [0.2, 0.4, 0.8]

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=2,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=2,
    )

    assert level_result.metadata["inference"] == "analytic"
    assert level_result.std_error == pytest.approx([0.2, 0.2, 0.2])
    assert level_result.confidence_interval != [
        [point, point] for point in level_result.estimate
    ]

    assert slope_result.metadata["inference"] == "analytic"
    assert slope_result.std_error == pytest.approx([0.0, 0.0, 0.0])
    assert slope_result.confidence_interval == [
        [point, point] for point in slope_result.estimate
    ]


def test_parametric_dose_inference_uses_treated_sandwich_covariance() -> None:
    panel = _make_heteroskedastic_treated_fit_panel()
    explicit_grid = [0.2, 1.2]
    treated_dose = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    treated_delta = [0.2, 0.4, 0.0, 1.4, -0.2, 2.4]
    design = [[1.0, dose] for dose in treated_dose]

    x = np.asarray(design, dtype=float)
    y = np.asarray(treated_delta, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    sandwich_covariance = xtx_inv @ (x.T @ np.diag(residual**2) @ x) @ xtx_inv
    level_loadings = np.asarray([[1.0, value] for value in explicit_grid], dtype=float)
    slope_loadings = np.asarray([[0.0, 1.0] for _ in explicit_grid], dtype=float)
    expected_level_se = np.sqrt(
        np.einsum("ij,jk,ik->i", level_loadings, sandwich_covariance, level_loadings)
    )
    expected_slope_se = np.sqrt(
        np.einsum("ij,jk,ik->i", slope_loadings, sandwich_covariance, slope_loadings)
    )

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=1,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=1,
    )

    assert level_result.std_error == pytest.approx(
        expected_level_se.tolist(), abs=1e-12
    )
    assert slope_result.std_error == pytest.approx(
        expected_slope_se.tolist(), abs=1e-12
    )


def test_default_cubic_dose_inference_matches_reference_bspline_loadings() -> None:
    panel = _make_cubic_bspline_inference_panel()
    treated_dose = np.asarray([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 7.0])
    untreated_delta = np.asarray([-0.20, 0.15, 0.35])
    untreated_mean = float(np.mean(untreated_delta))
    untreated_mean_variance = float(
        np.var(untreated_delta, ddof=1) / untreated_delta.size
    )
    residual = np.asarray([0.0, 0.18, -0.24, 0.31, -0.16, 0.22, -0.08])
    treated_delta = (
        0.4
        + 0.25 * treated_dose
        - 0.08 * treated_dose**2
        + 0.015 * treated_dose**3
        + residual
        - untreated_mean
    )
    explicit_grid = [1.25, 2.5, 5.0, 6.5]
    grid = np.asarray(explicit_grid)
    lower = float(treated_dose.min())
    upper = float(treated_dose.max())

    design = np.column_stack(
        [
            np.ones_like(treated_dose),
            _cubic_bspline_no_intercept_columns(
                treated_dose,
                lower=lower,
                upper=upper,
            ),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, treated_delta, rcond=None)
    regression_residual = treated_delta - design @ coefficients
    xtx_inv = np.linalg.pinv(design.T @ design)
    sandwich_covariance = xtx_inv @ (
        design.T @ np.diag(regression_residual**2) @ design
    ) @ xtx_inv
    derivative_loadings = np.column_stack(
        [
            np.zeros_like(grid),
            _cubic_bspline_no_intercept_derivative_columns(
                grid,
                lower=lower,
                upper=upper,
            ),
        ]
    )
    expected_slope_se = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            derivative_loadings,
            sandwich_covariance,
            derivative_loadings,
        )
    )
    level_loadings = np.column_stack(
        [
            np.ones_like(grid),
            _cubic_bspline_no_intercept_columns(
                grid,
                lower=lower,
                upper=upper,
            ),
        ]
    )
    expected_level_se = np.sqrt(
        np.einsum("ij,jk,ik->i", level_loadings, sandwich_covariance, level_loadings)
        + untreated_mean_variance
    )

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", bstrap=False, cband=False),
        dvals=explicit_grid,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", bstrap=False, cband=False),
        dvals=explicit_grid,
    )

    expected_basis = {
        "type": "global_polynomial",
        "degree": 3,
        "num_knots": 0,
        "interior_knots": [],
    }
    assert level_result.metadata["basis"] == expected_basis
    assert slope_result.metadata["basis"] == expected_basis
    assert level_result.std_error == pytest.approx(
        expected_level_se.tolist(),
        abs=1e-12,
    )
    assert slope_result.std_error == pytest.approx(
        expected_slope_se.tolist(),
        abs=1e-12,
    )


def test_linear_spline_dose_inference_matches_reference_bspline_loadings() -> None:
    panel = _make_linear_bspline_inference_panel()
    treated_dose = np.asarray([1.0, 2.0, 3.0, 5.0, 7.0])
    untreated_delta = np.asarray([-0.10, 0.20, 0.40])
    untreated_mean = float(np.mean(untreated_delta))
    untreated_mean_variance = float(
        np.var(untreated_delta, ddof=1) / untreated_delta.size
    )
    residual = np.asarray([0.0, 0.35, -0.25, 0.40, -0.15])
    treated_delta = (
        0.5
        + 0.25 * treated_dose
        + 0.45 * np.clip(treated_dose - 3.0, a_min=0.0, a_max=None)
        + residual
        - untreated_mean
    )
    explicit_grid = [2.0, 3.0, 5.0, 6.0]
    grid = np.asarray(explicit_grid)
    lower = float(treated_dose.min())
    knot = float(np.quantile(treated_dose, 0.5))
    upper = float(treated_dose.max())

    design = np.column_stack(
        [
            np.ones_like(treated_dose),
            _linear_bspline_no_intercept_columns(
                treated_dose,
                lower=lower,
                knot=knot,
                upper=upper,
            ),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, treated_delta, rcond=None)
    regression_residual = treated_delta - design @ coefficients
    xtx_inv = np.linalg.pinv(design.T @ design)
    sandwich_covariance = xtx_inv @ (
        design.T @ np.diag(regression_residual**2) @ design
    ) @ xtx_inv
    level_loadings = np.column_stack(
        [
            np.ones_like(grid),
            _linear_bspline_no_intercept_columns(
                grid,
                lower=lower,
                knot=knot,
                upper=upper,
            ),
        ]
    )
    derivative_loadings = np.column_stack(
        [
            np.zeros_like(grid),
            _linear_bspline_no_intercept_derivative_columns(
                grid,
                lower=lower,
                knot=knot,
                upper=upper,
            ),
        ]
    )
    expected_level_se = np.sqrt(
        np.einsum("ij,jk,ik->i", level_loadings, sandwich_covariance, level_loadings)
        + untreated_mean_variance
    )
    expected_slope_se = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            derivative_loadings,
            sandwich_covariance,
            derivative_loadings,
        )
    )

    level_result = estimate_dose_effects(
        panel,
        _make_inference_spec(target_parameter="level", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=1,
        num_knots=1,
    )
    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=1,
        num_knots=1,
    )

    assert level_result.metadata["basis"]["interior_knots"] == pytest.approx([knot])
    assert slope_result.metadata["basis"]["interior_knots"] == pytest.approx([knot])
    assert level_result.std_error == pytest.approx(
        expected_level_se.tolist(),
        abs=1e-12,
    )
    assert slope_result.std_error == pytest.approx(
        expected_slope_se.tolist(),
        abs=1e-12,
    )


def test_parametric_level_inference_requires_two_untreated_units() -> None:
    panel = _make_single_untreated_panel()
    explicit_grid = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="untreated benchmark variance requires at least two untreated units"
    ):
        estimate_dose_effects(
            panel,
            _make_inference_spec(target_parameter="level", bstrap=False, cband=False),
            dvals=explicit_grid,
            degree=2,
        )

    slope_result = estimate_dose_slope_effects(
        panel,
        _make_inference_spec(target_parameter="slope", bstrap=False, cband=False),
        dvals=explicit_grid,
        degree=2,
    )
    assert slope_result.metadata["inference"] == "analytic"


def test_analytic_parametric_dose_routes_reject_exact_fit_support_without_inference_df() -> None:
    panel = _make_perfect_fit_panel()
    expected_points = [0.2, 0.5, 0.8]

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_effects(
            panel,
            _make_inference_spec(target_parameter="level", cband=True, bstrap=False),
            dvals=expected_points,
            degree=2,
        )

    with pytest.raises(
        ContDIDValidationError, match="residual degrees of freedom for inference"
    ):
        estimate_dose_slope_effects(
            panel,
            _make_inference_spec(target_parameter="slope", cband=False, bstrap=False),
            dvals=expected_points,
            degree=2,
        )


def test_simulate_contdid_data_supports_phase6_sim004_anchor() -> None:
    panel = simulate_contdid_data(
        n=4000,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=20260407,
    )

    frame = panel.frame
    assert frame["time_period"].nunique() == 4
    assert sorted(frame["G"].unique().tolist()) == [0, 2, 3, 4]
    assert (frame.loc[frame["G"] == 0, "D"] == 0.0).all()
    treated_doses = frame.loc[frame["G"] != 0, "D"]
    assert treated_doses.gt(0.0).any()
