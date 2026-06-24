from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from contdid import ContDIDSpec, PanelData, simulate_contdid_data
from contdid.validation import ContDIDValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE6_FIXTURE_PATH = (
    REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase6_inference_expected.json"
)


def _load_fixture() -> dict:
    return json.loads(PHASE6_FIXTURE_PATH.read_text(encoding="utf-8"))


def _make_eventstudy_spec(
    *, target_parameter: str, dose_est_method: str = "parametric"
) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method=dose_est_method,
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )


def _make_unbalanced_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    group_specs = {
        2: [(1.0, 0.0), (2.0, 0.2), (1.5, -0.1), (2.5, 0.1)],
        3: [(1.0, -0.15), (2.0, 0.25)],
        4: [(1.0, 0.1), (2.0, -0.2)],
    }
    group_scale = {2: 1.0, 3: 3.0, 4: 5.0}

    for group, dose_shocks in group_specs.items():
        for dose, shock in dose_shocks:
            for time_period in [1, 2, 3, 4]:
                outcome = 0.0
                if time_period >= group:
                    duration = time_period - group + 1
                    outcome = duration * (group_scale[group] * dose + shock)
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_control_noise_eventstudy_panel() -> PanelData:
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "time_period": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Y": [0.0, 2.5, 0.0, 4.5, 0.0, 3.5, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0],
            "G": [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
            "D": [1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return PanelData(frame=frame)


def _make_exact_dose_fit_binary_level_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0, 4.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, dose**2, 2, dose),
            ]
        )
        unit_id += 1

    for _ in range(4):
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, 0.0, 0, 0.0),
            ]
        )
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_quadratic_slope_distribution_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0, 4.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, dose**2, 2, dose),
                (unit_id, 3, 2.0 * dose**2, 2, dose),
            ]
        )
        unit_id += 1

    for _ in range(4):
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, 0.0, 0, 0.0),
                (unit_id, 3, 0.0, 0, 0.0),
            ]
        )
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_shared_control_overlap_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, 10.0 * dose, 2, dose),
                (unit_id, 3, 20.0 * dose, 2, dose),
            ]
        )
        unit_id += 1

    for dose in [1.0, 2.0, 3.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 3, dose),
                (unit_id, 2, 0.0, 3, dose),
                (unit_id, 3, 20.0 * dose, 3, dose),
            ]
        )
        unit_id += 1

    for slope in [1.0, 2.0, 3.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, slope, 0, 0.0),
                (unit_id, 3, 2.0 * slope, 0, 0.0),
            ]
        )
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_perfectly_correlated_benchmark_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, 10.0 * dose, 2, dose),
                (unit_id, 3, 20.0 * dose, 2, dose),
            ]
        )
        unit_id += 1

    for slope in [1.0, 2.0, 3.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, slope, 0, 0.0),
                (unit_id, 3, 2.0 * slope, 0, 0.0),
            ]
        )
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_fixed_baseline_pretrend_covariance_panel() -> PanelData:
    rows = [
        ("t1", 1, -1.0, 3, 1.0),
        ("t1", 2, 0.0, 3, 1.0),
        ("t1", 3, 1.0, 3, 1.0),
        ("t2", 1, 1.0, 3, 2.0),
        ("t2", 2, 0.0, 3, 2.0),
        ("t2", 3, 2.0, 3, 2.0),
        ("t3", 1, 0.0, 3, 3.0),
        ("t3", 2, 0.0, 3, 3.0),
        ("t3", 3, 3.0, 3, 3.0),
        ("c1", 1, 0.0, 0, 0.0),
        ("c1", 2, 0.0, 0, 0.0),
        ("c1", 3, 0.0, 0, 0.0),
        ("c2", 1, -1.0, 0, 0.0),
        ("c2", 2, 0.0, 0, 0.0),
        ("c2", 3, -2.0, 0, 0.0),
        ("c3", 1, 1.0, 0, 0.0),
        ("c3", 2, 0.0, 0, 0.0),
        ("c3", 3, 2.0, 0, 0.0),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_shared_treated_residual_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []

    for unit_id, (dose, event0_residual) in enumerate(
        [(1.0, -1.0), (2.0, 2.0), (3.0, -1.0)],
        start=1,
    ):
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, 10.0 * dose + event0_residual, 2, dose),
                (unit_id, 3, 20.0 * dose + 2.0 * event0_residual, 2, dose),
            ]
        )

    for unit_id in [4, 5, 6]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, 0.0, 0, 0.0),
                (unit_id, 3, 0.0, 0, 0.0),
            ]
        )

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_notyettreated_cross_role_covariance_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []

    for unit_index, dose in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        rows.extend(
            [
                (f"g2_{unit_index}", 1, 0.0, 2, dose),
                (f"g2_{unit_index}", 2, dose, 2, dose),
                (f"g2_{unit_index}", 3, 2.0 * dose, 2, dose),
            ]
        )

    for unit_index, (dose, pre_delta, post_residual) in enumerate(
        [
            (1.0, -1.0, -1.0),
            (2.0, 1.0, 2.0),
            (3.0, -1.0, -1.0),
            (4.0, 1.0, 2.0),
        ],
        start=1,
    ):
        rows.extend(
            [
                (f"g3_{unit_index}", 1, 0.0, 3, dose),
                (f"g3_{unit_index}", 2, pre_delta, 3, dose),
                (
                    f"g3_{unit_index}",
                    3,
                    pre_delta + 10.0 * dose + post_residual,
                    3,
                    dose,
                ),
            ]
        )

    for unit_index in range(1, 5):
        rows.extend(
            [
                (f"u{unit_index}", 1, 0.0, 0, 0.0),
                (f"u{unit_index}", 2, 0.0, 0, 0.0),
                (f"u{unit_index}", 3, 0.0, 0, 0.0),
            ]
        )

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_pairwise_overlap_psd_eventstudy_panel() -> PanelData:
    rows = [
        (1, 1, -0.06724584648435186, 2, 0.576),
        (1, 2, 1.4317630689464176, 2, 0.576),
        (1, 3, 0.10416894751927833, 2, 0.576),
        (1, 4, 3.755208806276734, 2, 0.576),
        (2, 1, 0.8752375681740906, 2, 1.412),
        (2, 2, 2.8111239046852186, 2, 1.412),
        (2, 3, 0.3712227800873885, 2, 1.412),
        (2, 4, 3.323605098018995, 2, 1.412),
        (3, 1, -1.8994135345839847, 3, 2.83),
        (3, 2, -0.30411129214310473, 3, 2.83),
        (3, 3, -3.3211586153910266, 3, 2.83),
        (3, 4, -5.684020271736424, 3, 2.83),
        (4, 1, 0.29333288574224836, 3, 1.487),
        (4, 2, -1.22791606383464, 3, 1.487),
        (4, 3, -0.13531101075817364, 3, 1.487),
        (4, 4, -4.092252883347414, 3, 1.487),
        (5, 1, 1.3045641730895912, 4, 2.22),
        (5, 2, -0.7524648429855153, 4, 2.22),
        (5, 3, 1.5474321245561427, 4, 2.22),
        (5, 4, 2.826139196091131, 4, 2.22),
        (6, 1, -0.2832631532860423, 4, 1.849),
        (6, 2, 0.1297965258684221, 4, 1.849),
        (6, 3, 0.7385514744224353, 4, 1.849),
        (6, 4, -4.063668170206491, 4, 1.849),
        (7, 1, 1.164329389500474, 0, 0.0),
        (7, 2, 0.18487152945236263, 0, 0.0),
        (7, 3, -0.27530283164342295, 0, 0.0),
        (7, 4, 0.16100979984859665, 0, 0.0),
        (8, 1, -4.648601054772447, 0, 0.0),
        (8, 2, -4.831572077981771, 0, 0.0),
        (8, 3, -6.5348997793569765, 0, 0.0),
        (8, 4, -10.917465943274, 0, 0.0),
    ]
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_perfect_fit_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, dose in [
        (2, 0.2),
        (2, 0.8),
        (3, 0.3),
        (3, 0.9),
        (0, 0.0),
        (0, 0.0),
    ]:
        for time_period in [1, 2, 3, 4]:
            outcome = 0.0
            if group and time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_partial_local_support_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, dose in [
        (2, 1.0),
        (2, 1.0),
        (3, 1.0),
        (3, 2.0),
        (0, 0.0),
        (0, 0.0),
    ]:
        for time_period in [1, 2, 3, 4]:
            outcome = 0.0
            if group and time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_mixed_local_inference_support_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    group2_specs = [
        (1.0, (0.0, 1.0)),
        (2.0, (1.0, -1.0)),
        (3.0, (-1.0, 0.5)),
    ]
    for dose, shocks in group2_specs:
        for time_period in [1, 2, 3]:
            outcome = 0.0
            if time_period == 2:
                outcome = 10.0 * dose + shocks[0]
            elif time_period == 3:
                outcome = 20.0 * dose + shocks[1]
            rows.append((unit_id, time_period, outcome, 2, dose))
        unit_id += 1

    for dose in [1.0, 2.0]:
        for time_period in [1, 2, 3]:
            outcome = 50.0 * dose if time_period == 3 else 0.0
            rows.append((unit_id, time_period, outcome, 3, dose))
        unit_id += 1

    for slope in [0.0, 1.0, 2.0]:
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, slope * time_period, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_spline_slope_cross_term_eventstudy_panel() -> PanelData:
    doses = [
        1.0,
        1.8181818181818183,
        2.6363636363636367,
        3.4545454545454546,
        4.272727272727273,
        5.090909090909091,
        5.909090909090909,
        6.7272727272727275,
        7.545454545454546,
        8.363636363636363,
        9.181818181818182,
        10.0,
    ]
    outcomes = [
        6.4530640598906555,
        7.876242704644181,
        13.01198806488132,
        11.090148722337153,
        9.62928007326326,
        17.977362448674768,
        28.717099526820043,
        34.05571176476945,
        32.51074378431763,
        35.00402836521919,
        42.26072916839166,
        49.57456656406811,
    ]
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose, outcome in zip(doses, outcomes):
        rows.extend(
            [
                (unit_id, 1, 0.0, 2, dose),
                (unit_id, 2, outcome, 2, dose),
            ]
        )
        unit_id += 1

    for _ in doses:
        rows.extend(
            [
                (unit_id, 1, 0.0, 0, 0.0),
                (unit_id, 2, 0.0, 0, 0.0),
            ]
        )
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _spline_cross_term_doses() -> list[float]:
    return [
        1.0,
        1.8181818181818183,
        2.6363636363636367,
        3.4545454545454546,
        4.272727272727273,
        5.090909090909091,
        5.909090909090909,
        6.7272727272727275,
        7.545454545454546,
        8.363636363636363,
        9.181818181818182,
        10.0,
    ]


def _treated_share_weighted_se(entry: dict[str, object]) -> float:
    cohort_estimates = entry["cohort_estimates"]
    variance = sum(
        (item["aggregation_weight"] * item["std_error"]) ** 2
        for item in cohort_estimates
    )
    return math.sqrt(variance)


def test_phase6_fixture_covers_sim004_and_sim005() -> None:
    fixture = _load_fixture()

    assert fixture["schema_version"] == "0.1"
    assert fixture["application_id"] == "phase6-package-inference-expected"
    assert list(fixture["scenarios"]) == [
        "SIM-004-staggered-eventstudy-null",
        "SIM-005-cck-two-period",
    ]


def test_sim004_eventstudy_level_requires_phase6_inference_payload_and_exact_rules() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    fixture = _load_fixture()["scenarios"]["SIM-004-staggered-eventstudy-null"]
    panel = simulate_contdid_data(
        n=16000,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=fixture["default_seed"],
    )

    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        degree=1,
    )

    assert result.estimand == "ATT(event_time)"
    assert result.event_time_grid == fixture["grid"]["values"]
    assert (
        result.metadata["event_time_grid"]
        == fixture["exact_rule_targets"]["event_time_grid"]
    )
    assert result.metadata["support"] == fixture["exact_rule_targets"]["support"]
    assert result.metadata["summary_aggregates"]["overall_level"] == pytest.approx(
        fixture["summary_aggregates"]["overall_level"],
        abs=1e-10,
    )
    assert result.metadata["summary_aggregates"][
        "post_treatment_mean_level"
    ] == pytest.approx(
        fixture["summary_aggregates"]["post_treatment_mean_level"],
        abs=1e-10,
    )
    assert result.std_error == pytest.approx(fixture["std_error"], abs=1e-10)
    assert result.critical_value == pytest.approx(fixture["critical_value"], abs=1e-10)
    assert result.confidence_band["critical_value"] == pytest.approx(
        fixture["confidence_band"]["critical_value"],
        abs=1e-10,
    )
    assert result.confidence_band["lower"] == pytest.approx(
        fixture["confidence_band"]["lower"],
        abs=1e-10,
    )
    assert result.confidence_band["upper"] == pytest.approx(
        fixture["confidence_band"]["upper"],
        abs=1e-10,
    )
    assert result.metadata["inference"] == "bootstrap"
    assert result.metadata["identification"] == {
        "paper_estimand": "ATT(event_time)",
        "identifying_assumption": "PT-MP",
        "ordinary_pt_interpretation": (
            "post-treatment ATT(event_time); negative event-time cells are "
            "pre-trend diagnostics"
        ),
        "identification_note": (
            "Post-treatment ATT(event_time) cells are identified by PT-MP/local "
            "binary event-study comparisons; negative event-time cells diagnose "
            "pre-treatment parallel-trends plausibility rather than treatment effects."
        ),
    }


def test_sim004_eventstudy_slope_requires_phase6_inference_payload_and_exact_rules() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects

    scenario = _load_fixture()["scenarios"]["SIM-004-staggered-eventstudy-null"]
    fixture = scenario["slope"]
    panel = simulate_contdid_data(
        n=16000,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=scenario["default_seed"],
    )

    result = estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        degree=1,
    )

    assert result.estimand == "ACRT(event_time)"
    assert result.event_time_grid == scenario["grid"]["values"]
    assert (
        result.metadata["event_time_grid"]
        == fixture["exact_rule_targets"]["event_time_grid"]
    )
    assert result.metadata["support"] == fixture["exact_rule_targets"]["support"]
    assert result.metadata["summary_aggregates"]["overall_slope"] == pytest.approx(
        fixture["summary_aggregates"]["overall_slope"],
        abs=1e-10,
    )
    assert result.metadata["summary_aggregates"][
        "post_treatment_mean_slope"
    ] == pytest.approx(
        fixture["summary_aggregates"]["post_treatment_mean_slope"],
        abs=1e-10,
    )
    assert result.std_error == pytest.approx(fixture["std_error"], abs=1e-10)
    assert result.critical_value == pytest.approx(fixture["critical_value"], abs=1e-10)
    assert result.confidence_band["critical_value"] == pytest.approx(
        fixture["confidence_band"]["critical_value"],
        abs=1e-10,
    )
    assert result.confidence_band["lower"] == pytest.approx(
        fixture["confidence_band"]["lower"],
        abs=1e-10,
    )
    assert result.confidence_band["upper"] == pytest.approx(
        fixture["confidence_band"]["upper"],
        abs=1e-10,
    )
    assert result.metadata["inference"] == "bootstrap"
    assert result.metadata["identification"] == {
        "paper_estimand": "ACRT(event_time)",
        "identifying_assumption": "SPT-MP + continuous dose support",
        "ordinary_pt_interpretation": (
            "derivative of event-time LATT path with local selection-bias "
            "contamination under PT-MP alone"
        ),
        "identification_note": (
            "The public slope event-study route reports the SPT-MP causal-response "
            "label; under PT-MP alone, differentiating event-time paths can retain "
            "selection-bias terms."
        ),
    }


def test_eventstudy_slope_inference_uses_treated_share_weights_for_standard_errors() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects

    panel = _make_unbalanced_eventstudy_panel()
    result = estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[1.0],
        degree=1,
    )

    entry0 = next(entry for entry in result.cohort_summary if entry["event_time"] == 0)
    entry1 = next(entry for entry in result.cohort_summary if entry["event_time"] == 1)
    expected0 = _treated_share_weighted_se(entry0)
    expected1 = _treated_share_weighted_se(entry1)
    idx0 = result.event_time.index(0)
    idx1 = result.event_time.index(1)

    assert sum(
        cohort["aggregation_weight"] for cohort in entry0["cohort_estimates"]
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert [
        cohort["aggregation_weight"] for cohort in entry1["cohort_estimates"]
    ] == pytest.approx(
        [
            cohort["treated_count"]
            / sum(item["treated_count"] for item in entry1["cohort_estimates"])
            for cohort in entry1["cohort_estimates"]
        ],
        abs=1e-12,
    )
    assert entry0["std_error"] == pytest.approx(expected0, abs=1e-12)
    assert entry1["std_error"] == pytest.approx(expected1, abs=1e-12)
    assert result.std_error[idx0] == pytest.approx(expected0, abs=1e-12)
    assert result.std_error[idx1] == pytest.approx(expected1, abs=1e-12)
    assert result.confidence_band["lower"][idx0] == pytest.approx(
        result.estimate[idx0] - result.critical_value * expected0,
        abs=1e-12,
    )
    assert result.confidence_band["upper"][idx1] == pytest.approx(
        result.estimate[idx1] + result.critical_value * expected1,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_inference_supports_nevertreated_control_group(
    target_parameter: str, estimator_name: str
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    panel = _make_unbalanced_eventstudy_panel()
    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        panel,
        ContDIDSpec(
            target_parameter=target_parameter,
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0],
        degree=1,
    )

    assert result.metadata["control_group"] == "nevertreated"
    assert result.metadata["inference"] == "bootstrap"
    assert result.metadata["timing_group_support"]["timing_groups"] == (
        [2, 3, 4] if target_parameter == "level" else [2]
    )
    assert result.confidence_band is not None
    assert result.confidence_interval is not None
    assert len(result.std_error) == len(result.event_time)


def test_eventstudy_level_inference_is_invariant_to_explicit_dose_grid() -> None:
    from contdid import estimate_eventstudy_effects

    panel = _make_unbalanced_eventstudy_panel()
    low_grid = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[1.0],
        degree=1,
    )
    high_grid = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[2.0],
        degree=1,
    )

    assert high_grid.std_error == pytest.approx(low_grid.std_error, abs=1e-12)
    assert high_grid.confidence_interval == [
        pytest.approx(interval, abs=1e-12) for interval in low_grid.confidence_interval
    ]
    assert high_grid.confidence_band["lower"] == pytest.approx(
        low_grid.confidence_band["lower"],
        abs=1e-12,
    )
    assert high_grid.confidence_band["upper"] == pytest.approx(
        low_grid.confidence_band["upper"],
        abs=1e-12,
    )
    assert high_grid.confidence_band["critical_value"] == pytest.approx(
        low_grid.confidence_band["critical_value"],
        abs=1e-12,
    )
    assert low_grid.metadata["dose_grid"] == [1.0]
    assert high_grid.metadata["dose_grid"] == [2.0]
    assert low_grid.metadata["summary"] == low_grid.metadata["summary_aggregates"]
    assert high_grid.metadata["summary"] == high_grid.metadata["summary_aggregates"]


def test_eventstudy_slope_inference_is_invariant_to_explicit_dose_grid() -> None:
    from contdid import estimate_eventstudy_slope_effects

    panel = _make_unbalanced_eventstudy_panel()
    low_grid = estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[1.0],
        degree=1,
    )
    high_grid = estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[2.0],
        degree=1,
    )

    assert high_grid.std_error == pytest.approx(low_grid.std_error, abs=1e-12)
    assert high_grid.confidence_interval == [
        pytest.approx(interval, abs=1e-12) for interval in low_grid.confidence_interval
    ]
    assert high_grid.confidence_band["lower"] == pytest.approx(
        low_grid.confidence_band["lower"],
        abs=1e-12,
    )
    assert high_grid.confidence_band["upper"] == pytest.approx(
        low_grid.confidence_band["upper"],
        abs=1e-12,
    )
    assert high_grid.confidence_band["critical_value"] == pytest.approx(
        low_grid.confidence_band["critical_value"],
        abs=1e-12,
    )
    assert low_grid.metadata["dose_grid"] == [1.0]
    assert high_grid.metadata["dose_grid"] == [2.0]
    assert low_grid.metadata["summary"] == low_grid.metadata["summary_aggregates"]
    assert high_grid.metadata["summary"] == high_grid.metadata["summary_aggregates"]


def test_eventstudy_level_standard_error_keeps_untreated_benchmark_noise() -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    panel = _make_control_noise_eventstudy_panel()
    level_result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[1.0, 2.0],
        degree=1,
    )
    slope_result = estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[1.0, 2.0],
        degree=1,
    )

    treated_delta = pd.Series([2.5, 4.5, 3.5], dtype=float)
    untreated_delta = pd.Series([0.0, 1.0, 2.0], dtype=float)
    expected_binary_did_se = math.sqrt(
        treated_delta.var(ddof=1) / treated_delta.size
        + untreated_delta.var(ddof=1) / untreated_delta.size
    )

    assert level_result.estimate == pytest.approx([2.5], abs=1e-12)
    assert level_result.std_error == pytest.approx([expected_binary_did_se], abs=1e-12)
    assert level_result.cohort_summary[0]["std_error"] == pytest.approx(
        expected_binary_did_se,
        abs=1e-12,
    )
    assert slope_result.estimate == pytest.approx([2.0], abs=1e-12)
    assert slope_result.std_error == pytest.approx([0.0], abs=1e-12)
    assert slope_result.cohort_summary[0]["std_error"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_eventstudy_level_standard_error_uses_binary_treated_mean_variation() -> None:
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_exact_dose_fit_binary_level_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=False,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=2,
    )

    treated_delta = pd.Series([1.0, 4.0, 9.0, 16.0], dtype=float)
    expected_treated_mean_se = math.sqrt(treated_delta.var(ddof=1) / treated_delta.size)

    assert result.estimate == pytest.approx([7.5], abs=1e-12)
    assert result.std_error == pytest.approx([expected_treated_mean_se], abs=1e-12)
    assert result.cohort_summary[0]["std_error"] == pytest.approx(
        expected_treated_mean_se,
        abs=1e-12,
    )


def test_eventstudy_slope_inference_keeps_derivative_distribution_covariance() -> None:
    from contdid import estimate_eventstudy_slope_effects
    from contdid.inference import compute_multiplier_bootstrap

    result = estimate_eventstudy_slope_effects(
        _make_quadratic_slope_distribution_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=2,
    )
    expected_bootstrap = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[5.0 / 3.0, 10.0 / 3.0], [10.0 / 3.0, 20.0 / 3.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert result.event_time == [0, 1]
    assert result.estimate == pytest.approx([5.0, 10.0], abs=1e-12)
    assert result.std_error == pytest.approx(
        [math.sqrt(5.0 / 3.0), math.sqrt(20.0 / 3.0)],
        abs=1e-12,
    )
    assert result.cohort_summary[0]["std_error"] == pytest.approx(
        math.sqrt(5.0 / 3.0),
        abs=1e-12,
    )
    assert result.critical_value == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )
    assert result.confidence_band["critical_value"] == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_slope_inference_uses_full_average_derivative_influence_function() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects

    result = estimate_eventstudy_slope_effects(
        _make_spline_slope_cross_term_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=False,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=_spline_cross_term_doses(),
        degree=1,
        num_knots=2,
    )

    assert result.event_time == [0]
    assert result.estimate == pytest.approx([4.345736076762626], abs=1e-12)
    assert result.std_error == pytest.approx([0.7219062535449848], abs=1e-12)
    assert result.std_error[0] < 0.7884668629303326
    assert result.cohort_summary[0]["std_error"] == pytest.approx(
        result.std_error[0],
        abs=1e-12,
    )


def test_eventstudy_level_standard_error_preserves_shared_control_benchmark_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_shared_control_overlap_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=False,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
    )

    expected_binary_did_se = math.sqrt(42.0)
    idx = result.event_time.index(0)

    assert result.estimate[idx] == pytest.approx(28.0, abs=1e-12)
    assert result.std_error[idx] == pytest.approx(expected_binary_did_se, abs=1e-12)
    assert result.cohort_summary[idx]["std_error"] == pytest.approx(
        expected_binary_did_se,
        abs=1e-12,
    )


def test_eventstudy_level_confidence_band_preserves_shared_control_benchmark_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_effects
    from contdid.inference import compute_multiplier_bootstrap

    result = estimate_eventstudy_effects(
        _make_perfectly_correlated_benchmark_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
    )

    expected_bootstrap = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[
            [101.0 / 3.0, 202.0 / 3.0],
            [202.0 / 3.0, 404.0 / 3.0],
        ],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert result.event_time == [0, 1]
    assert result.std_error == pytest.approx(
        [math.sqrt(101.0 / 3.0), math.sqrt(404.0 / 3.0)],
        abs=1e-12,
    )
    assert result.critical_value == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )
    assert result.confidence_band["critical_value"] == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_level_fixed_baseline_pretrend_band_preserves_cross_event_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_effects
    from contdid.inference import compute_multiplier_bootstrap

    result = estimate_eventstudy_effects(
        _make_fixed_baseline_pretrend_covariance_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
        base_period=2,
    )
    expected_bootstrap = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[
            [2.0 / 3.0, 5.0 / 6.0],
            [5.0 / 6.0, 5.0 / 3.0],
        ],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert result.event_time == [-2, 0]
    assert result.estimate == pytest.approx([0.0, 2.0], abs=1e-12)
    assert result.metadata["base_period"] == 2
    assert result.metadata["timing_group_support"]["base_period_strategy"] == "fixed"
    assert result.std_error == pytest.approx(
        [math.sqrt(2.0 / 3.0), math.sqrt(5.0 / 3.0)],
        abs=1e-12,
    )
    assert result.critical_value == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )
    assert result.confidence_band["critical_value"] == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_level_covariance_matrix_remains_psd_with_sparse_pairwise_overlap() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_pairwise_overlap_psd_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        degree=1,
    )

    assert result.event_time == [-2, -1, 0, 1, 2]
    assert result.confidence_band is not None
    assert result.critical_value is not None and result.critical_value > 0.0
    assert all(value >= 0.0 for value in result.std_error)


def test_eventstudy_level_confidence_band_preserves_shared_treated_fit_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_effects
    from contdid.inference import compute_multiplier_bootstrap

    result = estimate_eventstudy_effects(
        _make_shared_treated_residual_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
    )

    expected_bootstrap = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[103.0 / 3.0, 206.0 / 3.0], [206.0 / 3.0, 412.0 / 3.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert result.event_time == [0, 1]
    assert result.std_error == pytest.approx(
        [math.sqrt(103.0 / 3.0), math.sqrt(412.0 / 3.0)],
        abs=1e-12,
    )
    assert result.critical_value == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )
    assert result.confidence_band["critical_value"] == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_level_notyettreated_variance_preserves_cross_role_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_effects
    from contdid.inference import compute_multiplier_bootstrap

    panel = _make_notyettreated_cross_role_covariance_panel()
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=False,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    result = estimate_eventstudy_effects(
        panel,
        spec,
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=1,
    )
    band_result = estimate_eventstudy_effects(
        panel,
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=1,
    )

    event0 = result.cohort_summary[result.event_time.index(0)]
    event0_variance = 11.474704588813678
    event_minus1_event0_covariance = 1.0061816583522876

    assert event0["timing_groups"] == [2, 3]
    assert event0["std_error"] == pytest.approx(
        math.sqrt(event0_variance),
        abs=1e-12,
    )
    assert result.std_error[result.event_time.index(0)] == pytest.approx(
        math.sqrt(event0_variance),
        abs=1e-12,
    )
    expected_band = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        covariance=[
            [1.0 / 3.0, event_minus1_event0_covariance, 0.0],
            [event_minus1_event0_covariance, event0_variance, 5.0 / 12.0],
            [0.0, 5.0 / 12.0, 5.0 / 3.0],
        ],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert band_result.event_time == [-1, 0, 1]
    assert band_result.std_error == pytest.approx(
        expected_band["std_error"],
        abs=1e-12,
    )
    assert band_result.confidence_band["critical_value"] == pytest.approx(
        expected_band["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_slope_confidence_band_preserves_shared_treated_fit_covariance() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects
    from contdid.inference import compute_multiplier_bootstrap

    result = estimate_eventstudy_slope_effects(
        _make_shared_treated_residual_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
    )

    expected_bootstrap = compute_multiplier_bootstrap(
        loadings=[[1.0, 0.0], [0.0, 1.0]],
        covariance=[[0.75, 1.5], [1.5, 3.0]],
        alp=0.1,
        bstrap=True,
        cband=True,
        boot_type="multiplier",
        biters=199,
    )

    assert result.event_time == [0, 1]
    assert result.std_error == pytest.approx([0.75**0.5, 3.0**0.5], abs=1e-12)
    assert result.critical_value == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )
    assert result.confidence_band["critical_value"] == pytest.approx(
        expected_bootstrap["critical_value"],
        abs=1e-12,
    )


def test_eventstudy_slope_inference_rejects_post_treatment_rows_without_local_inference_df() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "requires at least one locally identified post-treatment event time "
            "with positive-dose support and inference degrees of freedom"
        ),
    ):
        estimate_eventstudy_slope_effects(
            _make_partial_local_support_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter="slope"),
            dvals=[1.0, 2.0],
            degree=1,
        )


def test_eventstudy_slope_rejects_perfect_local_fit_without_inference_df() -> None:
    from contdid import estimate_eventstudy_slope_effects

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "requires at least one locally identified post-treatment event time "
            "with positive-dose support and inference degrees of freedom"
        ),
    ):
        estimate_eventstudy_slope_effects(
            _make_perfect_fit_eventstudy_panel(),
            ContDIDSpec(
                target_parameter="slope",
                aggregation="eventstudy",
                dose_est_method="parametric",
                control_group="notyettreated",
                treatment_type="continuous",
                anticipation=0,
                alp=0.1,
                bstrap=False,
                cband=True,
                boot_type="multiplier",
                biters=199,
            ),
            dvals=[0.2, 0.8],
            degree=1,
        )


def test_eventstudy_omits_local_rows_without_inference_df_before_weighting() -> None:
    from contdid import estimate_eventstudy_slope_effects

    result = estimate_eventstudy_slope_effects(
        _make_mixed_local_inference_support_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=False,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
    )

    assert result.event_time == [0, 1]
    assert result.estimate == pytest.approx([9.5, 19.75], abs=1e-12)
    assert result.std_error == pytest.approx(
        [0.4330127018922188, 0.5051814855409108],
        abs=1e-12,
    )
    assert [entry["timing_groups"] for entry in result.cohort_summary] == [[2], [2]]


def test_sim005_cck_eventstudy_fixed_dimension_succeeds() -> None:
    """CCK with fixed dimension should now succeed in event study."""
    from contdid import estimate_eventstudy_effects

    fixture = _load_fixture()["scenarios"]["SIM-005-cck-two-period"]
    panel = simulate_contdid_data(
        n=8000,
        dgp_id="SIM-005-cck-two-period",
        seed=fixture["default_seed"],
    )

    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level", dose_est_method="cck"),
        degree=2,
        num_knots=0,
    )
    assert result.estimand == "ATT(event_time)"
    assert result.metadata["dose_est_method"] == "cck"
