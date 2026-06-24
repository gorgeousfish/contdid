from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contdid import ContDIDSpec, PanelData, simulate_contdid_data
from contdid.validation import ContDIDValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTSTUDY_FIXTURE = (
    REPO_ROOT / "contdid-py" / "tests" / "fixtures" / "phase5_eventstudy_expected.json"
)


def _load_fixture() -> dict:
    return json.loads(EVENTSTUDY_FIXTURE.read_text(encoding="utf-8"))


def _make_eventstudy_spec(
    *,
    target_parameter: str,
    dose_est_method: str = "parametric",
    control_group: str = "notyettreated",
) -> ContDIDSpec:
    return ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method=dose_est_method,
        control_group=control_group,
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


def _make_slope_multicohort_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    group_specs = {
        2: [(1.0, 0.0), (1.5, 0.3), (2.0, -0.2), (2.5, 0.1)],
        3: [(1.0, -0.1), (1.5, 0.2), (2.0, -0.15)],
        4: [(1.0, 0.15), (1.5, -0.05), (2.0, 0.25)],
    }
    group_scale = {2: 1.0, 3: 2.5, 4: 4.0}

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


def _make_anticipation_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    for unit_id, (group, dose) in enumerate(
        [(3, 1.0), (3, 2.0), (4, 1.5), (0, 0.0), (0, 0.0)],
        start=1,
    ):
        for time_period in [1, 2, 3, 4, 5]:
            outcome = 0.0
            if group and time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
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


def _make_future_treated_only_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    for unit_id, (group, dose) in enumerate(
        [(6, 1.5), (6, 2.5), (0, 0.0), (0, 0.0)],
        start=1,
    ):
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, group, dose))

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_single_cohort_multiperiod_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    for unit_id, (group, dose) in enumerate(
        [(3, 1.0), (3, 2.0), (0, 0.0), (0, 0.0)],
        start=1,
    ):
        for time_period in [1, 2, 3, 4]:
            outcome = 0.0
            if group and time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_cubic_slope_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 7.0]:
        dose_effect = 0.4 + 0.25 * dose - 0.08 * dose**2 + 0.015 * dose**3
        for time_period in [1, 2, 3]:
            duration = max(time_period - 1, 0)
            rows.append((unit_id, time_period, duration * dose_effect, 2, dose))
        unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_linear_slope_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0, 4.0]:
        for time_period in [1, 2, 3]:
            outcome = 0.0
            if time_period >= 2:
                duration = float(time_period - 1)
                outcome = duration * (5.0 + 2.0 * dose)
            rows.append((unit_id, time_period, outcome, 2, dose))
        unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_staggered_linear_slope_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for timing_group in [2, 3]:
        for dose in [1.0, 2.0, 3.0, 4.0]:
            for time_period in [1, 2, 3, 4]:
                exposure = max(time_period - timing_group + 1, 0)
                outcome = float(exposure) * (5.0 + 2.0 * dose)
                rows.append((unit_id, time_period, outcome, timing_group, dose))
            unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_future_comparison_eventstudy_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []

    for unit_index, dose in enumerate([1.0, 2.0], start=1):
        for time_period in [1, 2, 3]:
            outcome = 0.0
            if time_period >= 2:
                outcome = 10.0 * float(time_period - 1) * dose
            rows.append((f"g2_{unit_index}", time_period, outcome, 2, dose))

    for unit_index, dose in enumerate([1.5, 2.5], start=1):
        for time_period in [1, 2, 3]:
            rows.append(
                (f"g4_{unit_index}", time_period, float(time_period - 1), 4, dose)
            )

    for time_period in [1, 2, 3]:
        rows.append(("u1", time_period, 0.0, 0, 0.0))

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_first_period_and_identified_later_cohort_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, doses in [(1, [100.0, 200.0, 300.0]), (2, [1.0, 2.0, 3.0])]:
        for dose in doses:
            for time_period in [1, 2, 3]:
                outcome = 0.0
                if time_period >= group:
                    outcome = float(time_period - group + 1) * dose
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_global_knot_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, doses in [
        (2, [1.0, 2.0, 3.0, 4.0, 5.0]),
        (3, [3.0, 4.0, 5.0, 6.0, 7.0]),
    ]:
        for dose in doses:
            for time_period in [1, 2, 3]:
                outcome = 0.0
                if time_period >= group:
                    outcome = max(dose - 4.0, 0.0) * float(time_period - group + 1)
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    for _ in range(5):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_preonly_identified_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []

    for time_period, outcome in [(1, 0.0), (2, 1.5), (3, 3.0), (4, 4.5)]:
        rows.append((1, time_period, outcome, 2, 1.5))

    for unit_id, dose in [(2, 0.5), (3, 1.5), (4, 2.0)]:
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, 6, dose))

    for unit_id in [5, 6]:
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_dropped_high_dose_cohort_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, doses in [
        (2, [100.0]),
        (3, [1.0, 1.5, 2.0]),
        (0, [0.0, 0.0, 0.0]),
    ]:
        for dose in doses:
            for time_period in [1, 2, 3, 4]:
                outcome = 0.0
                if group and time_period >= group:
                    outcome = float(time_period - group + 1) * dose
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_notyettreated_self_only_pre_period_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, dose in [(2, 1.0), (2, 2.0), (2, 3.0), (3, 1.0), (3, 2.0)]:
        for time_period in [1, 2, 3]:
            outcome = 0.0
            if time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_pretrend_contaminated_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0, 3.0, 4.0]:
        rows.extend(
            [
                (unit_id, 1, 0.0, 3, dose),
                (unit_id, 2, 100.0 * dose, 3, dose),
                (unit_id, 3, 101.0 * dose, 3, dose),
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


def _make_universal_base_period_eventstudy_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group in [2, 4]:
        for dose in [1.0, 2.0, 3.0]:
            for time_period in [1, 2, 3, 4, 5]:
                outcome = 0.0
                if time_period >= group:
                    outcome = float(time_period - group + 1) * dose
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    for _ in range(3):
        for time_period in [1, 2, 3, 4, 5]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _make_universal_baseline_notyettreated_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []

    for unit_id, dose in [("g3a", 1.0), ("g3b", 2.0)]:
        for time_period in [1, 2, 3, 4]:
            outcome = 100.0 if time_period >= 3 else 0.0
            rows.append((unit_id, time_period, outcome, 3, dose))

    for unit_id, dose in [("g4a", 1.0), ("g4b", 2.0)]:
        for time_period in [1, 2, 3, 4]:
            outcome = 10.0 * dose if time_period >= 4 else 0.0
            rows.append((unit_id, time_period, outcome, 4, dose))

    for unit_id in ["n1", "n2"]:
        for time_period in [1, 2, 3, 4]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _rename_panel_columns(panel: PanelData) -> PanelData:
    return PanelData(
        frame=panel.frame.rename(
            columns={
                panel.id_column: "unit_id",
                panel.time_column: "period",
                panel.outcome_column: "outcome",
                panel.group_column: "cohort",
                panel.dose_column: "dose",
            }
        ),
        id_column="unit_id",
        time_column="period",
        outcome_column="outcome",
        group_column="cohort",
        dose_column="dose",
    )


def _treated_share_weighted_mean(entry: dict[str, object]) -> float:
    cohort_estimates = entry["cohort_estimates"]
    return sum(
        item["estimate"] * item["aggregation_weight"] for item in cohort_estimates
    )


def _centered_mean_influence(values: dict[object, float]) -> dict[object, float]:
    sample_size = len(values)
    if sample_size <= 1:
        return {unit_id: 0.0 for unit_id in values}
    mean_value = float(np.mean(list(values.values())))
    scale = math.sqrt(sample_size / (sample_size - 1)) / sample_size
    return {
        unit_id: (float(value) - mean_value) * scale
        for unit_id, value in values.items()
    }


def _add_scaled_oracle_influence(
    target: dict[object, float],
    source: dict[object, float],
    *,
    scale: float,
) -> None:
    for unit_id, value in source.items():
        target[unit_id] = target.get(unit_id, 0.0) + scale * float(value)


def _level_oracle_entries_by_event_time(
    panel: PanelData,
    spec: ContDIDSpec,
) -> dict[int, list[dict[str, object]]]:
    import contdid.eventstudy as eventstudy_module
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        panel,
        control_group=spec.control_group,
        anticipation=spec.anticipation,
    )

    oracle_by_event_time: dict[int, list[dict[str, object]]] = {}
    for row in prepared.loc[prepared["support"], :].itertuples(index=False):
        local_panel = eventstudy_module._build_local_eventstudy_panel(
            panel,
            timing_group=int(row.timing_group),
            time_period=int(row.time_period),
            base_period=int(row.base_period),
            control_group=str(row.comparison_type),
        )
        treated_by_id, comparison_by_id = (
            eventstudy_module._local_level_delta_maps_by_id(local_panel)
        )
        if not eventstudy_module._local_level_has_inference_df(
            treated_by_id,
            comparison_by_id,
        ):
            continue
        cohort_influence: dict[object, float] = {}
        _add_scaled_oracle_influence(
            cohort_influence,
            _centered_mean_influence(treated_by_id),
            scale=1.0,
        )
        _add_scaled_oracle_influence(
            cohort_influence,
            _centered_mean_influence(comparison_by_id),
            scale=-1.0,
        )
        oracle_by_event_time.setdefault(int(row.event_time), []).append(
            {
                "timing_group": int(row.timing_group),
                "treated_count": int(row.treated_count),
                "influence": cohort_influence,
            }
        )
    return oracle_by_event_time


def _oracle_derivative_matrix(
    dose: np.ndarray,
    *,
    degree: int,
    knots: list[float],
) -> np.ndarray:
    from contdid.estimation import _build_derivative_matrix
    return _build_derivative_matrix(dose, degree, knots)


def _slope_summary_oracle_influence(fit: object) -> dict[object, float]:
    support = _oracle_derivative_matrix(
        fit.treated_dose,
        degree=fit.degree,
        knots=fit.knots,
    )
    curve = support @ fit.coefficients
    centered_curve = curve - float(np.mean(curve))
    loading = support.mean(axis=0)
    score = fit.design * fit.residual[:, None]
    bread = np.linalg.pinv((fit.design.T @ fit.design) / fit.treated_count)
    coefficient_influence = score @ bread @ loading
    influence = centered_curve + coefficient_influence
    return {
        unit_id: float(value)
        for unit_id, value in zip(fit.treated_unit_ids, influence.tolist())
    }


def _slope_oracle_entries_by_event_time(
    panel: PanelData,
    spec: ContDIDSpec,
    *,
    dvals: list[float],
    degree: int,
    num_knots: int = 0,
) -> dict[int, list[dict[str, object]]]:
    import contdid.eventstudy as eventstudy_module
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        panel,
        control_group=spec.control_group,
        anticipation=spec.anticipation,
    )
    local_spec = eventstudy_module._eventstudy_local_spec(
        spec,
        expected_target="slope",
    )
    oracle_by_event_time: dict[int, list[dict[str, object]]] = {}
    for row in prepared.loc[prepared["support"], :].itertuples(index=False):
        local_panel = eventstudy_module._build_local_eventstudy_panel(
            panel,
            timing_group=int(row.timing_group),
            time_period=int(row.time_period),
            base_period=int(row.base_period),
            control_group=str(row.comparison_type),
        )
        local_fit = eventstudy_module._fit_shared_dose_design(
            local_panel,
            local_spec,
            expected_target="slope",
            dvals=dvals,
            degree=degree,
            num_knots=num_knots,
            enforce_observed_support=False,
            require_inference_df=False,
            require_untreated_variance_df=False,
            require_public_dose_control_group=False,
            require_strict_explicit_grid=False,
            assume_valid_panel=True,
            knots=[],
        )
        if not eventstudy_module._local_fit_has_inference_df(
            local_fit,
            expected_target="slope",
        ):
            continue
        oracle_by_event_time.setdefault(int(row.event_time), []).append(
            {
                "timing_group": int(row.timing_group),
                "treated_count": int(row.treated_count),
                "influence": _centered_mean_influence(
                    _slope_summary_oracle_influence(local_fit)
                ),
            }
        )
    return oracle_by_event_time


def _aggregate_oracle_event_influence(
    oracle_entries: list[dict[str, object]],
) -> dict[object, float]:
    total_treated = sum(int(item["treated_count"]) for item in oracle_entries)
    aggregate_influence: dict[object, float] = {}
    for oracle_entry in oracle_entries:
        weight = int(oracle_entry["treated_count"]) / total_treated
        _add_scaled_oracle_influence(
            aggregate_influence,
            oracle_entry["influence"],
            scale=weight,
        )
    return aggregate_influence


def _oracle_eventstudy_covariance(
    event_times: list[int],
    oracle_by_event_time: dict[int, list[dict[str, object]]],
) -> np.ndarray:
    aggregate_by_event_time = {
        event_time: _aggregate_oracle_event_influence(oracle_by_event_time[event_time])
        for event_time in event_times
    }
    covariance = np.zeros((len(event_times), len(event_times)), dtype=float)
    for row_index, left_event_time in enumerate(event_times):
        left_influence = aggregate_by_event_time[left_event_time]
        for col_index, right_event_time in enumerate(event_times):
            right_influence = aggregate_by_event_time[right_event_time]
            covariance[row_index, col_index] = sum(
                left_value * right_influence.get(unit_id, 0.0)
                for unit_id, left_value in left_influence.items()
            )
    return covariance


def test_eventstudy_level_path_tracks_null_surface_near_zero() -> None:
    from contdid import estimate_eventstudy_effects

    fixture = _load_fixture()
    panel = simulate_contdid_data(
        n=16000, dgp_id="SIM-001-null-dose", seed=fixture["default_seed"]
    )

    result = estimate_eventstudy_effects(
        panel, _make_eventstudy_spec(target_parameter="level"), degree=1
    )

    assert result.estimand == "ATT(event_time)"
    assert result.grid == fixture["event_time_grid"]
    assert result.event_time == fixture["event_time_grid"]
    assert result.event_time_grid == fixture["event_time_grid"]
    assert max(abs(value) for value in result.estimate) < 0.08
    assert result.metadata["timing_group_support"]["timing_groups"] == [2, 3, 4]
    assert result.metadata["support"] == fixture["support"]
    assert result.metadata["inference"] == "bootstrap"
    assert result.critical_value is not None and result.critical_value > 0.0
    assert result.confidence_band is not None
    assert result.cohort_summary is not None
    assert len(result.cohort_summary) == len(fixture["event_time_grid"])


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_routes_respect_paneldata_column_overrides(
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    canonical_panel = simulate_contdid_data(
        n=500,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=20260407,
    )
    custom_panel = _rename_panel_columns(canonical_panel)
    spec = ContDIDSpec(
        target_parameter=target_parameter,
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
        alp=0.1,
        bstrap=False,
        cband=False,
        boot_type="multiplier",
        biters=199,
    )
    estimators = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }
    estimator = estimators[estimator_name]

    canonical = estimator(
        canonical_panel,
        spec,
        dvals=[0.25, 0.5, 0.75],
        degree=1,
    )
    aliased = estimator(
        custom_panel,
        spec,
        dvals=[0.25, 0.5, 0.75],
        degree=1,
    )

    assert aliased.event_time_grid == canonical.event_time_grid
    assert aliased.estimate == pytest.approx(canonical.estimate, abs=0.0)
    assert aliased.std_error == pytest.approx(canonical.std_error, abs=0.0)
    assert aliased.metadata["support"] == canonical.metadata["support"]
    assert (
        aliased.metadata["timing_group_support"]
        == canonical.metadata["timing_group_support"]
    )


def test_eventstudy_default_grid_remains_available_on_multi_period_panels() -> None:
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_unbalanced_eventstudy_panel(),
        _make_eventstudy_spec(target_parameter="level"),
        degree=1,
    )

    assert result.event_time == [-2, -1, 0, 1, 2]
    assert len(result.metadata["dose_grid"]) == 90
    assert result.metadata["dose_grid"][0] >= 1.0
    assert result.metadata["dose_grid"][-1] <= 2.5


def test_eventstudy_slope_path_preserves_positive_post_treatment_direction() -> None:
    from contdid import estimate_eventstudy_slope_effects

    panel = simulate_contdid_data(n=16000, dgp_id="SIM-002-linear-dose", seed=20260407)

    result = estimate_eventstudy_slope_effects(
        panel, _make_eventstudy_spec(target_parameter="slope"), degree=1
    )

    pre_period = [
        value
        for event_time, value in zip(result.event_time or [], result.estimate)
        if event_time < 0
    ]
    post_period = [
        value
        for event_time, value in zip(result.event_time or [], result.estimate)
        if event_time >= 0
    ]

    assert result.estimand == "ACRT(event_time)"
    assert result.event_time == [-2, -1, 0, 1, 2]
    assert max(abs(value) for value in pre_period) < 0.08
    assert all(value > 0.0 for value in post_period)
    assert result.metadata["event_time_grid"] == [-2, -1, 0, 1, 2]
    assert result.metadata["inference"] == "bootstrap"
    assert result.confidence_band is not None


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_nonnull_surfaces_do_not_claim_flat_zero_shape_constraints(
    target_parameter: str, estimator_name: str
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    panel = simulate_contdid_data(n=16000, dgp_id="SIM-002-linear-dose", seed=20260407)
    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        panel,
        _make_eventstudy_spec(target_parameter=target_parameter),
        degree=1,
    )

    assert max(abs(value) for value in result.estimate) > 0.05
    assert result.metadata["shape_constraints"] == {
        "level_curve": "no flat-zero shape restriction across event time",
        "slope_curve": "no flat-zero shape restriction across event time",
        "event_time_order": "ascending integers",
    }


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_metadata_carries_public_dose_est_method(
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
        _make_eventstudy_spec(target_parameter=target_parameter),
        dvals=[1.0],
        degree=1,
    )

    assert result.metadata["dose_est_method"] == "parametric"
    assert result.metadata["source_estimator"] == (
        "binary_eventstudy_mean"
        if target_parameter == "level"
        else "phase4_shared_dose_stack"
    )
    assert result.metadata["inference_covariance"] == "full_event_time_covariance"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"degree": 1.5}, "degree must be an integer"),
        ({"num_knots": 0.5}, "num_knots must be an integer"),
    ],
)
def test_eventstudy_level_validates_public_basis_controls(
    kwargs: dict[str, float],
    message: str,
) -> None:
    from contdid import estimate_eventstudy_effects

    with pytest.raises(ContDIDValidationError, match=message):
        estimate_eventstudy_effects(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter="level"),
            dvals=[1.0],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"degree": 0}, "degree must be at least 1 for dose estimation"),
        ({"num_knots": -1}, "num_knots must be nonnegative"),
    ],
)
def test_eventstudy_routes_reject_out_of_range_basis_controls(
    target_parameter: str,
    estimator_name: str,
    kwargs: dict[str, int],
    message: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(ContDIDValidationError, match=message):
        estimator(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter=target_parameter),
            dvals=[1.0],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_uses_treated_share_weights_across_timing_groups(
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
        _make_eventstudy_spec(target_parameter=target_parameter),
        dvals=[1.0],
        degree=1,
    )

    entry0 = next(entry for entry in result.cohort_summary if entry["event_time"] == 0)
    entry1 = next(entry for entry in result.cohort_summary if entry["event_time"] == 1)
    expected0 = _treated_share_weighted_mean(entry0)
    expected1 = _treated_share_weighted_mean(entry1)

    assert sum(
        cohort["aggregation_weight"] for cohort in entry0["cohort_estimates"]
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert sum(
        cohort["aggregation_weight"] for cohort in entry1["cohort_estimates"]
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert [
        cohort["aggregation_weight"] for cohort in entry0["cohort_estimates"]
    ] == pytest.approx(
        [
            cohort["treated_count"]
            / sum(item["treated_count"] for item in entry0["cohort_estimates"])
            for cohort in entry0["cohort_estimates"]
        ],
        abs=1e-12,
    )
    assert entry0["mean_estimate"] == pytest.approx(expected0, abs=1e-12)
    assert entry1["mean_estimate"] == pytest.approx(expected1, abs=1e-12)
    assert result.estimate[result.event_time.index(0)] == pytest.approx(
        expected0, abs=1e-12
    )
    assert result.estimate[result.event_time.index(1)] == pytest.approx(
        expected1, abs=1e-12
    )


def test_eventstudy_private_weighting_accepts_positive_integer_counts() -> None:
    import contdid.eventstudy as eventstudy_module

    first = {"treated_count": 2, "estimate": 1.0}
    second = {"treated_count": np.int64(3), "estimate": 2.0}

    weighted = eventstudy_module._weighted_entries([first, second])

    assert weighted == [
        (pytest.approx(0.4, abs=1e-12), first),
        (pytest.approx(0.6, abs=1e-12), second),
    ]
    assert eventstudy_module._weighted_entries([]) == []


@pytest.mark.parametrize(
    "treated_count",
    [0, -1, 1.5, True, np.bool_(True), None, "2", math.nan, math.inf],
)
def test_eventstudy_private_weighting_rejects_invalid_treated_counts(
    treated_count: object,
) -> None:
    import contdid.eventstudy as eventstudy_module

    with pytest.raises(
        ContDIDValidationError,
        match="event-study cohort treated_count must be a positive integer",
    ):
        eventstudy_module._weighted_entries([{"treated_count": treated_count}])


def test_eventstudy_private_weighting_rejects_missing_treated_count() -> None:
    import contdid.eventstudy as eventstudy_module

    with pytest.raises(
        ContDIDValidationError,
        match="event-study cohort treated_count must be a positive integer",
    ):
        eventstudy_module._weighted_entries([{"estimate": 1.0}])


def test_eventstudy_private_weighting_rejects_nonfinite_float_counts() -> None:
    import contdid.eventstudy as eventstudy_module

    with pytest.raises(
        ContDIDValidationError,
        match="event-study cohort treated_count must be a finite positive integer",
    ):
        eventstudy_module._weighted_entries(
            [{"treated_count": 2}, {"treated_count": 10**400}]
        )


def test_eventstudy_level_multicohort_standard_errors_match_influence_oracle() -> None:
    from contdid import estimate_eventstudy_effects

    panel = _make_unbalanced_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="level")
    result = estimate_eventstudy_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    oracle_by_event_time = _level_oracle_entries_by_event_time(panel, spec)

    checked_multi_cohort_cells = 0
    for entry in result.cohort_summary:
        event_time = int(entry["event_time"])
        oracle_entries = oracle_by_event_time[event_time]
        if len(oracle_entries) <= 1:
            continue
        checked_multi_cohort_cells += 1
        aggregate_influence = _aggregate_oracle_event_influence(oracle_entries)
        expected_se = math.sqrt(
            sum(value * value for value in aggregate_influence.values())
        )

        assert entry["std_error"] == pytest.approx(expected_se, abs=1e-12)
        assert result.std_error[result.event_time.index(event_time)] == pytest.approx(
            expected_se,
            abs=1e-12,
        )

    assert checked_multi_cohort_cells >= 2


def test_eventstudy_level_confidence_band_matches_cross_event_influence_oracle() -> (
    None
):
    from contdid import estimate_eventstudy_effects
    from contdid.inference import compute_multiplier_bootstrap

    panel = _make_unbalanced_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="level")
    result = estimate_eventstudy_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    event_times = [int(value) for value in result.event_time]
    oracle_by_event_time = _level_oracle_entries_by_event_time(panel, spec)
    oracle_covariance = _oracle_eventstudy_covariance(
        event_times,
        oracle_by_event_time,
    )
    oracle_bootstrap = compute_multiplier_bootstrap(
        np.eye(len(event_times), dtype=float),
        oracle_covariance,
        alp=spec.alp,
        bstrap=spec.bstrap,
        cband=spec.cband,
        boot_type=spec.boot_type,
        biters=spec.biters,
    )
    expected_std_error = oracle_bootstrap["std_error"]
    expected_critical_value = oracle_bootstrap["critical_value"]
    expected_lower = [
        estimate - expected_critical_value * se
        for estimate, se in zip(result.estimate, expected_std_error)
    ]
    expected_upper = [
        estimate + expected_critical_value * se
        for estimate, se in zip(result.estimate, expected_std_error)
    ]

    assert result.std_error == pytest.approx(expected_std_error, abs=1e-12)
    assert result.critical_value == pytest.approx(expected_critical_value, abs=1e-12)
    assert result.confidence_band["lower"] == pytest.approx(expected_lower, abs=1e-12)
    assert result.confidence_band["upper"] == pytest.approx(expected_upper, abs=1e-12)


def test_eventstudy_slope_multicohort_standard_errors_match_influence_oracle() -> None:
    from contdid import estimate_eventstudy_slope_effects

    panel = _make_slope_multicohort_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="slope")
    result = estimate_eventstudy_slope_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    oracle_by_event_time = _slope_oracle_entries_by_event_time(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )

    checked_multi_cohort_cells = 0
    for entry in result.cohort_summary:
        event_time = int(entry["event_time"])
        oracle_entries = oracle_by_event_time[event_time]
        if len(oracle_entries) <= 1:
            continue
        checked_multi_cohort_cells += 1
        aggregate_influence = _aggregate_oracle_event_influence(oracle_entries)
        expected_se = math.sqrt(
            sum(value * value for value in aggregate_influence.values())
        )

        assert entry["std_error"] == pytest.approx(expected_se, abs=1e-12)
        assert result.std_error[result.event_time.index(event_time)] == pytest.approx(
            expected_se,
            abs=1e-12,
        )

    assert checked_multi_cohort_cells >= 2


def test_eventstudy_slope_confidence_band_matches_cross_event_influence_oracle() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects
    from contdid.inference import compute_multiplier_bootstrap

    panel = _make_slope_multicohort_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="slope")
    result = estimate_eventstudy_slope_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    event_times = [int(value) for value in result.event_time]
    oracle_by_event_time = _slope_oracle_entries_by_event_time(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    oracle_covariance = _oracle_eventstudy_covariance(
        event_times,
        oracle_by_event_time,
    )
    oracle_bootstrap = compute_multiplier_bootstrap(
        np.eye(len(event_times), dtype=float),
        oracle_covariance,
        alp=spec.alp,
        bstrap=spec.bstrap,
        cband=spec.cband,
        boot_type=spec.boot_type,
        biters=spec.biters,
    )
    expected_std_error = oracle_bootstrap["std_error"]
    expected_critical_value = oracle_bootstrap["critical_value"]
    expected_lower = [
        estimate - expected_critical_value * se
        for estimate, se in zip(result.estimate, expected_std_error)
    ]
    expected_upper = [
        estimate + expected_critical_value * se
        for estimate, se in zip(result.estimate, expected_std_error)
    ]

    assert result.std_error == pytest.approx(expected_std_error, abs=1e-12)
    assert result.critical_value == pytest.approx(expected_critical_value, abs=1e-12)
    assert result.confidence_band["lower"] == pytest.approx(expected_lower, abs=1e-12)
    assert result.confidence_band["upper"] == pytest.approx(expected_upper, abs=1e-12)


def test_eventstudy_slope_uses_shared_global_knots_across_local_fits() -> None:
    from contdid import estimate_eventstudy_slope_effects

    result = estimate_eventstudy_slope_effects(
        _make_global_knot_eventstudy_panel(),
        _make_eventstudy_spec(
            target_parameter="slope",
            control_group="nevertreated",
        ),
        dvals=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        degree=1,
        num_knots=1,
    )

    assert result.metadata["basis"] == {
        "type": "bspline",
        "degree": 1,
        "num_knots": 1,
        "interior_knots": [4.0],
    }
    assert result.event_time == [-1, 0, 1]
    assert result.estimate == pytest.approx([0.0, 0.6, 0.8], abs=1e-12)
    assert result.cohort_summary[1]["cohort_estimates"][0]["estimate"] == pytest.approx(
        0.4, abs=1e-12
    )
    assert result.cohort_summary[1]["cohort_estimates"][1]["estimate"] == pytest.approx(
        0.8, abs=1e-12
    )


def test_eventstudy_slope_default_cubic_summary_matches_treated_support_derivative() -> (
    None
):
    from contdid import estimate_eventstudy_slope_effects

    treated_dose = np.asarray([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 7.0])
    expected_derivative_mean = float(
        np.mean(0.25 - 0.16 * treated_dose + 0.045 * treated_dose**2)
    )

    result = estimate_eventstudy_slope_effects(
        _make_cubic_slope_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            bstrap=False,
        ),
    )

    assert result.metadata["basis"] == {
        "type": "global_polynomial",
        "degree": 3,
        "num_knots": 0,
        "interior_knots": [],
    }
    assert result.event_time == [0, 1]
    assert result.estimate == pytest.approx(
        [expected_derivative_mean, 2.0 * expected_derivative_mean],
        abs=1e-12,
    )
    assert result.cohort_summary[0]["cohort_estimates"][0]["estimate"] == pytest.approx(
        expected_derivative_mean,
        abs=1e-12,
    )
    assert result.metadata["summary"]["overall_slope"] == pytest.approx(
        1.5 * expected_derivative_mean,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_supports_nevertreated_control_group_on_public_surface(
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
        _make_eventstudy_spec(
            target_parameter=target_parameter,
            control_group="nevertreated",
        ),
        dvals=[1.0],
        degree=1,
    )

    assert result.metadata["control_group"] == "nevertreated"
    if target_parameter == "level":
        assert result.event_time == [-2, -1, 0, 1, 2]
        assert result.metadata["timing_group_support"]["timing_groups"] == [2, 3, 4]
        assert result.metadata["support"] == [True, True, True, True, True]
    else:
        assert result.event_time == [0, 1, 2]
        assert result.metadata["timing_group_support"]["timing_groups"] == [2]
        assert result.metadata["support"] == [True, True, True]
    assert result.confidence_band is not None
    assert len(result.cohort_summary) == len(result.event_time)


@pytest.mark.parametrize(
    (
        "target_parameter",
        "estimator_name",
        "expected_event_time",
        "expected_estimate",
        "expected_overall",
    ),
    [
        (
            "level",
            "estimate_eventstudy_effects",
            [-2, -1, 0, 1, 2],
            [0.0, 0.0, 3.9, 5.433333333333333, 5.4],
            4.744444444444444,
        ),
        (
            "slope",
            "estimate_eventstudy_slope_effects",
            [0, 1, 2],
            [1.12, 2.24, 3.360000000000001],
            2.24,
        ),
    ],
)
def test_eventstudy_fixed_base_period_supports_notyettreated_control_group(
    target_parameter: str,
    estimator_name: str,
    expected_event_time: list[int],
    expected_estimate: list[float],
    expected_overall: float,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        _make_unbalanced_eventstudy_panel(),
        _make_eventstudy_spec(
            target_parameter=target_parameter,
            control_group="notyettreated",
        ),
        dvals=[1.0],
        degree=1,
        base_period=1,
    )
    summary_key = f"overall_{target_parameter}"

    assert result.metadata["control_group"] == "notyettreated"
    assert result.metadata["base_period"] == 1
    assert result.metadata["timing_group_support"]["base_period_strategy"] == "fixed"
    assert result.event_time == expected_event_time
    assert result.estimate == pytest.approx(expected_estimate, abs=1e-12)
    assert result.metadata["summary_aggregates"][summary_key] == pytest.approx(
        expected_overall, abs=1e-12
    )
    assert {
        int(cohort["base_period"])
        for event_time_entry in result.cohort_summary
        for cohort in event_time_entry["cohort_estimates"]
    } == {1}


@pytest.mark.parametrize(
    (
        "target_parameter",
        "estimator_name",
        "summary_key",
        "post_key",
        "expected_estimate",
        "expected_overall",
        "expected_post_mean",
    ),
    [
        (
            "level",
            "estimate_eventstudy_effects",
            "overall_level",
            "post_treatment_mean_level",
            [0.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            26.0 / 6.0,
            5.0,
        ),
        (
            "slope",
            "estimate_eventstudy_slope_effects",
            "overall_slope",
            "post_treatment_mean_slope",
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            13.0 / 6.0,
            2.5,
        ),
    ],
)
def test_eventstudy_universal_base_period_uses_cohort_specific_reference(
    target_parameter: str,
    estimator_name: str,
    summary_key: str,
    post_key: str,
    expected_estimate: list[float],
    expected_overall: float,
    expected_post_mean: float,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        _make_universal_base_period_eventstudy_panel(),
        ContDIDSpec(
            target_parameter=target_parameter,
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            treatment_type="continuous",
            anticipation=0,
            alp=0.1,
            bstrap=False,
            cband=False,
            boot_type="multiplier",
            biters=99,
        ),
        dvals=[1.0, 2.0, 3.0],
        degree=1,
        base_period="universal",
    )

    assert result.event_time == [-3, -2, 0, 1, 2, 3]
    assert -1 not in result.event_time
    assert result.estimate == pytest.approx(expected_estimate, abs=1e-12)
    assert result.metadata["base_period"] == "universal"
    assert (
        result.metadata["timing_group_support"]["base_period_strategy"] == "universal"
    )
    assert {
        (int(cohort["timing_group"]), int(cohort["base_period"]))
        for event_time_entry in result.cohort_summary
        for cohort in event_time_entry["cohort_estimates"]
    } == {(2, 1), (4, 3)}
    assert result.metadata["summary_aggregates"][summary_key] == pytest.approx(
        expected_overall, abs=1e-12
    )
    assert result.metadata["summary_aggregates"][post_key] == pytest.approx(
        expected_post_mean, abs=1e-12
    )


def test_eventstudy_universal_notyettreated_controls_stay_untreated_at_reference_endpoint() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_universal_baseline_notyettreated_panel(),
        _make_eventstudy_spec(
            target_parameter="level",
            control_group="notyettreated",
        ),
        dvals=[1.0, 2.0],
        degree=1,
        base_period="universal",
    )

    event_minus_two = next(
        entry for entry in result.cohort_summary if entry["event_time"] == -2
    )
    group_four = next(
        cohort
        for cohort in event_minus_two["cohort_estimates"]
        if cohort["timing_group"] == 4
    )

    assert group_four["time_period"] == 2
    assert group_four["base_period"] == 3
    assert group_four["comparison_count"] == 2
    assert group_four["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert event_minus_two["mean_estimate"] == pytest.approx(0.0, abs=1e-12)


def test_eventstudy_varying_base_period_string_matches_default() -> None:
    from contdid import estimate_eventstudy_effects

    panel = _make_unbalanced_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="level")

    default_result = estimate_eventstudy_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
    )
    varying_result = estimate_eventstudy_effects(
        panel,
        spec,
        dvals=[1.0],
        degree=1,
        base_period="varying",
    )

    assert varying_result.metadata["base_period"] is None
    assert (
        varying_result.metadata["timing_group_support"]["base_period_strategy"]
        == "varying_pre_period"
    )
    assert varying_result.event_time == default_result.event_time
    assert varying_result.estimate == pytest.approx(default_result.estimate, abs=1e-12)


def test_eventstudy_public_validation_errors_do_not_leak_internal_phase_labels() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    panel = _make_unbalanced_eventstudy_panel()

    with pytest.raises(
        ContDIDValidationError,
        match="event-study estimators require aggregation='eventstudy'",
    ) as aggregation_error:
        estimate_eventstudy_effects(
            panel,
            ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
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
            dvals=[1.0],
            degree=1,
        )

    assert "Phase 5" not in str(aggregation_error.value)

    with pytest.raises(
        ContDIDValidationError,
        match="control_group='eventuallytreated' is not supported",
    ) as control_group_error:
        estimate_eventstudy_effects(
            panel,
            _make_eventstudy_spec(
                target_parameter="level",
                control_group="eventuallytreated",
            ),
            dvals=[1.0],
            degree=1,
        )

    assert "Phase 5" not in str(control_group_error.value)


def test_eventstudy_level_estimate_is_invariant_to_explicit_dose_grid() -> None:
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

    assert high_grid.event_time == low_grid.event_time
    assert high_grid.estimate == pytest.approx(low_grid.estimate, abs=1e-12)
    assert low_grid.metadata["dose_grid"] == [1.0]
    assert high_grid.metadata["dose_grid"] == [2.0]
    assert low_grid.metadata["summary"] == low_grid.metadata["summary_aggregates"]
    assert high_grid.metadata["summary"] == high_grid.metadata["summary_aggregates"]
    assert high_grid.metadata["summary_aggregates"] == pytest.approx(
        low_grid.metadata["summary_aggregates"],
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name", "overall_key", "post_key", "expected"),
    [
        (
            "level",
            "estimate_eventstudy_effects",
            "overall_level",
            "post_treatment_mean_level",
            2.5,
        ),
        (
            "slope",
            "estimate_eventstudy_slope_effects",
            "overall_slope",
            "post_treatment_mean_slope",
            1.0,
        ),
    ],
)
def test_eventstudy_overall_summary_excludes_pretrend_cells(
    target_parameter: str,
    estimator_name: str,
    overall_key: str,
    post_key: str,
    expected: float,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]
    result = estimator(
        _make_pretrend_contaminated_eventstudy_panel(),
        _make_eventstudy_spec(
            target_parameter=target_parameter,
            control_group="nevertreated",
        ),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=1,
    )

    assert result.event_time == [-1, 0]
    assert result.estimate[0] > 90.0
    assert result.metadata["summary_aggregates"][overall_key] == pytest.approx(
        expected,
        abs=1e-12,
    )
    assert result.metadata["summary_aggregates"][post_key] == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_eventstudy_slope_reuses_single_local_fit_per_supported_event_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contdid.estimation as estimation_module
    import contdid.eventstudy as eventstudy_module
    from contdid import estimate_eventstudy_slope_effects
    from contdid.timing import prepare_timing_groups

    panel = _make_unbalanced_eventstudy_panel()
    prepared = prepare_timing_groups(
        panel, control_group="notyettreated", anticipation=0
    )
    supported_rows = int(prepared["support"].sum())

    original_fit = estimation_module._fit_shared_dose_design
    fit_calls = 0

    def _counting_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(estimation_module, "_fit_shared_dose_design", _counting_fit)
    monkeypatch.setattr(eventstudy_module, "_fit_shared_dose_design", _counting_fit)

    estimate_eventstudy_slope_effects(
        panel,
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[1.0],
        degree=1,
    )

    assert fit_calls == supported_rows


def test_eventstudy_level_reuses_single_delta_collapse_per_supported_event_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contdid.eventstudy as eventstudy_module
    from contdid import estimate_eventstudy_effects
    from contdid.timing import prepare_timing_groups

    panel = _make_unbalanced_eventstudy_panel()
    prepared = prepare_timing_groups(
        panel, control_group="notyettreated", anticipation=0
    )
    supported_rows = int(prepared["support"].sum())
    original_collapse = eventstudy_module._collapse_to_unit_differences
    collapse_calls = 0

    def _counting_collapse(*args, **kwargs):
        nonlocal collapse_calls
        collapse_calls += 1
        return original_collapse(*args, **kwargs)

    monkeypatch.setattr(
        eventstudy_module,
        "_collapse_to_unit_differences",
        _counting_collapse,
    )

    estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[1.0],
        degree=1,
    )

    # The global positive-dose support scan is reused for both the default
    # evaluation grid and knots; each local level row then reuses one
    # treated/comparison delta collapse instead of doing two.
    assert supported_rows > 1
    assert collapse_calls == supported_rows + 1


def test_eventstudy_local_panel_builder_avoids_full_panel_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contdid.eventstudy as eventstudy_module

    panel = _make_unbalanced_eventstudy_panel()
    original_copy = pd.DataFrame.copy
    full_panel_copy_calls = 0

    def _counting_copy(self, *args, **kwargs):
        nonlocal full_panel_copy_calls
        if self.shape == panel.frame.shape:
            full_panel_copy_calls += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    local_panel = eventstudy_module._build_local_eventstudy_panel(
        panel,
        timing_group=2,
        time_period=3,
        base_period=1,
        control_group="notyettreated",
    )

    assert set(local_panel.frame["G"].unique()) == {0, 2}
    assert set(local_panel.frame["time_period"].unique()) == {1, 2}
    assert full_panel_copy_calls == 0


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_local_fits_reuse_validated_local_panels(
    monkeypatch: pytest.MonkeyPatch,
    target_parameter: str,
    estimator_name: str,
) -> None:
    import contdid.estimation as estimation_module
    import contdid.eventstudy as eventstudy_module
    import contdid.timing as timing_module
    import contdid.validation as validation_module
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects
    from contdid.timing import prepare_timing_groups

    panel = _make_unbalanced_eventstudy_panel()
    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]
    supported_rows = int(
        prepare_timing_groups(panel, control_group="notyettreated", anticipation=0)[
            "support"
        ].sum()
    )
    original_validate = validation_module.validate_panel_data
    validation_calls = 0

    def _counting_validate(panel_arg):
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(panel_arg)

    monkeypatch.setattr(estimation_module, "validate_panel_data", _counting_validate)
    monkeypatch.setattr(eventstudy_module, "validate_panel_data", _counting_validate)
    monkeypatch.setattr(timing_module, "validate_panel_data", _counting_validate)
    monkeypatch.setattr(validation_module, "validate_panel_data", _counting_validate)

    estimator(
        panel,
        _make_eventstudy_spec(target_parameter=target_parameter),
        dvals=[1.0],
        degree=1,
    )

    assert supported_rows > 1
    assert validation_calls == 1


def test_eventstudy_level_summary_contract_uses_binary_mean_difference() -> None:
    import contdid.eventstudy as eventstudy_module
    from contdid import estimate_eventstudy_effects
    from contdid.timing import prepare_timing_groups

    panel = _make_unbalanced_eventstudy_panel()
    spec = _make_eventstudy_spec(target_parameter="level")
    result = estimate_eventstudy_effects(panel, spec, dvals=[1.0], degree=1)
    prepared = prepare_timing_groups(
        panel, control_group="notyettreated", anticipation=0
    )

    cohort_entries = {
        (entry["event_time"], cohort["timing_group"]): cohort
        for entry in result.cohort_summary
        for cohort in entry["cohort_estimates"]
    }

    for row in prepared.loc[prepared["support"], :].itertuples(index=False):
        local_panel = eventstudy_module._build_local_eventstudy_panel(
            panel,
            timing_group=int(row.timing_group),
            time_period=int(row.time_period),
            base_period=int(row.base_period),
            control_group=str(row.comparison_type),
        )
        treated_by_id = eventstudy_module._local_treated_delta_by_id(local_panel)
        comparison_by_id = eventstudy_module._local_untreated_delta_by_id(local_panel)
        if not eventstudy_module._local_level_has_inference_df(
            treated_by_id,
            comparison_by_id,
        ):
            assert (int(row.event_time), int(row.timing_group)) not in cohort_entries
            continue
        expected_estimate, expected_se = eventstudy_module._local_level_summary(
            treated_by_id,
            comparison_by_id,
        )
        cohort = cohort_entries[(int(row.event_time), int(row.timing_group))]

        assert cohort["estimate"] == pytest.approx(expected_estimate, abs=1e-12)
        assert cohort["std_error"] == pytest.approx(expected_se, abs=1e-12)


def test_eventstudy_slope_matches_closed_form_linear_duration_path() -> None:
    from contdid import estimate_eventstudy_slope_effects

    result = estimate_eventstudy_slope_effects(
        _make_linear_slope_eventstudy_panel(),
        _make_eventstudy_spec(target_parameter="slope"),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=1,
        base_period=1,
    )

    assert result.event_time_grid == [0, 1]
    assert result.metadata["support"] == [True, True]
    assert result.metadata["timing_group_support"]["timing_groups"] == [2]
    assert result.estimate == pytest.approx([2.0, 4.0], abs=1e-12)
    assert result.std_error == pytest.approx([0.0, 0.0], abs=1e-12)
    assert result.metadata["summary_aggregates"] == {
        "overall_slope": pytest.approx(3.0, abs=1e-12),
        "post_treatment_mean_slope": pytest.approx(3.0, abs=1e-12),
    }
    assert [
        entry["cohort_estimates"][0]["estimate"] for entry in result.cohort_summary
    ] == pytest.approx([2.0, 4.0], abs=1e-12)


def test_eventstudy_slope_matches_staggered_closed_form_linear_duration_path() -> None:
    from contdid import estimate_eventstudy_slope_effects

    result = estimate_eventstudy_slope_effects(
        _make_staggered_linear_slope_eventstudy_panel(),
        _make_eventstudy_spec(target_parameter="slope", control_group="nevertreated"),
        dvals=[1.0, 2.0, 3.0, 4.0],
        degree=1,
        base_period=1,
    )

    assert result.event_time_grid == [-1, 0, 1, 2]
    assert result.metadata["support"] == [True, True, True, True]
    assert result.metadata["timing_group_support"]["timing_groups"] == [2, 3]
    assert result.estimate == pytest.approx([0.0, 2.0, 4.0, 6.0], abs=1e-12)
    assert result.std_error == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert result.metadata["summary_aggregates"] == {
        "overall_slope": pytest.approx(3.6, abs=1e-12),
        "post_treatment_mean_slope": pytest.approx(4.0, abs=1e-12),
    }

    by_event_time = {entry["event_time"]: entry for entry in result.cohort_summary}
    assert by_event_time[-1]["cohort_estimates"][0]["timing_group"] == 3
    assert by_event_time[-1]["cohort_estimates"][0]["aggregation_weight"] == (
        pytest.approx(1.0, abs=1e-12)
    )
    assert by_event_time[-1]["cohort_estimates"][0]["estimate"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert [
        cohort["aggregation_weight"]
        for cohort in by_event_time[0]["cohort_estimates"]
    ] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert [
        cohort["estimate"] for cohort in by_event_time[0]["cohort_estimates"]
    ] == pytest.approx([2.0, 2.0], abs=1e-12)
    assert [
        cohort["aggregation_weight"]
        for cohort in by_event_time[1]["cohort_estimates"]
    ] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert [
        cohort["estimate"] for cohort in by_event_time[1]["cohort_estimates"]
    ] == pytest.approx([4.0, 4.0], abs=1e-12)
    assert by_event_time[2]["cohort_estimates"][0]["timing_group"] == 2
    assert by_event_time[2]["cohort_estimates"][0]["aggregation_weight"] == (
        pytest.approx(1.0, abs=1e-12)
    )
    assert by_event_time[2]["cohort_estimates"][0]["estimate"] == pytest.approx(
        6.0, abs=1e-12
    )


def test_eventstudy_slope_rejects_post_treatment_rows_without_local_inference_df() -> (
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


def test_eventstudy_drops_notyettreated_self_only_pre_rows() -> None:
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_notyettreated_self_only_pre_period_panel(),
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[1.0, 2.0],
        degree=1,
    )

    assert result.event_time == [0]
    assert result.event_time_grid == [0]
    assert result.metadata["timing_group_support"]["timing_groups"] == [2]
    assert result.metadata["support"] == [True]
    assert result.cohort_summary == [
        {
            "event_time": 0,
            "timing_groups": [2],
            "cohort_estimates": [
                {
                    "timing_group": 2,
                    "time_period": 2,
                    "base_period": 1,
                    "comparison_count": 2,
                    "treated_count": 3,
                    "estimate": pytest.approx(2.0, abs=1e-12),
                    "std_error": pytest.approx(1.0 / math.sqrt(3.0), abs=1e-12),
                    "aggregation_weight": pytest.approx(1.0, abs=1e-12),
                }
            ],
            "mean_estimate": pytest.approx(2.0, abs=1e-12),
            "std_error": pytest.approx(1.0 / math.sqrt(3.0), abs=1e-12),
            "support": True,
        }
    ]


def test_eventstudy_uses_future_timing_as_notyettreated_comparison_only_support() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_future_comparison_eventstudy_panel(),
        _make_eventstudy_spec(target_parameter="level"),
        dvals=[1.0, 2.0],
        degree=1,
    )

    assert result.event_time == [0, 1]
    assert result.metadata["timing_group_support"]["timing_groups"] == [2]
    assert result.metadata["support"] == [True, True]
    assert result.cohort_summary[0]["cohort_estimates"][0]["comparison_count"] == 3
    assert result.cohort_summary[1]["cohort_estimates"][0]["comparison_count"] == 3
    assert result.estimate == pytest.approx([14.333333333333334, 28.666666666666668])


def test_eventstudy_dose_grid_excludes_future_comparison_only_doses() -> None:
    from contdid import estimate_eventstudy_effects

    panel = _make_future_comparison_eventstudy_panel()

    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level"),
        degree=1,
    )

    assert result.metadata["timing_group_support"]["timing_groups"] == [2]
    assert result.metadata["dose_grid"][0] == pytest.approx(1.1, abs=1e-12)
    assert result.metadata["dose_grid"][-1] == pytest.approx(1.99, abs=1e-12)
    assert max(result.metadata["dose_grid"]) < 2.5

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must stay within the observed positive-dose treated support",
    ):
        estimate_eventstudy_effects(
            panel,
            _make_eventstudy_spec(target_parameter="level"),
            dvals=[2.5],
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_dose_grid_excludes_first_period_treated_cohorts_without_baseline(
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    panel = _make_first_period_and_identified_later_cohort_eventstudy_panel()
    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        panel,
        _make_eventstudy_spec(
            target_parameter=target_parameter,
            control_group="nevertreated",
        ),
        degree=1,
    )

    assert result.metadata["timing_group_support"]["timing_groups"] == [2]
    assert result.event_time == [0, 1]
    assert result.metadata["dose_grid"][0] == pytest.approx(1.2, abs=1e-12)
    assert result.metadata["dose_grid"][-1] == pytest.approx(2.98, abs=1e-12)
    assert max(result.metadata["dose_grid"]) < 100.0

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must stay within the observed positive-dose treated support",
    ):
        estimator(
            panel,
            _make_eventstudy_spec(
                target_parameter=target_parameter,
                control_group="nevertreated",
            ),
            dvals=[100.0],
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_dose_grid_excludes_cohorts_dropped_by_local_inference_df(
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    panel = _make_dropped_high_dose_cohort_eventstudy_panel()
    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    result = estimator(
        panel,
        _make_eventstudy_spec(
            target_parameter=target_parameter,
            control_group="nevertreated",
        ),
        degree=1,
    )

    assert result.metadata["timing_group_support"]["timing_groups"] == [3]
    assert result.metadata["dose_grid"][0] == pytest.approx(1.1, abs=1e-12)
    assert result.metadata["dose_grid"][-1] == pytest.approx(1.99, abs=1e-12)
    assert max(result.metadata["dose_grid"]) < 100.0

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must stay within the observed positive-dose treated support",
    ):
        estimator(
            panel,
            _make_eventstudy_spec(
                target_parameter=target_parameter,
                control_group="nevertreated",
            ),
            dvals=[100.0],
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_routes_accept_scalar_explicit_dvals_as_length_one(
    target_parameter: str, estimator_name: str
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    panel = _make_unbalanced_eventstudy_panel()
    result = estimator(
        panel,
        _make_eventstudy_spec(target_parameter=target_parameter),
        dvals=1.5,
        degree=1,
    )

    assert result.metadata["dose_grid"] == pytest.approx([1.5])
    assert len(result.metadata["dose_grid"]) == 1


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
@pytest.mark.parametrize(
    "dvals",
    [
        "1.5",
        ["1.0", "2.0"],
    ],
)
def test_eventstudy_routes_reject_string_valued_explicit_dvals(
    dvals: object,
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimator(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter=target_parameter),
            dvals=dvals,
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
@pytest.mark.parametrize(
    "dvals",
    [
        True,
        np.bool_(True),
        [True, 1.5, 2.0],
        [np.bool_(True), 1.5, 2.0],
    ],
)
def test_eventstudy_routes_reject_boolean_explicit_dvals(
    dvals: object,
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(
        ContDIDValidationError,
        match="dose grid must contain only finite non-boolean numeric values",
    ):
        estimator(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter=target_parameter),
            dvals=dvals,
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
@pytest.mark.parametrize(
    "explicit_grid",
    [
        [1.0, 1.0, 2.0],
        [2.0, 1.0],
    ],
)
def test_eventstudy_routes_reject_non_strict_explicit_dvals(
    explicit_grid: list[float],
    target_parameter: str,
    estimator_name: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(
        ContDIDValidationError,
        match="explicit dvals must be strictly increasing with no duplicate dose values",
    ):
        estimator(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter=target_parameter),
            dvals=explicit_grid,
            degree=1,
        )


def test_eventstudy_slope_notyettreated_route_normalizes_internal_dose_control_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contdid.eventstudy as eventstudy_module
    from contdid import estimate_eventstudy_slope_effects

    observed_control_groups: list[str] = []

    def _stop_after_first_local_fit(*args, **kwargs):
        local_spec = kwargs.get("spec", args[1])
        observed_control_groups.append(local_spec.control_group)
        raise RuntimeError("captured local fit")

    monkeypatch.setattr(
        eventstudy_module,
        "_fit_shared_dose_design",
        _stop_after_first_local_fit,
    )

    with pytest.raises(RuntimeError, match="captured local fit"):
        estimate_eventstudy_slope_effects(
            _make_unbalanced_eventstudy_panel(),
            _make_eventstudy_spec(
                target_parameter="slope",
                control_group="notyettreated",
            ),
            dvals=[1.0],
            degree=1,
        )

    assert observed_control_groups == ["nevertreated"]


def test_eventstudy_accepts_cck_fixed_dimension() -> None:
    """CCK with fixed dimension is now supported in event study."""
    from contdid import estimate_eventstudy_effects

    panel = simulate_contdid_data(n=4000, dgp_id="SIM-005-cck-two-period")

    # CCK with fixed dimension should succeed (two-period panel is ideal for CCK)
    result = estimate_eventstudy_effects(
        panel,
        _make_eventstudy_spec(target_parameter="level", dose_est_method="cck"),
        degree=2,
        num_knots=0,
    )
    assert result.estimand == "ATT(event_time)"
    assert result.metadata["dose_est_method"] == "cck"


def test_eventstudy_cck_boundary_rejects_invalid_control_group() -> None:
    from contdid import estimate_eventstudy_effects

    panel = simulate_contdid_data(n=600, dgp_id="SIM-005-cck-two-period")

    with pytest.raises(
        ContDIDValidationError,
        match="eventuallytreated.*not supported",
    ):
        estimate_eventstudy_effects(
            panel,
            _make_eventstudy_spec(
                target_parameter="level",
                dose_est_method="cck",
                control_group="eventuallytreated",
            ),
            base_period="universal",
        )


def test_eventstudy_cck_staggered_rejects_invalid_control_group() -> (
    None
):
    from contdid import estimate_eventstudy_effects

    panel = simulate_contdid_data(n=600, dgp_id="SIM-004-staggered-eventstudy-null")

    with pytest.raises(
        ContDIDValidationError,
        match="eventuallytreated.*not supported",
    ):
        estimate_eventstudy_effects(
            panel,
            _make_eventstudy_spec(
                target_parameter="level",
                dose_est_method="cck",
                control_group="eventuallytreated",
            ),
            base_period="universal",
        )


def test_eventstudy_cck_small_sample_raises_appropriate_error() -> (
    None
):
    """CCK + eventstudy with tiny panel should fail with sample-size error."""
    from contdid import estimate_eventstudy_effects

    with pytest.raises(
        ContDIDValidationError,
    ):
        estimate_eventstudy_effects(
            _make_single_cohort_multiperiod_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter="level", dose_est_method="cck"),
        )


def test_eventstudy_accepts_nonzero_anticipation() -> (
    None
):
    """Anticipation > 0 is now supported (CGBS Assumption 3-MP(a))."""
    from contdid import estimate_eventstudy_effects

    result = estimate_eventstudy_effects(
        _make_anticipation_eventstudy_panel(),
        ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            treatment_type="continuous",
            anticipation=1,
            alp=0.1,
            bstrap=True,
            cband=True,
            boot_type="multiplier",
            biters=199,
        ),
        dvals=[1.0],
        degree=1,
    )
    assert result.estimand == "ATT(event_time)"
    assert len(result.estimate) > 0


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name"),
    [
        ("level", "estimate_eventstudy_effects"),
        ("slope", "estimate_eventstudy_slope_effects"),
    ],
)
def test_eventstudy_rejects_future_timing_when_no_treated_cohort_is_observed(
    target_parameter: str, estimator_name: str
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(
        ContDIDValidationError,
        match="timing-group preparation requires at least one treated cohort observed during the panel",
    ):
        estimator(
            _make_future_treated_only_eventstudy_panel(),
            _make_eventstudy_spec(target_parameter=target_parameter),
            dvals=[1.5, 2.5],
            degree=1,
        )


@pytest.mark.parametrize(
    ("target_parameter", "estimator_name", "expected_message"),
    [
        (
            "level",
            "estimate_eventstudy_effects",
            "positive-dose support and inference degrees of freedom",
        ),
        (
            "slope",
            "estimate_eventstudy_slope_effects",
            "found no locally identified positive-dose support",
        ),
    ],
)
@pytest.mark.parametrize("control_group", ["notyettreated", "nevertreated"])
def test_eventstudy_requires_locally_identified_post_treatment_support(
    target_parameter: str,
    estimator_name: str,
    expected_message: str,
    control_group: str,
) -> None:
    from contdid import estimate_eventstudy_effects, estimate_eventstudy_slope_effects

    estimator = {
        "estimate_eventstudy_effects": estimate_eventstudy_effects,
        "estimate_eventstudy_slope_effects": estimate_eventstudy_slope_effects,
    }[estimator_name]

    with pytest.raises(
        ContDIDValidationError,
        match=expected_message,
    ):
        estimator(
            _make_preonly_identified_eventstudy_panel(),
            _make_eventstudy_spec(
                target_parameter=target_parameter,
                control_group=control_group,
            ),
            degree=1,
        )
