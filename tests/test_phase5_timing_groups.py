from __future__ import annotations

import pandas as pd
import pytest

from contdid import PanelData, simulate_contdid_data
from contdid.validation import ContDIDValidationError


def _small_staggered_panel() -> PanelData:
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time_period": [1, 2, 1, 2, 1, 2, 1, 2],
            "Y": [1.0, 2.0, 1.5, 2.5, 1.2, 2.1, 0.8, 1.1],
            "G": [2, 2, 2, 2, 0, 0, 0, 0],
            "D": [0.4, 0.4, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return PanelData(frame=frame)


def _anticipation_panel() -> PanelData:
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


def _notyettreated_self_only_pre_period_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1
    for group, dose in [(2, 1.0), (2, 2.0), (3, 1.0), (3, 2.0)]:
        for time_period in [1, 2, 3]:
            outcome = 0.0
            if time_period >= group:
                outcome = float(time_period - group + 1) * dose
            rows.append((unit_id, time_period, outcome, group, dose))
        unit_id += 1
    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _future_comparison_panel() -> PanelData:
    rows: list[tuple[str, int, float, int, float]] = []

    for unit_index, dose in enumerate([1.0, 2.0], start=1):
        for time_period in [1, 2, 3]:
            rows.append(
                (
                    f"g2_{unit_index}",
                    time_period,
                    float(time_period >= 2) * dose,
                    2,
                    dose,
                )
            )

    for unit_index, dose in enumerate([1.5, 2.5], start=1):
        for time_period in [1, 2, 3]:
            rows.append((f"g4_{unit_index}", time_period, 0.0, 4, dose))

    rows.extend(
        [
            ("u1", 1, 0.0, 0, 0.0),
            ("u1", 2, 0.0, 0, 0.0),
            ("u1", 3, 0.0, 0, 0.0),
        ]
    )

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _first_period_and_identified_later_cohort_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for group, doses in [(1, [1.0, 2.0]), (2, [1.0, 2.0])]:
        for dose in doses:
            for time_period in [1, 2, 3]:
                outcome = 0.0
                if time_period >= group:
                    outcome = float(time_period - group + 1) * dose
                rows.append((unit_id, time_period, outcome, group, dose))
            unit_id += 1

    for _ in range(2):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _only_first_period_treated_panel() -> PanelData:
    rows: list[tuple[int, int, float, int, float]] = []
    unit_id = 1

    for dose in [1.0, 2.0]:
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, float(time_period) * dose, 1, dose))
        unit_id += 1

    for _ in range(2):
        for time_period in [1, 2, 3]:
            rows.append((unit_id, time_period, 0.0, 0, 0.0))
        unit_id += 1

    return PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )


def _universal_baseline_notyettreated_panel() -> PanelData:
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


def test_prepare_timing_groups_builds_phase5_event_time_surface() -> None:
    from contdid.timing import build_event_time_index, prepare_timing_groups

    panel = simulate_contdid_data(n=1200, dgp_id="SIM-001-null-dose", seed=20260407)

    prepared = prepare_timing_groups(panel, control_group="notyettreated")

    assert prepared["timing_group"].drop_duplicates().tolist() == [2, 3, 4]
    assert build_event_time_index(prepared) == [-2, -1, 0, 1, 2]

    row = prepared.loc[
        (prepared["timing_group"] == 4) & (prepared["event_time"] == -2)
    ].iloc[0]
    assert row["base_period"] == 1
    assert row["comparison_type"] == "notyettreated"
    assert bool(row["support"]) is True
    assert row["comparison_count"] > 0


def test_prepare_timing_groups_excludes_focal_group_from_notyettreated_support() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _notyettreated_self_only_pre_period_panel(),
        control_group="notyettreated",
    )

    self_only_pre = prepared.loc[
        (prepared["timing_group"] == 3) & (prepared["event_time"] == -1)
    ].iloc[0]
    identified_post = prepared.loc[
        (prepared["timing_group"] == 2) & (prepared["event_time"] == 0)
    ].iloc[0]

    assert self_only_pre["comparison_count"] == 0
    assert bool(self_only_pre["support"]) is False
    assert identified_post["comparison_count"] == 2
    assert bool(identified_post["support"]) is True


def test_prepare_timing_groups_uses_future_timing_as_comparison_only_support() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _future_comparison_panel(),
        control_group="notyettreated",
    )

    assert prepared["timing_group"].drop_duplicates().tolist() == [2]
    post_rows = prepared.loc[prepared["post_treatment"]].sort_values("event_time")

    assert post_rows["event_time"].tolist() == [0, 1]
    assert post_rows["comparison_count"].tolist() == [3, 3]
    assert post_rows["support"].tolist() == [True, True]


def test_prepare_timing_groups_omits_first_period_treated_cohorts_with_no_baseline() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _first_period_and_identified_later_cohort_panel(),
        control_group="nevertreated",
    )

    assert prepared["timing_group"].drop_duplicates().tolist() == [2]
    assert prepared["event_time"].tolist() == [0, 1]
    assert prepared["base_period"].tolist() == [1, 1]


def test_prepare_timing_groups_omits_fixed_baseline_cohorts_not_preceded_by_baseline() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _first_period_and_identified_later_cohort_panel(),
        control_group="nevertreated",
        base_period=1,
    )

    assert prepared["timing_group"].drop_duplicates().tolist() == [2]
    assert prepared["event_time"].tolist() == [0, 1]
    assert prepared["base_period"].tolist() == [1, 1]


def test_prepare_timing_groups_hard_fails_when_no_treated_cohort_has_a_baseline() -> None:
    from contdid.timing import prepare_timing_groups

    with pytest.raises(
        ContDIDValidationError,
        match="timing-group preparation found no treated cohorts with an admissible base period",
    ):
        prepare_timing_groups(
            _only_first_period_treated_panel(),
            control_group="nevertreated",
        )


def test_prepare_timing_groups_allows_fixed_base_period_with_notyettreated() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _small_staggered_panel(),
        control_group="notyettreated",
        base_period=1,
    )

    assert prepared["timing_group"].tolist() == [2]
    assert prepared["time_period"].tolist() == [2]
    assert prepared["base_period"].tolist() == [1]
    assert prepared["event_time"].tolist() == [0]
    assert prepared["comparison_type"].tolist() == ["notyettreated"]
    assert prepared["comparison_count"].tolist() == [2]
    assert prepared["support"].tolist() == [True]


def test_prepare_timing_groups_reuses_validated_panel_without_full_frame_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contdid.timing import prepare_timing_groups

    panel = _anticipation_panel()
    original_copy = pd.DataFrame.copy
    full_panel_copy_calls = 0

    def _counting_copy(self, *args, **kwargs):
        nonlocal full_panel_copy_calls
        if self.shape == panel.frame.shape:
            full_panel_copy_calls += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _counting_copy)

    prepared = prepare_timing_groups(
        panel,
        control_group="notyettreated",
        assume_valid_panel=True,
    )

    assert prepared["support"].sum() > 1
    assert full_panel_copy_calls == 0


def test_prepare_timing_groups_uses_long_difference_endpoint_for_notyettreated_support() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(
        _universal_baseline_notyettreated_panel(),
        control_group="notyettreated",
        base_period="universal",
    )

    row = prepared.loc[
        (prepared["timing_group"] == 4) & (prepared["event_time"] == -2)
    ].iloc[0]

    assert row["time_period"] == 2
    assert row["base_period"] == 3
    assert row["comparison_count"] == 2
    assert bool(row["support"]) is True


def test_prepare_timing_groups_marks_never_treated_support_explicitly() -> None:
    from contdid.timing import prepare_timing_groups

    prepared = prepare_timing_groups(_small_staggered_panel(), control_group="nevertreated")

    post_row = prepared.loc[(prepared["timing_group"] == 2) & (prepared["event_time"] == 0)].iloc[0]
    assert post_row["comparison_type"] == "nevertreated"
    assert bool(post_row["support"]) is True
    assert post_row["comparison_count"] == 2


def test_prepare_timing_groups_rejects_missing_never_treated_support() -> None:
    from contdid.timing import prepare_timing_groups

    frame = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time_period": [1, 2, 1, 2],
            "Y": [1.0, 2.0, 1.5, 2.5],
            "G": [2, 2, 2, 2],
            "D": [0.4, 0.4, 0.7, 0.7],
        }
    )
    panel = PanelData(frame=frame)

    with pytest.raises(ContDIDValidationError, match="never-treated comparison units"):
        prepare_timing_groups(panel, control_group="nevertreated")


def test_prepare_timing_groups_accepts_nonzero_anticipation() -> None:
    """anticipation > 0 is now supported (CGBS Assumption 3-MP(a))."""
    from contdid.timing import prepare_timing_groups

    result = prepare_timing_groups(
        _anticipation_panel(),
        control_group="notyettreated",
        anticipation=1,
    )
    assert not result.empty
    # With anticipation=1, base period for group 3 is 3-1-1=1
    # and for group 4 is 4-1-1=2
    group3_rows = result[result["timing_group"] == 3]
    assert all(group3_rows["base_period"].isin([1]))  # universal base is period 1


def test_prepare_timing_groups_rejects_group_timing_off_the_observed_integer_grid() -> None:
    from contdid.timing import prepare_timing_groups

    panel = _small_staggered_panel()
    panel.frame["G"] = panel.frame["G"].astype(float)
    panel.frame.loc[panel.frame["G"] > 0, "G"] = 1.5

    with pytest.raises(
        ContDIDValidationError,
        match="group timing values must align with or follow the observed integer time-period grid",
    ):
        prepare_timing_groups(panel, control_group="notyettreated")


def test_prepare_timing_groups_uses_phase_neutral_control_group_error() -> None:
    from contdid.timing import prepare_timing_groups

    with pytest.raises(
        ContDIDValidationError,
        match=(
            "timing-group preparation supports control_group values "
            "'notyettreated' and 'nevertreated' only"
        ),
    ) as error:
        prepare_timing_groups(
            _small_staggered_panel(),
            control_group="eventuallytreated",
        )

    assert "Phase 5" not in str(error.value)
