from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest


def _make_valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time_period": [1, 2, 1, 2, 1, 2],
            "Y": [0.1, 0.4, 1.2, 1.7, -0.5, 0.0],
            "G": [2, 2, 0, 0, 2, 2],
            "D": [0.3, 0.3, 0.0, 0.0, 0.8, 0.8],
        }
    )


def test_phase3_objects_accept_valid_balanced_panel_inputs() -> None:
    from contdid import (
        ContDIDResult,
        ContDIDSpec,
        PanelData,
        validate_panel_data,
        validate_spec,
    )

    panel = PanelData(frame=_make_valid_frame())
    validated_panel = validate_panel_data(panel)

    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        treatment_type="continuous",
        anticipation=0,
    )
    validated_spec = validate_spec(spec, panel=validated_panel)

    result = ContDIDResult(
        estimand="att_curve",
        grid=[0.0, 0.5, 1.0],
        estimate=[0.0, 0.1, 0.2],
        std_error=[0.05, 0.05, 0.05],
        metadata={"target_parameter": validated_spec.target_parameter},
    )

    assert validated_panel is panel
    assert validated_spec is spec
    assert result.metadata["target_parameter"] == "level"
    assert result.grid == [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "grid": [0.2],
                "estimate": [1.0, 2.0],
                "std_error": [0.1, 0.2],
            },
            "grid must have the same shape as estimate",
        ),
        (
            {
                "grid": [0.2, 0.4],
                "estimate": [1.0, 2.0],
                "std_error": [0.1],
            },
            "estimate and std_error must have the same shape",
        ),
        (
            {
                "grid": [0.2],
                "estimate": [math.nan],
                "std_error": [0.1],
            },
            "estimate must contain only finite non-boolean values",
        ),
        (
            {
                "grid": [0.2],
                "estimate": [True],
                "std_error": [0.1],
            },
            "estimate must contain only finite non-boolean values",
        ),
        (
            {
                "grid": [0.2],
                "estimate": [1.0],
                "std_error": [-0.1],
            },
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            {
                "grid": [0.2],
                "estimate": [1.0],
                "std_error": [False],
            },
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            {
                "grid": [math.inf],
                "estimate": [1.0],
                "std_error": [0.1],
            },
            "grid must contain only finite non-boolean values",
        ),
        (
            {
                "grid": ["stale"],
                "estimate": [1.0],
                "std_error": [0.1],
            },
            "grid must contain only finite non-boolean values",
        ),
        (
            {
                "grid": [True],
                "estimate": [1.0],
                "std_error": [0.1],
            },
            "grid must contain only finite non-boolean values",
        ),
        (
            {
                "grid": [0.2],
                "estimate": ["stale"],
                "std_error": [0.1],
            },
            "estimate must contain only finite non-boolean values",
        ),
        (
            {
                "grid": [0.2],
                "estimate": [1.0],
                "std_error": ["stale"],
            },
            "std_error must contain only finite non-boolean nonnegative values",
        ),
        (
            {
                "grid": [],
                "estimate": [],
                "std_error": [],
            },
            "estimate and std_error must contain at least one value",
        ),
    ],
)
def test_result_container_rejects_invalid_public_payload_shapes(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match=message):
        ContDIDResult(estimand="ATT(d)", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"confidence_interval": [[0.3, 0.5]]},
            "confidence_interval must have one lower/upper pair per estimate",
        ),
        (
            {"confidence_interval": [[0.3, 0.5], [math.nan, 0.9]]},
            "confidence_interval must contain only finite non-boolean values",
        ),
        (
            {"confidence_interval": [[0.3, 0.5], ["stale", 0.9]]},
            "confidence_interval must contain only finite non-boolean values",
        ),
        (
            {"confidence_interval": [[False, 0.5], [0.6, 1.0]]},
            "confidence_interval must contain only finite non-boolean values",
        ),
        (
            {"confidence_interval": [[0.3, True], [0.6, 1.0]]},
            "confidence_interval must contain only finite non-boolean values",
        ),
        (
            {"confidence_interval": [[0.6, 0.5], [0.7, 0.9]]},
            "confidence_interval lower bounds must not exceed upper bounds",
        ),
        (
            {"confidence_interval": [[0.1, 0.3], [0.9, 1.1]]},
            "confidence_interval must contain the point estimate",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1],
                    "upper": [0.9],
                    "critical_value": 1.96,
                }
            },
            "confidence_band lower and upper must match estimate shape",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, 0.7],
                    "upper": [0.9, 1.1],
                    "critical_value": math.inf,
                }
            },
            "confidence_band critical_value must be finite and nonnegative",
        ),
        (
            {
                "confidence_band": {
                    "lower": [False, 0.4],
                    "upper": [0.7, 1.2],
                    "critical_value": 1.96,
                }
            },
            "confidence_band lower and upper must contain only finite non-boolean values",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, "stale"],
                    "upper": [0.7, 1.2],
                    "critical_value": 1.96,
                }
            },
            "confidence_band lower and upper must contain only finite non-boolean values",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, 0.4],
                    "upper": [True, 1.2],
                    "critical_value": 1.96,
                }
            },
            "confidence_band lower and upper must contain only finite non-boolean values",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, 0.9],
                    "upper": [0.7, 1.1],
                    "critical_value": 1.96,
                }
            },
            "confidence_band must contain the point estimate",
        ),
        (
            {"critical_value": math.nan},
            "critical_value must be finite and nonnegative",
        ),
        (
            {"critical_value": "stale"},
            "critical_value must be finite and nonnegative",
        ),
        (
            {"critical_value": True},
            "critical_value must be finite and nonnegative",
        ),
        (
            {"metadata": {"critical_value": True}},
            "critical_value must be finite and nonnegative",
        ),
        (
            {"metadata": {"critical_value": "stale"}},
            "critical_value must be finite and nonnegative",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, 0.4],
                    "upper": [0.7, 1.2],
                    "critical_value": True,
                }
            },
            "confidence_band critical_value must be finite and nonnegative",
        ),
        (
            {
                "confidence_band": {
                    "lower": [0.1, 0.4],
                    "upper": [0.7, 1.2],
                    "critical_value": "stale",
                }
            },
            "confidence_band critical_value must be finite and nonnegative",
        ),
        (
            {
                "critical_value": 2.5,
                "confidence_band": {
                    "lower": [0.1, 0.4],
                    "upper": [0.7, 1.2],
                    "critical_value": 1.96,
                },
            },
            "critical_value must match confidence_band critical_value",
        ),
    ],
)
def test_result_container_rejects_invalid_inference_payload_shapes(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match=message):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0.2, 0.5],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            **kwargs,
        )


def test_result_container_normalizes_inference_payload_metadata_mirrors() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        confidence_interval=(("0.3", "0.5"), ("0.6", "1.0")),
        confidence_band={
            "lower": ("0.1", "0.4"),
            "upper": ("0.7", "1.2"),
            "critical_value": "1.96",
        },
        metadata={
            "grid": ["stale"],
            "estimate": ["stale"],
            "confidence_interval": [["stale"]],
        },
    )

    assert result.confidence_interval == [[0.3, 0.5], [0.6, 1.0]]
    assert result.confidence_band == {
        "lower": [0.1, 0.4],
        "upper": [0.7, 1.2],
        "critical_value": 1.96,
    }
    assert result.metadata["grid"] == result.grid
    assert result.metadata["estimate"] == result.estimate
    assert result.metadata["confidence_interval"] == result.confidence_interval
    assert result.metadata["confidence_band"] == result.confidence_band
    for eventstudy_only_key in (
        "event_time",
        "event_time_grid",
        "timing_group",
        "cohort_summary",
        "support",
        "timing_group_support",
    ):
        assert eventstudy_only_key not in result.metadata


def test_result_container_backfills_critical_value_from_confidence_band() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        confidence_band={
            "lower": [0.1, 0.4],
            "upper": [0.7, 1.2],
            "critical_value": "1.96",
        },
    )

    assert result.critical_value == 1.96
    assert result.metadata["critical_value"] == 1.96
    assert result.confidence_band["critical_value"] == 1.96


def test_result_container_exposes_readable_dose_frame_and_compact_repr() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        confidence_interval=[[0.3, 0.5], [0.6, 1.0]],
        confidence_band={
            "lower": [0.1, 0.4],
            "upper": [0.7, 1.2],
            "critical_value": 1.96,
        },
        metadata={
            "confidence_band_kind": "simultaneous_multiplier",
            "large_private_payload": {"do_not_render": list(range(20))},
        },
    )

    frame = result.to_frame()

    assert repr(result) == "ContDIDResult(estimand='ATT(d)', grid_size=2)"
    assert list(frame.columns) == [
        "dose",
        "estimate",
        "std_error",
        "ci_lower",
        "ci_upper",
        "band_lower",
        "band_upper",
    ]
    assert frame.to_dict(orient="list") == {
        "dose": [0.2, 0.5],
        "estimate": [0.4, 0.8],
        "std_error": [0.1, 0.2],
        "ci_lower": [0.3, 0.6],
        "ci_upper": [0.5, 1.0],
        "band_lower": [0.1, 0.4],
        "band_upper": [0.7, 1.2],
    }
    assert result.to_markdown() == "\n".join(
        [
            "| Dose | Estimate | Std. error | Pointwise CI | Uniform band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.200000 | 0.400000 | 0.100000 | [0.300000, 0.500000] | [0.100000, 0.700000] |",
            "| 0.500000 | 0.800000 | 0.200000 | [0.600000, 1.000000] | [0.400000, 1.200000] |",
        ]
    )
    assert result.to_markdown(include_caption=True) == "\n".join(
        [
            "ContDIDResult: ATT(d), 2 rows, dose axis, critical value 1.960000.",
            "",
            "| Dose | Estimate | Std. error | Pointwise CI | Uniform band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.200000 | 0.400000 | 0.100000 | [0.300000, 0.500000] | [0.100000, 0.700000] |",
            "| 0.500000 | 0.800000 | 0.200000 | [0.600000, 1.000000] | [0.400000, 1.200000] |",
        ]
    )
    assert result.to_markdown(include_caption=True, digits=3) == "\n".join(
        [
            "ContDIDResult: ATT(d), 2 rows, dose axis, critical value 1.960.",
            "",
            "| Dose | Estimate | Std. error | Pointwise CI | Uniform band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.200 | 0.400 | 0.100 | [0.300, 0.500] | [0.100, 0.700] |",
            "| 0.500 | 0.800 | 0.200 | [0.600, 1.000] | [0.400, 1.200] |",
        ]
    )


def test_result_to_markdown_explains_missing_interval_and_band_cells() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2],
        estimate=[0.4],
        std_error=[0.1],
    )

    assert result.to_markdown(digits=2) == "\n".join(
        [
            "| Dose | Estimate | Std. error | Pointwise CI | Confidence band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.20 | 0.40 | 0.10 | not estimated (pointwise CI) | not estimated (confidence band) |",
        ]
    )


def test_result_to_markdown_suppresses_display_only_negative_zero() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.25],
        estimate=[-1e-9],
        std_error=[0.0],
        confidence_interval=[[-1e-9, 1e-9]],
        confidence_band={
            "lower": [-1e-9],
            "upper": [1e-9],
            "critical_value": 0.0,
        },
        metadata={"confidence_band_kind": "pointwise_analytic"},
    )

    assert result.estimate == [-1e-9]
    assert result.to_markdown(digits=6) == "\n".join(
        [
            "| Dose | Estimate | Std. error | Pointwise CI | Pointwise band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.250000 | 0.000000 | 0.000000 | [0.000000, 0.000000] | same as pointwise CI |",
        ]
    )


def test_result_to_markdown_keeps_distinct_pointwise_band_bounds() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.25],
        estimate=[0.5],
        std_error=[0.1],
        confidence_interval=[[0.3, 0.7]],
        confidence_band={
            "lower": [0.2],
            "upper": [0.8],
            "critical_value": 3.0,
        },
        metadata={"confidence_band_kind": "pointwise_multiplier"},
    )

    assert result.to_markdown(digits=1) == "\n".join(
        [
            "| Dose | Estimate | Std. error | Pointwise CI | Pointwise band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.2 | 0.5 | 0.1 | [0.3, 0.7] | [0.2, 0.8] |",
        ]
    )


def test_result_container_saves_display_ready_dose_plot(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDResult

    output = tmp_path / "dose-plot.png"
    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5, 0.8],
        estimate=[0.4, 0.7, 0.5],
        std_error=[0.1, 0.1, 0.2],
        confidence_interval=[[0.2, 0.6], [0.5, 0.9], [0.1, 0.9]],
        confidence_band={
            "lower": [0.1, 0.4, 0.0],
            "upper": [0.7, 1.0, 1.0],
            "critical_value": 2.1,
        },
    )

    saved = result.save_plot(output)

    assert saved == output
    image = Image.open(saved)
    assert image.size == (1260, 810)
    colors = image.getcolors(maxcolors=1_000_000)
    assert colors is not None
    rendered_colors = {color for _, color in colors}
    assert (191, 219, 254) in rendered_colors
    assert (37, 99, 235) in rendered_colors
    assert len(rendered_colors) > 10


def test_result_plot_draws_confidence_band_for_single_dose_point(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "single-dose-band.png"
    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.5],
        estimate=[0.4],
        std_error=[0.1],
        confidence_band={
            "lower": [0.2],
            "upper": [0.6],
            "critical_value": 2.0,
        },
    )

    result.save_plot(output)

    image = Image.open(output)
    _, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    band_probe_x, band_probe_y = results_module._plot_scale(
        axis_values,
        [0.3],
        x_bounds,
        y_bounds,
    )[0]
    assert image.getpixel((round(band_probe_x), round(band_probe_y))) == (
        191,
        219,
        254,
    )


def test_result_plot_reports_band_legend_for_single_supported_eventstudy_point(
    tmp_path,
    monkeypatch,
) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    captured_has_band: list[bool] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, has_band: bool, **kwargs) -> None:
        captured_has_band.append(has_band)
        original_draw_legend(*args, has_band=has_band, **kwargs)

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.0, 0.1],
        std_error=[0.1, 0.1, 0.1],
        confidence_band={
            "lower": [-0.2, -0.3, -0.4],
            "upper": [0.2, 0.3, 0.4],
            "critical_value": 2.0,
        },
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [False, True, False]},
    )

    output = tmp_path / "single-supported-eventstudy-band.png"
    result.save_plot(output)

    assert captured_has_band == [True]
    image = Image.open(output)
    _, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    band_probe_x, band_probe_y = results_module._plot_scale(
        [axis_values[1]],
        [0.15],
        x_bounds,
        y_bounds,
    )[0]
    assert image.getpixel((round(band_probe_x) + 5, round(band_probe_y))) == (
        191,
        219,
        254,
    )


def test_result_container_plot_legend_uses_inference_band_label(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "pointwise-band-plot.png"
    captured_labels: list[str] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, band_label: str, **kwargs) -> None:
        captured_labels.append(band_label)
        original_draw_legend(*args, band_label=band_label, **kwargs)

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        confidence_band={
            "lower": [0.2, 0.4],
            "upper": [0.6, 1.2],
            "critical_value": 1.96,
        },
        metadata={"confidence_band_kind": "pointwise_analytic"},
    )

    result.save_plot(output)

    assert captured_labels == ["Pointwise band"]


def test_result_plot_suppresses_redundant_pointwise_band_layer(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "pointwise-interval-only.png"
    captured_layers: list[tuple[bool, bool]] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, has_band: bool, has_interval: bool, **kwargs) -> None:
        captured_layers.append((has_band, has_interval))
        original_draw_legend(
            *args,
            has_band=has_band,
            has_interval=has_interval,
            **kwargs,
        )

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5, 0.8],
        estimate=[0.4, 0.8, 0.6],
        std_error=[0.1, 0.2, 0.1],
        confidence_interval=[[0.2, 0.6], [0.4, 1.2], [0.4, 0.8]],
        confidence_band={
            "lower": [0.2, 0.4, 0.4],
            "upper": [0.6, 1.2, 0.8],
            "critical_value": 1.96,
        },
        metadata={"confidence_band_kind": "pointwise_multiplier"},
    )

    result.save_plot(output)

    assert captured_layers == [(False, True)]


def test_result_plot_suppresses_redundant_pointwise_band_on_supported_rows(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "supported-row-pointwise-band.png"
    captured_layers: list[tuple[bool, bool]] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, has_band: bool, has_interval: bool, **kwargs) -> None:
        captured_layers.append((has_band, has_interval))
        original_draw_legend(
            *args,
            has_band=has_band,
            has_interval=has_interval,
            **kwargs,
        )

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[0.0, 0.2, 0.4],
        std_error=[0.1, 0.1, 0.1],
        event_time_grid=[-1, 0, 1],
        confidence_interval=[[-0.2, 0.2], [0.0, 0.4], [0.2, 0.6]],
        confidence_band={
            "lower": [-0.2, 0.0, -99.0],
            "upper": [0.2, 0.4, 99.0],
            "critical_value": 1.96,
        },
        metadata={
            "confidence_band_kind": "pointwise_multiplier",
            "support": [True, True, False],
        },
    )

    result.save_plot(output)

    assert output.exists()
    assert captured_layers == [(False, True)]


def test_result_plot_keeps_distinct_pointwise_band_layer(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "distinct-pointwise-band.png"
    captured_layers: list[tuple[bool, bool]] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, has_band: bool, has_interval: bool, **kwargs) -> None:
        captured_layers.append((has_band, has_interval))
        original_draw_legend(
            *args,
            has_band=has_band,
            has_interval=has_interval,
            **kwargs,
        )

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2, 0.5, 0.8],
        estimate=[0.4, 0.8, 0.6],
        std_error=[0.1, 0.2, 0.1],
        confidence_interval=[[0.2, 0.6], [0.4, 1.2], [0.4, 0.8]],
        confidence_band={
            "lower": [0.1, 0.3, 0.3],
            "upper": [0.7, 1.3, 0.9],
            "critical_value": 2.20,
        },
        metadata={"confidence_band_kind": "pointwise_multiplier"},
    )

    result.save_plot(output)

    assert captured_layers == [(True, True)]


def test_result_container_saves_display_ready_eventstudy_plot_with_support(
    tmp_path,
) -> None:
    from PIL import Image

    from contdid import ContDIDResult

    output = tmp_path / "event-study.png"
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, True, False]},
    )

    saved = result.save_plot(output, title="Checked event-study plot")

    image = Image.open(saved)
    assert image.size == (1260, 810)
    colors = image.getcolors(maxcolors=1_000_000)
    assert colors is not None
    rendered_colors = {color for _, color in colors}
    assert (100, 116, 139) in rendered_colors
    assert (148, 163, 184) in rendered_colors


def test_result_plot_omits_unsupported_legend_when_all_rows_have_support(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    captured_support_args: list[list[bool] | None] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, support: list[bool] | None, **kwargs) -> None:
        captured_support_args.append(support)
        original_draw_legend(*args, support=support, **kwargs)

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, True, True]},
    )

    result.save_plot(tmp_path / "all-supported-event-study.png")

    assert result.metadata["support"] == [True, True, True]
    assert captured_support_args == [None]


def test_result_plot_keeps_unsupported_legend_when_any_row_lacks_support(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    captured_support_args: list[list[bool] | None] = []
    original_draw_legend = results_module._draw_plot_legend

    def capture_draw_legend(*args, support: list[bool] | None, **kwargs) -> None:
        captured_support_args.append(support)
        original_draw_legend(*args, support=support, **kwargs)

    monkeypatch.setattr(results_module, "_draw_plot_legend", capture_draw_legend)
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, True, False]},
    )

    result.save_plot(tmp_path / "partly-supported-event-study.png")

    assert captured_support_args == [[True, True, False]]


def test_result_plot_unsupported_legend_reports_missing_support_count() -> None:
    from PIL import Image, ImageDraw

    from contdid.results import _draw_plot_legend, _plot_font_stack

    class RecordingDraw:
        def __init__(self) -> None:
            image = Image.new("RGB", (1260, 810), color="white")
            self.delegate = ImageDraw.Draw(image)
            self.text_values: list[str] = []

        def rectangle(self, *args, **kwargs) -> None:
            return None

        def line(self, *args, **kwargs) -> None:
            return None

        def ellipse(self, *args, **kwargs) -> None:
            return None

        def text(self, position, text, **kwargs) -> None:
            self.text_values.append(str(text))

        def textbbox(self, *args, **kwargs):
            return self.delegate.textbbox(*args, **kwargs)

    draw = RecordingDraw()

    _draw_plot_legend(
        draw,
        legend_font=_plot_font_stack()[3],
        band_label="Uniform band",
        has_band=False,
        has_interval=False,
        support=[True, False, True, False],
        data_points=[],
    )

    assert "No local support (2/4)" in draw.text_values


def test_result_plot_legend_fits_long_support_count_label(monkeypatch) -> None:
    from PIL import Image, ImageDraw

    import contdid.results as results_module

    image = Image.new("RGB", (1260, 810), color="white")
    draw = ImageDraw.Draw(image)
    legend_font = results_module._plot_font_stack()[3]
    support = [False] * 1_000_000
    fitted_labels: list[tuple[str, str, int]] = []
    original_fit_plot_text = results_module._fit_plot_text

    def capture_fit_plot_text(draw, text, font, *, max_width):
        fitted = original_fit_plot_text(draw, text, font, max_width=max_width)
        fitted_labels.append((str(text), fitted, max_width))
        return fitted

    monkeypatch.setattr(results_module, "_fit_plot_text", capture_fit_plot_text)

    results_module._draw_plot_legend(
        draw,
        legend_font=legend_font,
        band_label="Uniform band",
        has_band=False,
        has_interval=False,
        support=support,
        data_points=[],
    )

    left, _, right, _ = results_module._plot_legend_bounds(
        has_band=False,
        has_interval=False,
        support=support,
        data_points=[],
    )
    label_width = right - (left + 58) - 14
    long_label = next(
        fitted
        for original, fitted, _ in fitted_labels
        if original == "No local support (1000000/1000000)"
    )

    assert long_label.endswith("...")
    assert results_module._text_size(draw, long_label, legend_font)[0] <= label_width


def test_result_plot_line_segments_do_not_cross_unsupported_eventstudy_rows() -> None:
    from contdid.results import _supported_index_segments, _supported_line_segments

    points = [(0.0, 1.0), (1.0, 1.5), (2.0, 0.5), (3.0, 1.0), (4.0, 1.25)]

    assert _supported_index_segments(None, len(points)) == [[0, 1, 2, 3, 4]]
    assert _supported_index_segments([True, True, False, True, True], len(points)) == [
        [0, 1],
        [3, 4],
    ]
    assert _supported_line_segments(points, None) == [points]
    assert _supported_line_segments(points, [True, True, False, True, True]) == [
        [(0.0, 1.0), (1.0, 1.5)],
        [(3.0, 1.0), (4.0, 1.25)],
    ]
    assert _supported_line_segments(points, [True, False, True, False, True]) == []


def test_result_plot_pixels_do_not_bridge_unsupported_eventstudy_rows(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "unsupported-gap.png"
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[0.0, 1.0, 0.0],
        std_error=[0.1, 0.1, 0.1],
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, False, True]},
    )

    axis_name, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    points = results_module._plot_scale(axis_values, result.estimate, x_bounds, y_bounds)
    midpoint_x = int(round((points[0][0] + points[1][0]) / 2.0))
    midpoint_y = int(round((points[0][1] + points[1][1]) / 2.0))

    assert axis_name == "event_time"
    result.save_plot(output)

    image = Image.open(output)
    line_rgb = (37, 99, 235)
    bridge_probe = [
        image.getpixel((midpoint_x + dx, midpoint_y + dy))
        for dx in range(-2, 3)
        for dy in range(-2, 3)
    ]
    assert line_rgb not in bridge_probe


def test_result_plot_pixels_skip_unsupported_eventstudy_uncertainty(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "unsupported-uncertainty.png"
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[0.0, 0.0, 0.0],
        std_error=[0.1, 0.1, 0.1],
        confidence_interval=[[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]],
        confidence_band={
            "lower": [-0.3, -0.3, -0.3],
            "upper": [0.3, 0.3, 0.3],
            "critical_value": 2.0,
        },
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, False, True]},
    )

    _, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    unsupported_x = int(
        round(results_module._plot_scale([axis_values[1]], [0.0], x_bounds, y_bounds)[0][0])
    )
    interval_upper_y = int(
        round(results_module._plot_scale([axis_values[1]], [0.2], x_bounds, y_bounds)[0][1])
    )
    band_center_y = int(
        round(results_module._plot_scale([axis_values[1]], [0.0], x_bounds, y_bounds)[0][1])
    )

    result.save_plot(output)

    image = Image.open(output)
    interval_rgb = (30, 64, 175)
    band_rgb = (191, 219, 254)
    interval_probe = [
        image.getpixel((unsupported_x + dx, interval_upper_y + dy))
        for dx in range(-3, 4)
        for dy in range(-3, 4)
    ]
    band_probe = [
        image.getpixel((unsupported_x + dx, band_center_y + dy))
        for dx in range(-3, 4)
        for dy in range(-3, 4)
    ]

    assert interval_rgb not in interval_probe
    assert band_rgb not in band_probe


def test_result_plot_bounds_include_unsupported_markers_not_uncertainty() -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    unsupported_result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[0.0, 100.0, 0.0],
        std_error=[0.1, 100.0, 0.1],
        confidence_interval=[[-0.2, 0.2], [-100.0, 100.0], [-0.2, 0.2]],
        confidence_band={
            "lower": [-0.3, -150.0, -0.3],
            "upper": [0.3, 150.0, 0.3],
            "critical_value": 2.0,
        },
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, False, True]},
    )

    assert results_module._plot_y_values(unsupported_result) == pytest.approx(
        [0.0, 100.0, 0.0, -0.2, 0.2, -0.2, 0.2, -0.3, -0.3, 0.3, 0.3],
        abs=0.0,
    )
    x_bounds, y_bounds = results_module._plot_bounds(unsupported_result)
    assert x_bounds == (-1.0, 1.0)
    assert y_bounds == pytest.approx((-10.33, 110.03))


def test_result_plot_keeps_extreme_unsupported_marker_visible(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    output = tmp_path / "extreme-unsupported-marker.png"
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[0.0, 100.0, 0.0],
        std_error=[0.1, 100.0, 0.1],
        confidence_interval=[[-0.2, 0.2], [-100.0, 100.0], [-0.2, 0.2]],
        confidence_band={
            "lower": [-0.3, -150.0, -0.3],
            "upper": [0.3, 150.0, 0.3],
            "critical_value": 2.0,
        },
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, False, True]},
    )

    _, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    unsupported_x, unsupported_y = results_module._plot_scale(
        [axis_values[1]],
        [result.estimate[1]],
        x_bounds,
        y_bounds,
    )[0]

    result.save_plot(output)

    image = Image.open(output)
    unsupported_rgb = (148, 163, 184)
    marker_probe = [
        image.getpixel((round(unsupported_x) + dx, round(unsupported_y) + dy))
        for dx in range(-7, 8)
        for dy in range(-7, 8)
    ]
    assert unsupported_rgb in marker_probe


def test_result_plot_draws_every_unsupported_marker_on_long_eventstudy_axis(
    tmp_path,
) -> None:
    from PIL import Image

    from contdid import ContDIDResult
    import contdid.results as results_module

    event_time = list(range(72))
    support = [True] * len(event_time)
    unsupported_index = 1
    support[unsupported_index] = False
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=event_time,
        estimate=[0.25] * len(event_time),
        std_error=[0.1] * len(event_time),
        event_time=event_time,
        event_time_grid=event_time,
        metadata={"support": support},
    )
    marker_stride = max(1, len(event_time) // 24)
    assert unsupported_index % marker_stride != 0

    _, axis_values = results_module._plot_axis(result)
    x_bounds, y_bounds = results_module._plot_bounds(result)
    unsupported_x, unsupported_y = results_module._plot_scale(
        [axis_values[unsupported_index]],
        [result.estimate[unsupported_index]],
        x_bounds,
        y_bounds,
    )[0]

    output = tmp_path / "long-axis-unsupported-marker.png"
    result.save_plot(output)

    image = Image.open(output)
    unsupported_rgb = (148, 163, 184)
    marker_probe = [
        image.getpixel((round(unsupported_x) + dx, round(unsupported_y) + dy))
        for dx in range(-7, 8)
        for dy in range(-7, 8)
    ]
    assert unsupported_rgb in marker_probe


def test_result_plot_bounds_ignore_all_unsupported_eventstudy_uncertainty() -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.0, 0.1],
        std_error=[100.0, 100.0, 100.0],
        confidence_interval=[[-100.0, 100.0], [-200.0, 200.0], [-300.0, 300.0]],
        confidence_band={
            "lower": [-400.0, -500.0, -600.0],
            "upper": [400.0, 500.0, 600.0],
            "critical_value": 2.0,
        },
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [False, False, False]},
    )

    assert results_module._plot_supported_indices(result) == []
    assert results_module._plot_y_values(result) == pytest.approx([-0.1, 0.0, 0.1])
    assert results_module._plot_bounds(result)[0] == (-1.0, 1.0)
    assert results_module._plot_bounds(result)[1] == pytest.approx((-0.12, 0.12))


def test_result_plot_uses_data_aligned_ticks_for_small_discrete_axes() -> None:
    from contdid.results import _plot_x_ticks

    assert _plot_x_ticks(
        axis_name="event_time",
        axis_values=[-2.0, -1.0, 0.0, 1.0, 2.0],
        x_bounds=(-2.0, 2.0),
    ) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert _plot_x_ticks(
        axis_name="dose",
        axis_values=[0.2, 0.5, 0.8],
        x_bounds=(0.2, 0.8),
    ) == [0.2, 0.5, 0.8]


def test_result_plot_uses_integer_ticks_for_long_eventstudy_axes() -> None:
    from contdid.results import _format_tick, _plot_x_ticks

    ticks = _plot_x_ticks(
        axis_name="event_time",
        axis_values=[float(value) for value in [-6, -5, -4, -3, -2, *range(14)]],
        x_bounds=(-6.0, 13.0),
    )

    assert ticks[0] == -6.0
    assert ticks[-1] == 13.0
    assert 0.0 in ticks
    assert all(float(value).is_integer() for value in ticks)
    assert all(
        "." not in _format_tick(value, axis_name="event_time") for value in ticks
    )
    assert len(ticks) <= 9


def test_result_plot_uses_readable_y_axis_ticks() -> None:
    from contdid.results import _format_tick, _format_y_tick, _plot_y_ticks

    assert _plot_y_ticks((0.0, 1.1)) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert _plot_y_ticks((-0.26, 0.26)) == [-0.2, -0.1, 0.0, 0.1, 0.2]
    assert [_format_y_tick(value) for value in _plot_y_ticks((0.0, 1.1))] == [
        "0",
        "0.25",
        "0.5",
        "0.75",
        "1",
    ]
    assert _format_tick(-1e-9, axis_name="dose") == "0.00"
    assert _format_y_tick(-1e-9) == "0"
    assert _format_y_tick(-0.0001) == "0"
    assert _format_y_tick(123.456) == "123"
    assert _format_y_tick(-123.456) == "-123"


def test_result_plot_legend_moves_away_from_data() -> None:
    from contdid.results import _plot_legend_bounds

    assert _plot_legend_bounds(
        has_band=False,
        has_interval=False,
        support=None,
        data_points=[],
    ) == (130, 122, 402, 171)
    assert _plot_legend_bounds(
        has_band=False,
        has_interval=False,
        support=None,
        data_points=[(160, 145)],
    ) == (914, 122, 1186, 171)


def test_result_plot_long_header_text_is_fitted_to_plot_width() -> None:
    from PIL import Image, ImageDraw

    from contdid.results import (
        _PLOT_MARGIN_LEFT,
        _PLOT_MARGIN_RIGHT,
        _PLOT_WIDTH,
        _fit_plot_text,
        _plot_font_stack,
        _text_size,
    )

    image = Image.new("RGB", (_PLOT_WIDTH, 160), color="white")
    draw = ImageDraw.Draw(image)
    title_font = _plot_font_stack()[0]
    max_width = _PLOT_WIDTH - _PLOT_MARGIN_LEFT - _PLOT_MARGIN_RIGHT
    long_title = (
        "ATT(event_time) by event time with simultaneous multiplier bands, "
        "cohort diagnostics, support diagnostics, and an intentionally long "
        "manuscript-ready title"
    )

    fitted = _fit_plot_text(
        draw,
        long_title,
        title_font,
        max_width=max_width,
    )

    assert fitted.endswith("...")
    assert len(fitted) < len(long_title)
    assert _text_size(draw, fitted, title_font)[0] <= max_width


def test_result_container_fits_long_plot_text_before_render(
    tmp_path,
    monkeypatch,
) -> None:
    from contdid import ContDIDResult
    import contdid.results as results_module

    captured: list[tuple[str, int]] = []
    original_fit_plot_text = results_module._fit_plot_text

    def capture_fit_plot_text(draw, text, font, *, max_width):
        fitted = original_fit_plot_text(draw, text, font, max_width=max_width)
        captured.append((fitted, max_width))
        return fitted

    monkeypatch.setattr(results_module, "_fit_plot_text", capture_fit_plot_text)
    long_text = 2 * (
        "ATT(event_time) release report with long support, timing, inference, "
        "and source diagnostics that should never overflow the PNG canvas"
    )
    result = ContDIDResult(
        estimand=long_text + " ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time=[-1, 0, 1],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, True, True]},
    )

    result.save_plot(
        tmp_path / "long-header-eventstudy.png",
        title=long_text,
        subtitle=long_text,
    )

    header_and_axis_labels = captured[:3]
    assert len(captured) == 4
    assert all(text.endswith("...") for text, _ in header_and_axis_labels)
    assert [max_width for _, max_width in header_and_axis_labels] == [
        results_module._PLOT_WIDTH
        - results_module._PLOT_MARGIN_LEFT
        - results_module._PLOT_MARGIN_RIGHT,
        results_module._PLOT_WIDTH
        - results_module._PLOT_MARGIN_LEFT
        - results_module._PLOT_MARGIN_RIGHT,
        results_module._PLOT_HEIGHT
        - results_module._PLOT_MARGIN_TOP
        - results_module._PLOT_MARGIN_BOTTOM,
    ]


def test_result_container_plot_path_must_target_png(tmp_path) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.5],
        estimate=[1.0],
        std_error=[0.1],
    )

    with pytest.raises(ValueError, match="path must end with .png"):
        result.save_plot(tmp_path / "plot.svg")


def test_public_estimator_result_saves_display_ready_plot(tmp_path) -> None:
    from PIL import Image

    from contdid import ContDIDSpec, PanelData, estimate_dose_effects

    rows = [
        ("u0", 1, 0.0, 0, 0.0),
        ("u0", 2, 0.0, 0, 0.0),
        ("u1", 1, 0.0, 0, 0.0),
        ("u1", 2, 0.0, 0, 0.0),
        ("t1", 1, 0.0, 2, 0.2),
        ("t1", 2, 0.12, 2, 0.2),
        ("t2", 1, 0.0, 2, 0.4),
        ("t2", 2, 0.21, 2, 0.4),
        ("t3", 1, 0.0, 2, 0.6),
        ("t3", 2, 0.35, 2, 0.6),
        ("t4", 1, 0.0, 2, 0.8),
        ("t4", 2, 0.61, 2, 0.8),
    ]
    panel = PanelData(
        frame=pd.DataFrame(rows, columns=["id", "time_period", "Y", "G", "D"])
    )
    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
        bstrap=False,
    )

    result = estimate_dose_effects(
        panel,
        spec,
        dvals=[0.2, 0.5, 0.8],
        degree=1,
        num_knots=0,
    )
    saved = result.save_plot(tmp_path / "estimated-dose.png")

    assert saved.exists()
    assert result.metadata["inference"] == "analytic"
    image = Image.open(saved)
    assert image.size == (1260, 810)
    colors = image.getcolors(maxcolors=1_000_000)
    assert colors is not None
    assert (37, 99, 235) in {color for _, color in colors}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"event_time": [1, 2], "event_time_grid": [1, 2]},
            "event_time and event_time_grid must match result grid",
        ),
        (
            {"event_time": [0, 1], "event_time_grid": [0, 2]},
            "event_time and event_time_grid must match result grid",
        ),
        (
            {"event_time": [0]},
            "event_time must match result estimate length",
        ),
        (
            {"event_time": [0.5, 1.0]},
            "event_time must contain only integer values",
        ),
        (
            {"event_time": ["stale", 1]},
            "event_time must contain only integer values",
        ),
        (
            {},
            "event-study result requires event_time or event_time_grid",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [{"event_time": 0}],
            },
            "cohort_summary must match event_time length",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [{}, {"event_time": 1}],
            },
            "cohort_summary rows must include event_time",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [{"event_time": 0}, {"event_time": 2}],
            },
            "cohort_summary event_time must match event_time grid",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [{"event_time": 0.5}, {"event_time": 1}],
            },
            "cohort_summary event_time must contain only integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [{"event_time": False}, {"event_time": 1}],
            },
            "cohort_summary event_time must contain only integer values",
        ),
        (
            {
                "cohort_summary": [{"event_time": 0}, {"event_time": 1}],
            },
            "event-study result requires event_time or event_time_grid",
        ),
        (
            {
                "cohort_summary": [{}, {}],
            },
            "event-study result requires event_time or event_time_grid",
        ),
        (
            {"event_time": [0, 1], "timing_group": [0]},
            "timing_group must contain only positive integer values",
        ),
        (
            {"event_time": [0, 1], "timing_group": [-1]},
            "timing_group must contain only positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0, "timing_groups": [0]},
                    {"event_time": 1},
                ],
            },
            "cohort_summary timing_groups must contain only positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": -1}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates timing_group must contain only positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 3}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates timing_group must be listed in cohort_summary timing_groups",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2}],
                        "mean_estimate": 999.0,
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_summary mean_estimate must match result estimate",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0, "std_error": -0.1},
                    {"event_time": 1},
                ],
            },
            "cohort_summary std_error must be finite and nonnegative",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0, "std_error": 99.0},
                    {"event_time": 1},
                ],
            },
            "cohort_summary std_error must match result std_error",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2}],
                        "support": False,
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_summary support must match cohort_estimates presence",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "cohort_estimates": [],
                        "support": True,
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_summary support must match cohort_estimates presence",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0, "support": 1},
                    {"event_time": 1},
                ],
            },
            "cohort_summary support must be a boolean",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0, "support": True},
                    {"event_time": 1},
                ],
            },
            "cohort_summary support must be present for every event-time row",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {"event_time": 0},
                    {"event_time": 1},
                ],
                "metadata": {"support": [True, False]},
            },
            "cohort_summary support must be present for every event-time row",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2}],
                        "support": True,
                    },
                    {"event_time": 1, "support": False},
                ],
                "metadata": {"support": [False, False]},
            },
            "metadata support must match cohort_summary support",
        ),
        (
            {
                "event_time": [0, 1],
                "metadata": {"support": [True]},
            },
            "metadata support must match result estimate length",
        ),
        (
            {
                "event_time": [0, 1],
                "metadata": {"support": [1, False]},
            },
            "metadata support must contain only boolean values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "estimate": float("nan")}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates estimate must be finite",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2, "std_error": -0.1}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates std_error must be finite and nonnegative",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2, "treated_count": -1}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates treated_count must contain only finite positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "comparison_count": 0}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates comparison_count must contain only finite positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "treated_count": 10**400}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates treated_count must contain only finite positive integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2, "time_period": 2.5}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates time_period must contain only integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "time_period": "stale"}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates time_period must contain only integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [{"timing_group": 2, "base_period": False}],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates base_period must contain only integer values",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2, 3],
                        "cohort_estimates": [
                            {"timing_group": 2, "aggregation_weight": 0.4},
                            {"timing_group": 3},
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight must be present for every cohort",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2, 3],
                        "cohort_estimates": [
                            {"timing_group": 2, "aggregation_weight": 0.4},
                            {"timing_group": 3, "aggregation_weight": 0.6},
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight requires treated_count",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "aggregation_weight": False}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight must be finite and nonnegative",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2],
                        "cohort_estimates": [
                            {"timing_group": 2, "aggregation_weight": -0.1}
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight must be finite and nonnegative",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2, 3],
                        "cohort_estimates": [
                            {
                                "timing_group": 2,
                                "treated_count": 4,
                                "aggregation_weight": 0.4,
                            },
                            {
                                "timing_group": 3,
                                "treated_count": 5,
                                "aggregation_weight": 0.5,
                            },
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight must sum to one",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2, 3],
                        "cohort_estimates": [
                            {
                                "timing_group": 2,
                                "treated_count": 1,
                                "estimate": 0.0,
                                "aggregation_weight": 0.5,
                            },
                            {
                                "timing_group": 3,
                                "treated_count": 1,
                                "estimate": 1.0,
                                "aggregation_weight": 0.5,
                            },
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight values must reconstruct result estimate",
        ),
        (
            {
                "event_time": [0, 1],
                "cohort_summary": [
                    {
                        "event_time": 0,
                        "timing_groups": [2, 3],
                        "cohort_estimates": [
                            {
                                "timing_group": 2,
                                "treated_count": 1,
                                "aggregation_weight": 0.5,
                            },
                            {
                                "timing_group": 3,
                                "treated_count": 3,
                                "aggregation_weight": 0.5,
                            },
                        ],
                    },
                    {"event_time": 1},
                ],
            },
            "cohort_estimates aggregation_weight must equal treated_count share",
        ),
    ],
)
def test_result_container_rejects_eventstudy_index_drift(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match=message):
        ContDIDResult(
            estimand="ATT(event_time)",
            grid=[0, 1],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            **kwargs,
        )


def test_result_container_normalizes_eventstudy_index_metadata_mirrors() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0, 1],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        event_time=("0", "1"),
        event_time_grid=(0.0, 1.0),
        timing_group=("2", "4"),
        cohort_summary=[
            {
                "event_time": 0.0,
                "timing_groups": ("2",),
                "cohort_estimates": [
                    {
                        "timing_group": "2",
                        "time_period": "3",
                        "base_period": "1",
                        "comparison_count": "4",
                        "treated_count": "5",
                        "aggregation_weight": "1.0",
                        "estimate": "0.4",
                        "std_error": "0.1",
                    }
                ],
                "mean_estimate": "0.4",
                "std_error": "0.1",
                "support": True,
            },
            {"event_time": "1", "support": False},
        ],
        metadata={
            "event_time": ["stale"],
            "event_time_grid": ["stale"],
            "timing_group": ["stale"],
            "cohort_summary": ["stale"],
        },
    )

    assert result.event_time == [0, 1]
    assert result.event_time_grid == [0, 1]
    assert result.timing_group == [2, 4]
    assert result.cohort_summary == [
        {
            "event_time": 0,
            "timing_groups": [2],
            "cohort_estimates": [
                {
                    "timing_group": 2,
                    "time_period": 3,
                    "base_period": 1,
                    "comparison_count": 4,
                    "treated_count": 5,
                    "aggregation_weight": 1.0,
                    "estimate": 0.4,
                    "std_error": 0.1,
                }
            ],
            "mean_estimate": 0.4,
            "std_error": 0.1,
            "support": True,
        },
        {"event_time": 1, "support": False},
    ]
    assert result.metadata["event_time"] == result.event_time
    assert result.metadata["event_time_grid"] == result.event_time_grid
    assert result.metadata["timing_group"] == result.timing_group
    assert result.metadata["cohort_summary"] == result.cohort_summary
    assert result.metadata["support"] == [True, False]


def test_result_container_preserves_large_exact_integer_cohort_counts() -> None:
    from contdid import ContDIDResult

    large_count = 2**53 + 1
    large_period = 2**53 + 3
    large_timing_group = 2**53 + 5
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0],
        estimate=[0.4],
        std_error=[0.1],
        event_time=[0],
        timing_group=[large_timing_group],
        cohort_summary=[
            {
                "event_time": 0,
                "timing_groups": [large_timing_group],
                "cohort_estimates": [
                    {
                        "timing_group": large_timing_group,
                        "time_period": large_period,
                        "base_period": large_period - 1,
                        "treated_count": large_count,
                        "comparison_count": large_count,
                        "aggregation_weight": 1.0,
                        "estimate": 0.4,
                    }
                ],
                "mean_estimate": 0.4,
                "std_error": 0.1,
                "support": True,
            }
        ],
    )

    cohort = result.cohort_summary[0]["cohort_estimates"][0]
    assert result.timing_group == [large_timing_group]
    assert result.cohort_summary[0]["timing_groups"] == [large_timing_group]
    assert cohort["timing_group"] == large_timing_group
    assert cohort["time_period"] == large_period
    assert cohort["base_period"] == large_period - 1
    assert cohort["treated_count"] == large_count
    assert cohort["comparison_count"] == large_count


def test_result_container_rejects_large_event_time_grid_rounding() -> None:
    from contdid import ContDIDResult

    large_event_time = 2**53 + 1
    with pytest.raises(ValueError, match="event_time and event_time_grid must match"):
        ContDIDResult(
            estimand="ATT(event_time)",
            grid=[large_event_time],
            estimate=[0.4],
            std_error=[0.1],
            event_time=[large_event_time],
        )


@pytest.mark.parametrize(
    ("timing_group_support", "message"),
    [
        ("stale", "timing_group_support must be a mapping"),
        ({"timing_groups": []}, "timing_group_support timing_groups must contain"),
        (
            {
                "timing_groups": [3],
                "never_treated_group": 0,
                "reporting_scale": "length of exposure to treatment",
                "base_period_strategy": "fixed",
            },
            "timing_group_support timing_groups must match public timing groups",
        ),
        (
            {
                "timing_groups": [2],
                "reporting_scale": "length of exposure to treatment",
                "base_period_strategy": "fixed",
            },
            "timing_group_support must include never_treated_group",
        ),
        (
            {
                "timing_groups": [2],
                "never_treated_group": 1,
                "reporting_scale": "length of exposure to treatment",
                "base_period_strategy": "fixed",
            },
            "timing_group_support never_treated_group must be 0",
        ),
        (
            {
                "timing_groups": [2],
                "never_treated_group": 0,
                "base_period_strategy": "fixed",
            },
            "timing_group_support must include reporting_scale",
        ),
        (
            {
                "timing_groups": [2],
                "never_treated_group": 0,
                "reporting_scale": "calendar time",
                "base_period_strategy": "fixed",
            },
            "timing_group_support reporting_scale must be length of exposure to treatment",
        ),
        (
            {
                "timing_groups": [2],
                "never_treated_group": 0,
                "reporting_scale": "length of exposure to treatment",
            },
            "timing_group_support must include base_period_strategy",
        ),
        (
            {
                "timing_groups": [2],
                "never_treated_group": 0,
                "reporting_scale": "length of exposure to treatment",
                "base_period_strategy": "rolling",
            },
            "timing_group_support base_period_strategy must be fixed, universal, or varying_pre_period",
        ),
    ],
)
def test_result_container_rejects_invalid_timing_group_support_contract(
    timing_group_support: object,
    message: str,
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match=message):
        ContDIDResult(
            estimand="ATT(event_time)",
            grid=[0, 1],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            event_time=[0, 1],
            timing_group=[2],
            metadata={"timing_group_support": timing_group_support},
        )


def test_result_container_normalizes_timing_group_support_contract() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[0, 1],
        estimate=[0.4, 0.8],
        std_error=[0.1, 0.2],
        event_time=[0, 1],
        timing_group=[2],
        metadata={
            "timing_group_support": {
                "timing_groups": ("2",),
                "never_treated_group": "0",
                "reporting_scale": "length of exposure to treatment",
                "base_period_strategy": "universal",
            }
        },
    )

    assert result.metadata["timing_group_support"] == {
        "timing_groups": [2],
        "never_treated_group": 0,
        "reporting_scale": "length of exposure to treatment",
        "base_period_strategy": "universal",
    }


@pytest.mark.parametrize(
    ("estimand", "valid_identification"),
    [
        (
            "ATT(d)",
            {
                "paper_estimand": "ATT(d)",
                "identifying_assumption": "SPT",
                "ordinary_pt_interpretation": "LATT(d|d)",
                "identification_note": (
                    "The same dose-specific contrast identifies LATT(d|d) under "
                    "ordinary PT; interpreting it as ATT(d) requires SPT."
                ),
            },
        ),
        (
            "ACRT(d)",
            {
                "paper_estimand": "ACRT(d)",
                "identifying_assumption": "SPT + continuous dose support",
                "ordinary_pt_interpretation": (
                    "derivative of LATT(d|d) with local selection-bias contamination"
                ),
                "identification_note": (
                    "Ordinary PT is not enough for a causal ACRT(d) interpretation; "
                    "the public slope route reports the SPT-based causal-response label."
                ),
            },
        ),
        (
            "ATT(event_time)",
            {
                "paper_estimand": "ATT(event_time)",
                "identifying_assumption": "PT-MP",
                "ordinary_pt_interpretation": (
                    "post-treatment ATT(event_time); negative event-time cells are "
                    "pre-trend diagnostics"
                ),
                "identification_note": (
                    "Post-treatment ATT(event_time) cells are identified by "
                    "PT-MP/local binary event-study comparisons; negative event-time "
                    "cells diagnose pre-treatment parallel-trends plausibility rather "
                    "than treatment effects."
                ),
            },
        ),
        (
            "ACRT(event_time)",
            {
                "paper_estimand": "ACRT(event_time)",
                "identifying_assumption": "SPT-MP + continuous dose support",
                "ordinary_pt_interpretation": (
                    "derivative of event-time LATT path with local selection-bias "
                    "contamination under PT-MP alone"
                ),
                "identification_note": (
                    "The public slope event-study route reports the SPT-MP "
                    "causal-response label; under PT-MP alone, differentiating "
                    "event-time paths can retain selection-bias terms."
                ),
            },
        ),
    ],
)
def test_result_container_preserves_checked_identification_payloads(
    estimand: str,
    valid_identification: dict[str, str],
) -> None:
    from contdid import ContDIDResult

    is_eventstudy = "event_time" in estimand
    result = ContDIDResult(
        estimand=estimand,
        grid=[0],
        estimate=[0.4],
        std_error=[0.1],
        event_time=[0] if is_eventstudy else None,
        metadata={"identification": dict(valid_identification)},
    )

    assert result.metadata["identification"] == valid_identification


@pytest.mark.parametrize(
    ("identification", "message"),
    [
        ("SPT", "identification metadata must be a mapping"),
        (
            {"paper_estimand": "ATT(d)"},
            "identification metadata identifying_assumption must be a non-empty string",
        ),
        (
            {
                "paper_estimand": "ATT(d)",
                "identifying_assumption": "PT",
                "ordinary_pt_interpretation": "LATT(d|d)",
                "identification_note": (
                    "The same dose-specific contrast identifies LATT(d|d) under "
                    "ordinary PT; interpreting it as ATT(d) requires SPT."
                ),
            },
            "identification metadata must match the checked public estimand interpretation",
        ),
        (
            {
                "paper_estimand": "ATT(event_time)",
                "identifying_assumption": "PT-MP",
                "ordinary_pt_interpretation": (
                    "post-treatment ATT(event_time); negative event-time cells are "
                    "pre-trend diagnostics"
                ),
                "identification_note": (
                    "Post-treatment ATT(event_time) cells are identified by "
                    "PT-MP/local binary event-study comparisons; negative event-time "
                    "cells diagnose pre-treatment parallel-trends plausibility rather "
                    "than treatment effects."
                ),
            },
            "identification metadata must match the checked public estimand interpretation",
        ),
    ],
)
def test_result_container_rejects_identification_payload_drift(
    identification: object,
    message: str,
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match=message):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0],
            estimate=[0.4],
            std_error=[0.1],
            metadata={"identification": identification},
        )


def test_result_container_rejects_cohort_summary_standard_error_drift() -> None:
    from contdid import ContDIDResult

    with pytest.raises(
        ValueError,
        match="cohort_summary std_error must match result std_error",
    ):
        ContDIDResult(
            estimand="ATT(event_time)",
            grid=[0],
            estimate=[0.4],
            std_error=[0.2],
            event_time=[0],
            cohort_summary=[
                {
                    "event_time": 0,
                    "mean_estimate": 0.4,
                    "std_error": 0.1,
                    "support": False,
                }
            ],
        )


def test_result_container_exposes_readable_eventstudy_frame_with_support() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time_grid=[-1, 0, 1],
        confidence_interval=[[-0.5, 0.3], [0.0, 0.4], [0.1, 0.9]],
        cohort_summary=[
            {"event_time": -1, "support": True},
            {"event_time": 0, "support": True},
            {"event_time": 1, "support": False},
        ],
    )

    frame = result.to_frame()

    assert repr(result) == "ContDIDResult(estimand='ATT(event_time)', grid_size=3)"
    assert list(frame.columns) == [
        "event_time",
        "estimate",
        "std_error",
        "ci_lower",
        "ci_upper",
        "support",
    ]
    assert frame.to_dict(orient="list") == {
        "event_time": [-1, 0, 1],
        "estimate": [-0.1, 0.2, 0.5],
        "std_error": [0.2, 0.1, 0.3],
        "ci_lower": [-0.5, 0.0, 0.1],
        "ci_upper": [0.3, 0.4, 0.9],
        "support": [True, True, False],
    }
    assert result.to_markdown() == "\n".join(
        [
            "| Event time | Estimate | Std. error | Pointwise CI | Confidence band | Support |",
            "| ---: | ---: | ---: | --- | --- | --- |",
            "| -1 | -0.100000 | 0.200000 | [-0.500000, 0.300000] | not estimated (confidence band) | yes |",
            "| 0 | 0.200000 | 0.100000 | [0.000000, 0.400000] | not estimated (confidence band) | yes |",
            "| 1 | 0.500000 | 0.300000 | not estimated (local support) | not estimated (local support) | no |",
        ]
    )
    assert result.to_markdown(include_caption=True) == "\n".join(
        [
            "ContDIDResult: ATT(event_time), 3 rows, event_time axis, support 2/3 rows.",
            "",
            "| Event time | Estimate | Std. error | Pointwise CI | Confidence band | Support |",
            "| ---: | ---: | ---: | --- | --- | --- |",
            "| -1 | -0.100000 | 0.200000 | [-0.500000, 0.300000] | not estimated (confidence band) | yes |",
            "| 0 | 0.200000 | 0.100000 | [0.000000, 0.400000] | not estimated (confidence band) | yes |",
            "| 1 | 0.500000 | 0.300000 | not estimated (local support) | not estimated (local support) | no |",
        ]
    )
    assert result.to_markdown(digits=2) == "\n".join(
        [
            "| Event time | Estimate | Std. error | Pointwise CI | Confidence band | Support |",
            "| ---: | ---: | ---: | --- | --- | --- |",
            "| -1 | -0.10 | 0.20 | [-0.50, 0.30] | not estimated (confidence band) | yes |",
            "| 0 | 0.20 | 0.10 | [0.00, 0.40] | not estimated (confidence band) | yes |",
            "| 1 | 0.50 | 0.30 | not estimated (local support) | not estimated (local support) | no |",
        ]
    )


def test_result_to_markdown_compacts_long_eventstudy_tables_display_only() -> None:
    from contdid import ContDIDResult

    event_time = list(range(-5, 5))
    estimate = [value / 10 for value in event_time]
    confidence_interval = [[value - 0.2, value + 0.2] for value in estimate]
    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=event_time,
        estimate=estimate,
        std_error=[0.1] * len(event_time),
        event_time_grid=event_time,
        confidence_interval=confidence_interval,
        confidence_band={
            "lower": [value - 0.4 for value in estimate],
            "upper": [value + 0.4 for value in estimate],
            "critical_value": 2.5,
        },
        metadata={
            "confidence_band_kind": "simultaneous_multiplier",
            "inference_covariance": "full_event_time_covariance",
            "support": [
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
            ],
        },
    )

    compact = result.to_markdown(include_caption=True, digits=2, max_rows=4)

    assert compact == "\n".join(
        [
            (
                "ContDIDResult: ATT(event_time), 10 rows, event_time axis, "
                "showing 4/10 rows, critical value 2.50, "
                "full event-time covariance band, support 7/10 rows."
            ),
            "",
            "| Event time | Estimate | Std. error | Pointwise CI | Uniform band | Support |",
            "| ---: | ---: | ---: | --- | --- | --- |",
            "| -5 | -0.50 | 0.10 | [-0.70, -0.30] | [-0.90, -0.10] | yes |",
            "| -4 | -0.40 | 0.10 | [-0.60, -0.20] | [-0.80, 0.00] | yes |",
            "| ... 6 rows omitted (event times -3 to 2; support 4/6) ... | ... | ... | ... | ... | ... |",
            "| 3 | 0.30 | 0.10 | not estimated (local support) | not estimated (local support) | no |",
            "| 4 | 0.40 | 0.10 | [0.20, 0.60] | [0.00, 0.80] | yes |",
        ]
    )
    assert result.to_markdown(max_rows=None, digits=2) == result.to_markdown(digits=2)
    assert "... rows omitted ..." not in result.to_markdown(digits=2)
    assert len(result.to_frame()) == 10


def test_result_to_markdown_compacted_dose_tables_label_omitted_dose_range() -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.1 * value for value in range(1, 9)],
        estimate=[0.2 * value for value in range(1, 9)],
        std_error=[0.05] * 8,
    )

    compact = result.to_markdown(digits=1, max_rows=4)

    assert compact == "\n".join(
        [
            "| Dose | Estimate | Std. error | Pointwise CI | Confidence band |",
            "| ---: | ---: | ---: | --- | --- |",
            "| 0.1 | 0.2 | 0.1 | not estimated (pointwise CI) | not estimated (confidence band) |",
            "| 0.2 | 0.4 | 0.1 | not estimated (pointwise CI) | not estimated (confidence band) |",
            "| ... 4 rows omitted (dose values 0.3 to 0.6) ... | ... | ... | ... | ... |",
            "| 0.7 | 1.4 | 0.1 | not estimated (pointwise CI) | not estimated (confidence band) |",
            "| 0.8 | 1.6 | 0.1 | not estimated (pointwise CI) | not estimated (confidence band) |",
        ]
    )
    assert len(result.to_frame()) == 8


def test_result_display_rejects_mutated_nonboolean_support(
    tmp_path: Path,
) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time_grid=[-1, 0, 1],
        confidence_interval=[[-0.5, 0.3], [0.0, 0.4], [0.1, 0.9]],
        metadata={"support": [True, True, False]},
    )
    result.metadata["support"] = [True, "yes", False]
    output = tmp_path / "new-output-dir" / "mutated-support.png"

    with pytest.raises(
        ValueError,
        match="display support must contain only boolean values",
    ):
        result.to_frame()
    with pytest.raises(
        ValueError,
        match="display support must contain only boolean values",
    ):
        result.to_markdown()
    with pytest.raises(
        ValueError,
        match="display support must contain only boolean values",
    ):
        result.save_plot(output)
    assert not output.parent.exists()


def test_result_display_rejects_mutated_support_length(tmp_path: Path) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(event_time)",
        grid=[-1, 0, 1],
        estimate=[-0.1, 0.2, 0.5],
        std_error=[0.2, 0.1, 0.3],
        event_time_grid=[-1, 0, 1],
        metadata={"support": [True, True, False]},
    )
    result.metadata["support"] = [True, False]
    output = tmp_path / "new-output-dir" / "mutated-support-length.png"

    with pytest.raises(
        ValueError,
        match="display support must match result estimate length",
    ):
        result.to_frame()
    with pytest.raises(
        ValueError,
        match="display support must match result estimate length",
    ):
        result.to_markdown()
    with pytest.raises(
        ValueError,
        match="display support must match result estimate length",
    ):
        result.save_plot(output)
    assert not output.parent.exists()


@pytest.mark.parametrize("digits", [-1, 13, 1.5, True])
def test_result_to_markdown_rejects_invalid_digits(digits: object) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2],
        estimate=[0.4],
        std_error=[0.1],
    )

    with pytest.raises(ValueError, match="digits must be an integer between 0 and 12"):
        result.to_markdown(digits=digits)  # type: ignore[arg-type]


@pytest.mark.parametrize("include_caption", ["yes", 1, None])
def test_result_to_markdown_rejects_nonboolean_caption_switch(
    include_caption: object,
) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2],
        estimate=[0.4],
        std_error=[0.1],
    )

    with pytest.raises(ValueError, match="include_caption must be a boolean"):
        result.to_markdown(include_caption=include_caption)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_rows", [1, 0, -1, 1.5, True, "4"])
def test_result_to_markdown_rejects_invalid_max_rows(max_rows: object) -> None:
    from contdid import ContDIDResult

    result = ContDIDResult(
        estimand="ATT(d)",
        grid=[0.2],
        estimate=[0.4],
        std_error=[0.1],
    )

    with pytest.raises(
        ValueError,
        match="max_rows must be None or an integer greater than or equal to 2",
    ):
        result.to_markdown(max_rows=max_rows)  # type: ignore[arg-type]


def test_result_container_rejects_cohort_summary_on_dose_result() -> None:
    from contdid import ContDIDResult

    with pytest.raises(ValueError, match="cohort_summary requires event_time"):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0.2, 0.5],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            cohort_summary=[{"event_time": 0}, {"event_time": 1}],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_time": [0, 1]},
        {"event_time_grid": [0, 1]},
        {"timing_group": [2, 4]},
    ],
)
def test_result_container_rejects_event_time_fields_on_dose_result(
    kwargs: dict[str, object],
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(
        ValueError,
        match=(
            "event-time fields require event-study estimand|"
            "timing_group requires event-study estimand"
        ),
    ):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0, 1],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            **kwargs,
        )


def test_result_container_rejects_support_metadata_on_dose_result() -> None:
    from contdid import ContDIDResult

    with pytest.raises(
        ValueError,
        match="event-study metadata fields require event-study estimand: support",
    ):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0.2, 0.5],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            metadata={"support": [True, True]},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"event_time": [0, 1]},
        {"event_time_grid": [0, 1]},
        {"timing_group": [2]},
        {"cohort_summary": [{"event_time": 0}, {"event_time": 1}]},
        {"timing_group_support": {"timing_groups": [2]}},
    ],
)
def test_result_container_rejects_eventstudy_metadata_on_dose_result(
    metadata: dict[str, object],
) -> None:
    from contdid import ContDIDResult

    with pytest.raises(
        ValueError,
        match="event-study metadata fields require event-study estimand",
    ):
        ContDIDResult(
            estimand="ATT(d)",
            grid=[0.2, 0.5],
            estimate=[0.4, 0.8],
            std_error=[0.1, 0.2],
            metadata=metadata,
        )


def test_panel_data_from_records_preserves_column_override_contract() -> None:
    from contdid import PanelData, validate_panel_data

    records = (
        _make_valid_frame()
        .rename(
            columns={
                "id": "unit_id",
                "time_period": "period",
                "Y": "outcome",
                "G": "cohort",
                "D": "dose",
            }
        )
        .to_dict("records")
    )

    panel = PanelData.from_records(
        records,
        id_column="unit_id",
        time_column="period",
        outcome_column="outcome",
        group_column="cohort",
        dose_column="dose",
    )

    assert list(panel.frame.columns) == [
        "unit_id",
        "period",
        "outcome",
        "cohort",
        "dose",
    ]
    assert validate_panel_data(panel) is panel


def test_validate_panel_data_rejects_unbalanced_panel() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    frame = _make_valid_frame().drop(index=5)

    with pytest.raises(ContDIDValidationError, match="balanced"):
        validate_panel_data(PanelData(frame=frame.reset_index(drop=True)))


def test_validate_panel_data_rejects_positive_dose_for_never_treated_units() -> None:
    from contdid import ContDIDValidationError, PanelData, validate_panel_data

    frame = _make_valid_frame()
    frame.loc[frame["G"] == 0, "D"] = 0.4

    with pytest.raises(ContDIDValidationError, match="never-treated"):
        validate_panel_data(PanelData(frame=frame))


def test_validate_spec_rejects_unsupported_none_aggregation() -> None:
    from contdid import ContDIDSpec, ContDIDValidationError, validate_spec

    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="none",
        dose_est_method="parametric",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
    )

    with pytest.raises(ContDIDValidationError, match="aggregation"):
        validate_spec(spec)


def test_validate_spec_accepts_cck_eventstudy_combination() -> None:
    """CCK + eventstudy is now supported with fixed dimension."""
    from contdid import ContDIDSpec, validate_spec

    spec = ContDIDSpec(
        target_parameter="slope",
        aggregation="eventstudy",
        dose_est_method="cck",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=0,
    )

    validated = validate_spec(spec)
    assert validated.dose_est_method == "cck"
    assert validated.aggregation == "eventstudy"


def test_validate_spec_accepts_nonzero_anticipation_for_eventstudy() -> None:
    """Anticipation > 0 is now supported for event study (CGBS Assumption 3-MP(a))."""
    from contdid import ContDIDSpec, validate_spec

    spec = ContDIDSpec(
        target_parameter="level",
        aggregation="eventstudy",
        dose_est_method="parametric",
        control_group="notyettreated",
        treatment_type="continuous",
        anticipation=1,
    )

    validated = validate_spec(spec)
    assert validated.anticipation == 1
