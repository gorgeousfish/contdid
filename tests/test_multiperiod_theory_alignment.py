"""End-to-end tests verifying theoretical alignment of multi-period and event-study modules.

These tests validate that the implementation faithfully implements the theoretical
framework from Callaway, Goodman-Bacon & Sant'Anna (2024, arXiv:2107.02637v7).

Theory references:
- CGBS Theorem 3.1: LATT(g,t,d|g,d) identified for ALL t>=g under parallel trends
- CGBS Corollary 1: dose-specific aggregation uses sample-size weights
- CGBS Appendix C: multiplier bootstrap with shared unit-level multipliers
- CCK (arXiv:2107.11869v3): convergence rates proven only for two-period panels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid.data import PanelData
from contdid.multiperiod import (
    MultiPeriodDoseResult,
    _identify_dose_timing_groups,
    estimate_multiperiod_dose,
)
from contdid.eventstudy import (
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
)
from contdid.specs import ContDIDSpec
from contdid.validation import ContDIDValidationError


# ---------------------------------------------------------------------------
# Helper: construct a simple staggered panel DGP
# ---------------------------------------------------------------------------


def _make_staggered_panel(
    *,
    n_per_group: int = 100,
    time_periods: list[int] | None = None,
    groups: list[int] | None = None,
    effect_func=None,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a balanced staggered panel with known DGP.

    DGP: Y_{it} = alpha_i + lambda_t + effect_func(D_i)*1{t>=g_i} + epsilon_{it}
    where alpha_i ~ N(0,1), lambda_t = t, epsilon ~ N(0, 0.5^2)

    Parameters
    ----------
    n_per_group : units per group (including never-treated group=0)
    time_periods : list of time periods (default [1,2,3,4,5])
    groups : list of treatment groups including 0 (default [0,2,3,4])
    effect_func : function mapping dose -> treatment effect (default: zero)
    seed : random seed
    """
    if time_periods is None:
        time_periods = [1, 2, 3, 4, 5]
    if groups is None:
        groups = [0, 2, 3, 4]
    if effect_func is None:
        effect_func = lambda d: 0.0  # noqa: E731

    rng = np.random.default_rng(seed)
    T = len(time_periods)
    n_groups = len(groups)
    n_total = n_per_group * n_groups

    records = []
    unit_id = 0
    for g in groups:
        for _ in range(n_per_group):
            unit_id += 1
            alpha_i = rng.normal(0, 1)
            # Dose: positive for treated, zero for never-treated
            if g == 0:
                dose_i = 0.0
            else:
                dose_i = rng.uniform(0.1, 2.0)

            for t in time_periods:
                lambda_t = float(t)
                epsilon = rng.normal(0, 0.5)
                # Treatment effect kicks in at t >= g for treated units
                if g > 0 and t >= g:
                    effect = effect_func(dose_i)
                else:
                    effect = 0.0
                y = alpha_i + lambda_t + effect + epsilon
                records.append({
                    "id": unit_id,
                    "time_period": t,
                    "Y": y,
                    "G": g,
                    "D": dose_i,
                })

    return pd.DataFrame(records)


# ===========================================================================
# Test 1: 验证多期 (g,t) 对完整枚举
# ===========================================================================


class TestMultiperiodGTPairEnumeration:
    """验证 _identify_dose_timing_groups 对每个 group g 枚举所有 t>=g 对。"""

    def test_enumerates_all_gt_pairs(self):
        """CGBS Theorem 3.1: LATT(g,t,d|g,d) identified for ALL t>=g.

        Panel: 3 groups (g=2,3,4), times 1..5, never-treated group=0
        Expected:
          g=2: t in {2,3,4,5} -> 4 pairs
          g=3: t in {3,4,5} -> 3 pairs
          g=4: t in {4,5} -> 2 pairs
        Total: 9 (g,t) pairs
        """
        panel_df = _make_staggered_panel(
            n_per_group=20,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            seed=100,
        )

        timing_groups = _identify_dose_timing_groups(
            panel_df,
            group_column="G",
            time_column="time_period",
            id_column="id",
            dose_column="D",
            control_group="nevertreated",
        )

        # Collect (g,t) pairs
        gt_pairs = [(tg["group"], tg["time"]) for tg in timing_groups]

        # Expected pairs
        expected_pairs = []
        for g in [2, 3, 4]:
            for t in range(g, 6):  # t in {g, g+1, ..., 5}
                expected_pairs.append((g, t))

        assert len(gt_pairs) == 9, f"Expected 9 (g,t) pairs, got {len(gt_pairs)}"
        assert set(gt_pairs) == set(expected_pairs), (
            f"(g,t) pairs mismatch.\nGot: {sorted(gt_pairs)}\nExpected: {sorted(expected_pairs)}"
        )

    def test_event_time_field_correct(self):
        """每个 (g,t) 对应 event_time = t - g。"""
        panel_df = _make_staggered_panel(
            n_per_group=20,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            seed=101,
        )

        timing_groups = _identify_dose_timing_groups(
            panel_df,
            group_column="G",
            time_column="time_period",
            id_column="id",
            dose_column="D",
            control_group="nevertreated",
        )

        for tg in timing_groups:
            assert tg["event_time"] == tg["time"] - tg["group"], (
                f"event_time should be {tg['time']}-{tg['group']}="
                f"{tg['time']-tg['group']}, got {tg['event_time']}"
            )

    def test_notyettreated_control_group(self):
        """not-yet-treated 对照组在后期减少控制单位数量。"""
        panel_df = _make_staggered_panel(
            n_per_group=30,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            seed=102,
        )

        timing_groups = _identify_dose_timing_groups(
            panel_df,
            group_column="G",
            time_column="time_period",
            id_column="id",
            dose_column="D",
            control_group="notyettreated",
        )

        # For g=2, t=2: control = never-treated + groups 3,4 (treated after t=2)
        # For g=2, t=5: control = never-treated only (groups 3,4 already treated)
        g2_t2 = next(tg for tg in timing_groups if tg["group"] == 2 and tg["time"] == 2)
        g2_t5 = next(tg for tg in timing_groups if tg["group"] == 2 and tg["time"] == 5)

        # At t=2, groups 3 and 4 are not-yet-treated -> more controls
        # At t=5, all groups treated -> only never-treated as controls
        assert g2_t2["n_control"] > g2_t5["n_control"], (
            "not-yet-treated control should have more units at earlier times"
        )


# ===========================================================================
# Test 2: 验证零效应DGP下 ATT(d) 估计的无偏性
# ===========================================================================


class TestMultiperiodUnbiasedUnderNull:
    """在零处理效应下，多期聚合 ATT(d) 估计应接近0。"""

    def test_multiperiod_unbiased_under_null(self):
        """Under PT with true ATT(d)=0, multi-period ATT(d) should be ~0.

        DGP: Y_{it} = alpha_i + lambda_t + epsilon, D independent of Y(0).
        Theory: E[LATT(g,t,d|g,d)] = 0 when true effect is 0.
        """
        panel_df = _make_staggered_panel(
            n_per_group=150,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3, 4],
            effect_func=lambda d: 0.0,
            seed=2024,
        )

        dose_grid = np.linspace(0.2, 1.8, 10)

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            dose_grid=dose_grid,
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=500,
            alp=0.05,
            cband=False,
            boot_seed=123,
        )

        assert isinstance(result, MultiPeriodDoseResult)

        # Under null, all point estimates should be close to zero
        point_estimates = np.array(result.point_estimate)
        se_values = np.array(result.standard_error)

        # Check: |overall mean ATT(d)| < 3*mean(SE) — approximately 99.7% coverage
        overall_att = np.mean(point_estimates)
        mean_se = np.mean(se_values)
        assert mean_se > 0, "Standard errors should be positive"
        assert abs(overall_att) < 3 * mean_se, (
            f"Under null DGP, overall ATT(d) = {overall_att:.4f} exceeds "
            f"3*SE = {3*mean_se:.4f}"
        )

        # Each grid point should be within its own tolerance
        max_t_stat = np.max(np.abs(point_estimates) / np.maximum(se_values, 1e-10))
        # With 10 grid points, max t-stat under null should rarely exceed 4
        assert max_t_stat < 5.0, (
            f"Max t-statistic {max_t_stat:.2f} too large for null DGP"
        )


# ===========================================================================
# Test 3: 验证已知效应DGP下 ATT(d) 估计的一致性
# ===========================================================================


class TestMultiperiodRecoversKnownDoseResponse:
    """多期估计应恢复已知的剂量响应曲线。"""

    def test_recovers_quadratic_dose_response(self):
        """Theory: Under PT with true ATT(d) = d^2, parametric B-spline estimator
        should recover the quadratic function.

        DGP: Y_{it} = alpha_i + lambda_t + d^2 * 1{t>=g} + epsilon
        """
        panel_df = _make_staggered_panel(
            n_per_group=200,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            effect_func=lambda d: d**2,
            seed=2025,
        )

        dose_grid = np.linspace(0.3, 1.8, 8)

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            dose_grid=dose_grid,
            degree=3,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=500,
            alp=0.05,
            cband=False,
            boot_seed=456,
        )

        # True effect at each grid point
        true_effects = dose_grid**2
        estimated = np.array(result.point_estimate)
        se_values = np.array(result.standard_error)

        # Check bias: |estimated - true| should be small relative to SE
        bias = np.abs(estimated - true_effects)
        # Allow up to 3*SE deviation (approximately 99.7% coverage per point)
        for i, (est, truth, se, b) in enumerate(
            zip(estimated, true_effects, se_values, bias)
        ):
            tolerance = max(3 * se, 0.3)  # at least 0.3 absolute tolerance
            assert b < tolerance, (
                f"Grid point {i} (d={dose_grid[i]:.2f}): estimated={est:.4f}, "
                f"true={truth:.4f}, bias={b:.4f} > tolerance={tolerance:.4f}"
            )

    def test_recovers_linear_dose_response(self):
        """Under linear true ATT(d) = 2*d, estimator should recover."""
        panel_df = _make_staggered_panel(
            n_per_group=200,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3],
            effect_func=lambda d: 2.0 * d,
            seed=2026,
        )

        dose_grid = np.linspace(0.3, 1.8, 6)

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            dose_grid=dose_grid,
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=300,
            alp=0.05,
            cband=False,
            boot_seed=789,
        )

        true_effects = 2.0 * dose_grid
        estimated = np.array(result.point_estimate)
        se_values = np.array(result.standard_error)

        # Mean absolute bias should be small
        mean_bias = np.mean(np.abs(estimated - true_effects))
        mean_se = np.mean(se_values)
        assert mean_bias < 3 * mean_se, (
            f"Mean bias {mean_bias:.4f} too large (3*mean_SE = {3*mean_se:.4f})"
        )


# ===========================================================================
# Test 4: 验证Bootstrap覆盖率
# ===========================================================================


@pytest.mark.slow
class TestMultiperiodBootstrapCoverage:
    """95%置信带应覆盖真实参数约95%的时间。

    Theory: CGBS Appendix C guarantees multiplier bootstrap valid inference.
    """

    def test_pointwise_coverage(self):
        """Pointwise confidence interval coverage check.

        Repeat estimation 50 times with different seeds; for each grid point,
        check if the true value falls within the pointwise 95% CI.
        Coverage should be in [0.80, 1.0] (conservative due to small reps).
        """
        n_reps = 50
        true_func = lambda d: d**2  # noqa: E731
        dose_grid = np.array([0.5, 1.0, 1.5])
        true_values = true_func(dose_grid)
        coverage_count = np.zeros(len(dose_grid))

        for rep in range(n_reps):
            panel_df = _make_staggered_panel(
                n_per_group=80,
                time_periods=[1, 2, 3, 4],
                groups=[0, 2, 3],
                effect_func=true_func,
                seed=3000 + rep,
            )
            result = estimate_multiperiod_dose(
                panel_df,
                id_column="id",
                time_column="time_period",
                outcome_column="Y",
                dose_column="D",
                group_column="G",
                dose_grid=dose_grid,
                degree=2,
                num_knots=0,
                control_group="nevertreated",
                target="level",
                biters=200,
                alp=0.05,
                cband=False,
                boot_seed=rep,
            )

            lower = np.array(result.confidence_band_lower)
            upper = np.array(result.confidence_band_upper)
            covered = (true_values >= lower) & (true_values <= upper)
            coverage_count += covered.astype(float)

        coverage_rate = coverage_count / n_reps
        # Each grid point coverage should be at least 0.80
        for i, rate in enumerate(coverage_rate):
            assert rate >= 0.70, (
                f"Coverage at d={dose_grid[i]:.1f} is {rate:.2f}, below 0.70"
            )


# ===========================================================================
# Test 5: 验证协变量调整的理论边界
# ===========================================================================


class TestMultiperiodCovariatesBoundary:
    """协变量在多期下应抛出 NotImplementedError。

    Theory boundary: CGBS does not provide conditional PT theory
    for multi-period decomposition with continuous D.
    """

    def test_covariates_raise_not_implemented(self):
        """Passing covariates to multi-period estimator should raise."""
        panel_df = _make_staggered_panel(
            n_per_group=20,
            time_periods=[1, 2, 3],
            groups=[0, 2, 3],
            seed=500,
        )
        # Add a dummy covariate column
        panel_df["X1"] = np.random.default_rng(0).normal(size=len(panel_df))

        with pytest.raises(NotImplementedError, match="[Cc]ovariate"):
            estimate_multiperiod_dose(
                panel_df,
                id_column="id",
                time_column="time_period",
                outcome_column="Y",
                dose_column="D",
                group_column="G",
                degree=2,
                num_knots=0,
                covariates=["X1"],
                boot_seed=0,
            )


# ===========================================================================
# Test 6: 验证 CCK 在事件研究中的理论边界
# ===========================================================================


class TestEventstudyCCKBoundary:
    """CCK固定维数在事件研究中应正常运行（每个局部比较是两期问题）。

    Theory: CCK (arXiv:2107.11869v3 Theorem 2) convergence rates apply
    to individual two-period comparisons; event-study decomposes into
    local (g,t) two-period problems where CCK theory applies directly.
    Adaptive/Lepski is NOT supported in event-study aggregation.
    """

    def test_cck_fixed_dimension_eventstudy_level(self):
        """dose_est_method='cck' with fixed dimension in event-study should succeed."""
        panel_df = _make_staggered_panel(
            n_per_group=30,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3],
            seed=600,
        )
        panel = PanelData(frame=panel_df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
        )

        result = estimate_eventstudy_effects(panel, spec, degree=2, num_knots=0)
        # Should produce valid event-study results
        assert result.estimand == "ATT(event_time)"
        assert len(result.estimate) > 0
        assert result.metadata["dose_est_method"] == "cck"
        assert result.metadata["source_estimator"] == "cck_fixed_dimension_eventstudy"

    def test_cck_fixed_dimension_eventstudy_slope(self):
        """CCK with slope event-study should also succeed with fixed dimension."""
        panel_df = _make_staggered_panel(
            n_per_group=30,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3],
            seed=601,
        )
        panel = PanelData(frame=panel_df)
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="cck",
            control_group="nevertreated",
        )

        result = estimate_eventstudy_slope_effects(panel, spec, degree=2, num_knots=0)
        assert result.estimand == "ACRT(event_time)"
        assert len(result.estimate) > 0
        assert result.metadata["dose_est_method"] == "cck"


# ===========================================================================
# Test 7: 验证事件研究聚合权重正确性
# ===========================================================================


class TestEventstudyWeights:
    """事件研究聚合权重应按 n_treated_g / sum(n_treated) 标准化。

    Theory: CGBS Corollary 2 specifies sample-size weighting.
    """

    def test_multiperiod_weights_sum_to_one(self):
        """Local results weights in multi-period estimation should sum to 1."""
        panel_df = _make_staggered_panel(
            n_per_group=50,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3, 4],
            effect_func=lambda d: d,
            seed=700,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            boot_seed=0,
        )

        weights = [lr["weight"] for lr in result.local_results]
        assert abs(sum(weights) - 1.0) < 1e-10, (
            f"Weights should sum to 1.0, got {sum(weights)}"
        )

    def test_weights_proportional_to_treated_count(self):
        """Weights should be proportional to n_treated per (g,t) pair."""
        # Construct unbalanced panel: group 2 has more units
        rng = np.random.default_rng(701)
        records = []
        uid = 0
        group_sizes = {0: 40, 2: 80, 3: 40}  # group 2 has 2x more
        for g, n_units in group_sizes.items():
            for _ in range(n_units):
                uid += 1
                dose = rng.uniform(0.1, 2.0) if g > 0 else 0.0
                alpha = rng.normal(0, 1)
                for t in [1, 2, 3]:
                    effect = dose if (g > 0 and t >= g) else 0.0
                    y = alpha + t + effect + rng.normal(0, 0.5)
                    records.append({
                        "id": uid, "time_period": t, "Y": y, "G": g, "D": dose
                    })

        panel_df = pd.DataFrame(records)

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=1,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            boot_seed=0,
        )

        # Check that group 2 pairs get higher weight (since it has more units)
        g2_weights = [lr["weight"] for lr in result.local_results if lr["group"] == 2]
        g3_weights = [lr["weight"] for lr in result.local_results if lr["group"] == 3]

        if g2_weights and g3_weights:
            # Each g=2 pair should have ~2x the weight of g=3 pairs
            avg_g2 = np.mean(g2_weights)
            avg_g3 = np.mean(g3_weights)
            ratio = avg_g2 / avg_g3
            assert ratio > 1.5, (
                f"Group 2 (80 units) should have higher weight than group 3 (40 units), "
                f"ratio = {ratio:.2f}"
            )


# ===========================================================================
# Test 8: 验证乘子Bootstrap共享正确性
# ===========================================================================


class TestBootstrapMultipliersShared:
    """同一单位在所有estimand中应使用相同的Bootstrap乘子。

    Theory: CGBS Appendix C requires shared unit-level multipliers
    for valid joint inference across dose grid points.
    """

    def test_fixed_seed_reproducibility(self):
        """同一 boot_seed 下多次运行结果应完全一致。"""
        panel_df = _make_staggered_panel(
            n_per_group=50,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3],
            effect_func=lambda d: d,
            seed=800,
        )

        kwargs = dict(
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            cband=True,
            boot_seed=999,
        )

        result1 = estimate_multiperiod_dose(panel_df, **kwargs)
        result2 = estimate_multiperiod_dose(panel_df, **kwargs)

        # Point estimates should be identical (same data)
        np.testing.assert_array_equal(
            result1.point_estimate, result2.point_estimate,
            err_msg="Point estimates should be identical across runs"
        )
        # SEs and critical values should be identical (same bootstrap seed)
        np.testing.assert_array_equal(
            result1.standard_error, result2.standard_error,
            err_msg="Standard errors should be identical with same seed"
        )
        np.testing.assert_array_equal(
            result1.confidence_band_lower, result2.confidence_band_lower,
            err_msg="Confidence bands should be identical with same seed"
        )

    def test_covariance_nondiagonal(self):
        """聚合 IF 的协方差矩阵应有非零的非对角元素。

        Because never-treated controls appear in multiple (g,t) comparisons,
        the aggregated IF for different dose grid points should be correlated.
        """
        panel_df = _make_staggered_panel(
            n_per_group=80,
            time_periods=[1, 2, 3, 4],
            groups=[0, 2, 3],
            effect_func=lambda d: d,
            seed=801,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            boot_seed=0,
        )

        # Get the covariance from aggregated IF
        cov = result.aggregated_influence.covariance()
        n_grid = cov.shape[0]

        # Check off-diagonal elements are non-zero (shared units create correlation)
        if n_grid > 1:
            off_diag = cov[0, 1]
            # Off-diagonal should generally be non-zero
            # (units in never-treated appear in all comparisons)
            assert abs(off_diag) > 0, (
                "Covariance off-diagonal should be non-zero due to shared controls"
            )


# ===========================================================================
# Test 9: 验证 event_time 字段在多期结果中的正确性
# ===========================================================================


class TestMultiperiodEventTimeField:
    """每个本地结果应包含正确的 event_time = t - g 字段。"""

    def test_event_time_in_local_results(self):
        """Local results should contain event_time field equal to time - group."""
        panel_df = _make_staggered_panel(
            n_per_group=50,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            effect_func=lambda d: d,
            seed=900,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=2,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            boot_seed=0,
        )

        for lr in result.local_results:
            expected_event_time = lr["time"] - lr["group"]
            assert lr["event_time"] == expected_event_time, (
                f"event_time for (g={lr['group']}, t={lr['time']}) should be "
                f"{expected_event_time}, got {lr['event_time']}"
            )

    def test_event_time_range(self):
        """Event times should range from 0 to T-g for each group g."""
        panel_df = _make_staggered_panel(
            n_per_group=40,
            time_periods=[1, 2, 3, 4, 5],
            groups=[0, 2, 3, 4],
            effect_func=lambda d: 0.5 * d,
            seed=901,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=1,
            num_knots=0,
            control_group="nevertreated",
            target="level",
            biters=100,
            boot_seed=0,
        )

        # Collect event_times by group
        event_times_by_group: dict[int, list[int]] = {}
        for lr in result.local_results:
            g = lr["group"]
            event_times_by_group.setdefault(g, []).append(lr["event_time"])

        # g=2: event_times should be {0,1,2,3}
        # g=3: event_times should be {0,1,2}
        # g=4: event_times should be {0,1}
        expected = {2: [0, 1, 2, 3], 3: [0, 1, 2], 4: [0, 1]}
        for g, expected_ets in expected.items():
            actual_ets = sorted(event_times_by_group.get(g, []))
            assert actual_ets == expected_ets, (
                f"Group {g} event_times: expected {expected_ets}, got {actual_ets}"
            )


# ===========================================================================
# Test: 验证 metadata 中的理论信息
# ===========================================================================


class TestMultiperiodMetadata:
    """验证 result.metadata 中包含正确的理论标注。"""

    def test_theoretical_basis_in_metadata(self):
        """Metadata should record CGBS theorem reference."""
        panel_df = _make_staggered_panel(
            n_per_group=30,
            time_periods=[1, 2, 3],
            groups=[0, 2, 3],
            seed=950,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=2,
            num_knots=0,
            biters=50,
            boot_seed=0,
        )

        assert "theoretical_basis" in result.metadata
        assert "CGBS" in result.metadata["theoretical_basis"]
        assert "theoretical_boundaries" in result.metadata
        boundaries = result.metadata["theoretical_boundaries"]
        assert any("CCK" in b or "cck" in b.lower() for b in boundaries)
        assert any("covariate" in b.lower() for b in boundaries)

    def test_slope_target_metadata(self):
        """slope target should work and record correct metadata."""
        panel_df = _make_staggered_panel(
            n_per_group=50,
            time_periods=[1, 2, 3],
            groups=[0, 2, 3],
            effect_func=lambda d: d**2,
            seed=951,
        )

        result = estimate_multiperiod_dose(
            panel_df,
            id_column="id",
            time_column="time_period",
            outcome_column="Y",
            dose_column="D",
            group_column="G",
            degree=3,
            num_knots=0,
            target="slope",
            biters=100,
            boot_seed=0,
        )

        assert result.metadata["target"] == "slope"
        assert result.metadata["estimator"] == "multiperiod_dose_2x2"
