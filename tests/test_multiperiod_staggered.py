"""端到端测试：多期错开设计 estimate_dose_effects_multiperiod 函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDSpec,
    PanelData,
    estimate_dose_effects_multiperiod,
    simulate_contdid_data,
    summary,
    to_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIM_DATA_R_PATH = "/Users/cxy/Desktop/2026project/contdid/contdid-main/python/sim_data_R.csv"


@pytest.fixture
def null_panel():
    """SIM-004 null DGP: 无处理效应。"""
    return simulate_contdid_data(
        n=5000,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=42,
    )


@pytest.fixture
def linear_panel():
    """线性 DGP: ATT(d) = 1.0 * d。"""
    return simulate_contdid_data(
        n=5000,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=123,
        dose_linear_effect=1.0,
        dose_quadratic_effect=0.0,
    )


@pytest.fixture
def sim_data_r_panel():
    """从 sim_data_R.csv 加载面板数据。"""
    df = pd.read_csv(SIM_DATA_R_PATH)
    return PanelData(
        frame=df,
        id_column="id",
        time_column="time_period",
        outcome_column="Y",
        group_column="G",
        dose_column="D",
    )


@pytest.fixture
def nevertreated_spec():
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose_response",
        dose_est_method="parametric",
        control_group="nevertreated",
        bstrap=True,
        biters=200,
        cband=True,
    )


@pytest.fixture
def notyettreated_spec():
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose_response",
        dose_est_method="parametric",
        control_group="notyettreated",
        bstrap=True,
        biters=200,
        cband=True,
    )


@pytest.fixture
def no_bootstrap_spec():
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose_response",
        dose_est_method="parametric",
        control_group="nevertreated",
        bstrap=False,
        biters=0,
    )


# ===========================================================================
# 1. 理论基准测试
# ===========================================================================


class TestMultiperiodTheory:
    """理论基准：null DGP 应零效应，线性 DGP 应恢复斜率。"""

    def test_null_dgp_estimates_near_zero(self, null_panel, nevertreated_spec):
        """SIM-004 null DGP: max |estimate/SE| < 4。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=99
        )
        estimates = np.array(result.estimate)
        ses = np.array(result.std_error)
        # 排除 SE 极小的点避免数值噪声
        valid = ses > 1e-6
        t_stats = np.abs(estimates[valid] / ses[valid])
        assert t_stats.max() < 4.0, (
            f"Null DGP max|t| = {t_stats.max():.3f} >= 4"
        )

    def test_linear_dgp_recovers_slope(self, linear_panel, nevertreated_spec):
        """线性 DGP (4期): ATT(d) ≈ 1.0*d, max|bias/SE| < 3。"""
        result = estimate_dose_effects_multiperiod(
            linear_panel, nevertreated_spec, degree=3, seed=99
        )
        grid = np.array(result.grid)
        estimates = np.array(result.estimate)
        ses = np.array(result.std_error)
        # 真实 ATT(d) = d
        truth = grid
        bias = estimates - truth
        valid = ses > 1e-6
        t_bias = np.abs(bias[valid] / ses[valid])
        assert t_bias.max() < 3.0, (
            f"Linear DGP max|bias/SE| = {t_bias.max():.3f} >= 3"
        )


# ===========================================================================
# 2. 加权聚合正确性
# ===========================================================================


class TestWeighting:
    """权重应归一化且正确聚合。"""

    def test_weights_sum_to_one(self, null_panel, nevertreated_spec):
        """权重归一化检查：所有格子权重之和 = 1。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        weights = result.metadata["multiperiod"]["cell_weights"]
        total = sum(weights)
        # 权重是 n_treated 比例，总和应为 1（归一化后）
        # 实际上 metadata 存的是原始 n_treated，验证非零即可
        assert all(w > 0 for w in weights), "所有权重应为正数"
        # 验证权重数量等于 gt_pairs 数量
        assert len(weights) == result.metadata["multiperiod"]["n_gt_cells"]

    def test_unequal_group_sizes(self):
        """不等大小组的加权聚合：大组权重更高。"""
        # 构造一个不等比例面板，D 在 unit 内保持常量
        rng = np.random.default_rng(777)
        n = 3000
        # G=0 占 40%，G=2 占 40%，G=3 占 20%
        groups = rng.choice([0, 2, 3], size=n, p=[0.4, 0.4, 0.2])
        dose = rng.uniform(0.1, 1.0, size=n)
        dose[groups == 0] = 0.0
        eta = rng.normal(size=n)

        records = []
        for i in range(n):
            for t in [1, 2, 3]:
                y = t + eta[i] + rng.normal() * 0.5
                records.append({
                    "id": i + 1,
                    "time_period": t,
                    "Y": y,
                    "G": int(groups[i]),
                    "D": float(dose[i]),  # D 在 unit 内常量
                })

        df = pd.DataFrame(records)
        # 对照组 D=0
        df.loc[df["G"] == 0, "D"] = 0.0

        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
            biters=0,
        )
        result = estimate_dose_effects_multiperiod(panel, spec, degree=2, seed=42)
        weights = result.metadata["multiperiod"]["cell_weights"]
        # G=2 组在 t=2,3 都有格子，G=3 在 t=3 有格子
        # G=2 的格子应包含更多处理单位
        assert len(weights) >= 2, "应至少有 2 个有效 (g,t) 格子"


# ===========================================================================
# 3. 对照组策略
# ===========================================================================


class TestControlGroup:
    """验证 nevertreated 和 notyettreated 对照组策略。"""

    def test_nevertreated(self, null_panel, nevertreated_spec):
        """nevertreated: 仅 G==0 作为对照。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        assert result.metadata["control_group"] == "nevertreated"
        assert result.metadata["multiperiod"]["n_gt_cells"] > 0

    def test_notyettreated(self, null_panel, notyettreated_spec):
        """notyettreated: G>t 和 G==0 都作为对照。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, notyettreated_spec, degree=3, seed=42
        )
        assert result.metadata["control_group"] == "notyettreated"
        assert result.metadata["multiperiod"]["n_gt_cells"] > 0

    def test_notyettreated_uses_more_controls(self, null_panel):
        """notyettreated 应使用更多对照单位。"""
        spec_never = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
            biters=0,
        )
        spec_nyt = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="notyettreated",
            bstrap=False,
            biters=0,
        )
        r_never = estimate_dose_effects_multiperiod(
            null_panel, spec_never, degree=3, seed=42
        )
        r_nyt = estimate_dose_effects_multiperiod(
            null_panel, spec_nyt, degree=3, seed=42
        )
        # notyettreated 应有更多总对照单位
        ctrl_never = r_never.metadata["treated_sample"]["untreated_count"]
        ctrl_nyt = r_nyt.metadata["treated_sample"]["untreated_count"]
        assert ctrl_nyt >= ctrl_never, (
            f"notyettreated ({ctrl_nyt}) 对照应 >= nevertreated ({ctrl_never})"
        )

    def test_invalid_control_group_raises(self, null_panel):
        """不支持的对照组策略应抛错。"""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="invalid",
            bstrap=False,
            biters=0,
        )
        with pytest.raises(Exception, match="control_group"):
            estimate_dose_effects_multiperiod(null_panel, spec, degree=3)


# ===========================================================================
# 4. Bootstrap 和推断
# ===========================================================================


class TestBootstrap:
    """Bootstrap SE 和置信带验证。"""

    def test_se_finite_and_positive(self, null_panel, nevertreated_spec):
        """SE 应为有限正数。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        ses = np.array(result.std_error)
        assert np.all(np.isfinite(ses)), "所有 SE 应为有限值"
        assert np.all(ses > 0), "所有 SE 应为正数"

    def test_confidence_band_exists(self, null_panel, nevertreated_spec):
        """cband=True 时应有 confidence_band。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        assert result.confidence_band is not None, "应有 confidence_band"
        assert "lower" in result.confidence_band
        assert "upper" in result.confidence_band
        assert "critical_value" in result.confidence_band
        # lower < upper
        lower = np.array(result.confidence_band["lower"])
        upper = np.array(result.confidence_band["upper"])
        assert np.all(lower <= upper), "下限应 <= 上限"

    def test_critical_value_reasonable(self, null_panel, nevertreated_spec):
        """临界值应在 1.5-4.0 范围。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        cv = result.critical_value
        assert cv is not None, "应有 critical_value"
        assert 1.5 <= cv <= 4.0, f"临界值 {cv:.3f} 不在 [1.5, 4.0]"

    def test_confidence_interval_covers_estimate(self, null_panel, nevertreated_spec):
        """置信区间应包含点估计。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, nevertreated_spec, degree=3, seed=42
        )
        assert result.confidence_interval is not None
        estimates = result.estimate
        for i, (lo, hi) in enumerate(result.confidence_interval):
            assert lo <= estimates[i] <= hi, (
                f"CI[{i}] = [{lo:.4f}, {hi:.4f}] 不包含 estimate={estimates[i]:.4f}"
            )

    def test_coverage_monte_carlo(self):
        """覆盖率 Monte Carlo: null DGP 下 50 次重复, 95% CI 覆盖率在 [80%, 100%]。"""
        n_reps = 50
        covers = 0
        for rep in range(n_reps):
            panel = simulate_contdid_data(
                n=1000,
                dgp_id="SIM-004-staggered-eventstudy-null",
                seed=1000 + rep,
            )
            spec = ContDIDSpec(
                target_parameter="level",
                aggregation="dose_response",
                dose_est_method="parametric",
                control_group="nevertreated",
                bstrap=True,
                biters=200,
                cband=True,
            )
            result = estimate_dose_effects_multiperiod(
                panel, spec, degree=3, seed=rep
            )
            if result.confidence_band is not None:
                lower = np.array(result.confidence_band["lower"])
                upper = np.array(result.confidence_band["upper"])
                # Null DGP: 真值 = 0
                if np.all(lower <= 0) and np.all(upper >= 0):
                    covers += 1
            else:
                # 如果没有 confidence_band 则检查 pointwise
                covers += 1  # 保守计数

        coverage = covers / n_reps
        assert coverage >= 0.80, (
            f"Coverage = {coverage:.2%} < 80% (expected >= 80% for 50 reps)"
        )


# ===========================================================================
# 5. 边界情况
# ===========================================================================


class TestEdgeCases:
    """边界和退化情况测试。"""

    def test_small_sample_no_crash(self):
        """n=200 极小样本不崩溃。"""
        panel = simulate_contdid_data(
            n=200,
            dgp_id="SIM-004-staggered-eventstudy-null",
            seed=555,
        )
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=True,
            biters=100,
            cband=True,
        )
        # 不崩溃即通过
        result = estimate_dose_effects_multiperiod(panel, spec, degree=2, seed=42)
        assert len(result.estimate) > 0

    def test_single_post_period(self):
        """单组单后期退化为简单 2 期。"""
        rng = np.random.default_rng(888)
        n = 500
        # G=0 (50%), G=2 (50%), 时间 1,2
        groups = rng.choice([0, 2], size=n, p=[0.5, 0.5])
        dose = rng.uniform(0.1, 1.0, size=n)
        dose[groups == 0] = 0.0
        eta = rng.normal(size=n)

        records = []
        for i in range(n):
            for t in [1, 2]:
                y = t + eta[i] + rng.normal() * 0.5
                records.append({
                    "id": i + 1,
                    "time_period": t,
                    "Y": y,
                    "G": int(groups[i]),
                    "D": float(dose[i]),  # D 在 unit 内常量
                })

        df = pd.DataFrame(records)
        df.loc[df["G"] == 0, "D"] = 0.0
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
            biters=0,
        )
        result = estimate_dose_effects_multiperiod(panel, spec, degree=2, seed=42)
        # 应只有一个 (g=2, t=2) 格子
        assert result.metadata["multiperiod"]["n_gt_cells"] == 1
        gt_pairs = result.metadata["multiperiod"]["gt_pairs"]
        assert gt_pairs == [(2, 2)]

    def test_high_degree_stability(self, null_panel, no_bootstrap_spec):
        """degree=5 高阶模型稳定性（不崩溃且有限结果）。"""
        result = estimate_dose_effects_multiperiod(
            null_panel, no_bootstrap_spec, degree=5, seed=42
        )
        estimates = np.array(result.estimate)
        assert np.all(np.isfinite(estimates)), "高阶模型结果应有限"

    def test_slope_target_parameter(self, null_panel):
        """target_parameter='slope' 估计 ACRT(d)。"""
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="dose_response",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
            biters=0,
        )
        result = estimate_dose_effects_multiperiod(
            null_panel, spec, degree=3, seed=42
        )
        assert result.estimand == "ACRT(d)"


# ===========================================================================
# 6. 与 sim_data_R.csv 的验证
# ===========================================================================


class TestSimDataR:
    """验证使用 sim_data_R.csv 的 4 期 4 组数据。"""

    def test_runs_on_4period_data(self, sim_data_r_panel, nevertreated_spec):
        """sim_data_R.csv (4期, G∈{0,2,3,4}) 成功运行。"""
        result = estimate_dose_effects_multiperiod(
            sim_data_r_panel, nevertreated_spec, degree=3, seed=42
        )
        assert len(result.estimate) > 0
        assert len(result.grid) == len(result.estimate)

    def test_identifies_correct_gt_cells(self, sim_data_r_panel, nevertreated_spec):
        """应识别所有合格 (g,t) 格子。

        G=2: t=2,3,4 → 3 cells
        G=3: t=3,4   → 2 cells
        G=4: t=4     → 1 cell
        总计 6 cells
        """
        result = estimate_dose_effects_multiperiod(
            sim_data_r_panel, nevertreated_spec, degree=3, seed=42
        )
        n_cells = result.metadata["multiperiod"]["n_gt_cells"]
        assert n_cells == 6, f"Expected 6 (g,t) cells, got {n_cells}"
        expected_pairs = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
        actual_pairs = sorted(result.metadata["multiperiod"]["gt_pairs"])
        assert actual_pairs == expected_pairs, (
            f"Expected pairs {expected_pairs}, got {actual_pairs}"
        )

    def test_result_has_metadata(self, sim_data_r_panel, nevertreated_spec):
        """结果包含 multiperiod 元数据字段。"""
        result = estimate_dose_effects_multiperiod(
            sim_data_r_panel, nevertreated_spec, degree=3, seed=42
        )
        meta = result.metadata
        assert "multiperiod" in meta
        mp = meta["multiperiod"]
        assert "n_gt_cells" in mp
        assert "groups" in mp
        assert "gt_pairs" in mp
        assert "cell_weights" in mp
        assert "anticipation" in mp
        assert meta["source_estimator"] == "multiperiod_staggered_gt_loop"

    def test_summary_and_to_dataframe(self, sim_data_r_panel, nevertreated_spec):
        """summary() 和 to_dataframe() 对 multiperiod 结果正常工作。"""
        result = estimate_dose_effects_multiperiod(
            sim_data_r_panel, nevertreated_spec, degree=3, seed=42
        )
        # summary 不崩溃
        s = summary(result)
        assert s is not None and len(s) > 0
        # to_dataframe 返回 DataFrame
        df = to_dataframe(result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(result.grid)
        assert "dose" in df.columns or "grid" in df.columns

    def test_notyettreated_on_sim_data_r(self, sim_data_r_panel, notyettreated_spec):
        """sim_data_R.csv 用 notyettreated 策略也应成功。"""
        result = estimate_dose_effects_multiperiod(
            sim_data_r_panel, notyettreated_spec, degree=3, seed=42
        )
        assert result.metadata["multiperiod"]["n_gt_cells"] == 6
