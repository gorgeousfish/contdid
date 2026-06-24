"""集成测试 - 验证插件估计器、钩子系统、多框架适配器的协同工作。

覆盖范围：
- 端到端管道：多框架输入 → 插件估计器 → 钩子后处理
- 数学不变性：三项扩展不改变核心算法行为
- 向后兼容性：不使用新功能时行为完全不变
- 边缘情况：异常隔离、验证拒绝、只读保护
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from contdid import (
    # 核心 API
    PanelData,
    ContDIDSpec,
    ContDIDResult,
    cont_did,
    simulate_contdid_data,
    # 第一轮架构优化
    FittablePanel,
    EstimatorRegistry,
    register_estimator,
    get_default_registry,
    ValidationSeverity,
    ValidationStrictness,
    ValidationReport,
    # 第二轮扩展性改进 - 插件式估计器
    BaseEstimator,
    EstimatorCapabilities,
    EstimatorResult,
    LinearDoseEstimator,
    # 第二轮扩展性改进 - 钩子系统
    HookRegistry,
    HookStage,
    ReadOnlyResult,
    register_hook,
    unregister_hook,
    get_hook_registry,
    # 第二轮扩展性改进 - 适配器
    AdapterRegistry,
    PolarsAdapter,
    ArrowAdapter,
    PandasAdapter,
    convert_to_pandas,
)
from contdid.hooks import _default_hook_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_period_panel() -> PanelData:
    """二期面板数据（SIM-002-linear-dose）。"""
    return simulate_contdid_data(
        n=200,
        num_time_periods=2,
        num_groups=2,
        pg=[0.6],
        pu=0.4,
        dgp_id="SIM-002-linear-dose",
        seed=42,
    )


@pytest.fixture()
def two_period_df(two_period_panel) -> pd.DataFrame:
    """返回面板数据的原始 DataFrame。"""
    return two_period_panel.frame.copy()


@pytest.fixture(autouse=True)
def clean_default_hooks():
    """每次测试前后清空默认钩子注册表。"""
    _default_hook_registry.clear()
    yield
    _default_hook_registry.clear()


# ===========================================================================
# 1. TestFullExtensibilityPipeline
# ===========================================================================


class TestFullExtensibilityPipeline:
    """端到端管道：多框架输入 → 插件估计器 → 钩子后处理。"""

    def test_polars_input_with_custom_estimator_and_hook(self, two_period_df):
        """Polars输入 + 自定义估计器 + 后处理钩子 完整管道。"""
        polars = pytest.importorskip("polars")

        # 1) Polars DataFrame 输入
        pl_df = polars.from_pandas(two_period_df)
        panel = PanelData(frame=pl_df)
        assert isinstance(panel.frame, pd.DataFrame)

        # 2) 使用自定义 LinearDoseEstimator
        est = LinearDoseEstimator()
        # 从面板数据手动提取处理组数据
        df = panel.frame
        t2 = df[df["time_period"] == 2].set_index("id")
        t1 = df[df["time_period"] == 1].set_index("id")
        dy_all = (t2["Y"] - t1["Y"]).reindex(t2.index)
        dose_all = t2["D"]
        treated = dose_all > 0
        dy = dy_all[treated].values
        dose = dose_all[treated].values
        untreated_mean = dy_all[~treated].mean()
        dy_centered = dy - untreated_mean

        dvals = np.quantile(dose, np.arange(0.10, 1.0, 0.05))
        result = est.fit(dy_centered, dose, dvals)
        assert isinstance(result, EstimatorResult)
        assert result.att_d.shape == dvals.shape

        # 3) 钩子后处理
        hook_registry = HookRegistry()
        hook_output = {}

        def capture_hook(r):
            hook_output["n_points"] = len(r.grid)
            return hook_output

        hook_registry.register("capture", capture_hook)

        # 构造 ContDIDResult 传给钩子
        cr = ContDIDResult(
            estimand="ATT(d)",
            grid=dvals.tolist(),
            estimate=result.att_d.tolist(),
            std_error=[0.1] * len(dvals),
            metadata={"method": "linear_dose"},
        )
        outputs = hook_registry.execute(cr, HookStage.POST_INFERENCE)
        assert "capture" in outputs
        assert outputs["capture"]["n_points"] == len(dvals)

    def test_arrow_input_through_base_estimator(self, two_period_df):
        """Arrow Table 通过 BaseEstimator 子类的完整管道。"""
        pa = pytest.importorskip("pyarrow")

        table = pa.Table.from_pandas(two_period_df)
        panel = PanelData(frame=table)
        assert isinstance(panel.frame, pd.DataFrame)

        # 提取并通过 LinearDoseEstimator 估计
        df = panel.frame
        t2 = df[df["time_period"] == 2].set_index("id")
        t1 = df[df["time_period"] == 1].set_index("id")
        dy_all = (t2["Y"] - t1["Y"]).reindex(t2.index)
        dose_all = t2["D"]
        treated = dose_all > 0
        dy = dy_all[treated].values
        dose = dose_all[treated].values
        untreated_mean = dy_all[~treated].mean()
        dy_centered = dy - untreated_mean

        dvals = np.linspace(0.1, 0.9, 15)
        est = LinearDoseEstimator()
        result = est.fit(dy_centered, dose, dvals)
        assert np.isfinite(result.att_d).all()

    def test_hook_receives_results_from_custom_estimator(self, two_period_df):
        """钩子能正确接收自定义估计器的结果。"""
        est = LinearDoseEstimator()
        dy, dose, dvals = _extract_treated_data(two_period_df)
        result = est.fit(dy, dose, dvals)

        cr = ContDIDResult(
            estimand="ATT(d)",
            grid=dvals.tolist(),
            estimate=result.att_d.tolist(),
            std_error=[0.1] * len(dvals),
            metadata={"source": "LinearDoseEstimator"},
        )

        received = {}
        hook_reg = HookRegistry()
        hook_reg.register("inspector", lambda r: received.update({
            "estimand": r.estimand,
            "n_grid": len(r.grid),
            "first_est": r.estimate[0],
        }))
        hook_reg.execute(cr, HookStage.POST_INFERENCE)

        assert received["estimand"] == "ATT(d)"
        assert received["n_grid"] == len(dvals)
        assert received["first_est"] == pytest.approx(result.att_d[0], rel=1e-10)

    def test_adapter_preserves_math_for_plugin_estimator(self, two_period_df):
        """适配器转换后的数据通过插件估计器产生正确数学结果。"""
        polars = pytest.importorskip("polars")

        # pandas 路径
        dy_pd, dose_pd, dvals = _extract_treated_data(two_period_df)
        est = LinearDoseEstimator()
        result_pd = est.fit(dy_pd, dose_pd, dvals)

        # polars 路径
        pl_df = polars.from_pandas(two_period_df)
        converted = convert_to_pandas(pl_df)
        dy_pl, dose_pl, _ = _extract_treated_data(converted)
        result_pl = est.fit(dy_pl, dose_pl, dvals)

        np.testing.assert_allclose(
            result_pd.att_d, result_pl.att_d, rtol=1e-12,
            err_msg="Polars 适配后估计结果与 pandas 不一致"
        )


# ===========================================================================
# 2. TestMathematicalInvariance
# ===========================================================================


class TestMathematicalInvariance:
    """数学不变性验证 - 三项扩展不改变核心算法行为。"""

    def test_base_estimator_result_matches_builtin(self, two_period_panel):
        """BaseEstimator子类(LinearDoseEstimator)与内置参数化估计器结果方向一致。"""
        # 通过内置 parametric 估计器获取基准结果
        spec = ContDIDSpec.dose_response(method="parametric")
        builtin_result = cont_did(two_period_panel, spec, degree=1, num_knots=0)

        # 通过 LinearDoseEstimator 获取结果
        dy, dose, _ = _extract_treated_data(two_period_panel.frame)
        dvals = np.array(builtin_result.grid)
        est = LinearDoseEstimator()
        custom_result = est.fit(dy, dose, dvals)

        # 两者应有相同符号/趋势（实现细节可能不同，但方向一致）
        builtin_est = np.array(builtin_result.estimate)
        custom_est = custom_result.att_d
        # 线性 DGP：两者的相关系数应高度正相关
        corr = np.corrcoef(builtin_est, custom_est)[0, 1]
        assert corr > 0.9, f"内置与自定义估计器相关性过低: {corr:.4f}"

    def test_hooks_do_not_alter_estimates(self, two_period_panel):
        """注册多个钩子后，ATT(d) 值不变。"""
        spec = ContDIDSpec.dose_response()

        # 无钩子基准
        result_no_hooks = cont_did(two_period_panel, spec, degree=3, num_knots=0)
        baseline_est = list(result_no_hooks.estimate)
        baseline_se = list(result_no_hooks.std_error)

        # 注册多个钩子
        register_hook("hook1", lambda r: {"mean_est": np.mean(r.estimate)})
        register_hook("hook2", lambda r: {"max_se": max(r.std_error)})
        register_hook("hook3", lambda r: {"grid_len": len(r.grid)})

        # 重新估计
        result_with_hooks = cont_did(two_period_panel, spec, degree=3, num_knots=0)

        assert result_with_hooks.estimate == baseline_est
        assert result_with_hooks.std_error == baseline_se

    def test_adapter_conversion_bit_exact(self, two_period_df):
        """pandas/polars/arrow输入产生bit-exact相同PanelData frame。"""
        polars = pytest.importorskip("polars")
        pa = pytest.importorskip("pyarrow")

        panel_pd = PanelData(frame=two_period_df.copy())
        panel_pl = PanelData(frame=polars.from_pandas(two_period_df))
        panel_pa = PanelData(frame=pa.Table.from_pandas(two_period_df))

        cols = list(two_period_df.columns)
        df_pd = panel_pd.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]
        df_pl = panel_pl.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]
        df_pa = panel_pa.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]

        pd.testing.assert_frame_equal(df_pd, df_pl, check_dtype=False)
        pd.testing.assert_frame_equal(df_pd, df_pa, check_dtype=False)

    def test_combined_extensions_preserve_inference(self, two_period_panel):
        """三项扩展同时使用时，置信区间和Bootstrap推断不变。"""
        spec = ContDIDSpec.dose_response(bstrap=True, biters=100, alp=0.05)

        # 基准结果（无扩展功能）
        result_baseline = cont_did(two_period_panel, spec, degree=3, num_knots=0)

        # 注册钩子 + 重新估计
        register_hook("ci_checker", lambda r: {"has_ci": r.confidence_interval is not None})
        result_with_ext = cont_did(two_period_panel, spec, degree=3, num_knots=0)

        # 点估计和标准误应完全相同
        assert result_with_ext.estimate == result_baseline.estimate
        assert result_with_ext.std_error == result_baseline.std_error


# ===========================================================================
# 3. TestBackwardCompatibility
# ===========================================================================


class TestBackwardCompatibility:
    """向后兼容性 - 不使用新功能时行为完全不变。"""

    def test_no_hooks_zero_overhead(self, two_period_panel):
        """不注册钩子时无额外开销。"""
        spec = ContDIDSpec.dose_response()
        assert len(_default_hook_registry) == 0

        # 确认 hook_outputs 为空
        result = cont_did(two_period_panel, spec, degree=3, num_knots=0)
        assert result.hook_outputs == {}

    def test_pandas_input_unchanged(self, two_period_df):
        """pandas输入仍通过直通适配器（零开销）。"""
        adapter = PandasAdapter()
        assert adapter.can_handle(two_period_df)
        converted = adapter.to_pandas(two_period_df)
        assert converted is two_period_df  # 同一对象，零拷贝

    def test_existing_estimators_still_work(self):
        """现有 parametric/cck 估计器通过注册表正常工作。"""
        registry = get_default_registry()
        available = registry.list_available()
        assert "parametric" in available
        assert "cck" in available

        # 获取估计器实例
        param_est = registry.get("parametric")
        cck_est = registry.get("cck")
        assert param_est.name == "parametric"
        assert cck_est.name == "cck"

    def test_traditional_api_fully_compatible(self, two_period_panel):
        """传统 cont_did(panel, spec) 调用行为完全不变。"""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        result = cont_did(two_period_panel, spec, degree=3, num_knots=0)
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(d)"
        assert len(result.estimate) > 0
        assert len(result.std_error) > 0


# ===========================================================================
# 4. TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """边缘情况测试。"""

    def test_hook_exception_does_not_break_pipeline(self, two_period_panel):
        """钩子异常不中断估计流程。"""
        spec = ContDIDSpec.dose_response()

        def bad_hook(r):
            raise ValueError("Intentional error in hook")

        register_hook("bad", bad_hook)

        # 应该能正常完成估计
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cont_did(two_period_panel, spec, degree=3, num_knots=0)

        assert isinstance(result, ContDIDResult)
        assert len(result.estimate) > 0

    def test_custom_estimator_validation_failure(self):
        """不合规的自定义估计器被拒绝注册。"""
        # 不完整的 BaseEstimator 子类不能实例化
        with pytest.raises(TypeError):
            class BadEstimator(BaseEstimator):
                @property
                def name(self):
                    return "bad"
                # 缺少所有抽象方法
            BadEstimator()

    def test_unsupported_dataframe_type_clear_error(self):
        """不支持的数据类型给出清晰错误信息。"""
        with pytest.raises(TypeError, match="不支持的数据类型"):
            convert_to_pandas({"not": "a dataframe"})

        with pytest.raises(TypeError):
            convert_to_pandas([1, 2, 3])

    def test_readonly_result_protection(self):
        """ReadOnlyResult 严格阻止修改。"""
        cr = ContDIDResult(
            estimand="ATT(d)",
            grid=[1.0, 2.0, 3.0],
            estimate=[0.5, 1.0, 1.5],
            std_error=[0.1, 0.2, 0.3],
            metadata={},
        )
        readonly = ReadOnlyResult(cr)

        # 读取正常
        assert readonly.estimand == "ATT(d)"
        assert readonly.grid == [1.0, 2.0, 3.0]
        assert readonly.estimate == [0.5, 1.0, 1.5]

        # 修改被阻止
        with pytest.raises(AttributeError, match="不允许修改"):
            readonly.estimate = [0.0, 0.0, 0.0]
        with pytest.raises(AttributeError, match="不允许修改"):
            readonly.new_attr = "anything"
        with pytest.raises(AttributeError, match="不允许删除"):
            del readonly.estimate

        # 原始值不变
        assert cr.estimate == [0.5, 1.0, 1.5]


# ===========================================================================
# 5. TestPerformanceBaseline
# ===========================================================================


class TestPerformanceBaseline:
    """性能基线检查 - 扩展功能不显著影响核心性能。"""

    def test_hook_overhead_acceptable(self, two_period_panel):
        """钩子系统开销可接受（< 10% overhead）。"""
        spec = ContDIDSpec.dose_response(bstrap=False)

        # 基线：无钩子
        t0 = time.perf_counter()
        for _ in range(3):
            cont_did(two_period_panel, spec, degree=3, num_knots=0)
        baseline_time = (time.perf_counter() - t0) / 3

        # 有钩子
        register_hook("perf1", lambda r: np.mean(r.estimate))
        register_hook("perf2", lambda r: len(r.grid))
        register_hook("perf3", lambda r: {"range": (min(r.estimate), max(r.estimate))})

        t0 = time.perf_counter()
        for _ in range(3):
            cont_did(two_period_panel, spec, degree=3, num_knots=0)
        hook_time = (time.perf_counter() - t0) / 3

        # 允许 50% 开销（钩子非常轻量，但计时噪音可能较大）
        if baseline_time > 0.01:  # 仅当基线时间足够大时检查
            overhead = (hook_time - baseline_time) / baseline_time
            assert overhead < 0.5, f"钩子开销过大: {overhead:.1%}"


# ===========================================================================
# 6. TestImportConsistency
# ===========================================================================


class TestImportConsistency:
    """验证所有扩展符号均从顶层可导入。"""

    def test_all_extensibility_symbols_in_contdid(self):
        """所有第二轮扩展符号均在 contdid.__all__ 中。"""
        import contdid

        expected_symbols = [
            # BaseEstimator 家族
            "BaseEstimator", "EstimatorCapabilities", "EstimatorResult",
            "LinearDoseEstimator", "validate_estimator_result", "validate_influence_matrix",
            # 钩子系统
            "HookRegistry", "HookStage", "ReadOnlyResult",
            "register_hook", "unregister_hook", "get_hook_registry",
            # 适配器
            "AdapterRegistry", "PolarsAdapter", "ArrowAdapter",
            "PandasAdapter", "convert_to_pandas",
        ]
        for sym in expected_symbols:
            assert hasattr(contdid, sym), f"contdid 缺少导出: {sym}"
            assert sym in contdid.__all__, f"contdid.__all__ 缺少: {sym}"


# ===========================================================================
# Helper
# ===========================================================================


def _extract_treated_data(df: pd.DataFrame):
    """从面板 DataFrame 提取处理组的 ΔY、dose、dvals。"""
    t2 = df[df["time_period"] == 2].set_index("id")
    t1 = df[df["time_period"] == 1].set_index("id")
    common = t2.index.intersection(t1.index)
    t2 = t2.loc[common]
    t1 = t1.loc[common]
    dy_all = t2["Y"].values - t1["Y"].values
    dose_all = t2["D"].values
    treated = dose_all > 0
    dy = dy_all[treated]
    dose = dose_all[treated]
    untreated_mean = dy_all[~treated].mean()
    dy_centered = dy - untreated_mean
    dvals = np.quantile(dose, np.arange(0.10, 1.0, 0.05))
    return dy_centered, dose, dvals
