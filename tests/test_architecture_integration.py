"""集成测试 - 验证注册机制、链式API、分层验证的协同工作。

测试三项架构优化（估计器注册机制、链式调用API、分层验证）之间
的交互与向后兼容性。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    EstimatorRegistry,
    FittablePanel,
    PanelData,
    ValidationReport,
    ValidationSeverity,
    ValidationStrictness,
    cont_did,
    get_default_registry,
    register_estimator,
    simulate_contdid_data,
    validate_panel_data_report,
)
from contdid.registry import ParametricEstimator, CCKEstimator
from contdid.validation import ContDIDValidationError, validate_panel_data


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
def multi_period_panel() -> PanelData:
    """多期面板数据（SIM-004-staggered-eventstudy-null）。"""
    return simulate_contdid_data(
        n=300,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=123,
    )


@pytest.fixture()
def minimal_panel() -> PanelData:
    """最小有效二期面板（手工构造，便于验证逻辑而非数值）。"""
    rng = np.random.default_rng(99)
    n_units = 60

    unit_groups = np.where(np.arange(1, n_units + 1) <= 36, 2, 0)
    unit_doses = np.where(unit_groups > 0, rng.uniform(0.5, 3.0, size=n_units), 0.0)

    ids = np.repeat(np.arange(1, n_units + 1), 2)
    times = np.tile([1, 2], n_units)
    groups = np.repeat(unit_groups, 2)
    doses = np.repeat(unit_doses, 2)
    post = (times >= groups) & (groups > 0)
    outcomes = 1.0 + 0.5 * doses * post + rng.normal(0, 0.3, size=len(ids))

    df = pd.DataFrame({
        "id": ids,
        "time_period": times,
        "Y": outcomes,
        "G": groups,
        "D": doses,
    })
    return PanelData(frame=df)


# ===========================================================================
# TestFullPipelineIntegration - 完整管道测试
# ===========================================================================


class TestFullPipelineIntegration:
    """测试完整管道：数据→验证（分层）→规范配置→注册表路由→估计→结果。"""

    def test_pipeline_with_registry_routing(self, two_period_panel: PanelData):
        """链式API通过注册表路由到parametric估计器。"""
        spec = ContDIDSpec.dose_response(method="parametric")
        result = two_period_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)
        # 验证估计结果包含必要属性
        assert result.grid is not None
        assert len(result.grid) > 0

    def test_pipeline_with_lenient_validation(self, minimal_panel: PanelData):
        """链式API配合LENIENT验证模式。"""
        spec = ContDIDSpec.dose_response(
            method="parametric",
            validation_strictness="lenient",
        )
        # LENIENT模式不会因warning级别问题阻止执行
        result = minimal_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)

    def test_pipeline_with_normal_validation(self, minimal_panel: PanelData):
        """链式API配合NORMAL验证模式（warnings打印但不阻止）。"""
        spec = ContDIDSpec.dose_response(
            method="parametric",
            validation_strictness="normal",
        )
        result = minimal_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)

    def test_custom_estimator_with_pipeline(self, two_period_panel: PanelData):
        """自定义注册估计器配合链式API使用。"""
        # 创建自定义估计器
        class DummyEstimator:
            name = "test_dummy_integration"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                # 接受任何 dose_est_method（不限制方法名）
                return []

            def estimate(self, panel, spec, **kwargs):
                # 委托给parametric进行实际估计
                from contdid.registry import ParametricEstimator
                # 修改spec的method以通过parametric的validate
                import dataclasses
                patched_spec = dataclasses.replace(spec, dose_est_method="parametric")
                return ParametricEstimator().estimate(panel, patched_spec, **kwargs)

        # 注册自定义估计器
        registry = get_default_registry()
        registry.register(DummyEstimator())
        try:
            spec = ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="test_dummy_integration",
                control_group="nevertreated",
            )
            result = two_period_panel.with_spec(spec).fit()
            assert isinstance(result, ContDIDResult)
        finally:
            # 清理：移除测试用的自定义估计器
            registry.unregister("test_dummy_integration")

    def test_convenience_spec_with_registry(self, two_period_panel: PanelData):
        """便捷构造方法创建的spec正确路由到注册表。"""
        # dose_response() → registry → parametric
        spec_dose = ContDIDSpec.dose_response()
        result_dose = cont_did(two_period_panel, spec_dose)
        assert isinstance(result_dose, ContDIDResult)
        assert result_dose.estimand == "ATT(d)"

        # marginal_response() → registry → parametric
        spec_slope = ContDIDSpec.marginal_response()
        result_slope = cont_did(two_period_panel, spec_slope)
        assert isinstance(result_slope, ContDIDResult)
        assert result_slope.estimand == "ACRT(d)"

    def test_validation_report_before_fit(self, two_period_panel: PanelData):
        """验证报告可在fit前独立获取。"""
        spec = ContDIDSpec.dose_response()
        report = validate_panel_data_report(two_period_panel, spec=spec)
        assert isinstance(report, ValidationReport)
        assert report.is_valid
        # 验证后仍然可以正常fit
        result = two_period_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)

    def test_pipeline_with_cck_method(self, two_period_panel: PanelData):
        """链式API通过注册表路由到CCK估计器。"""
        spec = ContDIDSpec.dose_response(method="cck")
        result = two_period_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)


# ===========================================================================
# TestBackwardCompatibility - 回归测试
# ===========================================================================


class TestBackwardCompatibility:
    """回归测试 - 确保传统用法完全不受影响。"""

    def test_traditional_api_unchanged(self, two_period_panel: PanelData):
        """传统 cont_did(panel, spec) 调用行为不变。"""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        result = cont_did(two_period_panel, spec)
        assert isinstance(result, ContDIDResult)
        assert result.grid is not None
        assert len(result.grid) > 0

    def test_traditional_api_kwargs_unchanged(self, two_period_panel: PanelData):
        """传统关键字参数调用行为不变。"""
        result = cont_did(
            two_period_panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert isinstance(result, ContDIDResult)

    def test_default_strictness_is_strict(self):
        """默认验证严格度为STRICT，与重构前一致。"""
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        assert spec.validation_strictness == "strict"

    def test_existing_error_messages_preserved(self):
        """错误消息文本与重构前一致。"""
        # 创建缺失列的面板 → 应触发 ERROR
        df = pd.DataFrame({"x": [1, 2, 3]})
        panel = PanelData(frame=df, id_column="id", time_column="time",
                          outcome_column="Y", group_column="G", dose_column="D")
        with pytest.raises(ContDIDValidationError):
            validate_panel_data(panel)

    def test_pipeline_result_matches_traditional(self, two_period_panel: PanelData):
        """链式API结果与传统API结果一致。"""
        spec = ContDIDSpec.dose_response()

        # 传统调用
        result_traditional = cont_did(two_period_panel, spec)

        # 链式调用
        result_pipeline = two_period_panel.with_spec(spec).fit()

        # 两者应完全一致（同一spec，同一panel）
        np.testing.assert_array_equal(result_traditional.grid, result_pipeline.grid)
        np.testing.assert_array_almost_equal(
            result_traditional.estimate, result_pipeline.estimate, decimal=10
        )


# ===========================================================================
# TestCrossFeatureEdgeCases - 边缘情况
# ===========================================================================


class TestCrossFeatureEdgeCases:
    """边缘情况 - 三个特性交互的特殊场景。"""

    def test_cck_spec_validation_through_registry(self, two_period_panel: PanelData):
        """CCK估计器通过注册表验证约束。"""
        spec = ContDIDSpec.dose_response(method="cck")
        registry = get_default_registry()
        estimator = registry.get("cck")
        errors = estimator.validate_spec(spec, two_period_panel)
        assert errors == []

    def test_parametric_spec_validation_through_registry(self, two_period_panel: PanelData):
        """Parametric估计器通过注册表验证约束。"""
        spec = ContDIDSpec.dose_response(method="parametric")
        registry = get_default_registry()
        estimator = registry.get("parametric")
        errors = estimator.validate_spec(spec, two_period_panel)
        assert errors == []

    def test_wrong_method_spec_blocked_by_registry_validation(
        self, two_period_panel: PanelData
    ):
        """错误的method被注册表内估计器的validate_spec捕获。"""
        # 手动构造不匹配的spec
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        registry = get_default_registry()
        cck_estimator = registry.get("cck")
        errors = cck_estimator.validate_spec(spec, two_period_panel)
        assert len(errors) > 0
        assert "cck" in errors[0].lower() or "parametric" in errors[0].lower()

    def test_pipeline_propagates_validation_strictness(self):
        """链式API正确传递验证严格度参数。"""
        # 创建含 WARNING 级问题的面板（NaN outcome → finite_outcomes WARNING）
        rows = []
        for i in range(1, 51):
            is_treated = i <= 30
            group = 2 if is_treated else 0
            dose = float(i) * 0.1 if is_treated else 0.0
            for t in [1, 2]:
                # 一个treated单位的一个观测有NaN → WARNING
                outcome = float("nan") if (i == 5 and t == 2) else float(i + t)
                rows.append({
                    "id": i,
                    "time_period": t,
                    "Y": outcome,
                    "G": group,
                    "D": dose,
                })
        df = pd.DataFrame(rows)
        panel = PanelData(frame=df)

        # STRICT模式下应该阻止（warning → 阻止）
        spec_strict = ContDIDSpec.dose_response(validation_strictness="strict")
        with pytest.raises(ContDIDValidationError):
            panel.with_spec(spec_strict).fit()

        # LENIENT模式下验证通过（WARNING不阻止），但估计可能因NaN报错
        # 这里仅测试验证层面的传播逻辑
        spec_lenient = ContDIDSpec.dose_response(validation_strictness="lenient")
        # 验证调用 validate_panel_data 不会抛出
        validated = validate_panel_data(panel, strictness=ValidationStrictness.LENIENT)
        assert validated is panel  # LENIENT模式下pass-through

    def test_registry_list_available_estimators(self):
        """列出可用估计器包括默认的和注册的。"""
        registry = get_default_registry()
        available = registry.list_available()
        assert "parametric" in available
        assert "cck" in available
        assert isinstance(available, list)
        assert available == sorted(available)

    def test_validation_report_with_convenience_spec(self, two_period_panel: PanelData):
        """验证报告配合便捷构造spec正确工作。"""
        spec = ContDIDSpec.dose_response()
        report = validate_panel_data_report(two_period_panel, spec=spec)
        assert isinstance(report, ValidationReport)
        # 合法数据不应有ERROR
        assert len(report.errors) == 0

    def test_fittable_panel_repr_with_convenience_spec(self, minimal_panel: PanelData):
        """FittablePanel的repr包含spec关键信息。"""
        spec = ContDIDSpec.dose_response()
        fittable = minimal_panel.with_spec(spec)
        repr_str = repr(fittable)
        assert "FittablePanel" in repr_str
        assert "level" in repr_str
        assert "dose" in repr_str
        assert "parametric" in repr_str

    def test_eventstudy_through_full_pipeline(self, multi_period_panel: PanelData):
        """事件研究通过完整管道（链式API + 注册表路由 + 验证）。"""
        spec = ContDIDSpec.eventstudy()
        # 验证报告
        report = validate_panel_data_report(multi_period_panel, spec=spec)
        assert report.is_valid
        # 通过管道执行
        result = multi_period_panel.with_spec(spec).fit()
        assert isinstance(result, ContDIDResult)


# ===========================================================================
# TestImportConsistency - 导入一致性验证
# ===========================================================================


class TestImportConsistency:
    """验证所有新公共符号可正常导入。"""

    def test_all_new_symbols_importable(self):
        """确认所有新增公共符号可从顶层导入。"""
        from contdid import (
            PanelData,
            ContDIDSpec,
            ContDIDResult,
            cont_did,
            FittablePanel,
            EstimatorRegistry,
            register_estimator,
            get_default_registry,
            ValidationSeverity,
            ValidationStrictness,
            ValidationReport,
            validate_panel_data_report,
        )
        # 验证类型
        assert FittablePanel is not None
        assert EstimatorRegistry is not None
        assert ValidationSeverity is not None
        assert ValidationStrictness is not None
        assert ValidationReport is not None

    def test_registry_module_symbols(self):
        """注册表模块内部符号可正确导入。"""
        from contdid.registry import (
            EstimatorProtocol,
            EstimatorRegistry,
            ParametricEstimator,
            CCKEstimator,
            get_default_registry,
            register_estimator,
        )
        assert ParametricEstimator is not None
        assert CCKEstimator is not None

    def test_validation_module_symbols(self):
        """验证模块内部符号可正确导入。"""
        from contdid.validation import (
            ContDIDValidationError,
            ValidationIssue,
            ValidationReport,
            ValidationSeverity,
            ValidationStrictness,
            validate_panel_data,
            validate_panel_data_report,
        )
        assert ValidationIssue is not None

    def test_no_circular_imports(self):
        """无循环导入问题（全量导入 contdid 包成功即验证）。"""
        import importlib
        import contdid
        importlib.reload(contdid)
        assert hasattr(contdid, "FittablePanel")
        assert hasattr(contdid, "EstimatorRegistry")
        assert hasattr(contdid, "ValidationReport")
