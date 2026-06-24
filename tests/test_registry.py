"""Tests for estimator registry mechanism.

验证：
- 默认注册表包含 "parametric" 和 "cck"
- 自定义估计器注册和调用
- 无效估计器名称的错误处理
- 估计器约束验证（如CCK不支持多期剂量聚合）
- 回归测试：注册表路由结果与直接调用一致
"""

from __future__ import annotations

import numpy as np
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    PanelData,
    cont_did,
    simulate_contdid_data,
    EstimatorRegistry,
    register_estimator,
    get_default_registry,
)
from contdid.registry import (
    EstimatorProtocol,
    ParametricEstimator,
    CCKEstimator,
    _default_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_period_panel() -> PanelData:
    """创建二期面板数据用于测试。"""
    return simulate_contdid_data(
        n=200,
        num_time_periods=2,
        num_groups=2,
        pg=[0.6],
        pu=0.4,
        dgp_id="SIM-002-linear-dose",
        seed=42,
    )


@pytest.fixture
def multi_period_panel() -> PanelData:
    """创建多期面板数据用于测试。"""
    return simulate_contdid_data(
        n=300,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=123,
    )


@pytest.fixture
def parametric_spec() -> ContDIDSpec:
    """参数化估计规范。"""
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="parametric",
        control_group="nevertreated",
    )


@pytest.fixture
def cck_spec() -> ContDIDSpec:
    """CCK 估计规范。"""
    return ContDIDSpec(
        target_parameter="level",
        aggregation="dose",
        dose_est_method="cck",
        control_group="nevertreated",
    )


# ---------------------------------------------------------------------------
# 默认注册表测试
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    """测试默认注册表包含正确的内置估计器。"""

    def test_contains_parametric(self) -> None:
        registry = get_default_registry()
        assert "parametric" in registry

    def test_contains_cck(self) -> None:
        registry = get_default_registry()
        assert "cck" in registry

    def test_list_available(self) -> None:
        registry = get_default_registry()
        available = registry.list_available()
        assert "parametric" in available
        assert "cck" in available

    def test_len(self) -> None:
        registry = get_default_registry()
        assert len(registry) >= 2

    def test_get_parametric(self) -> None:
        registry = get_default_registry()
        est = registry.get("parametric")
        assert est.name == "parametric"
        assert est.supports_multiperiod is True
        assert est.supports_adaptive is False

    def test_get_cck(self) -> None:
        registry = get_default_registry()
        est = registry.get("cck")
        assert est.name == "cck"
        assert est.supports_multiperiod is False
        assert est.supports_adaptive is True


# ---------------------------------------------------------------------------
# EstimatorRegistry 单元测试
# ---------------------------------------------------------------------------


class TestEstimatorRegistry:
    """测试 EstimatorRegistry 核心功能。"""

    def test_register_and_get(self) -> None:
        """测试自定义估计器注册和获取。"""
        registry = EstimatorRegistry()

        class MyEstimator:
            name = "my_method"
            supports_multiperiod = True
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                return ContDIDResult(
                    estimand="ATT(d)",
                    grid=[1.0, 2.0],
                    estimate=[0.5, 1.0],
                    std_error=[0.1, 0.2],
                    metadata={"method": "my_method"},
                )

        est = MyEstimator()
        registry.register(est)
        assert "my_method" in registry
        assert registry.get("my_method") is est
        assert "my_method" in registry.list_available()

    def test_register_duplicate_raises(self) -> None:
        """测试重复注册同名估计器应报错。"""
        registry = EstimatorRegistry()

        class Est:
            name = "dup"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                pass

        registry.register(Est())
        with pytest.raises(ValueError, match="已注册"):
            registry.register(Est())

    def test_register_invalid_missing_name(self) -> None:
        """测试缺少必需接口属性时报 TypeError。"""
        registry = EstimatorRegistry()

        class BadEstimator:
            supports_multiperiod = True
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                pass

        with pytest.raises(TypeError, match="缺少必需的接口成员"):
            registry.register(BadEstimator())

    def test_register_invalid_missing_method(self) -> None:
        """测试缺少 estimate 方法时报 TypeError。"""
        registry = EstimatorRegistry()

        class BadEstimator:
            name = "bad"
            supports_multiperiod = True
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

        with pytest.raises(TypeError, match="缺少必需的接口成员"):
            registry.register(BadEstimator())

    def test_get_unknown_raises(self) -> None:
        """测试获取不存在的估计器应报错并提示可用方法。"""
        registry = EstimatorRegistry()

        class Est:
            name = "only"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                pass

        registry.register(Est())
        with pytest.raises(KeyError, match="未知的估计方法.*nonexistent.*可用方法.*only"):
            registry.get("nonexistent")

    def test_unregister(self) -> None:
        """测试移除估计器。"""
        registry = EstimatorRegistry()

        class Est:
            name = "removable"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                pass

        registry.register(Est())
        assert "removable" in registry
        registry.unregister("removable")
        assert "removable" not in registry

    def test_unregister_unknown_raises(self) -> None:
        """测试移除不存在的估计器应报错。"""
        registry = EstimatorRegistry()
        with pytest.raises(KeyError, match="未注册"):
            registry.unregister("ghost")


# ---------------------------------------------------------------------------
# 约束验证测试
# ---------------------------------------------------------------------------


class TestEstimatorValidation:
    """测试估计器约束验证。"""

    def test_parametric_validates_ok(
        self, two_period_panel: PanelData, parametric_spec: ContDIDSpec
    ) -> None:
        est = ParametricEstimator()
        errors = est.validate_spec(parametric_spec, two_period_panel)
        assert errors == []

    def test_cck_validates_ok_two_period(
        self, two_period_panel: PanelData, cck_spec: ContDIDSpec
    ) -> None:
        est = CCKEstimator()
        errors = est.validate_spec(cck_spec, two_period_panel)
        assert errors == []

    def test_cck_multiperiod_dose_falls_back(self, multi_period_panel: PanelData) -> None:
        """CCK 对多期面板 dose 聚合自动回退到参数化路径（向后兼容）。"""
        est = CCKEstimator()
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
        )
        # validate_spec 不报错（回退逻辑在 estimate() 中处理）
        errors = est.validate_spec(spec, multi_period_panel)
        assert errors == []
        # estimate() 应回退到多期参数化路径
        result = est.estimate(multi_period_panel, spec, dvals=None, degree=3, num_knots=0)
        assert result.metadata.get("panel_type") == "multiperiod"

    def test_parametric_wrong_method(
        self, two_period_panel: PanelData
    ) -> None:
        """ParametricEstimator 拒绝非 parametric 方法。"""
        est = ParametricEstimator()
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
        )
        errors = est.validate_spec(spec, two_period_panel)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# 通过 cont_did() 的错误处理
# ---------------------------------------------------------------------------


class TestContDidRegistryErrors:
    """测试 cont_did() 通过注册表路由的错误处理。"""

    def test_unknown_method_raises(self, two_period_panel: PanelData) -> None:
        """通过 cont_did() 请求不存在的估计方法应报错。"""
        with pytest.raises(KeyError, match="未知的估计方法.*nonexistent"):
            cont_did(
                two_period_panel,
                dose_est_method="nonexistent",
            )

    def test_unsupported_aggregation_raises(self, two_period_panel: PanelData) -> None:
        """不支持的 aggregation 应在注册表路由前被拦截。"""
        from contdid.validation import ContDIDValidationError

        with pytest.raises(ContDIDValidationError, match="Unsupported aggregation"):
            cont_did(
                two_period_panel,
                aggregation="unknown_agg",
            )


# ---------------------------------------------------------------------------
# 回归测试：注册表路由结果与直接调用一致
# ---------------------------------------------------------------------------


class TestRegistryRegression:
    """回归测试：确保通过注册表路由的结果与直接调用完全一致。"""

    def test_parametric_two_period_dose(self, two_period_panel: PanelData) -> None:
        """参数化二期剂量结果通过两条路径应一致。"""
        from contdid.estimation import estimate_dose_level_effects

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            bstrap=False,
        )

        # 通过注册表路由
        result_registry = cont_did(two_period_panel, spec=spec, degree=3, num_knots=0)

        # 直接调用
        from contdid.validation import validate_panel_data

        validated = validate_panel_data(two_period_panel)
        result_direct = estimate_dose_level_effects(
            validated, spec, dvals=None, degree=3, num_knots=0
        )

        # 结果应完全一致
        np.testing.assert_array_almost_equal(
            result_registry.estimate, result_direct.estimate, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result_registry.std_error, result_direct.std_error, decimal=10
        )
        assert result_registry.grid == result_direct.grid

    def test_cck_two_period_dose(self, two_period_panel: PanelData) -> None:
        """CCK 二期剂量结果通过两条路径应一致。"""
        from contdid.estimation import estimate_dose_level_effects

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
            bstrap=False,
        )

        # 通过注册表路由
        result_registry = cont_did(two_period_panel, spec=spec)

        # 直接调用
        from contdid.validation import validate_panel_data

        validated = validate_panel_data(two_period_panel)
        result_direct = estimate_dose_level_effects(validated, spec)

        # 结果应完全一致
        np.testing.assert_array_almost_equal(
            result_registry.estimate, result_direct.estimate, decimal=10
        )
        np.testing.assert_array_almost_equal(
            result_registry.std_error, result_direct.std_error, decimal=10
        )


# ---------------------------------------------------------------------------
# 第三方扩展测试
# ---------------------------------------------------------------------------


class TestThirdPartyExtension:
    """测试第三方通过 register_estimator() 扩展。"""

    def test_register_and_call_custom_estimator(self, two_period_panel: PanelData) -> None:
        """注册自定义估计器并通过 cont_did() 调用。"""
        registry = get_default_registry()

        # 确保测试估计器未注册
        test_name = "_test_custom_est"
        if test_name in registry:
            registry.unregister(test_name)

        class CustomEstimator:
            name = "_test_custom_est"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                return ContDIDResult(
                    estimand="CUSTOM(d)",
                    grid=[1.0, 2.0, 3.0],
                    estimate=[0.1, 0.2, 0.3],
                    std_error=[0.01, 0.02, 0.03],
                    metadata={"custom": True},
                )

        try:
            registry.register(CustomEstimator())

            # 通过 cont_did() 调用
            spec = ContDIDSpec(
                target_parameter="level",
                aggregation="dose",
                dose_est_method="_test_custom_est",
                control_group="nevertreated",
            )
            result = cont_did(two_period_panel, spec=spec)
            assert result.estimand == "CUSTOM(d)"
            assert result.estimate == [0.1, 0.2, 0.3]
            assert result.metadata["custom"] is True
        finally:
            # 清理
            if test_name in registry:
                registry.unregister(test_name)
