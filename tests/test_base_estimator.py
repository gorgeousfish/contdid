"""测试 BaseEstimator 抽象基类及插件注册机制。

覆盖范围：
- BaseEstimator 不可直接实例化
- 正确实现子类的注册和调用
- 不完整实现（缺少方法）的错误处理
- EstimatorResult 结构验证
- LinearDoseEstimator 教学示例
- capabilities 声明与验证
- 回归测试：现有 Protocol 方式仍然工作
"""

from __future__ import annotations

import numpy as np
import pytest

from contdid.base_estimator import (
    BaseEstimator,
    EstimatorCapabilities,
    EstimatorResult,
    LinearDoseEstimator,
    validate_estimator_result,
    validate_influence_matrix,
)
from contdid.registry import (
    EstimatorRegistry,
    _BaseEstimatorAdapter,
    get_default_registry,
    register_estimator,
)


# ---------------------------------------------------------------------------
# 辅助：生成简单的测试数据
# ---------------------------------------------------------------------------


def _make_test_data(n: int = 100, seed: int = 42):
    """生成线性剂量响应测试数据。

    DGP: ΔY_i = 2 + 3*D_i + ε_i, D_i ~ Uniform(0.5, 5)
    ATT(d) = 2 + 3*d
    """
    rng = np.random.default_rng(seed)
    dose = rng.uniform(0.5, 5.0, size=n)
    dy = 2.0 + 3.0 * dose + rng.normal(0, 0.5, size=n)
    dvals = np.linspace(1.0, 4.0, 20)
    return dy, dose, dvals


# ===========================================================================
# 1. BaseEstimator 抽象类基本行为
# ===========================================================================


class TestBaseEstimatorAbstract:
    """测试 BaseEstimator 不可直接实例化。"""

    def test_cannot_instantiate_directly(self):
        """BaseEstimator 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseEstimator()

    def test_incomplete_subclass_raises(self):
        """缺少抽象方法的子类不能实例化。"""

        class IncompleteEstimator(BaseEstimator):
            @property
            def name(self) -> str:
                return "incomplete"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities()

            # 缺少 fit, influence_function, validate_assumptions

        with pytest.raises(TypeError):
            IncompleteEstimator()

    def test_partial_subclass_missing_fit(self):
        """只实现部分方法的子类也不能实例化。"""

        class PartialEstimator(BaseEstimator):
            @property
            def name(self) -> str:
                return "partial"

            @property
            def capabilities(self) -> EstimatorCapabilities:
                return EstimatorCapabilities()

            def influence_function(self, dy, dose, dvals, coefficients, **kwargs):
                return np.zeros((len(dy), len(dvals)))

            def validate_assumptions(self, panel, spec):
                return []

            # 缺少 fit

        with pytest.raises(TypeError):
            PartialEstimator()


# ===========================================================================
# 2. EstimatorResult 结构验证
# ===========================================================================


class TestEstimatorResult:
    """测试 EstimatorResult 数据类。"""

    def test_valid_construction(self):
        """正常构造 EstimatorResult。"""
        dvals = np.array([1.0, 2.0, 3.0])
        result = EstimatorResult(
            att_d=np.array([0.5, 1.0, 1.5]),
            acrt_d=None,
            coefficients=np.array([0.5, 0.5]),
            dvals=dvals,
            basis_info={"degree": 1},
        )
        assert result.att_d.shape == (3,)
        assert result.dvals.shape == (3,)
        assert result.acrt_d is None

    def test_att_dvals_length_mismatch(self):
        """att_d 与 dvals 长度不一致应报错。"""
        with pytest.raises(ValueError, match="att_d.*dvals.*一致"):
            EstimatorResult(
                att_d=np.array([1.0, 2.0]),
                acrt_d=None,
                coefficients=np.array([1.0]),
                dvals=np.array([1.0, 2.0, 3.0]),
                basis_info={},
            )

    def test_acrt_dvals_length_mismatch(self):
        """acrt_d 与 dvals 长度不一致应报错。"""
        with pytest.raises(ValueError, match="acrt_d.*dvals.*一致"):
            EstimatorResult(
                att_d=np.array([1.0, 2.0, 3.0]),
                acrt_d=np.array([0.5, 0.5]),  # 长度 2 != 3
                coefficients=np.array([1.0]),
                dvals=np.array([1.0, 2.0, 3.0]),
                basis_info={},
            )

    def test_auto_coercion_to_float(self):
        """输入应自动转为 float ndarray。"""
        result = EstimatorResult(
            att_d=[1, 2, 3],
            acrt_d=None,
            coefficients=[1, 2],
            dvals=[1, 2, 3],
            basis_info={},
        )
        assert result.att_d.dtype == np.float64
        assert result.coefficients.dtype == np.float64


# ===========================================================================
# 3. validate_estimator_result 和 validate_influence_matrix
# ===========================================================================


class TestValidationHelpers:
    """测试验证辅助函数。"""

    def test_validate_estimator_result_pass(self):
        """合法结果通过验证。"""
        result = EstimatorResult(
            att_d=np.array([1.0, 2.0, 3.0]),
            acrt_d=None,
            coefficients=np.array([0.5, 0.5]),
            dvals=np.array([1.0, 2.0, 3.0]),
            basis_info={},
        )
        errors = validate_estimator_result(result, n_obs=100, n_dvals=3)
        assert errors == []

    def test_validate_estimator_result_wrong_type(self):
        """非 EstimatorResult 类型应报错。"""
        errors = validate_estimator_result("not a result", n_obs=100, n_dvals=3)
        assert len(errors) == 1
        assert "EstimatorResult" in errors[0]

    def test_validate_estimator_result_nan(self):
        """含 NaN 的结果应报错。"""
        result = EstimatorResult(
            att_d=np.array([1.0, np.nan, 3.0]),
            acrt_d=None,
            coefficients=np.array([0.5, 0.5]),
            dvals=np.array([1.0, 2.0, 3.0]),
            basis_info={},
        )
        errors = validate_estimator_result(result, n_obs=100, n_dvals=3)
        assert any("非有限值" in e for e in errors)

    def test_validate_influence_matrix_pass(self):
        """合法影响函数矩阵通过验证。"""
        inf = np.random.default_rng(0).normal(0, 1, (50, 10))
        errors = validate_influence_matrix(inf, n_obs=50, n_dvals=10)
        assert errors == []

    def test_validate_influence_matrix_wrong_shape(self):
        """维度不匹配应报错。"""
        inf = np.zeros((50, 5))
        errors = validate_influence_matrix(inf, n_obs=50, n_dvals=10)
        assert any("列数" in e for e in errors)

    def test_validate_influence_matrix_not_array(self):
        """非 ndarray 应报错。"""
        errors = validate_influence_matrix([[1, 2], [3, 4]], n_obs=2, n_dvals=2)
        assert any("np.ndarray" in e for e in errors)


# ===========================================================================
# 4. LinearDoseEstimator 教学示例
# ===========================================================================


class TestLinearDoseEstimator:
    """测试 LinearDoseEstimator 教学实现。"""

    def test_basic_properties(self):
        """属性检查。"""
        est = LinearDoseEstimator()
        assert est.name == "linear_dose"
        assert est.capabilities.max_spline_degree == 1
        assert est.capabilities.supports_multiperiod is False
        assert est.capabilities.supports_adaptive is False

    def test_fit_level(self):
        """线性估计 ATT(d)。"""
        dy, dose, dvals = _make_test_data()
        est = LinearDoseEstimator()
        result = est.fit(dy, dose, dvals, target_parameter="level")

        assert isinstance(result, EstimatorResult)
        assert result.att_d.shape == (len(dvals),)
        assert result.dvals.shape == (len(dvals),)
        assert result.acrt_d is None
        assert result.coefficients.shape == (2,)

        # 检查线性关系：ATT(d) ≈ α + β*d
        # DGP: ΔY = 2 + 3*D + ε → 系数应接近 [2, 3]
        np.testing.assert_allclose(result.coefficients, [2.0, 3.0], atol=0.3)

    def test_fit_slope(self):
        """线性估计 ACRT(d)。"""
        dy, dose, dvals = _make_test_data()
        est = LinearDoseEstimator()
        result = est.fit(dy, dose, dvals, target_parameter="slope")

        assert result.acrt_d is not None
        assert result.acrt_d.shape == (len(dvals),)
        # 线性模型 ACRT(d) = β（常数）
        np.testing.assert_allclose(result.acrt_d, result.coefficients[1], atol=1e-10)

    def test_influence_function_shape(self):
        """影响函数矩阵维度正确。"""
        dy, dose, dvals = _make_test_data()
        est = LinearDoseEstimator()
        result = est.fit(dy, dose, dvals)

        inf_mat = est.influence_function(dy, dose, dvals, result.coefficients)
        assert inf_mat.shape == (len(dy), len(dvals))
        assert np.isfinite(inf_mat).all()

    def test_influence_function_slope(self):
        """slope 模式影响函数。"""
        dy, dose, dvals = _make_test_data()
        est = LinearDoseEstimator()
        result = est.fit(dy, dose, dvals, target_parameter="slope")

        inf_mat = est.influence_function(
            dy, dose, dvals, result.coefficients, target_parameter="slope"
        )
        assert inf_mat.shape == (len(dy), len(dvals))

    def test_validate_assumptions_empty(self):
        """LinearDoseEstimator 的验证返回空列表。"""
        est = LinearDoseEstimator()
        warnings = est.validate_assumptions(None, None)
        assert warnings == []

    def test_get_set_params(self):
        """get_params/set_params 基本功能。"""
        est = LinearDoseEstimator()
        assert est.get_params() == {}
        # set_params 返回 self
        ret = est.set_params()
        assert ret is est


# ===========================================================================
# 5. 注册表集成
# ===========================================================================


class TestRegistryIntegration:
    """测试 BaseEstimator 子类与注册表的集成。"""

    def test_register_base_estimator(self):
        """BaseEstimator 子类可以注册到注册表。"""
        registry = EstimatorRegistry()
        est = LinearDoseEstimator()
        # 手动使用适配器注册
        adapter = _BaseEstimatorAdapter(est)
        registry.register(adapter)

        assert "linear_dose" in registry
        retrieved = registry.get("linear_dose")
        assert retrieved.name == "linear_dose"
        assert retrieved.supports_multiperiod is False
        assert retrieved.supports_adaptive is False

    def test_register_estimator_function_with_base(self):
        """register_estimator() 自动包装 BaseEstimator 子类。"""
        # 创建独立的注册表来避免污染全局状态
        registry = EstimatorRegistry()
        est = LinearDoseEstimator()
        adapter = _BaseEstimatorAdapter(est)
        registry.register(adapter)

        # 验证适配器属性
        retrieved = registry.get("linear_dose")
        assert hasattr(retrieved, "base_estimator")
        assert retrieved.base_estimator is est

    def test_adapter_validate_spec(self):
        """适配器正确代理 validate_assumptions。"""
        est = LinearDoseEstimator()
        adapter = _BaseEstimatorAdapter(est)
        errors = adapter.validate_spec(None, None)
        assert errors == []

    def test_existing_protocol_still_works(self):
        """回归测试：现有 Protocol 风格估计器仍可注册。"""
        registry = EstimatorRegistry()

        class ProtocolEstimator:
            name = "proto_test"
            supports_multiperiod = False
            supports_adaptive = False

            def validate_spec(self, spec, panel):
                return []

            def estimate(self, panel, spec, **kwargs):
                return None

        registry.register(ProtocolEstimator())
        assert "proto_test" in registry
        assert registry.get("proto_test").name == "proto_test"

    def test_default_registry_has_builtins(self):
        """默认注册表包含内置估计器。"""
        registry = get_default_registry()
        available = registry.list_available()
        assert "parametric" in available
        assert "cck" in available


# ===========================================================================
# 6. EstimatorCapabilities
# ===========================================================================


class TestEstimatorCapabilities:
    """测试 EstimatorCapabilities 数据类。"""

    def test_defaults(self):
        """默认值全部为 False/3。"""
        caps = EstimatorCapabilities()
        assert caps.supports_multiperiod is False
        assert caps.supports_eventstudy is False
        assert caps.supports_adaptive is False
        assert caps.supports_covariates is False
        assert caps.supports_clustering is False
        assert caps.max_spline_degree == 3

    def test_custom_values(self):
        """自定义值正确存储。"""
        caps = EstimatorCapabilities(
            supports_multiperiod=True,
            supports_eventstudy=True,
            max_spline_degree=5,
        )
        assert caps.supports_multiperiod is True
        assert caps.supports_eventstudy is True
        assert caps.max_spline_degree == 5

    def test_frozen(self):
        """frozen=True 不可修改。"""
        caps = EstimatorCapabilities()
        with pytest.raises(Exception):  # FrozenInstanceError
            caps.supports_multiperiod = True


# ===========================================================================
# 7. 公共导出验证
# ===========================================================================


class TestPublicExports:
    """确认新符号从 contdid 包顶层可访问。"""

    def test_base_estimator_import(self):
        import contdid

        assert hasattr(contdid, "BaseEstimator")
        assert hasattr(contdid, "EstimatorCapabilities")
        assert hasattr(contdid, "EstimatorResult")
        assert hasattr(contdid, "LinearDoseEstimator")
        assert hasattr(contdid, "validate_estimator_result")
        assert hasattr(contdid, "validate_influence_matrix")

    def test_all_list_contains_new_symbols(self):
        import contdid

        for sym in [
            "BaseEstimator",
            "EstimatorCapabilities",
            "EstimatorResult",
            "LinearDoseEstimator",
            "validate_estimator_result",
            "validate_influence_matrix",
        ]:
            assert sym in contdid.__all__
