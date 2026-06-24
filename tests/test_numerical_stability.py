"""数值稳定性与性能优化正确性测试。

测试 B 样条节点去重、设计矩阵数值性质、导数精度、极端情况稳定性、
Lepski n_jobs 参数、Bootstrap 并行化接口，以及与 scipy 参考实现的对比。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

from contdid.bspline import (
    build_bspline_derivative_design,
    build_bspline_design,
    quantile_knots,
)
from contdid.influence import InfluenceFunction
from contdid.lepski import select_lepski_dimension


# =============================================================================
# 1. B 样条节点去重测试
# =============================================================================


class TestQuantileKnotsDeduplication:
    """quantile_knots 去重逻辑的正确性。"""

    def test_quantile_knots_deduplicates(self):
        """数据有大量重复值时，quantile_knots 应返回唯一节点。"""
        dose = np.concatenate([
            np.full(100, 0.1),
            np.full(80, 0.5),
            np.full(20, 0.9),
        ])
        knots = quantile_knots(dose, num_knots=5)
        # 应返回 <= 5 个唯一节点
        assert len(knots) <= 5
        # 所有返回节点应两两不同
        assert len(knots) == len(set(knots))

    def test_quantile_knots_empty_dose(self):
        """空数组应返回空列表。"""
        knots = quantile_knots(np.array([]), num_knots=5)
        assert knots == []

    def test_quantile_knots_single_value(self):
        """单一值（零 range）应返回空列表。"""
        dose = np.full(50, 3.14)
        knots = quantile_knots(dose, num_knots=5)
        assert knots == []

    def test_quantile_knots_normal_case(self):
        """正常分布数据应返回请求数量的唯一节点。"""
        rng = np.random.default_rng(42)
        dose = rng.normal(5.0, 1.0, size=500)
        knots = quantile_knots(dose, num_knots=5)
        # 正常分布数据有足够的唯一值，应该返回 5 个节点
        assert len(knots) == 5
        assert len(knots) == len(set(knots))

    def test_quantile_knots_preserves_order(self):
        """返回的节点应严格递增。"""
        rng = np.random.default_rng(123)
        dose = rng.uniform(0.0, 10.0, size=200)
        knots = quantile_knots(dose, num_knots=8)
        for i in range(len(knots) - 1):
            assert knots[i] < knots[i + 1], (
                f"节点不严格递增: knots[{i}]={knots[i]}, knots[{i+1}]={knots[i+1]}"
            )


# =============================================================================
# 2. 设计矩阵数值稳定性测试
# =============================================================================


class TestDesignMatrixStability:
    """设计矩阵的数值性质。"""

    def test_design_matrix_full_rank_with_deduplication(self):
        """去重后的节点应产生满秩设计矩阵。"""
        # 使用含大量重复的数据，但保证有足够唯一值支撑满秩
        rng = np.random.default_rng(11)
        # 50个重复值 + 150个连续值，模拟现实中的离散化剂量
        dose = np.concatenate([
            np.full(50, 2.0),
            np.full(30, 5.0),
            rng.uniform(0.0, 10.0, size=150),
        ])
        knots = quantile_knots(dose, num_knots=5)
        if len(knots) == 0:
            pytest.skip("去重后无内部节点可测试")
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        rank = np.linalg.matrix_rank(design)
        assert rank == design.shape[1], (
            f"设计矩阵不满秩: rank={rank}, 列数={design.shape[1]}"
        )

    def test_design_matrix_partition_of_unity(self):
        """设计矩阵行和应等于 1（误差 < 1e-10）。"""
        rng = np.random.default_rng(7)
        dose = rng.uniform(1.0, 5.0, size=100)
        knots = quantile_knots(dose, num_knots=4)
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        row_sums = design.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_design_matrix_nonnegative(self):
        """所有 B 样条值应非负。"""
        rng = np.random.default_rng(99)
        dose = rng.uniform(0.0, 10.0, size=200)
        knots = quantile_knots(dose, num_knots=6)
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        assert np.all(design >= -1e-15), (
            f"存在负值: min={design.min():.2e}"
        )

    def test_design_matrix_boundary_points(self):
        """评估点恰好在边界时应产生有效值（无 NaN）。"""
        dose = np.linspace(0.0, 1.0, 50)
        knots = [0.25, 0.5, 0.75]
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        assert not np.any(np.isnan(design)), "边界点处存在 NaN"
        # 确认边界点行和仍满足分区性质
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)

    def test_design_matrix_outside_range_clipped(self):
        """评估点超出数据范围时应被 clip 到边界。"""
        # 构造节点基于 [0, 1] 范围内的数据
        dose_train = np.linspace(0.0, 1.0, 50)
        knots = [0.25, 0.5, 0.75]
        xmin, xmax = 0.0, 1.0
        # 评估点包含超出范围的值
        dose_eval = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        design = build_bspline_design(
            dose_eval, degree=3, interior_knots=knots, xmin=xmin, xmax=xmax
        )
        assert not np.any(np.isnan(design)), "超出范围的点产生了 NaN"
        # 行和仍应为 1
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)


# =============================================================================
# 3. 导数精度测试
# =============================================================================


class TestDerivativeAccuracy:
    """导数设计矩阵的精度验证。"""

    def test_derivative_matches_numerical(self):
        """解析导数应与数值微分一致（误差 < 1e-6）。"""
        rng = np.random.default_rng(42)
        dose = rng.uniform(1.0, 5.0, size=100)
        knots = quantile_knots(dose, num_knots=4)
        # 在内部点（避开边界）上做数值微分对比
        dose_interior = dose[(dose > 1.1) & (dose < 4.9)]
        h = 1e-7

        deriv_analytic = build_bspline_derivative_design(
            dose_interior, degree=3, interior_knots=knots,
            xmin=dose.min(), xmax=dose.max(),
        )
        design_plus = build_bspline_design(
            dose_interior + h, degree=3, interior_knots=knots,
            xmin=dose.min(), xmax=dose.max(),
        )
        design_minus = build_bspline_design(
            dose_interior - h, degree=3, interior_knots=knots,
            xmin=dose.min(), xmax=dose.max(),
        )
        deriv_numerical = (design_plus - design_minus) / (2 * h)
        np.testing.assert_allclose(
            deriv_analytic, deriv_numerical, atol=1e-6, rtol=1e-5
        )

    def test_derivative_sum_is_zero(self):
        """导数设计矩阵行和应为 0（分区性质的导数）。"""
        rng = np.random.default_rng(10)
        dose = rng.uniform(0.0, 10.0, size=150)
        knots = quantile_knots(dose, num_knots=5)
        deriv = build_bspline_derivative_design(dose, degree=3, interior_knots=knots)
        row_sums = deriv.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-8)

    def test_derivative_boundary_stability(self):
        """边界点处导数应有限且稳定。"""
        dose = np.linspace(0.0, 1.0, 50)
        knots = [0.25, 0.5, 0.75]
        deriv = build_bspline_derivative_design(dose, degree=3, interior_knots=knots)
        assert np.all(np.isfinite(deriv)), "边界点处导数出现 Inf/NaN"
        # 导数值应在合理范围内
        assert np.all(np.abs(deriv) < 1000), (
            f"导数值异常大: max|deriv|={np.abs(deriv).max():.2e}"
        )


# =============================================================================
# 4. 极端情况稳定性测试
# =============================================================================


class TestExtremeCases:
    """极端数据范围和参数下的稳定性。"""

    def test_extreme_dose_range(self):
        """大尺度剂量（1e6 范围）不应产生数值问题。"""
        rng = np.random.default_rng(55)
        dose = rng.uniform(0.0, 1e6, size=200)
        knots = quantile_knots(dose, num_knots=5)
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        assert np.all(np.isfinite(design))
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)

    def test_small_dose_range(self):
        """小尺度剂量（1e-6 范围）不应产生数值问题。"""
        rng = np.random.default_rng(66)
        dose = rng.uniform(0.0, 1e-6, size=200)
        knots = quantile_knots(dose, num_knots=5)
        if len(knots) == 0:
            pytest.skip("小尺度数据去重后无内部节点")
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        assert np.all(np.isfinite(design))
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)

    def test_high_degree_stability(self):
        """高阶 B 样条（degree=5）应保持数值稳定。"""
        rng = np.random.default_rng(77)
        dose = rng.uniform(0.0, 10.0, size=300)
        knots = quantile_knots(dose, num_knots=6)
        design = build_bspline_design(dose, degree=5, interior_knots=knots)
        assert np.all(np.isfinite(design))
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)
        assert np.all(design >= -1e-14)

    def test_many_knots_stability(self):
        """大量节点（num_knots=20）应保持稳定。"""
        rng = np.random.default_rng(88)
        dose = rng.uniform(0.0, 10.0, size=500)
        knots = quantile_knots(dose, num_knots=20)
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        assert np.all(np.isfinite(design))
        np.testing.assert_allclose(design.sum(axis=1), 1.0, atol=1e-10)
        rank = np.linalg.matrix_rank(design)
        assert rank == design.shape[1]

    def test_condition_number_acceptable(self):
        """B 样条设计矩阵条件数应合理（< 1000）。"""
        rng = np.random.default_rng(33)
        dose = rng.uniform(0.0, 10.0, size=300)
        knots = quantile_knots(dose, num_knots=5)
        design = build_bspline_design(dose, degree=3, interior_knots=knots)
        # 使用 X'X 的条件数
        xtx = design.T @ design
        cond = np.linalg.cond(xtx)
        assert cond < 1000, f"条件数过大: cond(X'X)={cond:.1f}"


# =============================================================================
# 5. Lepski n_jobs 参数测试
# =============================================================================


class TestLepskiNJobs:
    """Lepski 自适应维度选择的 n_jobs 参数。"""

    @pytest.fixture()
    def lepski_data(self):
        """生成 Lepski 测试数据。"""
        rng = np.random.default_rng(2024)
        n = 200
        dose = rng.uniform(1.0, 5.0, size=n)
        # 简单的二次响应加噪声
        delta_outcome = 0.5 * dose + 0.1 * dose**2 + rng.normal(0, 0.5, size=n)
        return delta_outcome, dose

    def test_lepski_n_jobs_parameter_accepted(self, lepski_data):
        """select_lepski_dimension 应接受 n_jobs=1 参数。"""
        delta_outcome, dose = lepski_data
        # 不应抛出异常
        result = select_lepski_dimension(
            delta_outcome, dose,
            degree=3, bootstrap_reps=100, seed=42, n_jobs=1,
        )
        assert result.selected_dimension >= 4  # 至少 degree+1

    def test_lepski_result_consistent(self, lepski_data):
        """固定 seed 下 Lepski 结果应可复现。"""
        delta_outcome, dose = lepski_data
        result1 = select_lepski_dimension(
            delta_outcome, dose,
            degree=3, bootstrap_reps=200, seed=12345, n_jobs=1,
        )
        result2 = select_lepski_dimension(
            delta_outcome, dose,
            degree=3, bootstrap_reps=200, seed=12345, n_jobs=1,
        )
        assert result1.selected_dimension == result2.selected_dimension
        np.testing.assert_array_equal(result1.fitted_values, result2.fitted_values)


# =============================================================================
# 6. Bootstrap 并行化接口测试
# =============================================================================


class TestBootstrapParallel:
    """multiplier_bootstrap 的 n_jobs 参数和可复现性。"""

    @pytest.fixture()
    def influence_obj(self):
        """创建一个简单的 InfluenceFunction 对象。"""
        rng = np.random.default_rng(777)
        n_units = 100
        n_estimands = 10
        values = rng.normal(0, 1, size=(n_units, n_estimands))
        unit_ids = tuple(range(n_units))
        estimand_labels = tuple(f"d{i}" for i in range(n_estimands))
        return InfluenceFunction(
            unit_ids=unit_ids,
            values=values,
            estimand_labels=estimand_labels,
            n_total=n_units,
        )

    def test_bootstrap_n_jobs_parameter(self, influence_obj):
        """multiplier_bootstrap 应接受 n_jobs=1 参数。"""
        result = influence_obj.multiplier_bootstrap(
            biters=100, alp=0.05, cband=True, seed=42, n_jobs=1,
        )
        assert "critical_value" in result
        assert result["critical_value"] > 0

    def test_bootstrap_reproducible_with_seed(self, influence_obj):
        """固定 seed 下 Bootstrap 结果应完全可复现。"""
        result1 = influence_obj.multiplier_bootstrap(
            biters=500, alp=0.05, cband=True, seed=9999, n_jobs=1,
        )
        result2 = influence_obj.multiplier_bootstrap(
            biters=500, alp=0.05, cband=True, seed=9999, n_jobs=1,
        )
        assert result1["critical_value"] == result2["critical_value"]
        assert result1["std_error"] == result2["std_error"]


# =============================================================================
# 7. 与 scipy 参考实现对比
# =============================================================================


class TestScipyReference:
    """与 scipy.interpolate.BSpline 参考实现的一致性。"""

    def test_bspline_matches_scipy_reference(self):
        """自实现的 B 样条应与 scipy.interpolate.BSpline 一致。"""
        rng = np.random.default_rng(101)
        dose = rng.uniform(0.0, 10.0, size=100)
        interior_knots = [2.5, 5.0, 7.5]
        degree = 3

        # 使用 contdid 构建设计矩阵
        design = build_bspline_design(dose, degree=degree, interior_knots=interior_knots)
        num_basis = len(interior_knots) + degree + 1  # 7

        # 构造等价的 scipy BSpline 并逐列对比
        xmin, xmax = dose.min(), dose.max()
        knot_vector = (
            [xmin] * (degree + 1)
            + interior_knots
            + [xmax] * (degree + 1)
        )
        knot_vector = np.array(knot_vector, dtype=float)

        for j in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[j] = 1.0
            spl = ScipyBSpline(knot_vector, coeffs, degree, extrapolate=True)
            expected = spl(np.clip(dose, xmin, xmax))
            np.testing.assert_allclose(
                design[:, j], expected, atol=1e-12,
                err_msg=f"第 {j} 列与 scipy 参考不一致",
            )

    def test_derivative_matches_scipy_derivative(self):
        """导数应与 scipy BSpline.derivative() 一致。"""
        rng = np.random.default_rng(202)
        dose = rng.uniform(0.0, 10.0, size=100)
        interior_knots = [2.5, 5.0, 7.5]
        degree = 3

        # contdid 导数设计矩阵
        deriv_design = build_bspline_derivative_design(
            dose, degree=degree, interior_knots=interior_knots
        )
        num_basis = len(interior_knots) + degree + 1

        # scipy 参考
        xmin, xmax = dose.min(), dose.max()
        knot_vector = np.array(
            [xmin] * (degree + 1) + interior_knots + [xmax] * (degree + 1),
            dtype=float,
        )

        for j in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[j] = 1.0
            spl = ScipyBSpline(knot_vector, coeffs, degree, extrapolate=True)
            dspl = spl.derivative(nu=1)
            expected = dspl(np.clip(dose, xmin, xmax))
            np.testing.assert_allclose(
                deriv_design[:, j], expected, atol=1e-10,
                err_msg=f"第 {j} 列导数与 scipy 参考不一致",
            )
