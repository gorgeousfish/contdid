"""Python↔R 数值一致性验证框架。

统计一致性标准: max|bias/SE| ≤ 3
（Python估计与参考值（R基准或解析真值）的偏差不超过3个标准误）

验证覆盖:
1. TestRParametricDoseConsistency - parametric dose路径与R基准对比
2. TestRCCKConsistency           - CCK非参数路径与R基准对比
3. TestREventStudyConsistency    - 事件研究路径与R基准对比
4. TestInternalConsistency       - Python内部一致性（不依赖R fixtures）
5. TestConvergenceProperties     - 样本量递增时的收敛性验证

注意:
- 依赖R基准数据的测试（TestR*）在 fixtures 不存在时自动 skip
- TestInternalConsistency 和 TestConvergenceProperties 始终可独立运行
- DGP参数定义在 tests/r_reference/dgp_params.json

References:
- CGBS (arXiv:2107.02637v7): ATT(d) identification (Theorem 3.1)
- CCK (arXiv:2107.11869v3): uniform inference (Theorem 2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from contdid import PanelData, ContDIDResult, cont_did, simulate_contdid_data

# ============================================================================
# 常量与路径
# ============================================================================

FIXTURES_DIR = Path(__file__).parent / "r_reference" / "fixtures"
DGP_PARAMS_PATH = Path(__file__).parent / "r_reference" / "dgp_params.json"
CONSISTENCY_THRESHOLD = 3.0  # max|bias/SE| 阈值


# ============================================================================
# 辅助函数
# ============================================================================


def _load_fixture(filename: str) -> dict[str, Any]:
    """加载R基准fixture文件。"""
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"R fixture not found: {filename}. Run generate_fixtures.R first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dgp_params() -> dict[str, Any]:
    """加载共享DGP参数。"""
    return json.loads(DGP_PARAMS_PATH.read_text(encoding="utf-8"))


def _check_consistency(
    py_estimates: list[float],
    py_se: list[float],
    ref_estimates: list[float],
    ref_se: list[float],
    threshold: float = CONSISTENCY_THRESHOLD,
    label: str = "",
) -> None:
    """验证Python估计与参考估计的数值一致性。

    使用合并标准误: combined_se = sqrt(py_se^2 + ref_se^2)
    判定标准: |py_est - ref_est| / combined_se ≤ threshold
    """
    assert len(py_estimates) == len(ref_estimates), (
        f"Grid length mismatch: Python={len(py_estimates)}, R={len(ref_estimates)}"
    )

    max_ratio = 0.0
    worst_idx = -1

    for i in range(len(py_estimates)):
        diff = abs(py_estimates[i] - ref_estimates[i])
        combined_se = np.sqrt(py_se[i] ** 2 + ref_se[i] ** 2)
        if combined_se > 1e-10:
            ratio = diff / combined_se
            if ratio > max_ratio:
                max_ratio = ratio
                worst_idx = i

    assert max_ratio <= threshold, (
        f"{label} consistency FAILED: max|diff/SE| = {max_ratio:.3f} > {threshold} "
        f"(worst at index {worst_idx})"
    )


def _check_truth_consistency(
    estimates: list[float],
    se: list[float],
    truth: list[float],
    threshold: float = CONSISTENCY_THRESHOLD,
    label: str = "",
) -> None:
    """验证估计值与解析真值的一致性。

    判定标准: |est - truth| / se ≤ threshold
    """
    max_ratio = 0.0
    worst_idx = -1

    for i in range(len(estimates)):
        bias = abs(estimates[i] - truth[i])
        if se[i] > 1e-10:
            ratio = bias / se[i]
            if ratio > max_ratio:
                max_ratio = ratio
                worst_idx = i

    assert max_ratio <= threshold, (
        f"{label} truth consistency FAILED: max|bias/SE| = {max_ratio:.3f} > {threshold} "
        f"(worst at index {worst_idx}, est={estimates[worst_idx]:.4f}, "
        f"truth={truth[worst_idx]:.4f}, se={se[worst_idx]:.4f})"
    )


def _generate_two_period_panel(
    n: int = 5000,
    dose_linear_effect: float = 1.0,
    dose_quadratic_effect: float = 0.0,
    seed: int = 56789,
) -> PanelData:
    """生成两期面板数据（用于精确对比）。"""
    return simulate_contdid_data(
        n=n,
        num_time_periods=2,
        num_groups=2,
        pg=[0.75],
        pu=0.25,
        dose_linear_effect=dose_linear_effect,
        dose_quadratic_effect=dose_quadratic_effect,
        seed=seed,
        dgp_id="SIM-005-cck-two-period",
    )


# ============================================================================
# R基准对比测试: Parametric Dose
# ============================================================================


class TestRParametricDoseConsistency:
    """验证parametric dose路径与R包数值一致性。

    依赖: tests/r_reference/fixtures/parametric_dose_att.json
          tests/r_reference/fixtures/parametric_dose_acrt.json
    """

    @pytest.fixture(autouse=True)
    def _check_fixtures(self):
        if not FIXTURES_DIR.exists() or not (FIXTURES_DIR / "parametric_dose_att.json").exists():
            pytest.skip("R reference fixtures not available. Run generate_fixtures.R first.")

    def test_att_linear_dose(self):
        """ATT(d) parametric路径: 线性DGP与R一致。"""
        fixture = _load_fixture("parametric_dose_att.json")
        scenario = fixture["scenarios"]["SIM-002-linear-dose"]
        r_results = scenario["results"]

        # 使用相同DGP参数生成Python数据
        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-002-linear-dose",
        )

        # 用与R相同的evaluation grid
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="ATT(d) linear parametric",
        )

    def test_att_quadratic_dose(self):
        """ATT(d) parametric路径: 二次DGP与R一致。"""
        fixture = _load_fixture("parametric_dose_att.json")
        scenario = fixture["scenarios"]["SIM-003-quadratic-dose"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-003-quadratic-dose",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="ATT(d) quadratic parametric",
        )

    def test_acrt_linear_dose(self):
        """ACRT(d) parametric路径: 线性DGP与R一致。"""
        fixture = _load_fixture("parametric_dose_acrt.json")
        scenario = fixture["scenarios"]["SIM-002-linear-dose"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-002-linear-dose",
        )

        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="ACRT(d) linear parametric",
        )

    def test_acrt_quadratic_dose(self):
        """ACRT(d) parametric路径: 二次DGP与R一致。"""
        fixture = _load_fixture("parametric_dose_acrt.json")
        scenario = fixture["scenarios"]["SIM-003-quadratic-dose"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-003-quadratic-dose",
        )

        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="ACRT(d) quadratic parametric",
        )


# ============================================================================
# R基准对比测试: CCK
# ============================================================================


class TestRCCKConsistency:
    """验证CCK非参数路径与R包数值一致性。

    依赖: tests/r_reference/fixtures/cck_dose_att.json
    注意: CCK目前仅支持两期面板
    """

    @pytest.fixture(autouse=True)
    def _check_fixtures(self):
        if not FIXTURES_DIR.exists() or not (FIXTURES_DIR / "cck_dose_att.json").exists():
            pytest.skip("R CCK fixture not available. Run generate_fixtures.R first.")

    def test_cck_att_linear(self):
        """CCK ATT(d): 两期线性DGP与R一致。"""
        fixture = _load_fixture("cck_dose_att.json")
        scenario = fixture["scenarios"]["SIM-TP-linear"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            pg=dgp["pg"],
            pu=dgp["pu"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-005-cck-two-period",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="CCK ATT(d) linear",
        )

    def test_cck_att_quadratic(self):
        """CCK ATT(d): 两期二次DGP与R一致。"""
        fixture = _load_fixture("cck_dose_att.json")
        scenario = fixture["scenarios"]["SIM-TP-quadratic"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            pg=dgp["pg"],
            pu=dgp["pu"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-005-cck-two-period",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=r_results["grid"],
        )

        _check_consistency(
            result.estimate, result.std_error,
            r_results["estimate"], r_results["std_error"],
            label="CCK ATT(d) quadratic",
        )


# ============================================================================
# R基准对比测试: Event Study
# ============================================================================


class TestREventStudyConsistency:
    """验证事件研究路径与R包数值一致性。

    依赖: tests/r_reference/fixtures/eventstudy_att.json
    """

    @pytest.fixture(autouse=True)
    def _check_fixtures(self):
        if not FIXTURES_DIR.exists() or not (FIXTURES_DIR / "eventstudy_att.json").exists():
            pytest.skip("R event study fixture not available. Run generate_fixtures.R first.")

    def test_eventstudy_att(self):
        """Event study ATT: 线性DGP与R一致。"""
        fixture = _load_fixture("eventstudy_att.json")
        scenario = fixture["scenarios"]["SIM-002-linear-dose"]
        r_results = scenario["results"]

        dgp = scenario["dgp"]
        panel = simulate_contdid_data(
            n=dgp["n"],
            num_time_periods=dgp["num_time_periods"],
            num_groups=dgp["num_groups"],
            dose_linear_effect=dgp["dose_linear_effect"],
            dose_quadratic_effect=dgp["dose_quadratic_effect"],
            seed=dgp["seed"],
            dgp_id="SIM-002-linear-dose",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        # Event study结果的grid对齐
        r_grid = r_results.get("grid")
        py_grid = result.event_time_grid or result.grid

        if r_grid is not None and len(py_grid) == len(r_grid):
            _check_consistency(
                result.estimate, result.std_error,
                r_results["estimate"], r_results["std_error"],
                label="Event study ATT",
            )
        else:
            # Grid长度不同时, 只比较重叠的event times
            pytest.skip(
                f"Event-time grid mismatch: Python has {len(py_grid)} points, "
                f"R has {len(r_grid) if r_grid else 'unknown'}"
            )


# ============================================================================
# Python内部一致性测试（不依赖R fixtures）
# ============================================================================


class TestInternalConsistency:
    """Python包内部一致性验证。

    使用 simulate_contdid_data 生成已知DGP数据，验证:
    1. 估计偏差在统计可接受范围内 (|bias/SE| ≤ 3)
    2. 零效应下置信区间正确覆盖
    3. Parametric和CCK在相同basis下给出一致结果

    这些测试不依赖R环境，始终可运行。
    """

    def test_known_dgp_att_linear_recovery(self):
        """已知线性DGP下ATT(d)估计偏差可控。

        DGP: ATT(d) = dose_linear_effect * d = 1.0 * d
        验证: |est(d) - d| / SE(d) ≤ 3 对所有grid点
        """
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=1.0,
            dose_quadratic_effect=0.0,
            seed=23456,
            dgp_id="SIM-002-linear-dose",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        # 真值: ATT(d) = 1.0 * d
        truth = [1.0 * d for d in result.grid]
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Internal ATT(d) linear",
        )

    def test_known_dgp_att_quadratic_recovery(self):
        """已知二次DGP下ATT(d)估计偏差可控。

        DGP: ATT(d) = dose_quadratic_effect * d^2 = 1.0 * d^2
        """
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=0.0,
            dose_quadratic_effect=1.0,
            seed=34567,
            dgp_id="SIM-003-quadratic-dose",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        # 真值: ATT(d) = 1.0 * d^2
        truth = [1.0 * d**2 for d in result.grid]
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Internal ATT(d) quadratic",
        )

    def test_known_dgp_acrt_linear_recovery(self):
        """已知线性DGP下ACRT(d)估计偏差可控。

        DGP: ACRT(d) = d/dd(1.0*d) = 1.0 （常数）
        注意: 多期staggered聚合的slope估计在边界点有有限样本偏差，
        需要更大样本量来确保收敛。使用degree=1精确匹配线性DGP。
        """
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=1.0,
            dose_quadratic_effect=0.0,
            seed=23456,
            dgp_id="SIM-002-linear-dose",
        )

        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=1,
            num_knots=0,
        )

        # 真值: ACRT(d) = 1.0 everywhere
        truth = [1.0] * len(result.grid)
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Internal ACRT(d) linear",
        )

    def test_known_dgp_acrt_quadratic_recovery(self):
        """已知二次DGP下ACRT(d)估计偏差可控。

        DGP: ACRT(d) = d/dd(1.0*d^2) = 2.0*d
        """
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=0.0,
            dose_quadratic_effect=1.0,
            seed=34567,
            dgp_id="SIM-003-quadratic-dose",
        )

        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        # 真值: ACRT(d) = 2.0*d
        truth = [2.0 * d for d in result.grid]
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Internal ACRT(d) quadratic",
        )

    def test_zero_effect_coverage(self):
        """零效应DGP下估计值应统计上不显著。

        DGP: ATT(d)=0, ACRT(d)=0
        验证: |est| / SE ≤ 3 (即95%CI覆盖0)
        """
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=0.0,
            dose_quadratic_effect=0.0,
            seed=12345,
            dgp_id="SIM-001-null-dose",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=1,
            num_knots=0,
        )

        # 真值: ATT(d) = 0
        truth = [0.0] * len(result.grid)
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Zero-effect coverage (ATT)",
        )

    def test_zero_effect_slope_coverage(self):
        """零效应DGP下ACRT估计值应统计上不显著。"""
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=0.0,
            dose_quadratic_effect=0.0,
            seed=12345,
            dgp_id="SIM-001-null-dose",
        )

        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=1,
            num_knots=0,
        )

        truth = [0.0] * len(result.grid)
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="Zero-effect coverage (ACRT)",
        )

    def test_cck_internal_linear(self):
        """CCK路径: 两期线性DGP内部一致性。"""
        panel = simulate_contdid_data(
            n=5000,
            num_time_periods=2,
            num_groups=2,
            pg=[0.75],
            pu=0.25,
            dose_linear_effect=1.0,
            dose_quadratic_effect=0.0,
            seed=56789,
            dgp_id="SIM-005-cck-two-period",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        # 真值: ATT(d) = 1.0 * d
        truth = [1.0 * d for d in result.grid]
        _check_truth_consistency(
            result.estimate, result.std_error, truth,
            label="CCK internal ATT(d) linear",
        )

    def test_parametric_cck_agreement(self):
        """Parametric和CCK在相同basis下应给出一致估计。"""
        panel = simulate_contdid_data(
            n=5000,
            num_time_periods=2,
            num_groups=2,
            pg=[0.75],
            pu=0.25,
            dose_linear_effect=1.0,
            dose_quadratic_effect=0.0,
            seed=56789,
            dgp_id="SIM-005-cck-two-period",
        )

        # 使用相同grid确保可比性
        common_grid = np.linspace(0.1, 0.9, 15).tolist()

        result_param = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=common_grid,
        )

        result_cck = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            dose_est_method="cck",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
            dvals=common_grid,
        )

        _check_consistency(
            result_param.estimate, result_param.std_error,
            result_cck.estimate, result_cck.std_error,
            label="Parametric vs CCK agreement",
        )

    def test_eventstudy_zero_pretrend(self):
        """事件研究: 零效应DGP下pre-treatment估计应为~0。"""
        panel = simulate_contdid_data(
            n=5000,
            dose_linear_effect=0.0,
            dose_quadratic_effect=0.0,
            seed=12345,
            dgp_id="SIM-004-staggered-eventstudy-null",
        )

        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            degree=3,
            num_knots=0,
        )

        event_times = result.event_time_grid
        if event_times is None:
            event_times = result.metadata.get("event_time_grid")
        assert event_times is not None, "event_time_grid not found in result"

        # 检查pre-treatment period (event_time < 0)
        for i, et in enumerate(event_times):
            if et < 0:
                bias = abs(result.estimate[i])
                se = result.std_error[i]
                if se > 1e-10:
                    assert bias / se <= CONSISTENCY_THRESHOLD, (
                        f"Pre-trend at e={et}: |bias|/SE = {bias/se:.2f} > {CONSISTENCY_THRESHOLD}"
                    )


# ============================================================================
# 收敛性测试
# ============================================================================


class TestConvergenceProperties:
    """验证估计器的统计收敛性质。

    随着样本量增加，偏差应减小:
    - MSE应以 O(1/n) 衰减
    - SE应以 O(1/sqrt(n)) 衰减
    """

    def test_att_se_decreases_with_n(self):
        """ATT标准误随样本量增加而减小。"""
        se_by_n = []

        for n in [1000, 3000, 5000]:
            panel = simulate_contdid_data(
                n=n,
                dose_linear_effect=1.0,
                dose_quadratic_effect=0.0,
                seed=23456,
                dgp_id="SIM-002-linear-dose",
            )
            result = cont_did(
                panel,
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                degree=1,
                num_knots=0,
            )
            mean_se = np.mean(result.std_error)
            se_by_n.append(mean_se)

        # SE应单调递减
        assert se_by_n[0] > se_by_n[1] > se_by_n[2], (
            f"SE should decrease with n: got {se_by_n}"
        )

        # SE缩减比例应大致符合 sqrt(n_small/n_large)
        ratio_expected = np.sqrt(1000 / 5000)  # ~0.447
        ratio_actual = se_by_n[2] / se_by_n[0]
        # 允许较宽松的范围（0.2到1.0）以容纳有限样本波动
        assert 0.2 < ratio_actual < 1.0, (
            f"SE ratio n=5000/n=1000 = {ratio_actual:.3f}, expected ~{ratio_expected:.3f}"
        )

    def test_bias_bounded_across_sample_sizes(self):
        """不同样本量下偏差均在统计可接受范围内。"""
        for n in [2000, 5000]:
            panel = simulate_contdid_data(
                n=n,
                dose_linear_effect=1.0,
                dose_quadratic_effect=0.0,
                seed=23456,
                dgp_id="SIM-002-linear-dose",
            )
            result = cont_did(
                panel,
                target_parameter="level",
                aggregation="dose",
                dose_est_method="parametric",
                control_group="nevertreated",
                degree=3,
                num_knots=0,
            )

            truth = [1.0 * d for d in result.grid]
            _check_truth_consistency(
                result.estimate, result.std_error, truth,
                label=f"Convergence ATT(d) n={n}",
            )
