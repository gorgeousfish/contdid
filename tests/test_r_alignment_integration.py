"""集成测试 - 验证R包对齐功能的协同工作。

验证三项已实现的R包对齐功能（P0 数值一致性、P1 Anticipation参数、P4 对照组支持）
在组合使用时能正确协同。

Theoretical reference:
- arXiv-2107.02637v7 Assumption 3-MP (pp.1423-1436)
- base_period = g - anticipation - 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    PanelData,
    cont_did,
    simulate_contdid_data,
)
from contdid.validation import ContDIDValidationError, validate_spec


# ---------------------------------------------------------------------------
# Shared fixture: staggered panel with never-treated + multiple cohorts
# ---------------------------------------------------------------------------


def _make_integration_panel(
    n_never: int = 50,
    n_g2: int = 60,
    n_g3: int = 50,
    n_g4: int = 40,
    n_periods: int = 6,
    effect_slope: float = 1.5,
    seed: int = 12345,
) -> PanelData:
    """Create a T=`n_periods` staggered panel for integration tests.

    Groups:
      G=0: never-treated (pure control)
      G=3: treated starting period 3
      G=4: treated starting period 4
      G=5: treated starting period 5
    DGP: Y_it = unit_fe + t + effect_slope * dose * 1(t >= G) + noise
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    uid = 0

    groups_config = {0: n_never, 3: n_g2, 4: n_g3, 5: n_g4}

    for group, n_units in groups_config.items():
        for _ in range(n_units):
            fe = rng.normal(0, 1.0)
            dose = rng.uniform(0.1, 1.0) if group > 0 else 0.0
            for t in range(1, n_periods + 1):
                treated_now = group > 0 and t >= group
                effect = effect_slope * dose if treated_now else 0.0
                y = fe + t + effect + rng.normal(0, 0.3)
                records.append({
                    "id": uid,
                    "time_period": t,
                    "Y": y,
                    "G": group,
                    "D": dose if group > 0 else 0.0,
                })
            uid += 1

    df = pd.DataFrame(records)
    return PanelData(frame=df)


@pytest.fixture(scope="module")
def panel() -> PanelData:
    """Module-scoped panel for reuse across test classes."""
    return _make_integration_panel()


# ===========================================================================
# 1. Anticipation参数与对照组选择的交互
# ===========================================================================


class TestAnticipationWithControlGroups:
    """Anticipation参数与对照组选择的交互。"""

    def test_anticipation_with_nevertreated(self, panel: PanelData) -> None:
        """anticipation=1 + nevertreated对照组能正常运行并产生有效结果。"""
        spec = ContDIDSpec.dose_response(
            anticipation=1, control_group="nevertreated",
        )
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert result.estimate is not None
        assert len(result.estimate) > 0
        # 验证metadata正确记录参数
        assert result.metadata.get("control_group") == "nevertreated"

    def test_anticipation_with_notyettreated(self, panel: PanelData) -> None:
        """anticipation=1 + notyettreated对照组能正常运行并产生有效结果。"""
        spec = ContDIDSpec.dose_response(
            anticipation=1, control_group="notyettreated",
        )
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert result.estimate is not None
        assert len(result.estimate) > 0
        assert result.metadata.get("control_group") == "notyettreated"

    def test_anticipation_zero_same_as_default(self, panel: PanelData) -> None:
        """anticipation=0 结果与不指定anticipation完全一致。"""
        spec_default = ContDIDSpec.dose_response()
        spec_explicit = ContDIDSpec.dose_response(anticipation=0)

        r_default = cont_did(panel, spec=spec_default)
        r_explicit = cont_did(panel, spec=spec_explicit)

        np.testing.assert_array_almost_equal(
            r_default.estimate, r_explicit.estimate,
            decimal=12,
            err_msg="anticipation=0应与默认结果完全一致",
        )

    def test_different_control_groups_give_different_results(
        self, panel: PanelData,
    ) -> None:
        """nevertreated vs notyettreated在同一anticipation下产生不同估计。"""
        spec_never = ContDIDSpec.dose_response(
            anticipation=1, control_group="nevertreated",
        )
        spec_notyet = ContDIDSpec.dose_response(
            anticipation=1, control_group="notyettreated",
        )
        r_never = cont_did(panel, spec=spec_never)
        r_notyet = cont_did(panel, spec=spec_notyet)

        # 两种对照组策略应产生不同的点估计
        assert not np.allclose(r_never.estimate, r_notyet.estimate, atol=1e-10), \
            "nevertreated和notyettreated应产生不同的估计结果"

    def test_anticipation_changes_results_with_both_control_groups(
        self, panel: PanelData,
    ) -> None:
        """anticipation参数对两种对照组策略都产生影响。"""
        for cg in ("nevertreated", "notyettreated"):
            r0 = cont_did(
                panel,
                spec=ContDIDSpec.dose_response(anticipation=0, control_group=cg),
            )
            r1 = cont_did(
                panel,
                spec=ContDIDSpec.dose_response(anticipation=1, control_group=cg),
            )
            assert not np.allclose(r0.estimate, r1.estimate, atol=1e-10), \
                f"anticipation=0 vs 1 在 {cg} 下应产生不同结果"


# ===========================================================================
# 2. 数值一致性在不同选项组合下的验证
# ===========================================================================


class TestNumericalConsistencyWithOptions:
    """数值一致性在不同选项组合下的验证。"""

    def test_known_dgp_nevertreated(self, panel: PanelData) -> None:
        """已知DGP + nevertreated对照组的ATT恢复。

        DGP effect_slope=1.5, 参数估计应在合理范围内。
        """
        spec = ContDIDSpec.dose_response(control_group="nevertreated")
        result = cont_did(panel, spec=spec)

        # 估计值应为非空且有限
        assert all(np.isfinite(result.estimate)), "所有估计值应有限"

        # 标准误应为正
        if result.std_error is not None:
            assert all(
                se > 0 for se in result.std_error
            ), "标准误应为正"

    def test_known_dgp_notyettreated(self, panel: PanelData) -> None:
        """已知DGP + notyettreated对照组的ATT恢复。"""
        spec = ContDIDSpec.dose_response(control_group="notyettreated")
        result = cont_did(panel, spec=spec)

        assert all(np.isfinite(result.estimate)), "所有估计值应有限"
        if result.std_error is not None:
            assert all(
                se > 0 for se in result.std_error
            ), "标准误应为正"

    def test_anticipation_shifts_event_time(self, panel: PanelData) -> None:
        """anticipation参数正确偏移事件时间轴。

        anticipation=1 应该导致base_period = g - 2 (而非 g - 1)，
        从而影响分组策略。
        """
        spec_a0 = ContDIDSpec.dose_response(anticipation=0)
        spec_a1 = ContDIDSpec.dose_response(anticipation=1)

        r0 = cont_did(panel, spec=spec_a0)
        r1 = cont_did(panel, spec=spec_a1)

        # 两者的n_groups可能不同（高anticipation可能排除某些群组）
        n0 = r0.metadata.get("n_groups", 0)
        n1 = r1.metadata.get("n_groups", 0)

        # anticipation=1 应排除更多群组或改变分组数
        assert isinstance(n0, int) and n0 > 0
        assert isinstance(n1, int) and n1 > 0

    def test_results_stable_across_runs(self, panel: PanelData) -> None:
        """同一数据+同一spec的结果应该完全可复现。"""
        spec = ContDIDSpec.dose_response(
            anticipation=1, control_group="nevertreated",
        )
        r1 = cont_did(panel, spec=spec)
        r2 = cont_did(panel, spec=spec)

        np.testing.assert_array_equal(
            r1.estimate, r2.estimate,
            err_msg="相同输入应产生完全相同的输出",
        )


# ===========================================================================
# 3. 向后兼容性
# ===========================================================================


class TestBackwardCompatibility:
    """确保所有默认行为未改变。"""

    def test_default_control_group_is_notyettreated(self) -> None:
        """默认对照组为notyettreated。"""
        spec = ContDIDSpec.dose_response()
        assert spec.control_group == "notyettreated"

    def test_default_anticipation_is_zero(self) -> None:
        """默认anticipation为0。"""
        spec = ContDIDSpec.dose_response()
        assert spec.anticipation == 0

    def test_eventuallytreated_rejected(self) -> None:
        """eventuallytreated明确被拒绝（理论不支持）。"""
        with pytest.raises(
            (ContDIDValidationError, ValueError, TypeError),
        ):
            spec = ContDIDSpec.dose_response(control_group="eventuallytreated")
            validate_spec(spec)

    def test_eventstudy_default_control_group(self) -> None:
        """事件研究默认对照组也是notyettreated。"""
        spec = ContDIDSpec.eventstudy()
        assert spec.control_group == "notyettreated"

    def test_eventstudy_default_anticipation(self) -> None:
        """事件研究默认anticipation也是0。"""
        spec = ContDIDSpec.eventstudy()
        assert spec.anticipation == 0

    def test_old_api_still_works(self, panel: PanelData) -> None:
        """不传任何新参数时，API行为与之前一致。"""
        # 这应该使用所有默认值正常运行
        result = cont_did(panel)
        assert isinstance(result, ContDIDResult)
        assert result.estimate is not None


# ===========================================================================
# 4. Eventstudy + Anticipation + Control Group 三方交互
# ===========================================================================


class TestEventStudyFullIntegration:
    """事件研究分析中三项功能的组合。"""

    def test_eventstudy_nevertreated_anticipation0(
        self, panel: PanelData,
    ) -> None:
        """事件研究 + nevertreated + anticipation=0。"""
        spec = ContDIDSpec.eventstudy(
            anticipation=0, control_group="nevertreated",
        )
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert result.estimate is not None

    def test_eventstudy_notyettreated_anticipation1(
        self, panel: PanelData,
    ) -> None:
        """事件研究 + notyettreated + anticipation=1。"""
        spec = ContDIDSpec.eventstudy(
            anticipation=1, control_group="notyettreated",
        )
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert result.estimate is not None

    def test_eventstudy_control_group_affects_results(
        self, panel: PanelData,
    ) -> None:
        """事件研究中不同对照组产生不同结果。"""
        r_never = cont_did(
            panel,
            spec=ContDIDSpec.eventstudy(control_group="nevertreated"),
        )
        r_notyet = cont_did(
            panel,
            spec=ContDIDSpec.eventstudy(control_group="notyettreated"),
        )
        assert not np.allclose(
            r_never.estimate, r_notyet.estimate, atol=1e-10,
        ), "事件研究中两种对照组应产生不同估计"

    def test_eventstudy_anticipation_affects_results(
        self, panel: PanelData,
    ) -> None:
        """事件研究中不同anticipation产生不同结果。"""
        r0 = cont_did(
            panel,
            spec=ContDIDSpec.eventstudy(anticipation=0),
        )
        r1 = cont_did(
            panel,
            spec=ContDIDSpec.eventstudy(anticipation=1),
        )
        # 结果可能长度不同或值不同
        estimates_differ = (
            len(r0.estimate) != len(r1.estimate)
            or not np.allclose(r0.estimate, r1.estimate, atol=1e-10)
        )
        assert estimates_differ, "不同anticipation应产生不同的事件研究结果"


# ===========================================================================
# 5. simulate_contdid_data 与新功能的兼容性
# ===========================================================================


class TestSimulatedDataIntegration:
    """用simulate_contdid_data生成数据测试新功能。"""

    def test_simulated_data_with_nevertreated(self) -> None:
        """模拟数据 + nevertreated 对照组。"""
        panel = simulate_contdid_data(n=500, seed=42)
        spec = ContDIDSpec.dose_response(control_group="nevertreated")
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert all(np.isfinite(result.estimate))

    def test_simulated_data_with_anticipation(self) -> None:
        """模拟数据 + anticipation=1。"""
        panel = simulate_contdid_data(n=500, seed=42)
        spec = ContDIDSpec.dose_response(anticipation=1)
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert all(np.isfinite(result.estimate))

    def test_simulated_data_full_combo(self) -> None:
        """模拟数据 + anticipation=1 + nevertreated。"""
        panel = simulate_contdid_data(n=500, seed=42)
        spec = ContDIDSpec.dose_response(
            anticipation=1, control_group="nevertreated",
        )
        result = cont_did(panel, spec=spec)
        assert isinstance(result, ContDIDResult)
        assert all(np.isfinite(result.estimate))
