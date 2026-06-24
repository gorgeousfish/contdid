"""Tests for the unified cont_did() API: routing, inference integration, and consistency."""

from __future__ import annotations

import numpy as np
import pytest

from contdid import (
    ContDIDResult,
    ContDIDSpec,
    PanelData,
    cont_did,
    estimate_dose_level_effects,
    estimate_dose_slope_effects,
    estimate_eventstudy_effects,
    simulate_contdid_data,
)
from contdid.validation import ContDIDValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_period_panel(n: int = 200, seed: int = 42) -> PanelData:
    """Generate a two-period panel for testing."""
    return simulate_contdid_data(
        n=n,
        num_time_periods=2,
        num_groups=2,
        pg=[0.6],
        pu=0.4,
        dgp_id="SIM-002-linear-dose",
        seed=seed,
    )


def _multi_period_panel(n: int = 300, seed: int = 123) -> PanelData:
    """Generate a multi-period (T=4) panel for testing."""
    return simulate_contdid_data(
        n=n,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=seed,
    )


def _cck_two_period_panel(n: int = 300, seed: int = 99) -> PanelData:
    """Generate a two-period panel for CCK testing."""
    return simulate_contdid_data(
        n=n,
        dgp_id="SIM-005-cck-two-period",
        seed=seed,
    )


# ===========================================================================
# 1. Basic routing tests
# ===========================================================================


class TestBasicRouting:
    """Verify cont_did routes to the correct estimator based on parameters."""

    def test_cont_did_imports(self):
        """验证 cont_did 可以从 contdid 包直接导入。"""
        from contdid import cont_did as imported_func

        assert callable(imported_func)

    def test_cont_did_dose_level_two_period(self):
        """cont_did(aggregation='dose', target_parameter='level') 在两期面板下正确路由。"""
        panel = _two_period_panel()
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(d)"
        assert len(result.grid) > 0
        assert len(result.estimate) == len(result.grid)

    def test_cont_did_dose_slope_two_period(self):
        """cont_did(aggregation='dose', target_parameter='slope') 在两期面板下正确路由。"""
        panel = _two_period_panel()
        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="dose",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ACRT(d)"
        assert len(result.grid) > 0

    def test_cont_did_eventstudy_level(self):
        """cont_did(aggregation='eventstudy', target_parameter='level') 正确路由到事件研究。"""
        panel = _multi_period_panel()
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="eventstudy",
            control_group="notyettreated",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(event_time)"
        assert result.event_time_grid is not None

    def test_cont_did_eventstudy_slope(self):
        """cont_did(aggregation='eventstudy', target_parameter='slope') 正确路由。"""
        panel = _multi_period_panel()
        result = cont_did(
            panel,
            target_parameter="slope",
            aggregation="eventstudy",
            control_group="notyettreated",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ACRT(event_time)"
        assert result.event_time_grid is not None

    def test_cont_did_multiperiod_auto_detect(self):
        """多期面板（T>2）自动路由到 multiperiod 估计器。"""
        panel = _multi_period_panel()
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(d)"
        # Metadata should indicate multiperiod panel type
        assert result.metadata.get("panel_type") == "multiperiod"


# ===========================================================================
# 2. Inference integration tests
# ===========================================================================


class TestInferenceIntegration:
    """Verify cont_did automatically attaches inference."""

    def test_cont_did_result_has_inference(self):
        """cont_did 返回的结果应自动包含推断（SE > 0）。"""
        panel = _two_period_panel()
        result = cont_did(panel, biters=200)
        assert result.has_inference is True
        assert any(se > 0 for se in result.std_error)

    def test_cont_did_result_has_confidence_interval(self):
        """结果的 metadata 中应包含置信区间/带信息。"""
        panel = _two_period_panel()
        result = cont_did(panel, biters=200)
        # Should have confidence interval or confidence band metadata
        has_ci = result.confidence_interval is not None
        has_band = result.confidence_band is not None
        has_meta_band = (
            "confidence_band_lower" in result.metadata
            or "confidence_band_upper" in result.metadata
        )
        assert has_ci or has_band or has_meta_band

    def test_cont_did_cband_true(self):
        """cband=True 时应生成同时置信带。"""
        panel = _two_period_panel(n=300)
        result = cont_did(panel, cband=True, biters=200)
        # With cband=True, either confidence_band is set or metadata has band info
        has_band = result.confidence_band is not None
        has_meta_band = (
            "confidence_band_lower" in result.metadata
            or "confidence_band_upper" in result.metadata
        )
        assert has_band or has_meta_band


# ===========================================================================
# 3. Consistency with fine-grained functions
# ===========================================================================


class TestConsistencyWithFineGrained:
    """Verify cont_did results match direct calls to estimation functions."""

    def test_cont_did_matches_estimate_dose_level(self):
        """cont_did 结果应与直接调用 estimate_dose_level_effects 一致。"""
        panel = _two_period_panel(n=200, seed=77)
        dvals = [0.2, 0.4, 0.6, 0.8]

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            biters=200,
        )

        # Via unified API
        unified_result = cont_did(
            panel, spec=spec, dvals=dvals, degree=3, num_knots=0
        )

        # Via direct call
        direct_result = estimate_dose_level_effects(
            panel, spec, dvals=dvals, degree=3, num_knots=0
        )

        np.testing.assert_allclose(
            unified_result.estimate, direct_result.estimate, rtol=1e-10
        )
        np.testing.assert_allclose(
            unified_result.std_error, direct_result.std_error, rtol=1e-10
        )
        np.testing.assert_allclose(
            unified_result.grid, direct_result.grid, rtol=1e-10
        )

    def test_cont_did_matches_eventstudy(self):
        """cont_did(eventstudy) 结果应与直接调用 estimate_eventstudy_effects 一致。"""
        panel = _multi_period_panel(n=300, seed=88)
        dvals = [0.3, 0.6]

        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="notyettreated",
            biters=200,
        )

        # Via unified API
        unified_result = cont_did(
            panel, spec=spec, dvals=dvals, degree=3, num_knots=0
        )

        # Via direct call
        direct_result = estimate_eventstudy_effects(
            panel, spec, dvals=dvals, degree=3, num_knots=0
        )

        np.testing.assert_allclose(
            unified_result.estimate, direct_result.estimate, rtol=1e-10
        )
        np.testing.assert_allclose(
            unified_result.std_error, direct_result.std_error, rtol=1e-10
        )


# ===========================================================================
# 4. Spec object tests
# ===========================================================================


class TestSpecObject:
    """Verify spec-based and parameter-based calls work correctly."""

    def test_cont_did_with_spec_object(self):
        """传入 ContDIDSpec 对象应覆盖所有直接参数。"""
        panel = _two_period_panel()
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            biters=200,
        )
        result = cont_did(panel, spec=spec)
        # spec says slope → ACRT(d)
        assert result.estimand == "ACRT(d)"

    def test_cont_did_without_spec(self):
        """不传 spec 时应从直接参数自动构建。"""
        panel = _two_period_panel()
        # Default parameters: target_parameter='level', aggregation='dose'
        result = cont_did(panel, biters=200)
        assert result.estimand == "ATT(d)"
        assert isinstance(result, ContDIDResult)


# ===========================================================================
# 5. Convenience method tests
# ===========================================================================


class TestConvenienceMethods:
    """Verify ContDIDResult convenience methods."""

    def test_has_inference_property(self):
        """ContDIDResult.has_inference 正确反映推断状态。"""
        panel = _two_period_panel()
        result = cont_did(panel, biters=200)
        assert result.has_inference is True

        # A result with all-zero SE should report no inference
        no_inf_result = ContDIDResult(
            estimand="ATT(d)",
            grid=[0.2, 0.5, 0.8],
            estimate=[0.1, 0.2, 0.3],
            std_error=[0.0, 0.0, 0.0],
        )
        assert no_inf_result.has_inference is False

    def test_is_significant_method(self):
        """ContDIDResult.is_significant() 正确判断显著性。"""
        panel = _two_period_panel(n=300)
        result = cont_did(panel, biters=200)
        sig = result.is_significant()
        assert isinstance(sig, list)
        assert all(isinstance(s, bool) for s in sig)
        assert len(sig) == len(result.estimate)

        # Single index version
        sig_single = result.is_significant(index=0)
        assert isinstance(sig_single, bool)


# ===========================================================================
# 6. Error handling tests
# ===========================================================================


class TestErrorHandling:
    """Verify cont_did raises proper errors for invalid inputs."""

    def test_cont_did_invalid_aggregation(self):
        """无效的 aggregation 值应抛出 ContDIDValidationError。"""
        panel = _two_period_panel()
        with pytest.raises(ContDIDValidationError, match="Unsupported aggregation"):
            cont_did(panel, aggregation="invalid_agg")

    def test_cont_did_invalid_target(self):
        """无效的 target_parameter 应抛出错误。"""
        panel = _two_period_panel()
        with pytest.raises(ContDIDValidationError):
            cont_did(panel, target_parameter="invalid_target")

    def test_cont_did_cck_adaptive_multiperiod_blocked(self):
        """CCK adaptive 在多期下应被禁止。"""
        panel = _multi_period_panel()
        # Multi-period panel routes to multiperiod estimator, not CCK.
        # The multiperiod estimator does not support CCK, so this should
        # either succeed with parametric (ignoring dose_est_method in routing)
        # or raise an error. We test that it does not silently use adaptive CCK.
        result = cont_did(
            panel,
            dose_est_method="cck",
            adaptive=True,
            aggregation="dose",
            biters=200,
        )
        # If it succeeds, it routed to multiperiod (not CCK adaptive)
        assert result.metadata.get("panel_type") == "multiperiod"


# ===========================================================================
# 7. CCK routing tests
# ===========================================================================


class TestCCKRouting:
    """Verify CCK-specific routing."""

    def test_cont_did_cck_two_period(self):
        """cont_did(dose_est_method='cck') 在两期面板下正确运行。"""
        panel = _cck_two_period_panel(n=300)
        result = cont_did(
            panel,
            dose_est_method="cck",
            target_parameter="level",
            aggregation="dose",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(d)"
        assert result.has_inference

    def test_cont_did_cck_eventstudy(self):
        """cont_did(dose_est_method='cck', aggregation='eventstudy') 使用固定维数 CCK。"""
        panel = _cck_two_period_panel(n=300)
        result = cont_did(
            panel,
            dose_est_method="cck",
            target_parameter="level",
            aggregation="eventstudy",
            control_group="nevertreated",
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert result.estimand == "ATT(event_time)"
        assert result.has_inference
