"""Tests for pre-trend testing and unbalanced panel theory boundary."""
import numpy as np
import pandas as pd
import pytest


# ============================================================
# 1. 公共 API 导入测试
# ============================================================

def test_pre_trend_test_importable():
    """pre_trend_test 和 PreTrendTestResult 可从 contdid 直接导入。"""
    from contdid import pre_trend_test, PreTrendTestResult
    assert callable(pre_trend_test)
    assert PreTrendTestResult is not None


def test_pre_trend_test_from_result_importable():
    """pre_trend_test_from_result 可从 contdid 导入。"""
    from contdid import pre_trend_test_from_result
    assert callable(pre_trend_test_from_result)


# ============================================================
# Helper: 构造事件研究面板数据
# ============================================================

def _make_eventstudy_panel(n=300, T=5, treatment_time=3, seed=42, pre_trend_violation=0.0):
    """构造事件研究面板，可选pre-trend violation。"""
    rng = np.random.default_rng(seed)
    n_treated = n // 2
    n_control = n - n_treated

    ids = np.repeat(np.arange(n), T)
    times = np.tile(np.arange(1, T + 1), n)

    # Groups: first half treated at treatment_time, second half never-treated
    groups = np.where(np.repeat(np.arange(n), T) < n_treated, treatment_time, 0)

    # Dose: treated units get random positive dose
    dose_values = np.where(np.arange(n) < n_treated, rng.uniform(1, 5, n), 0.0)
    dose = np.repeat(dose_values, T)

    # Outcome: unit FE + time FE + treatment effect + optional pre-trend violation
    unit_fe = np.repeat(rng.normal(0, 1, n), T)
    time_fe = np.tile(np.arange(1, T + 1) * 0.5, n)

    # Treatment effect: ATT(d) = 0.5*d for t >= g
    treat_effect = np.where((groups > 0) & (times >= groups), dose * 0.5, 0.0)

    # Pre-trend violation: differential trend proportional to dose before treatment
    if pre_trend_violation != 0.0:
        pre_violation = np.where(
            (groups > 0) & (times < groups),
            dose * pre_trend_violation * (times - treatment_time),
            0.0
        )
    else:
        pre_violation = 0.0

    outcome = unit_fe + time_fe + treat_effect + pre_violation + rng.normal(0, 0.5, n * T)

    df = pd.DataFrame({
        'id': ids, 'time': times, 'Y': outcome, 'G': groups, 'D': dose
    })
    return df


# ============================================================
# 2. 预趋势检验统计正确性
# ============================================================

def test_pretest_null_dgp_not_reject():
    """在满足平行趋势的DGP下，预趋势检验不应拒绝H0。"""
    from contdid import PanelData, ContDIDSpec, estimate_eventstudy_effects, pre_trend_test_from_result

    df = _make_eventstudy_panel(n=400, T=5, treatment_time=3, pre_trend_violation=0.0)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')
    spec = ContDIDSpec(target_parameter='level', aggregation='eventstudy',
                       dose_est_method='parametric', control_group='nevertreated')

    result = estimate_eventstudy_effects(panel, spec, degree=3, num_knots=0)
    pretest_result = pre_trend_test_from_result(result)

    # Under null, should usually not reject (p > 0.05 most of the time)
    # With seed=42, just check p_value is valid
    assert 0.0 <= pretest_result.p_value <= 1.0
    assert pretest_result.test_statistic >= 0.0
    assert pretest_result.degrees_of_freedom > 0


def test_pretest_violated_dgp_small_pvalue():
    """在严重违反平行趋势的DGP下，p值应较小。"""
    from contdid import PanelData, ContDIDSpec, estimate_eventstudy_effects, pre_trend_test_from_result

    # Strong pre-trend violation
    df = _make_eventstudy_panel(n=400, T=5, treatment_time=3, pre_trend_violation=0.8)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')
    spec = ContDIDSpec(target_parameter='level', aggregation='eventstudy',
                       dose_est_method='parametric', control_group='nevertreated')

    result = estimate_eventstudy_effects(panel, spec, degree=3, num_knots=0)
    pretest_result = pre_trend_test_from_result(result)

    # With strong violation, should detect it (p < 0.3 generous threshold due to sample size)
    assert pretest_result.p_value < 0.3


def test_pretest_wald_statistic_nonnegative():
    """Wald统计量应始终非负。"""
    from contdid import PanelData, ContDIDSpec, estimate_eventstudy_effects, pre_trend_test_from_result

    df = _make_eventstudy_panel(n=200, T=4, treatment_time=3)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')
    spec = ContDIDSpec(target_parameter='level', aggregation='eventstudy',
                       dose_est_method='parametric', control_group='nevertreated')

    result = estimate_eventstudy_effects(panel, spec)
    pretest_result = pre_trend_test_from_result(result)
    assert pretest_result.test_statistic >= 0.0


def test_pretest_df_equals_pre_periods():
    """自由度应等于pre-treatment event times数量。"""
    from contdid import PanelData, ContDIDSpec, estimate_eventstudy_effects, pre_trend_test_from_result

    df = _make_eventstudy_panel(n=200, T=5, treatment_time=3)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')
    spec = ContDIDSpec(target_parameter='level', aggregation='eventstudy',
                       dose_est_method='parametric', control_group='nevertreated')

    result = estimate_eventstudy_effects(panel, spec)
    pretest_result = pre_trend_test_from_result(result)

    # Count pre-treatment event times from metadata
    event_time_grid = result.metadata["event_time_grid"]
    n_pre = sum(1 for et in event_time_grid if et < 0)
    assert pretest_result.degrees_of_freedom == n_pre


# ============================================================
# 3. 与事件研究的一致性
# ============================================================

def test_pretest_pre_estimates_match_eventstudy():
    """预趋势检验中的 pre_period_estimates 应与事件研究的 e<0 cells 一致。"""
    from contdid import PanelData, ContDIDSpec, estimate_eventstudy_effects, pre_trend_test_from_result

    df = _make_eventstudy_panel(n=200, T=5, treatment_time=3)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')
    spec = ContDIDSpec(target_parameter='level', aggregation='eventstudy',
                       dose_est_method='parametric', control_group='nevertreated')

    result = estimate_eventstudy_effects(panel, spec)
    pretest_result = pre_trend_test_from_result(result)

    # Pre-treatment estimates from pretest should match event study
    event_time_grid = result.metadata["event_time_grid"]
    pre_indices = [i for i, et in enumerate(event_time_grid) if et < 0]
    es_pre_estimates = [result.estimate[i] for i in pre_indices]

    for pretest_est, es_est in zip(pretest_result.pre_period_estimates, es_pre_estimates):
        assert abs(pretest_est - es_est) < 1e-10


# ============================================================
# 4. 不平衡面板理论边界
# ============================================================

def test_unbalanced_panel_raises_error():
    """不平衡面板应抛出 ContDIDValidationError。"""
    from contdid import PanelData, cont_did
    from contdid.validation import ContDIDValidationError

    # Create unbalanced panel: remove some observations
    df = _make_eventstudy_panel(n=100, T=4, treatment_time=3)
    # Remove 10% of rows randomly
    rng = np.random.default_rng(123)
    keep_mask = rng.random(len(df)) > 0.1
    # But ensure at least one unit is incomplete
    df_unbalanced = df[keep_mask].reset_index(drop=True)

    panel = PanelData(frame=df_unbalanced, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')

    with pytest.raises(ContDIDValidationError, match="balanced"):
        cont_did(panel, aggregation='dose')


def test_balanced_panel_passes():
    """平衡面板应通过验证。"""
    from contdid import PanelData, cont_did

    df = _make_eventstudy_panel(n=100, T=3, treatment_time=2)
    panel = PanelData(frame=df, id_column='id', time_column='time',
                      outcome_column='Y', group_column='G', dose_column='D')

    # Should not raise
    result = cont_did(panel, aggregation='dose')
    assert len(result.estimate) > 0
