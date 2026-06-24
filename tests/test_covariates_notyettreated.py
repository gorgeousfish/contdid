"""Tests for covariate adjustment and notyettreated control group correctness.

Covers:
1. notyettreated in two-period dose aggregation
2. Event-study covariate support
3. Two-period covariate correctness
4. Theoretical boundary gating
5. Unified API routing consistency
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
    estimate_dose_level_effects,
    estimate_dose_slope_effects,
    estimate_eventstudy_effects,
    estimate_eventstudy_slope_effects,
)
from contdid.validation import ContDIDValidationError


# ---------------------------------------------------------------------------
# Helpers: panel data construction
# ---------------------------------------------------------------------------

def _make_two_period_panel(
    n: int = 200,
    seed: int = 42,
    *,
    add_covariate: bool = False,
    covariate_effect: float = 2.0,
    treatment_effect: float = 0.0,
    notyettreated_group: bool = False,
) -> pd.DataFrame:
    """Construct a balanced two-period panel for dose estimation.

    Parameters
    ----------
    n : int
        Number of units.
    seed : int
        Random seed.
    add_covariate : bool
        If True, add a continuous covariate X that partially determines Y.
    covariate_effect : float
        Coefficient of X in the outcome equation.
    treatment_effect : float
        Linear dose effect coefficient (tau).
    notyettreated_group : bool
        If True, include a "not-yet-treated" group (G=4) alongside never-treated (G=0)
        and treated (G=2). The panel has time_period in {1,2}.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)

    # Assign groups
    if notyettreated_group:
        # 3 groups: never(0), treated-now(2), not-yet(4)
        group_probs = [0.4, 0.4, 0.2]
        groups = rng.choice([0, 2, 4], size=n, p=group_probs)
    else:
        # 2 groups: never(0), treated(2)
        groups = np.where(rng.random(n) < 0.5, 0, 2)

    # Dose: positive for treated at G=2, zero for others
    dose = np.where(groups == 2, rng.uniform(0.5, 5.0, size=n), 0.0)

    # Unit fixed effects
    alpha_i = rng.normal(0, 1, size=n)
    # Time effects
    lambda_t = np.array([0.0, 1.0])

    # Covariate
    X = rng.normal(0, 1, size=n) if add_covariate else np.zeros(n)

    rows = []
    for t_idx, t in enumerate([1, 2]):
        eps = rng.normal(0, 1, size=n)
        # Treatment effect only in period 2 for treated units (G=2)
        tau = np.where((groups == 2) & (t == 2), treatment_effect * dose, 0.0)
        Y = alpha_i + lambda_t[t_idx] + covariate_effect * X + tau + eps
        for i in range(n):
            row = {
                "id": int(ids[i]),
                "time_period": t,
                "Y": float(Y[i]),
                "G": int(groups[i]),
                "D": float(dose[i]),
            }
            if add_covariate:
                row["X"] = float(X[i])
            rows.append(row)

    return pd.DataFrame(rows)


def _make_multigroup_panel(
    n_per_group: int = 80,
    seed: int = 123,
) -> pd.DataFrame:
    """Construct a multi-group panel with never(0), early(2), late(4) groups.

    Time periods: 1, 2, 3, 4.
    - G=0: never treated
    - G=2: treated starting at t=2
    - G=4: treated starting at t=4

    This panel has >2 periods, suitable for multiperiod estimation or
    notyettreated logic verification.
    """
    rng = np.random.default_rng(seed)
    groups = [0] * n_per_group + [2] * n_per_group + [4] * n_per_group
    n = len(groups)
    ids = list(range(1, n + 1))
    groups = np.array(groups)
    dose = np.where(groups > 0, rng.uniform(0.5, 4.0, size=n), 0.0)
    alpha_i = rng.normal(0, 1, size=n)

    rows = []
    for t in [1, 2, 3, 4]:
        eps = rng.normal(0, 0.5, size=n)
        Y = alpha_i + 0.5 * t + eps
        for i in range(n):
            rows.append({
                "id": ids[i],
                "time_period": t,
                "Y": float(Y[i]),
                "G": int(groups[i]),
                "D": float(dose[i]),
            })
    return pd.DataFrame(rows)


def _make_eventstudy_panel(
    n_per_group: int = 60,
    seed: int = 77,
    *,
    add_covariate: bool = False,
    covariate_effect: float = 2.0,
    treatment_effect: float = 0.0,
) -> pd.DataFrame:
    """Construct a staggered panel suitable for event-study estimation.

    Groups: G=0 (never), G=3 (treated at t=3), G=5 (treated at t=5)
    Time: 1..6
    """
    rng = np.random.default_rng(seed)
    groups = [0] * n_per_group + [3] * n_per_group + [5] * n_per_group
    n = len(groups)
    ids = list(range(1, n + 1))
    groups = np.array(groups)
    dose = np.where(groups > 0, rng.uniform(0.5, 4.0, size=n), 0.0)
    alpha_i = rng.normal(0, 1, size=n)
    X = rng.normal(0, 1, size=n) if add_covariate else np.zeros(n)

    rows = []
    for t in range(1, 7):
        eps = rng.normal(0, 1, size=n)
        # Treatment effect: for treated units, after their treatment time
        tau = np.zeros(n)
        if treatment_effect != 0.0:
            for i in range(n):
                if groups[i] > 0 and t >= groups[i]:
                    tau[i] = treatment_effect * dose[i]
        Y = alpha_i + 0.3 * t + covariate_effect * X + tau + eps
        for i in range(n):
            row = {
                "id": ids[i],
                "time_period": t,
                "Y": float(Y[i]),
                "G": int(groups[i]),
                "D": float(dose[i]),
            }
            if add_covariate:
                row["X"] = float(X[i])
            rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# 1. notyettreated in two-period dose aggregation
# ===========================================================================


class TestNotyettreatedDoseLevel:
    """notyettreated control group in dose-level (ATT(d)) estimation."""

    def test_notyettreated_dose_level_runs(self):
        """notyettreated 对照组在 dose level 估计中应正常运行。"""
        df = _make_two_period_panel(n=200, seed=10, notyettreated_group=True)
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="notyettreated",
        )
        result = estimate_dose_level_effects(panel, spec, degree=2)
        assert isinstance(result, ContDIDResult)
        assert len(result.estimate) > 0
        assert len(result.std_error) > 0

    def test_notyettreated_dose_slope_runs(self):
        """notyettreated 对照组在 dose slope 估计中应正常运行。"""
        df = _make_two_period_panel(n=200, seed=11, notyettreated_group=True)
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="notyettreated",
        )
        result = estimate_dose_slope_effects(panel, spec, degree=2)
        assert isinstance(result, ContDIDResult)
        assert len(result.estimate) > 0

    def test_notyettreated_vs_nevertreated_similar(self):
        """notyettreated 和 nevertreated 在同一数据上结果应接近。

        当所有对照确实都是 nevertreated 时，两种 control_group 策略
        应产生非常相似的结果。
        """
        # Only never-treated and treated groups (no "not yet" units)
        df = _make_two_period_panel(n=300, seed=20, notyettreated_group=False)
        panel = PanelData(frame=df)

        spec_never = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        spec_notyet = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="notyettreated",
        )
        result_never = estimate_dose_level_effects(panel, spec_never, degree=2)
        result_notyet = estimate_dose_level_effects(panel, spec_notyet, degree=2)

        # When there are no "not yet" units, results should be identical
        np.testing.assert_allclose(
            result_never.estimate, result_notyet.estimate, atol=1e-10
        )

    def test_notyettreated_dose_zero_effect_unbiased(self):
        """零效应 DGP 下，notyettreated dose 估计应接近 0。"""
        df = _make_two_period_panel(
            n=400, seed=30, notyettreated_group=True, treatment_effect=0.0
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="notyettreated",
        )
        result = estimate_dose_level_effects(panel, spec, degree=1)
        estimates = np.array(result.estimate)
        # Under zero effect, mean estimate should be close to 0
        assert abs(np.mean(estimates)) < 1.5, (
            f"Mean estimate = {np.mean(estimates):.3f}, expected close to 0"
        )

    def test_notyettreated_dose_has_inference(self):
        """notyettreated dose 结果应包含有效推断（SE > 0）。"""
        df = _make_two_period_panel(n=200, seed=31, notyettreated_group=True)
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="notyettreated",
        )
        result = estimate_dose_level_effects(panel, spec, degree=2)
        se = np.array(result.std_error)
        assert np.all(se > 0), "All standard errors should be positive"

    def test_notyettreated_with_multigroup_panel(self):
        """在有多个处理组的面板中，notyettreated 应正确使用尚未处理的组作为对照。

        构造：groups = {0: never, 2: treated at t=2, 4: treated at t=4}
        对于 g=2 在 t=2：notyettreated 应包括 g=0 和 g=4 作为对照。
        使用 cont_did 的 multiperiod 路径。
        """
        df = _make_multigroup_panel(n_per_group=80, seed=32)
        panel = PanelData(frame=df)
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            control_group="notyettreated",
            degree=2,
            biters=200,
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.estimate) > 0
        # Metadata should indicate multiperiod panel
        assert result.metadata.get("panel_type") == "multiperiod"


# ===========================================================================
# 2. Event-study covariate tests
# ===========================================================================


class TestEventstudyCovariates:
    """Event-study estimation with covariate adjustment."""

    def test_eventstudy_covariates_level(self):
        """事件研究 level 目标传入协变量应抛出 NotImplementedError。

        论文(arXiv:2107.02637v7)未提供协变量情形的完整估计理论。
        """
        df = _make_eventstudy_panel(
            n_per_group=60, seed=40, add_covariate=True
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_eventstudy_effects(panel, spec, degree=2)

    def test_eventstudy_covariates_slope(self):
        """事件研究 slope 目标传入协变量应抛出 NotImplementedError。"""
        df = _make_eventstudy_panel(
            n_per_group=60, seed=41, add_covariate=True
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="slope",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_eventstudy_slope_effects(panel, spec, degree=2)

    def test_eventstudy_covariates_reduces_variance(self):
        """协变量调整当前被禁用，传入协变量应抛出 NotImplementedError。"""
        rng = np.random.default_rng(42)
        n_per_group = 80
        groups_arr = [0] * n_per_group + [3] * n_per_group + [5] * n_per_group
        n = len(groups_arr)
        ids = list(range(1, n + 1))
        groups_arr = np.array(groups_arr)
        dose = np.where(groups_arr > 0, rng.uniform(0.5, 4.0, size=n), 0.0)
        alpha_i = rng.normal(0, 1, size=n)
        X = rng.normal(0, 1, size=n)

        rows = []
        for t in range(1, 7):
            eps = rng.normal(0, 1, size=n)
            Y = alpha_i + 0.3 * t + 3.0 * X * t + eps
            for i in range(n):
                rows.append({
                    "id": ids[i], "time_period": t,
                    "Y": float(Y[i]), "G": int(groups_arr[i]),
                    "D": float(dose[i]), "X": float(X[i]),
                })
        df = pd.DataFrame(rows)
        panel = PanelData(frame=df)

        # Without covariates should still work
        spec_no_cov = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        result_no_cov = estimate_eventstudy_effects(panel, spec_no_cov, degree=1)
        assert len(result_no_cov.estimate) > 0

        # With covariates should raise NotImplementedError
        spec_cov = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_eventstudy_effects(panel, spec_cov, degree=1)

    def test_eventstudy_covariates_zero_effect(self):
        """传入协变量应抛出 NotImplementedError。"""
        df = _make_eventstudy_panel(
            n_per_group=80, seed=43, add_covariate=True,
            treatment_effect=0.0,
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_eventstudy_effects(panel, spec, degree=1)

    def test_eventstudy_covariates_preserved_in_metadata(self):
        """传入协变量应抛出 NotImplementedError（无法到达 metadata 阶段）。"""
        df = _make_eventstudy_panel(
            n_per_group=60, seed=44, add_covariate=True
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_eventstudy_effects(panel, spec, degree=2)


# ===========================================================================
# 3. Two-period covariate correctness
# ===========================================================================


class TestTwoPeriodCovariates:
    """Two-period dose estimation with covariates."""

    def test_two_period_covariates_basic(self):
        """两期 dose 估计中传入协变量应抛出 NotImplementedError。"""
        df = _make_two_period_panel(
            n=200, seed=50, add_covariate=True, treatment_effect=1.0
        )
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_dose_level_effects(panel, spec, degree=2)

    def test_two_period_covariates_reduces_se(self):
        """协变量调整当前被禁用，传入协变量应抛出 NotImplementedError。

        无协变量的路径应正常运行。
        """
        rng = np.random.default_rng(51)
        n = 300
        ids = np.arange(1, n + 1)
        groups = np.where(rng.random(n) < 0.5, 0, 2)
        dose = np.where(groups == 2, rng.uniform(0.5, 5.0, size=n), 0.0)
        alpha_i = rng.normal(0, 1, size=n)
        X = rng.normal(0, 1, size=n)

        rows = []
        for t_idx, t in enumerate([1, 2]):
            eps = rng.normal(0, 1, size=n)
            tau_effect = np.where((groups == 2) & (t == 2), 1.0 * dose, 0.0)
            Y = alpha_i + 0.5 * t + 3.0 * X * t + tau_effect + eps
            for i in range(n):
                rows.append({
                    "id": int(ids[i]), "time_period": t,
                    "Y": float(Y[i]), "G": int(groups[i]),
                    "D": float(dose[i]), "X": float(X[i]),
                })
        df = pd.DataFrame(rows)
        panel = PanelData(frame=df)

        # Without covariates should work
        spec_no_cov = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
        )
        result_no_cov = estimate_dose_level_effects(panel, spec_no_cov, degree=2)
        assert len(result_no_cov.estimate) > 0

        # With covariates should raise NotImplementedError
        spec_cov = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_dose_level_effects(panel, spec_cov, degree=2)

    def test_two_period_covariates_dose_curve_unchanged(self):
        """协变量调整当前被禁用，传入协变量应抛出 NotImplementedError。"""
        df = _make_two_period_panel(
            n=400, seed=52, add_covariate=True,
            covariate_effect=2.0, treatment_effect=1.0,
        )
        panel = PanelData(frame=df)

        dvals = [1.0, 2.0, 3.0, 4.0]
        spec_cov = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="nevertreated",
            covariates=("X",),
        )
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            estimate_dose_level_effects(
                panel, spec_cov, degree=1, dvals=dvals
            )


# ===========================================================================
# 4. Theoretical boundary tests
# ===========================================================================


class TestTheoreticalBoundaries:
    """Gating tests for unsupported theoretical combinations."""

    def test_multiperiod_covariates_still_blocked(self):
        """多期协变量仍应抛出 NotImplementedError（理论不支持）。

        直接调用 estimate_multiperiod_dose 来验证理论门控，
        因为 cont_did 的路由层可能未传递 covariates 参数。
        """
        from contdid.multiperiod import estimate_multiperiod_dose

        df = _make_multigroup_panel(n_per_group=50, seed=60)
        # Add a covariate column
        df["X"] = np.random.default_rng(60).normal(0, 1, size=len(df))

        with pytest.raises(NotImplementedError, match="[Cc]ovariate"):
            estimate_multiperiod_dose(
                panel_df=df,
                id_column="id",
                time_column="time_period",
                outcome_column="Y",
                dose_column="D",
                group_column="G",
                control_group="nevertreated",
                target="level",
                degree=2,
                covariates=["X"],
            )

    def test_notyettreated_invalid_raises(self):
        """无效的 control_group 值应抛出验证错误。"""
        df = _make_two_period_panel(n=50, seed=61)
        panel = PanelData(frame=df)
        spec = ContDIDSpec(
            target_parameter="level",
            aggregation="dose",
            dose_est_method="parametric",
            control_group="invalidgroup",
        )
        with pytest.raises(ContDIDValidationError):
            estimate_dose_level_effects(panel, spec, degree=2)


# ===========================================================================
# 5. Unified API routing consistency
# ===========================================================================


class TestUnifiedAPIRouting:
    """Verify cont_did() correctly routes notyettreated and covariates."""

    def test_cont_did_notyettreated_routes_correctly(self):
        """统一 API cont_did() 应正确路由 notyettreated dose 请求。"""
        df = _make_two_period_panel(n=200, seed=70, notyettreated_group=True)
        panel = PanelData(frame=df)
        result = cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            control_group="notyettreated",
            degree=2,
            biters=100,
        )
        assert isinstance(result, ContDIDResult)
        assert len(result.estimate) > 0

    def test_cont_did_eventstudy_with_covariates(self):
        """统一 API cont_did() 传入协变量应抛出 NotImplementedError。"""
        df = _make_eventstudy_panel(
            n_per_group=60, seed=71, add_covariate=True
        )
        panel = PanelData(frame=df)
        with pytest.raises(NotImplementedError, match="Covariate conditioning is not available"):
            cont_did(
                panel,
                target_parameter="level",
                aggregation="eventstudy",
                control_group="nevertreated",
                covariates=("X",),
                degree=2,
                biters=100,
            )
