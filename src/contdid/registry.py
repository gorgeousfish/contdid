"""估计器注册机制 - 支持动态注册和扩展估计方法。

本模块实现 Registry Pattern，将 cont_did() 的硬编码 if-elif 路由
替换为可扩展的注册表查找，便于第三方扩展新的估计方法。

理论基础
--------
当前内置两条估计路径：
- parametric: B样条参数化估计 (arXiv-2107.02637v7, Theorem 3.1)
- cck: CCK非参数筛估计 (arXiv-2107.11869v3, Theorem 2)
"""

from __future__ import annotations

from typing import Any, Protocol

from .base_estimator import BaseEstimator
from .data import PanelData
from .results import ContDIDResult
from .specs import ContDIDSpec


class EstimatorProtocol(Protocol):
    """估计器协议 - 所有估计方法必须实现此接口。

    使用 Protocol（结构化子类型）而非抽象基类，
    更符合 Python 鸭子类型风格，无需显式继承。
    """

    @property
    def name(self) -> str:
        """估计器唯一标识名。"""
        ...

    @property
    def supports_multiperiod(self) -> bool:
        """是否支持多期/事件研究设定。"""
        ...

    @property
    def supports_adaptive(self) -> bool:
        """是否支持 Lepski 自适应维度选择。"""
        ...

    def validate_spec(self, spec: ContDIDSpec, panel: PanelData) -> list[str]:
        """验证 ContDIDSpec 和 PanelData 是否满足此估计器的约束。

        Parameters
        ----------
        spec : ContDIDSpec
            估计规范对象。
        panel : PanelData
            验证后的面板数据。

        Returns
        -------
        list[str]
            错误消息列表，空列表表示验证通过。
        """
        ...

    def estimate(self, panel: PanelData, spec: ContDIDSpec, **kwargs: Any) -> ContDIDResult:
        """执行估计，返回 ContDIDResult。

        Parameters
        ----------
        panel : PanelData
            验证后的面板数据。
        spec : ContDIDSpec
            估计规范对象。
        **kwargs
            传递给底层估计函数的额外参数（dvals, degree, num_knots 等）。

        Returns
        -------
        ContDIDResult
            包含点估计、标准误、置信区间/带和元数据的完整结果。
        """
        ...


class EstimatorRegistry:
    """估计器注册表 - 管理可用的估计方法。

    提供注册、查找和列举功能，支持第三方通过 register() 扩展。

    Examples
    --------
    >>> from contdid.registry import EstimatorRegistry
    >>> registry = EstimatorRegistry()
    >>> registry.list_available()
    ['parametric', 'cck']
    >>> estimator = registry.get("parametric")
    >>> estimator.supports_multiperiod
    True
    """

    def __init__(self) -> None:
        self._estimators: dict[str, Any] = {}

    def register(self, estimator: Any) -> None:
        """注册一个估计器实例。

        Parameters
        ----------
        estimator
            必须实现 EstimatorProtocol 协议的对象：
            具有 name, supports_multiperiod, supports_adaptive 属性，
            以及 validate_spec() 和 estimate() 方法。

        Raises
        ------
        TypeError
            如果估计器缺少必需的属性或方法。
        ValueError
            如果已存在同名的估计器。
        """
        # 验证必需接口
        missing = []
        for attr in ("name", "supports_multiperiod", "supports_adaptive"):
            if not hasattr(estimator, attr):
                missing.append(attr)
        for method in ("validate_spec", "estimate"):
            if not hasattr(estimator, method) or not callable(getattr(estimator, method)):
                missing.append(f"{method}()")
        if missing:
            raise TypeError(
                f"估计器缺少必需的接口成员: {', '.join(missing)}。"
                f"请实现 EstimatorProtocol 协议。"
            )

        name = estimator.name
        if name in self._estimators:
            raise ValueError(
                f"估计器 {name!r} 已注册。如需替换，请先调用 unregister({name!r})。"
            )
        self._estimators[name] = estimator

    def unregister(self, name: str) -> None:
        """移除已注册的估计器。

        Parameters
        ----------
        name : str
            估计器标识名。

        Raises
        ------
        KeyError
            如果指定名称的估计器不存在。
        """
        if name not in self._estimators:
            raise KeyError(f"估计器 {name!r} 未注册。")
        del self._estimators[name]

    def get(self, name: str) -> Any:
        """获取已注册的估计器。

        Parameters
        ----------
        name : str
            估计器标识名。

        Returns
        -------
        估计器对象（满足 EstimatorProtocol 协议）。

        Raises
        ------
        KeyError
            如果指定名称的估计器不存在，提供可用方法列表。
        """
        if name not in self._estimators:
            available = ", ".join(sorted(self._estimators.keys()))
            raise KeyError(
                f"未知的估计方法: {name!r}。"
                f"可用方法: [{available}]"
            )
        return self._estimators[name]

    def list_available(self) -> list[str]:
        """列出所有已注册的估计器名称。

        Returns
        -------
        list[str]
            已注册估计器名称的排序列表。
        """
        return sorted(self._estimators.keys())

    def __contains__(self, name: str) -> bool:
        """检查估计器是否已注册。"""
        return name in self._estimators

    def __len__(self) -> int:
        """返回已注册估计器数量。"""
        return len(self._estimators)


# ---------------------------------------------------------------------------
# 内置估计器适配器
# ---------------------------------------------------------------------------


class ParametricEstimator:
    """参数化 B样条估计器适配器。

    包装 estimation.py 中的参数化估计逻辑，支持：
    - 二期剂量响应估计 (ATT(d) / ACRT(d))
    - 多期剂量响应聚合
    - 事件研究聚合

    理论基础: arXiv-2107.02637v7, Theorem 3.1
    """

    @property
    def name(self) -> str:
        """估计器唯一标识名。"""
        return "parametric"

    @property
    def supports_multiperiod(self) -> bool:
        """支持多期/事件研究设定。"""
        return True

    @property
    def supports_adaptive(self) -> bool:
        """不支持 Lepski 自适应维度选择。"""
        return False

    def validate_spec(self, spec: ContDIDSpec, panel: PanelData) -> list[str]:
        """验证规范满足参数化估计器的约束。"""
        errors: list[str] = []
        if spec.dose_est_method != "parametric":
            errors.append(
                f"ParametricEstimator 仅处理 dose_est_method='parametric'，"
                f"收到 {spec.dose_est_method!r}"
            )
        return errors

    def estimate(self, panel: PanelData, spec: ContDIDSpec, **kwargs: Any) -> ContDIDResult:
        """执行参数化 B样条估计。

        Parameters
        ----------
        panel : PanelData
            验证后的面板数据。
        spec : ContDIDSpec
            估计规范（dose_est_method 应为 "parametric"）。
        **kwargs
            支持的关键字参数：
            - dvals: 剂量网格评估点
            - degree: B样条阶数（默认 3）
            - num_knots: 内部结点数（默认 0）
            - base_period: 事件研究基期
        """
        from .api import _route_eventstudy, _route_multiperiod_dose, _route_two_period_dose

        n_periods = panel.frame[panel.time_column].nunique()
        is_multiperiod = n_periods > 2

        if spec.aggregation == "eventstudy":
            return _route_eventstudy(
                panel,
                spec,
                dvals=kwargs.get("dvals"),
                degree=kwargs.get("degree", 3),
                num_knots=kwargs.get("num_knots", 0),
                base_period=kwargs.get("base_period"),
            )
        elif is_multiperiod:
            return _route_multiperiod_dose(
                panel,
                spec,
                dvals=kwargs.get("dvals"),
                degree=kwargs.get("degree", 3),
                num_knots=kwargs.get("num_knots", 0),
            )
        else:
            return _route_two_period_dose(
                panel,
                spec,
                dvals=kwargs.get("dvals"),
                degree=kwargs.get("degree", 3),
                num_knots=kwargs.get("num_knots", 0),
                knot_method=kwargs.get("knot_method", "quantile"),
                adaptive=False,
                adaptive_k_min=None,
                adaptive_k_max=None,
                adaptive_seed=None,
            )


class CCKEstimator:
    """CCK 非参数筛估计器适配器。

    包装 cck.py 中的非参数估计逻辑，支持：
    - 二期剂量响应估计 (ATT(d) / ACRT(d))
    - 事件研究聚合（各 (g,t) 格子独立二期估计）
    - Lepski 自适应维度选择（仅限二期剂量响应）

    理论基础: arXiv-2107.11869v3, Theorem 2
    """

    @property
    def name(self) -> str:
        """估计器唯一标识名。"""
        return "cck"

    @property
    def supports_multiperiod(self) -> bool:
        """不支持多期剂量聚合（但事件研究内部各格子独立二期估计可用）。"""
        return False

    @property
    def supports_adaptive(self) -> bool:
        """支持 Lepski 自适应维度选择。"""
        return True

    def validate_spec(self, spec: ContDIDSpec, panel: PanelData) -> list[str]:
        """验证规范满足 CCK 估计器的约束。

        注意：CCK 不支持多期剂量聚合，但为保持向后兼容，
        多期场景由 estimate() 自动回退到参数化路径而非报错。
        """
        errors: list[str] = []
        if spec.dose_est_method != "cck":
            errors.append(
                f"CCKEstimator 仅处理 dose_est_method='cck'，"
                f"收到 {spec.dose_est_method!r}"
            )
        return errors

    def estimate(self, panel: PanelData, spec: ContDIDSpec, **kwargs: Any) -> ContDIDResult:
        """执行 CCK 非参数筛估计。

        Parameters
        ----------
        panel : PanelData
            验证后的面板数据。
        spec : ContDIDSpec
            估计规范（dose_est_method 应为 "cck"）。
        **kwargs
            支持的关键字参数：
            - dvals: 剂量网格评估点
            - degree: B样条阶数（默认 3）
            - num_knots: 内部结点数（默认 0）
            - base_period: 事件研究基期
            - adaptive: 是否使用 Lepski 自适应（默认 False）
            - adaptive_k_min, adaptive_k_max, adaptive_seed: 自适应参数
        """
        from .api import _route_eventstudy, _route_multiperiod_dose, _route_two_period_dose

        if spec.aggregation == "eventstudy":
            return _route_eventstudy(
                panel,
                spec,
                dvals=kwargs.get("dvals"),
                degree=kwargs.get("degree", 3),
                num_knots=kwargs.get("num_knots", 0),
                base_period=kwargs.get("base_period"),
            )

        # CCK 不支持多期剂量聚合，自动回退到参数化多期路径（保持向后兼容）
        n_periods = panel.frame[panel.time_column].nunique()
        if n_periods > 2:
            return _route_multiperiod_dose(
                panel,
                spec,
                dvals=kwargs.get("dvals"),
                degree=kwargs.get("degree", 3),
                num_knots=kwargs.get("num_knots", 0),
            )

        return _route_two_period_dose(
            panel,
            spec,
            dvals=kwargs.get("dvals"),
            degree=kwargs.get("degree", 3),
            num_knots=kwargs.get("num_knots", 0),
            knot_method=kwargs.get("knot_method", "quantile"),
            adaptive=kwargs.get("adaptive", False),
            adaptive_k_min=kwargs.get("adaptive_k_min"),
            adaptive_k_max=kwargs.get("adaptive_k_max"),
            adaptive_seed=kwargs.get("adaptive_seed"),
        )


# ---------------------------------------------------------------------------
# 模块级默认注册表实例
# ---------------------------------------------------------------------------

_default_registry = EstimatorRegistry()
_default_registry.register(ParametricEstimator())
_default_registry.register(CCKEstimator())


def get_default_registry() -> EstimatorRegistry:
    """获取模块级默认注册表实例（包含内置估计器）。

    Returns
    -------
    EstimatorRegistry
        包含 "parametric" 和 "cck" 两个内置估计器的注册表。
    """
    return _default_registry


def register_estimator(estimator: Any) -> None:
    """向默认注册表注册新的估计器。

    这是第三方扩展的主要入口点。支持两种方式：
    1. 实现 EstimatorProtocol 协议的对象（鸭子类型）
    2. BaseEstimator 子类实例（结构化继承）

    对于 BaseEstimator 子类，会自动包装为 Protocol 兼容接口后注册。

    Parameters
    ----------
    estimator
        实现 EstimatorProtocol 协议的对象，或 BaseEstimator 子类实例。

    Examples
    --------
    >>> from contdid.registry import register_estimator
    >>> class MyEstimator:
    ...     name = "my_method"
    ...     supports_multiperiod = True
    ...     supports_adaptive = False
    ...     def validate_spec(self, spec, panel): return []
    ...     def estimate(self, panel, spec, **kwargs): ...
    >>> register_estimator(MyEstimator())

    >>> from contdid.base_estimator import BaseEstimator, LinearDoseEstimator
    >>> register_estimator(LinearDoseEstimator())
    """
    if isinstance(estimator, BaseEstimator):
        estimator = _BaseEstimatorAdapter(estimator)
    _default_registry.register(estimator)


class _BaseEstimatorAdapter:
    """将 BaseEstimator 子类包装为 EstimatorProtocol 兼容接口。

    这是内部适配器，使 BaseEstimator 的继承式接口与现有
    Protocol 鸭子类型注册表无缝兼容。
    """

    def __init__(self, base_estimator: BaseEstimator) -> None:
        self._impl = base_estimator

    @property
    def name(self) -> str:
        return self._impl.name

    @property
    def supports_multiperiod(self) -> bool:
        return self._impl.capabilities.supports_multiperiod

    @property
    def supports_adaptive(self) -> bool:
        return self._impl.capabilities.supports_adaptive

    @property
    def base_estimator(self) -> BaseEstimator:
        """访问底层 BaseEstimator 实例。"""
        return self._impl

    def validate_spec(self, spec: ContDIDSpec, panel: PanelData) -> list[str]:
        return self._impl.validate_assumptions(panel, spec)

    def estimate(self, panel: PanelData, spec: ContDIDSpec, **kwargs: Any) -> ContDIDResult:
        """通过 BaseEstimator 执行估计并包装为 ContDIDResult。

        注意：此适配器提供基本的转换逻辑。复杂场景下（多期、事件研究），
        第三方开发者应直接实现 EstimatorProtocol 或在 BaseEstimator 子类中
        处理完整的 PanelData → ContDIDResult 流程。
        """
        from .estimation import _collapse_to_unit_differences

        # 提取处理组的 ΔY 和 dose
        collapsed = _collapse_to_unit_differences(panel)
        dose_col = panel.dose_column
        treated_mask = collapsed[dose_col] > 0
        dy = collapsed.loc[treated_mask, "delta_outcome"].values
        dose = collapsed.loc[treated_mask, dose_col].values

        # 减去未处理组均值（SPT 识别）
        untreated_mean = collapsed.loc[~treated_mask, "delta_outcome"].mean()
        dy_centered = dy - untreated_mean

        dvals = kwargs.get("dvals")
        if dvals is None:
            import numpy as _np
            dvals = _np.quantile(dose, _np.arange(0.10, 1.0, 0.01))
        dvals = __import__("numpy").asarray(dvals, dtype=float)

        degree = kwargs.get("degree", 3)
        num_knots = kwargs.get("num_knots", 0)
        target = spec.target_parameter if hasattr(spec, "target_parameter") else "level"

        result = self._impl.fit(
            dy_centered, dose, dvals,
            target_parameter=target, degree=degree, num_knots=num_knots,
            **{k: v for k, v in kwargs.items()
               if k not in ("dvals", "degree", "num_knots")},
        )

        # 构造 ContDIDResult
        from .results import ContDIDResult
        return ContDIDResult(
            spec=spec,
            dose_grid=dvals.tolist(),
            att_d=result.att_d.tolist(),
            se_d=[0.0] * len(dvals),  # 需要通过 influence_function + bootstrap 获得
            ci_lower=[0.0] * len(dvals),
            ci_upper=[0.0] * len(dvals),
            att_overall=result.att_overall,
            acrt_d=result.acrt_d.tolist() if result.acrt_d is not None else None,
            method=self._impl.name,
        )
