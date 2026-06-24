"""结果后处理钩子系统。

允许用户注册在估计完成后自动执行的回调函数，
用于摘要统计、可视化、格式转换等后处理操作。

设计原则：
- 钩子接收只读的结果视图，不可修改核心估计值
- 钩子的异常不会中断主流程（catch and warn）
- 支持优先级排序和条件触发

理论约束：
- ATT(d)、ACRT(d) 的点估计值不可被钩子修改
- 置信区间和置信带不可被钩子修改
- 影响函数矩阵不可被钩子修改
- 钩子仅用于读取结果并执行展示、摘要、导出等操作
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import warnings


class HookStage(Enum):
    """钩子触发时机。"""

    POST_ESTIMATION = auto()  # 估计完成后
    POST_INFERENCE = auto()  # 推断完成后（含置信区间）
    POST_AGGREGATION = auto()  # 聚合完成后（事件研究）


@dataclass(frozen=True)
class HookSpec:
    """钩子规范。

    Attributes
    ----------
    name : str
        钩子唯一名称。
    callback : Callable
        回调函数，接收 ReadOnlyResult 并返回任意值。
    stage : HookStage
        触发时机。
    priority : int
        优先级（越小越先执行）。
    condition : Callable or None
        条件触发函数，返回 True 时才执行钩子。
    """

    name: str
    callback: Callable[["ReadOnlyResult"], Any]
    stage: HookStage = HookStage.POST_INFERENCE
    priority: int = 100
    condition: Callable[["ReadOnlyResult"], bool] | None = None


class ReadOnlyResult:
    """ContDIDResult 的只读视图 - 防止钩子修改核心结果。

    暴露所有读取接口但禁止赋值操作。这是保证数学结果
    不被钩子篡改的关键安全机制。
    """

    __slots__ = ("_result",)

    def __init__(self, result: Any) -> None:
        object.__setattr__(self, "_result", result)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"ReadOnlyResult 不允许修改属性 '{name}'。"
            "钩子函数仅可读取结果，不可修改核心估计值。"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"ReadOnlyResult 不允许删除属性 '{name}'。"
            "钩子函数仅可读取结果，不可修改核心估计值。"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_result"), name)

    @property
    def estimand(self) -> str:
        """估计目标名称（只读）。"""
        return object.__getattribute__(self, "_result").estimand

    @property
    def grid(self) -> list[float]:
        """评估网格点（只读）。"""
        return object.__getattribute__(self, "_result").grid

    @property
    def estimate(self) -> list[float]:
        """点估计值（只读）。"""
        return object.__getattribute__(self, "_result").estimate

    @property
    def std_error(self) -> list[float]:
        """标准误（只读）。"""
        return object.__getattribute__(self, "_result").std_error

    @property
    def confidence_interval(self) -> list[list[float]] | None:
        """逐点置信区间（只读）。"""
        return object.__getattribute__(self, "_result").confidence_interval

    @property
    def confidence_band(self) -> dict[str, Any] | None:
        """同时置信带（只读）。"""
        return object.__getattribute__(self, "_result").confidence_band

    @property
    def critical_value(self) -> float | None:
        """临界值（只读）。"""
        return object.__getattribute__(self, "_result").critical_value

    @property
    def metadata(self) -> dict[str, Any]:
        """元数据（只读）。"""
        return object.__getattribute__(self, "_result").metadata

    @property
    def event_time(self) -> list[int] | None:
        """事件时间索引（只读）。"""
        return object.__getattribute__(self, "_result").event_time

    @property
    def event_time_grid(self) -> list[int] | None:
        """事件时间网格（只读）。"""
        return object.__getattribute__(self, "_result").event_time_grid

    @property
    def has_inference(self) -> bool:
        """是否包含推断结果（只读）。"""
        return object.__getattribute__(self, "_result").has_inference


class HookRegistry:
    """钩子注册表 - 管理后处理钩子。

    支持注册、移除、按优先级执行钩子，以及条件触发。
    单个钩子的异常不会阻止其他钩子执行。
    """

    def __init__(self) -> None:
        self._hooks: list[HookSpec] = []

    def register(
        self,
        name: str,
        callback: Callable[["ReadOnlyResult"], Any],
        *,
        stage: HookStage = HookStage.POST_INFERENCE,
        priority: int = 100,
        condition: Callable[["ReadOnlyResult"], bool] | None = None,
    ) -> None:
        """注册后处理钩子。

        Parameters
        ----------
        name : str
            钩子唯一名称。如果名称已存在则覆盖。
        callback : Callable
            回调函数，接收 ReadOnlyResult 返回任意值。
        stage : HookStage
            触发时机，默认 POST_INFERENCE。
        priority : int
            优先级，越小越先执行，默认 100。
        condition : Callable or None
            条件函数，返回 True 时才执行钩子。
        """
        # 移除同名旧钩子（覆盖语义）
        self._hooks = [h for h in self._hooks if h.name != name]
        spec = HookSpec(
            name=name,
            callback=callback,
            stage=stage,
            priority=priority,
            condition=condition,
        )
        self._hooks.append(spec)

    def unregister(self, name: str) -> None:
        """移除已注册的钩子。

        Parameters
        ----------
        name : str
            要移除的钩子名称。

        Raises
        ------
        KeyError
            如果指定名称的钩子不存在。
        """
        original_count = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        if len(self._hooks) == original_count:
            raise KeyError(f"钩子 '{name}' 未注册")

    def execute(self, result: Any, stage: HookStage) -> dict[str, Any]:
        """执行指定阶段的所有钩子，返回各钩子输出。

        异常处理：单个钩子的异常不会阻止其他钩子执行，
        失败的钩子会通过 warnings.warn() 报告。

        Parameters
        ----------
        result : ContDIDResult
            估计结果对象。
        stage : HookStage
            当前触发阶段。

        Returns
        -------
        dict[str, Any]
            各钩子的输出，键为钩子名称。
        """
        outputs: dict[str, Any] = {}
        readonly = ReadOnlyResult(result)

        # 按优先级排序（稳定排序，相同优先级保持注册顺序）
        staged_hooks = sorted(
            (h for h in self._hooks if h.stage == stage),
            key=lambda h: h.priority,
        )

        for hook in staged_hooks:
            try:
                # 检查条件
                if hook.condition is not None:
                    if not hook.condition(readonly):
                        continue

                # 执行回调
                output = hook.callback(readonly)
                outputs[hook.name] = output
            except Exception as exc:
                warnings.warn(
                    f"钩子 '{hook.name}' 执行失败: {exc!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return outputs

    def list_hooks(self) -> list[str]:
        """列出所有已注册钩子名称。

        Returns
        -------
        list[str]
            按注册顺序排列的钩子名称列表。
        """
        return [h.name for h in self._hooks]

    def clear(self) -> None:
        """清除所有钩子。"""
        self._hooks.clear()

    def __len__(self) -> int:
        return len(self._hooks)

    def __repr__(self) -> str:
        return f"HookRegistry(hooks={self.list_hooks()})"


# 模块级默认钩子注册表
_default_hook_registry = HookRegistry()


def register_hook(
    name: str,
    callback: Callable[["ReadOnlyResult"], Any],
    **kwargs: Any,
) -> None:
    """注册钩子到默认注册表（便捷函数）。

    Parameters
    ----------
    name : str
        钩子唯一名称。
    callback : Callable
        回调函数。
    **kwargs
        传递给 HookRegistry.register 的其他参数。
    """
    _default_hook_registry.register(name, callback, **kwargs)


def unregister_hook(name: str) -> None:
    """从默认注册表移除钩子。

    Parameters
    ----------
    name : str
        要移除的钩子名称。
    """
    _default_hook_registry.unregister(name)


def get_hook_registry() -> HookRegistry:
    """获取默认钩子注册表。

    Returns
    -------
    HookRegistry
        模块级默认注册表实例。
    """
    return _default_hook_registry


# --------------------------------------------------------------------------
# 内置示例钩子（不默认注册，用户按需启用）
# --------------------------------------------------------------------------


def dose_summary_hook(result: ReadOnlyResult) -> dict[str, Any]:
    """剂量响应摘要统计钩子。

    计算评估点数量、ATT 范围、显著点数等摘要统计量。

    Parameters
    ----------
    result : ReadOnlyResult
        只读结果视图。

    Returns
    -------
    dict
        包含 n_eval_points、estimate_range、significant_points 的字典。
    """
    import numpy as np

    estimates = np.asarray(result.estimate)
    std_errors = np.asarray(result.std_error)

    # 计算显著点（|z| > 1.96）
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = np.where(std_errors > 0, np.abs(estimates / std_errors), 0.0)

    return {
        "n_eval_points": len(result.grid),
        "estimate_range": (float(np.min(estimates)), float(np.max(estimates))),
        "significant_points": int(np.sum(z_stats > 1.96)),
    }
