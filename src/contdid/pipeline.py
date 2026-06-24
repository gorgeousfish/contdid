"""管道化API - 支持链式调用模式。

提供 :class:`FittablePanel` 中间对象，支持如下链式调用：

    panel.with_spec(spec).fit().summary()

该模块是对现有 ``cont_did()`` 接口的薄封装，不引入新的估计逻辑。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import PanelData
    from .results import ContDIDResult
    from .specs import ContDIDSpec


class FittablePanel:
    """绑定了数据和规范的可拟合对象。

    支持链式调用：``panel.with_spec(spec).fit().summary()``

    Parameters
    ----------
    panel : PanelData
        面板数据容器
    spec : ContDIDSpec
        估计规范

    Examples
    --------
    >>> from contdid import PanelData, ContDIDSpec
    >>> panel = PanelData(df, id_column="id", time_column="t",
    ...                   outcome_column="Y", group_column="G", dose_column="D")
    >>> spec = ContDIDSpec.dose_response()
    >>> result = panel.with_spec(spec).fit()
    >>> print(result.summary())
    """

    __slots__ = ("_panel", "_spec")

    def __init__(self, panel: "PanelData", spec: "ContDIDSpec") -> None:
        self._panel = panel
        self._spec = spec

    @property
    def panel(self) -> "PanelData":
        """获取绑定的面板数据。"""
        return self._panel

    @property
    def spec(self) -> "ContDIDSpec":
        """获取绑定的估计规范。"""
        return self._spec

    def fit(self, **kwargs: Any) -> "ContDIDResult":
        """执行估计，返回结果对象。

        Parameters
        ----------
        **kwargs
            传递给 ``cont_did()`` 的额外参数（如 ``dvals``, ``degree``,
            ``num_knots``, ``base_period``, ``adaptive`` 等）。

        Returns
        -------
        ContDIDResult
            估计结果，包含点估计、标准误、置信区间等。

        Examples
        --------
        >>> result = panel.with_spec(spec).fit(degree=3, num_knots=2)
        """
        from .api import cont_did

        return cont_did(self._panel, self._spec, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FittablePanel(panel=<{self._panel.frame.shape[0]} rows>, "
            f"spec=<{self._spec.target_parameter}/{self._spec.aggregation}/{self._spec.dose_est_method}>)"
        )
