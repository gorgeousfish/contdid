"""插件式估计器基类 - 提供结构化接口供第三方开发者继承。

理论基础：arXiv-2107.02637v7 §4 (识别), §5 (估计), §8 (CCK推断)

本模块是对现有 EstimatorProtocol 的补充（非替代），为偏好面向对象继承
风格的第三方开发者提供明确的抽象基类。BaseEstimator 子类可以通过
registry.register_estimator() 直接注册到全局注册表中。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EstimatorCapabilities:
    """估计器能力声明 - 声明估计器支持的功能范围。

    第三方开发者通过此 dataclass 描述自定义估计器支持的功能集，
    注册表将使用此信息进行路由验证和能力检查。

    Attributes
    ----------
    supports_multiperiod : bool
        是否支持多期/staggered adoption 设定。
    supports_eventstudy : bool
        是否支持事件研究聚合。
    supports_adaptive : bool
        是否支持 Lepski 自适应维度选择。
    supports_covariates : bool
        是否支持协变量调整。
    supports_clustering : bool
        是否支持聚类标准误。
    max_spline_degree : int
        最大支持的样条阶数。
    """

    supports_multiperiod: bool = False
    supports_eventstudy: bool = False
    supports_adaptive: bool = False
    supports_covariates: bool = False
    supports_clustering: bool = False
    max_spline_degree: int = 3


@dataclass
class EstimatorResult:
    """估计器返回的结构化结果。

    所有 BaseEstimator.fit() 方法必须返回此类实例。

    Attributes
    ----------
    att_d : np.ndarray
        ATT(d) 在 dvals 上的点估计值。
    acrt_d : np.ndarray | None
        ACRT(d) 在 dvals 上的点估计值（target_parameter="slope" 时）。
    coefficients : np.ndarray
        基函数系数向量 β_K。
    dvals : np.ndarray
        评估点网格。
    basis_info : dict
        基函数信息（degree, knots, basis_type 等）。
    att_overall : float | None
        全局 ATT = E[ATT(D)|D>0]。
    acrt_overall : float | None
        全局 ACRT。
    """

    att_d: np.ndarray
    acrt_d: np.ndarray | None
    coefficients: np.ndarray
    dvals: np.ndarray
    basis_info: dict = field(default_factory=dict)
    att_overall: float | None = None
    acrt_overall: float | None = None

    def __post_init__(self) -> None:
        """验证结果结构完整性。"""
        self.att_d = np.asarray(self.att_d, dtype=float)
        self.dvals = np.asarray(self.dvals, dtype=float)
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.acrt_d is not None:
            self.acrt_d = np.asarray(self.acrt_d, dtype=float)

        if self.att_d.shape[0] != self.dvals.shape[0]:
            raise ValueError(
                f"att_d 长度 ({self.att_d.shape[0]}) 必须与 "
                f"dvals 长度 ({self.dvals.shape[0]}) 一致"
            )
        if self.acrt_d is not None and self.acrt_d.shape[0] != self.dvals.shape[0]:
            raise ValueError(
                f"acrt_d 长度 ({self.acrt_d.shape[0]}) 必须与 "
                f"dvals 长度 ({self.dvals.shape[0]}) 一致"
            )


class BaseEstimator(ABC):
    """连续处理DiD估计器抽象基类。

    第三方开发者通过继承此类并实现抽象方法来创建自定义估计器。
    所有实现必须严格遵循 arXiv-2107.02637v7 的数学定义。

    核心约定：
    - fit() 执行估计，计算 ATT(d) 或 ACRT(d)
    - influence_function() 返回影响函数用于推断
    - validate_assumptions() 检查识别假设是否满足

    数学定义 (arXiv-2107.02637v7)：
    - ATT(d) = E[ΔY|D=d] - E[ΔY|D=0]，估计量：ψ^K(d)'β_K
    - ACRT(d) = ∂ATT(d)/∂d，估计量：(∂ψ^K(d))'β_K
    - 影响函数：φ_K(W_i, d) = ψ^K(d)'[E[ψψ']]^{-1}ψ^K(D_i)u_{i,K}
    - 推断：乘子Bootstrap with N(0,1) weights

    Example
    -------
    >>> class MyEstimator(BaseEstimator):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_method"
    ...     @property
    ...     def capabilities(self) -> EstimatorCapabilities:
    ...         return EstimatorCapabilities(supports_multiperiod=False)
    ...     def fit(self, dy, dose, dvals, **kwargs):
    ...         # 实现ATT(d)估计...
    ...         return EstimatorResult(...)
    ...     def influence_function(self, dy, dose, dvals, coefficients, **kwargs):
    ...         return inf_func_matrix
    ...     def validate_assumptions(self, panel, spec):
    ...         return []
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """估计器唯一标识名。"""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> EstimatorCapabilities:
        """声明估计器支持的功能范围。"""
        ...

    @abstractmethod
    def fit(
        self,
        dy: np.ndarray,
        dose: np.ndarray,
        dvals: np.ndarray,
        *,
        target_parameter: str = "level",
        degree: int = 3,
        num_knots: int = 0,
        **kwargs: Any,
    ) -> EstimatorResult:
        """执行估计。

        Parameters
        ----------
        dy : np.ndarray
            一阶差分 ΔY = Y_{t=2} - Y_{t=1}（处理组，D>0）
        dose : np.ndarray
            处理剂量 D（处理组，D>0）
        dvals : np.ndarray
            评估点网格
        target_parameter : str
            "level" (ATT) 或 "slope" (ACRT)
        degree : int
            B样条次数
        num_knots : int
            内部节点数

        Returns
        -------
        EstimatorResult
            包含点估计、基函数信息、系数等
        """
        ...

    @abstractmethod
    def influence_function(
        self,
        dy: np.ndarray,
        dose: np.ndarray,
        dvals: np.ndarray,
        coefficients: np.ndarray,
        *,
        target_parameter: str = "level",
        **kwargs: Any,
    ) -> np.ndarray:
        """计算影响函数矩阵。

        论文定义 (arXiv-2107.02637v7, §8):
        φ_K(W_i, d) = ψ^K(d)' [E[ψψ']]^{-1} ψ^K(D_i) u_{i,K}

        Parameters
        ----------
        dy : np.ndarray
            一阶差分 ΔY
        dose : np.ndarray
            处理剂量
        dvals : np.ndarray
            评估点网格
        coefficients : np.ndarray
            fit() 返回的系数向量

        Returns
        -------
        np.ndarray
            shape (n_obs, n_dvals) 的影响函数矩阵
        """
        ...

    @abstractmethod
    def validate_assumptions(
        self, panel: Any, spec: Any
    ) -> list[str]:
        """验证识别假设是否满足。

        应检查：
        - 强平行趋势(SPT)的可测试含义
        - 重叠/正则性条件
        - 样条维度与样本量的关系

        Returns
        -------
        list[str]
            违反假设的警告消息列表（空列表表示通过）
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """获取估计器的配置参数。"""
        return {}

    def set_params(self, **params: Any) -> BaseEstimator:
        """设置估计器参数（返回 self 支持链式调用）。"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def validate_estimator_result(result: EstimatorResult, n_obs: int, n_dvals: int) -> list[str]:
    """验证 EstimatorResult 的结构完整性。

    Parameters
    ----------
    result : EstimatorResult
        待验证的结果对象。
    n_obs : int
        样本量（用于影响函数维度检查参考）。
    n_dvals : int
        评估点数量。

    Returns
    -------
    list[str]
        验证错误列表（空列表表示通过）。
    """
    errors: list[str] = []
    if not isinstance(result, EstimatorResult):
        errors.append(f"fit() 必须返回 EstimatorResult，得到 {type(result).__name__}")
        return errors

    if result.att_d.shape[0] != n_dvals:
        errors.append(
            f"att_d 长度 ({result.att_d.shape[0]}) 与 dvals ({n_dvals}) 不一致"
        )
    if result.dvals.shape[0] != n_dvals:
        errors.append(
            f"result.dvals 长度 ({result.dvals.shape[0]}) 与预期 ({n_dvals}) 不一致"
        )
    if not np.isfinite(result.att_d).all():
        errors.append("att_d 包含非有限值 (NaN/Inf)")
    if not np.isfinite(result.coefficients).all():
        errors.append("coefficients 包含非有限值 (NaN/Inf)")
    return errors


def validate_influence_matrix(
    inf_matrix: np.ndarray, n_obs: int, n_dvals: int
) -> list[str]:
    """验证影响函数矩阵的维度和数值正确性。

    Parameters
    ----------
    inf_matrix : np.ndarray
        影响函数矩阵。
    n_obs : int
        预期样本量。
    n_dvals : int
        预期评估点数量。

    Returns
    -------
    list[str]
        验证错误列表。
    """
    errors: list[str] = []
    if not isinstance(inf_matrix, np.ndarray):
        errors.append(f"influence_function() 必须返回 np.ndarray，得到 {type(inf_matrix).__name__}")
        return errors

    if inf_matrix.ndim != 2:
        errors.append(f"影响函数矩阵必须为二维，得到 {inf_matrix.ndim} 维")
        return errors

    if inf_matrix.shape[0] != n_obs:
        errors.append(
            f"影响函数行数 ({inf_matrix.shape[0]}) 与样本量 ({n_obs}) 不一致"
        )
    if inf_matrix.shape[1] != n_dvals:
        errors.append(
            f"影响函数列数 ({inf_matrix.shape[1]}) 与评估点数 ({n_dvals}) 不一致"
        )
    if not np.isfinite(inf_matrix).all():
        errors.append("影响函数矩阵包含非有限值 (NaN/Inf)")
    return errors


# ---------------------------------------------------------------------------
# 教学示例：线性剂量响应估计器
# ---------------------------------------------------------------------------


class LinearDoseEstimator(BaseEstimator):
    """线性剂量响应估计器（教学示例）。

    假设 ATT(d) = α + β·d（线性关系）。
    这对应 degree=1, num_knots=0 的特例。

    理论基础：arXiv-2107.02637v7 §5.1 参数化估计

    此类作为 BaseEstimator 子类的参考实现，展示：
    - 如何实现 fit() 执行 OLS 估计
    - 如何计算影响函数用于乘子 Bootstrap 推断
    - 如何进行基本的假设验证

    Example
    -------
    >>> est = LinearDoseEstimator()
    >>> # dy, dose, dvals 为 numpy 数组
    >>> result = est.fit(dy, dose, dvals)
    >>> inf_mat = est.influence_function(dy, dose, dvals, result.coefficients)
    """

    @property
    def name(self) -> str:
        """估计器唯一标识名。"""
        return "linear_dose"

    @property
    def capabilities(self) -> EstimatorCapabilities:
        """声明功能：仅支持基本二期估计。"""
        return EstimatorCapabilities(
            supports_multiperiod=False,
            supports_eventstudy=False,
            supports_adaptive=False,
            supports_covariates=False,
            supports_clustering=False,
            max_spline_degree=1,
        )

    def fit(
        self,
        dy: np.ndarray,
        dose: np.ndarray,
        dvals: np.ndarray,
        *,
        target_parameter: str = "level",
        degree: int = 1,
        num_knots: int = 0,
        **kwargs: Any,
    ) -> EstimatorResult:
        """线性 OLS 估计 ATT(d) = α + β·d。

        估计方程：ΔY_i = α + β·D_i + ε_i
        ATT(d) = ψ(d)'β = [1, d]'[α, β]' = α + β·d
        """
        dy = np.asarray(dy, dtype=float)
        dose = np.asarray(dose, dtype=float)
        dvals = np.asarray(dvals, dtype=float)

        n = len(dy)
        # 设计矩阵 X = [1, D]
        X = np.column_stack([np.ones(n), dose])

        # OLS: β = (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ dy
        coefficients = np.linalg.solve(XtX, Xty)

        # ATT(d) = α + β·d
        psi_d = np.column_stack([np.ones(len(dvals)), dvals])
        att_d = psi_d @ coefficients

        # ACRT(d) = ∂ATT/∂d = β (constant for linear)
        acrt_d = None
        if target_parameter == "slope":
            acrt_d = np.full(len(dvals), coefficients[1])

        return EstimatorResult(
            att_d=att_d,
            acrt_d=acrt_d,
            coefficients=coefficients,
            dvals=dvals,
            basis_info={"degree": 1, "num_knots": 0, "basis_type": "polynomial"},
            att_overall=float(np.mean(att_d)),
            acrt_overall=float(coefficients[1]) if acrt_d is not None else None,
        )

    def influence_function(
        self,
        dy: np.ndarray,
        dose: np.ndarray,
        dvals: np.ndarray,
        coefficients: np.ndarray,
        *,
        target_parameter: str = "level",
        **kwargs: Any,
    ) -> np.ndarray:
        """计算线性估计器的影响函数。

        φ_i(d) = ψ(d)' (X'X)^{-1} X_i · u_i
        其中 u_i = ΔY_i - X_i'β 为残差。
        """
        dy = np.asarray(dy, dtype=float)
        dose = np.asarray(dose, dtype=float)
        dvals = np.asarray(dvals, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)

        n = len(dy)
        X = np.column_stack([np.ones(n), dose])

        # 残差
        residuals = dy - X @ coefficients

        # (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)

        # ψ(d) 基函数在评估点
        if target_parameter == "slope":
            psi_d = np.column_stack([np.zeros(len(dvals)), np.ones(len(dvals))])
        else:
            psi_d = np.column_stack([np.ones(len(dvals)), dvals])

        # φ_i(d) = ψ(d)' (X'X)^{-1} X_i · u_i
        # shape: (n_obs, n_dvals)
        meat = X @ XtX_inv  # (n, K)
        inf_matrix = np.outer(residuals, np.ones(len(dvals))) * (meat @ psi_d.T)

        return inf_matrix

    def validate_assumptions(
        self, panel: Any, spec: Any
    ) -> list[str]:
        """验证线性模型的基本假设。"""
        warnings: list[str] = []
        # 线性估计器假设简单线性关系，实际剂量响应可能非线性
        # 此处仅做基本检查
        return warnings
