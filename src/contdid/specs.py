"""Runtime spec objects and Phase 2 enum fidelity helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from ._asset_paths import resolve_runtime_asset

_SYMBOL_MAP_CONTRACT_PATH = resolve_runtime_asset(
    package_relative="runtime_assets/symbol_map_contract.json",
    repo_relative="contdid-py/runtime-assets/symbol_map_contract.json",
)


@lru_cache(maxsize=1)
def load_symbol_map_contract() -> dict:
    """Load the Phase 2 symbol-map contract used for runtime enum fidelity."""

    return json.loads(_SYMBOL_MAP_CONTRACT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_runtime_enum_map() -> dict[str, tuple[str, ...]]:
    """Return Phase 2 runtime enums keyed by target_parameter/aggregation/etc."""

    fidelity_rules = load_symbol_map_contract()["fidelity_rules"]
    return {
        rule["name"]: tuple(rule["allowed_values"])
        for rule in fidelity_rules
        if rule["name"] in {"target_parameter", "aggregation", "dose_est_method", "control_group"}
    }


@dataclass(slots=True)
class ContDIDSpec:
    """Specification for continuous-dose DiD estimation.

    Holds all configuration parameters that control estimator routing,
    basis construction, inference method, and treatment assumptions.

    Attributes:
        target_parameter: "level" for ATT(d) or "slope" for ACRT(d).
        aggregation: "dose" for dose-response or "eventstudy" for event-study.
        dose_est_method: "parametric" for B-spline OLS or "cck" for CCK sieve.
        control_group: Which units serve as the comparison group.

            - "notyettreated" (default): Uses never-treated (G=0) AND
              not-yet-treated (G>t) units as controls. Recommended for
              staggered adoption designs; maximizes control-group sample size
              and statistical power. Consistent with R package `contdid`
              default.
            - "nevertreated": Uses only never-treated (G=0) units as controls.
              Suitable when a sufficient pool of never-treated units exists
              and the researcher prefers a fixed comparison group.

            Both strategies are supported by the multi-period identification
            theory in arXiv-2107.02637v7, Section A3 (Assumption 3-MP).
        treatment_type: Treatment type (default "continuous").
        anticipation: Number of anticipation periods (default 0).
        alp: Significance level (default 0.05).
        bstrap: Whether to use bootstrap inference (default True).
        cband: Whether to compute simultaneous confidence band (default False).
        boot_type: Bootstrap weight distribution (default "multiplier").
        biters: Number of bootstrap iterations (default 1000).
        covariates: Optional covariate column names (RESERVED, NOT AVAILABLE).
            The paper (arXiv:2107.02637v7, §Extensions) provides only a conceptual
            framework for conditional parallel trends without the full estimation
            theory. Passing a non-None value raises NotImplementedError.
        cluster_column: Optional column name for cluster-robust SEs.
    """

    target_parameter: str
    aggregation: str
    dose_est_method: str
    control_group: str
    # NOTE: 论文(arXiv-2107.02637v7) Assumption 4 理论上同时覆盖连续处理和多值离散处理，
    # 但当前实现仅支持 "continuous"。多值离散处理的饱和回归估计器尚未实现。
    treatment_type: str = "continuous"
    # Anticipation parameter (CGBS Assumption 3-MP(a)): when anticipation=a > 0,
    # the base period shifts from g-1 to g-1-a, allowing for up to 'a' periods
    # of anticipatory behavior before actual treatment onset.
    anticipation: int = 0
    alp: float = 0.05
    bstrap: bool = True
    cband: bool = False
    boot_type: str = "multiplier"
    biters: int = 1000
    covariates: tuple[str, ...] | None = None  # 预留参数：协变量列名（当前不可用，传入非None触发NotImplementedError）
    cluster_column: str | None = None  # Column name for cluster-robust SEs
    validation_strictness: str = "strict"  # "strict", "normal", or "lenient"

    @classmethod
    def dose_response(cls, method: str = "parametric", **kwargs) -> "ContDIDSpec":
        """快捷构造剂量响应规范。

        Parameters
        ----------
        method : str
            估计方法，"parametric" 或 "cck"。
        **kwargs
            其他 ContDIDSpec 参数。

        Returns
        -------
        ContDIDSpec

        Examples
        --------
        >>> spec = ContDIDSpec.dose_response()
        >>> spec = ContDIDSpec.dose_response(method="cck", cband=True)
        """
        kwargs.setdefault("control_group", "notyettreated")
        return cls(
            target_parameter="level",
            aggregation="dose",
            dose_est_method=method,
            **kwargs,
        )

    @classmethod
    def eventstudy(cls, method: str = "parametric", **kwargs) -> "ContDIDSpec":
        """快捷构造事件研究规范。

        Parameters
        ----------
        method : str
            估计方法，"parametric" 或 "cck"。
        **kwargs
            其他 ContDIDSpec 参数。

        Returns
        -------
        ContDIDSpec

        Examples
        --------
        >>> spec = ContDIDSpec.eventstudy()
        >>> spec = ContDIDSpec.eventstudy(anticipation=1)
        """
        kwargs.setdefault("control_group", "notyettreated")
        return cls(
            target_parameter="level",
            aggregation="eventstudy",
            dose_est_method=method,
            **kwargs,
        )

    @classmethod
    def marginal_response(cls, method: str = "parametric", **kwargs) -> "ContDIDSpec":
        """快捷构造边际响应（ACRT）规范。

        Parameters
        ----------
        method : str
            估计方法，"parametric" 或 "cck"。
        **kwargs
            其他 ContDIDSpec 参数。

        Returns
        -------
        ContDIDSpec

        Examples
        --------
        >>> spec = ContDIDSpec.marginal_response()
        >>> spec = ContDIDSpec.marginal_response(method="cck")
        """
        kwargs.setdefault("control_group", "notyettreated")
        return cls(
            target_parameter="slope",
            aggregation="dose",
            dose_est_method=method,
            **kwargs,
        )
