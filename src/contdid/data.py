"""Core panel-data containers for contdid.

Provides the :class:`PanelData` dataclass that wraps a pandas DataFrame
with named column roles required by contdid estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pandas as pd

if TYPE_CHECKING:
    from .pipeline import FittablePanel
    from .specs import ContDIDSpec


@dataclass(slots=True)
class PanelData:
    """Balanced long-panel input for continuous-dose DiD estimation.

    Wraps a pandas DataFrame together with column-role mappings so that
    estimators can access unit identifiers, time periods, outcomes,
    treatment timing (group), and treatment dose by name.

    The ``frame`` argument accepts pandas DataFrame, Polars DataFrame/LazyFrame,
    or PyArrow Table. Non-pandas inputs are automatically converted via the
    adapter registry (see :mod:`contdid.adapters`).

    Attributes:
        frame: The underlying pandas DataFrame in long format.
        id_column: Column name for unit identifiers.
        time_column: Column name for time period values.
        outcome_column: Column name for the outcome variable.
        group_column: Column name for treatment timing group (0 = never-treated).
        dose_column: Column name for treatment dose (0 = untreated).
    """

    frame: pd.DataFrame
    id_column: str = "id"
    time_column: str = "time_period"
    outcome_column: str = "Y"
    group_column: str = "G"
    dose_column: str = "D"

    def __post_init__(self) -> None:
        """自动适配非 pandas 输入数据类型。"""
        if not isinstance(self.frame, pd.DataFrame):
            from .adapters import convert_to_pandas

            object.__setattr__(self, "frame", convert_to_pandas(self.frame))

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        id_column: str = "id",
        time_column: str = "time_period",
        outcome_column: str = "Y",
        group_column: str = "G",
        dose_column: str = "D",
    ) -> "PanelData":
        """Build a PanelData from row dictionaries or mapping records.

        Args:
            records: Iterable of row-level mappings (e.g., list of dicts).
            id_column: Column name for unit identifiers.
            time_column: Column name for time period values.
            outcome_column: Column name for the outcome variable.
            group_column: Column name for treatment timing group.
            dose_column: Column name for treatment dose.

        Returns:
            A new PanelData instance backed by the constructed DataFrame.
        """
        return cls(
            frame=pd.DataFrame.from_records(records),
            id_column=id_column,
            time_column=time_column,
            outcome_column=outcome_column,
            group_column=group_column,
            dose_column=dose_column,
        )

    def with_spec(self, spec: "ContDIDSpec") -> "FittablePanel":
        """将数据与估计规范绑定，返回可拟合对象。

        Parameters
        ----------
        spec : ContDIDSpec
            估计规范（目标参数、聚合方式、估计方法等）

        Returns
        -------
        FittablePanel
            可调用 ``.fit()`` 的中间对象

        Examples
        --------
        >>> result = panel.with_spec(spec).fit()
        >>> result = panel.with_spec(spec).fit(degree=3, num_knots=2)
        """
        from .pipeline import FittablePanel

        return FittablePanel(self, spec)
