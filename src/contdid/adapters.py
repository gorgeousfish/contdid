"""多数据框架适配器 - 支持 Polars 和 Arrow 输入。

确保数据转换过程中保持面板数据的数学属性不变，
满足论文 arXiv-2107.02637v7 假设 A1 (随机抽样/iid面板结构) 的要求。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class DataFrameAdapter(Protocol):
    """数据框架适配器协议。"""

    def can_handle(self, data: Any) -> bool:
        """检查是否能处理此数据类型。"""
        ...

    def to_pandas(self, data: Any) -> pd.DataFrame:
        """转换为pandas DataFrame，保持数学属性。"""
        ...

    def validate_conversion(self, original: Any, converted: pd.DataFrame) -> list[str]:
        """验证转换后数据的完整性。"""
        ...


class PandasAdapter:
    """Pandas DataFrame 适配器（直通）。"""

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, pd.DataFrame)

    def to_pandas(self, data: Any) -> pd.DataFrame:
        return data

    def validate_conversion(self, original: Any, converted: pd.DataFrame) -> list[str]:
        return []


class PolarsAdapter:
    """Polars DataFrame 适配器。

    转换注意事项：
    - Polars 的 null 与 pandas 的 NaN 语义不同
    - Polars 的整数类型不含 NaN（使用 null）
    - 排序可能不同（需显式排序）
    """

    def can_handle(self, data: Any) -> bool:
        try:
            import polars as pl

            return isinstance(data, (pl.DataFrame, pl.LazyFrame))
        except ImportError:
            return False

    def to_pandas(self, data: Any) -> pd.DataFrame:
        """转换 Polars → pandas，保持面板结构。"""
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # 使用 Polars 原生方法转换，保持类型映射
        df = data.to_pandas()

        return df

    def validate_conversion(self, original: Any, converted: pd.DataFrame) -> list[str]:
        """验证转换完整性。"""
        import polars as pl

        issues: list[str] = []

        # 收集原始 DataFrame（如果是 LazyFrame 则 collect）
        orig_df = original if isinstance(original, pl.DataFrame) else original.collect()

        # 1. 行数一致
        if len(converted) != len(orig_df):
            issues.append(f"行数不一致: 原始{len(orig_df)}, 转换后{len(converted)}")

        # 2. 列名一致
        orig_cols = set(orig_df.columns)
        conv_cols = set(converted.columns)
        if orig_cols != conv_cols:
            missing = orig_cols - conv_cols
            extra = conv_cols - orig_cols
            msg = "列名不一致:"
            if missing:
                msg += f" 缺少{missing}"
            if extra:
                msg += f" 多余{extra}"
            issues.append(msg)

        # 3. 数值精度检查（浮点列）
        for col in orig_df.columns:
            if col not in converted.columns:
                continue
            if orig_df[col].dtype in (pl.Float32, pl.Float64):
                orig_vals = orig_df[col].to_numpy()
                conv_vals = converted[col].to_numpy()
                # 排除 NaN 位置
                mask = ~(np.isnan(orig_vals) | np.isnan(conv_vals))
                if mask.any() and not np.allclose(
                    orig_vals[mask], conv_vals[mask], rtol=1e-15, atol=0
                ):
                    issues.append(f"列 '{col}' 浮点精度损失")

        return issues


class ArrowAdapter:
    """PyArrow Table 适配器。

    转换注意事项：
    - Arrow 类型系统更严格
    - 大型数据集使用零拷贝转换
    """

    def can_handle(self, data: Any) -> bool:
        try:
            import pyarrow as pa

            return isinstance(data, pa.Table)
        except ImportError:
            return False

    def to_pandas(self, data: Any) -> pd.DataFrame:
        """转换 Arrow → pandas，保持面板结构。"""
        # self_destruct=False 保证原始 Table 不被释放
        df = data.to_pandas(self_destruct=False)
        return df

    def validate_conversion(self, original: Any, converted: pd.DataFrame) -> list[str]:
        """验证转换完整性。"""
        import pyarrow as pa

        issues: list[str] = []

        # 1. 行数一致
        if len(converted) != original.num_rows:
            issues.append(f"行数不一致: 原始{original.num_rows}, 转换后{len(converted)}")

        # 2. 列名一致
        orig_cols = set(original.column_names)
        conv_cols = set(converted.columns)
        if orig_cols != conv_cols:
            missing = orig_cols - conv_cols
            extra = conv_cols - orig_cols
            msg = "列名不一致:"
            if missing:
                msg += f" 缺少{missing}"
            if extra:
                msg += f" 多余{extra}"
            issues.append(msg)

        # 3. 数值精度检查（浮点列）
        for i, field in enumerate(original.schema):
            if field.name not in converted.columns:
                continue
            if pa.types.is_floating(field.type):
                orig_vals = original.column(field.name).to_numpy()
                conv_vals = converted[field.name].to_numpy()
                mask = ~(np.isnan(orig_vals) | np.isnan(conv_vals))
                if mask.any() and not np.allclose(
                    orig_vals[mask], conv_vals[mask], rtol=1e-15, atol=0
                ):
                    issues.append(f"列 '{field.name}' 浮点精度损失")

        return issues


class AdapterRegistry:
    """适配器注册表 - 管理可用的数据框架适配器。"""

    def __init__(self) -> None:
        self._adapters: list[DataFrameAdapter] = [
            PandasAdapter(),
            PolarsAdapter(),
            ArrowAdapter(),
        ]

    def register(self, adapter: DataFrameAdapter) -> None:
        """注册新的数据框架适配器。

        新注册的适配器具有最高优先级。
        """
        self._adapters.insert(0, adapter)

    def convert(self, data: Any) -> pd.DataFrame:
        """自动检测并转换输入数据为 pandas DataFrame。

        Parameters
        ----------
        data : Any
            输入数据（pandas/polars/arrow）

        Returns
        -------
        pd.DataFrame
            转换后的 pandas DataFrame

        Raises
        ------
        TypeError
            不支持的数据类型
        ValueError
            转换验证失败
        """
        for adapter in self._adapters:
            if adapter.can_handle(data):
                converted = adapter.to_pandas(data)
                issues = adapter.validate_conversion(data, converted)
                if issues:
                    raise ValueError(
                        "数据转换验证失败:\n" + "\n".join(f"  - {i}" for i in issues)
                    )
                return converted

        type_name = type(data).__name__
        raise TypeError(
            f"不支持的数据类型: {type_name}。"
            f"支持的类型: pandas.DataFrame, polars.DataFrame, "
            f"polars.LazyFrame, pyarrow.Table"
        )


# 模块级默认注册表
_default_adapter_registry = AdapterRegistry()


def convert_to_pandas(data: Any) -> pd.DataFrame:
    """便捷函数 - 自动转换输入数据为 pandas DataFrame。

    支持 pandas DataFrame（直通）、Polars DataFrame/LazyFrame、PyArrow Table。
    转换过程中自动验证行数、列名和浮点精度的一致性。

    Parameters
    ----------
    data : Any
        输入数据

    Returns
    -------
    pd.DataFrame
        转换后的 pandas DataFrame

    Raises
    ------
    TypeError
        不支持的数据类型
    ValueError
        转换验证失败
    """
    return _default_adapter_registry.convert(data)


def get_adapter_registry() -> AdapterRegistry:
    """获取默认适配器注册表实例。"""
    return _default_adapter_registry
