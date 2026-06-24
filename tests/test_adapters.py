"""测试多数据框架适配器。

验证 Polars/Arrow 输入经过适配器转换后，
面板数据数学属性保持不变，估计结果一致。
"""

import numpy as np
import pandas as pd
import pytest

from contdid.adapters import (
    AdapterRegistry,
    ArrowAdapter,
    PandasAdapter,
    PolarsAdapter,
    convert_to_pandas,
    get_adapter_registry,
)
from contdid.data import PanelData


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_panel_df() -> pd.DataFrame:
    """生成一个简单的两期平衡面板 DataFrame。"""
    np.random.seed(42)
    n_units = 50
    n_periods = 2
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)

    # 前 20 个 unit 是处理组（G=2），其余是控制组（G=0）
    groups = np.where(ids <= 20, 2, 0)
    # 剂量：处理组 dose ~ Uniform(0.5, 2.0)，控制组 dose=0
    doses = np.where(
        groups > 0,
        np.repeat(np.random.uniform(0.5, 2.0, n_units), n_periods),
        0.0,
    )
    # 结果变量
    Y = np.random.randn(n_units * n_periods) + doses * 1.5

    return pd.DataFrame(
        {
            "id": ids,
            "time_period": times,
            "Y": Y,
            "G": groups,
            "D": doses,
        }
    )


@pytest.fixture
def sample_panel_df_with_floats() -> pd.DataFrame:
    """面板数据包含高精度浮点数。"""
    np.random.seed(123)
    n = 100
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, 26), 4),
            "time_period": np.tile(np.arange(1, 5), 25),
            "Y": np.random.randn(n) * 1e-8 + np.pi,
            "G": np.where(np.repeat(np.arange(1, 26), 4) <= 10, 3, 0),
            "D": np.where(
                np.repeat(np.arange(1, 26), 4) <= 10,
                np.repeat(np.random.uniform(0.1, 5.0, 25), 4),
                0.0,
            ),
        }
    )


# ─── PandasAdapter Tests ────────────────────────────────────────────────────


class TestPandasAdapter:
    """测试 Pandas 适配器（直通）。"""

    def test_can_handle(self, sample_panel_df):
        adapter = PandasAdapter()
        assert adapter.can_handle(sample_panel_df)

    def test_passthrough(self, sample_panel_df):
        adapter = PandasAdapter()
        result = adapter.to_pandas(sample_panel_df)
        assert result is sample_panel_df  # 同一对象

    def test_cannot_handle_dict(self):
        adapter = PandasAdapter()
        assert not adapter.can_handle({"a": [1, 2]})


# ─── PolarsAdapter Tests ────────────────────────────────────────────────────


polars = pytest.importorskip("polars", reason="Polars not installed")


class TestPolarsAdapter:
    """测试 Polars → pandas 转换。"""

    def test_can_handle_dataframe(self, sample_panel_df):
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.from_pandas(sample_panel_df)
        assert adapter.can_handle(pl_df)

    def test_can_handle_lazyframe(self, sample_panel_df):
        import polars as pl

        adapter = PolarsAdapter()
        pl_lf = pl.from_pandas(sample_panel_df).lazy()
        assert adapter.can_handle(pl_lf)

    def test_basic_conversion(self, sample_panel_df):
        """基本 Polars DataFrame 转换。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.from_pandas(sample_panel_df)
        result = adapter.to_pandas(pl_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_panel_df)
        assert set(result.columns) == set(sample_panel_df.columns)

    def test_panel_structure_preserved(self, sample_panel_df):
        """面板数据结构在转换后保持。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.from_pandas(sample_panel_df)
        result = adapter.to_pandas(pl_df)

        # 所有 (id, time_period) 对唯一
        orig_pairs = set(zip(sample_panel_df["id"], sample_panel_df["time_period"]))
        conv_pairs = set(zip(result["id"], result["time_period"]))
        assert orig_pairs == conv_pairs

    def test_numeric_precision(self, sample_panel_df_with_floats):
        """浮点精度在转换后保持。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.from_pandas(sample_panel_df_with_floats)
        result = adapter.to_pandas(pl_df)

        np.testing.assert_array_equal(
            result["Y"].values, sample_panel_df_with_floats["Y"].values
        )

    def test_null_handling(self):
        """Polars null 正确转换为 pandas NaN。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time_period": [1, 2, 1, 2],
                "Y": [1.0, None, 3.0, 4.0],
                "G": [0, 0, 2, 2],
                "D": [0.0, 0.0, 1.5, 1.5],
            }
        )
        result = adapter.to_pandas(pl_df)

        assert pd.isna(result["Y"].iloc[1])
        assert result["Y"].iloc[0] == 1.0

    def test_lazyframe_conversion(self, sample_panel_df):
        """LazyFrame 也能正确转换。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_lf = pl.from_pandas(sample_panel_df).lazy()
        result = adapter.to_pandas(pl_lf)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_panel_df)

    def test_validation_passes(self, sample_panel_df):
        """验证转换完整性检查通过。"""
        import polars as pl

        adapter = PolarsAdapter()
        pl_df = pl.from_pandas(sample_panel_df)
        result = adapter.to_pandas(pl_df)
        issues = adapter.validate_conversion(pl_df, result)
        assert issues == []

    def test_full_pipeline_with_polars(self, sample_panel_df):
        """Polars 输入通过完整 PanelData 构造。"""
        import polars as pl

        pl_df = pl.from_pandas(sample_panel_df)
        panel = PanelData(frame=pl_df)

        # 验证 frame 已转换为 pandas
        assert isinstance(panel.frame, pd.DataFrame)
        assert len(panel.frame) == len(sample_panel_df)


# ─── ArrowAdapter Tests ─────────────────────────────────────────────────────


pyarrow = pytest.importorskip("pyarrow", reason="PyArrow not installed")


class TestArrowAdapter:
    """测试 Arrow → pandas 转换。"""

    def test_can_handle(self, sample_panel_df):
        import pyarrow as pa

        adapter = ArrowAdapter()
        table = pa.Table.from_pandas(sample_panel_df)
        assert adapter.can_handle(table)

    def test_basic_conversion(self, sample_panel_df):
        """基本 Arrow Table 转换。"""
        import pyarrow as pa

        adapter = ArrowAdapter()
        table = pa.Table.from_pandas(sample_panel_df)
        result = adapter.to_pandas(table)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_panel_df)
        # Arrow 可能添加 __index_level_0__ 列，过滤掉
        orig_cols = set(sample_panel_df.columns)
        result_cols = set(result.columns) - {"__index_level_0__"}
        assert orig_cols.issubset(result_cols)

    def test_panel_structure_preserved(self, sample_panel_df):
        """面板数据结构在转换后保持。"""
        import pyarrow as pa

        adapter = ArrowAdapter()
        table = pa.Table.from_pandas(sample_panel_df)
        result = adapter.to_pandas(table)

        orig_pairs = set(zip(sample_panel_df["id"], sample_panel_df["time_period"]))
        conv_pairs = set(zip(result["id"], result["time_period"]))
        assert orig_pairs == conv_pairs

    def test_numeric_precision(self, sample_panel_df_with_floats):
        """浮点精度在转换后保持。"""
        import pyarrow as pa

        adapter = ArrowAdapter()
        table = pa.Table.from_pandas(sample_panel_df_with_floats)
        result = adapter.to_pandas(table)

        np.testing.assert_array_equal(
            result["Y"].values, sample_panel_df_with_floats["Y"].values
        )

    def test_validation_passes(self, sample_panel_df):
        """验证转换完整性检查通过。"""
        import pyarrow as pa

        adapter = ArrowAdapter()
        table = pa.Table.from_pandas(sample_panel_df)
        result = adapter.to_pandas(table)
        issues = adapter.validate_conversion(table, result)
        assert issues == []

    def test_full_pipeline_with_arrow(self, sample_panel_df):
        """Arrow 输入通过完整 PanelData 构造。"""
        import pyarrow as pa

        table = pa.Table.from_pandas(sample_panel_df)
        panel = PanelData(frame=table)

        assert isinstance(panel.frame, pd.DataFrame)
        assert len(panel.frame) == len(sample_panel_df)


# ─── AdapterRegistry Tests ──────────────────────────────────────────────────


class TestAdapterRegistry:
    """测试适配器注册和路由。"""

    def test_auto_detection_pandas(self, sample_panel_df):
        """自动检测 pandas 输入。"""
        result = convert_to_pandas(sample_panel_df)
        assert result is sample_panel_df

    def test_auto_detection_polars(self, sample_panel_df):
        """自动检测 Polars 输入。"""
        import polars as pl

        pl_df = pl.from_pandas(sample_panel_df)
        result = convert_to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_panel_df)

    def test_auto_detection_arrow(self, sample_panel_df):
        """自动检测 Arrow 输入。"""
        import pyarrow as pa

        table = pa.Table.from_pandas(sample_panel_df)
        result = convert_to_pandas(table)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_panel_df)

    def test_unsupported_type_error(self):
        """不支持的类型抛出 TypeError。"""
        with pytest.raises(TypeError, match="不支持的数据类型"):
            convert_to_pandas({"a": [1, 2, 3]})

    def test_unsupported_type_list(self):
        """列表类型抛出 TypeError。"""
        with pytest.raises(TypeError):
            convert_to_pandas([[1, 2], [3, 4]])

    def test_custom_adapter_registration(self, sample_panel_df):
        """自定义适配器注册。"""
        registry = AdapterRegistry()

        class DictAdapter:
            def can_handle(self, data):
                return isinstance(data, dict)

            def to_pandas(self, data):
                return pd.DataFrame(data)

            def validate_conversion(self, original, converted):
                return []

        registry.register(DictAdapter())
        result = registry.convert({"id": [1, 2], "val": [3, 4]})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_adapter_registry(self):
        """获取默认注册表。"""
        reg = get_adapter_registry()
        assert isinstance(reg, AdapterRegistry)


# ─── Mathematical Consistency Tests ─────────────────────────────────────────


class TestMathematicalConsistency:
    """测试数据转换的数学一致性。

    关键验证：用完全相同的数据（仅格式不同），
    三种适配器输入后产生完全相同的 PanelData frame。
    """

    def test_pandas_polars_identical_frame(self, sample_panel_df):
        """pandas 和 polars 输入产生完全相同的 frame 内容。"""
        import polars as pl

        panel_pd = PanelData(frame=sample_panel_df.copy())
        panel_pl = PanelData(frame=pl.from_pandas(sample_panel_df))

        # 排序后比较（Polars 可能不保证原始行顺序）
        df_pd = panel_pd.frame.sort_values(["id", "time_period"]).reset_index(drop=True)
        df_pl = panel_pl.frame.sort_values(["id", "time_period"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(df_pd, df_pl, check_dtype=False)

    def test_pandas_arrow_identical_frame(self, sample_panel_df):
        """pandas 和 arrow 输入产生完全相同的 frame 内容。"""
        import pyarrow as pa

        panel_pd = PanelData(frame=sample_panel_df.copy())
        panel_pa = PanelData(frame=pa.Table.from_pandas(sample_panel_df))

        df_pd = panel_pd.frame.sort_values(["id", "time_period"]).reset_index(drop=True)
        df_pa = panel_pa.frame.sort_values(["id", "time_period"]).reset_index(drop=True)

        # Arrow 可能带 __index_level_0__，只比较原始列
        common_cols = list(sample_panel_df.columns)
        pd.testing.assert_frame_equal(
            df_pd[common_cols], df_pa[common_cols], check_dtype=False
        )

    def test_all_three_formats_identical(self, sample_panel_df):
        """三种格式输入产生完全相同的数据。"""
        import polars as pl
        import pyarrow as pa

        panel_pd = PanelData(frame=sample_panel_df.copy())
        panel_pl = PanelData(frame=pl.from_pandas(sample_panel_df))
        panel_pa = PanelData(frame=pa.Table.from_pandas(sample_panel_df))

        cols = list(sample_panel_df.columns)

        df_pd = panel_pd.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]
        df_pl = panel_pl.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]
        df_pa = panel_pa.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]

        pd.testing.assert_frame_equal(df_pd, df_pl, check_dtype=False)
        pd.testing.assert_frame_equal(df_pd, df_pa, check_dtype=False)

    def test_float_precision_maintained(self, sample_panel_df_with_floats):
        """高精度浮点数在所有路径中保持 bit-exact。"""
        import polars as pl
        import pyarrow as pa

        orig_Y = sample_panel_df_with_floats["Y"].values.copy()

        panel_pl = PanelData(frame=pl.from_pandas(sample_panel_df_with_floats))
        panel_pa = PanelData(frame=pa.Table.from_pandas(sample_panel_df_with_floats))

        # 排序一致后比较
        df_pl = panel_pl.frame.sort_values(["id", "time_period"]).reset_index(drop=True)
        df_pa = panel_pa.frame.sort_values(["id", "time_period"]).reset_index(drop=True)
        df_orig = sample_panel_df_with_floats.sort_values(
            ["id", "time_period"]
        ).reset_index(drop=True)

        np.testing.assert_array_equal(df_pl["Y"].values, df_orig["Y"].values)
        np.testing.assert_array_equal(df_pa["Y"].values, df_orig["Y"].values)

    def test_sorting_invariance(self, sample_panel_df):
        """结果不依赖输入行顺序。"""
        import polars as pl

        # 随机打乱行顺序
        shuffled = sample_panel_df.sample(frac=1, random_state=99).reset_index(drop=True)
        pl_shuffled = pl.from_pandas(shuffled)

        panel_orig = PanelData(frame=sample_panel_df.copy())
        panel_shuffled = PanelData(frame=pl_shuffled)

        cols = list(sample_panel_df.columns)
        df1 = panel_orig.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[cols]
        df2 = panel_shuffled.frame.sort_values(["id", "time_period"]).reset_index(drop=True)[
            cols
        ]

        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


# ─── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """测试边缘情况。"""

    def test_empty_dataframe_polars(self):
        """空 Polars DataFrame 转换。"""
        import polars as pl

        pl_df = pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "time_period": pl.Series([], dtype=pl.Int64),
                "Y": pl.Series([], dtype=pl.Float64),
                "G": pl.Series([], dtype=pl.Int64),
                "D": pl.Series([], dtype=pl.Float64),
            }
        )
        result = convert_to_pandas(pl_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(result.columns) == {"id", "time_period", "Y", "G", "D"}

    def test_empty_table_arrow(self):
        """空 Arrow Table 转换。"""
        import pyarrow as pa

        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("time_period", pa.int64()),
                ("Y", pa.float64()),
                ("G", pa.int64()),
                ("D", pa.float64()),
            ]
        )
        table = pa.table(
            {
                "id": pa.array([], type=pa.int64()),
                "time_period": pa.array([], type=pa.int64()),
                "Y": pa.array([], type=pa.float64()),
                "G": pa.array([], type=pa.int64()),
                "D": pa.array([], type=pa.float64()),
            },
            schema=schema,
        )
        result = convert_to_pandas(table)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_large_dataframe_polars(self):
        """较大数据集的 Polars 转换性能可接受。"""
        import polars as pl

        np.random.seed(7)
        n_units = 1000
        n_periods = 10
        n_rows = n_units * n_periods

        pl_df = pl.DataFrame(
            {
                "id": np.repeat(np.arange(n_units), n_periods),
                "time_period": np.tile(np.arange(n_periods), n_units),
                "Y": np.random.randn(n_rows),
                "G": np.where(np.repeat(np.arange(n_units), n_periods) < 400, 3, 0),
                "D": np.random.uniform(0, 3, n_rows),
            }
        )
        result = convert_to_pandas(pl_df)
        assert len(result) == n_rows

    def test_integer_columns_polars(self):
        """Polars 整数列正确转换。"""
        import polars as pl

        pl_df = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time_period": [1, 2, 1, 2],
                "Y": [10.0, 20.0, 30.0, 40.0],
                "G": [0, 0, 2, 2],
                "D": [0, 0, 5, 5],
            }
        )
        result = convert_to_pandas(pl_df)
        # 整数列保持为整数（或可转换为整数）
        assert result["id"].dtype in [np.int64, np.int32, int]
        assert result["G"].dtype in [np.int64, np.int32, int]
