# R Reference Fixtures for Python↔R Numerical Consistency

## 概述

本目录包含用于验证 `contdid-py` 与 R `contdid` 包数值一致性的基准数据。

## 一致性标准

```
max|bias/SE| ≤ 3
```

即Python估计与R估计的偏差不超过3个标准误。

## 目录结构

```
r_reference/
├── README.md              # 本文件
├── generate_fixtures.R    # R基准数据生成脚本
├── dgp_params.json        # 共享DGP参数（两包必须一致）
└── fixtures/              # R包生成的基准结果（JSON）
    ├── parametric_dose_att.json
    ├── parametric_dose_acrt.json
    ├── cck_dose_att.json
    ├── eventstudy_att.json
    └── two_period_parametric.json
```

## 生成R基准数据

### 前提条件

```r
install.packages(c("contdid", "jsonlite", "tidyr"))
# 或者从GitHub安装开发版:
# remotes::install_github("bcallaway11/contdid")
```

### 运行

```bash
cd tests/r_reference/
Rscript generate_fixtures.R
```

### 输出

脚本会在 `fixtures/` 目录下生成JSON文件，每个文件包含：
- `meta`: 生成环境信息（R版本、包版本、日期）
- `estimation_params`: 估计参数
- `scenarios`: 各DGP场景的结果（grid、estimate、std_error、CI）

## DGP参数说明

DGP参数定义在 `dgp_params.json` 中，与Python包的 `simulate_contdid_data()` 和
R包的 `simulate_contdid_data()` 使用完全相同的参数。

关键DGP:
- **SIM-001-null-dose**: 零效应（用于type-I error校准）
- **SIM-002-linear-dose**: ATT(d) = d（线性）
- **SIM-003-quadratic-dose**: ATT(d) = d²（二次）
- **SIM-TP-linear/quadratic**: 两期场景（CCK适用）

## Python测试使用方式

```python
# tests/test_numerical_consistency.py 会自动检测fixtures是否存在
# 如不存在，相关测试会 pytest.skip()
# 内部一致性测试（TestInternalConsistency）不依赖fixtures，始终可运行
```

## 注意事项

1. R和Python使用相同seed但RNG实现不同，因此生成的数据不完全相同
2. 一致性验证比较的是**估计器输出**，不是原始数据
3. 对于相同DGP，两个估计器应给出统计上不可区分的结果（|diff/SE| ≤ 3）
4. 两期场景是最精确的对比（无staggered聚合差异）
