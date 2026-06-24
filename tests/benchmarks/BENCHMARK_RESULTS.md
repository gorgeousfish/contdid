# contdid-py Performance Benchmark Results

## System Information

| Key | Value |
|-----|-------|
| platform | macOS-15.3-arm64-arm-64bit-Mach-O |
| python_version | 3.13.2 |
| numpy_version | 2.2.4 |
| pandas_version | 2.2.3 |
| scipy_version | 1.17.1 |
| cpu | arm |
| machine | arm64 |

## Benchmark Configuration

- Sample sizes: [1000, 5000, 10000, 20000]
- Bootstrap iterations (n<=1000): 1000
- Bootstrap iterations (n>1000): 100
- B-spline num_knots: 5

## Results Summary

| Scenario | n_units | n_rows | Wall Time | Peak Memory (MB) | biters |
|----------|---------|--------|-----------|------------------|--------|
| two_period_dose | 1000 | 2000 | 55.9 ms | 1.6 | 1000 |
| two_period_dose | 5000 | 10000 | 117.7 ms | 8.0 | 100 |
| two_period_dose | 10000 | 20000 | 214.7 ms | 15.9 | 100 |
| two_period_dose | 20000 | 40000 | 376.3 ms | 31.7 | 100 |
| multiperiod_dose | 1000 | 4000 | 143.6 ms | 1.0 | 1000 |
| multiperiod_dose | 5000 | 20000 | 327.1 ms | 4.5 | 100 |
| multiperiod_dose | 10000 | 40000 | 665.0 ms | 8.9 | 100 |
| multiperiod_dose | 20000 | 80000 | 1.08 s | 17.7 | 100 |
| eventstudy | 1000 | 4000 | 374.5 ms | 0.8 | 1000 |
| eventstudy | 5000 | 20000 | 859.0 ms | 3.6 | 100 |
| eventstudy | 10000 | 40000 | 1.16 s | 7.1 | 100 |
| eventstudy | 20000 | 80000 | 2.17 s | 14.1 | 100 |

## Scaling Analysis

### two_period_dose

- n=5000 vs n=1000: n_ratio=5.0x, time_ratio=2.1x
- n=10000 vs n=1000: n_ratio=10.0x, time_ratio=3.8x
- n=20000 vs n=1000: n_ratio=20.0x, time_ratio=6.7x

### multiperiod_dose

- n=5000 vs n=1000: n_ratio=5.0x, time_ratio=2.3x
- n=10000 vs n=1000: n_ratio=10.0x, time_ratio=4.6x
- n=20000 vs n=1000: n_ratio=20.0x, time_ratio=7.5x

### eventstudy

- n=5000 vs n=1000: n_ratio=5.0x, time_ratio=2.3x
- n=10000 vs n=1000: n_ratio=10.0x, time_ratio=3.1x
- n=20000 vs n=1000: n_ratio=20.0x, time_ratio=5.8x

## cProfile Hotspot Analysis

### two_period_dose

```
36721 function calls (36085 primitive calls) in 0.055 seconds

   Ordered by: cumulative time
   List reduced from 907 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.055    0.055 /Users/cxy/Desktop/2026project/contdid/contdid-py/tests/benchmarks/bench_performance.py:263(run)
        1    0.000    0.000    0.055    0.055 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:33(cont_did)
        1    0.000    0.000    0.044    0.044 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:311(_route_two_period_dose)
        1    0.000    0.000    0.044    0.044 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:1091(estimate_dose_level_effects)
        1    0.000    0.000    0.044    0.044 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:1054(estimate_dose_effects)
        2    0.000    0.000    0.024    0.012 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/validation.py:380(validate_panel_data)
        1    0.000    0.000    0.023    0.023 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:523(_fit_shared_dose_design)
        1    0.000    0.000    0.014    0.014 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:405(_prepare_dose_sample)
        2    0.003    0.002    0.010    0.005 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/validation.py:110(_check_balanced_panel)
        1    0.000    0.000    0.008    0.008 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:96(_collapse_to_unit_differences)
        6    0.002    0.000    0.008    0.001 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/groupby/generic.py:635(nunique)
        4    0.000    0.000    0.007    0.002 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:312(_build_design_matrix)
        4    0.000    0.000    0.007    0.002 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/bspline.py:63(build_bspline_design)
        1    0.000    0.000    0.006    0.006 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:897(_build_parametric_result_from_fit)
        2    0.000    0.000    0.006    0.003 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/validation.py:177(_check_unit_constancy)
```

### multiperiod_dose

```
61774 function calls (60527 primitive calls) in 0.107 seconds

   Ordered by: cumulative time
   List reduced from 624 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.107    0.107 /Users/cxy/Desktop/2026project/contdid/contdid-py/tests/benchmarks/bench_performance.py:268(run)
        1    0.000    0.000    0.107    0.107 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:33(cont_did)
        1    0.000    0.000    0.092    0.092 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:252(_route_multiperiod_dose)
        1    0.000    0.000    0.091    0.091 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/multiperiod.py:300(estimate_multiperiod_dose)
        6    0.001    0.000    0.053    0.009 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/multiperiod.py:156(_run_local_dose_estimation)
        1    0.029    0.029    0.030    0.030 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/influence.py:416(aggregate_influence_functions)
   104/64    0.000    0.000    0.020    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexing.py:1176(__getitem__)
      104    0.000    0.000    0.018    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexing.py:1397(_getitem_axis)
        1    0.000    0.000    0.015    0.015 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/validation.py:380(validate_panel_data)
       40    0.000    0.000    0.013    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexing.py:1365(_getitem_tuple)
       40    0.000    0.000    0.013    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexing.py:1032(_getitem_lowerdim)
       12    0.001    0.000    0.012    0.001 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/bspline.py:63(build_bspline_design)
       49    0.000    0.000    0.009    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/generic.py:4142(_take_with_is_copy)
       49    0.000    0.000    0.008    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/generic.py:4027(take)
       34    0.000    0.000    0.008    0.000 /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/pandas/core/indexing.py:1205(_getbool_axis)
```

### eventstudy

```
543150 function calls (539635 primitive calls) in 0.491 seconds

   Ordered by: cumulative time
   List reduced from 1017 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.491    0.491 /Users/cxy/Desktop/2026project/contdid/contdid-py/tests/benchmarks/bench_performance.py:273(run)
        1    0.000    0.000    0.491    0.491 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:33(cont_did)
        1    0.000    0.000    0.476    0.476 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/api.py:213(_route_eventstudy)
        1    0.002    0.002    0.476    0.476 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:1099(estimate_eventstudy_effects)
        1    0.005    0.005    0.475    0.475 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:641(_aggregate_eventstudy)
       25    0.002    0.000    0.225    0.009 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:527(_aggregate_eventstudy_influence_by_id)
      135    0.140    0.001    0.178    0.001 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:468(_add_scaled_influence)
       10    0.000    0.000    0.143    0.014 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:593(_cross_eventstudy_covariance)
        5    0.000    0.000    0.128    0.026 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:617(_aggregate_eventstudy_variance)
       45    0.000    0.000    0.109    0.002 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:506(_entry_influence_by_id)
       45    0.005    0.000    0.109    0.002 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:480(_level_entry_influence_by_id)
       10    0.001    0.000    0.088    0.009 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/estimation.py:96(_collapse_to_unit_differences)
        9    0.004    0.000    0.083    0.009 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:405(_local_level_delta_maps_by_id)
        9    0.001    0.000    0.062    0.007 /Users/cxy/Desktop/2026project/contdid/contdid-py/src/contdid/eventstudy.py:156(_build_local_eventstudy_panel)
   324409    0.054    0.000    0.054    0.000 {method 'get' of 'dict' objects}
```

## Identified Bottlenecks

### 已优化的瓶颈

| # | 函数 | 优化前占比 | 问题 | 优化方法 | 加速效果 |
|---|------|-----------|------|----------|----------|
| 1 | `validation._contains_boolean_values` | 40-64% | 对每个元素做 `isinstance(v, (bool, np.bool_))` Python 级别迭代 | 增加 dtype 快速路径：非 object dtype 直接返回 False | **6-7x (两期/多期)** |
| 2 | `eventstudy._add_scaled_influence` | 17-25% (事件研究) | `float(value)` 冗余转换、重复属性查找 | 移除冗余 `float()` 转换，缓存 `target.get` 方法引用 | ~15% 改善 |
| 3 | `eventstudy._centered_mean_influence_by_id` | 事件研究子调用 | 逐元素 dict comprehension 做减均值+缩放 | 改用 `np.fromiter` 向量化计算后用 `dict(zip(...))` 构建结果 | ~10% 改善 |
| 4 | `eventstudy._local_level_delta_maps_by_id` | ~15% (事件研究) | Python for 循环分割 treated/comparison | 改用 numpy boolean mask 向量化分割 | ~20% 改善 |

### 剩余瓶颈（未优化，需架构级改动）

| # | 函数 | 占比 | 原因 | 可能的优化方向 |
|---|------|------|------|---------------|
| 1 | `eventstudy._add_scaled_influence` dict iteration | ~36% (事件研究) | 325k 次 `dict.get()` 调用（n=5000×多轮聚合） | 将影响函数从 `dict[id, float]` 改为 numpy 数组（需架构重构） |
| 2 | `estimation._collapse_to_unit_differences` | ~18% (事件研究) | pandas groupby.agg + sort_values 在每个 local panel 上重复执行 | 缓存/预计算或改用纯 numpy 差分 |
| 3 | `multiperiod.aggregate_influence_functions` | ~28% (多期) | 大规模影响函数矩阵聚合 | numpy 向量化替代逐单元聚合 |
| 4 | `bspline.build_bspline_design` | ~11% (多期) | scipy BSpline 对象重复构建 | 缓存 knot 序列相同的 B-spline basis |

### 优化前后对比 (n=5000, biters=100)

| Scenario | 优化前 | 优化后 | 加速比 |
|----------|--------|--------|--------|
| two_period_dose | 898.9 ms | 117.7 ms | **7.6x** |
| multiperiod_dose | 2.21 s | 327.1 ms | **6.8x** |
| eventstudy | 2.89 s | 859.0 ms | **3.4x** |

### 优化前后对比 (n=10000, biters=100)

| Scenario | 优化前 | 优化后 | 加速比 |
|----------|--------|--------|--------|
| two_period_dose | 1.55 s | 214.7 ms | **7.2x** |
| multiperiod_dose | 1.49 s | 665.0 ms | **2.2x** |
| eventstudy | 3.48 s | 1.16 s | **3.0x** |

### 关键发现

1. **验证层是最大瓶颈**：优化前 `_contains_boolean_values` 占总时间的 40-64%，原因是对数值型列做逐元素 `isinstance` 检查。添加 dtype 快速路径后，验证时间从 ~80ms 降至 ~15ms (n=5000)。

2. **事件研究的字典操作是次要瓶颈**：影响函数使用 `dict[unit_id, float]` 存储，大量 `dict.get()` 调用在 n=5000+ 时成为瓶颈。彻底优化需将数据结构改为 numpy 数组。

3. **内存使用线性增长**：峰值内存与样本量成线性关系（约 1.6 MB/1000 units），在 n=20000 时约 32 MB，完全可接受。

4. **时间复杂度接近线性**：优化后，各场景的时间增长约为 O(n) 至 O(n log n)，scaling ratio 显著优于优化前的超线性增长。

## 稀疏矩阵评估 (2026-06)

### 评估动机

B样条具有局部支撑性：阶为 k 的基函数在设计矩阵每行最多产生 (k+1) 个非零元素。
对于 degree=3，稀疏度 = 4 / (num_knots + 4)。

### 构建性能 (n=5000, degree=3)

| num_knots | K | nnz% | Dense循环(ms) | Sparse手动(ms) | BSpline.design_matrix(ms) | 构建加速 |
|-----------|---|------|-------------|--------------|--------------------------|----------|
| 3 | 7 | 57% | 2.05 | 8.47 | 0.84 | 2.4x |
| 5 | 9 | 44% | 2.80 | 9.13 | 0.83 | 3.4x |
| 10 | 14 | 29% | 4.54 | 16.60 | 1.19 | 3.8x |
| 20 | 24 | 17% | 18.25 | 16.46 | 1.01 | 18.1x |
| 40 | 44 | 9% | 15.44 | 21.80 | 0.94 | 16.4x |

### 矩阵运算性能 (n=5000, num_knots=5)

| 操作 | Dense(µs) | Sparse(µs) | Dense/Sparse |
|------|-----------|------------|-------------|
| B @ coef (矩阵-向量) | 17.5 | 20.5 | 0.85x |
| B^T @ y | 18.6 | 47.0 | 0.40x |
| B^T @ B (Gram 矩阵) | 15.4 | 383.7 | 0.04x |
| 最小二乘求解 | 43.3 | 429.3 | 0.10x |

### 结论与实施决策

**关键发现：**
1. `BSpline.design_matrix` (scipy C 实现) 构建矩阵比 Python 循环快 3-18x
2. 即使加上 `.toarray()` 转换，仍比原循环快 3-8x
3. 稀疏格式的矩阵运算 (Gram、最小二乘) 在 K<20 时比 dense 慢 5-25x
4. 数值精度完美一致：max|diff| = 0.0（bit-exact）

**实施决策：** 采用 `BSpline.design_matrix().toarray()` 替代 Python 循环构建
- 保留 dense 格式用于下游运算（Gram 矩阵、最小二乘）
- 不保留 sparse 格式（运算开销远超收益）
- `build_bspline_design` 从 ~2.8ms 降至 ~0.7ms (n=5000, num_knots=5)

**不实现 sparse="auto" 参数的原因：**
1. contdid 典型 num_knots=3-10，矩阵不够稀疏（nnz > 29%）
2. 下游运算全部需要 dense 格式，保留 sparse 毫无收益
3. 构建加速已通过 `BSpline.design_matrix` 实现，无需额外 API 参数

详见 `tests/benchmarks/bench_sparse.py`
