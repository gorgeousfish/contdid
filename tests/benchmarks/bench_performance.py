#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contdid-py 大样本性能基准测试
==============================

测试场景：
  1. 两期估计 (Two-period dose estimation)
  2. 多期估计 (Multi-period dose estimation)
  3. 事件研究估计 (Event-study estimation)

样本量：n = 1000, 5000, 10000

使用方法：
  cd contdid-py
  python -m tests.benchmarks.bench_performance

输出：格式化性能报告（wall-clock 时间 + 内存峰值）
"""

from __future__ import annotations

import cProfile
import io
import os
import platform
import pstats
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Ensure contdid-py/src is importable
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from contdid import cont_did, simulate_contdid_data  # noqa: E402
from contdid.data import PanelData  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_SIZES = [1000, 5000, 10000, 20000]
BOOTSTRAP_ITERS_SMALL = 1000  # n <= 1000
BOOTSTRAP_ITERS_LARGE = 100   # n > 1000 (减少 bootstrap 以避免超时)
NUM_KNOTS = 5
PROFILE_TOP_N = 15  # cProfile 报告前 N 个热点函数


@dataclass
class BenchmarkResult:
    """单次基准测试结果."""
    scenario: str
    n_units: int
    n_rows: int
    wall_time_sec: float
    peak_memory_mb: float
    profile_stats: pstats.Stats | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def get_system_info() -> dict[str, str]:
    """收集系统和依赖版本信息."""
    import scipy

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
    }
    return info


def run_with_timing_and_memory(
    func: Callable[[], Any],
    label: str = "",
) -> tuple[Any, float, float]:
    """执行函数并测量 wall-clock 时间和峰值内存.

    Returns
    -------
    (result, elapsed_sec, peak_memory_mb)
    """
    tracemalloc.start()
    tracemalloc.reset_peak()

    t_start = time.perf_counter()
    result = func()
    t_end = time.perf_counter()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t_end - t_start
    peak_mb = peak / (1024 * 1024)

    return result, elapsed, peak_mb


def run_with_cprofile(func: Callable[[], Any]) -> tuple[Any, pstats.Stats]:
    """执行函数并收集 cProfile 统计."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    return result, stats


def format_time(seconds: float) -> str:
    """格式化时间显示."""
    if seconds < 1.0:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"


def get_biters(n: int) -> int:
    """根据样本量确定 bootstrap 迭代次数."""
    if n <= 1000:
        return BOOTSTRAP_ITERS_SMALL
    return BOOTSTRAP_ITERS_LARGE


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark scenarios
# ──────────────────────────────────────────────────────────────────────────────

def bench_two_period(n: int) -> BenchmarkResult:
    """基准测试：两期 dose-response 估计."""
    biters = get_biters(n)

    # 生成两期数据
    panel = simulate_contdid_data(
        n=n,
        dgp_id="SIM-005-cck-two-period",
        seed=42,
    )
    n_rows = len(panel.frame)

    def run_estimation():
        return cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            num_knots=NUM_KNOTS,
            biters=biters,
            bstrap=True,
            cband=False,
        )

    _, elapsed, peak_mb = run_with_timing_and_memory(run_estimation, "two-period")

    return BenchmarkResult(
        scenario="two_period_dose",
        n_units=n,
        n_rows=n_rows,
        wall_time_sec=elapsed,
        peak_memory_mb=peak_mb,
    )


def bench_multiperiod(n: int) -> BenchmarkResult:
    """基准测试：多期 dose-response 估计."""
    biters = get_biters(n)

    # 生成4期数据（默认 SIM-001）
    panel = simulate_contdid_data(
        n=n,
        dgp_id="SIM-001-null-dose",
        seed=42,
    )
    n_rows = len(panel.frame)

    def run_estimation():
        return cont_did(
            panel,
            target_parameter="level",
            aggregation="dose",
            num_knots=NUM_KNOTS,
            biters=biters,
            bstrap=True,
            cband=False,
        )

    _, elapsed, peak_mb = run_with_timing_and_memory(run_estimation, "multiperiod")

    return BenchmarkResult(
        scenario="multiperiod_dose",
        n_units=n,
        n_rows=n_rows,
        wall_time_sec=elapsed,
        peak_memory_mb=peak_mb,
    )


def bench_eventstudy(n: int) -> BenchmarkResult:
    """基准测试：事件研究估计."""
    biters = get_biters(n)

    # 生成4期 staggered 数据
    panel = simulate_contdid_data(
        n=n,
        dgp_id="SIM-004-staggered-eventstudy-null",
        seed=42,
    )
    n_rows = len(panel.frame)

    def run_estimation():
        return cont_did(
            panel,
            target_parameter="level",
            aggregation="eventstudy",
            num_knots=NUM_KNOTS,
            biters=biters,
            bstrap=True,
            cband=False,
        )

    _, elapsed, peak_mb = run_with_timing_and_memory(run_estimation, "eventstudy")

    return BenchmarkResult(
        scenario="eventstudy",
        n_units=n,
        n_rows=n_rows,
        wall_time_sec=elapsed,
        peak_memory_mb=peak_mb,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Profiling (detailed for largest sample)
# ──────────────────────────────────────────────────────────────────────────────

def profile_scenario(scenario_name: str, n: int = 5000) -> str:
    """对指定场景做 cProfile，返回热点函数报告."""
    biters = get_biters(n)

    if scenario_name == "two_period_dose":
        panel = simulate_contdid_data(n=n, dgp_id="SIM-005-cck-two-period", seed=42)
        def run():
            return cont_did(panel, aggregation="dose", num_knots=NUM_KNOTS,
                            biters=biters, bstrap=True, cband=False)
    elif scenario_name == "multiperiod_dose":
        panel = simulate_contdid_data(n=n, dgp_id="SIM-001-null-dose", seed=42)
        def run():
            return cont_did(panel, aggregation="dose", num_knots=NUM_KNOTS,
                            biters=biters, bstrap=True, cband=False)
    elif scenario_name == "eventstudy":
        panel = simulate_contdid_data(n=n, dgp_id="SIM-004-staggered-eventstudy-null", seed=42)
        def run():
            return cont_did(panel, aggregation="eventstudy", num_knots=NUM_KNOTS,
                            biters=biters, bstrap=True, cband=False)
    else:
        return f"Unknown scenario: {scenario_name}"

    _, stats = run_with_cprofile(run)

    stream = io.StringIO()
    stats.stream = stream
    stats.print_stats(PROFILE_TOP_N)
    return stream.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks() -> tuple[list[BenchmarkResult], dict[str, str]]:
    """运行所有基准测试."""
    results: list[BenchmarkResult] = []
    sys_info = get_system_info()

    scenarios = [
        ("two_period_dose", bench_two_period),
        ("multiperiod_dose", bench_multiperiod),
        ("eventstudy", bench_eventstudy),
    ]

    print("=" * 72)
    print("contdid-py Performance Benchmark")
    print("=" * 72)
    print()

    # 系统信息
    print("System Information:")
    for k, v in sys_info.items():
        print(f"  {k}: {v}")
    print()

    # 运行各场景各样本量
    for scenario_name, bench_func in scenarios:
        print(f"--- Scenario: {scenario_name} ---")
        for n in SAMPLE_SIZES:
            biters = get_biters(n)
            print(f"  n={n:>6d} (biters={biters}) ... ", end="", flush=True)
            try:
                result = bench_func(n)
                results.append(result)
                print(
                    f"time={format_time(result.wall_time_sec):>10s}  "
                    f"mem={result.peak_memory_mb:.1f} MB  "
                    f"rows={result.n_rows}"
                )
            except Exception as e:
                print(f"FAILED: {e}")
        print()

    return results, sys_info


def print_summary_table(results: list[BenchmarkResult]) -> str:
    """打印汇总表."""
    lines: list[str] = []
    header = f"{'Scenario':<20} {'n_units':>8} {'n_rows':>8} {'Time':>12} {'Memory(MB)':>12} {'biters':>8}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in results:
        biters = get_biters(r.n_units)
        lines.append(
            f"{r.scenario:<20} {r.n_units:>8d} {r.n_rows:>8d} "
            f"{format_time(r.wall_time_sec):>12s} {r.peak_memory_mb:>12.1f} {biters:>8d}"
        )
    lines.append(sep)

    table_str = "\n".join(lines)
    print("\n" + table_str)
    return table_str


def run_profiling(n: int = 5000) -> dict[str, str]:
    """对各场景做 cProfile 分析."""
    print("\n" + "=" * 72)
    print(f"cProfile Analysis (n={n})")
    print("=" * 72)

    profiles: dict[str, str] = {}
    for scenario in ["two_period_dose", "multiperiod_dose", "eventstudy"]:
        print(f"\n--- Profile: {scenario} (n={n}) ---")
        report = profile_scenario(scenario, n=n)
        profiles[scenario] = report
        print(report)

    return profiles


def generate_markdown_report(
    results: list[BenchmarkResult],
    sys_info: dict[str, str],
    profiles: dict[str, str],
) -> str:
    """生成 Markdown 格式的基准报告."""
    lines: list[str] = []
    lines.append("# contdid-py Performance Benchmark Results")
    lines.append("")
    lines.append("## System Information")
    lines.append("")
    lines.append("| Key | Value |")
    lines.append("|-----|-------|")
    for k, v in sys_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Benchmark Configuration")
    lines.append("")
    lines.append(f"- Sample sizes: {SAMPLE_SIZES}")
    lines.append(f"- Bootstrap iterations (n<=1000): {BOOTSTRAP_ITERS_SMALL}")
    lines.append(f"- Bootstrap iterations (n>1000): {BOOTSTRAP_ITERS_LARGE}")
    lines.append(f"- B-spline num_knots: {NUM_KNOTS}")
    lines.append("")

    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Scenario | n_units | n_rows | Wall Time | Peak Memory (MB) | biters |")
    lines.append("|----------|---------|--------|-----------|------------------|--------|")
    for r in results:
        biters = get_biters(r.n_units)
        lines.append(
            f"| {r.scenario} | {r.n_units} | {r.n_rows} | "
            f"{format_time(r.wall_time_sec)} | {r.peak_memory_mb:.1f} | {biters} |"
        )
    lines.append("")

    # Scaling analysis
    lines.append("## Scaling Analysis")
    lines.append("")
    scenarios_seen = []
    for r in results:
        if r.scenario not in scenarios_seen:
            scenarios_seen.append(r.scenario)

    for scenario in scenarios_seen:
        scenario_results = [r for r in results if r.scenario == scenario]
        if len(scenario_results) >= 2:
            lines.append(f"### {scenario}")
            lines.append("")
            base = scenario_results[0]
            for r in scenario_results[1:]:
                n_ratio = r.n_units / base.n_units
                t_ratio = r.wall_time_sec / base.wall_time_sec if base.wall_time_sec > 0 else float('inf')
                lines.append(
                    f"- n={r.n_units} vs n={base.n_units}: "
                    f"n_ratio={n_ratio:.1f}x, time_ratio={t_ratio:.1f}x"
                )
            lines.append("")

    # Profiling
    lines.append("## cProfile Hotspot Analysis")
    lines.append("")
    for scenario, report in profiles.items():
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("```")
        lines.append(report.strip())
        lines.append("```")
        lines.append("")

    lines.append("## Identified Bottlenecks")
    lines.append("")
    lines.append("(Populated after profiling run)")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 运行基准测试
    results, sys_info = run_all_benchmarks()

    # 打印汇总表
    print_summary_table(results)

    # cProfile 分析（对 n=5000）
    profiles = run_profiling(n=5000)

    # 生成 Markdown 报告
    report_md = generate_markdown_report(results, sys_info, profiles)

    # 写入文件
    output_path = Path(__file__).parent / "BENCHMARK_RESULTS.md"
    output_path.write_text(report_md, encoding="utf-8")
    print(f"\nBenchmark report written to: {output_path}")
