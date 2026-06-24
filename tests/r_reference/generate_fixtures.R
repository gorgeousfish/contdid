#!/usr/bin/env Rscript
# =============================================================================
# Python↔R 数值一致性验证: R基准数据生成脚本
#
# 用途: 生成R包 contdid 的估计结果作为Python包验证的参考基准
# 运行: Rscript generate_fixtures.R
# 依赖: contdid (>= 0.1.0), jsonlite, tidyr
#
# 输出目录: fixtures/
#   - parametric_dose_att.json    : parametric路径ATT(d)结果
#   - parametric_dose_acrt.json   : parametric路径ACRT(d)结果
#   - cck_dose_att.json           : CCK路径ATT(d)结果
#   - eventstudy_att.json         : 事件研究ATT结果
#   - two_period_parametric.json  : 两期parametric结果（精确对比）
#   - two_period_cck.json         : 两期CCK结果（精确对比）
#
# 注意: 本脚本的DGP参数与 dgp_params.json 严格一致
# =============================================================================

library(contdid)
library(jsonlite)

cat("=== contdid R fixture generation ===\n")
cat("contdid version:", as.character(packageVersion("contdid")), "\n")
cat("R version:", R.version.string, "\n\n")

# 输出目录
output_dir <- file.path(dirname(sys.frame(1)$ofile), "fixtures")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ============================================================================
# 辅助函数
# ============================================================================

save_fixture <- function(data, filename) {
  path <- file.path(output_dir, filename)
  write_json(data, path, pretty = TRUE, auto_unbox = TRUE)
  cat("  Saved:", path, "\n")
}

extract_dose_results <- function(res, target = "att") {
  # 从 cont_did 结果对象中提取ATT(d)或ACRT(d)估计
  # cont_did 使用 ptetools 框架, 结果结构取决于版本
  if (inherits(res, "pte_results")) {
    # ptetools framework
    agg_res <- res$overall_att
    if (target == "acrt") {
      agg_res <- res$overall_acrt
    }
    # dose aggregation 的结果在 $dose_response 或类似字段
    dr <- res$dose_response
    if (!is.null(dr)) {
      return(list(
        grid = dr$dvals,
        estimate = dr$estimate,
        std_error = dr$se,
        ci_lower = dr$ci_lower,
        ci_upper = dr$ci_upper
      ))
    }
  }

  # Fallback: 尝试 summary 提取
  s <- summary(res)
  return(list(
    grid = s$dvals,
    estimate = s$estimate,
    std_error = s$se,
    ci_lower = if (!is.null(s$ci_lower)) s$ci_lower else s$estimate - 1.96 * s$se,
    ci_upper = if (!is.null(s$ci_upper)) s$ci_upper else s$estimate + 1.96 * s$se
  ))
}

# ============================================================================
# 场景1: 4期staggered - parametric dose ATT
# ============================================================================

cat("\n--- Scenario: 4-period staggered, parametric dose ATT ---\n")

# SIM-002-linear-dose
set.seed(23456)
df_linear <- simulate_contdid_data(
  n = 5000,
  num_time_periods = 4,
  num_groups = 4,
  pg = rep(0.25, 3),
  pu = 0.25,
  dose_linear_effect = 1.0,
  dose_quadratic_effect = 0.0
)

res_att_linear <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_linear,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

# SIM-003-quadratic-dose
set.seed(34567)
df_quad <- simulate_contdid_data(
  n = 5000,
  num_time_periods = 4,
  num_groups = 4,
  pg = rep(0.25, 3),
  pu = 0.25,
  dose_linear_effect = 0.0,
  dose_quadratic_effect = 1.0
)

res_att_quad <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_quad,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

fixture_att <- list(
  meta = list(
    generator = "contdid R package",
    generator_version = as.character(packageVersion("contdid")),
    r_version = R.version.string,
    generation_date = Sys.time(),
    consistency_threshold = 3.0,
    consistency_metric = "max|bias/SE|"
  ),
  estimation_params = list(
    target_parameter = "level",
    aggregation = "dose",
    dose_est_method = "parametric",
    control_group = "nevertreated",
    degree = 3,
    num_knots = 0,
    biters = 1000
  ),
  scenarios = list(
    "SIM-002-linear-dose" = list(
      dgp = list(n = 5000, num_time_periods = 4, num_groups = 4,
                 dose_linear_effect = 1.0, dose_quadratic_effect = 0.0, seed = 23456),
      results = extract_dose_results(res_att_linear, "att")
    ),
    "SIM-003-quadratic-dose" = list(
      dgp = list(n = 5000, num_time_periods = 4, num_groups = 4,
                 dose_linear_effect = 0.0, dose_quadratic_effect = 1.0, seed = 34567),
      results = extract_dose_results(res_att_quad, "att")
    )
  )
)

save_fixture(fixture_att, "parametric_dose_att.json")

# ============================================================================
# 场景2: 4期staggered - parametric dose ACRT
# ============================================================================

cat("\n--- Scenario: 4-period staggered, parametric dose ACRT ---\n")

res_acrt_linear <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_linear,
  target_parameter = "slope",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

res_acrt_quad <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_quad,
  target_parameter = "slope",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

fixture_acrt <- list(
  meta = list(
    generator = "contdid R package",
    generator_version = as.character(packageVersion("contdid")),
    r_version = R.version.string,
    generation_date = Sys.time(),
    consistency_threshold = 3.0,
    consistency_metric = "max|bias/SE|"
  ),
  estimation_params = list(
    target_parameter = "slope",
    aggregation = "dose",
    dose_est_method = "parametric",
    control_group = "nevertreated",
    degree = 3,
    num_knots = 0,
    biters = 1000
  ),
  scenarios = list(
    "SIM-002-linear-dose" = list(
      dgp = list(n = 5000, num_time_periods = 4, num_groups = 4,
                 dose_linear_effect = 1.0, dose_quadratic_effect = 0.0, seed = 23456),
      results = extract_dose_results(res_acrt_linear, "acrt")
    ),
    "SIM-003-quadratic-dose" = list(
      dgp = list(n = 5000, num_time_periods = 4, num_groups = 4,
                 dose_linear_effect = 0.0, dose_quadratic_effect = 1.0, seed = 34567),
      results = extract_dose_results(res_acrt_quad, "acrt")
    )
  )
)

save_fixture(fixture_acrt, "parametric_dose_acrt.json")

# ============================================================================
# 场景3: 两期 CCK ATT（两期是CCK目前支持的场景）
# ============================================================================

cat("\n--- Scenario: 2-period, CCK dose ATT ---\n")

set.seed(56789)
df_tp_linear <- simulate_contdid_data(
  n = 5000,
  num_time_periods = 2,
  num_groups = 2,
  pg = c(0.75),
  pu = 0.25,
  dose_linear_effect = 1.0,
  dose_quadratic_effect = 0.0
)

res_cck_linear <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_tp_linear,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "cck",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

set.seed(67890)
df_tp_quad <- simulate_contdid_data(
  n = 5000,
  num_time_periods = 2,
  num_groups = 2,
  pg = c(0.75),
  pu = 0.25,
  dose_linear_effect = 0.0,
  dose_quadratic_effect = 1.0
)

res_cck_quad <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_tp_quad,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "cck",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

fixture_cck <- list(
  meta = list(
    generator = "contdid R package",
    generator_version = as.character(packageVersion("contdid")),
    r_version = R.version.string,
    generation_date = Sys.time(),
    consistency_threshold = 3.0,
    consistency_metric = "max|bias/SE|"
  ),
  estimation_params = list(
    target_parameter = "level",
    aggregation = "dose",
    dose_est_method = "cck",
    control_group = "nevertreated",
    degree = 3,
    num_knots = 0,
    biters = 1000
  ),
  scenarios = list(
    "SIM-TP-linear" = list(
      dgp = list(n = 5000, num_time_periods = 2, num_groups = 2,
                 pg = c(0.75), pu = 0.25,
                 dose_linear_effect = 1.0, dose_quadratic_effect = 0.0, seed = 56789),
      results = extract_dose_results(res_cck_linear, "att")
    ),
    "SIM-TP-quadratic" = list(
      dgp = list(n = 5000, num_time_periods = 2, num_groups = 2,
                 pg = c(0.75), pu = 0.25,
                 dose_linear_effect = 0.0, dose_quadratic_effect = 1.0, seed = 67890),
      results = extract_dose_results(res_cck_quad, "att")
    )
  )
)

save_fixture(fixture_cck, "cck_dose_att.json")

# ============================================================================
# 场景4: 事件研究
# ============================================================================

cat("\n--- Scenario: 4-period staggered, event study ATT ---\n")

# 使用线性效应场景做事件研究
res_es <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_linear,
  target_parameter = "level",
  aggregation = "eventstudy",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

fixture_es <- list(
  meta = list(
    generator = "contdid R package",
    generator_version = as.character(packageVersion("contdid")),
    r_version = R.version.string,
    generation_date = Sys.time(),
    consistency_threshold = 3.0,
    consistency_metric = "max|bias/SE|"
  ),
  estimation_params = list(
    target_parameter = "level",
    aggregation = "eventstudy",
    dose_est_method = "parametric",
    control_group = "nevertreated",
    degree = 3,
    num_knots = 0,
    biters = 1000
  ),
  scenarios = list(
    "SIM-002-linear-dose" = list(
      dgp = list(n = 5000, num_time_periods = 4, num_groups = 4,
                 dose_linear_effect = 1.0, dose_quadratic_effect = 0.0, seed = 23456),
      results = extract_dose_results(res_es, "att")
    )
  )
)

save_fixture(fixture_es, "eventstudy_att.json")

# ============================================================================
# 场景5: 两期 parametric（用于精确数值对比）
# ============================================================================

cat("\n--- Scenario: 2-period, parametric (exact comparison) ---\n")

res_tp_param_linear <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_tp_linear,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

res_tp_param_quad <- cont_did(
  yname = "Y", dname = "D", gname = "G",
  tname = "time_period", idname = "id",
  data = df_tp_quad,
  target_parameter = "level",
  aggregation = "dose",
  dose_est_method = "parametric",
  control_group = "nevertreated",
  degree = 3, num_knots = 0,
  biters = 1000, cband = FALSE
)

fixture_tp_param <- list(
  meta = list(
    generator = "contdid R package",
    generator_version = as.character(packageVersion("contdid")),
    r_version = R.version.string,
    generation_date = Sys.time(),
    consistency_threshold = 3.0,
    consistency_metric = "max|bias/SE|"
  ),
  estimation_params = list(
    target_parameter = "level",
    aggregation = "dose",
    dose_est_method = "parametric",
    control_group = "nevertreated",
    degree = 3,
    num_knots = 0,
    biters = 1000
  ),
  scenarios = list(
    "SIM-TP-linear" = list(
      dgp = list(n = 5000, num_time_periods = 2, num_groups = 2,
                 pg = c(0.75), pu = 0.25,
                 dose_linear_effect = 1.0, dose_quadratic_effect = 0.0, seed = 56789),
      results = extract_dose_results(res_tp_param_linear, "att")
    ),
    "SIM-TP-quadratic" = list(
      dgp = list(n = 5000, num_time_periods = 2, num_groups = 2,
                 pg = c(0.75), pu = 0.25,
                 dose_linear_effect = 0.0, dose_quadratic_effect = 1.0, seed = 67890),
      results = extract_dose_results(res_tp_param_quad, "att")
    )
  )
)

save_fixture(fixture_tp_param, "two_period_parametric.json")

# ============================================================================
cat("\n=== All fixtures generated successfully ===\n")
cat("Output directory:", output_dir, "\n")
