# ── R/08_eda.R — EDA and model comparison (presentation quality) ──────────────
#
# Requires: df, REGS, MODEL_LABELS, MODEL_COLORS, fit_model(), get_X(),
#           summarise_fit(), oos_window(), print_rmse_table()
#           (all sourced by 00_main.R before this script runs)
#
# FIX: demand_total_real_min / demand_total_real_max must be columns in the
#      data frame passed to ggplot(), not bare global variables. They are
#      computed here in the monthly_cic summary and used via aes(y = ...).

# ── 1. Prepare EDA data ───────────────────────────────────────────────────────

eda_df <- df[!is.na(df$Currency), c("Date", "Currency", "Change")]

# Rename Currency → demand_total_real for consistency with the EDA convention
eda_df$demand_total_real <- eda_df$Currency

# Monthly summary: mean, min, max of CIC level (demand_total_real)
# demand_total_real_min / demand_total_real_max are columns in this data frame.
# ggplot2 resolves them from the data argument — NOT the global environment.
monthly_cic <- eda_df |>
  dplyr::mutate(year_month = lubridate::floor_date(Date, "month")) |>
  dplyr::group_by(year_month) |>
  dplyr::summarise(
    demand_total_real     = mean(demand_total_real, na.rm = TRUE),
    demand_total_real_min = min(demand_total_real,  na.rm = TRUE),
    demand_total_real_max = max(demand_total_real,  na.rm = TRUE),
    .groups = "drop"
  )

# ── 2. Figure A — CIC Level with intra-month range ribbon ────────────────────

p_cic_level <- ggplot2::ggplot(monthly_cic, ggplot2::aes(x = year_month)) +
  ggplot2::geom_ribbon(
    ggplot2::aes(ymin = demand_total_real_min, ymax = demand_total_real_max),
    fill = "#1f77b4", alpha = 0.15
  ) +
  ggplot2::geom_line(
    ggplot2::aes(y = demand_total_real),
    color = "#1f77b4", linewidth = 0.9
  ) +
  ggplot2::geom_line(
    ggplot2::aes(y = demand_total_real_min),   # 2nd geom_line — now resolves correctly
    color = "#1f77b4", linewidth = 0.35, linetype = "dashed"
  ) +
  ggplot2::geom_line(
    ggplot2::aes(y = demand_total_real_max),
    color = "#1f77b4", linewidth = 0.35, linetype = "dashed"
  ) +
  ggplot2::annotate(
    "rect",
    xmin = as.Date("2020-03-01"), xmax = as.Date("2020-12-31"),
    ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.08
  ) +
  ggplot2::labs(
    title    = "Currency in Circulation — Monthly Level",
    subtitle = "Ribbon = intra-month min/max range  |  Solid = monthly mean  |  Shaded = COVID 2020",
    x = NULL, y = "CIC Level (THB billion)"
  ) +
  ggplot2::scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  ggplot2::scale_y_continuous(labels = scales::comma) +
  ggplot2::theme_minimal(base_size = 12)

print(p_cic_level)

# ── 3. Figure B — Daily ΔCIC series ──────────────────────────────────────────

p_change <- ggplot2::ggplot(df, ggplot2::aes(x = Date, y = Change)) +
  ggplot2::geom_line(color = "#2ca02c", linewidth = 0.35, alpha = 0.75) +
  ggplot2::geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed") +
  ggplot2::annotate(
    "rect",
    xmin = as.Date("2020-03-01"), xmax = as.Date("2020-12-31"),
    ymin = -Inf, ymax = Inf, fill = "red", alpha = 0.08
  ) +
  ggplot2::labs(
    title = "Daily Change in CIC — Model Dependent Variable (ΔCIC)",
    x = NULL, y = "Daily Change (THB billion)"
  ) +
  ggplot2::scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  ggplot2::theme_minimal(base_size = 12)

print(p_change)

# ── 4. In-sample fit: model_2022 vs model_base ────────────────────────────────

cat("\n---- In-sample fit (full training data) ----\n")
fitted_models <- list()
for (mname in names(REGS)) {
  cat(sprintf("Fitting %s (%d regressors)...\n", mname, length(REGS[[mname]])))
  mdl <- fit_model(df, mname)
  summarise_fit(mdl, mname)
  fitted_models[[mname]] <- mdl
}

# Residual comparison
cat("\n---- Residual diagnostics ----\n")
for (mname in names(fitted_models)) {
  res <- residuals(fitted_models[[mname]])
  res <- res[!is.na(res)]
  adf_p <- tryCatch(tseries::adf.test(res)$p.value, error = function(e) NA_real_)
  cat(sprintf(
    "  %-35s  ResidSD=%.3f  ADF p=%.4f  %s\n",
    MODEL_LABELS[mname], sd(res),
    adf_p,
    ifelse(!is.na(adf_p) && adf_p < 0.05, "stationary ✓", "non-stationary ⚠")
  ))
}

# ── 5. OOS benchmark (Dec 2021 – May 2022) ───────────────────────────────────

cat("\n---- OOS Backtest: benchmark window (Dec 2021 – May 2022) ----\n")
bench <- oos_window(df, "2021-11-30", "2021-12-01", "2022-05-31")
if (!is.null(bench)) print_rmse_table(bench, "Dec 2021 – May 2022")

# ── 6. Figure C — Actual vs Forecast (benchmark window) ──────────────────────

if (!is.null(bench)) {
  df_eval <- df[df$Date >= as.Date("2021-12-01") & df$Date <= as.Date("2022-05-31"), ]

  fc_rows <- lapply(names(bench), function(mname) {
    data.frame(
      Date     = bench[[mname]]$dates,
      forecast = bench[[mname]]$forecast,
      model    = MODEL_LABELS[mname],
      stringsAsFactors = FALSE
    )
  })
  fc_df <- do.call(rbind, fc_rows)

  subtitle_txt <- paste(
    sapply(names(bench), function(m)
      sprintf("%s: RMSE=%.3f", MODEL_LABELS[m], bench[[m]]$RMSE)),
    collapse = "   |   "
  )

  color_map <- c(
    Actual = "#333333",
    setNames(unname(MODEL_COLORS), unname(MODEL_LABELS))
  )

  p_fc <- ggplot2::ggplot() +
    ggplot2::geom_line(
      data = df_eval,
      ggplot2::aes(x = Date, y = Change, color = "Actual"),
      linewidth = 1.2
    ) +
    ggplot2::geom_line(
      data = fc_df,
      ggplot2::aes(x = Date, y = forecast, color = model),
      linewidth = 0.85, alpha = 0.85
    ) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed") +
    ggplot2::scale_color_manual(values = color_map) +
    ggplot2::labs(
      title    = "Actual vs Forecast — Daily ΔCIC (Dec 2021 – May 2022)",
      subtitle = subtitle_txt,
      x = NULL, y = "Daily Change (THB billion)", color = NULL
    ) +
    ggplot2::scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(legend.position = "bottom")

  print(p_fc)

  # ── 7. Figure D — RMSE bar chart ─────────────────────────────────────────────

  rmse_df <- data.frame(
    model  = factor(MODEL_LABELS[names(bench)], levels = unname(MODEL_LABELS)),
    RMSE   = sapply(bench, `[[`, "RMSE"),
    color  = unname(MODEL_COLORS[names(bench)]),
    stringsAsFactors = FALSE
  )

  p_rmse <- ggplot2::ggplot(rmse_df, ggplot2::aes(x = model, y = RMSE, fill = model)) +
    ggplot2::geom_col(alpha = 0.85, width = 0.5, show.legend = FALSE) +
    ggplot2::geom_hline(yintercept = 4.96, linetype = "dashed",
                        color = "black", linewidth = 1.0) +
    ggplot2::geom_hline(yintercept = 7.31, linetype = "dotted",
                        color = "grey50", linewidth = 1.0) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", RMSE)),
                       vjust = -0.4, fontface = "bold", size = 4.5) +
    ggplot2::annotate("text", x = 0.55, y = 4.96 + 0.12,
                      label = "BOT 2022 paper: 4.96", hjust = 0, size = 3.5) +
    ggplot2::annotate("text", x = 0.55, y = 7.31 + 0.12,
                      label = "Pre-2022 model: 7.31", hjust = 0, size = 3.5) +
    ggplot2::scale_fill_manual(values = setNames(rmse_df$color, rmse_df$model)) +
    ggplot2::labs(
      title = "RMSE Comparison — model_2022 vs model_base",
      subtitle = "Benchmark window: Dec 2021 – May 2022  (lower is better)",
      x = NULL, y = "RMSE (THB billion)"
    ) +
    ggplot2::ylim(0, max(rmse_df$RMSE, 8.5) * 1.1) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 10, hjust = 0.5))

  print(p_rmse)
}

cat("\n---- 08_eda.R complete ----\n")
