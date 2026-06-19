# ── R/04_backtest.R — Out-of-sample evaluation ───────────────────────────────

oos_window <- function(df, train_end, eval_start, eval_end) {
  df_tr <- df[df$Date <= as.Date(train_end),  ]
  df_ev <- df[df$Date >= as.Date(eval_start) & df$Date <= as.Date(eval_end), ]

  if (nrow(df_tr) < 200L || nrow(df_ev) < 5L) {
    warning("Insufficient data for OOS window.")
    return(NULL)
  }

  results <- list()
  for (mname in names(REGS)) {
    mdl <- tryCatch(
      fit_model(df_tr, mname),
      error = function(e) { warning(sprintf("[%s] fit failed: %s", mname, e$message)); NULL }
    )
    if (is.null(mdl)) next

    X_ev <- get_X(df_ev, mname)
    fc <- if (!is.null(X_ev) && ncol(X_ev) > 0L) {
      as.numeric(forecast::forecast(mdl, xreg = X_ev)$mean)
    } else {
      as.numeric(forecast::forecast(mdl, h = nrow(df_ev))$mean)
    }

    actual <- df_ev$Change
    errors <- actual - fc
    results[[mname]] <- list(
      dates    = df_ev$Date,
      actual   = actual,
      forecast = fc,
      errors   = errors,
      RMSE     = sqrt(mean(errors^2L, na.rm = TRUE)),
      MAE      = mean(abs(errors),    na.rm = TRUE),
      Bias     = mean(errors,         na.rm = TRUE)
    )
  }
  results
}


print_rmse_table <- function(res, window_label = "") {
  if (!nzchar(window_label)) window_label <- "OOS"
  cat(sprintf("\n  Window: %s\n", window_label))
  cat(sprintf("  %-35s  %8s  %8s  %8s\n", "Model", "RMSE", "MAE", "Bias"))
  cat("  ", strrep("-", 64L), "\n", sep = "")
  for (mname in names(res)) {
    r <- res[[mname]]
    cat(sprintf("  %-35s  %8.3f  %8.3f  %8.3f\n",
                MODEL_LABELS[mname], r$RMSE, r$MAE, r$Bias))
  }
}
