# ── R/03_models.R — Model definitions ────────────────────────────────────────
#
# model_base : ARMA(1,1) + DOW + WOM dummies (8 regressors)
#              Simpler baseline capturing only day-of-week and week-of-month effects.
#
# model_2022 : ARMA(1,1) + full 55-dummy matrix replicating the BOT 2022 production
#              model (Python equivalent: Old_2022 / TwoStepARIMAX).
#              Adds day-of-month (30), month-of-year (11), and holiday dummies (6)
#              on top of model_base.

DOM_COLS <- paste0("Date_", sprintf("%02d", 2:31))   # 30 day-of-month dummies
DOW_COLS <- c("D_TUE", "D_WED", "D_THU", "D_FRI")  # 4  day-of-week
WOM_COLS <- c("D_WEEK2", "D_WEEK3", "D_WEEK4", "D_WEEK5")  # 4 week-of-month
MON_COLS <- c("D_JAN", "D_FEB", "D_MAR", "D_APR", "D_MAY", "D_JUN",
              "D_JUL", "D_AUG", "D_SEP", "D_OCT", "D_NOV")  # 11 month
HOL_OLD  <- c("D_PRE_LH1", "D_PRE_LH3", "D_POST_LH3",
               "D_PRE_SH1", "D_Covid_1st", "D_LWD")          # 6  holiday

REGS <- list(
  model_base = c(DOW_COLS, WOM_COLS),
  model_2022 = c(DOM_COLS, DOW_COLS, WOM_COLS, MON_COLS, HOL_OLD)
)

MODEL_LABELS <- c(
  model_base = "Baseline (DOW+WOM, 8 dummies)",
  model_2022 = "2022 BOT Model (55 dummies)"
)
MODEL_COLORS <- c(model_base = "#1f77b4", model_2022 = "#d62728")


get_X <- function(df_sub, model_name) {
  cols <- intersect(REGS[[model_name]], names(df_sub))
  if (length(cols) == 0L) return(NULL)
  as.matrix(df_sub[, cols, drop = FALSE])
}


fit_model <- function(df_sub, model_name) {
  y <- ts(df_sub$Change)
  X <- get_X(df_sub, model_name)
  if (!is.null(X) && ncol(X) > 0L) {
    forecast::Arima(y, order = c(1L, 0L, 1L), xreg = X, include.mean = TRUE)
  } else {
    forecast::Arima(y, order = c(1L, 0L, 1L), include.mean = TRUE)
  }
}


summarise_fit <- function(mdl, model_name) {
  cf    <- coef(mdl)
  ar1   <- if ("ar1" %in% names(cf)) cf["ar1"] else NA_real_
  ma1   <- if ("ma1" %in% names(cf)) cf["ma1"] else NA_real_
  sigma <- sqrt(mdl$sigma2)
  cat(sprintf(
    "  %-35s  AIC=%9.1f  BIC=%9.1f  sigma=%.3f  AR1=%.3f  MA1=%.3f\n",
    MODEL_LABELS[model_name], mdl$aic, mdl$bic, sigma, ar1, ma1
  ))
}
