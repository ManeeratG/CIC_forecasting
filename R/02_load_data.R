# ── R/02_load_data.R — Load CIC data from input.xlsx ────────────────────────
#
# The Excel sheet "RAW" has headers in row 2 (skip=1 in R, matching pandas header=1).
# Dummy columns (D_*, Date_*) are pre-computed in the sheet.
# Currency = CIC level in THB billion.
# Change   = daily first-difference of Currency (ΔCIC), the model target.

load_cic_data <- function(filepath = here::here("input.xlsx")) {
  raw <- readxl::read_excel(filepath, sheet = "RAW", skip = 1)

  raw$Date     <- as.Date(raw$Date)
  raw$Currency <- suppressWarnings(as.numeric(raw$Currency))
  raw          <- raw[!is.na(raw$Date), ]
  raw          <- raw[order(raw$Date), ]
  raw$Change   <- c(NA_real_, diff(raw$Currency))
  df           <- raw[!is.na(raw$Change), ]

  # Coerce all dummy columns to numeric, replacing NA with 0
  dummy_cols <- grep("^(D_|Date_)", names(df), value = TRUE)
  for (col in dummy_cols) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
    df[[col]][is.na(df[[col]])] <- 0
  }

  df
}

df <- load_cic_data()
cat(sprintf(
  "Data loaded: %d obs  %s → %s\n",
  nrow(df), format(min(df$Date)), format(max(df$Date))
))
