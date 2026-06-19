# ── R/01_setup.R — Package loading ───────────────────────────────────────────

required_pkgs <- c(
  "readxl", "dplyr", "lubridate", "ggplot2",
  "forecast", "here", "tidyr", "scales", "tseries"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, quiet = TRUE, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
