# ── R/00_main.R — Master runner ───────────────────────────────────────────────
#
# Sources all steps in sequence.  Run this file to reproduce the full analysis.
# Each step can also be sourced individually once earlier steps are in memory.

# ---- 1. Packages -------------------------------------------------------
source(here::here("R", "01_setup.R"))

# ---- 2. Load data ------------------------------------------------------
source(here::here("R", "02_load_data.R"))

# ---- 3. Model definitions & helpers ------------------------------------
source(here::here("R", "03_models.R"))

# ---- 4. Backtest helpers -----------------------------------------------
source(here::here("R", "04_backtest.R"))

# ---- 8. Exploratory data analysis (optional, presentation quality) -----
source(here::here("R", "08_eda.R"))
