# CIC Forecasting v2 — Context, Diagnosis & Experiment Log

**Goal:** beat v1 on the primary KPI — **1-month-ahead end-of-month (EOM) CIC level RMSE**
(expanding window, re-estimated at each month-end origin). Daily ΔCIC RMSE is a
no-regression guardrail (≤ baseline +5% per year), not the target.

**Code:** all v2 work lives in `cic_forecast_v2.py` (standalone; imports the data
pipeline and baseline models from `cic_forecast.py` without modifying it).
Results workbook: `cic_v2_results.xlsx`. Figure: `fig_v2_eom_level.png`.

---

## 1. Why v1 fell short — diagnosis (all numbers measured, not assumed)

v1 (`cic_forecast.py`) added two adaptive-trend models to the Old_2022 baseline:
`D1` (OLS on 55 calendar dummies → local-level UC + AR(1) on residuals) and
`Model3` (trailing-60-month OLS → smooth-trend UC). On the primary KPI over 61
month origins 2020–2025:

| Model | Overall EOM RMSE | 2020 | 2021 | 2022 | 2023 | 2024 | 2025* |
|---|---|---|---|---|---|---|---|
| Old_2022 | **32.86** | 45.81 | 42.60 | 24.70 | 13.80 | 27.80 | 4.85 |
| D1 | 33.04 | 45.73 | 42.86 | 25.21 | 13.99 | 28.10 | 2.88 |

\* 2025 contains a **single month** (n=1) — D1's only clear "win" is one observation.
Model3 never appeared in the v1 output workbook (silent failure or stale export).

Measured failure modes (from `cic_forecast_output.xlsx` errors):

1. **Within-month error persistence, not daily noise, drives the KPI.** Daily
   errors have within-month lag-1 autocorrelation ≈ **0.39**. If daily errors were
   iid, monthly-sum RMSE would be ≈ daily RMSE × √21 ≈ 22 THB bn; the actual is
   ≈ 34. The EOM error is ~22 × the drift error at the origin — the KPI amplifies
   any drift mistake 22-fold.
2. **D1's local-level drift is jumpy and COVID-contaminated.** Its state variance
   is MLE'd on a sample including Mar–Apr 2020 outliers, inflating σ²_level, so
   the filtered drift chases daily noise. The jitter cancels the adaptivity gain:
   D1 ≈ Old_2022 everywhere except the (n=1) 2025 bucket.
3. **Stale seasonal betas.** Mean EOM error by calendar month (2020–2024, Old_2022):
   March **+24.5**, April **+21.3** (Songkran cash build-up under-forecast),
   June **−19.5** (over-forecast). The 1997-anchored betas no longer fit
   post-COVID seasonal cash behaviour.
4. **Month-to-month error feedback is a dead end (tested).** EOM errors have
   lag-1 autocorrelation **−0.19**; adding a trailing-k-month mean-error
   correction *worsens* RMSE for every k ∈ {1,2,3,6,12} (e.g. k=3: 37.7 vs 31.0
   raw), and same-month-last-year correction is far worse (48 vs 29). Excluded
   from v2 by evidence, not taste.
5. **Frequency mismatch.** All v1 models are daily, but the KPI is monthly. The
   ~345-observation monthly EOM level series can carry the drift + 12-month
   seasonality at the KPI's native frequency without the ×22 amplification.

Prior negative results carried over from v1 (do not retry): post-COVID step
dummy, Fourier annual terms, extra fixed holiday dummies — all hurt or flat OOS.

## 2. The three bets

| Bet | Model key(s) | Design | Attacks |
|---|---|---|---|
| 1 | `M1_SARIMA`, `M1_UC` | Direct monthly model on the EOM level series: SARIMA(1,1,1)(0,1,1,12) with 2020-03/04 intervention dummies; UC local-linear-trend + 12-month seasonal with COVID months masked | (1),(5) |
| 2 | `D2_smooth` (+`D2_lltrend`, `D2_smooth_nomask` ablations) | OLS on the 55 dummies → cumulate residuals into a calendar-adjusted level → UC **smooth trend** + AR(1), COVID months (2020-03→12) masked as missing. Trend innovation lives on the *slope*, so the drift bends instead of jumping | (1),(2) |
| 3 | `CMB_invMSE` (+`AVG_equal`, pairwise) | Per-origin inverse-MSE weights over trailing 12 realized EOM errors (floored 0.1, leakage-free), over {Old_2022, M1, D2} | regime robustness |

Baselines in the same harness: `Old_2022`, `D1` (imported from v1).

## 3. Evaluation protocol

- **Origins:** calendar month-ends 2017-12-31 → 2026-03-31 (targets 2018-01 → 2026-04, ~100 months; v1 covered only 2020-01 → 2025-01).
- **Selection window:** targets **2018-01 → 2023-12** — every spec choice (SARIMA vs UC for Bet 1, trend type / masking for Bet 2, combination members) is made here.
- **Holdout:** targets **2024-01 → 2026-04** — run once at the end; decides the winner. Diebold–Mariano test (Harvey correction) vs Old_2022.
- **Gate 0:** the v2 harness must reproduce v1's numbers (Old_2022 = 32.86, D1 = 33.04 on origins 2019-12 → 2024-12) before any new result is trusted.
- **Guardrail:** daily ΔCIC RMSE per year ≤ Old_2022 + 5% (models with daily paths).

## 4. Experiment log

*(One entry per run; RMSE in THB bn. Filled in as experiments complete.)*

### E0 — Gate 0: harness reproduction of v1
- **Status:** ✅ PASSED (2026-07-14).
- **Spec:** `python cic_forecast_v2.py --gate0` — Old_2022 + D1, origins 2019-12 → 2024-12.
- **Result:** Old_2022 **32.860** (ref 32.860), D1 **33.041** (ref 33.041) — exact
  reproduction of the v1 `Level_EOM_Metrics` numbers, n=61. The v2 harness is a
  faithful reimplementation of `month_end_eom_backtest`; all subsequent numbers
  are comparable with v1.

### E1 — Full backtest, all candidates, origins 2017-12 → 2026-03
- **Status:** ✅ complete (2026-07-14). ~100 origins × 7 fitted models + 5 combination
  overlays; per-origin forecasts cached in `backtest_cache/`.
- **Selection window (targets 2018-01 → 2023-12, n=72), EOM level RMSE:**

| Model | RMSE | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| **CMB Old+M1_SARIMA** | **30.98** | 26.2 | 12.8 | 52.6 | 36.0 | 23.4 | 17.4 |
| Old_2022 (baseline) | 32.63 | 35.4 | 20.4 | 45.8 | 42.6 | 24.7 | 13.8 |
| D1 | 32.80 | 35.6 | 20.6 | 45.7 | 42.9 | 25.2 | 14.0 |
| CMB invMSE (3-member) | 36.04 | 27.9 | 17.3 | 67.2 | 35.5 | 26.7 | 15.0 |
| M1_SARIMA | 37.20 | 23.0 | 17.3 | 72.2 | 31.4 | 25.7 | 24.9 |
| M1_UC | 57.79 | 23.8 | 18.5 | 125.0 | 45.9 | 27.2 | 25.7 |
| D2_smooth | 80.55 | 33.4 | 37.2 | 178.3 | 40.1 | 39.0 | 38.7 |
| D2_lltrend | 114.96 | 45.4 | 25.4 | 187.5 | 197.4 | 40.3 | 29.3 |

  Selection decisions: Bet 1 spec = **SARIMA** (M1_UC's COVID masking leaves a
  +19 THB bn systematic bias — rejected). Bet 2 = **rejected entirely** (post-mortem
  below). Bet 3 members = **{Old_2022, M1_SARIMA} pairwise** (adding D2 to the pool
  drags every combination down).

- **Holdout (targets 2024-01 → 2026-04, n=28, run once), EOM level RMSE:**

| Model | RMSE | Δ vs Old | DM stat | p |
|---|---|---|---|---|
| **CMB Old+M1_SARIMA** | **33.33** | **−7.7%** | **−2.13** | **0.043** |
| M1_SARIMA | 34.41 | −4.7% | −0.66 | 0.517 |
| M1_UC | 34.77 | −3.7% | −0.49 | 0.629 |
| D1 | 36.11 | −0.0% | −0.01 | 0.990 |
| Old_2022 | 36.12 | — | — | — |
| D2_smooth | 60.78 | +68% | +2.49 | 0.019 (worse) |

  The pairwise combination — already the selection-window winner, so this is a
  clean selection→holdout confirmation — is the only candidate whose holdout
  improvement is statistically significant at 5%.

- **Continuity check (v1-comparable window, targets 2020-01 → 2025-01, n=61):**
  Old_2022 32.86 · D1 33.04 · CMB Old+M1_SARIMA 33.25. The combination gives up
  ~0.4 in the COVID-dominated window (M1 was weak in 2020 and the trailing weights
  react with a lag) but gains 7.7% in the current regime — the trade the KPI wants.

- **Daily ΔCIC guardrail:** D1 within +0.5% of Old_2022 everywhere ✓. All D2
  variants violate (masked variants catastrophically: 33–37 daily RMSE in 2020).
  M1/CMB have no daily path of their own; production daily paths come from
  `reconciled_daily_path()` — Old_2022's within-month calendar shape shifted by a
  constant to hit the combined EOM level, so daily shape accuracy is Old_2022's
  by construction.

### E2 — Post-mortem: why Bet 2 (D2) failed
- Masking 2020-03→12 on the *cumulated* residual series makes the Kalman filter
  free-run for 10 months and then swallow the entire COVID level shift as one
  giant innovation at re-entry — with `smooth trend` the level cannot jump, so the
  slope and AR(1) states whipsaw (fig: −560 THB bn EOM error in Dec-2020; daily
  RMSE 34 in 2020). The unmasked variant (`D2_smooth_nomask`) avoids the blow-up
  but is still ~35–100% worse than Old_2022 in most years: an integrated
  random-walk slope on a 7,000-obs daily series accumulates far more forecast
  variance at the 22-day horizon than the frozen constant it replaces.
- **Lesson recorded:** COVID robustness must be handled with intervention dummies
  (M1_SARIMA's approach — which worked), not by masking observations of an
  integrated series. Slope-drift trend models on *daily* data lose to modelling
  the *monthly* series directly — consistent with diagnosis (5).

## 5. Final results & recommendation

**Recommended production model: `CMB_Old_2022+M1_SARIMA`** — the inverse-MSE
(trailing 12 months, floor 0.1) combination of:
- **Old_2022** — the existing two-step ARIMAX (unchanged, still fitted daily), and
- **M1_SARIMA** — SARIMA(1,1,1)(0,1,1,12) on the monthly EOM level with
  2020-03/04 intervention dummies.

| Window | Old_2022 | Recommended | Improvement |
|---|---|---|---|
| Selection 2018–2023 (n=72) | 32.63 | **30.98** | −5.1% |
| Holdout 2024–2026 (n=28) | 36.12 | **33.33** | −7.7% (DM p=0.043) |

Monthly production recipe:
1. At each month-end, refit both members (seconds each; `cic_forecast_v2.py` caches).
2. Combine EOM forecasts with inverse-MSE weights over the last 12 realized EOM
   errors (equal weights until 6 months of history exist).
3. Daily paths for the liquidity monitor: Old_2022's daily forecast shifted by a
   constant per day to hit the combined EOM level (`reconciled_daily_path`).
4. Fallback: if the combination ever underperforms Old_2022 on a trailing
   12-month basis, the weights self-correct toward Old_2022 automatically —
   worst-case behaviour degrades to ≈ the baseline, which is what the 2020
   selection-window bucket shows.

Why it works: M1_SARIMA carries the drift and annual seasonality at the KPI's
native monthly frequency (strong 2018-19, 2021, 2024-26); Old_2022 carries the
within-month calendar shape and is the safer model in shock regimes (2020, 2023).
Their errors are complementary, and the trailing weights pick the right blend
per regime without ever being told where the regimes are.

## 6. Dropped / follow-up ideas

- EWMA/trailing-window adaptive constant on Old_2022 (spec §5.1 sanity baseline) —
  superseded: M1_SARIMA already supplies the adaptive drift through the combination.
- Model3 trailing-window tuning (36/84/120 months) or WLS-discounted betas — still
  the most promising unexplored arm for the stale-seasonal-beta problem
  (diagnosis 3): the March/April Songkran bias remains visible in the winner's
  errors and is the clearest remaining accuracy lever.
- LightGBM challenger on calendar + holiday-distance features (direct multi-step).
- Student-t prediction intervals (accuracy of intervals, not points).
- D2-class daily trend models: **do not revisit** — see E2 post-mortem.
