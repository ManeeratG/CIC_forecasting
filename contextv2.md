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
- **Status:** pending Gate 0.
- **Result:** _pending_

## 5. Final results & recommendation

_(to be completed after the holdout run)_

## 6. Dropped / follow-up ideas (if the bets underdeliver)

- EWMA/trailing-window adaptive constant on Old_2022 (spec §5.1 sanity baseline).
- Model3 trailing-window tuning (36/84/120 months) or WLS-discounted betas —
  the direct fix for diagnosis (3) if D2/M1 don't absorb it.
- LightGBM challenger on calendar + holiday-distance features (direct multi-step).
- Student-t prediction intervals (accuracy of intervals, not points).
