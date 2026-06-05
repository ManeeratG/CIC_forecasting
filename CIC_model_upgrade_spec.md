# CIC Forecasting — Model Upgrade Spec (Adaptive Trend, Univariate)

**For:** Claude Code, modifying the existing pipeline (`cic_forecast.py`, `generate_future_exog()`, `cic_forecast_output.xlsx`).
**Author intent:** Fix the one structural defect that makes every current variant cluster together and fail after trend shifts. Do NOT rewrite the pipeline — slot a new model in alongside the existing ones.

---

## 1. Context the implementer needs

- Current model is a two-step ARIMAX: OLS on ~55 calendar dummies → ARMA(1,1) on the residuals, forecasting daily **ΔCIC** (level change).
- The only thing carrying multi-month / multi-year trend information is the **fixed constant `c`**, estimated over the full sample. AR(1)≈0.28 decays to noise within ~2 trading days, so it carries no trend.
- Consequence: when cash demand changes regime (hoarding in 2020, digital-payment erosion from 2021+), the frozen drift is stale and the **end-of-month level** forecast drifts off. This is why Config 2 (2024–25) prefers the *old* model and why ExtDummy / Regime / Fourier all cluster — they refine a mean that was already fine and never touch the trend.
- Adding more fixed dummies, the post-COVID step dummy, and Fourier terms have all been tested and **do not help** (step dummy and Fourier hurt OOS).

## 2. Objective & constraints

- **Target metric (NEW):** 1-month-ahead **end-of-month CIC level** RMSE. Re-estimate at each month origin.
- **Hard constraint:** univariate only — calendar/dummy regressors + time-series structure. **No** exogenous macro / payment / digital-adoption variables.
- **Goal:** replace the fixed drift with an adaptive (state-space) trend that updates as the regime evolves, while reusing the existing calendar-dummy matrix unchanged.

## 3. Core change — Model D (StateSpaceTrend)

Replace the constant `c` with a time-varying drift estimated by a Kalman filter. Two equivalent formulations; **D1 is the low-risk default** because it reuses the existing dummy matrix exactly.

### D1 — change formulation (primary, minimal change)
Model the daily change with a random-walk drift instead of a constant:

```
ΔCIC_t = ν_t + β'·X_t + u_t
ν_t     = ν_{t-1} + ζ_t          # random-walk drift  (ζ_t ~ N(0, σ²_ζ))
u_t     = AR(1) irregular
```

`X_t` = the **existing** calendar dummy set (day-of-month, day-of-week, week-of-month, month, holiday, LWD), minus the constant. `β` estimated jointly (no separate OLS step needed).

> Why this works: a local-level drift on the *change* is mathematically a stochastic slope on the *level* (a local linear trend). The drift `ν_t` will bend down on its own as cash growth slows — no step dummy, no exogenous variable.

statsmodels reference:
```python
from statsmodels.tsa.statespace.structural import UnobservedComponents
mod = UnobservedComponents(
    endog=dCIC,            # daily change series
    exog=X,                # existing dummy matrix WITHOUT the constant column
    level='local level',   # ν_t random walk = adaptive drift
    autoregressive=1,      # AR(1) irregular (most of the persistence; MA(1) was tiny)
)
res = mod.fit(disp=False)
```

### D2 — level formulation (alternative, more flexible)
Model the daily **level** directly with a local linear trend, same dummy matrix as `exog`:
```python
UnobservedComponents(endog=CIC_level, exog=X, level='local linear trend', autoregressive=1)
```
Also test `level='smooth trend'` (level variance fixed to 0, only the slope drifts) — usually the **best for a month-end-level target** because the trend bends slowly instead of wandering. Forecast object is the level directly.

### D3 — irregular upgrade
If Ljung-Box on the winner of {D1, D2} is still significant, bump the irregular to ARMA(1,1) or AR with a seasonal lag-5 term (residual ACF shows weekly structure around lag 5–7).

**Selection:** fit D1, D2 (both trend types), D3. Pick by 1-month-ahead end-of-month level RMSE on the backtest; AIC as tiebreaker. Expect a `smooth trend` variant to win.

## 4. End-of-month level forecast procedure

For each origin = last business day of month M (level `L_M` known):
1. Build month M+1 business-day calendar + dummies via existing `generate_future_exog()`.
2. Kalman-forecast daily ΔCIC for all business days of M+1 (D1) — or forecast the level directly (D2).
3. **Month-end level forecast** = `L_M + Σ ΔCIC_hat` over M+1 (D1), or the model's terminal level (D2).
4. Error = actual end-of-(M+1) level − forecast.

**Re-run the filter at each month origin** — the Kalman filter delivers the *updated* slope automatically. This is the whole point; do not fit once on the full sample and roll forward statically.

## 5. Secondary upgrades that stay within the no-exog rule

1. **Quick sanity-check baseline (do this first, ~30 min):** re-estimate the *existing* ARMA model's constant on an EWMA / trailing-N-month window instead of the full sample. If this alone closes most of the 2024–25 gap, it confirms the trend diagnosis cheaply before the full state-space build.
2. **Forecast combination:** report a simple average of Old_2022 and Model D point forecasts. No single model dominated across windows; the average is usually more robust OOS. One extra column, near-zero cost.
3. **Direct monthly cross-check + reconciliation:** fit a separate **monthly** local-linear-trend UC model on the month-end level series, then reconcile the daily-summed forecast toward it (average the two month-end levels, or MinT if formal). Both univariate; directly targets the KPI.
4. **COVID robustness in estimation:** down-weight 2020 so it doesn't distort variance estimates — short intervention dummies for the extreme 2020 days, or WLS using the existing GARCH σ. Do **not** use the multi-year step dummy (already failed OOS).
5. **Heavy tails / intervals:** fig4 Q-Q shows fat tails. For the prediction bands feeding liquidity ops, calibrate with Student-t rather than Gaussian ±1.96σ.

## 6. Remove / do NOT add
- Post-COVID multi-year step dummy (worsened OOS).
- Fourier terms (redundant vs month dummies, higher AIC).
- Any further fixed holiday dummies — diminishing-to-negative returns demonstrated.

## 7. Evaluation protocol (update existing backtest)

| Item | Spec |
|---|---|
| **Primary metric** | 1-month-ahead end-of-month **level** RMSE |
| Secondary | monthly **change** RMSE (the fig7 monitor number); daily ΔCIC RMSE (continuity with old reports) |
| Windows | keep expanding-window backtest; **must include 2024 and 2025** (where staleness bites). Report 2019 / 2020 / 2021 / Dec21–May22 / 2024 / 2025 |
| Baselines | Old_2022 and ExtDummy, so improvement is attributable |
| Runtime | state-space MLE is seconds per fit; monthly re-estimation over the backtest is a few hundred fits — acceptable, no need to optimize |

**Acceptance criteria:** Model D beats Old_2022 on end-of-month level RMSE in the **2024–25** window (the trend-shift regime) **without** material regression pre-COVID.

## 8. Diagnostics to produce
- **Plot the estimated drift/slope `ν_t` over 1997–2026.** It must show the 2020 hump and post-2021 decline — visual proof the trend is now adaptive (contrast with Old_2022's flat constant). This is the key chart.
- Residual ACF / Ljung-Box and Q-Q vs current fig4.
- End-of-month level: actual vs Old_2022 vs Model D across the full backtest.

## 9. Deliverables
- Add Model D (D1/D2/D3) to `cic_forecast.py` alongside existing models; don't remove the old ones.
- Add the month-end-level metric to the backtest and a new sheet `Level_EOM_Metrics` in `cic_forecast_output.xlsx`.
- New figures: `fig10_trend_slope.png` (ν_t path) and `fig11_eom_level.png` (actual vs Old vs Model D).
