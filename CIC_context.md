# CIC Model Improvement Brief

## Background — What the 2022 Model Does

The Bank of Thailand's daily Currency in Circulation (CIC) forecasting model was last formally revised in 2022, updating an original model built in 2013. The motivation was accuracy deterioration after 2020.

**Dependent variable:** Daily level change in CIC (THB billion) — switched from % change in the 2022 revision, justified by stationarity analysis showing level change became more stationary post-2015 while % change degraded.

**Method:** ARMA Maximum Likelihood estimation, upgraded from AR(1) to ARMA(1,1) to suit the level-change dependent variable.

**Regressors (all calendar/dummy variables):**
- Day of month
- Day of week
- Week of month
- Month
- Long holiday dummies: Songkran and New Year, plus any holiday ≥ 4 consecutive days — with separate dummies for 1 day before, 3 consecutive days before, and 3 days after
- Short holiday dummy (before): all other holidays < 4 days
- Last working day of month dummy
- COVID first-wave dummy: 24–27 March 2020 level shift

**Benchmark performance (Dec-2021 → May-2022 hold-out):**

| Model | RMSE | Residual SD |
|---|---|---|
| Old model (pre-2022) | 7.31 | 4.75 |
| 2022 model | 4.96 | 4.14 |

---

## Proposed Improvements — 4 Directions

The goal is to improve forecast accuracy and robustness, particularly post-2020. All improvements should remain **univariate and self-contained** — no exogenous variables. These are directions to explore, not rigid specifications. Real data should drive final model choices.

---

### Direction 1 — Upgrade Seasonal Structure (SARIMA or Fourier)

The current model handles seasonality through fixed calendar dummies with ARMA(1,1). The goal is to let the model capture multi-frequency seasonal patterns more explicitly and flexibly.

Explore whether adding explicit seasonal ARMA orders (SARIMA) improves fit, and/or whether Fourier terms (sine/cosine harmonics for weekly and annual cycles) as regressors in the ARMA framework work better. Daily CIC has at least two seasonal frequencies — weekly (period 7) and annual (period ~365) — and the best approach may differ by frequency. Use information criteria (AIC/BIC) and out-of-sample RMSE to select orders rather than fixing them upfront.

---

### Direction 2 — Add GARCH(1,1) on the Variance Equation

The current model assumes constant variance in the residuals. Evidence from comparable central bank daily CIC models shows that daily CIC changes are heteroscedastic — variance is larger around holidays, payroll dates, and seasonal peaks.

Explore whether ARMA-GARCH(1,1) fits the residuals better than plain ARMA. Test for ARCH effects in the current model's residuals first (Engle's ARCH-LM test) to confirm heteroscedasticity is present before committing to this layer. If confirmed, fit ARMA(p,q)-GARCH(1,1) and assess whether it improves both point forecasts and prediction intervals.

---

### Direction 3 — Improve Trend/Regime Treatment (Structural Break or Stochastic Trend)

The 2022 model handles the COVID break with a single level-shift dummy for 24–27 March 2020. This is too narrow — it treats a multi-year regime change as a 4-day event. Post-2020, the trend growth rate of CIC also shifted (hoarding surge, then digital-payment drag), which no dummy can adequately track.

Explore two approaches and compare them:

- **(a) Formal structural break testing** (e.g. Bai-Perron) to identify break dates empirically, then allow separate intercept/slope regimes rather than hard-coding March 2020.
- **(b) Stochastic local trend component** (local level or local linear trend from the Structural Time Series / unobserved components framework), which lets the trend adapt continuously rather than jumping at a fixed dummy.

Pick or combine whichever fits the post-2020 data better.

---

### Direction 4 — Strengthen Validation Framework

The 2022 model was evaluated on a single 6-month hold-out window (Dec-2021 → May-2022). This is not robust enough to claim the model generalises across regimes.

Build a proper backtesting framework: rolling or expanding-window out-of-sample evaluation across multiple windows, covering both pre-COVID and post-COVID periods. Report RMSE at multiple forecast horizons (at minimum: 1-day, 5-day, 10-day, 22-day ahead). Apply this to all candidate models — old benchmark, 2022 model, and each new variant — so the comparison is fair and regime-robust.
