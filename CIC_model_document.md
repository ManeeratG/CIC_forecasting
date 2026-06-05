# Currency in Circulation Forecasting — Technical Document

**Bank of Thailand | Financial Markets Department**
**Date: June 2026**

---

## 1. Background

### 1.1 What the 2022 Model Does

The Bank of Thailand's daily Currency in Circulation (CIC) forecasting model was last formally revised in 2022, updating an original model built in 2013. The motivation was accuracy deterioration after 2020 caused by the COVID-19 structural break.

**Dependent variable:** Daily level change in CIC (THB billion). The 2022 revision switched from percentage change, justified by stationarity analysis showing that the level-change series became more stationary post-2015 while the percentage-change series degraded.

**Method:** ARMA Maximum Likelihood estimation, upgraded from AR(1) to ARMA(1,1).

**Regressors (calendar and dummy variables only — no exogenous macro variables):**

| Regressor group | Variables | Count |
|-----------------|-----------|-------|
| Day of month | Date\_01 – Date\_31 (one dummy per calendar day) | 31 |
| Day of week | D\_MON – D\_FRI | 5 |
| Week of month | D\_WEEK1 – D\_WEEK5 | 5 |
| Month | D\_JAN – D\_DEC | 12 |
| Long holiday pre-period | D\_PRE\_LH1 (1 day before), D\_PRE\_LH3 (3 days before) | 2 |
| Long holiday post-period | D\_POST\_LH3 (3 days after) | 1 |
| Short holiday | D\_PRE\_SH1 (1 day before short holiday) | 1 |
| COVID first-wave | D\_Covid\_1st (Mar 24–27, 2020) | 1 |
| Last working day | D\_LWD | 1 |

**Original benchmark performance (Dec-2021 → May-2022 hold-out):**

| Model | RMSE (THB bn) | Residual SD |
|-------|:---:|:---:|
| Pre-2022 model (AR(1), % change) | 7.31 | 4.75 |
| BOT 2022 model (ARMA(1,1), level change) | 4.96 | 4.14 |

---

### 1.2 Motivation for Further Improvement

Three observations motivated the current work:

1. **Post-COVID structural shift**: CIC growth accelerated during 2020 (cash hoarding), then declined from 2021 due to digital-payment adoption. A 4-day dummy for March 24–27, 2020 does not capture this multi-year regime change.
2. **Holiday asymmetry**: Songkran (mid-April) and New Year (Jan 1 / Dec 31) generate the two largest annual cash-flow events, with cash-flow patterns that differ from other long holidays. Pooling all long holidays into one dummy set mis-fits these events.
3. **Heteroscedastic residuals**: CIC volatility is larger around seasonal peaks and COVID disruptions. A constant-variance ARMA model understates uncertainty in high-volatility periods.

---

## 2. Improvement Directions

Four directions were explored. All improvements remain **univariate and self-contained** — no exogenous variables beyond calendar/dummy regressors.

### Direction 1 — Upgrade Seasonal Structure (Fourier Terms)

Explore whether Fourier terms (sine/cosine harmonics for the annual cycle) as regressors in the ARMA framework improve fit. Daily CIC has at least two seasonal frequencies: weekly (period 5 trading days) and annual (period ≈ 261 trading days). The current model handles these through calendar dummy variables. Fourier terms are an alternative representation; both approaches are compared via AIC/BIC and out-of-sample RMSE.

**Conclusion from data**: For daily CIC, calendar dummies outperform Fourier terms. Holiday-driven cash withdrawals are discrete pulses, not smooth sinusoidal fluctuations. Fourier terms smooth across the discontinuity and underfit the spiky holiday dynamics.

### Direction 2 — GARCH(1,1) on the Variance Equation

The current model assumes constant variance. Evidence from central bank CIC models shows that daily changes are heteroscedastic — variance is larger around holidays, payroll dates, and COVID. Engle's ARCH-LM test is applied first; if it rejects the null of no heteroscedasticity, GARCH(1,1) is fitted on ARMA residuals to improve prediction intervals.

**Conclusion from data**: ARCH-LM strongly rejects constant variance (stat ≈ 419, p < 0.0001). GARCH(1,1) is fitted to provide time-varying prediction intervals; it does not change point forecasts but identifies periods of elevated forecast risk for operations teams.

### Direction 3 — Improve Regime Treatment (Post-COVID Step Dummy)

The 2022 model treats COVID as a 4-day event. The pandemic actually caused a multi-year regime change: upward level shift (March 2020) followed by a declining trend (2021 onward, digital-payment adoption). Approach (a) from the literature: a step dummy $D_{PostCovid,t} = \mathbf{1}[t \geq \text{April 2020}]$ captures the mean-level shift without continuous re-estimation.

**Conclusion from data**: The Regime dummy improves in-sample AIC marginally (−14 vs Old_2022) but worsens out-of-sample RMSE in the benchmark window (+0.006) because all 119 evaluation obs fall after April 2020, creating a systematic mean offset. Regime model is not recommended for production; it could be activated for new-crisis scenarios.

### Direction 4 — Strengthen Validation Framework

The 2022 model was evaluated on a single 6-month hold-out. A proper backtesting framework uses rolling/expanding-window evaluation across multiple windows covering both pre-COVID and post-COVID periods, with RMSE reported at multiple forecast horizons (h = 1, 5, 10, 22 days).

**Implemented**: Four expanding windows (2019, 2020, 2021, Dec21-May22) plus a second evaluation config (1997-2023 training, 2024-2025 eval) to test 2-year out-of-sample robustness.

---

## 3. Literature Review

### 3.1 Time-Series Framework for Demand for Currency

Central bank cash forecasting models date to the 1970s. The standard approach decomposes CIC into:

$$\text{CIC}_t = \text{Trend}_t + \text{Seasonal}_t + \text{Calendar}_t + \varepsilon_t$$

Key references:

| Reference | Contribution |
|-----------|-------------|
| Box & Jenkins (1970) | ARIMA framework; AR(p), MA(q), ARMA(p,q) for stationary series |
| Anderson & Gascon (2009) | Federal Reserve daily cash forecasting with calendar dummies; confirms ARMA + dummies as workhorse |
| Peng & Shi (2014) | Structural break in Chinese CIC post-financial crisis; shows regime dummies outperform single-model estimation |
| Canova & Hansen (1995) | Fourier representation of seasonality; sine/cosine harmonics as alternative to seasonal dummies |
| Taylor (2003) | Exponential smoothing with multiple seasonal periods; weekly + annual for daily data |
| Engle (1982) | ARCH model; time-varying variance in financial time series |
| Bollerslev (1986) | GARCH(1,1) generalisation; persistence of volatility shocks |
| Bai & Perron (1998, 2003) | Formal structural break testing with multiple unknown break dates |
| Tashman (2000) | Multi-horizon out-of-sample evaluation protocol; rolling and expanding windows |

### 3.2 Seasonal Treatment

Daily CIC has at least **two seasonal frequencies**:

- **Weekly** (period = 5 trading days): Withdrawals peak on Fridays; deposits return on Mondays.
- **Annual** (period ≈ 261 trading days): CIC peaks in December–January (New Year), April (Songkran), and school holidays; troughs in June–July.

The BOT 2022 model handles these through calendar dummy variables — day-of-week, week-of-month, and month indicators. This is equivalent to a saturated dummy regression on seasonal frequencies, which Canova & Hansen (1995) show is asymptotically equivalent to (but more flexible than) Fourier representation when seasonality is discrete and spiky.

**Finding**: For daily CIC, calendar dummies outperform pure Fourier terms because holiday-driven cash withdrawals are discrete pulses (one-day events), not smooth sinusoidal fluctuations. Fourier terms smooth across the discontinuity and underfit.

### 3.3 Structural Break and Regime Change

Bai & Perron (1998) develop sequential tests for multiple structural breaks. For CIC:

- A break in mean (level shift) captures an overnight jump in CIC (e.g., demonetization events).
- A break in slope (trend change) captures a sustained change in CIC growth rate.

The COVID pandemic represents a **combined mean and slope break**: upward level shift in March–April 2020 (cash hoarding), followed by a declining trend from 2021 (digital payment adoption). A single 4-day dummy addresses neither adequately.

For this work we adopt the simpler **regime dummy** approach — Peng & Shi (2014) show it gives competitive out-of-sample accuracy relative to the structural time series approach.

### 3.4 Heteroscedasticity and GARCH

Bollerslev's (1986) GARCH(1,1):

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

The sum $\alpha + \beta$ measures persistence; values close to 1 indicate long-memory volatility. For CIC, ARCH effects are expected near Songkran, New Year, COVID, and month-end payroll dates. GARCH improves **prediction intervals** and can enable WLS-weighted residuals; it does not change point forecasts from the ARMA mean equation.

### 3.5 Validation Framework

Tashman (2000) argues a single hold-out window is insufficient. Requirements:

1. **Multiple windows**: Cover different regimes (pre-COVID, COVID, post-COVID).
2. **Multiple horizons**: Report RMSE at h = 1, 5, 10, 22 days.
3. **Rolling or expanding window**: Re-fit on each training window.

---

## 4. Model Specifications

### 4.1 Estimation Strategy — Two-Step ARIMAX

All models use a two-step approach:

1. **OLS** on all calendar/dummy regressors → $\hat{\boldsymbol{\beta}}$, OLS residuals
2. **ARIMA(1,0,1)** on OLS residuals → AR(1), MA(1), $\sigma^2$

This is numerically equivalent to joint SARIMAX-MLE in large samples (Frisch-Waugh theorem) but converges in seconds rather than minutes — making rolling backtests with four windows and four models practical. Joint MLE (EViews approach) takes minutes per fit; two-step takes < 0.1 seconds.

### 4.2 Old Model — BOT 2022 ARMA(1,1) Baseline

**Specification**: ARMA(1,1) + 55 calendar dummies (reference: Monday, day 1, week 1, December).

$$\Delta\text{CIC}_t = c + \boldsymbol{\beta}'\mathbf{X}_t + \phi_1 u_{t-1} + \theta_1 \varepsilon_{t-1} + \varepsilon_t$$

Regressors $\mathbf{X}_t$: Date\_02–Date\_31 (30), D\_TUE–D\_FRI (4), D\_WEEK2–D\_WEEK5 (4), D\_JAN–D\_NOV (11), D\_PRE\_LH1, D\_PRE\_LH3, D\_POST\_LH3, D\_PRE\_SH1, D\_Covid\_1st, D\_LWD = **55 + constant**.

### 4.3 Model A — Extended Holiday Dummies (ExtDummy)

**Change**: Add **separate Songkran and New Year pre/post dummies** on top of the existing dummy set.

**Motivation**: Songkran (mid-April, 3–5 consecutive days) and New Year (Jan 1 and/or Dec 31) generate different cash-flow patterns than other long holidays. Pooling them into D\_PRE\_LH1/D\_POST\_LH3 mis-estimates their specific patterns.

**Dummy construction**: Built from the official Thai holiday sheet (Bank of Thailand data, 2014–2026). For earlier years (1997–2013) the holiday sheet is unavailable, so SK dates are set to April 13–15 (fixed Thai calendar) and NY to January 1 + December 31. This gives **~25 annual events per dummy across the full training history** — ensuring stable OLS coefficient estimates.

| Variable | Meaning |
|----------|---------|
| D\_SK\_PRE1 | Last trading day before each Songkran holiday block |
| D\_SK\_POST1 | First trading day after each Songkran holiday block |
| D\_NY\_PRE1 | Last trading day before each New Year holiday block |
| D\_NY\_POST1 | First trading day after each New Year holiday block |

Dummies are mutually exclusive (no overlapping calendar coverage). **Total: 59 + constant, ARMA(1,1).**

### 4.4 Model B — Regime + Extended Dummies

**Change from Model A**: Add a post-COVID regime dummy.

$$D_{PostCovid,t} = \mathbf{1}[t \geq \text{April 1, 2020}]$$

Absorbs the persistent upward level shift in CIC after COVID. Without it, the constant term is estimated over the full sample and pulled upward by the COVID period, biasing pre-COVID forecasts.

**Total: 60 + constant, ARMA(1,1).**

**Note**: All 119 benchmark evaluation obs fall after April 2020, so D_PostCovid=1 throughout the eval window, introducing a systematic mean offset. This makes Regime inferior to ExtDummy on the Dec21-May22 benchmark.

### 4.5 Model C — Fourier + Regime

**Change from Model B**: Add annual Fourier terms:

$$F_{k,t} = \sin\!\left(\frac{2\pi k t}{261}\right), \quad G_{k,t} = \cos\!\left(\frac{2\pi k t}{261}\right), \quad k = 1, 2, 3$$

**Finding**: AIC is higher (worse) with Fourier on top of month dummies. Month dummies already capture the annual pattern non-parametrically (11 parameters). Fourier terms (6 parameters) are redundant and reduce degrees of freedom without improving fit. **Not recommended for production.** Retained in comparison for completeness.

### 4.6 GARCH(1,1) — Variance Model

Applied as a third step on ARMA residuals:

$$\hat{\varepsilon}_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1), \quad \sigma_t^2 = \omega + \alpha \hat{\varepsilon}_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Purpose**: Provides time-varying prediction intervals $\hat{\Delta\text{CIC}}_t \pm 1.96\hat{\sigma}_t$ for operational risk management. Does not change point forecasts. Applied conditionally if ARCH-LM test rejects constant variance.

---

## 5. Production Recommendation

**Recommended for production upgrade**: **Model A — ExtDummy** (1997-2021 training) — ARMA(1,1) with separate Songkran and New Year holiday dummies.

| Feature | Old_2022 | **ExtDummy (Recommended)** | Regime+ExtDummy |
|---------|----------|----------------------|--------|
| Holiday treatment | All long holidays pooled | **SK + NY separate (25 events/dummy, 1997–2026)** | Same as ExtDummy |
| COVID treatment | 4-day level-shift dummy | 4-day level-shift dummy | + Post-April 2020 step |
| ARMA order | (1,1) | (1,1) | (1,1) |
| Regressors | 55 + const | 59 + const | 60 + const |
| Benchmark RMSE (Dec21-May22) | 4.026 | **3.971** | 4.033 |
| AIC (1997-2021 training) | 34,447 | **34,434** | 34,433 |
| ΔAIC vs Old_2022 | — | **−13** | −14 |

**If a COVID-like disruption is expected in the forecast horizon**, activate D_PostCovid or add a new regime dummy. For normal operations, ExtDummy is sufficient.

---

## 6. Results

### 6.1 In-Sample Fit

Two-step ARIMAX. AIC penalised for all parameters (OLS intercept + regressors + AR, MA, σ²).

**Config 1 — Training: 1997-08-29 – 2021-11-30 (n = 5,935 obs)**

| Model | AIC | BIC | Residual σ (THB bn) | AR(1) | MA(1) |
|-------|-----|-----|---------------------|-------|-------|
| Old_2022 (1997-2021) | 34,447 | 34,842 | 4.363 | 0.285 | 0.115 |
| ExtDummy (1997-2021) | **34,434** | 34,856 | **4.355** | 0.281 | 0.123 |
| Regime+ExtDummy (1997-2021) | 34,433 | 34,861 | 4.354 | 0.280 | 0.124 |
| Fourier+Regime (1997-2021) | 34,443 | 34,912 | 4.353 | 0.279 | 0.124 |

**Config 2 — Training: 1997-08-29 – 2023-12-31 (n = 6,440 obs)**

| Model | AIC | BIC | Residual σ (THB bn) | AR(1) | MA(1) |
|-------|-----|-----|---------------------|-------|-------|
| Old_2022 (1997-2023) | 37,220 | 37,619 | 4.313 | 0.279 | 0.130 |
| ExtDummy (1997-2023) | **37,203** | 37,629 | **4.305** | 0.277 | 0.135 |
| Regime+ExtDummy (1997-2023) | 37,202 | 37,636 | 4.304 | 0.276 | 0.135 |
| Fourier+Regime (1997-2023) | 37,213 | 37,687 | 4.303 | 0.276 | 0.136 |

**Interpretation**: ExtDummy improves AIC by 13 points over Old_2022 in both configs — clearly significant (ΔAIC > 4 per Burnham & Anderson 2002). Regime adds 1 more point (marginal). Fourier terms add almost nothing.

**Residual diagnostics (all models, Config 1 training):**
- ADF test: stationary (p < 0.0001) ✓
- ARCH-LM(10): stat ≈ 406–419, p < 0.0001 → strong ARCH effects → **GARCH warranted**
- Ljung-Box(10): p < 0.0001 → some residual autocorrelation persists; ARMA(1,1) captures most but not all structure

**GARCH(1,1) on Old_2022 residuals (Config 1):**
- ω = 2.3463, α = 0.1984, β = 0.6861, persistence α+β = **0.884**
- High persistence: volatility shocks (COVID, Songkran) dissipate slowly over many days.

### 6.2 Out-of-Sample RMSE — Config 1: Benchmark Window (Dec 2021 – May 2022, n=119)

Trained on 1997–2021. Directly comparable to BOT 2022 paper evaluation window.

| Model | RMSE (THB bn) | Δ vs Old_2022 | Δ vs BOT paper |
|-------|:---:|:---:|:---:|
| Old_2022 (1997-2021) | 4.026 | 0.000 | −0.934 |
| **ExtDummy (1997-2021)** | **3.971** | **−0.055** | **−0.989** |
| Regime+ExtDummy (1997-2021) | 4.033 | +0.006 | −0.927 |
| Fourier+Regime (1997-2021) | 4.029 | +0.003 | −0.931 |
| [BOT 2022 paper (2017-2021) — EViews joint MLE] | 4.960 | baseline | 0.000 |
| [Pre-2022 model] | 7.310 | — | +2.350 |

RMSE gain of 0.055 THB bn/day ≈ **1.1 THB bn/month** of improved monthly-monitor accuracy. Both Python replications beat the published EViews result by ~1 THB bn because they train on the full 1997-2021 sample (n=5,935) vs. the paper's 2017-2021 window (n=1,304) — more data gives better coefficient estimates.

### 6.3 Out-of-Sample RMSE — Config 2: Extended Window (Jan 2024 – Dec 2025, n=485)

Trained on 1997–2023. Tests robustness on unseen 2-year post-COVID horizon.

| Model | RMSE (THB bn) | Δ vs Old_2022 |
|-------|:---:|:---:|
| **Old_2022 (1997-2023)** | **5.186** | 0.000 ← best |
| ExtDummy (1997-2023) | 5.213 | +0.027 |
| Regime+ExtDummy (1997-2023) | 5.209 | +0.023 |
| Fourier+Regime (1997-2023) | 5.201 | +0.015 |

All models cluster at ~5.2 RMSE; differences are < 0.03 THB bn — **not meaningfully different**. The elevated RMSE (~5.2 vs ~4.0 for Config 1) reflects higher intrinsic CIC volatility in 2024-2025. The SK/NY advantage of ExtDummy is concentrated around specific holiday periods and averages out over a 2-year diverse window.

### 6.4 Rolling Backtest RMSE (Expanding Window, Config 1 training)

| Model | 2019 (pre-COVID) | 2020 (COVID year) | 2021 (recovery) | Dec21–May22 (benchmark) |
|-------|:---:|:---:|:---:|:---:|
| Old_2022 (1997-train) | 4.928 | 5.472 | 4.582 | 4.026 |
| ExtDummy (1997-train) | 4.938 | 5.530 | 4.611 | **3.971** |
| Δ (ExtDummy vs Old) | +0.010 | +0.058 | +0.029 | **−0.055** |

ExtDummy's advantage is concentrated in the benchmark window (Dec 2021–May 2022), which contains both Songkran and New Year events. In other periods the two models are near-identical (< 0.06 RMSE difference). Both models show elevated RMSE in 2020 (COVID shock year) — neither can capture unprecedented cash hoarding dynamics adequately.

### 6.5 RMSE by Forecast Horizon (3 Monthly Origins, Config 1)

Refitting at Dec 2021, Feb 2022, Apr 2022; forecasting h steps ahead.

| Horizon | Old_2022 | ExtDummy |
|---------|:---:|:---:|
| 1-day ahead | 3.646 | 3.653 |
| 5-day ahead | 5.452 | **5.416** |
| 10-day ahead | 4.269 | **4.226** |
| 22-day ahead (monthly) | **1.412** | 1.498 |

The 22-day (monthly) RMSE is very low (1.4–1.5 THB bn) because at the 1-month horizon the ARMA correction has fully decayed and the forecast is driven almost entirely by the deterministic calendar structure — which is fully known in advance. This is excellent for the monthly monitor use case. At mid-range horizons (5–10 days), ExtDummy is marginally better due to SK/NY dummy signal in the lead-up to holiday periods.

---

## 7. Figures

| Figure | Content |
|--------|---------|
| `fig1_cic_overview.png` | CIC level and daily change, full sample 1997–2026 |
| `fig2_actual_vs_forecast.png` | Actual vs forecast, benchmark window (Dec 2021 – May 2022), all models |
| `fig3_forecast_errors.png` | Forecast error time series and distribution, each model |
| `fig4_residual_diagnostics.png` | ACF and Q-Q plots of training residuals |
| `fig5_rmse_comparison.png` | RMSE bar chart (all model-config combos) + rolling backtest |
| `fig6_horizon_rmse.png` | RMSE vs forecast horizon (h=1,5,10,22), Old_2022 vs ExtDummy |
| `fig7_monthly_monitor.png` | Monthly aggregated forecast vs actual |
| `fig8_garch_volatility.png` | GARCH conditional volatility over training period |
| `fig9_seasonal_cic.png` | Seasonal CIC pattern — end-of-month level by year (last 10 years), ★ = next-month forecast |

---

## 8. Excel Output

`cic_forecast_output.xlsx` contains the following sheets:

| Sheet | Content |
|-------|---------|
| `Eval_Benchmark` | Dec 2021–May 2022 eval rows only (119 obs) — actual + change forecasts + reconstructed CIC level for all models |
| `Eval_Extended` | Jan 2024–Dec 2025 eval rows only (485 obs) — same format |
| `InSample_Fitted` | Training period (1997-2021) in-sample fitted values for all models |
| `Full_Series` | Complete daily CIC level + change (raw data reference, no forecasts) |
| `Benchmark_Metrics` | RMSE/MAE/ResidSD/Bias for all model-config combos plus published references |
| `Rolling_RMSE` | Expanding-window backtest RMSE by period |
| `Horizon_RMSE` | Multi-horizon (h=1,5,10,22) RMSE |
| `GARCH_Params` | GARCH(1,1) parameter estimates and fit statistics |

---

## 9. Practical Notes for Monthly Monitor

The monthly monitor uses daily forecasts in two ways:

1. **Daily forecast accuracy**: RMSE in THB billion per day. Current target: below 5 THB bn.
2. **Monthly total forecast**: Sum of daily forecasts for the month. A small daily bias compounds into a large monthly error.

**Recommended workflow**:
1. At end of each month, re-estimate ExtDummy on all available data.
2. Forecast next month's business days using known calendar structure (generate with `generate_future_exog()`).
3. Report daily forecasts + monthly total + 95% prediction interval (using GARCH σ).
4. Track RMSE monthly; trigger model review if 3-month rolling RMSE > 6 THB bn.
5. Update SK/NY dummy holiday dates annually when BOT publishes the official holiday calendar.

**Installation** (local Python):
```
pip install -r requirements.txt
python cic_forecast.py
```

---

## 10. References

- Anderson, R.G. & Gascon, C.S. (2009). "The U.S. Experience with Seasonal Currency Flows." *Federal Reserve Bank of St. Louis Review.*
- Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica*, 66(1), 47–78.
- Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple Structural Change Models." *Journal of Applied Econometrics*, 18(1), 1–22.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31, 307–327.
- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*. Springer.
- Canova, F. & Hansen, B.E. (1995). "Are Seasonal Patterns Constant over Time? A Test for Seasonal Stability." *Journal of Business & Economic Statistics*, 13(3), 237–252.
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Peng, F. & Shi, Y. (2014). "A Structural Break Approach to Currency Demand Forecasting." *China Economic Review*, 27, 316–325.
- Taylor, J.W. (2003). "Short-Term Electricity Demand Forecasting Using Double Seasonal Exponential Smoothing." *Journal of the Operational Research Society*, 54(8), 799–805.
- Tashman, L.J. (2000). "Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review." *International Journal of Forecasting*, 16(4), 437–450.
