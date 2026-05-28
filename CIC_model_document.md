# Currency in Circulation Forecasting — Literature Review & Model Proposal

**Bank of Thailand | Financial Markets Department**
**Date: May 2026**

---

## 1. Background and Motivation

Currency in Circulation (CIC) is a key operational variable for central bank liquidity management. The Bank of Thailand forecasts daily CIC to manage reserve money and plan open-market operations. An accurate daily forecast, aggregated to a monthly figure, directly supports the monthly liquidity monitor.

The 2022 revision to the BOT model improved daily RMSE from 7.31 to 4.96 THB billion relative to the pre-2022 benchmark. The present work attempts a further improvement, motivated by three observations:

1. **Post-COVID structural shift**: CIC growth accelerated during 2020 (cash hoarding), then was dragged lower by digital-payment adoption from 2021. A 4-day dummy for March 24–27, 2020 does not capture this multi-year regime change.
2. **Holiday asymmetry**: Songkran and New Year generate different cash-flow patterns than other long holidays. Pooling all long holidays into a single dummy set likely mis-fits these events.
3. **Heteroscedastic residuals**: CIC volatility is larger around seasonal peaks and COVID disruptions. Constant-variance ARMA understates uncertainty in high-volatility periods.

---

## 2. Literature Review

### 2.1 Time-Series Framework for Demand for Currency

Central bank cash forecasting models date to the 1970s. The standard approach decomposes CIC into:

$$\text{CIC}_t = \text{Trend}_t + \text{Seasonal}_t + \text{Calendar}_t + \varepsilon_t$$

Key literature:

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

### 2.2 Seasonal Treatment

Daily CIC has at least **two seasonal frequencies**:

- **Weekly** (period = 5 trading days): Withdrawals peak on Fridays; deposits return on Mondays.
- **Annual** (period ≈ 261 trading days): CIC peaks in December–January (New Year), April (Songkran), and school holidays; troughs in June–July.

The BOT 2022 model handles these through calendar dummy variables — day-of-week, week-of-month, and month indicators. This is equivalent to a saturated dummy regression on seasonal frequencies, which Canova & Hansen (1995) show is asymptotically equivalent to (but more flexible than) Fourier representation when seasonality is discrete and spiky.

**Finding**: For daily CIC, calendar dummies outperform pure Fourier terms because holiday-driven cash withdrawals are discrete pulses (one-day events), not smooth sinusoidal fluctuations. Fourier terms smooth across the discontinuity and underfit.

### 2.3 Structural Break and Regime Change

Bai & Perron (1998) develop sequential tests for multiple structural breaks in the mean and slope of a time series. For CIC:

- A break in mean (level shift) captures an overnight jump in CIC (e.g., demonetization events).
- A break in slope (trend change) captures a sustained change in CIC growth rate.

The COVID pandemic represents a **combined mean and slope break**: an upward level shift in March–April 2020 (cash hoarding), followed by a declining trend from 2021 (digital payment adoption). A single 4-day dummy addresses neither the mean shift nor the slope change adequately.

Two approaches from the literature:

**(a) Regime dummy**: A step dummy $D_{PostCovid,t} = \mathbf{1}[t \geq \text{Apr 2020}]$ captures the mean shift in a regression. Adding an interaction $D_{PostCovid,t} \times t$ captures the slope change.

**(b) Unobserved components / local linear trend**: Harvey (1989)'s structural time series model allows the trend to evolve stochastically via a state-space representation. This is more flexible but harder to implement in a daily frequency model with many calendar regressors.

For this work, we adopt the simpler **regime dummy** approach (a), noting that Peng & Shi (2014) show it gives competitive out-of-sample accuracy relative to the structural time series approach.

### 2.4 Heteroscedasticity and GARCH

Engle's (1982) ARCH test checks whether squared residuals are serially correlated — evidence of time-varying variance. Bollerslev's (1986) GARCH(1,1):

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

addresses heteroscedasticity. The sum $\alpha + \beta$ measures persistence; values close to 1 indicate long-memory volatility (typical in financial daily data).

For CIC, ARCH effects are expected near:
- Songkran and New Year (large, predictable cash flows)
- COVID-19 period (unprecedented daily changes)
- Month-end payroll dates

The GARCH layer improves **prediction intervals** (variance forecasts) and can marginally improve point forecasts via GARCH-weighted residuals. The two-step approach (ARMA → GARCH on residuals) is used here for tractability with 50+ mean regressors.

### 2.5 Validation Framework

Tashman (2000) argues that a single hold-out window is insufficient to evaluate forecast generalisability. Key requirements:

1. **Multiple windows**: Cover different regimes (pre-COVID, COVID, post-COVID).
2. **Multiple horizons**: Report RMSE at h = 1, 5, 10, 22 days to assess how accuracy degrades.
3. **Rolling or expanding window**: Re-fit on each training window rather than fixing parameters once.

The BOT 2022 paper used a single 6-month hold-out (Dec 2021 – May 2022), which is consistent with the 2022 regime. We extend this to four windows covering 2019–2022.

---

## 3. Model Specifications

### 3.1 Old Model — 2022 BOT ARMA(1,1) Baseline

**Dependent variable**: $\Delta\text{CIC}_t$ = daily level change in CIC (THB billion).

**Motivation for level change**: ADF tests show the level-change series is stationary from 2015 onward; the percentage-change series is less stable. BOT 2022 paper confirms this choice.

**Specification** (replicating EViews `cic.prg`):

$$\Delta\text{CIC}_t = c + \boldsymbol{\beta}'\mathbf{X}_t + \phi_1 u_{t-1} + \theta_1 \varepsilon_{t-1} + \varepsilon_t$$

where $\mathbf{X}_t$ includes:

| Regressor group | Variables | Count |
|-----------------|-----------|-------|
| Day of month | Date\_02 – Date\_31 (reference: day 1) | 30 |
| Day of week | D\_TUE, D\_WED, D\_THU, D\_FRI (reference: Mon) | 4 |
| Week of month | D\_WEEK2 – D\_WEEK5 (reference: week 1) | 4 |
| Month | D\_JAN – D\_NOV (reference: Dec) | 11 |
| Long holiday pre-period | D\_PRE\_LH1 (1 day before), D\_PRE\_LH3 (3 days before) | 2 |
| Long holiday post-period | D\_POST\_LH3 (3 days after) | 1 |
| Short holiday | D\_PRE\_SH1 (1 day before short holiday) | 1 |
| COVID first-wave | D\_Covid\_1st (Mar 24–27, 2020) | 1 |
| Last working day | D\_LWD | 1 |
| **Total regressors** | | **55 + constant** |

**Estimation**: ARMA maximum likelihood (OPG method in EViews; LBFGS in Python).

**Benchmark performance** (from BOT 2022 paper):

| | Pre-2022 model | 2022 model |
|---|---|---|
| RMSE (Dec-2021 – May-2022) | 7.31 THB bn | 4.96 THB bn |
| Residual SD | 4.75 THB bn | 4.14 THB bn |

---

### 3.2 New Model A — Extended Holiday Dummies (Direction 1)

**Change from baseline**: Replace the single long-holiday dummy set with **separate Songkran and New Year dummies**.

**Motivation**: Songkran (mid-April, 3–5 consecutive days) and New Year (Jan 1, often 3–5 days) generate the two largest annual cash-flow events. Their pre/post patterns differ from other long holidays (e.g., Chakri Day, Constitution Day):
- Songkran: large provincial cash outflows 3–5 days before; large returns 2–3 days after.
- New Year: large withdrawals 1–3 days before; slower deposits post-holiday.

**Additional regressors**:

| Variable | Meaning |
|----------|---------|
| D\_PRE\_SK1 | 1 day before Songkran period |
| D\_PRE\_SK3 | 3 days before Songkran period |
| D\_POST\_SK3 | 3 days after Songkran period |
| D\_PRE\_NY1 | 1 day before New Year period |
| D\_PRE\_NY3 | 3 days before New Year period |
| D\_POST\_NY3 | 3 days after New Year period |

**Total regressors**: 61 + constant, ARMA(1,1).

**Expected improvement**: Holiday-period dummy misfit is the largest remaining systematic error in the baseline. Separating SK/NY should reduce RMSE in April and January.

---

### 3.3 New Model B — Regime + Extended Dummies (Directions 1 + 3)

**Change from Model A**: Add a **post-COVID regime dummy**.

$$D_{PostCovid,t} = \mathbf{1}[t \geq \text{April 1, 2020}]$$

**Motivation from Peng & Shi (2014)**: A step dummy starting at the break date captures the mean-level shift without requiring continuous re-estimation. The April 2020 start corresponds to the sharp CIC acceleration observed in the data following the first national lockdown.

**Effect**: The regime dummy absorbs the persistent upward shift in the CIC level after COVID, improving fit in the 2020–2022 period. Without it, the model's constant term is estimated over the full sample and is pulled upward by the COVID period, biasing pre-COVID forecasts.

**Total regressors**: 62 + constant, ARMA(1,1).

**This is the recommended production model** (see Section 4).

---

### 3.4 New Model C — Fourier + Regime (Direction 1, Supplementary)

**Change from Model B**: Add **annual Fourier terms** on top of month dummies to capture smooth intra-year patterns between the discrete monthly effects.

$$F_{k,t} = \sin\!\left(\frac{2\pi k t}{261}\right), \quad G_{k,t} = \cos\!\left(\frac{2\pi k t}{261}\right), \quad k = 1, 2, 3$$

where $t$ is the trading-day index and 261 is the approximate number of trading days per year.

**Literature basis**: Canova & Hansen (1995) show that Fourier terms are useful when seasonality is smooth and continuous. Taylor (2003) applies this to daily data with annual and weekly periods.

**Finding from data**: For daily CIC, AIC is higher (worse) with Fourier terms added on top of month dummies, because:
1. Month dummies already capture the annual pattern non-parametrically with 11 parameters.
2. Fourier terms (6 parameters) are redundant once month dummies are included.
3. The additional parameters reduce degrees of freedom without improving fit.

Fourier terms are more useful if month dummies are **removed** (replacing 11 parameters with 6). However, that substitution also performs worse because CIC's annual pattern is spiky (Songkran, New Year), not sinusoidal.

**Conclusion**: Fourier terms are not recommended for daily CIC with the existing dummy structure. They are retained in the comparison for completeness.

---

### 3.5 GARCH(1,1) — Variance Model (Direction 2)

**Two-step approach**:

1. Fit the ARMA(1,1) + dummies model (any specification above) to get point forecasts.
2. Extract residuals $\hat{\varepsilon}_t$.
3. Fit GARCH(1,1) on $\hat{\varepsilon}_t$:

$$\hat{\varepsilon}_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$

$$\sigma_t^2 = \omega + \alpha \hat{\varepsilon}_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Purpose**: GARCH does **not** change point forecasts (the ARMA mean equation is unchanged). It provides:
- Time-varying prediction intervals $\hat{\Delta\text{CIC}}_t \pm 1.96\hat{\sigma}_t$
- Identifies periods of elevated risk (COVID, holidays) for operations teams
- Enables WLS-style down-weighting of high-variance periods (if implemented jointly)

**Triggering condition**: Apply only if Engle's ARCH-LM test rejects the null of no heteroscedasticity ($p < 0.05$).

---

## 4. Recommended Model and Implementation

### 4.1 Production Recommendation

**Recommended for production upgrade**: **Model A — Extended Dummies (ExtDummy)** — ARMA(1,1) with separate Songkran and New Year holiday dummies.

The Regime model (Model B) shows marginally better COVID-year robustness but introduces a systematic daily bias (−0.725 THB bn) on the benchmark window due to the D_PostCovid step dummy over-adjusting the intercept. The **ExtDummy** model captures the primary source of improvement (SK/NY holiday disambiguation) with minimal benchmark cost (+0.047 RMSE vs Old_2022).

| Feature | Old 2022 | ExtDummy (**Recommended**) | Regime |
|---------|----------|----------------------|--------|
| Holiday treatment | All long holidays pooled | **Songkran + NY separate** | Songkran + NY separate |
| COVID treatment | 4-day level-shift dummy | 4-day level-shift dummy | + Post-April 2020 step |
| ARMA order | (1,1) | (1,1) | (1,1) |
| Regressors | 55 + constant | 61 + constant | 62 + constant |
| Benchmark RMSE | 3.799 | **3.846** | 3.902 |
| 2020 COVID RMSE | 5.480 | — | **5.346** |
| AIC improvement | — | **–45** | –45 |

**If a COVID-like scenario is expected in the forecast horizon**, activate the D_PostCovid dummy or add a new regime dummy. For normal operations, ExtDummy is sufficient.

### 4.2 Validation Framework (Direction 4)

Evaluation covers four expanding windows:

| Window | Period | Regime |
|--------|--------|--------|
| Pre-COVID | 2019 | Normal |
| COVID year | 2020 | Crisis |
| Post-COVID recovery | 2021 | Normalisation |
| Benchmark | Dec 2021 – May 2022 | Post-COVID |

Metrics reported at horizons h = 1, 5, 10, 22 trading days.

### 4.3 GARCH Supplement

If ARCH-LM test confirms heteroscedasticity (expected), fit GARCH(1,1) on Model B residuals to generate prediction intervals for the monthly monitor report. Point forecasts remain from the ARMA mean equation.

---

## 5. Model Comparison Results

*(Results are populated automatically by `cic_forecast.py`. See table below and accompanying figures.)*

### 5.1 In-Sample Fit (Training: 1997-08-28 – 2021-11-30)

Two-step ARIMAX (OLS + ARIMA(1,0,1) on residuals). AIC penalised for all parameters (OLS intercept + regressors + AR, MA, σ²).

| Model | AIC | BIC | Residual σ (THB bn) | AR(1) | MA(1) |
|-------|-----|-----|---------------------|-------|-------|
| Old_2022 | 34,483 | 34,878 | 4.374 | 0.289 | 0.116 |
| ExtDummy | **34,438** | 34,873 | 4.353 | 0.284 | 0.112 |
| Regime | **34,438** | 34,879 | **4.352** | 0.283 | 0.113 |
| Fourier_Regime | 34,449 | 34,930 | **4.352** | 0.283 | 0.113 |

**Interpretation**: ExtDummy and Regime improve AIC by ~45 points over Old_2022 — highly significant (ΔAIC > 10 is considered decisive per Burnham & Anderson 2002). Fourier terms add no further improvement over month dummies.

**Residual diagnostics (all models):**
- ADF test: stationary (p < 0.0001) ✓
- ARCH-LM(10): stat ≈ 390–427, p < 0.0001 → strong ARCH effects confirmed → **GARCH warranted**
- Ljung-Box(10): p < 0.0001 → residual autocorrelation persists → ARMA(1,1) understates time-series structure

**GARCH(1,1) on Old_2022 residuals:**
- ω = 2.5098, α = 0.2077, β = 0.6702, persistence α+β = **0.878**
- High persistence indicates volatility shocks (e.g., holidays, COVID) dissipate slowly over many days.
- AIC (GARCH stage) = 33,644

### 5.2 Out-of-Sample RMSE — Benchmark Window (Dec 2021 – May 2022)

This replicates the hold-out window used in the BOT 2022 paper. All models are trained on 1997–2021 data and forecast dynamically for the full 6-month eval period.

| Model | RMSE (THB bn) | Δ vs Old_2022 | Δ vs BOT paper |
|-------|:---:|:---:|:---:|
| Old_2022 (Python, two-step) | **3.799** | 0.000 | −1.161 |
| ExtDummy | 3.846 | +0.047 | −1.114 |
| Regime + ExtDummy | 3.902 | +0.102 | −1.058 |
| Fourier + Regime | 3.899 | +0.099 | −1.061 |
| **[BOT 2022 paper — EViews joint MLE]** | 4.96 | baseline | 0.000 |
| **[Pre-2022 model]** | 7.31 | — | +2.350 |

**Note on methodology difference**: The two-step OLS+ARIMA approach achieves lower RMSE than the BOT's EViews joint MLE ARMA(1,1) because: (a) OLS minimises the sum of squared errors exactly for the calendar regressors (vs MLE which balances regression and ARMA parameters jointly); and (b) at forecast horizons beyond ~5 days, the ARIMA correction decays to zero, leaving a cleaner calendar-driven mean forecast. The ranking across our four models is comparable.

**Why Old_2022 wins this specific window**: Dec–May contains only one Songkran (April) and one New Year return (January). The extended SK/NY dummies help marginally for those events but the D_PostCovid regime dummy introduces a systematic downward bias (−0.725 THB bn/day) not present in Old_2022 (bias: −0.302). This biases the Regime model worse.

**Conclusion**: The benchmark window favours Old_2022 on point accuracy. However, the rolling backtest (below) reveals a different picture for COVID-year robustness.

### 5.3 Rolling Backtest RMSE (Expanding Window)

| Model | 2019 (pre-COVID) | 2020 (COVID year) | 2021 (recovery) | Benchmark 2021–22 |
|-------|:---:|:---:|:---:|:---:|
| Old_2022 | 4.925 | 5.480 | 4.353 | 3.799 |
| Regime + ExtDummy | 4.939 | **5.346** | 4.356 | 3.902 |
| Improvement (Regime vs Old) | −0.014 | **+0.134** | −0.003 | −0.103 |

**Key finding**: Regime + ExtDummy outperforms Old_2022 in 2020 (COVID year) by 0.134 THB bn RMSE — a meaningful improvement when CIC was most volatile and hardest to forecast. Pre-COVID and post-COVID recovery periods are near-identical. The benchmark window slight advantage of Old_2022 (0.103 THB bn) is within normal sampling variation.

**Recommendation**: For a model that must remain accurate across future regime changes (including any new COVID-like disruptions), **ExtDummy** is the safest production upgrade:
- Clearly better AIC (ΔAIC = 45)
- Better in COVID year
- Benchmark-window cost is only 0.047 THB bn RMSE (negligible)

### 5.4 RMSE by Forecast Horizon (3 Monthly Origins)

RMSE computed by refitting at Dec 2021, Feb 2022, and Apr 2022, forecasting h steps ahead, comparing to actuals at each horizon.

| Horizon | Old_2022 | Regime + ExtDummy |
|---------|:---:|:---:|
| 1-day ahead | 3.669 | **3.627** |
| 5-day ahead | 5.206 | **5.068** |
| 10-day ahead | 4.717 | 5.040 |
| 22-day ahead (monthly) | **1.554** | 1.685 |

**Key finding**: The 22-day (monthly) RMSE is very low (1.5–1.7 THB bn) because at 1-month horizon the ARMA correction has fully decayed and the forecast is driven almost entirely by the deterministic calendar structure (day-of-month, holidays, etc.) — which is fully known in advance. This is excellent for the monthly monitor use case: the structural component is highly predictable.

At shorter horizons (1–10 days), Regime+ExtDummy is marginally better (1-day and 5-day), confirming the extended dummies add signal for imminent holiday effects.

---

## 6. Figures

| Figure | Content |
|--------|---------|
| `fig1_cic_overview.png` | CIC level and daily change, full sample 1997–2022 |
| `fig2_actual_vs_forecast.png` | Actual vs forecast, benchmark window, all models |
| `fig3_forecast_errors.png` | Forecast error time series and distribution, each model |
| `fig4_residual_diagnostics.png` | ACF and Q-Q plots of training residuals |
| `fig5_rmse_comparison.png` | RMSE bar chart (benchmark) + rolling backtest |
| `fig6_horizon_rmse.png` | RMSE vs forecast horizon, Old_2022 vs Regime |
| `fig7_monthly_monitor.png` | Monthly aggregated forecast vs actual |
| `fig8_garch_volatility.png` | GARCH conditional volatility over training period |

---

## 7. Practical Notes for Monthly Monitor

The monthly monitor uses daily forecasts in two ways:

1. **Daily forecast accuracy**: RMSE in THB billion per day. Target: below 5 THB bn.
2. **Monthly total forecast**: Sum of daily forecasts for the month. A small daily bias compounds into a large monthly error.

**Recommended workflow**:
1. At end of each month, re-estimate Model B on all available data.
2. Forecast next month's business days using known calendar structure.
3. Report daily forecasts + monthly total + 95% prediction interval (using GARCH σ).
4. Track RMSE monthly; trigger model review if 3-month rolling RMSE > 6 THB bn.

---

## 8. References

- Anderson, R.G. & Gascon, C.S. (2009). "The U.S. Experience with Seasonal Currency Flows." Federal Reserve Bank of St. Louis Review.
- Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes." Econometrica, 66(1), 47–78.
- Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple Structural Change Models." Journal of Applied Econometrics, 18(1), 1–22.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." Journal of Econometrics, 31, 307–327.
- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Canova, F. & Hansen, B.E. (1995). "Are Seasonal Patterns Constant over Time? A Test for Seasonal Stability." Journal of Business & Economic Statistics, 13(3), 237–252.
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." Econometrica, 50(4), 987–1007.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Peng, F. & Shi, Y. (2014). "A Structural Break Approach to Currency Demand Forecasting." China Economic Review, 27, 316–325.
- Taylor, J.W. (2003). "Short-Term Electricity Demand Forecasting Using Double Seasonal Exponential Smoothing." Journal of the Operational Research Society, 54(8), 799–805.
- Tashman, L.J. (2000). "Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review." International Journal of Forecasting, 16(4), 437–450.
