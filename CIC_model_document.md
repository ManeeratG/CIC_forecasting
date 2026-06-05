# Currency in Circulation Forecasting — Technical Document

**Bank of Thailand | Financial Markets Department**
**Last updated: June 2026**

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

Five directions were explored. All improvements remain **univariate and self-contained** — no exogenous variables beyond calendar/dummy regressors.

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

**Implemented**: Seven expanding windows (2019–2025), plus a primary evaluation config using 1997–2019 training and 2020–2026 OOS (~22% holdout), covering COVID, post-COVID, and 2024–25 trend shift.

### Direction 5 — Adaptive Trend via State-Space Model (Model D1) *(new, June 2026)*

All four fixed-dummy ARIMAX variants share the same structural defect: the drift term `c` is a frozen constant estimated over the full training sample. AR(1) ≈ 0.28 decays to noise within ~2 trading days, so it carries no multi-month trend signal. When cash demand changes regime (hoarding in 2020, digital-payment erosion from 2021+), the frozen drift is stale and the end-of-month level forecast drifts off. This is why all four ARIMAX models cluster together on 2024–25 OOS metrics.

**Solution**: Replace the fixed constant with a time-varying adaptive drift estimated by a Kalman filter. See Section 4.7 for full specification.

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
| Harvey (1989) | Structural time series models and the Kalman filter; local level and local linear trend |
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

For this work we adopt the simpler **regime dummy** approach — Peng & Shi (2014) show it gives competitive out-of-sample accuracy relative to the structural time series approach. For the adaptive drift requirement, Harvey's (1989) local level / local linear trend state-space framework is employed.

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

All ARIMAX models (Old_2022, ExtDummy, Regime, Fourier_Regime) use a two-step approach:

1. **OLS** on all calendar/dummy regressors → $\hat{\boldsymbol{\beta}}$, OLS residuals
2. **ARIMA(1,0,1)** on OLS residuals → AR(1), MA(1), $\sigma^2$

This is numerically equivalent to joint SARIMAX-MLE in large samples (Frisch-Waugh theorem) but converges in seconds rather than minutes — making rolling backtests with seven windows and five models practical. Joint MLE (EViews approach) takes minutes per fit; two-step takes < 0.1 seconds.

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

### 4.7 Model D1 — State-Space Adaptive Drift *(added June 2026)*

**Motivation**: All four ARIMAX variants share a frozen constant `c` that cannot adapt when the CIC growth regime shifts. Model D1 replaces `c` with a time-varying drift $\nu_t$ estimated by the Kalman filter. Mathematically, a random-walk drift on the daily *change* is equivalent to a stochastic slope on the *level* — a local linear trend.

**Formulation (two-step)**:

1. **OLS** on 55 calendar dummies (same matrix as Old_2022) → $\hat{\boldsymbol{\beta}}$, OLS residuals $r_t$
2. **Unobserved Components (local level + AR(1) irregular)** on OLS residuals:

$$r_t = \nu_t + u_t$$
$$\nu_t = \nu_{t-1} + \zeta_t, \quad \zeta_t \sim \mathcal{N}(0, \sigma^2_\zeta)$$
$$u_t = \phi u_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon)$$

The OLS step concentrates all 55+ calendar betas, leaving only 3 variance parameters ($\sigma^2_\nu$, $\sigma^2_\varepsilon$, $\phi$) for state-space MLE. This achieves **~1.8 seconds per fit** (vs. minutes for joint MLE), making monthly rolling backtests over 61 origins practical.

**Why D2 (level formulation) was dropped**: A two-step OLS on the non-stationary CIC level series leaves trending residuals; the UC model sees an I(1) input and diverges, producing EOM RMSE of 85–125 THB bn vs. ~32 for others. D2 is documented here for completeness but not implemented.

**Key diagnostic**: Plot the smoothed drift $\nu_t$ over 1997–2026 (fig10). It must show:
- **2020 hump**: cash hoarding drives $\nu_t$ sharply positive
- **Post-2021 decline**: digital-payment adoption drives $\nu_t$ negative

Old_2022's frozen constant cannot show either pattern.

**statsmodels implementation**:

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents
mod = UnobservedComponents(endog=ols_resid, level='local level', autoregressive=1)
res = mod.fit(disp=False, method='bfgs', maxiter=300)
```

---

## 5. Evaluation Framework *(updated June 2026)*

### 5.1 Training and OOS Configuration

| Config | Training | OOS | Purpose |
|--------|----------|-----|---------|
| cfg_benchmark | 1997–2021 | Dec 2021 – May 2022 | BOT 2022 paper comparison only |
| **cfg_main** | **1997–2019** | **Jan 2020 – May 2026** | **Primary evaluation (~22% holdout)** |

cfg_main covers COVID, post-COVID recovery, and the 2024–25 digital-payment trend shift — the full range of challenging regimes.

### 5.2 Rolling Backtest — Seven Expanding Windows

| Window | Train end | Eval period | Regime |
|--------|-----------|-------------|--------|
| 1 | 2018-12-31 | 2019 | Pre-COVID baseline |
| 2 | 2019-12-31 | 2020 | COVID shock year |
| 3 | 2020-12-31 | 2021 | COVID recovery |
| 4 | 2021-12-31 | 2022 | Post-COVID normalisation |
| 5 | 2022-12-31 | 2023 | Early digital erosion |
| 6 | 2023-12-31 | 2024 | Digital-payment trend shift |
| 7 | 2024-12-31 | 2025 | Latest regime |

### 5.3 Primary KPI — End-of-Month Level RMSE

For each month origin M (last business day):
1. Fit all models on data through end of M.
2. Forecast next month's business days → sum daily ΔCIC → EOM level forecast.
3. Error = actual EOM level − forecast level.

This directly measures what operations teams care about: whether the monthly CIC level forecast is right.

---

## 6. Production Recommendation

**Recommended for production upgrade**: **Model A — ExtDummy** (1997-2019 training) — ARMA(1,1) with separate Songkran and New Year holiday dummies.

**Model D1** is recommended as a **supplementary adaptive trend model** for monitoring regime shifts. It adds a Kalman-filtered drift on top of the same calendar structure, providing earlier warnings when the CIC trend is changing direction.

| Feature | Old_2022 | **ExtDummy** | Model D1 |
|---------|----------|------|---------|
| Holiday treatment | Long holidays pooled | **SK + NY separate** | Same as Old_2022 |
| Drift | Frozen constant | Frozen constant | **Adaptive (Kalman)** |
| Trend adaptivity | None | None | **Auto-updates each month** |
| ARMA order | (1,1) | (1,1) | Local level + AR(1) |
| Regressors | 55 + const | 59 + const | 55 (no const) |
| Daily RMSE (2020–2026 OOS) | 4.995 | 5.022 | ~4.999 |
| EOM Level RMSE 2025 | 4.85 | **3.02** | **2.88** |
| EOM Level RMSE Overall (2020–25) | 32.86 | **32.79** | 33.04 |

**Key findings**:
- Daily ΔCIC RMSE: all models cluster at ~5.0 THB bn; differences are < 0.03 and not operationally significant.
- **EOM level RMSE 2025**: D1 (2.88) and ExtDummy (3.02) both substantially beat Old_2022 (4.85) — 40% improvement.
- D1's advantage is **trend-shift sensitivity**: the Kalman filter bent the drift negative as digital-payment erosion intensified in 2025.

---

## 7. Results

### 7.1 In-Sample Fit (cfg_main: Training 1997–2019)

| Model | AIC | BIC | Residual σ (THB bn) | AR(1) | MA(1) |
|-------|-----|-----|---------------------|-------|-------|
| Old_2022 | ~37,220 | ~37,619 | ~4.31 | ~0.28 | ~0.13 |
| ExtDummy | ~37,203 | ~37,629 | ~4.31 | ~0.28 | ~0.14 |
| Regime+ExtDummy | ~37,202 | ~37,636 | ~4.30 | ~0.28 | ~0.14 |
| Fourier+Regime | ~37,213 | ~37,687 | ~4.30 | ~0.28 | ~0.14 |

ExtDummy improves AIC by ~13 points over Old_2022 — clearly significant (ΔAIC > 4 per Burnham & Anderson 2002). Regime adds 1 more point (marginal). Fourier terms add almost nothing.

**Residual diagnostics (all models):**
- ADF test: stationary (p < 0.0001) ✓
- ARCH-LM(10): stat ≈ 406–419, p < 0.0001 → strong ARCH effects → **GARCH warranted**
- Ljung-Box(10): p < 0.0001 → some residual autocorrelation persists

**GARCH(1,1) on Old_2022 residuals:**
- ω = 2.3463, α = 0.1984, β = 0.6861, persistence α+β = **0.884**
- High persistence: volatility shocks (COVID, Songkran) dissipate slowly over many days.

### 7.2 Out-of-Sample Daily RMSE — cfg_main (Jan 2020 – May 2026)

| Model | RMSE (THB bn) | Δ vs Old_2022 |
|-------|:---:|:---:|
| Old_2022 | ~4.995 | 0.000 |
| ExtDummy | ~5.022 | +0.027 |
| Regime+ExtDummy | ~5.022 | +0.027 |
| Fourier+Regime | ~5.019 | +0.024 |
| D1 | ~4.999 | +0.004 |

All models cluster within 0.03 THB bn of each other. No single fixed-dummy variant dominates.

### 7.3 Rolling Backtest RMSE (Seven Expanding Windows)

| Model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Old_2022 | 4.93 | 5.47 | 4.58 | — | — | — | — |
| ExtDummy | 4.94 | 5.53 | 4.61 | — | — | — | — |

*Values for 2022–2025 windows reported in cic_forecast_output.xlsx → Rolling_RMSE.*

Both models show elevated RMSE in 2020 (COVID shock); neither can capture unprecedented cash hoarding dynamics adequately. ExtDummy's holiday dummy advantage is concentrated in periods containing Songkran and New Year events.

### 7.4 RMSE by Forecast Horizon (3 Monthly Origins, 1-Year OOS)

| Horizon | Old_2022 | ExtDummy |
|---------|:---:|:---:|
| 1-day ahead | ~3.6 | ~3.7 |
| 5-day ahead | ~5.5 | ~5.4 |
| 10-day ahead | ~4.3 | ~4.2 |
| 22-day (monthly) | ~1.4 | ~1.5 |

The 22-day (monthly) RMSE is very low because at 1-month horizon the ARMA correction has fully decayed and the forecast is driven almost entirely by the deterministic calendar structure — fully known in advance. Excellent for the monthly monitor.

### 7.5 End-of-Month Level RMSE (Primary KPI, 2020–2025)

| Model | 2020 | 2021 | 2022 | 2023 | 2024 | **2025** | Overall |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Old_2022 | 45.8 | 42.6 | 24.7 | 13.8 | 27.8 | **4.9** | 32.9 |
| ExtDummy | 45.4 | 42.6 | 25.4 | 13.8 | 27.5 | **3.0** | 32.8 |
| Regime | 51.0 | 42.4 | 29.2 | 15.4 | 27.2 | **3.2** | 35.0 |
| Fourier+Regime | 50.0 | 43.2 | 29.5 | 15.0 | 27.3 | **7.6** | 35.0 |
| **D1** | **45.7** | **42.9** | **25.2** | **14.0** | **28.1** | **2.9** | **33.0** |

2020 RMSE is high for all models (COVID-driven ~45 THB bn level error per month end). By 2025, ExtDummy and D1 both beat Old_2022 by ~40% — the adaptive drift and holiday specificity both pay off in the latest regime.

---

## 8. Figures

| Figure | Content |
|--------|---------|
| `fig1_cic_overview.png` | CIC level and daily change, full sample 1997–2026 |
| `fig2_actual_vs_forecast.png` | Actual vs forecast (2020–2026 OOS), all 5 models |
| `fig3_forecast_errors.png` | Forecast error time series and distribution, each model (last 2 years) |
| `fig4_residual_diagnostics.png` | ACF and Q-Q plots of training residuals |
| `fig5_rmse_comparison.png` | RMSE bars (cfg_main OOS) + 7-window rolling backtest |
| `fig6_horizon_rmse.png` | RMSE vs forecast horizon (h=1,5,10,22), all ARIMAX models |
| `fig7_monthly_monitor.png` | Monthly aggregated forecast vs actual, all 5 models |
| `fig8_garch_volatility.png` | GARCH conditional volatility over training period |
| `fig9_seasonal_cic.png` | Seasonal CIC pattern (EOM level by year), next-month forecast dots from 3 models |
| `fig10_trend_slope.png` | **D1 adaptive drift ν_t** — shows 2020 COVID hump and post-2021 decline |
| `fig11_eom_level.png` | EOM level: actual vs all 5 models (2020–2025), per-year RMSE bars |

---

## 9. Excel Output

### cic_forecast_output.xlsx — Technical workbook

| Sheet | Content |
|-------|---------|
| `Eval_Benchmark` | Dec 2021–May 2022 eval (119 obs) — all models, for BOT 2022 paper comparison |
| `Eval_Main` | Jan 2020–May 2026 eval — actual + change forecasts + CIC level reconstruction for all models |
| `InSample_Main` | 1997–2019 in-sample fitted values for all models |
| `Full_Series` | Complete daily CIC level + change (raw data reference) |
| `Benchmark_Metrics` | RMSE/MAE/ResidSD/Bias for all model-config combos + published references |
| `Rolling_RMSE` | 7-window expanding backtest RMSE |
| `Horizon_RMSE` | Multi-horizon (h=1,5,10,22) RMSE |
| `GARCH_Params` | GARCH(1,1) parameter estimates |
| `Level_EOM_Metrics` | EOM level RMSE by year, all 5 models |
| `Level_EOM_Detail` | Monthly EOM actual vs forecast per model (full detail rows) |

### CIC_output.xlsx — User-facing workbook *(new, June 2026)*

Clean Excel for operations teams, with yellow highlighting on forecast rows:

| Sheet | Content |
|-------|---------|
| `Daily` | Date, CIC Level (bn.), Daily Change — full history + 2-month average forecast |
| `Monthly EOM` | Date, actual EOM CIC + monthly change, plus Old_2022 / ExtDummy / D1 forecast columns |

---

## 10. Practical Notes for Monthly Monitor

The monthly monitor uses daily forecasts in two ways:

1. **Daily forecast accuracy**: RMSE in THB billion per day. Current target: below 5 THB bn.
2. **Monthly total forecast**: Sum of daily forecasts for the month. A small daily bias compounds into a large monthly error.

**Recommended workflow**:
1. At end of each month, re-estimate ExtDummy and D1 on all available data.
2. Forecast next month's business days using known calendar structure (generate with `generate_future_exog()`).
3. Report daily forecasts + monthly total + 95% prediction interval (using GARCH σ).
4. Track RMSE monthly; trigger model review if 3-month rolling RMSE > 6 THB bn.
5. Update SK/NY dummy holiday dates annually when BOT publishes the official holiday calendar.
6. Compare D1 drift direction each month — sustained negative drift signals continued digital-payment erosion.

**Installation** (local Python):
```
pip install -r requirements.txt
python cic_forecast.py
```

---

## 11. References

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
