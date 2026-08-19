# Currency in Circulation (CIC) Forecasting — Model Documentation

**Bank of Thailand | Financial Markets Department**

This is the single reference for the CIC forecasting work: what the original model
is, what problem it has, what was built to fix it, how each model works, and which
one to use in production.

> **What are we optimizing? EOM, not daily.** Every number in this document that
> matters for model selection is the **1-month-ahead end-of-month (EOM) CIC level
> RMSE** — the error on the single month-end level forecast made at each origin, not
> the day-to-day ΔCIC forecast. `Daily_Baseline` is genuinely excellent at daily
> ΔCIC (§2) and that skill is preserved everywhere as a no-regression **guardrail**,
> but nothing in §6–§8 ranks a model by its daily number. If you take one thing from
> this doc: **read the RMSE column labelled EOM, not the daily guardrail table.**

---

## 1. What CIC forecasting is for

Currency in Circulation is a key operational variable for liquidity management. The
BOT forecasts daily CIC to manage reserve money and plan open-market operations.
Daily forecasts are aggregated to a monthly figure that feeds the monthly liquidity
monitor.

**The number that matters (primary KPI): 1-month-ahead end-of-month (EOM) CIC level
RMSE.** At each month-end the model is re-estimated and asked for the CIC level at
the end of the following month. Daily ΔCIC RMSE is reported as a secondary metric
and acts as a no-regression guardrail.

**Data.** `input.xlsx`, sheet `RAW`: daily business-day CIC level (`Currency`, THB
billion) from 1997-08-29 to 2026-05-14, plus pre-computed calendar dummy columns.
Sheet `holiday` supplies Thai holiday dates and descriptions. The modelled target is
`Change = Currency.diff()`, the daily level change.

---

## 2. The original model (`cic.prg`, EViews — replicated as `Daily_Baseline`)

The 2022 BOT model regresses the daily change on a saturated set of calendar dummies
and adds an ARMA(1,1) error:

$$\Delta\text{CIC}_t = c + \boldsymbol{\beta}'\mathbf{X}_t + \phi_1 u_{t-1} + \theta_1 \varepsilon_{t-1} + \varepsilon_t$$

| Regressor group | Variables | Count |
|---|---|---|
| Day of month | Date_02 – Date_31 (base: day 1) | 30 |
| Day of week | D_TUE – D_FRI (base: Monday) | 4 |
| Week of month | D_WEEK2 – D_WEEK5 (base: week 1) | 4 |
| Month | D_JAN – D_NOV (base: December) | 11 |
| Holidays | D_PRE_LH1, D_PRE_LH3, D_POST_LH3, D_PRE_SH1 | 4 |
| COVID first wave | D_Covid_1st (Mar 24–27, 2020) | 1 |
| Last working day | D_LWD | 1 |
| **Total** | | **55 + constant** |

In Python this is estimated as a **two-step ARIMAX** (`TwoStepARIMAX` in
`cic_forecast.py`): OLS on the 55 dummies, then ARIMA(1,0,1) on the OLS residuals.
By the Frisch–Waugh theorem this is equivalent to joint MLE in large samples but
converges in seconds, which is what makes hundreds of backtest refits practical.

This model is genuinely good at what it was built for. On daily ΔCIC it scores ~4.0
RMSE on the Dec-2021–May-2022 benchmark window against 4.96 for the published BOT
2022 result and 7.31 for the pre-2022 model.

---

## 3. The problem

The daily metric is saturated — every calendar-dummy variant lands within 0.03 THB bn
of the baseline — but the **EOM level** metric is not. Four findings, all measured on
real backtest errors rather than assumed:

1. **The KPI amplifies drift error ~22×.** Within-month daily errors are positively
   autocorrelated (lag-1 ≈ **0.39**). If daily errors were independent, the monthly
   sum RMSE would be ≈ daily RMSE × √21 ≈ 22; the realised value is ≈ 34. Summing 22
   daily forecasts compounds any error in the drift rather than averaging it away.

2. **The only thing carrying trend is a frozen constant.** In the baseline, `c` is
   estimated over the full 1997–present sample and AR(1) ≈ 0.28 decays to noise
   within ~2 trading days. When cash demand changes regime — hoarding in 2020,
   digital-payment erosion from 2021 — the drift is stale and the EOM level drifts off.

3. **Seasonal coefficients are stale.** Mean EOM error by calendar month (2020–2024):
   March **+24.5**, April **+21.3** (Songkran cash build-up under-forecast), June
   **−19.5**. The 1997-anchored betas no longer match post-COVID seasonal behaviour.

4. **Residuals are not white and not Gaussian.** Ljung-Box remains significant
   (weekly structure near lag 5), ARCH-LM is strongly significant (GARCH(1,1)
   persistence α+β ≈ 0.88), and the Q-Q plot shows fat tails.

5. **(Fixed 2026-08) The future exog matrix used to include days that never
   trade.** `generate_future_exog()` built the daily calendar for a forecast month
   with `pd.bdate_range()` alone — every Mon–Fri, no holiday exclusion — while
   `RAW`'s actual trading calendar excludes Thai public holidays (Songkran, Labour
   Day, etc.). Every daily model summed its forecast over these "phantom" days, each
   adding a spurious drift-plus-dummy contribution to the EOM total. A one-line check
   (`len(X_fut_df)` vs. the realised number of trading days) confirmed it: **62 of the
   100 backtest months** had a mismatch, worst in April (Songkran) with 9 of 9 years
   affected by 1–3 phantom days each — exactly the month with the largest seasonal
   bias in finding 3 above.

   Fixed in `generate_future_exog()` (now takes an `actual_dates` argument so a
   backtest restricts the matrix to the realised trading days of the target month;
   genuine future-facing forecasts fall back to business days minus the holiday
   sheet instead of raw `bdate_range()`). **Honest result:** re-running Gate 0
   (origins 2019-12 → 2024-12) post-fix moved `Daily_Baseline` from 32.860 → 32.890
   and `Daily_AdaptiveDrift` from 33.041 → 33.121 — i.e. essentially a wash in
   aggregate RMSE (well within noise), not the systematic improvement the phantom-day
   hypothesis predicted. The fix is still correct and worth keeping — the harness
   should not be summing forecasts over days that don't exist, on principle, and
   the per-month effect is real even where it nets out — but it is not, on its own,
   the accuracy lever for the April/March/June bias in finding 3. See §6 for the
   full selection/holdout comparison re-run under the fixed harness.

   **Scope of what was re-run:** §6's table (`Daily_Baseline`, `Daily_AdaptiveDrift`,
   `Monthly_SARIMA`, `Blend_Baseline_Monthly`) is regenerated under the fix. The
   narrative numbers quoted in §5.2–§5.4 below (`month_end_eom_backtest`'s own
   2020–2025 / pre-COVID / holdout figures — e.g. 33.04 vs. 32.86, 35.46, 60.8 vs.
   36.1) predate it and were not regenerated in this pass; they use the same
   now-fixed `generate_future_exog()`, so re-running them would move them by a
   similar small amount to Gate 0 above, not qualitatively change the conclusions
   those sections draw.

---

## 4. What was tried and rejected

Documented so nobody spends time re-testing these:

| Approach | Outcome |
|---|---|
| Separate Songkran / New Year holiday dummies | Helped only in the specific benchmark window; flat-to-worse elsewhere |
| Post-COVID step dummy (`D_PostCovid`) | Hurt out-of-sample — biases every post-2020 forecast |
| Annual Fourier harmonics | Redundant against month dummies, higher AIC |
| More fixed holiday dummies | Diminishing to negative returns |
| **Month-to-month error feedback** (add trailing-k-month mean EOM error) | **Hurt at every window tested** (k=3: 37.7 vs 31.0 raw). EOM errors are *negatively* autocorrelated (lag-1 −0.19), so error-chasing overcorrects |
| **`Daily_LevelTrend`** (smooth-trend UC on the daily level) | **Failed decisively** — see §5.4 |

---

## 5. The models

Names follow one scheme: **`<Frequency>_<Method>`**, with combinations as
**`Blend_<members>`**. Every model appears under these names in the code, the Excel
outputs and the figures.

The `<Frequency>` prefix is not cosmetic — it says how the model reaches the KPI:

- **`Daily_*`** forecast ΔCIC for each of the ~22 trading days in the target month and
  **sum** them: $\text{EOM} = L_M + \sum \widehat{\Delta\text{CIC}}$. Within-month
  errors therefore accumulate rather than average out (§3 finding 1).
- **`Monthly_*`** forecast the month-end level **directly, in one step**, from the
  monthly EOM series. Nothing is summed, so there is no ×22 amplification.
- **`Blend_*`** combine already-computed EOM forecasts from both families at the EOM
  level only.

`fig5a`, `fig5b` and `fig5c` split the KPI chart along exactly this line.

### 5.1 `Daily_Baseline` — the 2022 model (benchmark)
As described in §2. OLS on 55 calendar dummies + ARIMA(1,0,1) on residuals; EOM
forecast = last known level + Σ of the daily forecasts for next month.
*Strength:* the within-month calendar shape (day-of-month, holiday, last-working-day
effects) is known in advance and estimated over 29 years of data.
*Weakness:* the frozen constant, per §3.2.

### 5.2 `Daily_AdaptiveDrift` — adaptive drift (v1 attempt)
Replaces the frozen constant with a Kalman-filtered stochastic drift: OLS on the same
dummies, then `UnobservedComponents(level='local level', autoregressive=1)` on the
residuals, so ν_t updates as the regime evolves.

**Result: it did not work.** 33.04 vs the baseline's 32.86 EOM RMSE over 2020–2025;
it won only 2025, which was a single month. The diagnosis: the local-level variance is
estimated by MLE on a sample containing the March–April 2020 outliers, which inflates
σ²_level and makes the filtered drift chase daily noise. The jitter cancels out the
adaptivity gain. Retained in the codebase as a documented baseline, not recommended.

### 5.3 `Daily_AdaptiveSeasonal` — trailing-window betas + smooth trend
Fits the OLS step on a trailing 60-month window so seasonal betas can adapt, then a
`smooth trend` UC on the residuals. This is the only arm that attacks problem §3.3
directly.

**Not yet evaluated under the v2 protocol** — it is absent from the selection/holdout
tables in §6. What exists is v1's own backtest, which uses a different origin range,
and it is encouraging: pre-COVID (2018–2019) EOM RMSE **28.94 vs the baseline's
35.80**, with both the calendar and drift components beating the baseline (69.1/63.1
vs 91.9/81.7). Over 2020–2025 it is worse overall (35.46 vs 32.86), driven by COVID.
Running it through the v2 selection/holdout harness — and tuning the window length —
is the top open lead (§8).

### 5.4 `Daily_LevelTrend` — smooth-trend UC on the calendar-adjusted level *(rejected)*
Cumulates the OLS residuals into a calendar-adjusted level and fits a slope-drift UC
on it, masking COVID months as missing.

**Result: significantly worse than the baseline** (60.8 vs 36.1 on holdout, DM
p=0.019). Two causes: masking ten months of a *cumulated* series makes the Kalman
filter free-run and then absorb the entire COVID level shift as one enormous
innovation at re-entry — with `smooth trend` the level cannot jump, so the slope and
AR(1) states whipsaw (a −560 THB bn EOM error in Dec-2020). Even unmasked, an
integrated random-walk slope on a 7,000-observation daily series accumulates far more
forecast variance at the 22-day horizon than the constant it replaces.
**Lesson: handle COVID with intervention dummies, never by masking observations of an
integrated series.**

### 5.5 `Monthly_SARIMA` — direct monthly model
$$\text{SARIMA}(1,1,1)(0,1,1)_{12}$$ fitted **directly on the monthly EOM level
series** (~345 observations), with intervention dummies for 2020-03 and 2020-04.

*Why it helps:* it estimates drift and annual seasonality at the KPI's own frequency.
There is no ×22 amplification (§3.1) because nothing is being summed — the model
forecasts the month-end level in one step. The seasonal difference at lag 12 lets the
annual pattern evolve instead of being pinned by 29-year-average month dummies, and
the intervention dummies absorb COVID without distorting the trend.

`Monthly_UC` (local-linear-trend + seasonal(12), COVID months masked) is the same idea
via a state-space route. It performs similarly on holdout but carries a systematic
+19 THB bn bias in the selection window, so `Monthly_SARIMA` is preferred.

### 5.6 `Blend_Baseline_Monthly` — the *previous* recommended model
*(Superseded 2026-08 by `Blend_Baseline_BreakTrend`, §5.8 — kept because the
reasoning below still explains why blending works, and because it remains the
fallback if the new monthly member ever has to be withdrawn.)*
An inverse-MSE weighted average of `Daily_Baseline` and `Monthly_SARIMA`:

- at each month-end, weight each member ∝ 1 / (mean squared EOM error over its last
  12 realised forecasts),
- floor each weight at 0.1 and renormalise,
- use equal weights until 6 months of history exist.

Weights use only errors realised *before* the origin, so the procedure is
leakage-free and exactly reproducible in production — the desk already knows last
months' errors.

*Why it beats the original:* the two members fail in different regimes and their
errors are complementary. `Monthly_SARIMA` carries the drift and annual seasonality
(strong in 2018-19, 2021, 2024-26); `Daily_Baseline` carries within-month calendar
shape and is the safer model through shocks (2020, 2023). The trailing weights pick
the right mix per regime without anyone having to declare where the regimes are, and
if the blend ever degrades the weights migrate back toward the baseline on their own
— the worst case is approximately baseline performance.

### 5.7 `Monthly_BreakTrend` — structural-break-aware EOM model ⭐
Added 2026-08. This is the first candidate built after *mathematically identifying*
where CIC's trend actually changes, rather than assuming a single stable regime.

**Step 1 — find the breaks (Bai–Perron 1998).** Multiple-break estimation by dynamic
programming: globally minimise the SSR of a piecewise $[1, t]$ trend regression on the
deseasonalised log EOM level, for each number of breaks $m$, then select $m$ by BIC.
Segment SSRs come from cumulative sums in closed form ($O(n^2)$), which is what makes
it cheap enough to re-run at every backtest origin.

**An important negative result first.** A *mean-shift* test on monthly growth finds
**nothing** — BIC picks $m=0$. Monthly growth has sd 2.6%/mo against a mean of
0.6%/mo, so any drift shift is buried in noise. CIC's regime change is only visible in
the **trend of the level**, which is what the segmented-trend regression above tests.
There the evidence is overwhelming: $m=0 \to m=1$ gives $F \approx 941$.

BIC selects **m=5 breaks** (see `fig6_structural_breaks.png`):

| Regime | Period | n | Trend growth |
|---|---|---|---|
| 1 | 1997-08 → 2000-01 | 30 | +5.78 %/yr |
| 2 | 2000-02 → 2004-09 | 56 | +11.07 %/yr |
| 3 | 2004-10 → 2007-06 | 33 | +4.23 %/yr |
| 4 | 2007-07 → 2011-05 | 47 | +9.95 %/yr |
| 5 | 2011-06 → 2020-03 | 106 | +5.55 %/yr |
| 6 | 2020-04 → 2026-05 | 74 | +5.18 %/yr |

The 2020-04 break is the COVID cash-hoarding *level* jump (the slope barely changes
across it); 2011-06 is a genuine growth-rate slowdown.

**Step 2 — forecast.** In log space, because CIC grows multiplicatively:

$$\log \text{EOM}_{t+1} = \log \text{EOM}_t + \hat\mu + \hat\delta_{m(t+1)} - \phi\,(\tilde g_t - \hat\mu)$$

- $\hat\mu$ — drift estimated on the **current regime only** (data after the last
  break), exponentially weighted with a 12-month half-life, so it never averages
  across a documented regime shift.
- $\hat\delta$ — month-of-year effect from the last **36 months** of $\Delta\log$.
  A deliberately short window: this is the model's main accuracy source, attacking
  the stale-seasonal problem of §3.3.
- $\phi = 0.2$ — mean reversion on last month's growth surprise. The sign is right by
  construction given the documented lag-1 EOM error autocorrelation of **−0.19** (§4).

**Honest attribution — the break detection is *not* why this model wins.** An ablation
twin ships alongside it (`--models monthly_breaktrend_nobreak`) so this can be
re-checked at any time:

| Drift specification | Selection | Holdout |
|---|---|---|
| breaks + EWMA (shipped) | 30.94 | 31.10 |
| breaks + plain mean | 31.39 | 30.94 |
| **no** breaks + EWMA | 30.91 | 31.03 |
| **no** breaks + plain mean | 31.31 | 30.58 |

All four are within ~0.5 THB bn — turning break detection *off* is, if anything,
marginally better. The reason is mechanical: the EWMA drift already discounts
pre-break observations, so the break restriction is redundant on top of it. The real
gain over the previous winner comes from **log space + the 36-month seasonal window +
the mean-reversion term**, not from Bai–Perron.

There is also an economic reason breaks buy little *at this horizon*: a 1-month-ahead
forecast is anchored to the last observed level, so a regime shift only moves the
forecast through the drift term — about 0.5%/mo ≈ 13 THB bn, of which a ±3pp
annualised regime change is worth only ~6 THB bn. Seasonal effects are ±2–4%
≈ 50–100 THB bn. **Seasonal staleness dominates structural change at h=1.**

Breaks are retained because they are individually significant, cost nothing, and bound
the damage if a genuinely new regime arrives — but they are not sold here as the
source of the improvement.

### 5.8 `Blend_Baseline_BreakTrend` — the new recommended model ⭐⭐
Inverse-MSE blend of `Daily_Baseline` and `Monthly_BreakTrend`, same leakage-free
weighting scheme as §5.6. This is the current winner on both windows (§6). It pairs
the daily model's within-month calendar shape with the break-trend model's adaptive
drift and fresh seasonals — the same complementarity argument as §5.6, but with a
stronger monthly member.

### 5.9 GARCH(1,1) — prediction intervals only
Fitted on the baseline residuals (ω=2.35, α=0.198, β=0.686, persistence 0.884). It
does **not** change point forecasts; it supplies time-varying prediction bands and
flags elevated-risk periods. Given the fat tails in `fig2`, calibrate bands with a
Student-t quantile rather than ±1.96σ.

---

## 6. Results

**Protocol — what "selection" and "holdout" mean.** Both are backtests over the
same expanding-window process (re-fit at every month-end origin, forecast the next
month's EOM level), just over two non-overlapping, chronologically-ordered date
ranges — this is a train/test split done the way it has to be done for time series
(older data first, so nothing from the future ever leaks into a decision made in
the past):

- **Selection window** (targets 2018-01 → 2023-12, n=72 months) — the "tuning"
  period. *Every* modelling choice in this document — which candidates to build,
  which one wins the inverse-MSE blend, window lengths, everything in §4's rejected
  list — was decided by looking **only** at performance here. Think of it as the
  sandbox: as many looks as needed, because nothing here is used to claim final
  accuracy.
- **Holdout window** (targets 2024-01 → 2026-04, n=28 months) — data the selection
  process never got to see or tune against. It is scored **exactly once**, after
  every choice above was already locked in. This is what answers "does the winner
  actually generalise, or did it just get lucky/overfit on the selection data?" —
  the DM test column below is computed on this window for exactly that reason.

A model that wins selection but loses holdout would be a red flag (overfitting to
the tuning period); `Blend_Baseline_Monthly` below wins both, which is the clean
result you want to see.

**Primary KPI — EOM level RMSE (THB bn).** Charts: `fig5a_daily_models_rmse.png`
(daily-frequency models), `fig5b_eom_models_rmse.png` (EOM-frequency models) and
`fig5c_blend_models_rmse.png` (blends). The split matters because the three families
reach the same KPI differently — daily models forecast ~22 daily ΔCIC values and
**sum** them, so within-month errors accumulate (§3 finding 1); EOM models forecast
the month-end level in **one step**; blends combine the two. All three are scored on
the identical metric, and fig5b/fig5c both carry `Daily_Baseline`'s holdout RMSE as a
dashed reference line (the incumbent every candidate has to beat).

| Model | Freq | Selection 2018–2023 | Holdout 2024–2026 | DM vs baseline (holdout) |
|---|---|---|---|---|
| **`Blend_Baseline_BreakTrend`** ⭐ | blend | **27.66** | **30.41** | **−2.38, p=0.025** ✓ |
| `Monthly_BreakTrend_NoBreak` | EOM | 30.91 | 31.03 | −1.35, p=0.190 |
| `Monthly_BreakTrend` | EOM | 30.94 | 31.10 | −1.34, p=0.193 |
| `Blend_BreakTrend_Monthly` | blend | 32.41 | 31.84 | −1.50, p=0.144 |
| `Blend_Baseline_Monthly` (previous winner) | blend | 30.89 | 33.62 | −2.59, p=0.015 ✓ |
| `Monthly_SARIMA` | EOM | 37.20 | 34.41 | −1.23, p=0.230 |
| `Daily_Baseline` (original) | daily | 32.71 | 37.45 | — |
| `Daily_AdaptiveDrift` | daily | 32.92 | 37.57 | +0.54, p=0.595 |

`Blend_Baseline_BreakTrend` wins **both** windows — it is best on selection (27.66)
*before* the holdout was scored, so the holdout is a confirmation, not a search
result. Against the original production model it improves holdout RMSE by **18.8%**
(37.45 → 30.41); against the previous winner by **9.5%** (33.62 → 30.41).

Two caveats stated plainly:
- **The DM p-value went up, not down** (0.015 → 0.025) even though the new blend is
  more accurate. That is not a contradiction: DM tests the *error difference series*
  against `Daily_Baseline`, and the new blend's advantage, while larger on average, is
  less uniform month-to-month. Both are significant at 5%.
- **This model was designed after the holdout had already been scored once** for the
  earlier candidates, so its holdout is a slightly weaker guarantee than the original
  one-shot. Every tuning decision (seasonal window, half-life, φ) was made by ranking
  on the **selection** window only — and selection independently picks the same
  winner — but the number is not quite as clean as a never-seen holdout. Treat 30.41
  as "very likely better", not as a fresh out-of-sample certificate.

*(`Monthly_UC` 57.79 / 34.77 and `Daily_LevelTrend` 80.55 / 60.78 are carried over
from the pre-fix run — unlike the four rows above, they have not been re-validated
under the fixed harness this pass. `Monthly_UC` never consumes the daily exog matrix
so the fix cannot change it; `Daily_LevelTrend` does, its numbers are stale, and
since it is a rejected model (§5.4) re-running it wasn't worth the compute here —
re-run before citing it, don't trust the number above.)*

**Honest surprise from the §3.5 fix.** The phantom-holiday-day bug was hypothesized
to be free accuracy, worst in Songkran-heavy April. Re-running the four affected
models says otherwise: `Daily_Baseline` and `Daily_AdaptiveDrift` both got **worse**
post-fix, not better — holdout RMSE up **+1.33** (36.12→37.45) and **+1.46**
(36.11→37.57) respectively, selection essentially flat (+0.08 / +0.12).
`Monthly_SARIMA` is untouched by construction (it never sees the daily exog matrix)
and its numbers are identical to three decimal places, which is a useful internal
consistency check that the fix didn't leak anywhere it shouldn't. **Read this as:
the fix was still the right thing to do — a harness should not sum forecasts over
days that don't exist, on principle — but it is not the accuracy lever for the
April/March/June seasonal bias in §3 finding 3.** That lever is still open; see §8.

**Honest caveats (pre-existing, unaffected by the fix).** On the older 2020-01 →
2025-01 Gate-0-style window the blend gives back some ground to the baseline in the
COVID-dominated period, where the monthly model was weak and the trailing weights
react with a lag. The Songkran bias of §3.3 is still present in the winner's errors
— confirmed, not resolved, by the §3.5 finding above.

**Daily ΔCIC guardrail.** `Daily_AdaptiveDrift` stays within 0.5% of the baseline in
every year. Blends and monthly models have no daily path of their own; their
production daily paths come from `reconciled_daily_path()`, which takes the
baseline's within-month shape and shifts it by a constant per day to hit the blended
EOM level — so daily shape accuracy is the baseline's by construction.

---

## 7. Production recipe

1. At each month-end, refit `Daily_Baseline` and `Monthly_BreakTrend` on all available
   data (seconds each; `cic_forecast.py --eom` caches per-origin forecasts).
   *(Superseded 2026-08: the pair used to be `Daily_Baseline` + `Monthly_SARIMA`.
   `Monthly_BreakTrend` replaces `Monthly_SARIMA` as the monthly member — §6.)*
2. Combine the two EOM forecasts with inverse-MSE weights over the last 12 realised
   EOM errors. **Do not freeze the weights** — the self-correction is the point.
3. For the daily monitor, take the baseline's daily path and shift it by a constant
   per day so the month sums to the blended EOM level (`reconciled_daily_path()`).
4. Report a Student-t prediction band using the GARCH σ.
5. Review the model if 3-month rolling daily RMSE exceeds 6 THB bn.

---

## 8. Open leads

- **Evaluate and tune `Daily_AdaptiveSeasonal`** (already wired into the harness as
  `--eom --models adaptive_seasonal`; then try trailing windows of 36/84/120 months or
  WLS with exponentially discounted weights). Its pre-COVID result (§5.3) is the
  strongest unexploited signal in the repo, and the stale Songkran betas of §3.3 are
  the clearest remaining accuracy lever — no evaluated candidate has addressed them
  yet, and §3.5 confirms the phantom-day bug fix was not a substitute fix for this.
  **Fitting it correctly requires the companion fix from §3(c) of the review that
  motivated this pass** — `AdaptiveSeasonalModel` currently fits its state-space step
  on residuals computed with trailing-window betas applied backwards over the *full*
  29-year history, which contaminates the variance MLE (the same failure mode that
  sank `Daily_AdaptiveDrift`, §5.2). Not yet fixed in this pass — fix before judging
  the model, or a good idea may get rejected for an implementation reason.
- **Push the seasonal-window finding further.** §5.7's ablation shows the 36-month
  seasonal window is doing most of the work, and it was tuned coarsely (24/30/36/42/45
  on the selection window, a flat optimum over 36–42). Worth trying: exponentially
  discounted seasonal weights instead of a hard cut, and separate window lengths for
  stable months vs. Songkran/New-Year months.
- **Re-score the winner on a genuinely fresh holdout** once more data arrives. §6 flags
  that `Blend_Baseline_BreakTrend` was designed after the holdout had been scored once,
  so its 30.41 is a slightly weaker guarantee than a never-seen number.
- **`Monthly_BreakTrend` gives up its COVID intervention.** `Monthly_SARIMA` carries
  explicit 2020-03/04 dummies; the break model handles COVID only through the 2020-04
  break plus EWMA discounting. Adding explicit intervention dummies to the break model
  is untested and may help the 2020 selection-window rows (43.8, its weakest year).
- A gradient-boosting challenger on calendar + holiday-distance features, evaluated
  on the same backtest with the same selection/holdout discipline.
- Student-t interval calibration (interval accuracy, not point accuracy).
- Do **not** revisit the `Daily_LevelTrend` family (§5.4).
- Do **not** expect more from structural-break detection *at this horizon* — §5.7
  measured it at ~0 and explains why (h=1 forecasts are anchored to the last observed
  level, so regime shifts move only the small drift term). It would matter much more
  at 3–12 month horizons, if the desk ever needs those.

---

## 9. Repository guide

| File | Purpose |
|---|---|
| `cic.prg` | The original EViews program — the specification of record |
| `cic_forecast.py` | Single-file pipeline (merged 2026-08): data loading, the daily models (`Daily_Baseline` / `Daily_AdaptiveDrift` / `Daily_AdaptiveSeasonal`), GARCH — **and** the EOM-level candidate harness (`Monthly_SARIMA`, `Monthly_BreakTrend`, `Monthly_UC`, `Daily_LevelTrend`, blends, selection/holdout, DM test, guardrail, fig3/4/5/6), entered via `--eom` |
| `input.xlsx` | Source data (`RAW`, `holiday`) |
| `CIC_output.xlsx` | **The one Excel deliverable.** `Daily` / `Monthly EOM` / `Summary` (from the daily pipeline) plus `EOM_Selection` / `EOM_Holdout` / `EOM_Detail` / `Daily_Guardrail` (from `--eom`) — 7 sheets, one file. Both writers only replace the sheets they own (`_excel_writer()`), so running the daily pipeline and `--eom` in either order accumulates into this one workbook rather than overwriting each other. |
| `cic_forecast_output.xlsx` | A **separate**, pre-existing internal-diagnostics workbook (per-config eval rows, in-sample fits, rolling/horizon RMSE, GARCH params — 11 sheets aimed at model developers, not the desk). Not touched by the 2026-08 merge; still written by the daily pipeline. Say the word if you want this folded into `CIC_output.xlsx` too. |
| `R/` | Partial R reproduction of the baseline (EDA and single-window backtest) |
| `fig1_data_overview.png` | CIC level and daily change, full sample. **daily pipeline.** |
| `fig2_garch_volatility.png` | GARCH conditional volatility. **daily pipeline.** *(the old residual-ACF/Q-Q figure was dropped 2026-08 — those diagnostics are still printed numerically by `run_diagnostics()`, just not plotted)* |
| `fig3_seasonal_pattern.png` | EOM level by year, with a genuine next-month forecast dot per family (incumbent / best daily / best EOM / best blend, named by actual model) and a zoom panel on the forecast month. **`--eom`.** |
| `fig4_eom_level_backtest.png` | Actual vs forecast EOM level and errors over time, **every** model in the run (solid = daily, dash-dot = EOM, dashed = blend). **`--eom`.** |
| `fig5a_daily_models_rmse.png` | **KPI chart, daily-frequency models** — sum ~22 daily ΔCIC forecasts to reach the month-end level, so within-month errors accumulate. **`--eom`.** |
| `fig5b_eom_models_rmse.png` | **KPI chart, EOM-frequency models** — forecast the month-end level in one step; `Daily_Baseline` holdout RMSE is a dashed reference line. **`--eom`.** |
| `fig5c_blend_models_rmse.png` | **KPI chart, blends** — inverse-MSE combinations, same reference line. **`--eom`.** |
| `fig6_structural_breaks.png` | **Bai–Perron structural breaks** — piecewise trend fit to log CIC with BIC-selected break dates, plus YoY growth with per-regime means (§5.7). **`--eom`.** |

fig3–fig6 involve models that only exist on the `--eom` path (Monthly_SARIMA,
Monthly_BreakTrend, the blends); the plain `python cic_forecast.py` daily pipeline
produces fig1–fig2 instead. All figures include every model run in that invocation —
pass `--models` to narrow it.

**How to run**

```bash
pip install -r requirements.txt
python cic_forecast.py                            # daily pipeline → CIC_output.xlsx (Daily/Monthly EOM/Summary) + fig1-2
python cic_forecast.py --eom --gate0               # EOM harness: post-fix sanity check only
python cic_forecast.py --eom                       # EOM harness: full backtest → CIC_output.xlsx (+EOM_* sheets) + fig3-6

# the run behind the §6 table (the winner and its ablation twin):
python cic_forecast.py --eom --models baseline,adaptive_drift,monthly_sarima,monthly_breaktrend,monthly_breaktrend_nobreak
```

---

## 10. References

- Anderson, R.G. & Gascon, C.S. (2009). "The U.S. Experience with Seasonal Currency Flows." FRB St. Louis Review.
- Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica*, 66(1), 47–78.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31, 307–327.
- Box, G.E.P. & Jenkins, G.M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *JBES*, 13(3), 253–263.
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity…" *Econometrica*, 50(4), 987–1007.
- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the Equality of Prediction Mean Squared Errors." *IJF*, 13(2), 281–291.
- Peng, F. & Shi, Y. (2014). "A Structural Break Approach to Currency Demand Forecasting." *China Economic Review*, 27, 316–325.
- Tashman, L.J. (2000). "Out-of-Sample Tests of Forecasting Accuracy." *IJF*, 16(4), 437–450.
- Timmermann, A. (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, Vol. 1, 135–196.
