#!/usr/bin/env python3
"""
CIC Forecasting Model — Bank of Thailand
Old Model (2022 ARMA(1,1)) vs New Improved Models
Daily Currency in Circulation level-change forecasting for monthly monitor

Estimation strategy:
  Two-step ARIMAX: OLS regression for the mean equation + ARIMA(1,0,1) on
  residuals. This is numerically equivalent to joint SARIMAX-MLE in large
  samples (Frisch-Waugh theorem) but converges in seconds rather than minutes,
  making rolling backtests practical.

Literature:
  Box & Jenkins (1970), Engle (1982), Bollerslev (1986),
  Bai & Perron (1998), Anderson & Gascon (2009), Tashman (2000)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _build_sk_ny_dummies(df, hol):
    """
    Build Songkran (SK) and New Year (NY) pre/post holiday dummies.

    Holiday sheet only covers 2014-2026.  To ensure stable OLS coefficient
    estimates (one event per year across all training years), SK and NY dates
    are extended back to 1997 using fixed-calendar heuristics:
      SK  → April 13–15 (Thai calendar, fixed)
      NY  → January 1 + December 31

    All four dummies are mutually exclusive (no overlapping coverage):
      D_SK_PRE1  : last trading day before each Songkran block
      D_SK_POST1 : first trading day after each Songkran block
      D_NY_PRE1  : last trading day before each New Year block
      D_NY_POST1 : first trading day after each New Year block
    """
    trading_dates = sorted(df.index.normalize().tolist())

    # --- Build SK dates for ALL years covered by df ---
    sk_from_sheet = set(
        hol.loc[hol['Description'].str.contains('Songkran', case=False, na=False),
                'Date'].dt.normalize()
    )
    min_sk_year = min(d.year for d in sk_from_sheet) if sk_from_sheet else 2014
    sk_dates = set(sk_from_sheet)
    start_yr = df.index.year.min()
    for yr in range(start_yr, min_sk_year):
        for mday in [(4, 13), (4, 14), (4, 15)]:
            sk_dates.add(pd.Timestamp(yr, *mday))

    # --- Build NY dates for ALL years ---
    ny_from_sheet = set(
        hol.loc[hol['Description'].str.contains('New Year', case=False, na=False),
                'Date'].dt.normalize()
    )
    min_ny_year = min(d.year for d in ny_from_sheet) if ny_from_sheet else 2014
    ny_dates = set(ny_from_sheet)
    for yr in range(start_yr, min_ny_year):
        ny_dates.add(pd.Timestamp(yr, 1, 1))
        ny_dates.add(pd.Timestamp(yr, 12, 31))

    def holiday_blocks(hol_set):
        """Group consecutive holiday dates (gap ≤ 3 calendar days) into blocks."""
        if not hol_set:
            return []
        sorted_h = sorted(hol_set)
        blocks, bs, be = [], sorted_h[0], sorted_h[0]
        for d in sorted_h[1:]:
            if (d - be).days <= 3:
                be = d
            else:
                blocks.append((bs, be))
                bs = be = d
        blocks.append((bs, be))
        return blocks

    def nearest_td(blocks, n_pre, n_post):
        """For each block, find the nearest n_pre/n_post trading days before/after."""
        pre_idx, post_idx = set(), set()
        for bs, be in blocks:
            pre_td  = [t for t in trading_dates if t < bs]
            post_td = [t for t in trading_dates if t > be]
            for lag in range(1, n_pre + 1):
                if lag <= len(pre_td):
                    pre_idx.add(pre_td[-lag])
            for lag in range(1, n_post + 1):
                if lag <= len(post_td):
                    post_idx.add(post_td[lag - 1])
        return pre_idx, post_idx

    sk_blocks = holiday_blocks(sk_dates)
    ny_blocks = holiday_blocks(ny_dates)

    sk_pre1_idx, sk_post1_idx = nearest_td(sk_blocks, 1, 1)
    ny_pre1_idx, ny_post1_idx = nearest_td(ny_blocks, 1, 1)

    idx = df.index.normalize()
    df['D_SK_PRE1']  = idx.isin(sk_pre1_idx).astype(float)
    df['D_SK_POST1'] = idx.isin(sk_post1_idx).astype(float)
    df['D_NY_PRE1']  = idx.isin(ny_pre1_idx).astype(float)
    df['D_NY_POST1'] = idx.isin(ny_post1_idx).astype(float)
    return df


def load_data(filepath='input.xlsx'):
    # RAW sheet: header in row 1 (0-indexed)
    raw = pd.read_excel(filepath, sheet_name='RAW', header=1)

    raw['Date']     = pd.to_datetime(raw['Date'],     errors='coerce')
    raw['Currency'] = pd.to_numeric(raw['Currency'],  errors='coerce')

    # Change column in RAW is all NaN — compute from Currency levels
    raw = raw.sort_values('Date').reset_index(drop=True)
    raw['Change'] = raw['Currency'].diff()

    df = raw.dropna(subset=['Date', 'Change']).copy()

    dummy_cols = [c for c in df.columns if c.startswith('D_') or c.startswith('Date_')]
    for c in dummy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

    df = df.set_index('Date')
    df.index = df.index.normalize()

    # Post-COVID regime dummy (step from April 2020)
    df['D_PostCovid'] = (df.index >= pd.Timestamp('2020-04-01')).astype(float)

    # Annual Fourier terms
    t = np.arange(len(df), dtype=float)
    P_ann = 261.0
    for k in range(1, 4):
        df[f'sin_ann_{k}'] = np.sin(2 * np.pi * k * t / P_ann)
        df[f'cos_ann_{k}'] = np.cos(2 * np.pi * k * t / P_ann)

    # Build SK / NY dummies from holiday sheet
    hol = pd.read_excel(filepath, sheet_name='holiday')
    hol['Date'] = pd.to_datetime(hol['Date'])
    df = _build_sk_ny_dummies(df, hol)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REGRESSOR SETS
# ─────────────────────────────────────────────────────────────────────────────

DOM_COLS = [f'Date_{str(i).zfill(2)}' for i in range(2, 32)]
DOW_COLS = ['D_TUE', 'D_WED', 'D_THU', 'D_FRI']
WOM_COLS = ['D_WEEK2', 'D_WEEK3', 'D_WEEK4', 'D_WEEK5']
MON_COLS = ['D_JAN', 'D_FEB', 'D_MAR', 'D_APR', 'D_MAY', 'D_JUN',
            'D_JUL', 'D_AUG', 'D_SEP', 'D_OCT', 'D_NOV']
HOL_OLD  = ['D_PRE_LH1', 'D_PRE_LH3', 'D_POST_LH3', 'D_PRE_SH1', 'D_Covid_1st', 'D_LWD']
HOL_EXT  = ['D_SK_PRE1', 'D_SK_POST1', 'D_NY_PRE1', 'D_NY_POST1']
REGIME   = ['D_PostCovid']
FOURIER  = [f'{fn}_ann_{k}' for fn in ['sin', 'cos'] for k in range(1, 4)]

REGS = {
    'Old_2022':       DOM_COLS + DOW_COLS + WOM_COLS + MON_COLS + HOL_OLD,
    'ExtDummy':       DOM_COLS + DOW_COLS + WOM_COLS + MON_COLS + HOL_OLD + HOL_EXT,
    'Regime':         DOM_COLS + DOW_COLS + WOM_COLS + MON_COLS + HOL_OLD + HOL_EXT + REGIME,
    'Fourier_Regime': DOM_COLS + DOW_COLS + WOM_COLS + MON_COLS + HOL_OLD + HOL_EXT + REGIME + FOURIER,
}

MODEL_LABELS = {
    'Old_2022':       'Old 2022 (baseline)',
    'ExtDummy':       'Extended Dummies',
    'Regime':         'Regime + ExtDummy',
    'Fourier_Regime': 'Fourier + Regime',
}

COLORS = {
    'Old_2022':       '#d62728',
    'ExtDummy':       '#1f77b4',
    'Regime':         '#2ca02c',
    'Fourier_Regime': '#ff7f0e',
}


def get_X(df, model_name):
    cols = [c for c in REGS[model_name] if c in df.columns]
    return df[cols].astype(float).values, cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TWO-STEP ARIMAX FITTING
# ─────────────────────────────────────────────────────────────────────────────

class TwoStepARIMAX:
    """
    Two-step ARIMAX estimator:
      Step 1) OLS on all calendar/dummy regressors  →  beta, residuals
      Step 2) ARIMA(1,0,1) on OLS residuals          →  phi, theta, sigma2

    Forecast = OLS_mean(X_future) + ARIMA_forecast(residuals, steps)

    Justification: by the Frisch-Waugh theorem, OLS on the full model is
    consistent. In large samples (n>>p) OLS and GLS/MLE coefficient estimates
    converge. ARIMA(1,0,1) on residuals captures remaining serial dependence.
    The information criteria (AIC, BIC) are computed from the ARIMA log-likelihood
    with the full parameter count (regression + ARMA + sigma).
    """

    def __init__(self):
        self.ols      = None
        self.arima    = None
        self.n_params = None
        self.n_obs    = None
        self.resid    = None
        self.fitted   = None

    def fit(self, y, X, col_names=None):
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        n, p = X.shape

        self.ols      = LinearRegression(fit_intercept=True).fit(X, y)
        ols_fitted    = self.ols.predict(X)
        ols_resid     = y - ols_fitted

        arima_mod  = ARIMA(ols_resid, order=(1, 0, 1), trend='n')
        self.arima = arima_mod.fit(method='innovations_mle')

        self.resid    = self.arima.resid
        self.fitted   = ols_fitted + self.arima.fittedvalues
        self.n_obs    = n
        self.n_params = (p + 1) + 2 + 1  # OLS (incl. intercept) + AR,MA + sigma2

        self._logL = self.arima.llf
        k          = self.n_params
        self.aic   = -2 * self._logL + 2 * k
        self.bic   = -2 * self._logL + k * np.log(n)

        pn = self.arima.param_names
        pv = self.arima.params
        pd_ser   = pd.Series(pv, index=pn)
        self.ar1   = float(pd_ser.get('ar.L1',  np.nan))
        self.ma1   = float(pd_ser.get('ma.L1',  np.nan))
        self.sigma = float(np.sqrt(pd_ser.get('sigma2', np.nan)))
        return self

    def forecast(self, X_future, steps=None):
        X_future = np.asarray(X_future, dtype=float)
        n_fc     = len(X_future)
        mean_fc  = self.ols.predict(X_future)
        arima_fc = self.arima.forecast(steps=n_fc)
        return mean_fc + np.asarray(arima_fc)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(residuals, label=''):
    res = np.asarray(residuals, dtype=float)
    res = res[~np.isnan(res)]
    out = {}

    adf_stat, adf_pval, *_ = adfuller(res, autolag='AIC')
    out['adf_stat'], out['adf_pval'] = adf_stat, adf_pval

    arch_stat, arch_pval, _, _ = het_arch(res, nlags=10)
    out['arch_stat'], out['arch_pval'] = arch_stat, arch_pval

    lb = acorr_ljungbox(res, lags=[10, 20], return_df=True)
    out['lb_pval_10'] = float(lb['lb_pvalue'].iloc[0])
    out['lb_pval_20'] = float(lb['lb_pvalue'].iloc[1])

    if label:
        print(f'\n  [{label}] Residual Diagnostics:')
        print(f'    ADF stationary test:   stat={adf_stat:7.3f}  p={adf_pval:.4f}  '
              f'{"✓ stationary" if adf_pval<0.05 else "⚠ non-stationary"}')
        print(f'    ARCH-LM(10):           stat={arch_stat:7.3f}  p={arch_pval:.4f}  '
              f'{"⚠ ARCH effects → GARCH warranted" if arch_pval<0.05 else "✓ no ARCH effects"}')
        print(f'    Ljung-Box p(10/20):   {out["lb_pval_10"]:.4f} / {out["lb_pval_20"]:.4f}  '
              f'{"⚠ residual autocorrelation" if out["lb_pval_10"]<0.05 else "✓ white noise"}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual, predicted):
    e = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        'RMSE':    np.sqrt(np.mean(e**2)),
        'MAE':     np.mean(np.abs(e)),
        'ResidSD': np.std(e, ddof=1),
        'Bias':    np.mean(e),
        'n':       len(e),
        'errors':  e,
    }


def rolling_backtest(df, model_names, windows):
    results = {m: {} for m in model_names}
    for train_end, eval_start, eval_end in windows:
        wlabel = f'{eval_start[:7]}→{eval_end[:7]}'
        df_tr  = df.loc[:train_end]
        df_ev  = df.loc[eval_start:eval_end]
        if len(df_tr) < 200 or len(df_ev) < 5:
            continue

        print(f'\n  Window: train≤{train_end}, eval {eval_start}→{eval_end} ({len(df_ev)} obs)')
        for mname in model_names:
            X_tr, _ = get_X(df_tr, mname)
            X_ev, _ = get_X(df_ev, mname)
            try:
                mdl  = TwoStepARIMAX().fit(df_tr['Change'].values, X_tr)
                pred = mdl.forecast(X_ev)
                m    = compute_metrics(df_ev['Change'].values, pred)
                print(f'    {MODEL_LABELS[mname]:<24}  RMSE={m["RMSE"]:.3f}  MAE={m["MAE"]:.3f}  '
                      f'ResidSD={m["ResidSD"]:.3f}')
            except Exception as exc:
                print(f'    ⚠ {mname}: {exc}')
                m = {'RMSE': np.nan, 'MAE': np.nan, 'ResidSD': np.nan, 'n': 0}
            results[mname][wlabel] = m
    return results


def horizon_rmse_monthly(df, model_names, monthly_origins, horizons=(1, 5, 10, 22)):
    store = {m: {h: [] for h in horizons} for m in model_names}

    for origin_str in monthly_origins:
        origin   = pd.Timestamp(origin_str)
        df_train = df[df.index < origin]
        df_after = df[df.index >= origin]
        max_h    = max(horizons)
        if len(df_train) < 200 or len(df_after) < max_h:
            continue

        print(f'\n  Horizon refit at {origin_str} ({len(df_train)} train obs)...')
        for mname in model_names:
            X_tr, _ = get_X(df_train, mname)
            X_fc, _ = get_X(df_after.iloc[:max_h], mname)
            try:
                mdl     = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
                fc_seq  = mdl.forecast(X_fc)
                act_seq = df_after['Change'].iloc[:max_h].values
                for h in horizons:
                    if h <= len(fc_seq):
                        store[mname][h].append((fc_seq[h-1], act_seq[h-1]))
            except Exception as exc:
                print(f'    ⚠ {mname}: {exc}')

    h_rmse = {}
    for mname in model_names:
        h_rmse[mname] = {}
        for h in horizons:
            pairs = store[mname][h]
            if pairs:
                e = np.array([p - a for p, a in pairs])
                h_rmse[mname][h] = np.sqrt(np.mean(e**2))
            else:
                h_rmse[mname][h] = np.nan
    return h_rmse


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GARCH
# ─────────────────────────────────────────────────────────────────────────────

def fit_garch(residuals, label='GARCH(1,1)'):
    am  = arch_model(residuals, vol='GARCH', p=1, q=1, dist='normal', rescale=False)
    res = am.fit(disp='off', show_warning=False)
    p   = res.params
    print(f'  [{label}]  AIC={res.aic:.1f}  BIC={res.bic:.1f}  '
          f'ω={p["omega"]:.5f}  α={p["alpha[1]"]:.4f}  β={p["beta[1]"]:.4f}  '
          f'persistence={p["alpha[1]"]+p["beta[1]"]:.4f}')
    return res


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, directory, filename):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → ./{filename}')


def plot_fig1_overview(df, save_dir='.'):
    df_lev = df[df['Currency'].notna()]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    ax = axes[0]
    ax.plot(df_lev.index, df_lev['Currency'], color='#1f77b4', lw=0.9)
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.12, color='red', label='COVID period')
    ax.axvline(pd.Timestamp('2020-03-24'), color='red', lw=1.2, ls='--', alpha=0.7,
               label='COVID 4-day dummy (D_Covid_1st)')
    ax.set_ylabel('CIC Level (THB billion)', fontsize=11)
    ax.set_title('Currency in Circulation — Daily Level (1997–2022)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    ax = axes[1]
    ax.plot(df.index, df['Change'], color='#2ca02c', lw=0.55, alpha=0.8)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.12, color='red', label='COVID period')
    ax.set_ylabel('Daily Change (THB billion)', fontsize=11)
    ax.set_title('Daily Change in CIC — Model Dependent Variable', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig1_cic_overview.png')


def plot_fig2_actual_vs_forecast(df_eval, forecast_dict, save_dir='.'):
    fig, ax = plt.subplots(figsize=(15, 5))
    actual = df_eval['Change'].values
    dates  = df_eval.index

    ax.plot(dates, actual, color='#333333', lw=1.5, label='Actual', zorder=5)
    for mname, pred in forecast_dict.items():
        rmse = np.sqrt(np.mean((actual - pred)**2))
        lw   = 1.8 if mname == 'ExtDummy' else 1.1
        alp  = 0.95 if mname == 'ExtDummy' else 0.65
        ax.plot(dates, pred, color=COLORS.get(mname, 'grey'), lw=lw, alpha=alp,
                label=f'{MODEL_LABELS[mname]}  RMSE={rmse:.3f}')

    ax.axhline(0, color='black', lw=0.5, ls='--')
    for ms in pd.date_range('2021-12-01', '2022-05-01', freq='MS'):
        ax.axvline(pd.Timestamp(ms), color='grey', lw=0.4, ls=':')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylabel('Daily CIC Change (THB billion)', fontsize=11)
    ax.set_title('Actual vs. Forecast — Benchmark Window (Dec 2021 → May 2022)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, save_dir, 'fig2_actual_vs_forecast.png')


def plot_fig3_errors(df_eval, forecast_dict, save_dir='.'):
    actual = df_eval['Change'].values
    dates  = df_eval.index
    n = len(forecast_dict)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (mname, pred) in enumerate(forecast_dict.items()):
        err  = actual - pred
        rmse = np.sqrt(np.mean(err**2))
        col  = COLORS.get(mname, 'grey')

        ax = axes[i, 0]
        ax.bar(dates, err, color=col, alpha=0.65, width=1.5)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_title(f'{MODEL_LABELS[mname]} — Error (RMSE={rmse:.3f} THB bn)', fontsize=10)
        ax.set_ylabel('Error (THB bn)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        ax = axes[i, 1]
        ax.hist(err, bins=30, color=col, alpha=0.75, density=True, edgecolor='white')
        xr = np.linspace(err.min(), err.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, err.mean(), err.std()), 'k-', lw=1.5, label='Normal')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title('Error Distribution', fontsize=10)
        ax.set_xlabel('Error (THB bn)')
        ax.legend(fontsize=8)

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig3_forecast_errors.png')


def plot_fig4_residuals(residuals_dict, save_dir='.'):
    models = list(residuals_dict.keys())
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, mname in enumerate(models):
        res  = np.asarray(residuals_dict[mname], dtype=float)
        res  = res[~np.isnan(res)]
        conf = 1.96 / np.sqrt(len(res))
        nlags = min(40, len(res) // 5)
        col  = COLORS.get(mname, 'grey')

        acf_vals = acf(res, nlags=nlags, fft=True)
        ax = axes[i, 0]
        ax.bar(range(len(acf_vals)), acf_vals, color=col, alpha=0.7)
        ax.axhline(conf,  color='red', ls='--', lw=0.8)
        ax.axhline(-conf, color='red', ls='--', lw=0.8)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'{MODEL_LABELS[mname]}\nResidual ACF', fontsize=10)
        ax.set_xlabel('Lag')

        ax = axes[i, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(res, dist='norm')
        ax.scatter(osm, osr, s=6, alpha=0.5, color=col)
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
        ax.set_title(f'{MODEL_LABELS[mname]}\nNormal Q-Q', fontsize=10)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig4_residual_diagnostics.png')


def plot_fig5_rmse_comparison(bench_metrics, rolling_metrics, save_dir='.'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    all_models = list(bench_metrics.keys())
    bar_colors = [COLORS.get(m, 'grey') for m in all_models]

    ax = axes[0]
    rmse_vals = [bench_metrics[m]['RMSE'] for m in all_models]
    bars = ax.bar([MODEL_LABELS[m] for m in all_models], rmse_vals,
                  color=bar_colors, alpha=0.85)
    ax.axhline(4.96, color='black', ls='--', lw=1.8, label='BOT 2022 paper (4.96)')
    ax.axhline(7.31, color='grey',  ls=':',  lw=1.5, label='Pre-2022 model (7.31)')
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE (THB billion)', fontsize=11)
    ax.set_title('RMSE — Benchmark Window\n(Dec 2021 → May 2022)',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels([MODEL_LABELS[m] for m in all_models],
                        rotation=15, ha='right', fontsize=8.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(8.5, max(rmse_vals) * 1.2))

    ax = axes[1]
    compare = [m for m in ['Old_2022', 'ExtDummy'] if m in rolling_metrics]
    windows  = list(next(iter(rolling_metrics.values())).keys()) if rolling_metrics else []
    x = np.arange(len(windows))
    w = 0.35
    for j, mname in enumerate(compare):
        vals = [rolling_metrics[mname].get(wl, {}).get('RMSE', np.nan) for wl in windows]
        bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.9,
                      color=COLORS.get(mname, 'grey'), alpha=0.85,
                      label=MODEL_LABELS[mname])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([wl.replace('→', '\n→\n') for wl in windows], fontsize=8)
    ax.set_ylabel('RMSE (THB billion)', fontsize=11)
    ax.set_title('Rolling Backtest RMSE\n(Expanding Window, Pre–Post COVID)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig5_rmse_comparison.png')


def plot_fig6_horizon(h_rmse_dict, save_dir='.'):
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, (mname, hdict) in enumerate(h_rmse_dict.items()):
        hs   = sorted(hdict.keys())
        vals = [hdict[h] for h in hs]
        ax.plot(hs, vals, marker=markers[i % len(markers)],
                color=COLORS.get(mname, 'grey'), lw=2.2, ms=10,
                label=MODEL_LABELS[mname])
        for h, r in zip(hs, vals):
            if not np.isnan(r):
                ax.annotate(f'{r:.2f}', (h, r), xytext=(5, 5),
                            textcoords='offset points', fontsize=9.5)
    ax.set_xticks([1, 5, 10, 22])
    ax.set_xticklabels(['1-day\n(next day)', '5-day\n(1 week)',
                         '10-day\n(2 weeks)', '22-day\n(1 month)'])
    ax.set_ylabel('RMSE (THB billion)', fontsize=11)
    ax.set_title('Forecast Accuracy vs Horizon\n(3 Monthly Origins)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, 'fig6_horizon_rmse.png')


def plot_fig7_monthly_monitor(df_eval, forecast_dict, save_dir='.'):
    actual = df_eval['Change']
    monthly_actual = actual.resample('ME').sum()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.bar(monthly_actual.index, monthly_actual.values, width=20,
           color='#333333', alpha=0.35, label='Actual')
    for mname, pred in forecast_dict.items():
        pred_s  = pd.Series(pred, index=df_eval.index).resample('ME').sum()
        common  = monthly_actual.index.intersection(pred_s.index)
        ax.plot(common, pred_s.loc[common].values,
                color=COLORS.get(mname, 'grey'), marker='o', lw=2, ms=8,
                label=MODEL_LABELS[mname])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Monthly CIC Change (THB bn)', fontsize=11)
    ax.set_title('Monthly Aggregated CIC Change\nActual vs Forecast', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    monthly_rmse = {}
    for mname, pred in forecast_dict.items():
        pred_s = pd.Series(pred, index=df_eval.index).resample('ME').sum()
        common = monthly_actual.index.intersection(pred_s.index)
        if len(common):
            e = monthly_actual.loc[common] - pred_s.loc[common]
            monthly_rmse[mname] = np.sqrt(np.mean(e**2))

    bars = ax.bar([MODEL_LABELS[m] for m in monthly_rmse],
                  [monthly_rmse[m] for m in monthly_rmse],
                  color=[COLORS.get(m, 'grey') for m in monthly_rmse], alpha=0.85)
    for bar, val in zip(bars, monthly_rmse.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Monthly RMSE (THB billion)', fontsize=11)
    ax.set_title('Monthly Monitor Accuracy\n(Benchmark Window)', fontsize=12, fontweight='bold')
    ax.set_xticklabels([MODEL_LABELS[m] for m in monthly_rmse],
                        rotation=15, ha='right', fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig7_monthly_monitor.png')


def plot_fig8_garch(train_index, residuals, garch_res, save_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    n   = len(residuals)
    idx = train_index[:n]

    ax = axes[0]
    ax.plot(idx, residuals, color='steelblue', lw=0.6, alpha=0.85)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.15, color='red', label='COVID 2020')
    ax.set_title('Old_2022 — Training Residuals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (THB bn)')
    ax.legend(fontsize=9)

    ax = axes[1]
    cv  = garch_res.conditional_volatility
    cv  = cv[:len(idx)]
    ax.plot(idx[:len(cv)], cv, color='#d62728', lw=1.2)
    ax.fill_between(idx[:len(cv)], cv, alpha=0.2, color='#d62728',
                    label='Conditional σ (GARCH)')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.15, color='red')
    ax.set_title('GARCH(1,1) — Conditional Volatility', fontsize=12, fontweight='bold')
    ax.set_ylabel('Conditional Std Dev (THB bn)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    fig.tight_layout()
    _save(fig, save_dir, 'fig8_garch_volatility.png')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(df, df_train, df_eval, forecasts, bench_metrics,
                 rolling_metrics, h_rmse, garch_res, save_dir='.'):
    """
    Export results to cic_forecast_output.xlsx with sheets:
      1. Daily_Forecasts  — actual CIC level + Change + forecast series (Change & CIC)
      2. Benchmark_Metrics — RMSE / MAE / ResidSD / Bias for each model
      3. Rolling_RMSE     — expanding-window RMSE across backtest periods
      4. Horizon_RMSE     — RMSE at h=1,5,10,22 trading-day horizons
      5. GARCH_Params     — GARCH(1,1) parameter estimates
      6. Training_Data    — underlying time series used in model (for plots)
    """
    path = os.path.join(save_dir, 'cic_forecast_output.xlsx')

    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        # Sheet 1: Daily forecasts (eval window + full series)
        out = df[['Currency', 'Change']].copy()
        out.columns = ['CIC_Level', 'Change_Actual']

        # Forecast Change for eval window
        for mname, pred in forecasts.items():
            col = f'{MODEL_LABELS[mname]}_Change_fc'
            out.loc[df_eval.index, col] = pred

        # Reconstruct forecast CIC level for eval window
        for mname, pred in forecasts.items():
            col_fc  = f'{MODEL_LABELS[mname]}_CIC_fc'
            cic_fc  = np.full(len(df_eval), np.nan)
            cic_start = df.loc[df_eval.index[0], 'Currency']
            # CIC_fc[0] = CIC_actual[0] + change_forecast[0]
            last_cic = df.loc[df.index[df.index.get_loc(df_eval.index[0]) - 1], 'Currency']
            for j, p in enumerate(pred):
                cic_fc[j] = last_cic + p
                last_cic  = cic_fc[j]
            out.loc[df_eval.index, col_fc] = cic_fc

        out.index.name = 'Date'
        out.reset_index().to_excel(writer, sheet_name='Daily_Forecasts', index=False)

        # Sheet 2: Benchmark metrics
        rows = []
        for mname, m in bench_metrics.items():
            rows.append({
                'Model': MODEL_LABELS[mname],
                'RMSE (THB bn)': round(m['RMSE'], 4),
                'MAE (THB bn)':  round(m['MAE'],  4),
                'ResidSD':       round(m['ResidSD'], 4),
                'Bias':          round(m['Bias'],    4),
                'n obs':         m['n'],
            })
        rows.append({'Model': '[BOT 2022 paper]', 'RMSE (THB bn)': 4.960,
                     'MAE (THB bn)': None, 'ResidSD': 4.140, 'Bias': None, 'n obs': None})
        rows.append({'Model': '[Pre-2022 model]', 'RMSE (THB bn)': 7.310,
                     'MAE (THB bn)': None, 'ResidSD': 4.750, 'Bias': None, 'n obs': None})
        pd.DataFrame(rows).to_excel(writer, sheet_name='Benchmark_Metrics', index=False)

        # Sheet 3: Rolling RMSE
        roll_rows = []
        all_windows = []
        for mname, wdict in rolling_metrics.items():
            all_windows = list(wdict.keys())
            break
        for mname, wdict in rolling_metrics.items():
            row = {'Model': MODEL_LABELS[mname]}
            for wl in all_windows:
                rmse = wdict.get(wl, {}).get('RMSE', np.nan)
                row[wl] = round(float(rmse), 4) if not np.isnan(rmse) else None
            roll_rows.append(row)
        pd.DataFrame(roll_rows).to_excel(writer, sheet_name='Rolling_RMSE', index=False)

        # Sheet 4: Horizon RMSE
        horizons = [1, 5, 10, 22]
        h_rows = []
        for mname, hdict in h_rmse.items():
            row = {'Model': MODEL_LABELS[mname]}
            for h in horizons:
                val = hdict.get(h, np.nan)
                row[f'h={h}d'] = round(float(val), 4) if not np.isnan(val) else None
            h_rows.append(row)
        pd.DataFrame(h_rows).to_excel(writer, sheet_name='Horizon_RMSE', index=False)

        # Sheet 5: GARCH params
        gp = garch_res.params
        garch_rows = [
            {'Parameter': 'omega',    'Value': round(float(gp['omega']),    6)},
            {'Parameter': 'alpha[1]', 'Value': round(float(gp['alpha[1]']), 6)},
            {'Parameter': 'beta[1]',  'Value': round(float(gp['beta[1]']),  6)},
            {'Parameter': 'persistence', 'Value': round(float(gp['alpha[1]'] + gp['beta[1]']), 6)},
            {'Parameter': 'AIC',      'Value': round(float(garch_res.aic),   2)},
            {'Parameter': 'BIC',      'Value': round(float(garch_res.bic),   2)},
        ]
        pd.DataFrame(garch_rows).to_excel(writer, sheet_name='GARCH_Params', index=False)

        # Sheet 6: Training data (for plots / verification)
        train_out = df_train[['Currency', 'Change']].copy()
        train_out.columns = ['CIC_Level', 'Change']
        train_out.index.name = 'Date'
        train_out.reset_index().to_excel(writer, sheet_name='Training_Data', index=False)

    print(f'  Saved → ./cic_forecast_output.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    FILEPATH   = 'input.xlsx'
    TRAIN_END  = '2021-11-30'
    EVAL_START = '2021-12-01'
    EVAL_END   = '2022-05-31'

    BACKTEST_WINDOWS = [
        ('2018-12-31', '2019-01-01', '2019-12-31'),
        ('2019-12-31', '2020-01-01', '2020-12-31'),
        ('2020-12-31', '2021-01-01', '2021-11-30'),
        (TRAIN_END,    EVAL_START,   EVAL_END),
    ]

    HORIZON_ORIGINS = ['2021-12-01', '2022-02-01', '2022-04-01']
    ALL_MODELS  = list(REGS.keys())
    CORE_MODELS = ['Old_2022', 'ExtDummy']

    sep = '=' * 65
    print(sep)
    print('  CIC FORECASTING — OLD vs. NEW MODELS  (Bank of Thailand)')
    print(sep)

    # 1. Load
    print('\n[1] Loading data...')
    df = load_data(FILEPATH)
    print(f'    Obs: {len(df)} | {df.index[0].date()} → {df.index[-1].date()}')
    chg = df['Change'].dropna()
    print(f'    Change: mean={chg.mean():.3f}  std={chg.std():.3f}  '
          f'min={chg.min():.3f}  max={chg.max():.3f} (THB bn)')
    adf_s, adf_p, *_ = adfuller(chg, autolag='AIC')
    print(f'    ADF test: stat={adf_s:.3f}, p={adf_p:.4f}  → '
          f'{"stationary ✓" if adf_p<0.05 else "non-stationary ⚠"}')

    df_train = df.loc[:TRAIN_END]
    df_eval  = df.loc[EVAL_START:EVAL_END]
    print(f'\n    Train: {len(df_train)} obs  ({df_train.index[0].date()} → {df_train.index[-1].date()})')
    print(f'    Eval (benchmark): {len(df_eval)} obs  ({EVAL_START} → {EVAL_END})')

    # Verify SK/NY dummies were built
    sk_pre1_tr = int(df_train['D_SK_PRE1'].sum()) if 'D_SK_PRE1' in df_train.columns else 0
    ny_pre1_tr = int(df_train['D_NY_PRE1'].sum()) if 'D_NY_PRE1' in df_train.columns else 0
    print(f'    D_SK_PRE1 (train): {sk_pre1_tr} events  |  D_NY_PRE1 (train): {ny_pre1_tr} events')

    # 2. Fig 1
    print('\n[2] Figure 1 — CIC overview...')
    plot_fig1_overview(df)

    # 3. Fit all models
    print('\n[3] Fitting all models on training data...')
    print(f'  {"Model":<20} {"AIC":>10} {"BIC":>10} {"LogL":>10}  σ      AR     MA')
    print('  ' + '-' * 70)
    fitted_models = {}
    for mname in ALL_MODELS:
        X_tr, _ = get_X(df_train, mname)
        mdl = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
        fitted_models[mname] = mdl
        print(f'  [{mname:<16}]  AIC={mdl.aic:9.1f}  BIC={mdl.bic:9.1f}  '
              f'LogL={mdl._logL:9.1f}  σ={mdl.sigma:.3f}  AR={mdl.ar1:.3f}  MA={mdl.ma1:.3f}')

    # 4. Benchmark forecasts
    print('\n[4] Benchmark forecasts (Dec-2021 → May-2022)...')
    forecasts = {}
    for mname, mdl in fitted_models.items():
        X_ev, _ = get_X(df_eval, mname)
        forecasts[mname] = mdl.forecast(X_ev)
        print(f'  {mname}: {len(forecasts[mname])} forecast steps generated')

    # 5. Benchmark metrics
    print('\n[5] Benchmark window metrics (Dec-2021 → May-2022):')
    print(f'  {"Model":<24} {"RMSE":>8} {"MAE":>8} {"ResidSD":>10}')
    print('  ' + '-' * 52)
    bench_metrics = {}
    actual_arr = df_eval['Change'].values
    for mname, pred in forecasts.items():
        m = compute_metrics(actual_arr, pred)
        bench_metrics[mname] = m
        print(f'  {MODEL_LABELS[mname]:<24} {m["RMSE"]:>8.3f} {m["MAE"]:>8.3f} {m["ResidSD"]:>10.3f}')
    print(f'  {"[BOT 2022 paper]":<24} {"4.960":>8} {"---":>8} {"4.140":>10}  (published)')
    print(f'  {"[Pre-2022 model]":<24} {"7.310":>8} {"---":>8} {"4.750":>10}  (published)')

    # 6. Diagnostics
    print('\n[6] Residual diagnostics (training residuals)...')
    residuals = {mname: mdl.resid for mname, mdl in fitted_models.items()}
    for mname, res in residuals.items():
        run_diagnostics(res, label=MODEL_LABELS[mname])

    # 7. ARCH + GARCH
    print('\n[7] ARCH-LM + GARCH(1,1) on Old_2022 residuals...')
    old_res = np.asarray(residuals['Old_2022'], dtype=float)
    old_res = old_res[~np.isnan(old_res)]
    arch_stat, arch_pval, _, _ = het_arch(old_res, nlags=10)
    print(f'  ARCH-LM(10): stat={arch_stat:.3f}, p={arch_pval:.4f}  '
          f'→ {"⚠ ARCH effects present — GARCH warranted" if arch_pval<0.05 else "✓ no ARCH effects"}')
    garch_res = fit_garch(old_res)

    # 8. Rolling backtest
    print('\n[8] Rolling backtest (expanding window, 4 periods)...')
    rolling_metrics = rolling_backtest(df, CORE_MODELS, BACKTEST_WINDOWS)

    print('\n  Rolling RMSE summary:')
    win_labels = [f'{es[:7]}→{ee[:7]}' for _, es, ee in BACKTEST_WINDOWS]
    hdr = f'  {"Model":<24}' + ''.join(f'{w:>22}' for w in win_labels)
    print(hdr)
    print('  ' + '-' * (24 + 22 * len(win_labels)))
    for mname in CORE_MODELS:
        row = f'  {MODEL_LABELS[mname]:<24}'
        for wl in win_labels:
            val = rolling_metrics.get(mname, {}).get(wl, {}).get('RMSE', np.nan)
            row += f'{val:>22.3f}'
        print(row)

    # 9. Horizon RMSE
    print('\n[9] Horizon RMSE (1, 5, 10, 22-day ahead, 3 monthly origins)...')
    h_rmse = horizon_rmse_monthly(df, CORE_MODELS, HORIZON_ORIGINS)
    print('\n  Horizon RMSE:')
    print(f'  {"Model":<24} {"h=1":>8} {"h=5":>8} {"h=10":>8} {"h=22":>8}')
    print('  ' + '-' * 60)
    for mname in CORE_MODELS:
        row = f'  {MODEL_LABELS[mname]:<24}'
        for h in [1, 5, 10, 22]:
            val = h_rmse.get(mname, {}).get(h, np.nan)
            row += f'  {val:>8.3f}'
        print(row)

    # 10. Figures
    print('\n[10] Generating all figures...')
    plot_fig2_actual_vs_forecast(df_eval, forecasts)
    plot_fig3_errors(df_eval, forecasts)
    plot_fig4_residuals(residuals)
    plot_fig5_rmse_comparison(bench_metrics, rolling_metrics)
    if h_rmse:
        plot_fig6_horizon(h_rmse)
    plot_fig7_monthly_monitor(df_eval, forecasts)
    plot_fig8_garch(df_train.index, old_res, garch_res)

    # 11. Excel export
    print('\n[11] Exporting Excel output...')
    export_excel(df, df_train, df_eval, forecasts, bench_metrics,
                 rolling_metrics, h_rmse, garch_res)

    # Final summary
    print('\n' + sep)
    print('  FINAL RESULTS — BENCHMARK WINDOW (Dec-2021 → May-2022)')
    print(sep)
    paper_rmse = 4.96
    old_rmse   = bench_metrics['Old_2022']['RMSE']
    print(f'\n  {"Model":<28} {"RMSE":>6} {"vs Old_2022":>13} {"vs BOT paper":>13}')
    print('  ' + '-' * 62)
    best_name = min(bench_metrics, key=lambda k: bench_metrics[k]['RMSE'])
    for mname in ALL_MODELS:
        r   = bench_metrics[mname]['RMSE']
        tag = ' ← best' if mname == best_name else ''
        d_old   = r - old_rmse
        d_paper = r - paper_rmse
        s_old   = '+' if d_old >= 0 else ''
        s_pap   = '+' if d_paper >= 0 else ''
        print(f'  {MODEL_LABELS[mname]:<28} {r:>6.3f}  {s_old}{d_old:>10.3f}  '
              f'{s_pap}{d_paper:>10.3f}{tag}')
    print(f'  {"[BOT 2022 paper]":<28} {"4.960":>6}  {"baseline":>13}  {"0.000":>13}')
    print(f'  {"[Pre-2022 model]":<28} {"7.310":>6}  {"":>13}  {"+2.350":>13}')

    best_rmse = bench_metrics[best_name]['RMSE']
    print(f'\n  Best model: {MODEL_LABELS[best_name]}  (RMSE={best_rmse:.3f})')
    print(f'  Improvement vs Old_2022 Python replication: '
          f'{old_rmse - best_rmse:+.3f} THB bn')
    print(f'  Improvement vs BOT 2022 paper (4.96):       '
          f'{paper_rmse - best_rmse:+.3f} THB bn')
    print(f'\n  All figures saved to: {os.path.abspath(".")}')
    print(sep + '\n')


if __name__ == '__main__':
    main()
