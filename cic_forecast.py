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
import matplotlib.cm as cm
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
    raw = pd.read_excel(filepath, sheet_name='RAW', header=1)
    raw['Date']     = pd.to_datetime(raw['Date'],     errors='coerce')
    raw['Currency'] = pd.to_numeric(raw['Currency'],  errors='coerce')
    raw = raw.sort_values('Date').reset_index(drop=True)
    raw['Change'] = raw['Currency'].diff()
    df = raw.dropna(subset=['Date', 'Change']).copy()
    dummy_cols = [c for c in df.columns if c.startswith('D_') or c.startswith('Date_')]
    for c in dummy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
    df = df.set_index('Date')
    df.index = df.index.normalize()

    df['D_PostCovid'] = (df.index >= pd.Timestamp('2020-04-01')).astype(float)
    t = np.arange(len(df), dtype=float)
    P_ann = 261.0
    for k in range(1, 4):
        df[f'sin_ann_{k}'] = np.sin(2 * np.pi * k * t / P_ann)
        df[f'cos_ann_{k}'] = np.cos(2 * np.pi * k * t / P_ann)

    hol = pd.read_excel(filepath, sheet_name='holiday')
    hol['Date'] = pd.to_datetime(hol['Date'])
    df = _build_sk_ny_dummies(df, hol)
    return df


def load_holiday(filepath='input.xlsx'):
    hol = pd.read_excel(filepath, sheet_name='holiday')
    hol['Date'] = pd.to_datetime(hol['Date'])
    return hol


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

BASE_LABELS = {
    'Old_2022':       'Old_2022',
    'ExtDummy':       'ExtDummy',
    'Regime':         'Regime+ExtDummy',
    'Fourier_Regime': 'Fourier+Regime',
}

COLORS = {
    'Old_2022':       '#d62728',
    'ExtDummy':       '#1f77b4',
    'Regime':         '#2ca02c',
    'Fourier_Regime': '#ff7f0e',
}


def model_label(mname, train_label):
    return f'{BASE_LABELS[mname]} ({train_label})'


def get_X(df, model_name):
    cols = [c for c in REGS[model_name] if c in df.columns]
    return df[cols].astype(float).values, cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FUTURE EXOG GENERATION (for seasonal forecast dot)
# ─────────────────────────────────────────────────────────────────────────────

def _holiday_blocks_from_set(hol_set):
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


def generate_future_exog(model_name, start_str, end_str, hol):
    """
    Build the exogenous variable matrix for business dates in [start, end].
    Used to forecast CIC for months beyond the last data point.
    """
    dates = pd.bdate_range(start_str, end_str)
    if len(dates) == 0:
        return pd.DataFrame()

    fut = pd.DataFrame(index=dates)
    fut.index = fut.index.normalize()

    # Day of month
    for d in range(1, 32):
        fut[f'Date_{d:02d}'] = (fut.index.day == d).astype(float)

    # Day of week
    for i, nm in enumerate(['D_MON', 'D_TUE', 'D_WED', 'D_THU', 'D_FRI']):
        fut[nm] = (fut.index.dayofweek == i).astype(float)

    # Month
    mnames = ['D_JAN','D_FEB','D_MAR','D_APR','D_MAY','D_JUN',
              'D_JUL','D_AUG','D_SEP','D_OCT','D_NOV','D_DEC']
    for m, nm in enumerate(mnames, 1):
        fut[nm] = (fut.index.month == m).astype(float)

    # Week of month (1-5 based on day)
    for w in range(1, 6):
        fut[f'D_WEEK{w}'] = ((fut.index.day - 1) // 7 + 1 == w).astype(float)

    # Last working day of month
    lwd = set()
    for (yr, mo), _ in fut.groupby([fut.index.year, fut.index.month]):
        subset = fut[(fut.index.year == yr) & (fut.index.month == mo)]
        lwd.add(subset.index[-1])
    fut['D_LWD'] = fut.index.isin(lwd).astype(float)

    # COVID dummy: 0 for future
    fut['D_Covid_1st'] = 0.0
    fut['D_PostCovid']  = 1.0  # post-April-2020

    # Fourier terms (relative to start of series ~1997)
    origin = pd.Timestamp('1997-08-29')
    P_ann  = 261.0
    t_vals = np.array([(d - origin).days * 5 / 7 for d in fut.index], dtype=float)
    for k in range(1, 4):
        fut[f'sin_ann_{k}'] = np.sin(2 * np.pi * k * t_vals / P_ann)
        fut[f'cos_ann_{k}'] = np.cos(2 * np.pi * k * t_vals / P_ann)

    # Holiday-based dummies from holiday sheet
    fut_dates_list = sorted(fut.index.tolist())
    window_start = pd.Timestamp(start_str) - pd.Timedelta(days=14)
    window_end   = pd.Timestamp(end_str)   + pd.Timedelta(days=14)
    near_hol = set(
        hol.loc[(hol['Date'] >= window_start) & (hol['Date'] <= window_end), 'Date'].dt.normalize()
    )
    blocks    = _holiday_blocks_from_set(near_hol)
    long_blk  = [(bs, be) for bs, be in blocks if (be - bs).days >= 2]
    short_blk = [(bs, be) for bs, be in blocks if (be - bs).days < 2]

    pre_lh3, pre_lh1, post_lh3, pre_sh1 = set(), set(), set(), set()
    for bs, be in long_blk:
        pre_td  = [t for t in fut_dates_list if t < bs]
        post_td = [t for t in fut_dates_list if t > be]
        for lag in range(1, 4):
            if lag <= len(pre_td):
                pre_lh3.add(pre_td[-lag])
        if pre_td:
            pre_lh1.add(pre_td[-1])
        for lag in range(1, 4):
            if lag <= len(post_td):
                post_lh3.add(post_td[lag - 1])
    for bs, be in short_blk:
        pre_td = [t for t in fut_dates_list if t < bs]
        if pre_td:
            pre_sh1.add(pre_td[-1])

    fi = fut.index
    fut['D_PRE_LH1']  = fi.isin(pre_lh1).astype(float)
    fut['D_PRE_LH3']  = fi.isin(pre_lh3).astype(float)
    fut['D_POST_LH3'] = fi.isin(post_lh3).astype(float)
    fut['D_PRE_SH1']  = fi.isin(pre_sh1).astype(float)

    # SK/NY dummies for future window (Songkran = April 13-15, NY = Jan 1 / Dec 31)
    fut_yr_range = range(fut.index.year.min(), fut.index.year.max() + 1)
    sk_fut = set()
    ny_fut = set()
    sk_from_sheet = set(
        hol.loc[hol['Description'].str.contains('Songkran', case=False, na=False),
                'Date'].dt.normalize()
    )
    ny_from_sheet = set(
        hol.loc[hol['Description'].str.contains('New Year', case=False, na=False),
                'Date'].dt.normalize()
    )
    for yr in fut_yr_range:
        # Use sheet if available, else heuristic
        yr_sk = {d for d in sk_from_sheet if d.year == yr}
        if yr_sk:
            sk_fut |= yr_sk
        else:
            for md in [(4, 13), (4, 14), (4, 15)]:
                sk_fut.add(pd.Timestamp(yr, *md))
        yr_ny = {d for d in ny_from_sheet if d.year == yr}
        if yr_ny:
            ny_fut |= yr_ny
        else:
            ny_fut.add(pd.Timestamp(yr, 1, 1))
            ny_fut.add(pd.Timestamp(yr, 12, 31))

    sk_blk = _holiday_blocks_from_set(sk_fut)
    ny_blk = _holiday_blocks_from_set(ny_fut)

    sk_pre1, sk_post1 = set(), set()
    ny_pre1, ny_post1 = set(), set()
    for bs, be in sk_blk:
        pre_td  = [t for t in fut_dates_list if t < bs]
        post_td = [t for t in fut_dates_list if t > be]
        if pre_td:  sk_pre1.add(pre_td[-1])
        if post_td: sk_post1.add(post_td[0])
    for bs, be in ny_blk:
        pre_td  = [t for t in fut_dates_list if t < bs]
        post_td = [t for t in fut_dates_list if t > be]
        if pre_td:  ny_pre1.add(pre_td[-1])
        if post_td: ny_post1.add(post_td[0])

    fut['D_SK_PRE1']  = fi.isin(sk_pre1).astype(float)
    fut['D_SK_POST1'] = fi.isin(sk_post1).astype(float)
    fut['D_NY_PRE1']  = fi.isin(ny_pre1).astype(float)
    fut['D_NY_POST1'] = fi.isin(ny_post1).astype(float)

    cols = [c for c in REGS[model_name] if c in fut.columns]
    return fut[cols]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TWO-STEP ARIMAX
# ─────────────────────────────────────────────────────────────────────────────

class TwoStepARIMAX:
    """
    Two-step ARIMAX:
      Step 1) OLS on calendar/dummy regressors  →  beta, residuals
      Step 2) ARIMA(1,0,1) on OLS residuals     →  phi, theta, sigma2
    """
    def __init__(self):
        self.ols = self.arima = self.resid = self.fitted = None
        self.n_obs = self.n_params = None

    def fit(self, y, X):
        y, X = np.asarray(y, float), np.asarray(X, float)
        n, p = X.shape
        self.ols   = LinearRegression(fit_intercept=True).fit(X, y)
        ols_fit    = self.ols.predict(X)
        ols_res    = y - ols_fit
        self.arima = ARIMA(ols_res, order=(1, 0, 1), trend='n').fit(
            method='innovations_mle')
        self.resid  = self.arima.resid
        self.fitted = ols_fit + self.arima.fittedvalues
        self.n_obs  = n
        k = (p + 1) + 2 + 1
        self.n_params = k
        self._logL = self.arima.llf
        self.aic   = -2 * self._logL + 2 * k
        self.bic   = -2 * self._logL + k * np.log(n)
        pn = self.arima.param_names
        pv = self.arima.params
        ps = pd.Series(pv, index=pn)
        self.ar1   = float(ps.get('ar.L1',   np.nan))
        self.ma1   = float(ps.get('ma.L1',   np.nan))
        self.sigma = float(np.sqrt(ps.get('sigma2', np.nan)))
        return self

    def forecast(self, X_future):
        X_future = np.asarray(X_future, float)
        mean_fc  = self.ols.predict(X_future)
        arima_fc = self.arima.forecast(steps=len(X_future))
        return mean_fc + np.asarray(arima_fc)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — METRICS AND DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(actual, predicted):
    e = np.asarray(actual, float) - np.asarray(predicted, float)
    return {'RMSE': np.sqrt(np.mean(e**2)), 'MAE': np.mean(np.abs(e)),
            'ResidSD': np.std(e, ddof=1), 'Bias': np.mean(e),
            'n': len(e), 'errors': e}


def run_diagnostics(residuals, label=''):
    res = np.asarray(residuals, float)
    res = res[~np.isnan(res)]
    adf  = adfuller(res, autolag='AIC')
    arch = het_arch(res, nlags=10)
    lb   = acorr_ljungbox(res, lags=[10, 20], return_df=True)
    out  = {'adf_stat': adf[0], 'adf_pval': adf[1],
            'arch_stat': arch[0], 'arch_pval': arch[1],
            'lb_pval_10': float(lb['lb_pvalue'].iloc[0]),
            'lb_pval_20': float(lb['lb_pvalue'].iloc[1])}
    if label:
        print(f'\n  [{label}] Residual Diagnostics:')
        print(f'    ADF stationary:   stat={adf[0]:7.3f}  p={adf[1]:.4f}  '
              f'{"✓ stationary" if adf[1]<0.05 else "⚠ non-stationary"}')
        print(f'    ARCH-LM(10):      stat={arch[0]:7.3f}  p={arch[1]:.4f}  '
              f'{"⚠ ARCH effects → GARCH warranted" if arch[1]<0.05 else "✓ no ARCH"}')
        print(f'    Ljung-Box p(10/20): {out["lb_pval_10"]:.4f} / {out["lb_pval_20"]:.4f}  '
              f'{"⚠ autocorrelation" if out["lb_pval_10"]<0.05 else "✓ white noise"}')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

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
                lbl  = BASE_LABELS[mname]
                print(f'    {lbl:<20}  RMSE={m["RMSE"]:.3f}  MAE={m["MAE"]:.3f}  ResidSD={m["ResidSD"]:.3f}')
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
# SECTION 7 — GARCH
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
# SECTION 8 — VISUALIZATION
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
    ax.set_title('Currency in Circulation — Daily Level (1997–2026)', fontsize=13, fontweight='bold')
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


def plot_fig2_actual_vs_forecast(df_eval, forecast_dict, train_label, save_dir='.'):
    fig, ax = plt.subplots(figsize=(15, 5))
    actual = df_eval['Change'].values
    dates  = df_eval.index
    ax.plot(dates, actual, color='#333333', lw=1.5, label='Actual', zorder=5)
    for mname, pred in forecast_dict.items():
        rmse = np.sqrt(np.mean((actual - pred)**2))
        lw   = 1.8 if mname == 'ExtDummy' else 1.1
        ax.plot(dates, pred, color=COLORS.get(mname, 'grey'), lw=lw, alpha=0.8,
                label=f'{model_label(mname, train_label)}  RMSE={rmse:.3f}')
    ax.axhline(0, color='black', lw=0.5, ls='--')
    for ms in pd.date_range(dates[0].strftime('%Y-%m-01'), dates[-1], freq='MS'):
        ax.axvline(pd.Timestamp(ms), color='grey', lw=0.4, ls=':')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_ylabel('Daily CIC Change (THB billion)', fontsize=11)
    eval_desc = f'{dates[0].strftime("%b %Y")} → {dates[-1].strftime("%b %Y")}'
    ax.set_title(f'Actual vs. Forecast — {eval_desc}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, save_dir, 'fig2_actual_vs_forecast.png')


def plot_fig3_errors(df_eval, forecast_dict, train_label, save_dir='.'):
    actual = df_eval['Change'].values
    dates  = df_eval.index
    n      = len(forecast_dict)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (mname, pred) in enumerate(forecast_dict.items()):
        err  = actual - pred
        rmse = np.sqrt(np.mean(err**2))
        col  = COLORS.get(mname, 'grey')
        ax   = axes[i, 0]
        ax.bar(dates, err, color=col, alpha=0.65, width=1.5)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_title(f'{model_label(mname, train_label)} — Error (RMSE={rmse:.3f} THB bn)', fontsize=10)
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


def plot_fig4_residuals(residuals_dict, train_label, save_dir='.'):
    models = list(residuals_dict.keys())
    n      = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, mname in enumerate(models):
        res  = np.asarray(residuals_dict[mname], float)
        res  = res[~np.isnan(res)]
        conf = 1.96 / np.sqrt(len(res))
        acf_vals = acf(res, nlags=min(40, len(res)//5), fft=True)
        col  = COLORS.get(mname, 'grey')
        ax   = axes[i, 0]
        ax.bar(range(len(acf_vals)), acf_vals, color=col, alpha=0.7)
        ax.axhline(conf,  color='red', ls='--', lw=0.8)
        ax.axhline(-conf, color='red', ls='--', lw=0.8)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'{model_label(mname, train_label)}\nResidual ACF', fontsize=10)
        ax.set_xlabel('Lag')
        ax = axes[i, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(res, dist='norm')
        ax.scatter(osm, osr, s=6, alpha=0.5, color=col)
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
        ax.set_title(f'{model_label(mname, train_label)}\nNormal Q-Q', fontsize=10)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig4_residual_diagnostics.png')


def plot_fig5_rmse_comparison(all_bench, rolling_metrics, save_dir='.'):
    """all_bench = list of (label, rmse) for all model-config combos."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    labels = [x[0] for x in all_bench]
    rmsev  = [x[1] for x in all_bench]
    colors = [x[2] for x in all_bench]
    bars   = ax.bar(range(len(labels)), rmsev, color=colors, alpha=0.85)
    ax.axhline(4.96, color='black', ls='--', lw=1.8, label='BOT 2022 paper (2017-2021): 4.96')
    ax.axhline(7.31, color='grey',  ls=':',  lw=1.5, label='Pre-2022 model: 7.31')
    for bar, val in zip(bars, rmsev):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('RMSE (THB billion)', fontsize=11)
    ax.set_title('RMSE — Benchmark Window (Dec 2021 → May 2022)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(8.5, max(rmsev) * 1.2))

    ax = axes[1]
    compare   = [m for m in ['Old_2022', 'ExtDummy'] if m in rolling_metrics]
    win_labels = list(next(iter(rolling_metrics.values())).keys()) if rolling_metrics else []
    x = np.arange(len(win_labels))
    w = 0.35
    for j, mname in enumerate(compare):
        vals = [rolling_metrics[mname].get(wl, {}).get('RMSE', np.nan) for wl in win_labels]
        bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.9,
                      color=COLORS.get(mname, 'grey'), alpha=0.85,
                      label=BASE_LABELS[mname])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([wl.replace('→', '\n→\n') for wl in win_labels], fontsize=8)
    ax.set_ylabel('RMSE (THB billion)', fontsize=11)
    ax.set_title('Rolling Backtest RMSE\n(Expanding Window)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig5_rmse_comparison.png')


def plot_fig6_horizon(h_rmse_dict, train_label, save_dir='.'):
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for i, (mname, hdict) in enumerate(h_rmse_dict.items()):
        hs   = sorted(hdict.keys())
        vals = [hdict[h] for h in hs]
        ax.plot(hs, vals, marker=markers[i % len(markers)],
                color=COLORS.get(mname, 'grey'), lw=2.2, ms=10,
                label=model_label(mname, train_label))
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


def plot_fig7_monthly_monitor(df_eval, forecast_dict, train_label, save_dir='.'):
    actual        = df_eval['Change']
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
                label=model_label(mname, train_label))
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
    bars = ax.bar([model_label(m, train_label) for m in monthly_rmse],
                  list(monthly_rmse.values()),
                  color=[COLORS.get(m, 'grey') for m in monthly_rmse], alpha=0.85)
    for bar, val in zip(bars, monthly_rmse.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Monthly RMSE (THB billion)', fontsize=11)
    ax.set_title('Monthly Monitor Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticklabels([model_label(m, train_label) for m in monthly_rmse],
                        rotation=15, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig7_monthly_monitor.png')


def plot_fig8_garch(train_index, residuals, garch_res, save_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    n   = len(residuals)
    idx = train_index[:n]
    ax  = axes[0]
    ax.plot(idx, residuals, color='steelblue', lw=0.6, alpha=0.85)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.15, color='red', label='COVID 2020')
    ax.set_title('Old_2022 — Training Residuals', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (THB bn)')
    ax.legend(fontsize=9)
    ax = axes[1]
    cv  = garch_res.conditional_volatility[:len(idx)]
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


def plot_fig9_seasonal_cic(df, fitted_model, hol, save_dir='.'):
    """
    Seasonal CIC pattern: monthly end-of-month CIC level by year.
    Y-axis  : CIC level (THB billion)
    X-axis  : Month (Jan–Dec)
    Lines   : each year (last 10 years highlighted)
    Dot     : next-month forecast CIC level
    """
    df_lev   = df[df['Currency'].notna()].copy()
    eom      = df_lev['Currency'].resample('ME').last()
    eom      = eom.dropna()

    pivot = pd.DataFrame({'month': eom.index.month,
                          'year':  eom.index.year,
                          'cic':   eom.values})
    pivot = pivot.pivot(index='month', columns='year', values='cic')

    recent_years = sorted([y for y in pivot.columns if y >= pivot.columns.max() - 9])
    n_yr = len(recent_years)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap_colors = cm.tab10(np.linspace(0, 0.9, n_yr))

    for i, yr in enumerate(recent_years):
        col_data = pivot.get(yr)
        if col_data is None:
            continue
        valid = col_data.dropna()
        style = {'lw': 2.0 if yr == recent_years[-1] else 1.2,
                 'alpha': 1.0 if yr >= recent_years[-2] else 0.65}
        ax.plot(valid.index, valid.values,
                color=cmap_colors[i], marker='o', ms=4,
                label=str(yr), **style)

    # Forecast dot: next month from last available data
    last_date = df_lev.index.max()
    last_cic  = df_lev['Currency'].iloc[-1]

    # Forecast the next ~30 business days after last data point
    fc_start  = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    fc_end    = (last_date + pd.Timedelta(days=45)).strftime('%Y-%m-%d')
    try:
        X_fut = generate_future_exog('ExtDummy', fc_start, fc_end, hol)
        if len(X_fut) > 0:
            fc_change = fitted_model.forecast(X_fut.values)
            # Reconstruct CIC level
            cic_fc = last_cic
            eom_forecasts = {}
            for j, (dt, chg) in enumerate(zip(X_fut.index, fc_change)):
                cic_fc += chg
                mo = dt.month
                yr = dt.year
                eom_forecasts[(yr, mo)] = cic_fc  # keep updating → last day = EOM

            for (yr, mo), val in eom_forecasts.items():
                ax.scatter(mo, val, s=180, zorder=10,
                           color='gold', edgecolors='black', linewidths=1.5,
                           marker='*', label=f'Forecast ({pd.Timestamp(yr, mo, 1).strftime("%b %Y")})')
    except Exception as e:
        print(f'  (Forecast dot skipped: {e})')

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=11)
    ax.set_ylabel('CIC Level (THB billion)', fontsize=11)
    ax.set_title('Seasonal CIC Pattern — End-of-Month Level by Year\n'
                 '(★ = next-month forecast from ExtDummy model)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, ncol=2, loc='upper left',
              title='Year', title_fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    fig.tight_layout()
    _save(fig, save_dir, 'fig9_seasonal_cic.png')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(df, configs_results, rolling_metrics, h_rmse, garch_res,
                 save_dir='.'):
    """
    Export results to cic_forecast_output.xlsx

    Sheets
    ------
    Eval_Benchmark  – Dec 2021–May 2022 eval rows only (all models, easy to read)
    Eval_Extended   – Jan 2024–Dec 2025 eval rows only (extended config)
    InSample_Fitted – in-sample fitted values for each model (benchmark config)
    Full_Series     – complete daily CIC level + change (no forecasts, for reference)
    Benchmark_Metrics – RMSE/MAE/ResidSD/Bias for all model-config combos
    Rolling_RMSE    – expanding-window backtest
    Horizon_RMSE    – multi-horizon RMSE
    GARCH_Params    – GARCH(1,1) estimates
    """
    path = os.path.join(save_dir, 'cic_forecast_output.xlsx')

    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        # ── Eval sheets: one row per eval observation, all models side by side ──
        for cfg_key, cfg_data in configs_results.items():
            df_eval     = cfg_data['df_eval']
            forecasts   = cfg_data['forecasts']
            fitted_mdls = cfg_data['fitted_models']
            train_lbl   = cfg_data['train_label']

            out = pd.DataFrame({'Date': df_eval.index,
                                'CIC_Level': df_eval['Currency'].values,
                                'Change_Actual': df_eval['Change'].values})
            for mname, pred in forecasts.items():
                lbl = model_label(mname, train_lbl)
                out[f'{lbl}_Change_Forecast'] = pred
                # Reconstruct CIC level forecast
                last_cic  = df[df.index < df_eval.index[0]]['Currency'].iloc[-1]
                cic_levels = np.empty(len(pred))
                prev = last_cic
                for j, chg in enumerate(pred):
                    prev += chg
                    cic_levels[j] = prev
                out[f'{lbl}_CIC_Forecast'] = cic_levels

            sheet_name = 'Eval_Benchmark' if cfg_key == 'cfg_benchmark' else 'Eval_Extended'
            out.to_excel(writer, sheet_name=sheet_name, index=False)

        # ── In-sample fitted values (benchmark config) ──
        if 'cfg_benchmark' in configs_results:
            bdata   = configs_results['cfg_benchmark']
            df_tr   = bdata['df_train']
            lbl     = bdata['train_label']
            isfit   = pd.DataFrame({'Date': df_tr.index,
                                    'CIC_Level': df_tr['Currency'].values,
                                    'Change_Actual': df_tr['Change'].values})
            for mname, mdl in bdata['fitted_models'].items():
                fitted_chg = np.asarray(mdl.fitted)
                n_fit = min(len(fitted_chg), len(df_tr))
                col   = np.full(len(df_tr), np.nan)
                col[:n_fit] = fitted_chg[:n_fit]
                isfit[f'{model_label(mname, lbl)}_Fitted'] = col
            isfit.to_excel(writer, sheet_name='InSample_Fitted', index=False)

        # ── Full series ──
        full = df[['Currency', 'Change']].copy()
        full.columns = ['CIC_Level', 'Change']
        full.index.name = 'Date'
        full.reset_index().to_excel(writer, sheet_name='Full_Series', index=False)

        # ── Benchmark metrics (all model-config combos) ──
        rows = []
        for cfg_key, cfg_data in configs_results.items():
            lbl = cfg_data['train_label']
            ev  = cfg_data['eval_label']
            for mname, m in cfg_data['bench_metrics'].items():
                rows.append({
                    'Model': model_label(mname, lbl),
                    'Eval window': ev,
                    'RMSE (THB bn)': round(m['RMSE'],    4),
                    'MAE (THB bn)':  round(m['MAE'],     4),
                    'ResidSD':       round(m['ResidSD'], 4),
                    'Bias':          round(m['Bias'],    4),
                    'n obs':         m['n'],
                })
        rows.append({'Model': '[BOT 2022 paper (2017-2021)]',
                     'Eval window': 'Dec 2021–May 2022',
                     'RMSE (THB bn)': 4.960, 'MAE (THB bn)': None,
                     'ResidSD': 4.140, 'Bias': None, 'n obs': 119})
        rows.append({'Model': '[Pre-2022 model]',
                     'Eval window': 'Dec 2021–May 2022',
                     'RMSE (THB bn)': 7.310, 'MAE (THB bn)': None,
                     'ResidSD': 4.750, 'Bias': None, 'n obs': 119})
        pd.DataFrame(rows).to_excel(writer, sheet_name='Benchmark_Metrics', index=False)

        # ── Rolling RMSE ──
        roll_rows = []
        all_win = []
        for mname, wdict in rolling_metrics.items():
            all_win = list(wdict.keys())
            break
        for mname, wdict in rolling_metrics.items():
            row = {'Model': BASE_LABELS[mname]}
            for wl in all_win:
                val = wdict.get(wl, {}).get('RMSE', np.nan)
                row[wl] = round(float(val), 4) if not np.isnan(val) else None
            roll_rows.append(row)
        pd.DataFrame(roll_rows).to_excel(writer, sheet_name='Rolling_RMSE', index=False)

        # ── Horizon RMSE ──
        h_rows = []
        for mname, hdict in h_rmse.items():
            row = {'Model': BASE_LABELS[mname]}
            for h in [1, 5, 10, 22]:
                val = hdict.get(h, np.nan)
                row[f'h={h}d'] = round(float(val), 4) if not np.isnan(val) else None
            h_rows.append(row)
        pd.DataFrame(h_rows).to_excel(writer, sheet_name='Horizon_RMSE', index=False)

        # ── GARCH ──
        gp = garch_res.params
        garch_rows = [
            {'Parameter': 'omega',       'Value': round(float(gp['omega']),    6)},
            {'Parameter': 'alpha[1]',    'Value': round(float(gp['alpha[1]']), 6)},
            {'Parameter': 'beta[1]',     'Value': round(float(gp['beta[1]']),  6)},
            {'Parameter': 'persistence', 'Value': round(float(gp['alpha[1]'] + gp['beta[1]']), 6)},
            {'Parameter': 'AIC',         'Value': round(float(garch_res.aic), 2)},
            {'Parameter': 'BIC',         'Value': round(float(garch_res.bic), 2)},
        ]
        pd.DataFrame(garch_rows).to_excel(writer, sheet_name='GARCH_Params', index=False)

    print(f'  Saved → ./cic_forecast_output.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_config(df, cfg, all_models):
    """Fit and evaluate all models for one train/eval configuration."""
    train_lbl  = cfg['train_label']
    train_end  = cfg['train_end']
    eval_start = cfg['eval_start']
    eval_end   = cfg['eval_end']

    df_train = df.loc[:train_end]
    df_eval  = df.loc[eval_start:eval_end]

    print(f'\n  Config ({train_lbl}):  '
          f'train {df_train.index[0].date()}→{df_train.index[-1].date()} '
          f'(n={len(df_train)}) | eval {eval_start}→{eval_end} (n={len(df_eval)})')

    fitted_models = {}
    for mname in all_models:
        X_tr, _ = get_X(df_train, mname)
        mdl = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
        fitted_models[mname] = mdl

    forecasts    = {}
    bench_metrics = {}
    actual_arr   = df_eval['Change'].values
    for mname, mdl in fitted_models.items():
        X_ev, _ = get_X(df_eval, mname)
        pred = mdl.forecast(X_ev)
        forecasts[mname]     = pred
        bench_metrics[mname] = compute_metrics(actual_arr, pred)

    return {
        'train_label':   train_lbl,
        'eval_label':    f'{eval_start[:7]}→{eval_end[:7]}',
        'df_train':      df_train,
        'df_eval':       df_eval,
        'fitted_models': fitted_models,
        'forecasts':     forecasts,
        'bench_metrics': bench_metrics,
    }


def main():
    FILEPATH = 'input.xlsx'

    # Config 1 — Benchmark window (for BOT paper comparison)
    CFG_BENCHMARK = {
        'key':         'cfg_benchmark',
        'train_label': '1997-2021',
        'train_end':   '2021-11-30',
        'eval_start':  '2021-12-01',
        'eval_end':    '2022-05-31',
    }

    # Config 2 — Extended: train through 2023, test on 2024-2025
    CFG_EXTENDED = {
        'key':         'cfg_extended',
        'train_label': '1997-2023',
        'train_end':   '2023-12-31',
        'eval_start':  '2024-01-01',
        'eval_end':    '2025-12-31',
    }

    BACKTEST_WINDOWS = [
        ('2018-12-31', '2019-01-01', '2019-12-31'),
        ('2019-12-31', '2020-01-01', '2020-12-31'),
        ('2020-12-31', '2021-01-01', '2021-11-30'),
        ('2021-11-30', '2021-12-01', '2022-05-31'),
    ]
    HORIZON_ORIGINS = ['2021-12-01', '2022-02-01', '2022-04-01']
    ALL_MODELS  = list(REGS.keys())
    CORE_MODELS = ['Old_2022', 'ExtDummy']

    sep = '=' * 65
    print(sep)
    print('  CIC FORECASTING — OLD vs. NEW MODELS  (Bank of Thailand)')
    print(sep)

    # ── 1. Load ──
    print('\n[1] Loading data...')
    df  = load_data(FILEPATH)
    hol = load_holiday(FILEPATH)
    print(f'    Obs: {len(df)} | {df.index[0].date()} → {df.index[-1].date()}')
    chg = df['Change'].dropna()
    print(f'    Change: mean={chg.mean():.3f}  std={chg.std():.3f}  '
          f'min={chg.min():.3f}  max={chg.max():.3f} (THB bn)')
    adf_s, adf_p, *_ = adfuller(chg, autolag='AIC')
    print(f'    ADF: stat={adf_s:.3f}, p={adf_p:.4f}  → '
          f'{"stationary ✓" if adf_p<0.05 else "non-stationary ⚠"}')
    sk_tr = int(df.loc[:CFG_BENCHMARK['train_end'], 'D_SK_PRE1'].sum())
    ny_tr = int(df.loc[:CFG_BENCHMARK['train_end'], 'D_NY_PRE1'].sum())
    print(f'    D_SK_PRE1 (to 2021): {sk_tr} events  |  D_NY_PRE1: {ny_tr} events')

    # ── 2. Fig 1 ──
    print('\n[2] Figure 1 — CIC overview...')
    plot_fig1_overview(df)

    # ── 3. Run both configs ──
    print('\n[3] Fitting models on both training configurations...')
    print(f'  {"Model":<20} {"AIC":>10} {"BIC":>10} {"σ":>7} {"AR":>6} {"MA":>6}  [config]')
    print('  ' + '-' * 72)

    configs_results = {}
    for cfg in [CFG_BENCHMARK, CFG_EXTENDED]:
        key       = cfg['key']
        lbl       = cfg['train_label']
        df_train  = df.loc[:cfg['train_end']]
        df_eval   = df.loc[cfg['eval_start']:cfg['eval_end']]
        fitted_models = {}
        for mname in ALL_MODELS:
            X_tr, _ = get_X(df_train, mname)
            mdl = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
            fitted_models[mname] = mdl
            print(f'  [{model_label(mname, lbl):<28}]  '
                  f'AIC={mdl.aic:9.1f}  BIC={mdl.bic:9.1f}  '
                  f'σ={mdl.sigma:.3f}  AR={mdl.ar1:.3f}  MA={mdl.ma1:.3f}')
        forecasts    = {}
        bench_metrics = {}
        actual_arr   = df_eval['Change'].values
        for mname, mdl in fitted_models.items():
            X_ev, _ = get_X(df_eval, mname)
            pred = mdl.forecast(X_ev)
            forecasts[mname]      = pred
            bench_metrics[mname]  = compute_metrics(actual_arr, pred)
        configs_results[key] = {
            'train_label':   lbl,
            'eval_label':    f'{cfg["eval_start"][:7]}→{cfg["eval_end"][:7]}',
            'df_train':      df_train,
            'df_eval':       df_eval,
            'fitted_models': fitted_models,
            'forecasts':     forecasts,
            'bench_metrics': bench_metrics,
        }

    # ── 4. Benchmark metrics both configs ──
    print('\n[4] Benchmark metrics:')
    for cfg_key, cfg_data in configs_results.items():
        lbl = cfg_data['train_label']
        ev  = cfg_data['eval_label']
        print(f'\n  Config ({lbl}) — eval {ev}:')
        print(f'  {"Model":<32} {"RMSE":>8} {"MAE":>8} {"ResidSD":>10}')
        print('  ' + '-' * 60)
        for mname, m in cfg_data['bench_metrics'].items():
            print(f'  {model_label(mname, lbl):<32} {m["RMSE"]:>8.3f} {m["MAE"]:>8.3f} {m["ResidSD"]:>10.3f}')
        if cfg_key == 'cfg_benchmark':
            print(f'  {"[BOT 2022 paper (2017-2021)]":<32} {"4.960":>8} {"---":>8} {"4.140":>10}  (published)')
            print(f'  {"[Pre-2022 model]":<32} {"7.310":>8} {"---":>8} {"4.750":>10}  (published)')

    # ── 5. Residual diagnostics (benchmark config) ──
    print('\n[5] Residual diagnostics (benchmark config training residuals)...')
    b_fitted = configs_results['cfg_benchmark']['fitted_models']
    b_lbl    = configs_results['cfg_benchmark']['train_label']
    residuals = {m: mdl.resid for m, mdl in b_fitted.items()}
    for mname, res in residuals.items():
        run_diagnostics(res, label=model_label(mname, b_lbl))

    # ── 6. ARCH + GARCH ──
    print('\n[6] ARCH-LM + GARCH(1,1) on Old_2022 residuals...')
    old_res = np.asarray(residuals['Old_2022'], float)
    old_res = old_res[~np.isnan(old_res)]
    arch_stat, arch_pval, _, _ = het_arch(old_res, nlags=10)
    print(f'  ARCH-LM(10): stat={arch_stat:.3f}, p={arch_pval:.4f}  '
          f'→ {"⚠ ARCH effects present" if arch_pval<0.05 else "✓ no ARCH"}')
    garch_res = fit_garch(old_res)

    # ── 7. Rolling backtest ──
    print('\n[7] Rolling backtest (expanding window, 4 periods) — benchmark config...')
    rolling_metrics = rolling_backtest(df, CORE_MODELS, BACKTEST_WINDOWS)
    print('\n  Rolling RMSE summary:')
    win_labels = [f'{es[:7]}→{ee[:7]}' for _, es, ee in BACKTEST_WINDOWS]
    print(f'  {"Model":<24}' + ''.join(f'{w:>22}' for w in win_labels))
    print('  ' + '-' * (24 + 22 * len(win_labels)))
    for mname in CORE_MODELS:
        row = f'  {BASE_LABELS[mname]:<24}'
        for wl in win_labels:
            val = rolling_metrics.get(mname, {}).get(wl, {}).get('RMSE', np.nan)
            row += f'{val:>22.3f}'
        print(row)

    # ── 8. Horizon RMSE ──
    print('\n[8] Horizon RMSE (1, 5, 10, 22-day ahead, 3 origins) — benchmark config...')
    h_rmse = horizon_rmse_monthly(df, CORE_MODELS, HORIZON_ORIGINS)
    print('\n  Horizon RMSE:')
    print(f'  {"Model":<24} {"h=1":>8} {"h=5":>8} {"h=10":>8} {"h=22":>8}')
    print('  ' + '-' * 60)
    for mname in CORE_MODELS:
        row = f'  {BASE_LABELS[mname]:<24}'
        for h in [1, 5, 10, 22]:
            val = h_rmse.get(mname, {}).get(h, np.nan)
            row += f'  {val:>8.3f}'
        print(row)

    # ── 9. Figures ──
    print('\n[9] Generating figures...')

    # All-model benchmark RMSE bars (both configs)
    all_bench = []
    for cfg_data in configs_results.values():
        lbl = cfg_data['train_label']
        for mname, m in cfg_data['bench_metrics'].items():
            all_bench.append((model_label(mname, lbl), m['RMSE'], COLORS.get(mname, 'grey')))

    b_data = configs_results['cfg_benchmark']
    e_data = configs_results['cfg_extended']

    plot_fig2_actual_vs_forecast(b_data['df_eval'], b_data['forecasts'], b_data['train_label'])
    plot_fig3_errors(b_data['df_eval'], b_data['forecasts'], b_data['train_label'])
    plot_fig4_residuals(residuals, b_lbl)
    plot_fig5_rmse_comparison(all_bench, rolling_metrics)
    plot_fig6_horizon(h_rmse, b_data['train_label'])
    plot_fig7_monthly_monitor(b_data['df_eval'], b_data['forecasts'], b_data['train_label'])
    plot_fig8_garch(b_data['df_train'].index, old_res, garch_res)

    # Fig 9 — Seasonal CIC with forecast dot
    print('  Generating fig9 (seasonal CIC)...')
    plot_fig9_seasonal_cic(df, b_fitted['ExtDummy'], hol)

    # ── 10. Excel ──
    print('\n[10] Exporting Excel output...')
    export_excel(df, configs_results, rolling_metrics, h_rmse, garch_res)

    # ── Final summary ──
    print('\n' + sep)
    print('  FINAL RESULTS')
    print(sep)
    paper_rmse = 4.96

    for cfg_key, cfg_data in configs_results.items():
        lbl      = cfg_data['train_label']
        ev       = cfg_data['eval_label']
        bm       = cfg_data['bench_metrics']
        old_rmse = bm['Old_2022']['RMSE']
        best     = min(bm, key=lambda k: bm[k]['RMSE'])
        print(f'\n  ── Config ({lbl}), eval {ev} ──')
        print(f'  {"Model":<36} {"RMSE":>6}  {"vs Old_2022":>11}  {"vs BOT paper":>13}')
        print('  ' + '-' * 70)
        for mname in ALL_MODELS:
            r      = bm[mname]['RMSE']
            tag    = ' ← best' if mname == best else ''
            d_old  = r - old_rmse
            d_pap  = r - paper_rmse if cfg_key == 'cfg_benchmark' else float('nan')
            s_old  = '+' if d_old >= 0 else ''
            s_pap  = '+' if not np.isnan(d_pap) and d_pap >= 0 else ''
            d_pap_str = f'{s_pap}{d_pap:+.3f}' if not np.isnan(d_pap) else '      —'
            print(f'  {model_label(mname, lbl):<36} {r:>6.3f}  {s_old}{d_old:>10.3f}  {d_pap_str:>13}{tag}')
        if cfg_key == 'cfg_benchmark':
            print(f'  {"[BOT 2022 paper (2017-2021)]":<36} {"4.960":>6}  {"baseline":>11}  {"0.000":>13}')

    print(f'\n  All figures and cic_forecast_output.xlsx saved to: {os.path.abspath(".")}')
    print(sep + '\n')


if __name__ == '__main__':
    main()
