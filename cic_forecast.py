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
from statsmodels.tsa.statespace.structural import UnobservedComponents
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
    'D1':             'Model D1',
}

COLORS = {
    'Old_2022':       '#d62728',
    'ExtDummy':       '#1f77b4',
    'Regime':         '#2ca02c',
    'Fourier_Regime': '#ff7f0e',
    'D1':             '#9467bd',
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
# SECTION 4b — STATE-SPACE TREND MODEL (Model D)
# ─────────────────────────────────────────────────────────────────────────────

class StateSpaceTrendModel:
    """
    Model D: adaptive-drift forecaster — two-step state-space.

    Step 1: OLS on the same calendar dummy matrix as Old_2022 (concentrates out
            regression betas analytically, same as TwoStepARIMAX Step 1).
    Step 2: UnobservedComponents on the OLS residuals — no exog, so MLE only
            optimises 2–3 variance parameters (20–50× faster than joint UC).

    D1  — local level on ΔCIC residuals:
            resid_t = ν_t + u_t,  ν_t = ν_{t-1} + ζ_t  (AR(1) irregular)
          ν_t is the stochastic drift; adapts as the regime changes.
          EOM forecast = L_last + Σ (OLS_mean + UC_level_forecast).

    D2_smooth — smooth trend on CIC-level residuals:
            resid_t = ℓ_t + b_t·t + u_t,  b_t drifts slowly (level var = 0).
          Directly targets the end-of-month level KPI.
          EOM forecast = terminal UC level forecast + OLS mean.
    """

    VARIANTS = {
        'D1': {'endog': 'change', 'level': 'local level'},
    }

    def __init__(self, variant='D1'):
        if variant not in self.VARIANTS:
            raise ValueError(f'Unknown variant {variant!r}. Choose from {list(self.VARIANTS)}')
        self.variant = variant
        self._cfg    = self.VARIANTS[variant]
        self.ols     = None
        self.uc_res  = None
        self.fitted  = None
        self.resid   = None
        self.aic     = self.bic = np.nan

    def fit(self, y_change, X, y_level=None):
        # Step 1 — OLS (concentrates calendar betas out of Kalman MLE)
        self.ols    = LinearRegression(fit_intercept=True).fit(X, y_change)
        ols_resid   = y_change - self.ols.predict(X)
        # Step 2 — local-level UC on residuals (2 variance params only → fast)
        mod         = UnobservedComponents(endog=ols_resid, level='local level', autoregressive=1)
        self.uc_res = mod.fit(disp=False, method='bfgs', maxiter=300)
        self.fitted = self.ols.predict(X) + np.asarray(self.uc_res.fittedvalues)
        self.resid  = y_change - self.fitted
        self.aic    = self.uc_res.aic
        self.bic    = self.uc_res.bic
        return self

    def forecast(self, X_future):
        ols_fc = self.ols.predict(X_future)
        uc_fc  = self.uc_res.get_forecast(steps=len(X_future))
        return ols_fc + np.asarray(uc_fc.predicted_mean)

    def smoothed_drift(self):
        """Smoothed local-level state ν_t on ΔCIC residuals (= adaptive drift)."""
        return self.uc_res.smoother_results.smoothed_state[0]


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
# SECTION 6b — MONTH-END LEVEL BACKTEST (Model D evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def month_end_eom_backtest(df, hol, start_year=2020, end_year=2025):
    """
    Rolling monthly backtest: primary KPI = 1-month-ahead end-of-month CIC level RMSE.

    For each origin = last trading day of month M (level L_M known):
      1. Fit all 5 models on data up to origin (expanding window).
      2. Forecast all business days of M+1 via generate_future_exog('Old_2022').
      3. EOM level forecast = L_M + Σ ΔCIC_hat over M+1.
      4. Error = actual EOM(M+1) level − forecast.

    All five ARIMAX/StateSpace variants share the same calendar dummy matrix so
    the same X_future is reused for all models.
    """
    arimax_models = ['Old_2022', 'ExtDummy', 'Regime', 'Fourier_Regime']
    ss_models     = ['D1']
    all_keys      = arimax_models + ss_models
    store         = {k: {'dates': [], 'actual': [], 'forecast': []} for k in all_keys}

    origins = pd.date_range(f'{start_year - 1}-12-31', f'{end_year - 1}-12-31', freq='ME')

    for origin in origins:
        avail = df.index[df.index <= origin]
        if len(avail) < 500:
            continue
        train_end = avail[-1]
        df_train  = df.loc[:train_end]

        nm_start = origin + pd.offsets.MonthBegin(1)
        nm_end   = nm_start + pd.offsets.MonthEnd(0)
        df_next  = df.loc[nm_start:nm_end]
        if len(df_next) < 5:
            continue

        lev_next = df_next['Currency'].dropna()
        if len(lev_next) == 0:
            continue
        actual_eom = float(lev_next.iloc[-1])
        lev_hist   = df_train['Currency'].dropna()
        if len(lev_hist) == 0:
            continue
        last_level = float(lev_hist.iloc[-1])

        fc_start = df_next.index[0].strftime('%Y-%m-%d')
        fc_end   = df_next.index[-1].strftime('%Y-%m-%d')
        try:
            X_fut_df = generate_future_exog('Old_2022', fc_start, fc_end, hol)
        except Exception:
            continue
        if len(X_fut_df) == 0:
            continue

        y_chg = df_train['Change'].values

        # ARIMAX models — reuse X_fut_df (Old_2022 columns, same for all ARIMAX)
        for mname in arimax_models:
            X_tr, _  = get_X(df_train, mname)
            X_fut_m  = generate_future_exog(mname, fc_start, fc_end, hol).values \
                       if mname != 'Old_2022' else X_fut_df.values
            try:
                mdl    = TwoStepARIMAX().fit(y_chg, X_tr)
                fc     = mdl.forecast(X_fut_m)
                fc_eom = last_level + float(fc.sum())
                store[mname]['dates'].append(nm_end)
                store[mname]['actual'].append(actual_eom)
                store[mname]['forecast'].append(fc_eom)
            except Exception as exc:
                print(f'    ⚠ {mname} EOM {origin.date()}: {exc}')

        # D1 — uses Old_2022 regressors
        X_tr_ss, _ = get_X(df_train, 'Old_2022')
        try:
            mdl    = StateSpaceTrendModel('D1').fit(y_chg, X_tr_ss)
            fc     = mdl.forecast(X_fut_df.values)
            fc_eom = last_level + float(fc.sum())
            store['D1']['dates'].append(nm_end)
            store['D1']['actual'].append(actual_eom)
            store['D1']['forecast'].append(fc_eom)
        except Exception as exc:
            print(f'    ⚠ D1 EOM {origin.date()}: {exc}')

    # Finalise store
    for k in all_keys:
        r = store[k]
        r['dates']    = pd.DatetimeIndex(r['dates'])
        r['actual']   = np.array(r['actual'],   dtype=float)
        r['forecast'] = np.array(r['forecast'], dtype=float)
        r['errors']   = r['actual'] - r['forecast']
        r['RMSE']     = np.sqrt(np.mean(r['errors'] ** 2)) if len(r['errors']) > 0 else np.nan

    # Print summary
    print(f'\n  EOM Level Backtest  ({start_year}–{end_year}):')
    print(f'  {"Model":<20}  {"Overall RMSE":>14}  {"n":>5}')
    print('  ' + '-' * 42)
    for k in all_keys:
        r = store[k]
        print(f'  {k:<20}  {r["RMSE"]:>14.3f}  {len(r["errors"]):>5}')

    report_years = list(range(start_year, end_year + 1))
    print(f'\n  Per-year EOM RMSE:')
    hdr = f'  {"Model":<20}' + ''.join(f'{y:>8}' for y in report_years)
    print(hdr)
    print('  ' + '-' * (20 + 8 * len(report_years)))
    for k in all_keys:
        r   = store[k]
        row = f'  {k:<20}'
        for yr in report_years:
            mask = r['dates'].year == yr
            row += f'{np.sqrt(np.mean(r["errors"][mask]**2)):>8.2f}' \
                   if mask.sum() > 0 else f'{"—":>8}'
        print(row)

    return store


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
        lw   = 2.0 if mname == 'D1' else (1.8 if mname == 'ExtDummy' else 1.1)
        ax.plot(dates, pred, color=COLORS.get(mname, 'grey'), lw=lw, alpha=0.8,
                label=f'{BASE_LABELS.get(mname, mname)}  RMSE={rmse:.3f}')
    ax.axhline(0, color='black', lw=0.5, ls='--')
    for ms in pd.date_range(dates[0].replace(day=1), dates[-1], freq='YS'):
        ax.axvline(pd.Timestamp(ms), color='grey', lw=0.6, ls=':')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel('Daily CIC Change (THB billion)', fontsize=11)
    eval_desc = f'{dates[0].strftime("%b %Y")} → {dates[-1].strftime("%b %Y")}'
    ax.set_title(f'Actual vs. Forecast — Daily ΔCIC  ({eval_desc})', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Train: {train_label}  |  OOS: {eval_desc}', fontsize=9, color='dimgrey')
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
        ax.set_title(f'{BASE_LABELS.get(mname, mname)} — Error (RMSE={rmse:.3f} THB bn)', fontsize=10)
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
    eval_desc = f'{dates[0].strftime("%b %Y")} → {dates[-1].strftime("%b %Y")}'
    fig.suptitle(f'Forecast Errors  |  Train: {train_label}  |  OOS: {eval_desc}',
                 fontsize=9, color='dimgrey', y=1.01)
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
        ax.set_title(f'{BASE_LABELS.get(mname, mname)}\nResidual ACF', fontsize=10)
        ax.set_xlabel('Lag')
        ax = axes[i, 1]
        (osm, osr), (slope, intercept, _) = stats.probplot(res, dist='norm')
        ax.scatter(osm, osr, s=6, alpha=0.5, color=col)
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
        ax.set_title(f'{BASE_LABELS.get(mname, mname)}\nNormal Q-Q', fontsize=10)
        ax.set_xlabel('Theoretical quantiles')
        ax.set_ylabel('Sample quantiles')
    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig4_residual_diagnostics.png')


def plot_fig5_rmse_comparison(all_bench, rolling_metrics, eval_period_label='', save_dir='.'):
    """
    all_bench = list of (model_name, rmse, color) for cfg_main only.
    LHS: daily RMSE bars for the main OOS period.
    RHS: 7-window rolling backtest for all 4 ARIMAX models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # LHS — RMSE bars, cfg_main OOS only
    ax = axes[0]
    labels = [BASE_LABELS.get(x[0], x[0]) for x in all_bench]
    rmsev  = [x[1] for x in all_bench]
    colors = [x[2] for x in all_bench]
    bars   = ax.bar(range(len(labels)), rmsev, color=colors, alpha=0.85)
    ax.axhline(4.96, color='black', ls='--', lw=1.8, label='BOT 2022 paper baseline: 4.96')
    ax.axhline(7.31, color='grey',  ls=':',  lw=1.5, label='Pre-2022 model: 7.31')
    for bar, val in zip(bars, rmsev):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Daily ΔCIC RMSE (THB billion)', fontsize=11)
    ax.set_title(f'Daily Forecast RMSE — OOS Period\n{eval_period_label}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(8.5, max(rmsev) * 1.2))

    # RHS — 7-window rolling backtest, all 4 ARIMAX models
    ax = axes[1]
    compare    = [m for m in ['Old_2022', 'ExtDummy', 'Regime', 'Fourier_Regime']
                  if m in rolling_metrics]
    win_labels = list(next(iter(rolling_metrics.values())).keys()) if rolling_metrics else []
    x  = np.arange(len(win_labels))
    n  = len(compare)
    w  = 0.8 / n
    for j, mname in enumerate(compare):
        vals   = [rolling_metrics[mname].get(wl, {}).get('RMSE', np.nan) for wl in win_labels]
        offset = (j - n / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w * 0.9,
                        color=COLORS.get(mname, 'grey'), alpha=0.85,
                        label=BASE_LABELS[mname])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([wl[:7] + '\n→' + wl[10:] for wl in win_labels], fontsize=8)
    ax.set_ylabel('Daily ΔCIC RMSE (THB billion)', fontsize=11)
    ax.set_title('Rolling Backtest RMSE — All Years\n(Expanding window, 1-year OOS each)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
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
                label=BASE_LABELS.get(mname, mname))
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
    ax.set_xlabel('Train: 1997–2019  |  OOS: Jan 2020 – May 2026', fontsize=9, color='dimgray')
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
                label=BASE_LABELS.get(mname, mname))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Monthly CIC Change (THB bn)', fontsize=11)
    ax.set_title('Monthly Aggregated CIC Change\nActual vs Forecast', fontsize=12, fontweight='bold')
    ax.set_xlabel('Train: 1997–2019  |  OOS: last 2 years shown', fontsize=9, color='dimgray')
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
    bar_labels = [BASE_LABELS.get(m, m) for m in monthly_rmse]
    bars = ax.bar(bar_labels,
                  list(monthly_rmse.values()),
                  color=[COLORS.get(m, 'grey') for m in monthly_rmse], alpha=0.85)
    for bar, val in zip(bars, monthly_rmse.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Monthly RMSE (THB billion)', fontsize=11)
    ax.set_title('Monthly Monitor Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=8)
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


def plot_fig9_seasonal_cic(df, fitted_models_dict, hol, save_dir='.'):
    """
    Seasonal CIC pattern: monthly end-of-month CIC level by year.
    Y-axis  : CIC level (THB billion)
    X-axis  : Month (Jan–Dec)
    Lines   : each year (last 10 years highlighted, older years faded)
    Fan     : 3-model forecast fan — Old_2022, ExtDummy, D1 — with shaded range
    """
    df_lev = df[df['Currency'].notna()].copy()
    eom    = df_lev['Currency'].resample('ME').last().dropna()

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

    # Fan chart: 3 model forecasts shown as lines + shaded range
    last_date = df_lev.index.max()
    last_cic  = df_lev['Currency'].iloc[-1]
    fc_start  = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    fc_end    = (last_date + pd.Timedelta(days=45)).strftime('%Y-%m-%d')

    fan_models = {k: v for k, v in fitted_models_dict.items()
                  if k in ('Old_2022', 'ExtDummy', 'D1')}
    fan_fc = {}  # mname -> {(yr, mo): eom_level}

    for mname, mdl in fan_models.items():
        try:
            X_fut = generate_future_exog('Old_2022' if mname == 'D1' else mname,
                                         fc_start, fc_end, hol)
            if len(X_fut) == 0:
                continue
            fc_change = mdl.forecast(X_fut.values)
            cic_fc = last_cic
            eom_fc = {}
            for dt, chg in zip(X_fut.index, fc_change):
                cic_fc += chg
                eom_fc[(dt.year, dt.month)] = cic_fc
            fan_fc[mname] = eom_fc
        except Exception as e:
            print(f'  (Fan forecast skipped for {mname}: {e})')

    if fan_fc:
        # Shaded range between min/max forecast across models
        all_keys = set()
        for eom_fc in fan_fc.values():
            all_keys.update(eom_fc.keys())
        for yr_mo in sorted(all_keys):
            mo = yr_mo[1]
            vals = [fan_fc[m][yr_mo] for m in fan_models if yr_mo in fan_fc.get(m, {})]
            if len(vals) >= 2:
                ax.fill_between([mo - 0.3, mo + 0.3],
                                [min(vals), min(vals)], [max(vals), max(vals)],
                                color='#cccccc', alpha=0.5, zorder=7)

        fan_label_map = {'Old_2022': 'Old_2022 fc', 'ExtDummy': 'ExtDummy fc', 'D1': 'D1 fc'}
        for mname in ('Old_2022', 'ExtDummy', 'D1'):
            eom_fc = fan_fc.get(mname, {})
            if not eom_fc:
                continue
            xs = [k[1] for k in sorted(eom_fc)]
            ys = [eom_fc[k] for k in sorted(eom_fc)]
            ax.plot(xs, ys, marker='o', ms=7, lw=0, zorder=10,
                    color=COLORS.get(mname, 'grey'),
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=fan_label_map[mname])

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=11)
    ax.set_ylabel('CIC Level (THB billion)', fontsize=11)
    ax.set_title('Seasonal CIC Pattern — End-of-Month Level by Year\n'
                 '(dots = next-month forecast; shaded band = range across Old_2022 / ExtDummy / D1)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, ncol=1, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              title='Year / Model', title_fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    fig.tight_layout()
    _save(fig, save_dir, 'fig9_seasonal_cic.png')


def plot_fig10_trend_slope(df_train, ss_d1_res, save_dir='.'):
    """
    fig10 — Smoothed adaptive drift ν_t from Model D1.

    Key diagnostic: ν_t must show the 2020 COVID cash-hoarding hump and the
    post-2021 digital-payment-erosion decline. Old_2022's frozen constant
    (shown as dashed baseline) cannot adapt to either regime change.
    """
    idx   = df_train.index[:len(ss_d1_res.fitted)]
    drift = ss_d1_res.smoothed_drift()[:len(idx)]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(idx, drift, color=COLORS['D1'], lw=1.2,
            label='D1 smoothed drift ν_t  (adaptive — Kalman-filtered)')
    ax.axhline(0, color='black', lw=0.7, ls='--', label='Zero baseline (≈ Old_2022 frozen constant)')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'),
               alpha=0.12, color='red', label='COVID 2020 (cash hoarding ↑)')
    ax.axvline(pd.Timestamp('2021-01-01'), color='orange', lw=1.4, ls='--', alpha=0.9,
               label='Post-2021 digital-payment erosion (drift ↓)')
    ax.set_ylabel('Adaptive drift ν_t on ΔCIC residuals (THB bn/day)', fontsize=11)
    ax.set_title('Model D1 — Smoothed Adaptive Drift ν_t  (1997 – training end)\n'
                 'Old_2022 uses a single frozen constant; D1 updates it via the Kalman filter',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, save_dir, 'fig10_trend_slope.png')


def plot_fig11_eom_level(eom_results, save_dir='.'):
    """
    fig11 — Actual vs forecast end-of-month CIC level, all 5 models.

    Left  : EOM level traces — actual (black) vs all model forecasts.
    Right : per-year RMSE bars — shows which model wins in which regime.
    """
    model_order = ['Old_2022', 'ExtDummy', 'Regime', 'Fourier_Regime', 'D1']
    model_labels = {
        'Old_2022':       'Old_2022 (frozen drift)',
        'ExtDummy':       'ExtDummy',
        'Regime':         'Regime+ExtDummy',
        'Fourier_Regime': 'Fourier+Regime',
        'D1':             'Model D1 (adaptive drift)',
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Left — actual vs forecast level traces
    ax = axes[0]
    ref = eom_results.get('Old_2022', {})
    if len(ref.get('dates', [])):
        ax.plot(ref['dates'], ref['actual'], color='#333333', lw=2.2,
                label='Actual EOM CIC Level', zorder=6)
    for k in model_order:
        r = eom_results.get(k, {})
        if len(r.get('dates', [])) == 0:
            continue
        rmse = r.get('RMSE', np.nan)
        lw   = 2.0 if k == 'D1' else 1.2
        ax.plot(r['dates'], r['forecast'], color=COLORS.get(k, 'grey'),
                lw=lw, alpha=0.85,
                label=f'{model_labels[k]}  RMSE={rmse:.1f}')
    ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'),
               alpha=0.10, color='red', label='COVID 2020')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel('EOM CIC Level (THB billion)', fontsize=11)
    ax.set_title('End-of-Month CIC Level — Actual vs Forecast (1-month-ahead, rolling refit)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Right — per-year RMSE bars
    ax = axes[1]
    ref_dates = eom_results.get('Old_2022', {}).get('dates', pd.DatetimeIndex([]))
    years = sorted(set(ref_dates.year)) if len(ref_dates) else []
    x = np.arange(len(years))
    n = len(model_order)
    w = 0.8 / n
    for j, k in enumerate(model_order):
        r    = eom_results.get(k, {})
        vals = []
        for yr in years:
            mask = r.get('dates', pd.DatetimeIndex([])).year == yr
            vals.append(np.sqrt(np.mean(r['errors'][mask] ** 2)) if mask.sum() > 0 else np.nan)
        offset = (j - n / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w * 0.9,
                        color=COLORS.get(k, 'grey'), alpha=0.85,
                        label=model_labels[k])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=9)
    ax.set_ylabel('EOM Level RMSE (THB billion)', fontsize=11)
    ax.set_title('1-Month-Ahead EOM Level RMSE by Year\n(lower = better)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout(pad=2)
    _save(fig, save_dir, 'fig11_eom_level.png')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8d — CIC OUTPUT EXCEL (clean user-facing workbook)
# ─────────────────────────────────────────────────────────────────────────────

def export_cic_output_excel(df, fitted_models_dict, hol, save_dir='.'):
    """
    Export a clean, user-facing CIC_output.xlsx with two sheets:

    Tab 1 — Daily
        Date | CIC Level (bn.) | Daily Change
        Full historical series. Forecast rows (after last actual date) in yellow.

    Tab 2 — Monthly EOM
        Date | CIC Level (actual) | Monthly Change (actual)
        | Old_2022 EOM Forecast | Old_2022 Monthly Change Forecast
        | ExtDummy EOM Forecast | ExtDummy Monthly Change Forecast
        | D1 EOM Forecast       | D1 Monthly Change Forecast
        Historical actuals + next 2 months of forecast. Forecast rows in yellow.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    YELLOW = PatternFill(start_color='FFFFC0', end_color='FFFFC0', fill_type='solid')
    HEADER = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
    HDR_FONT = Font(color='FFFFFF', bold=True)

    path = os.path.join(save_dir, 'CIC_output.xlsx')

    # ── Tab 1: Daily ──
    df_lev = df[['Currency', 'Change']].copy()
    df_lev.columns = ['CIC Level (bn.)', 'Daily Change (bn.)']
    df_lev.index.name = 'Date'
    df_lev = df_lev.dropna(subset=['CIC Level (bn.)'])
    last_actual = df_lev.index.max()

    # ── Build 2-month daily forecast for the 3 models ──
    fc_start = (last_actual + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    fc_end   = (last_actual + pd.offsets.MonthEnd(2)).strftime('%Y-%m-%d')
    model_order = ['Old_2022', 'ExtDummy', 'D1']

    future_fc = {}  # mname -> pd.Series(change, index=future_dates)
    for mname in model_order:
        mdl = fitted_models_dict.get(mname)
        if mdl is None:
            continue
        try:
            key = 'Old_2022' if mname == 'D1' else mname
            X_fut = generate_future_exog(key, fc_start, fc_end, hol)
            if len(X_fut):
                fc_change = mdl.forecast(X_fut.values)
                future_fc[mname] = pd.Series(fc_change, index=X_fut.index)
        except Exception as e:
            print(f'  (CIC output forecast skipped for {mname}: {e})')

    # Append forecast rows to daily tab
    if future_fc:
        ref_fc = next(iter(future_fc.values()))
        fc_rows = pd.DataFrame({
            'CIC Level (bn.)': np.nan,
            'Daily Change (bn.)': np.nan,
        }, index=ref_fc.index)
        # Reconstruct CIC level from last actual
        for mname, fc_s in future_fc.items():
            lvl = df_lev['CIC Level (bn.)'].iloc[-1]
            levels = []
            for chg in fc_s.values:
                lvl += chg
                levels.append(lvl)
        # Use the average across models for the daily tab level column
        all_levels = []
        for mname, fc_s in future_fc.items():
            lvl = df_lev['CIC Level (bn.)'].iloc[-1]
            lvls = []
            for chg in fc_s.values:
                lvl += chg
                lvls.append(lvl)
            all_levels.append(lvls)
        avg_levels = np.mean(all_levels, axis=0)
        avg_changes = np.mean([list(fc_s.values) for fc_s in future_fc.values()], axis=0)
        fc_rows['CIC Level (bn.)']    = avg_levels
        fc_rows['Daily Change (bn.)'] = avg_changes
        daily_full = pd.concat([df_lev, fc_rows])
    else:
        daily_full = df_lev.copy()

    # ── Tab 2: Monthly EOM ──
    eom_actual = df_lev['CIC Level (bn.)'].resample('ME').last().dropna()
    eom_change = eom_actual.diff()
    eom_df = pd.DataFrame({
        'CIC Level — Actual (bn.)': eom_actual,
        'Monthly Change — Actual (bn.)': eom_change,
    })
    eom_df.index.name = 'Date'

    # Add model EOM forecasts (historical rolling + next 2 months)
    for mname in model_order:
        eom_df[f'{BASE_LABELS.get(mname,mname)} — EOM Forecast (bn.)'] = np.nan
        eom_df[f'{BASE_LABELS.get(mname,mname)} — Monthly Chg Forecast (bn.)'] = np.nan

    # Next 2 months EOM forecasts
    if future_fc:
        for mname, fc_s in future_fc.items():
            lbl = BASE_LABELS.get(mname, mname)
            # Reconstruct cumulative level
            lvl = df_lev['CIC Level (bn.)'].iloc[-1]
            eom_forecasts = {}
            for dt, chg in zip(fc_s.index, fc_s.values):
                lvl += chg
                eom_forecasts[(dt.year, dt.month)] = lvl
            for (yr, mo), val in eom_forecasts.items():
                idx_ts = pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0)
                if idx_ts not in eom_df.index:
                    eom_df.loc[idx_ts, 'CIC Level — Actual (bn.)'] = np.nan
                    eom_df.loc[idx_ts, 'Monthly Change — Actual (bn.)'] = np.nan
                eom_df.loc[idx_ts, f'{lbl} — EOM Forecast (bn.)'] = val
                # Monthly change forecast = forecast EOM - previous actual EOM
                prev_eom = eom_actual.iloc[-1] if idx_ts not in eom_actual.index else eom_actual.get(idx_ts, np.nan)
                prev_idx = eom_df.index[eom_df.index < idx_ts]
                if len(prev_idx):
                    prev_lev = eom_df.loc[prev_idx[-1], f'{lbl} — EOM Forecast (bn.)']
                    if np.isnan(prev_lev):
                        prev_lev = eom_df.loc[prev_idx[-1], 'CIC Level — Actual (bn.)']
                    if not np.isnan(prev_lev):
                        eom_df.loc[idx_ts, f'{lbl} — Monthly Chg Forecast (bn.)'] = val - prev_lev
        eom_df = eom_df.sort_index()

    # Write to Excel
    daily_full.index.name = 'Date'
    eom_df.index.name = 'Date'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        daily_full.reset_index().to_excel(writer, sheet_name='Daily', index=False)
        eom_df.reset_index().to_excel(writer, sheet_name='Monthly EOM', index=False)

    # Apply styling with openpyxl
    wb = load_workbook(path)

    def style_sheet(ws, forecast_start_row, n_cols):
        thin = Side(style='thin', color='CCCCCC')
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        # Header row
        for col in range(1, n_cols + 1):
            cell = ws.cell(row=1, column=col)
            cell.fill   = HEADER
            cell.font   = HDR_FONT
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
        # Data rows
        for row in ws.iter_rows(min_row=2):
            is_fc = row[0].row >= forecast_start_row
            for cell in row:
                if is_fc:
                    cell.fill = YELLOW
                cell.border = border
                if cell.column == 1:
                    cell.number_format = 'YYYY-MM-DD'
                elif cell.value is not None and isinstance(cell.value, (int, float)):
                    cell.number_format = '#,##0.00'
        # Auto-width
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 30)

    # Daily sheet: forecast rows start after last actual
    ws_d = wb['Daily']
    daily_full.index.name = 'Date'
    daily_reset = daily_full.reset_index()
    date_col = daily_reset.columns[0]  # robust to index name
    fc_start_row_daily = 2 + int((daily_reset[date_col] <= last_actual).sum())
    style_sheet(ws_d, fc_start_row_daily, daily_reset.shape[1])

    # Monthly EOM sheet: forecast rows = months after last actual EOM
    ws_m = wb['Monthly EOM']
    eom_df.index.name = 'Date'
    eom_reset = eom_df.reset_index()
    date_col_eom = eom_reset.columns[0]
    fc_start_row_eom = 2 + int((eom_reset[date_col_eom] <= last_actual).sum())
    style_sheet(ws_m, fc_start_row_eom, eom_reset.shape[1])

    wb.save(path)
    print(f'  Saved → ./CIC_output.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_excel(df, configs_results, rolling_metrics, h_rmse, garch_res,
                 eom_results=None, save_dir='.'):
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

            sheet_name = 'Eval_Benchmark' if cfg_key == 'cfg_benchmark' else 'Eval_Main'
            out.to_excel(writer, sheet_name=sheet_name, index=False)

        # ── In-sample fitted values (main config) ──
        for ck in ['cfg_main', 'cfg_benchmark']:
            if ck not in configs_results:
                continue
            cdata   = configs_results[ck]
            df_tr   = cdata['df_train']
            lbl     = cdata['train_label']
            isfit   = pd.DataFrame({'Date': df_tr.index,
                                    'CIC_Level': df_tr['Currency'].values,
                                    'Change_Actual': df_tr['Change'].values})
            for mname, mdl in cdata['fitted_models'].items():
                fitted_chg = np.asarray(mdl.fitted)
                n_fit = min(len(fitted_chg), len(df_tr))
                col   = np.full(len(df_tr), np.nan)
                col[:n_fit] = fitted_chg[:n_fit]
                col_label = model_label(mname, lbl) if mname != 'D1' else f'D1 ({lbl})'
                isfit[f'{col_label}_Fitted'] = col
            sname = 'InSample_Main' if ck == 'cfg_main' else 'InSample_Benchmark'
            isfit.to_excel(writer, sheet_name=sname, index=False)
            break  # only write one

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

        # ── Level EOM Metrics (Model D backtest) ──
        if eom_results:
            eom_rows = []
            model_order = ['Old_2022', 'D1', 'D2_smooth']
            for k in model_order:
                r = eom_results.get(k, {})
                if not len(r.get('dates', [])):
                    continue
                dates  = r['dates']
                actual = r['actual']
                fc     = r['forecast']
                errs   = r['errors']
                years  = sorted(set(dates.year))
                base_row = {'Model': k, 'Overall_RMSE': round(r['RMSE'], 3), 'n': len(errs)}
                for yr in years:
                    mask = dates.year == yr
                    if mask.sum() > 0:
                        base_row[f'RMSE_{yr}'] = round(
                            float(np.sqrt(np.mean(errs[mask] ** 2))), 3)
                eom_rows.append(base_row)
            pd.DataFrame(eom_rows).to_excel(writer, sheet_name='Level_EOM_Metrics', index=False)

            # Also write detail rows (date, actual, forecast, error per model)
            detail_frames = []
            for k in model_order:
                r = eom_results.get(k, {})
                if not len(r.get('dates', [])):
                    continue
                tmp = pd.DataFrame({
                    'Model':    k,
                    'Date':     r['dates'],
                    'Actual_EOM_Level':   r['actual'],
                    'Forecast_EOM_Level': r['forecast'],
                    'Error':              r['errors'],
                })
                detail_frames.append(tmp)
            if detail_frames:
                pd.concat(detail_frames).sort_values(['Date', 'Model']).to_excel(
                    writer, sheet_name='Level_EOM_Detail', index=False)

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

    # Config 1 — Benchmark (BOT 2022 paper reference, kept for Excel comparison only)
    CFG_BENCHMARK = {
        'key':         'cfg_benchmark',
        'train_label': '1997-2021',
        'train_end':   '2021-11-30',
        'eval_start':  '2021-12-01',
        'eval_end':    '2022-05-31',
    }

    # Config 2 — Main: train pre-COVID (1997-2019), OOS = Jan 2020 – latest data
    # ~20% of 7000 obs is OOS; covers COVID, post-COVID, and 2024-25 trend shift
    CFG_MAIN = {
        'key':         'cfg_main',
        'train_label': '1997-2019',
        'train_end':   '2019-12-31',
        'eval_start':  '2020-01-01',
        'eval_end':    '2026-05-31',   # will be clipped to last available date
    }

    # Expanding-window rolling backtest: 1-year OOS each window, covers full history
    BACKTEST_WINDOWS = [
        ('2018-12-31', '2019-01-01', '2019-12-31'),
        ('2019-12-31', '2020-01-01', '2020-12-31'),
        ('2020-12-31', '2021-01-01', '2021-12-31'),
        ('2021-12-31', '2022-01-01', '2022-12-31'),
        ('2022-12-31', '2023-01-01', '2023-12-31'),
        ('2023-12-31', '2024-01-01', '2024-12-31'),
        ('2024-12-31', '2025-01-01', '2025-12-31'),
    ]
    # Horizon RMSE origins: spread across the OOS period
    HORIZON_ORIGINS = ['2020-01-01', '2022-06-01', '2024-01-01']
    ALL_MODELS      = list(REGS.keys())
    CORE_MODELS     = ALL_MODELS    # include all 4 ARIMAX variants in rolling backtest

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
    sk_tr = int(df.loc[:CFG_MAIN['train_end'], 'D_SK_PRE1'].sum())
    ny_tr = int(df.loc[:CFG_MAIN['train_end'], 'D_NY_PRE1'].sum())
    print(f'    D_SK_PRE1 (to {CFG_MAIN["train_end"][:4]}): {sk_tr} events  |  D_NY_PRE1: {ny_tr} events')

    # ── 2. Fig 1 ──
    print('\n[2] Figure 1 — CIC overview...')
    plot_fig1_overview(df)

    # ── 3. Fit all models on both configs (ARIMAX + D1) ──
    print('\n[3] Fitting ARIMAX models on both training configurations...')
    print(f'  {"Model":<20} {"AIC":>10} {"BIC":>10} {"σ":>7} {"AR":>6} {"MA":>6}  [config]')
    print('  ' + '-' * 72)

    configs_results = {}
    for cfg in [CFG_BENCHMARK, CFG_MAIN]:
        key      = cfg['key']
        lbl      = cfg['train_label']
        df_train = df.loc[:cfg['train_end']]
        # Clip eval end to last available date
        eval_end_actual = min(cfg['eval_end'], df.index[-1].strftime('%Y-%m-%d'))
        df_eval  = df.loc[cfg['eval_start']:eval_end_actual]

        # ARIMAX variants
        fitted_models = {}
        for mname in ALL_MODELS:
            X_tr, _ = get_X(df_train, mname)
            mdl = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
            fitted_models[mname] = mdl
            print(f'  [{model_label(mname, lbl):<28}]  '
                  f'AIC={mdl.aic:9.1f}  BIC={mdl.bic:9.1f}  '
                  f'σ={mdl.sigma:.3f}  AR={mdl.ar1:.3f}  MA={mdl.ma1:.3f}')

        # D1 state-space variant
        X_tr_ss, _ = get_X(df_train, 'Old_2022')
        try:
            d1_mdl = StateSpaceTrendModel('D1').fit(df_train['Change'].values, X_tr_ss)
            fitted_models['D1'] = d1_mdl
            print(f'  [{"D1 ("+lbl+")":<28}]  AIC={d1_mdl.aic:9.1f}  BIC={d1_mdl.bic:9.1f}')
        except Exception as exc:
            print(f'  ⚠ D1 ({lbl}): {exc}')

        # Forecasts and metrics for eval period
        forecasts     = {}
        bench_metrics = {}
        actual_arr    = df_eval['Change'].values
        for mname, mdl in fitted_models.items():
            if mname == 'D1':
                X_ev, _ = get_X(df_eval, 'Old_2022')
            else:
                X_ev, _ = get_X(df_eval, mname)
            pred = mdl.forecast(X_ev)
            forecasts[mname]     = pred
            bench_metrics[mname] = compute_metrics(actual_arr, pred)

        configs_results[key] = {
            'train_label':   lbl,
            'eval_label':    f'{cfg["eval_start"][:7]}→{eval_end_actual[:7]}',
            'df_train':      df_train,
            'df_eval':       df_eval,
            'fitted_models': fitted_models,
            'forecasts':     forecasts,
            'bench_metrics': bench_metrics,
        }

    ALL_MODELS_INCL_D1 = ALL_MODELS + ['D1']

    # ── 4. Benchmark metrics ──
    print('\n[4] Benchmark metrics:')
    for cfg_key, cfg_data in configs_results.items():
        lbl = cfg_data['train_label']
        ev  = cfg_data['eval_label']
        print(f'\n  Config ({lbl}) — eval {ev}:')
        print(f'  {"Model":<32} {"RMSE":>8} {"MAE":>8} {"ResidSD":>10}')
        print('  ' + '-' * 60)
        for mname, m in cfg_data['bench_metrics'].items():
            lbl2 = model_label(mname, lbl) if mname != 'D1' else f'D1 ({lbl})'
            print(f'  {lbl2:<32} {m["RMSE"]:>8.3f} {m["MAE"]:>8.3f} {m["ResidSD"]:>10.3f}')
        if cfg_key == 'cfg_benchmark':
            print(f'  {"[BOT 2022 paper (2017-2021)]":<32} {"4.960":>8} {"---":>8} {"4.140":>10}  (published)')
            print(f'  {"[Pre-2022 model]":<32} {"7.310":>8} {"---":>8} {"4.750":>10}  (published)')

    # ── 5. Residual diagnostics (main config, ARIMAX only) ──
    print('\n[5] Residual diagnostics (cfg_main training residuals)...')
    m_fitted  = configs_results['cfg_main']['fitted_models']
    m_lbl     = configs_results['cfg_main']['train_label']
    residuals = {k: v.resid for k, v in m_fitted.items() if k in ALL_MODELS}
    for mname, res in residuals.items():
        run_diagnostics(res, label=model_label(mname, m_lbl))

    # ── 6. ARCH + GARCH ──
    print('\n[6] ARCH-LM + GARCH(1,1) on Old_2022 residuals...')
    old_res = np.asarray(residuals['Old_2022'], float)
    old_res = old_res[~np.isnan(old_res)]
    arch_stat, arch_pval, _, _ = het_arch(old_res, nlags=10)
    print(f'  ARCH-LM(10): stat={arch_stat:.3f}, p={arch_pval:.4f}  '
          f'→ {"⚠ ARCH effects present" if arch_pval<0.05 else "✓ no ARCH"}')
    garch_res = fit_garch(old_res)

    # ── 7. Rolling backtest (all 7 windows, all 4 ARIMAX models) ──
    print('\n[7] Rolling backtest (expanding window, 7 periods, all ARIMAX models)...')
    rolling_metrics = rolling_backtest(df, CORE_MODELS, BACKTEST_WINDOWS)
    win_labels = [f'{es[:7]}→{ee[:7]}' for _, es, ee in BACKTEST_WINDOWS]
    print('\n  Rolling RMSE summary:')
    w_fmt = ''.join(f'{w:>16}' for w in win_labels)
    print(f'  {"Model":<24}{w_fmt}')
    print('  ' + '-' * (24 + 16 * len(win_labels)))
    for mname in CORE_MODELS:
        row = f'  {BASE_LABELS[mname]:<24}'
        for wl in win_labels:
            val = rolling_metrics.get(mname, {}).get(wl, {}).get('RMSE', np.nan)
            row += f'{val:>16.3f}' if not np.isnan(val) else f'{"—":>16}'
        print(row)

    # ── 8. Horizon RMSE ──
    print('\n[8] Horizon RMSE (1, 5, 10, 22-day ahead, 3 origins) — full OOS period...')
    h_rmse = horizon_rmse_monthly(df, CORE_MODELS, HORIZON_ORIGINS)
    print('\n  Horizon RMSE:')
    print(f'  {"Model":<24} {"h=1":>8} {"h=5":>8} {"h=10":>8} {"h=22":>8}')
    print('  ' + '-' * 60)
    for mname in CORE_MODELS:
        row = f'  {BASE_LABELS[mname]:<24}'
        for h in [1, 5, 10, 22]:
            val = h_rmse.get(mname, {}).get(h, np.nan)
            row += f'  {val:>8.3f}' if not np.isnan(val) else f'  {"—":>8}'
        print(row)

    # ── 8b. D1 fit on main training data (for fig10 drift plot) ──
    print('\n[8b] Fitting D1 on cfg_main training data (for drift visualisation)...')
    m_df_train = configs_results['cfg_main']['df_train']
    X_tr_ss, _ = get_X(m_df_train, 'Old_2022')
    ss_d1 = m_fitted.get('D1')
    if ss_d1 is None:
        try:
            ss_d1 = StateSpaceTrendModel('D1').fit(m_df_train['Change'].values, X_tr_ss)
            print(f'  D1  AIC={ss_d1.aic:.1f}  BIC={ss_d1.bic:.1f}')
        except Exception as exc:
            print(f'  ⚠ D1: {exc}')

    # ── 8c. EOM level backtest — all 5 models, 2020–2025 ──
    print('\n[8c] EOM level backtest — all 5 models, rolling monthly 2020–2025...')
    print('     (One UC fit + 4 ARIMAX fits per month-origin — ~5 min total)')
    eom_results = month_end_eom_backtest(df, hol, start_year=2020, end_year=2025)

    # ── 9. Figures ──
    print('\n[9] Generating figures...')

    # cfg_main is the primary config for all comparison figures
    m_data = configs_results['cfg_main']
    b_data = configs_results['cfg_benchmark']

    # fig2 — Actual vs forecast; use cfg_main eval (2020–2026) for full picture
    plot_fig2_actual_vs_forecast(m_data['df_eval'], m_data['forecasts'], m_data['train_label'])

    # fig3 — Error distributions; zoom to last 2 years for readability
    zoom_start = (m_data['df_eval'].index[-1] - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    df_zoom    = m_data['df_eval'].loc[zoom_start:]
    fc_zoom    = {k: v[-len(df_zoom):] for k, v in m_data['forecasts'].items()}
    plot_fig3_errors(df_zoom, fc_zoom, m_data['train_label'])

    # fig4 — Residual ACF/QQ on cfg_main training residuals (ARIMAX only)
    plot_fig4_residuals(residuals, m_lbl)

    # fig5 — RMSE comparison bars: cfg_main only + rolling backtest
    all_bench = []
    for mname, m in configs_results['cfg_main']['bench_metrics'].items():
        all_bench.append((BASE_LABELS.get(mname, mname), m['RMSE'], COLORS.get(mname, '#aaaaaa')))
    eval_period_label = 'Train: 1997–2019  |  OOS: Jan 2020 – May 2026'
    plot_fig5_rmse_comparison(all_bench, rolling_metrics, eval_period_label=eval_period_label)

    plot_fig6_horizon(h_rmse, m_data['train_label'])

    # fig7 — Monthly monitor; zoom to last 2 years
    plot_fig7_monthly_monitor(df_zoom, fc_zoom, m_data['train_label'])

    plot_fig8_garch(m_data['df_train'].index, old_res, garch_res)

    print('  Generating fig9 (seasonal CIC + 3-model fan chart)...')
    fan_models_dict = {k: m_fitted[k] for k in ('Old_2022', 'ExtDummy', 'D1') if k in m_fitted}
    plot_fig9_seasonal_cic(df, fan_models_dict, hol)

    # fig10 — D1 adaptive drift
    if ss_d1 is not None:
        print('  Generating fig10 (Model D1 adaptive drift)...')
        plot_fig10_trend_slope(m_df_train, ss_d1)
    else:
        print('  ⚠ Skipping fig10 — D1 failed to fit.')

    # fig11 — EOM level comparison, all 5 models
    print('  Generating fig11 (EOM level comparison, all 5 models)...')
    plot_fig11_eom_level(eom_results)

    # ── 10. Excel ──
    print('\n[10] Exporting Excel output...')
    export_excel(df, configs_results, rolling_metrics, h_rmse, garch_res, eom_results)

    # Clean user-facing workbook
    print('  Exporting CIC_output.xlsx (daily + monthly EOM, 3 models)...')
    export_cic_output_excel(df, m_fitted, hol)

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
        for mname in ALL_MODELS_INCL_D1:
            if mname not in bm:
                continue
            r      = bm[mname]['RMSE']
            tag    = ' ← best' if mname == best else ''
            d_old  = r - old_rmse
            d_pap  = r - paper_rmse if cfg_key == 'cfg_benchmark' else float('nan')
            s_old  = '+' if d_old >= 0 else ''
            s_pap  = '+' if not np.isnan(d_pap) and d_pap >= 0 else ''
            d_pap_str = f'{s_pap}{d_pap:+.3f}' if not np.isnan(d_pap) else '      —'
            lbl2   = model_label(mname, lbl) if mname != 'D1' else f'D1 ({lbl})'
            print(f'  {lbl2:<36} {r:>6.3f}  {s_old}{d_old:>10.3f}  {d_pap_str:>13}{tag}')
        if cfg_key == 'cfg_benchmark':
            print(f'  {"[BOT 2022 paper (2017-2021)]":<36} {"4.960":>6}  {"baseline":>11}  {"0.000":>13}')

    # EOM level RMSE summary
    print(f'\n  ── EOM Level RMSE (primary KPI — 1-month-ahead, 2020–2025) ──')
    eom_models = ['Old_2022', 'ExtDummy', 'Regime', 'Fourier_Regime', 'D1']
    print(f'  {"Model":<20}  {"2024–25 RMSE":>14}  {"Overall RMSE":>14}')
    print('  ' + '-' * 52)
    old24 = None
    for k in eom_models:
        r = eom_results.get(k, {})
        if not len(r.get('dates', [])):
            continue
        mask24   = r['dates'].year >= 2024
        rmse24   = np.sqrt(np.mean(r['errors'][mask24] ** 2)) if mask24.sum() > 0 else np.nan
        rmse_all = r['RMSE']
        if k == 'Old_2022':
            old24 = rmse24
        tag = ''
        if k != 'Old_2022' and old24 is not None and not np.isnan(rmse24):
            tag = '  ← better' if rmse24 < old24 else ''
        print(f'  {k:<20}  {rmse24:>14.3f}  {rmse_all:>14.3f}{tag}')

    print(f'\n  All figures and cic_forecast_output.xlsx saved to: {os.path.abspath(".")}')
    print(sep + '\n')


if __name__ == '__main__':
    main()
