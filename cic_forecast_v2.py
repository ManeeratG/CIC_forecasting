#!/usr/bin/env python3
"""
CIC Forecasting v2 — Bank of Thailand
Three-bet accuracy program targeting the primary KPI:
1-month-ahead end-of-month (EOM) CIC level RMSE.

Candidates (see contextv2.md for the diagnosis and experiment log):
  Bet 1  M1  — direct monthly EOM model (SARIMA / UC, 12-month seasonality)
  Bet 2  D2  — smooth-trend UC on the calendar-adjusted CIC level, COVID-robust
  Bet 3  CMB — inverse-MSE combination of {Old_2022, M1, D2}

Baselines run in the same harness for continuity: Old_2022 (TwoStepARIMAX)
and D1 (local-level state-space) imported from cic_forecast.py.

Protocol:
  origins   = calendar month-ends 2017-12-31 → 2026-03-31 (targets 2018-01 → 2026-04)
  selection = targets 2018-01 → 2023-12   (all spec choices made here)
  holdout   = targets 2024-01 → 2026-04   (run once; decides the winner)
  Gate 0    = harness must reproduce v1's Old_2022 = 32.86 / D1 = 33.04
              on the 2020–2025 origin range before results are trusted.

Usage:
  python cic_forecast_v2.py --gate0            # Gate-0 reproduction check only
  python cic_forecast_v2.py                    # full run (all models, all origins)
  python cic_forecast_v2.py --models old,m1_sarima,d2_smooth
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.linear_model import LinearRegression

# Reuse the v1 data pipeline and baseline models unchanged.
from cic_forecast import (
    load_data, load_holiday, REGS, get_X, generate_future_exog,
    TwoStepARIMAX, StateSpaceTrendModel,
)

CACHE_DIR   = 'backtest_cache'
COVID_START = pd.Timestamp('2020-03-01')
COVID_END   = pd.Timestamp('2020-12-31')

SELECTION_END  = pd.Timestamp('2023-12-31')   # last target month of selection window
FIRST_ORIGIN   = '2017-12-31'
LAST_ORIGIN    = '2026-03-31'                 # targets April 2026 (last complete month)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MODELS
# ─────────────────────────────────────────────────────────────────────────────
# Every model implements:
#   fit_forecast(df_train, X_fut_df, last_level, target_month) ->
#       {'eom_fc': float, 'daily_fc': np.ndarray | None}
# df_train      : full history up to the origin (expanding window)
# X_fut_df      : Old_2022 dummy matrix for the target month's trading days
# last_level    : CIC level at the origin (L_M)
# target_month  : pd.Period of month M+1

class OldModel:
    """Baseline: v1 Old_2022 two-step ARIMAX (OLS on 55 dummies + ARIMA(1,0,1))."""
    key = 'Old_2022'

    def fit_forecast(self, df_train, X_fut_df, last_level, target_month):
        X_tr, _ = get_X(df_train, 'Old_2022')
        mdl = TwoStepARIMAX().fit(df_train['Change'].values, X_tr)
        fc  = np.asarray(mdl.forecast(X_fut_df.values), float)
        return {'eom_fc': last_level + float(fc.sum()), 'daily_fc': fc}


class D1Model:
    """Baseline: v1 D1 local-level state-space (adaptive drift on ΔCIC residuals)."""
    key = 'D1'

    def fit_forecast(self, df_train, X_fut_df, last_level, target_month):
        X_tr, _ = get_X(df_train, 'Old_2022')
        mdl = StateSpaceTrendModel('D1').fit(df_train['Change'].values, X_tr)
        fc  = np.asarray(mdl.forecast(X_fut_df.values), float)
        return {'eom_fc': last_level + float(fc.sum()), 'daily_fc': fc}


def _monthly_eom_series(df_train):
    """Month-end CIC level series: last observed level per calendar month."""
    lev = df_train['Currency'].dropna()
    eom = lev.groupby(lev.index.to_period('M')).last()
    eom.index = eom.index.to_timestamp('M')
    return eom


class M1Sarima:
    """Bet 1a: SARIMA(1,1,1)(0,1,1,12) on the monthly EOM level with
    2020-03 / 2020-04 intervention dummies (COVID cash-hoarding pulse)."""
    key = 'M1_SARIMA'

    def fit_forecast(self, df_train, X_fut_df, last_level, target_month):
        eom = _monthly_eom_series(df_train)
        exog = pd.DataFrame(index=eom.index)
        exog['covid_mar20'] = (eom.index.to_period('M') == pd.Period('2020-03')).astype(float)
        exog['covid_apr20'] = (eom.index.to_period('M') == pd.Period('2020-04')).astype(float)
        mdl = SARIMAX(eom, exog=exog, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=300)
        exog_f = pd.DataFrame({'covid_mar20': [0.0], 'covid_apr20': [0.0]})
        fc = float(mdl.forecast(steps=1, exog=exog_f).iloc[0])
        return {'eom_fc': fc, 'daily_fc': None}


class M1UC:
    """Bet 1b: UC local-linear-trend + 12-month seasonal on the monthly EOM
    level, with COVID months (2020-03 → 2020-12) masked as missing."""
    key = 'M1_UC'

    def fit_forecast(self, df_train, X_fut_df, last_level, target_month):
        eom = _monthly_eom_series(df_train).astype(float)
        masked = eom.copy()
        masked[(masked.index >= COVID_START) & (masked.index <= COVID_END)] = np.nan
        mdl = UnobservedComponents(masked, level='local linear trend', seasonal=12
                                   ).fit(disp=False, method='bfgs', maxiter=300)
        fc = float(np.asarray(mdl.get_forecast(steps=1).predicted_mean)[0])
        return {'eom_fc': fc, 'daily_fc': None}


class D2Model:
    """Bet 2: smooth-trend UC on the calendar-adjusted CIC level.

    Step 1: full-sample OLS of ΔCIC on the Old_2022 dummy matrix (intercept
            included) → calendar mean m_t; cumulate residuals into the
            calendar-adjusted level C_t = Σ (ΔCIC_s − m_s).
    Step 2: UnobservedComponents(C_t, level=trend_spec, autoregressive=1),
            optionally masking COVID (2020-03 → 2020-12) as missing so the
            trend variances are not MLE'd off the hoarding shock.
    EOM forecast = L_M + Σ future m + (Ĉ_terminal − C_M).
    Daily path   = m_{t} + ΔĈ_{t} (for the daily-RMSE guardrail).
    """

    def __init__(self, trend='smooth trend', covid_nan=True, key=None):
        self.trend     = trend
        self.covid_nan = covid_nan
        self.key       = key or f"D2_{'smooth' if trend == 'smooth trend' else 'lltrend'}" \
                                + ('' if covid_nan else '_nocovidmask')

    def fit_forecast(self, df_train, X_fut_df, last_level, target_month):
        y = df_train['Change'].values.astype(float)
        X_tr, _ = get_X(df_train, 'Old_2022')
        ols = LinearRegression(fit_intercept=True).fit(X_tr, y)
        resid = y - ols.predict(X_tr)
        cum = pd.Series(np.cumsum(resid), index=df_train.index)
        endog = cum.copy()
        if self.covid_nan:
            endog[(endog.index >= COVID_START) & (endog.index <= COVID_END)] = np.nan
        mdl = UnobservedComponents(endog.values, level=self.trend, autoregressive=1
                                   ).fit(disp=False, method='bfgs', maxiter=300)
        n_fc  = len(X_fut_df)
        c_hat = np.asarray(mdl.get_forecast(steps=n_fc).predicted_mean, float)
        m_fut = ols.predict(X_fut_df.values)
        c_last = float(cum.iloc[-1])
        daily = m_fut + np.diff(np.concatenate([[c_last], c_hat]))
        eom_fc = last_level + float(m_fut.sum()) + (float(c_hat[-1]) - c_last)
        return {'eom_fc': eom_fc, 'daily_fc': daily}


MODEL_FACTORY = {
    'old':        OldModel,
    'd1':         D1Model,
    'm1_sarima':  M1Sarima,
    'm1_uc':      M1UC,
    'd2_smooth':  lambda: D2Model(trend='smooth trend', covid_nan=True),
    'd2_lltrend': lambda: D2Model(trend='local linear trend', covid_nan=True),
    'd2_smooth_nomask': lambda: D2Model(trend='smooth trend', covid_nan=False,
                                        key='D2_smooth_nomask'),
}
DEFAULT_MODELS = ['old', 'd1', 'm1_sarima', 'm1_uc', 'd2_smooth', 'd2_lltrend',
                  'd2_smooth_nomask']


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EOM BACKTEST HARNESS (expanding window, per-origin cache)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(model_key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f'{model_key}.pkl')


def _load_cache(model_key):
    p = _cache_path(model_key)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    return {}


def _save_cache(model_key, cache):
    with open(_cache_path(model_key), 'wb') as f:
        pickle.dump(cache, f)


def eom_backtest(df, hol, model_names, first_origin=FIRST_ORIGIN,
                 last_origin=LAST_ORIGIN, use_cache=True, verbose=True):
    """
    For each origin = calendar month-end M with data available:
      1. train = df up to origin (expanding window), L_M = last observed level
      2. build the Old_2022 dummy matrix for M+1's trading days
         (generate_future_exog — same call as v1's month_end_eom_backtest)
      3. each model forecasts the EOM level of M+1 (daily models: L_M + Σ ΔCIĈ)
      4. error = actual EOM(M+1) − forecast
    Returns {model_key: DataFrame[target, actual, forecast, error, ...]} plus
    a parallel dict of daily paths for the daily-RMSE guardrail.
    """
    origins = pd.date_range(first_origin, last_origin, freq='ME')
    results = {}
    daily_store = {}

    for name in model_names:
        factory = MODEL_FACTORY[name]
        model = factory()
        cache = _load_cache(model.key) if use_cache else {}
        rows, dirty = [], False
        daily_store[model.key] = {}

        for origin in origins:
            avail = df.index[df.index <= origin]
            if len(avail) < 500:
                continue
            df_train = df.loc[:origin]

            nm_start = origin + pd.offsets.MonthBegin(1)
            nm_end   = nm_start + pd.offsets.MonthEnd(0)
            df_next  = df.loc[nm_start:nm_end]
            if len(df_next) < 5:
                continue
            lev_next = df_next['Currency'].dropna()
            if len(lev_next) == 0:
                continue
            actual_eom = float(lev_next.iloc[-1])
            lev_hist = df_train['Currency'].dropna()
            if len(lev_hist) == 0:
                continue
            last_level = float(lev_hist.iloc[-1])

            okey = origin.strftime('%Y-%m-%d')
            if okey in cache:
                out = cache[okey]
            else:
                fc_start = df_next.index[0].strftime('%Y-%m-%d')
                fc_end   = df_next.index[-1].strftime('%Y-%m-%d')
                try:
                    X_fut_df = generate_future_exog('Old_2022', fc_start, fc_end, hol)
                    if len(X_fut_df) == 0:
                        continue
                    out = model.fit_forecast(df_train, X_fut_df, last_level,
                                             nm_start.to_period('M'))
                    out['daily_dates'] = X_fut_df.index
                except Exception as exc:
                    print(f'    ⚠ {model.key} @ {okey}: {exc}')
                    continue
                cache[okey] = out
                dirty = True
                if verbose:
                    print(f'    {model.key:<18} {okey}  EOM fc={out["eom_fc"]:.1f} '
                          f'actual={actual_eom:.1f}', flush=True)

            rows.append({'origin': origin, 'target': nm_end,
                         'actual': actual_eom, 'forecast': out['eom_fc'],
                         'last_level': last_level})
            if out.get('daily_fc') is not None:
                daily_store[model.key][nm_end] = pd.Series(
                    np.asarray(out['daily_fc'], float), index=out['daily_dates'])

        if dirty and use_cache:
            _save_cache(model.key, cache)

        res = pd.DataFrame(rows).set_index('target')
        res['error'] = res['actual'] - res['forecast']
        results[model.key] = res
        if verbose:
            print(f'  {model.key:<18} n={len(res)}  overall EOM RMSE='
                  f'{np.sqrt((res["error"] ** 2).mean()):.3f}', flush=True)

    return results, daily_store


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BET 3: FORECAST COMBINATION (post-processing on cached results)
# ─────────────────────────────────────────────────────────────────────────────

def combine_forecasts(results, member_keys, key=None, window=12, floor=0.1,
                      min_history=6):
    """
    Inverse-MSE combination of EOM forecasts, leakage-free:
    weights at origin M use only errors of targets < M's target month, over the
    trailing `window` months; floored at `floor` and renormalised.
    Before `min_history` common errors exist, fall back to equal weights.
    """
    key = key or 'CMB_' + '+'.join(member_keys)
    common = None
    for k in member_keys:
        idx = results[k].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()

    rows = []
    for t in common:
        fcs, ws = [], []
        for k in member_keys:
            r = results[k]
            past = r.loc[r.index < t, 'error'].tail(window)
            fcs.append(r.loc[t, 'forecast'])
            if len(past) >= min_history:
                ws.append(1.0 / max(float((past ** 2).mean()), 1e-9))
            else:
                ws.append(np.nan)
        ws = np.array(ws, float)
        if np.isnan(ws).any():
            ws = np.ones(len(member_keys))
        ws = ws / ws.sum()
        ws = np.maximum(ws, floor)
        ws = ws / ws.sum()
        fc = float(np.dot(ws, fcs))
        r0 = results[member_keys[0]].loc[t]
        rows.append({'target': t, 'origin': r0['origin'], 'actual': r0['actual'],
                     'forecast': fc, 'last_level': r0['last_level'],
                     **{f'w_{k}': w for k, w in zip(member_keys, ws)}})
    res = pd.DataFrame(rows).set_index('target')
    res['error'] = res['actual'] - res['forecast']
    return key, res


def equal_weight(results, member_keys, key=None):
    key = key or 'AVG_' + '+'.join(member_keys)
    common = None
    for k in member_keys:
        idx = results[k].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    fc = sum(results[k].loc[common, 'forecast'] for k in member_keys) / len(member_keys)
    r0 = results[member_keys[0]].loc[common]
    res = pd.DataFrame({'origin': r0['origin'], 'actual': r0['actual'],
                        'forecast': fc, 'last_level': r0['last_level']})
    res['error'] = res['actual'] - res['forecast']
    return key, res


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — METRICS, DM TEST, GUARDRAIL
# ─────────────────────────────────────────────────────────────────────────────

def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def summarize(results, lo=None, hi=None, label=''):
    """Per-model overall + per-year EOM RMSE over targets in [lo, hi]."""
    rows = []
    for k, r in results.items():
        m = r
        if lo is not None:
            m = m[m.index >= lo]
        if hi is not None:
            m = m[m.index <= hi]
        if len(m) == 0:
            continue
        row = {'Model': k, 'n': len(m), 'RMSE': rmse(m['error']),
               'MAE': float(m['error'].abs().mean()),
               'Bias': float(m['error'].mean())}
        for yr in sorted(m.index.year.unique()):
            row[f'RMSE_{yr}'] = rmse(m.loc[m.index.year == yr, 'error'])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index('Model').sort_values('RMSE')
    if label:
        print(f'\n=== {label} ===')
        print(out.round(3).to_string())
    return out


def dm_test(e1, e2):
    """Diebold–Mariano test on squared errors (h=1), Harvey small-sample corr.
    Negative stat → model 1 more accurate. Returns (stat, p_value)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dbar = d.mean()
    var = d.var(ddof=1) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    harvey = np.sqrt((n + 1 - 2 * 1 + 1 * (1 - 1) / n) / n)  # h=1
    stat = dm * harvey
    p = 2 * (1 - sps.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def daily_guardrail(df, daily_store, baseline_key='Old_2022', lo=None, hi=None):
    """Per-year daily ΔCIC RMSE per model (only models with daily paths)."""
    rows = {}
    for k, paths in daily_store.items():
        if not paths:
            continue
        err = []
        for t, fc in paths.items():
            if lo is not None and t < lo:
                continue
            if hi is not None and t > hi:
                continue
            act = df['Change'].reindex(fc.index)
            e = (act - fc).dropna()
            err.append(e)
        if not err:
            continue
        e = pd.concat(err).sort_index()
        rows[k] = {f'{yr}': rmse(e[e.index.year == yr]) for yr in sorted(e.index.year.unique())}
        rows[k]['Overall'] = rmse(e)
    out = pd.DataFrame(rows).T
    print('\n=== Daily ΔCIC RMSE guardrail (per target-month days) ===')
    print(out.round(3).to_string())
    if baseline_key in out.index:
        base = out.loc[baseline_key]
        viol = out[out.gt(base * 1.05).any(axis=1)].index.difference([baseline_key])
        if len(viol):
            print(f'  ⚠ guardrail exceeded (+5% vs {baseline_key} in ≥1 year): {list(viol)}')
        else:
            print('  ✓ all models within +5% of baseline in every year')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def reconciled_daily_path(daily_store, base_key, base_eom, target_eom, target):
    """Production daily path for an EOM-only forecast (M1 or a combination):
    take the daily-model's within-month calendar shape and shift it by the
    constant per-day amount needed to hit the reconciled EOM level."""
    base = daily_store[base_key][target]
    shift = (target_eom - base_eom) / len(base)
    return base + shift


def export_results(results, sel_tbl, hold_tbl, guard_tbl, path='cic_v2_results.xlsx'):
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        sel_tbl.round(3).to_excel(xw, sheet_name='EOM_Selection')
        hold_tbl.round(3).to_excel(xw, sheet_name='EOM_Holdout')
        detail = []
        for k, r in results.items():
            d = r.reset_index()
            d.insert(0, 'Model', k)
            detail.append(d)
        pd.concat(detail).round(3).to_excel(xw, sheet_name='EOM_Detail', index=False)
        if guard_tbl is not None:
            guard_tbl.round(3).to_excel(xw, sheet_name='Daily_Guardrail')
    print(f'\n  Results exported → {path}')


def plot_eom(results, keys, path='fig_v2_eom_level.png'):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    r0 = results[keys[0]]
    axes[0].plot(r0.index, r0['actual'], 'k-', lw=2, label='Actual EOM level')
    for k in keys:
        r = results[k]
        axes[0].plot(r.index, r['forecast'], lw=1.2, alpha=0.85, label=k)
        axes[1].plot(r.index, r['error'], lw=1.2, alpha=0.85, label=k)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].axvline(SELECTION_END, color='grey', ls='--', lw=1)
    axes[1].text(SELECTION_END, axes[1].get_ylim()[1] * 0.9, ' holdout →',
                 color='grey', fontsize=9)
    axes[0].set_title('1-month-ahead end-of-month CIC level — actual vs forecast')
    axes[1].set_title('EOM forecast error (THB bn)')
    for ax in axes:
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Figure saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

def gate0(df, hol, use_cache=True):
    """Reproduce v1's month_end_eom_backtest numbers: Old_2022 = 32.860,
    D1 = 33.041 over origins 2019-12-31 → 2024-12-31 (targets 2020-01 → 2025-01)."""
    print('\n──── GATE 0: v1 reproduction (origins 2019-12 → 2024-12) ────')
    results, _ = eom_backtest(df, hol, ['old', 'd1'],
                              first_origin='2019-12-31', last_origin='2024-12-31',
                              use_cache=use_cache, verbose=True)
    ok = True
    for k, ref in [('Old_2022', 32.860), ('D1', 33.041)]:
        got = rmse(results[k]['error'])
        match = abs(got - ref) < 0.15
        ok &= match
        print(f'  {k:<10} v2 harness: {got:.3f}   v1 reference: {ref:.3f}   '
              f'{"✓ match" if match else "✗ MISMATCH"}')
    print('  GATE 0 ' + ('PASSED' if ok else 'FAILED — investigate before trusting results'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS),
                    help='comma list from: ' + ','.join(MODEL_FACTORY))
    ap.add_argument('--gate0', action='store_true', help='run Gate-0 check only')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--first-origin', default=FIRST_ORIGIN)
    ap.add_argument('--last-origin', default=LAST_ORIGIN)
    args = ap.parse_args()
    use_cache = not args.no_cache

    print('Loading data…', flush=True)
    df  = load_data('input.xlsx')
    hol = load_holiday('input.xlsx')
    print(f'  {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}')

    if args.gate0:
        gate0(df, hol, use_cache=use_cache)
        return

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    print(f'\n──── EOM backtest: {model_names} '
          f'(origins {args.first_origin} → {args.last_origin}) ────', flush=True)
    results, daily_store = eom_backtest(df, hol, model_names,
                                        first_origin=args.first_origin,
                                        last_origin=args.last_origin,
                                        use_cache=use_cache)

    # Bet 3 — combinations (selection window decides members; defaults below)
    have = set(results)
    if {'Old_2022', 'M1_SARIMA', 'D2_smooth'} <= have:
        k, r = combine_forecasts(results, ['Old_2022', 'M1_SARIMA', 'D2_smooth'],
                                 key='CMB_invMSE')
        results[k] = r
        k, r = equal_weight(results, ['Old_2022', 'M1_SARIMA', 'D2_smooth'],
                            key='AVG_equal')
        results[k] = r
        for pair in [('Old_2022', 'M1_SARIMA'), ('Old_2022', 'D2_smooth'),
                     ('M1_SARIMA', 'D2_smooth')]:
            k, r = combine_forecasts(results, list(pair))
            results[k] = r

    sel_tbl  = summarize(results, hi=SELECTION_END,
                         label='SELECTION window (targets 2018-01 → 2023-12)')
    hold_tbl = summarize(results, lo=SELECTION_END + pd.Timedelta(days=1),
                         label='HOLDOUT window (targets 2024-01 → 2026-04)')

    # DM tests vs Old_2022 on the holdout
    if 'Old_2022' in results:
        print('\n=== Diebold–Mariano vs Old_2022 (holdout, squared errors) ===')
        base = results['Old_2022']
        base_h = base[base.index > SELECTION_END]
        for k, r in results.items():
            if k == 'Old_2022':
                continue
            rh = r[r.index > SELECTION_END]
            common = base_h.index.intersection(rh.index)
            if len(common) < 8:
                continue
            stat, p = dm_test(rh.loc[common, 'error'], base_h.loc[common, 'error'])
            print(f'  {k:<24} DM={stat:+.2f}  p={p:.3f}  '
                  f'({"better" if stat < 0 else "worse"} than baseline)')

    guard_tbl = daily_guardrail(df, daily_store)

    export_results(results, sel_tbl, hold_tbl, guard_tbl)
    plot_eom(results, [k for k in results if k in
                       ('Old_2022', 'D1', 'M1_SARIMA', 'D2_smooth',
                        'CMB_invMSE', 'CMB_Old_2022+M1_SARIMA')])


if __name__ == '__main__':
    main()
