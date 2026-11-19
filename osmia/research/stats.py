"""Trade statistics with honest uncertainty.

Trades opened on the same day share market conditions, so they are not
independent. Confidence intervals therefore use a *day-block bootstrap*:
resample whole days of trades with replacement, not individual trades. This
gives wider (more honest) intervals than a naive per-trade bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 2000


def _by_day(trades: list[dict]) -> list[np.ndarray]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    day = pd.to_datetime(df['entry_ts'], utc=True).dt.floor('D')
    return [g['ret_pct'].to_numpy(float) for _, g in df.groupby(day)]


def _boot_means(blocks: list[np.ndarray], rng, n_boot: int) -> np.ndarray:
    k = len(blocks)
    sums = np.array([b.sum() for b in blocks])
    cnts = np.array([len(b) for b in blocks])
    idx = rng.integers(0, k, size=(n_boot, k))
    return sums[idx].sum(axis=1) / np.maximum(cnts[idx].sum(axis=1), 1)


def summarize(trades: list[dict], seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Mean net return per trade (%), win rate, total, and a 95% day-block CI."""
    rets = np.array([t['ret_pct'] for t in trades], dtype=float)
    out = {'n': int(len(rets)), 'days': 0, 'mean_pct': None, 'median_pct': None,
           'win_rate': None, 'total_pct': 0.0, 'ci_low': None, 'ci_high': None,
           'avg_hold_hours': None}
    if len(rets) == 0:
        return out
    blocks = _by_day(trades)
    out.update({
        'days': len(blocks),
        'mean_pct': round(float(rets.mean()), 4),
        'median_pct': round(float(np.median(rets)), 4),
        'win_rate': round(float((rets > 0).mean()), 4),
        'total_pct': round(float(rets.sum()), 4),
        'avg_hold_hours': round(float(np.mean([t['hold_hours'] for t in trades])), 2),
    })
    if len(blocks) >= 2:
        boots = _boot_means(blocks, np.random.default_rng(seed), n_boot)
        out['ci_low'], out['ci_high'] = (round(float(x), 4) for x in np.percentile(boots, [2.5, 97.5]))
    return out


def compare(strategy: list[dict], baseline: list[dict], seed: int = 0,
            n_boot: int = N_BOOT) -> dict:
    """Difference in mean net return per trade (strategy - baseline) with a
    95% CI from independent day-block bootstraps of each side."""
    a, b = _by_day(strategy), _by_day(baseline)
    if len(a) < 2 or len(b) < 2:
        return {'diff_pct': None, 'ci_low': None, 'ci_high': None}
    rng = np.random.default_rng(seed)
    da = _boot_means(a, rng, n_boot)
    db = _boot_means(b, rng, n_boot)
    diff = da - db
    ma = np.mean([t['ret_pct'] for t in strategy])
    mb = np.mean([t['ret_pct'] for t in baseline])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {'diff_pct': round(float(ma - mb), 4), 'ci_low': round(float(lo), 4),
            'ci_high': round(float(hi), 4)}


def equity_curve(trades: list[dict]) -> list[dict]:
    """Cumulative net return (sum of per-trade %), one point per exit."""
    pts, total = [], 0.0
    for t in sorted(trades, key=lambda t: t['exit_ts']):
        total += t['ret_pct']
        pts.append({'ts': pd.Timestamp(t['exit_ts']).isoformat(), 'cum_pct': round(total, 4)})
    return pts


def verdict(holdout: dict, vs_random: dict, min_trades: int, min_days: int = 3) -> str:
    """Pre-registered decision rule.
    accepted     enough trades/days AND the 95% CI of (strategy - random) is above 0
                 AND the strategy's own mean net return is positive
    rejected     enough data AND that CI's upper bound is <= 0
    inconclusive anything else (too few trades, or the CI straddles 0)"""
    if holdout['n'] < min_trades or holdout['days'] < min_days or vs_random['ci_low'] is None:
        return 'inconclusive'
    if vs_random['ci_low'] > 0 and (holdout['mean_pct'] or 0) > 0:
        return 'accepted'
    if vs_random['ci_high'] <= 0:
        return 'rejected'
    return 'inconclusive'
