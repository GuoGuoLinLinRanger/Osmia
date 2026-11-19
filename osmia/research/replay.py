"""No-lookahead replay of pair mean-reversion strategies over stored prices.

A strategy is a small, bounded StrategySpec. At every decision step t the
replay only uses bars <= t: it computes a robust (median/MAD) z-score of each
pair's log price ratio over a trailing window, manages open positions, then
opens at most one new position chosen by the strategy's selector:

  rule    the most dislocated candidate (lowest z)  -- what the live system does
  random  a random tradable pair at the same moments -- the null baseline
  llm     a callable (e.g. a cached LLM) choosing among the candidates

Returns are net of round-trip taker fees and an assumed spread, measured on
the ratio (sell QUOTE, buy BASE), i.e. the same thing trade_pnl measures live.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import BINANCE_TAKER_FEE, MAD_CONSISTENCY, Z_SCORE_CLAMP
from ..core import trade_route

ASSUMED_SPREAD_PCT = 0.05      # per round trip, on top of fees (replay has no order book)
MAJOR_QUOTES       = {'BTC', 'ETH'}
SELECTORS          = ('rule', 'random', 'llm')

# Bounds every hypothesis is clamped into: (min, max)
BOUNDS = {
    'z_entry':        (-5.0, -1.5),
    'z_exit':         (-1.5, 1.5),
    'window_hours':   (6.0, 72.0),
    'recent_hours':   (0.5, 12.0),
    'max_hold_hours': (1.0, 72.0),
    'stop_loss_pct':  (-25.0, -1.0),
    'max_open':       (1, 5),
    'cooldown_hours': (0.0, 24.0),
}
CHOICES = {
    'quotes':     ('any', 'majors'),
    'btc_regime': ('any', 'up', 'down'),
    'selector':   SELECTORS,
}


@dataclass(frozen=True)
class StrategySpec:
    name:           str   = 'baseline-rule'
    z_entry:        float = -2.0     # open when z < z_entry (on both windows)
    z_exit:         float = 0.0      # close when z > z_exit (reverted)
    window_hours:   float = 24.0     # trailing window for the median/MAD
    recent_hours:   float = 2.0      # dislocation must also hold on this window
    max_hold_hours: float = 24.0
    stop_loss_pct:  float = -10.0    # net % on the position
    quotes:         str   = 'any'    # 'majors' = quote must be BTC or ETH
    btc_regime:     str   = 'any'    # only trade when BTC's 24h move is up/down
    selector:       str   = 'rule'
    max_open:       int   = 3
    cooldown_hours: float = 1.0      # per pair, after a close
    step_minutes:   int   = field(default=10, compare=False)

    def params(self) -> dict:
        d = asdict(self)
        d.pop('step_minutes', None)
        return d


def sanitize_spec(raw: dict, name: str | None = None) -> StrategySpec:
    """Clamp a proposed parameter dict into BOUNDS/CHOICES. Unknown keys are
    ignored; bad values fall back to defaults. Never raises."""
    base = StrategySpec()
    out = {}
    raw = raw if isinstance(raw, dict) else {}
    for key, (lo, hi) in BOUNDS.items():
        default = getattr(base, key)
        try:
            v = float(raw.get(key, default))
            if math.isnan(v):
                raise ValueError
        except (TypeError, ValueError):
            v = default
        v = min(max(v, lo), hi)
        out[key] = int(round(v)) if isinstance(default, int) else round(v, 4)
    for key, opts in CHOICES.items():
        v = str(raw.get(key, getattr(base, key))).strip().lower()
        out[key] = v if v in opts else getattr(base, key)
    out['recent_hours'] = min(out['recent_hours'], out['window_hours'] / 2)
    if out['z_exit'] <= out['z_entry']:
        out['z_exit'] = base.z_exit
    out['name'] = str(name or raw.get('name') or 'hypothesis')[:60]
    return StrategySpec(**out)


def round_trip_cost_pct(base: str, quote: str) -> float:
    _, legs = trade_route(base, quote)
    return 2 * legs * BINANCE_TAKER_FEE * 100 + ASSUMED_SPREAD_PCT


class ZPanel:
    """Robust z-scores of every pair's log ratio at every decision step.
    Shared by all strategies with the same window settings (the expensive part)."""

    def __init__(self, prices: pd.DataFrame, window_hours: float,
                 recent_hours: float, step_minutes: int = 10):
        prices = prices.sort_index()
        bar_min = _bar_minutes(prices)
        self.assets = [a for a in prices.columns if prices[a].notna().sum() > 0]
        self.index = prices.index
        self.bar_minutes = bar_min
        W = max(10, int(round(window_hours * 60 / bar_min)))
        R = max(3, int(round(recent_hours * 60 / bar_min)))
        S = max(1, int(round(step_minutes / bar_min)))
        lp = np.log(prices[self.assets].ffill(limit=3).to_numpy(dtype=float))
        pairs = [(b, q) for b, q in itertools.permutations(range(len(self.assets)), 2)]
        self.pairs = [(self.assets[b], self.assets[q]) for b, q in pairs]
        B = np.array([b for b, _ in pairs], dtype=int)
        Q = np.array([q for _, q in pairs], dtype=int)
        self.steps = list(range(W - 1, len(prices), S))
        n_steps, n_pairs = len(self.steps), len(pairs)
        self.z = np.full((n_steps, n_pairs), np.nan)
        self.z_recent = np.full((n_steps, n_pairs), np.nan)
        self.log_ratio = np.full((n_steps, n_pairs), np.nan)
        min_obs = max(20, W // 2)
        for i, t in enumerate(self.steps):
            win = lp[t - W + 1:t + 1]
            lr = win[:, B] - win[:, Q]                      # (W, pairs)
            cur = lr[-1]
            self.log_ratio[i] = cur
            enough = np.sum(~np.isnan(lr), axis=0) >= min_obs
            self.z[i] = np.where(enough, _robust_z(lr, cur), np.nan)
            self.z_recent[i] = np.where(enough, _robust_z(lr[-R:], cur), np.nan)
        np.clip(self.z, -Z_SCORE_CLAMP, Z_SCORE_CLAMP, out=self.z)
        # BTC 24h log return at each step, for regime filters
        self.btc_24h = np.full(n_steps, np.nan)
        if 'BTC' in self.assets:
            k = self.assets.index('BTC')
            lag = max(1, int(round(24 * 60 / bar_min)))
            for i, t in enumerate(self.steps):
                if t - lag >= 0:
                    self.btc_24h[i] = lp[t, k] - lp[t - lag, k]

    def step_time(self, i: int) -> pd.Timestamp:
        return self.index[self.steps[i]]


def _bar_minutes(prices: pd.DataFrame) -> float:
    if len(prices.index) < 2:
        return 5.0
    d = pd.Series(prices.index).diff().dropna().dt.total_seconds().median() / 60
    return float(d) if d and d > 0 else 5.0


def _robust_z(window: np.ndarray, cur: np.ndarray) -> np.ndarray:
    with np.errstate(all='ignore'):
        med = np.nanmedian(window, axis=0)
        mad = np.nanmedian(np.abs(window - med), axis=0) * MAD_CONSISTENCY
        z = (cur - med) / mad
    return np.where(mad > 0, z, 0.0)


_PANELS: dict = {}


def get_panel(prices: pd.DataFrame, spec: StrategySpec) -> ZPanel:
    key = (id(prices), len(prices), spec.window_hours, spec.recent_hours, spec.step_minutes)
    if key not in _PANELS:
        if len(_PANELS) > 16:
            _PANELS.clear()
        _PANELS[key] = ZPanel(prices, spec.window_hours, spec.recent_hours, spec.step_minutes)
    return _PANELS[key]


def replay(prices: pd.DataFrame, spec: StrategySpec, seed: int = 0,
           picker=None, start=None) -> list[dict]:
    """Run one strategy. `start` (timestamp) lets a test segment use earlier
    bars as warm-up history without trading on them. Returns closed trades."""
    if prices is None or prices.empty:
        return []
    panel = get_panel(prices, spec)
    rng = np.random.default_rng(seed)
    pair_idx = {p: j for j, p in enumerate(panel.pairs)}
    step_h = spec.step_minutes / 60
    open_pos: dict[tuple, dict] = {}
    cooldown_until: dict[tuple, int] = {}
    trades: list[dict] = []
    first = 0
    if start is not None:
        ts = pd.Timestamp(start)
        ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
        first = next((i for i in range(len(panel.steps)) if panel.step_time(i) >= ts),
                     len(panel.steps))

    def close(pair, pos, i, reason):
        j = pair_idx[pair]
        lr = panel.log_ratio[i, j]
        gross = (math.exp(lr - pos['entry_lr']) - 1) * 100 if not math.isnan(lr) else 0.0
        trades.append({
            'pair': f"{pair[0]}/{pair[1]}", 'base': pair[0], 'quote': pair[1],
            'entry_ts': panel.step_time(pos['i']), 'exit_ts': panel.step_time(i),
            'entry_z': pos['entry_z'], 'exit_z': None if math.isnan(panel.z[i, j]) else float(panel.z[i, j]),
            'gross_pct': round(gross, 4), 'cost_pct': pos['cost'],
            'ret_pct': round(gross - pos['cost'], 4),
            'hold_hours': round((i - pos['i']) * step_h, 3), 'reason': reason,
        })
        cooldown_until[pair] = i + int(math.ceil(spec.cooldown_hours / step_h))

    last = len(panel.steps) - 1
    for i in range(first, len(panel.steps)):
        # 1. manage open positions
        for pair in list(open_pos):
            pos = open_pos[pair]
            j = pair_idx[pair]
            lr, z = panel.log_ratio[i, j], panel.z[i, j]
            if math.isnan(lr):
                continue
            net = (math.exp(lr - pos['entry_lr']) - 1) * 100 - pos['cost']
            held = (i - pos['i']) * step_h
            reason = ('stop_loss' if net <= spec.stop_loss_pct else
                      'z_crossback' if (not math.isnan(z) and z > spec.z_exit) else
                      'max_hold' if held >= spec.max_hold_hours else
                      'end_of_data' if i == last else None)
            if reason:
                close(pair, pos, i, reason)
                del open_pos[pair]
        if i == last or len(open_pos) >= spec.max_open:
            continue
        # 2. regime gate
        if spec.btc_regime != 'any':
            r = panel.btc_24h[i]
            if math.isnan(r) or (spec.btc_regime == 'up') != (r > 0):
                continue
        # 3. eligible pairs and candidates (no lookahead: row i only)
        busy = {a for p in open_pos for a in p}
        eligible = [j for j, (b, q) in enumerate(panel.pairs)
                    if b not in busy and q not in busy
                    and cooldown_until.get((b, q), -1) <= i
                    and cooldown_until.get((q, b), -1) <= i
                    and (spec.quotes == 'any' or q in MAJOR_QUOTES)
                    and not math.isnan(panel.z[i, j])]
        cands = [j for j in eligible
                 if panel.z[i, j] < spec.z_entry and panel.z_recent[i, j] < spec.z_entry]
        if not cands:
            continue
        if spec.selector == 'rule':
            j = min(cands, key=lambda k: panel.z[i, k])
        elif spec.selector == 'random':
            j = int(rng.choice(eligible))            # same moments, random pair
        else:
            if picker is None:
                continue
            table = pd.DataFrame({'pair': [f"{panel.pairs[k][0]}/{panel.pairs[k][1]}" for k in cands],
                                  'z_score': [float(panel.z[i, k]) for k in cands]})
            choice = picker(table.sort_values('z_score').reset_index(drop=True), panel.step_time(i))
            if not choice or '/' not in choice:
                continue
            pair = tuple(choice.split('/', 1))
            if pair not in pair_idx or pair_idx[pair] not in cands:
                continue
            j = pair_idx[pair]
        pair = panel.pairs[j]
        open_pos[pair] = {'i': i, 'entry_lr': float(panel.log_ratio[i, j]),
                          'entry_z': float(panel.z[i, j]),
                          'cost': round(round_trip_cost_pct(*pair), 4)}
    return trades


def basket_return_pct(prices: pd.DataFrame, start=None, end=None,
                      assets: list[str] | None = None) -> float | None:
    """Equal-weight buy-and-hold return over [start, end] (no rebalancing, no fees)."""
    p = prices.sort_index()
    if start is not None:
        p = p[p.index >= pd.Timestamp(start)]
    if end is not None:
        p = p[p.index <= pd.Timestamp(end)]
    if assets:
        p = p[[a for a in assets if a in p.columns]]
    p = p.dropna(axis=1, how='any')
    if p.empty or len(p) < 2:
        return None
    return round(float(((p.iloc[-1] / p.iloc[0]) - 1).mean() * 100), 4)
