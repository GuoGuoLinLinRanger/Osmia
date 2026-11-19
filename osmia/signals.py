"""Statistical signal engine: robust z-scores, candidates, confidence."""

import duckdb
import numpy as np
import pandas as pd

from .config import (
    FNG_FEAR, FNG_GREED, MEME_ASSETS, MIN_PAIR_SAMPLES, RECENT_PAIR_SAMPLES,
    TOP_N_LLM, TOP_N_RAW, TREND_SLOPE_THRESHOLD, ZSCORE_BUY_THRESH,
    ZSCORE_BUY_THRESH_HEIGHTENED, Z_SCORE_CLAMP,
)
from .core import detect_dominance, mad_zscore, rule_based_score


def get_pair_reputations(con, pairs: list[str]) -> dict[str, float]:
    """Look up rep scores for directional pairs ('SOL/BTC' and 'BTC/SOL' differ)."""
    if not pairs:
        return {}
    valid = [(p, p) for p in pairs if '/' in p and len(p.split('/')) == 2]
    if not valid:
        return {}
    canonical_ids = list({cid for _, cid in valid})
    ph = ','.join(['?' for _ in canonical_ids])
    df = con.execute(f"""
        SELECT pair_id, n_signals AS n, rep_score
        FROM pair_reputation
        WHERE pair_id IN ({ph})
    """, canonical_ids).df()
    canonical_rep = {}
    for _, r in df.iterrows():
        canonical_rep[r['pair_id']] = float(r['rep_score']) if r['rep_score'] is not None else 0.0
    result = {}
    for orig_pair, cid in valid:
        result[orig_pair] = canonical_rep.get(cid, 0.0)
    return result

def get_pair_trend_slopes(con, pairs: list[str]) -> dict[str, float]:
    result = {p: 0.0 for p in pairs}
    if not pairs:
        return result
    # One wide [minute x asset] matrix over the recent window, then each pair's
    # ratio slope in pandas — avoids the per-pair DATE_TRUNC self-join that
    # mirrored the get_raw_anomalies hotspot (ran once per candidate pair).
    cutoff = con.execute("""
        SELECT MIN(ts) FROM (
            SELECT DISTINCT DATE_TRUNC('minute', timestamp) AS ts
            FROM market_history
            ORDER BY ts DESC LIMIT 500
        )
    """).fetchone()[0]
    long = con.execute("""
        SELECT DATE_TRUNC('minute', timestamp) AS ts, asset_id, price_usd
        FROM market_history
        WHERE timestamp >= ?
    """, [cutoff]).df()
    if long.empty:
        return result

    wide = (long.sort_values('ts')
                .pivot_table(index='ts', columns='asset_id',
                             values='price_usd', aggfunc='last'))
    for pair in pairs:
        parts = pair.split('/')
        if len(parts) != 2:
            continue
        base, quote = parts
        if base not in wide.columns or quote not in wide.columns:
            continue
        # wide is sorted ascending, so the ratio series is already chronological.
        y = (wide[base] / wide[quote]).replace(
            [np.inf, -np.inf], np.nan).dropna().values
        if len(y) < 20:
            continue
        x      = np.arange(len(y))
        slope  = float(np.polyfit(x, y, 1)[0])
        median = float(np.median(y))
        result[pair] = round(slope / median if median != 0 else 0.0, 6)
    return result

def _pair_price_matrix(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(latest per-asset row indexed by asset_id, wide [minute x asset] price matrix)."""
    latest = con.execute("""
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df().set_index('asset_id')
    if latest.empty:
        return latest, pd.DataFrame()
    cutoff = con.execute("""
        SELECT MIN(ts) FROM (
            SELECT DISTINCT DATE_TRUNC('minute', timestamp) AS ts
            FROM market_history
            ORDER BY ts DESC LIMIT 2000
        )
    """).fetchone()[0]
    long = con.execute("""
        SELECT DATE_TRUNC('minute', timestamp) AS ts, asset_id, price_usd
        FROM market_history
        WHERE timestamp >= ?
    """, [cutoff]).df()
    if long.empty:
        return latest, pd.DataFrame()
    wide = (long.sort_values('ts')
                .pivot_table(index='ts', columns='asset_id',
                             values='price_usd', aggfunc='last'))
    return latest, wide

def get_pair_zscores(con, pairs: list[str]) -> dict[str, float]:
    """Current clamped robust z-score for each directional 'BASE/QUOTE' pair that
    has enough history. Used for exits, where the pair is usually no longer a
    candidate (its z has already recovered above the entry threshold)."""
    if not pairs:
        return {}
    latest, wide = _pair_price_matrix(con)
    out = {}
    for p in set(pairs):
        if '/' not in p:
            continue
        base, quote = p.split('/', 1)
        if (base not in wide.columns or quote not in wide.columns
                or base not in latest.index or quote not in latest.index):
            continue
        quote_px = latest.loc[quote, 'price_usd']
        if not quote_px:
            continue
        ratios = (wide[base] / wide[quote]).replace(
            [np.inf, -np.inf], np.nan).dropna().values
        if len(ratios) < MIN_PAIR_SAMPLES:
            continue
        z = mad_zscore(ratios, latest.loc[base, 'price_usd'] / quote_px)
        out[p] = round(max(-Z_SCORE_CLAMP, min(Z_SCORE_CLAMP, z)), 3)
    return out

def get_raw_anomalies(con: duckdb.DuckDBPyConnection,
                      relaxed_assets: set[str] | None = None) -> pd.DataFrame:
    relaxed_assets = relaxed_assets or set()
    latest = con.execute("""
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df().set_index('asset_id')

    if latest.empty:
        return pd.DataFrame()

    assets = latest.index.tolist()

    # Pull recent minute-level prices for all assets in ONE query, then build a
    # wide [minute x asset] price matrix and compute every pair ratio in pandas.
    # Replaces the old 14x14 per-pair self-join (182 queries) which joined
    # market_history to itself on DATE_TRUNC('minute', ...) — a non-sargable key
    # that forced full-table scans and ran for minutes once the table grew.
    # Rows are already minute-truncated on insert, so DATE_TRUNC here is a no-op
    # safety net for any legacy sub-minute rows.
    cutoff = con.execute("""
        SELECT MIN(ts) FROM (
            SELECT DISTINCT DATE_TRUNC('minute', timestamp) AS ts
            FROM market_history
            ORDER BY ts DESC LIMIT 2000
        )
    """).fetchone()[0]
    long = con.execute("""
        SELECT DATE_TRUNC('minute', timestamp) AS ts, asset_id, price_usd
        FROM market_history
        WHERE timestamp >= ?
    """, [cutoff]).df()
    if long.empty:
        return pd.DataFrame()

    wide = (long.sort_values('ts')
                .pivot_table(index='ts', columns='asset_id',
                             values='price_usd', aggfunc='last'))

    rows = []
    for base in assets:
        if base not in wide.columns:
            continue
        base_px     = latest.loc[base, 'price_usd']
        base_series = wide[base]
        for quote in assets:
            if base == quote or quote not in wide.columns:
                continue
            quote_px = latest.loc[quote, 'price_usd']
            if not quote_px or quote_px == 0:
                continue
            current_ratio = base_px / quote_px
            ratios = (base_series / wide[quote]).replace(
                [np.inf, -np.inf], np.nan).dropna().values
            if len(ratios) < MIN_PAIR_SAMPLES:
                continue
            # Heightened mode: relax the dump threshold for pairs touching an asset
            # with an imminent unlock so smaller pre-unlock dislocations still surface.
            thresh = (ZSCORE_BUY_THRESH_HEIGHTENED
                      if (base in relaxed_assets or quote in relaxed_assets)
                      else ZSCORE_BUY_THRESH)
            z = mad_zscore(ratios, current_ratio)
            if z >= thresh:
                continue
            # Stale-median guard: a long window whose median lags a slow grind
            # (e.g. ADA bleeding vs the field for days) manufactures z=-20+ that is
            # a *trend*, not a live reversion spring. Require the deviation to still
            # be anomalous on a short recent window; if it has settled at a new
            # level there is nothing left to revert to — skip it.
            z_recent = mad_zscore(ratios[-RECENT_PAIR_SAMPLES:], current_ratio)
            if z_recent >= thresh:
                continue
            # Clamp the runaway so rank/confidence/sizing use an honest magnitude
            # instead of the stale-median artifact that pins one pair at 10/10.
            z = max(z, -Z_SCORE_CLAMP)
            med_ratio = float(np.median(ratios))
            rows.append({
                'pair':         f"{base}/{quote}",
                'base_asset':   base,
                'quote_asset':  quote,
                'ratio':        round(current_ratio, 6),
                'mean_ratio':   round(med_ratio, 6),
                'z_score':      round(z, 3),
                'vwap_ratio':   latest.loc[base, 'vwap_ratio'],
                'volume_z':     latest.loc[base, 'volume_z'],
                'spread_pct':   latest.loc[base, 'spread_pct'],
                'sample_count': len(ratios),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('z_score').head(TOP_N_RAW).reset_index(drop=True)

def get_pair_persistence(con, pairs):
    if not pairs: return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT top_pair, COUNT(*) as hits FROM llm_verdicts
        WHERE top_pair IN ({ph}) AND action = 'accumulate'
          AND timestamp >= NOW() - INTERVAL 24 HOURS
        GROUP BY top_pair
    """, pairs).df()
    return dict(zip(df['top_pair'], df['hits'])) if not df.empty else {}

def get_prev_z_scores(con, pairs):
    if not pairs: return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT DISTINCT ON (top_pair) top_pair, z_score FROM llm_verdicts
        WHERE top_pair IN ({ph})
        ORDER BY top_pair, timestamp DESC
    """, pairs).df()
    return dict(zip(df['top_pair'], df['z_score'])) if not df.empty else {}

def build_candidates(con, fng_adj=0.0,
                     relaxed_assets: set[str] | None = None
                     ) -> tuple[pd.DataFrame, str | None, str | None]:
    raw = get_raw_anomalies(con, relaxed_assets)
    if raw.empty:
        return raw, None, None

    pairs       = raw['pair'].tolist()
    persistence = get_pair_persistence(con, pairs)
    prev_z      = get_prev_z_scores(con, pairs)
    reputations = get_pair_reputations(con, pairs)
    slopes      = get_pair_trend_slopes(con, pairs)

    raw['persistence_24h'] = raw['pair'].map(lambda p: persistence.get(p, 0))
    raw['z_velocity']      = raw['pair'].map(
        lambda p: round(raw.loc[raw['pair'] == p, 'z_score'].iloc[0] - prev_z[p], 3)
        if p in prev_z else None
    )
    raw['reputation']  = raw['pair'].map(lambda p: reputations.get(p, None))
    raw['trend_slope'] = raw['pair'].map(lambda p: slopes.get(p, 0.0))
    raw['rule_score']  = raw.apply(
        lambda r: rule_based_score(r, persistence, prev_z, reputations, slopes, fng_adj), axis=1
    )

    quote_warn = detect_dominance(raw, 'quote_asset')
    base_warn  = detect_dominance(raw, 'base_asset')
    top = raw.sort_values('rule_score').head(TOP_N_LLM).reset_index(drop=True)
    return top, quote_warn, base_warn

def derive_confidence(candidates: pd.DataFrame, top_pair: str,
                      fng_value: int = 50) -> int:
    if candidates.empty or top_pair not in candidates['pair'].values:
        return 5
    scores = candidates.set_index('pair')['rule_score']
    winner = scores[top_pair]
    others = scores.drop(top_pair)
    gap    = abs(others.min() - winner) if not others.empty else 0
    conf   = 5 + min(int(gap / 0.25), 4)
    row    = candidates[candidates['pair'] == top_pair].iloc[0]
    if row['spread_pct'] is not None and row['spread_pct'] > 0.1: conf -= 1
    if row['quote_asset'] in MEME_ASSETS:                          conf -= 1
    rep = row.get('reputation')
    if rep is not None:
        if rep >= 0.5:    conf = min(conf + 1, 10)
        elif rep <= -0.5: conf = max(conf - 2, 1)
    slope = abs(row.get('trend_slope', 0.0))
    if slope > TREND_SLOPE_THRESHOLD:
        conf = max(conf - 1, 1)
    # F&G modifier
    if fng_value <= FNG_FEAR:   conf = min(conf + 1, 10)
    elif fng_value >= FNG_GREED: conf = max(conf - 1, 1)
    return max(1, min(conf, 10))

def get_fiat_candidates(con) -> tuple[pd.DataFrame, bool]:
    date_range = con.execute("""
        SELECT DATEDIFF('day', MIN(timestamp), MAX(timestamp))
        FROM market_history WHERE source = 'live'
    """).fetchone()[0] or 0
    short_history = date_range < 14

    latest = con.execute("""
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct, change_24h
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df()

    rows = []
    for _, r in latest.iterrows():
        asset = r['asset_id']
        hist  = con.execute(
            "SELECT price_usd FROM market_history WHERE asset_id=? ORDER BY timestamp DESC LIMIT 2000",
            [asset]).fetchdf()
        if len(hist) < MIN_PAIR_SAMPLES:
            continue
        prices_arr = hist['price_usd'].dropna().values
        z          = mad_zscore(prices_arr, r['price_usd'])
        if z >= ZSCORE_BUY_THRESH:
            continue
        rows.append({
            'asset_id':     asset,
            'price_usd':    round(r['price_usd'], 6),
            'mean_price':   round(float(np.median(prices_arr)), 6),
            'z_score':      round(z, 3),
            'vwap_ratio':   r['vwap_ratio'],
            'volume_z':     r['volume_z'],
            'spread_pct':   r['spread_pct'],
            'change_24h':   r['change_24h'],
            'sample_count': len(prices_arr),
        })

    df = pd.DataFrame(rows).sort_values('z_score').head(TOP_N_LLM).reset_index(drop=True) \
        if rows else pd.DataFrame()
    return df, short_history
