#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.4
Unified ingest + analyze script for the Osmia LLM trading monitor.

Usage:
    python osmia_trader.py              # ingest + analyze (pair mode)
    python osmia_trader.py --ingest     # ingest only
    python osmia_trader.py --analyze    # pair signal analysis
    python osmia_trader.py --fiat       # fiat entry analysis (USDT buy signals)
    python osmia_trader.py --init-db    # (re)create DB schema and exit
    python osmia_trader.py --status     # quick DB health summary

Changes from v0.3:
    - Confidence scoring: forced relative ranking within batch, anchored to
      rule_score delta between candidates — kills flat 8/10 anchoring
    - Pair output: actionable trade block with route, qty, fee breakdown,
      break-even threshold
    - Fiat output (--fiat): per-asset USDT buy signal with price, qty
      estimate, fee, and signal context
    - LLM prompt redesigned: rule_score rank is authoritative, LLM must
      justify any override; confidence derived from spread between candidates
    - BINANCE_TAKER_FEE_BNB: 0.075% (BNB discount applied)
    - Direct pair check: knows which cross pairs exist on Binance natively
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import ccxt
import duckdb
import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH    = '/opt/llm_trader/market_data.duckdb'
QWEN_URL   = "http://localhost:8081/v1/chat/completions"
QWEN_MODEL = "qwen-instruct"

ASSETS  = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP',
           'TRX', 'BNB', 'AVAX', 'LTC', 'XLM', 'APT', 'DOGE', 'SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]

# Fee model
BINANCE_TAKER_FEE = 0.00075   # 0.075% with BNB discount

# Known direct cross pairs on Binance (no USDT routing needed)
# Incomplete list — covers the most liquid; everything else routes via USDT
BINANCE_DIRECT_PAIRS = {
    'ETH/BTC', 'BTC/ETH',
    'SOL/BTC', 'BTC/SOL',
    'SOL/ETH', 'ETH/SOL',
    'BNB/BTC', 'BTC/BNB',
    'BNB/ETH', 'ETH/BNB',
    'ADA/BTC', 'BTC/ADA',
    'ADA/ETH', 'ETH/ADA',
    'XRP/BTC', 'BTC/XRP',
    'XRP/ETH', 'ETH/XRP',
    'DOGE/BTC', 'BTC/DOGE',
    'LTC/BTC', 'BTC/LTC',
    'LTC/ETH', 'ETH/LTC',
    'DOT/BTC', 'BTC/DOT',
    'DOT/ETH', 'ETH/DOT',
    'AVAX/BTC', 'BTC/AVAX',
    'AVAX/ETH', 'ETH/AVAX',
}

# Stats thresholds
MIN_PAIR_SAMPLES  = 20
VOLUME_WINDOW     = 50
ZSCORE_BUY_THRESH = -2.0
TOP_N_RAW         = 10
TOP_N_LLM         = 3

MEME_ASSETS = {'DOGE', 'XLM', 'TRX', 'XRP'}

# Notional trade size for fee/qty calculations (USDT)
NOTIONAL_USDT = 500.0

# ── Schema ────────────────────────────────────────────────────────────────────

DDL_MARKET_HISTORY = """
CREATE TABLE IF NOT EXISTS market_history (
    timestamp      TIMESTAMPTZ NOT NULL,
    asset_id       VARCHAR     NOT NULL,
    price_usd      DOUBLE,
    volume_24h     DOUBLE,
    change_24h     DOUBLE,
    market_cap     DOUBLE,
    vwap_ratio     DOUBLE,
    volume_z       DOUBLE,
    spread_pct     DOUBLE,
    high_low_pct   DOUBLE,
    PRIMARY KEY (timestamp, asset_id)
);
"""

DDL_LLM_VERDICTS = """
CREATE TABLE IF NOT EXISTS llm_verdicts (
    timestamp        TIMESTAMPTZ NOT NULL,
    mode             VARCHAR     NOT NULL DEFAULT 'pair',
    top_pair         VARCHAR,
    z_score          DOUBLE,
    confidence_score INTEGER,
    action           VARCHAR,
    thesis           TEXT,
    risks            TEXT,
    prose            TEXT,
    raw_json         TEXT,
    PRIMARY KEY (timestamp, top_pair)
);
"""

# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL_MARKET_HISTORY)
    con.execute(DDL_LLM_VERDICTS)
    # Add mode column if missing (migration from v0.3)
    cols = con.execute("DESCRIBE llm_verdicts").df()['column_name'].tolist()
    if 'mode' not in cols:
        con.execute("ALTER TABLE llm_verdicts ADD COLUMN mode VARCHAR DEFAULT 'pair'")


# ── Ingest ────────────────────────────────────────────────────────────────────

def _volume_z_scores(con: duckdb.DuckDBPyConnection,
                     new_df: pd.DataFrame) -> dict[str, float]:
    results = {}
    for _, row in new_df.iterrows():
        asset   = row['asset_id']
        cur_vol = row['volume_24h']
        hist = con.execute(
            "SELECT volume_24h FROM market_history "
            "WHERE asset_id = ? ORDER BY timestamp DESC LIMIT ?",
            [asset, VOLUME_WINDOW]
        ).fetchdf()
        if len(hist) < 5:
            results[asset] = 0.0
            continue
        mean = hist['volume_24h'].mean()
        std  = hist['volume_24h'].std()
        results[asset] = float((cur_vol - mean) / std) if std > 0 else 0.0
    return results


def ingest(con: duckdb.DuckDBPyConnection) -> int:
    print(f"[{datetime.now():%H:%M:%S}] Fetching tickers …")
    exchange = ccxt.binance({'enableRateLimit': True})

    try:
        tickers = exchange.fetch_tickers(SYMBOLS)
    except Exception as e:
        print(f"  ccxt error: {e}", file=sys.stderr)
        return 0

    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    rows = []
    for symbol in SYMBOLS:
        t = tickers.get(symbol)
        if not t:
            continue
        asset = symbol.split('/')[0]
        last  = t.get('last')  or 0.0
        ask   = t.get('ask')   or 0.0
        bid   = t.get('bid')   or 0.0
        mid   = (ask + bid) / 2 if (ask and bid) else last
        vwap  = t.get('vwap')  or 0.0
        high  = t.get('high')  or 0.0
        low   = t.get('low')   or 0.0

        rows.append({
            'timestamp':    ts,
            'asset_id':     asset,
            'price_usd':    last,
            'volume_24h':   t.get('quoteVolume') or 0.0,
            'change_24h':   t.get('percentage')  or 0.0,
            'market_cap':   0.0,
            'vwap_ratio':   round(last / vwap,            6) if vwap else None,
            'spread_pct':   round((ask - bid) / mid * 100, 6) if mid  else None,
            'high_low_pct': round((high - low) / low * 100, 6) if low  else None,
            'volume_z':     None,
        })

    if not rows:
        print("  No ticker data returned.", file=sys.stderr)
        return 0

    df = pd.DataFrame(rows)
    vzs = _volume_z_scores(con, df)
    df['volume_z'] = df['asset_id'].map(vzs)

    con.execute("""
        INSERT OR IGNORE INTO market_history
            (timestamp, asset_id, price_usd, volume_24h, change_24h,
             market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct)
        SELECT timestamp, asset_id, price_usd, volume_24h, change_24h,
               market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct
        FROM df
    """)

    n = len(df)
    print(f"  Ingested {n} records @ {ts:%Y-%m-%d %H:%M} UTC")
    return n


# ── Fee / trade helpers ───────────────────────────────────────────────────────

def trade_route(base: str, quote: str) -> tuple[str, int]:
    """
    Returns (route_description, num_legs).
    Checks if a direct Binance pair exists, otherwise routes via USDT.
    """
    direct = f"{base}/{quote}"
    if direct in BINANCE_DIRECT_PAIRS:
        return f"{base}/{quote} (direct)", 1
    return f"{base}/USDT → {quote}/USDT", 2


def fee_breakdown(notional: float, num_legs: int,
                  spread_pct: float | None) -> dict:
    """
    Returns fee components and total cost for a trade of `notional` USDT.
    spread_pct is the base asset spread as a percentage.
    """
    fee_per_leg   = notional * BINANCE_TAKER_FEE
    total_fee     = fee_per_leg * num_legs
    spread_cost   = notional * (spread_pct / 100) if spread_pct else 0.0
    total_cost    = total_fee + spread_cost
    breakeven_pct = (total_cost / notional) * 100

    return {
        'notional':      round(notional, 2),
        'fee_per_leg':   round(fee_per_leg, 4),
        'num_legs':      num_legs,
        'total_fee':     round(total_fee, 4),
        'spread_cost':   round(spread_cost, 4),
        'total_cost':    round(total_cost, 4),
        'breakeven_pct': round(breakeven_pct, 4),
    }


def get_current_prices(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """Latest price per asset from market_history."""
    df = con.execute("""
        SELECT asset_id, price_usd
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df()
    return dict(zip(df['asset_id'], df['price_usd']))


# ── Signal engine (shared) ────────────────────────────────────────────────────

def get_raw_anomalies(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
    WITH bucketed AS (
        SELECT DATE_TRUNC('minute', timestamp) AS ts_min,
               asset_id, price_usd, vwap_ratio, volume_z, spread_pct
        FROM market_history
    ),
    pairs AS (
        SELECT a.ts_min,
               a.asset_id || '/' || b.asset_id                     AS pair,
               a.asset_id                                           AS base_asset,
               b.asset_id                                           AS quote_asset,
               a.price_usd / NULLIF(b.price_usd, 0)                AS ratio,
               a.vwap_ratio                                         AS base_vwap_ratio,
               a.volume_z                                           AS base_volume_z,
               a.spread_pct                                         AS base_spread_pct
        FROM bucketed a
        JOIN bucketed b ON a.ts_min = b.ts_min AND a.asset_id != b.asset_id
    ),
    stats AS (
        SELECT pair,
               AVG(ratio)    AS mean_ratio,
               STDDEV(ratio) AS std_ratio,
               COUNT(*)      AS sample_count
        FROM pairs
        GROUP BY pair
        HAVING sample_count > {MIN_PAIR_SAMPLES}
    ),
    latest AS (
        SELECT pair, base_asset, quote_asset, ratio,
               base_vwap_ratio, base_volume_z, base_spread_pct
        FROM pairs
        WHERE ts_min = (
            SELECT MAX(DATE_TRUNC('minute', timestamp)) FROM market_history
        )
    )
    SELECT  l.pair,
            l.base_asset,
            l.quote_asset,
            ROUND(l.ratio, 6)                                             AS ratio,
            ROUND(s.mean_ratio, 6)                                        AS mean_ratio,
            ROUND((l.ratio - s.mean_ratio) / NULLIF(s.std_ratio, 0), 3)  AS z_score,
            ROUND(l.base_vwap_ratio, 4)                                   AS vwap_ratio,
            ROUND(l.base_volume_z,   3)                                   AS volume_z,
            ROUND(l.base_spread_pct, 4)                                   AS spread_pct,
            s.sample_count
    FROM latest l
    JOIN stats  s ON l.pair = s.pair
    WHERE (l.ratio - s.mean_ratio) / NULLIF(s.std_ratio, 0) < {ZSCORE_BUY_THRESH}
    ORDER BY z_score ASC
    LIMIT {TOP_N_RAW}
    """
    return con.execute(query).df()


def get_pair_persistence(con: duckdb.DuckDBPyConnection,
                         pairs: list[str]) -> dict[str, int]:
    if not pairs:
        return {}
    placeholders = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT top_pair, COUNT(*) as hits
        FROM llm_verdicts
        WHERE top_pair IN ({placeholders})
          AND action = 'accumulate'
          AND timestamp >= NOW() - INTERVAL 24 HOURS
        GROUP BY top_pair
    """, pairs).df()
    return dict(zip(df['top_pair'], df['hits'])) if not df.empty else {}


def get_prev_z_scores(con: duckdb.DuckDBPyConnection,
                      pairs: list[str]) -> dict[str, float]:
    if not pairs:
        return {}
    placeholders = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT DISTINCT ON (top_pair) top_pair, z_score
        FROM llm_verdicts
        WHERE top_pair IN ({placeholders})
        ORDER BY top_pair, timestamp DESC
    """, pairs).df()
    return dict(zip(df['top_pair'], df['z_score'])) if not df.empty else {}


def detect_quote_dominance(df: pd.DataFrame) -> str | None:
    counts   = df['quote_asset'].value_counts()
    dominant = counts[counts >= 3]
    return dominant.index[0] if not dominant.empty else None


def rule_based_score(row: pd.Series,
                     persistence: dict[str, int],
                     prev_z: dict[str, float]) -> float:
    score = row['z_score']
    hits  = persistence.get(row['pair'], 0)
    score -= min(hits * 0.1, 0.5)

    prev = prev_z.get(row['pair'])
    if prev is not None:
        score -= 0.2 if row['z_score'] < prev else -0.2

    if row['volume_z']  is not None and row['volume_z']  >  1.5: score -= 0.15
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99: score -= 0.1
    if row['spread_pct'] is not None and row['spread_pct'] > 0.1:  score += 0.3
    if row['quote_asset'] in MEME_ASSETS:                           score += 0.5

    return round(score, 4)


def build_candidates(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str | None]:
    raw = get_raw_anomalies(con)
    if raw.empty:
        return raw, None

    pairs       = raw['pair'].tolist()
    persistence = get_pair_persistence(con, pairs)
    prev_z      = get_prev_z_scores(con, pairs)

    raw['persistence_24h'] = raw['pair'].map(lambda p: persistence.get(p, 0))
    raw['z_velocity']      = raw['pair'].map(
        lambda p: round(raw.loc[raw['pair'] == p, 'z_score'].iloc[0] - prev_z[p], 3)
        if p in prev_z else None
    )
    raw['rule_score'] = raw.apply(
        lambda r: rule_based_score(r, persistence, prev_z), axis=1
    )

    quote_warn = detect_quote_dominance(raw)
    top = raw.sort_values('rule_score').head(TOP_N_LLM).reset_index(drop=True)
    return top, quote_warn


# ── Confidence derivation ─────────────────────────────────────────────────────

def derive_confidence(candidates: pd.DataFrame, top_pair: str) -> int:
    """
    Compute confidence from rule_score spread between winner and runner-up.
    Wider gap = higher confidence. Anchored to 5, scaled ±3.
    Prevents the flat 8/10 anchoring from the LLM.
    """
    if candidates.empty or top_pair not in candidates['pair'].values:
        return 5
    scores = candidates.set_index('pair')['rule_score']
    winner = scores[top_pair]
    others = scores.drop(top_pair)
    if others.empty:
        return 7
    runner_up = others.min()
    gap = abs(runner_up - winner)
    # gap=0 → conf=5, gap=0.5 → conf=7, gap=1.0+ → conf=9
    conf = 5 + min(int(gap / 0.25), 4)
    # Penalise if winner has spread_pct > 0.1 (liquidity concern)
    row = candidates[candidates['pair'] == top_pair].iloc[0]
    if row['spread_pct'] is not None and row['spread_pct'] > 0.1:
        conf -= 1
    # Penalise meme quote
    if row['quote_asset'] in MEME_ASSETS:
        conf -= 1
    return max(1, min(conf, 10))


# ── Pair analysis ─────────────────────────────────────────────────────────────

def build_pair_prompt(candidates: pd.DataFrame, quote_warn: str | None) -> str:
    rows = []
    for i, r in candidates.iterrows():
        vel_str = f"{r['z_velocity']:+.3f}" if r['z_velocity'] is not None else "n/a"
        spread  = f"{r['spread_pct']:.4f}%" if r['spread_pct'] is not None else "n/a"
        vwap    = f"{r['vwap_ratio']:.4f}"  if r['vwap_ratio'] is not None else "n/a"
        vol_z   = f"{r['volume_z']:+.3f}"   if r['volume_z']   is not None else "n/a"
        route, legs = trade_route(r['base_asset'], r['quote_asset'])
        rows.append(
            f"  [rank {i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel_str} | "
            f"persist={r['persistence_24h']} | vwap={vwap} | vol_z={vol_z} | "
            f"spread={spread} | route={route}"
        )
    table = "\n".join(rows)

    quote_block = ""
    if quote_warn:
        quote_block = (
            f"\n⚠ QUOTE DOMINANCE: {quote_warn} is the quote in multiple pairs. "
            f"This may mean {quote_warn} is overbought, not that the base assets are cheap. "
            f"Address this in your thesis.\n"
        )

    return f"""You are a disciplined spot-only crypto trader. A rule-based engine has pre-ranked {len(candidates)} candidates. Your job: validate the top-ranked pick or override with justification.

Candidates (rule engine rank, best first):
{table}
{quote_block}
Definitions:
- z: price-ratio Z-score. More negative = more oversold vs history.
- vel: Z-score change since last analysis. Negative = deepening (good). Positive = recovering (caution).
- persist: times this pair was top signal in last 24h.
- vwap: last/VWAP. <1.0 = below volume-weighted average.
- vol_z: volume anomaly score. >1.5 = unusual interest.
- spread: bid-ask spread %. >0.1% = liquidity concern. >0.15% = flag in risks.
- route: trade execution path on Binance.

Instructions:
- The rule engine rank is authoritative. To override rank 1, you must state why explicitly in thesis.
- Do NOT give all candidates the same confidence. Use the spread between their rule scores.
- action must be one of: accumulate | watch | pass
- If spread > 0.15% on winner, confidence must be <=6.
- thesis and risks: max 120 chars each.

Output: one JSON line, blank line, then 2-3 sentences prose.
JSON: {{"action": "accumulate"|"watch"|"pass", "top_pair": "<PAIR>", "rank": ["<p1>","<p2>",...], "thesis": "<str>", "risks": "<str>"}}
"""


def format_pair_trade(top_row: pd.Series, prices: dict[str, float],
                      verdict: dict, confidence: int) -> None:
    base      = top_row['base_asset']
    quote     = top_row['quote_asset']
    base_px   = prices.get(base,  0.0)
    quote_px  = prices.get(quote, 0.0)
    route, legs = trade_route(base, quote)
    fees      = fee_breakdown(NOTIONAL_USDT, legs, top_row['spread_pct'])

    sell_qty  = round(NOTIONAL_USDT / base_px,  6) if base_px  else 0
    buy_qty   = round(NOTIONAL_USDT / quote_px, 6) if quote_px else 0

    print("\n╔══ TRADE SIGNAL ══════════════════════════════════════════╗")
    print(f"  Mode       : PAIR REVERSION (spot)")
    print(f"  Action     : Sell {base} → Buy {quote}  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Route      : {route}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Sell       : {sell_qty} {base}  @ {base_px:,.4f} USDT")
    print(f"  Buy        : {buy_qty} {quote}  @ {quote_px:,.4f} USDT")
    print(f"  Notional   : {fees['notional']} USDT")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['fee_per_leg']} USDT × {fees['num_legs']} legs = {fees['total_fee']} USDT  (0.075% BNB)")
    print(f"  Spread     : {fees['spread_cost']} USDT  ({top_row['spread_pct']:.4f}%)" if top_row['spread_pct'] else "  Spread     : n/a")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.3f}% of notional)")
    print(f"  Break-even : price ratio must improve by {fees['breakeven_pct']:.3f}%")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Signal     : Z={top_row['z_score']:+.3f}  persist={top_row['persistence_24h']}  vel={top_row['z_velocity'] if top_row['z_velocity'] is not None else 'n/a'}")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_pairs(con: duckdb.DuckDBPyConnection) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running pair analysis …")
    candidates, quote_warn = build_candidates(con)

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {
            'action': 'pass', 'top_pair': 'none', 'thesis': 'Market stable.',
            'risks': '', 'prose': '', 'raw_json': ''
        }, 0, None, 'pair')
        return

    if quote_warn:
        print(f"\n  ⚠ Quote dominance: {quote_warn} — may be overbought, not independent signals")

    print("\n── Candidates (rule-scored) ──")
    display_cols = ['pair', 'z_score', 'z_velocity', 'persistence_24h',
                    'vwap_ratio', 'volume_z', 'spread_pct', 'rule_score']
    print(candidates[display_cols].to_string(index=False))

    prompt  = build_pair_prompt(candidates, quote_warn)
    verdict = ask_qwen(prompt)

    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair)
    prices     = get_current_prices(con)

    top_row = None
    if top_pair and not candidates.empty:
        match = candidates[candidates['pair'] == top_pair]
        if not match.empty:
            top_row = match.iloc[0]

    if top_row is not None and verdict.get('action') != 'pass':
        format_pair_trade(top_row, prices, verdict, confidence)
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'pair')


# ── Fiat analysis ─────────────────────────────────────────────────────────────

def get_fiat_candidates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Per-asset Z-score of price_usd vs rolling window.
    Returns assets where price is statistically cheap vs recent history.
    """
    query = f"""
    WITH stats AS (
        SELECT asset_id,
               AVG(price_usd)    AS mean_price,
               STDDEV(price_usd) AS std_price,
               COUNT(*)          AS sample_count
        FROM market_history
        GROUP BY asset_id
        HAVING sample_count > {MIN_PAIR_SAMPLES}
    ),
    latest AS (
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct, change_24h
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    )
    SELECT  l.asset_id,
            ROUND(l.price_usd, 6)                                          AS price_usd,
            ROUND(s.mean_price, 6)                                         AS mean_price,
            ROUND((l.price_usd - s.mean_price) / NULLIF(s.std_price, 0), 3) AS z_score,
            ROUND(l.vwap_ratio, 4)                                         AS vwap_ratio,
            ROUND(l.volume_z,   3)                                         AS volume_z,
            ROUND(l.spread_pct, 4)                                         AS spread_pct,
            ROUND(l.change_24h, 3)                                         AS change_24h,
            s.sample_count
    FROM latest l
    JOIN stats  s ON l.asset_id = s.asset_id
    WHERE (l.price_usd - s.mean_price) / NULLIF(s.std_price, 0) < {ZSCORE_BUY_THRESH}
    ORDER BY z_score ASC
    LIMIT {TOP_N_LLM}
    """
    return con.execute(query).df()


def build_fiat_prompt(candidates: pd.DataFrame) -> str:
    rows = []
    for i, r in candidates.iterrows():
        spread  = f"{r['spread_pct']:.4f}%" if r['spread_pct'] is not None else "n/a"
        vwap    = f"{r['vwap_ratio']:.4f}"  if r['vwap_ratio'] is not None else "n/a"
        vol_z   = f"{r['volume_z']:+.3f}"   if r['volume_z']   is not None else "n/a"
        chg     = f"{r['change_24h']:+.2f}%" if r['change_24h'] is not None else "n/a"
        rows.append(
            f"  [rank {i+1}] {r['asset_id']:>6s}/USDT | price={r['price_usd']:,.4f} | "
            f"z={r['z_score']:+.3f} | vwap={vwap} | vol_z={vol_z} | "
            f"24h={chg} | spread={spread}"
        )
    table = "\n".join(rows)

    return f"""You are a disciplined spot-only crypto trader advising on USDT buy entries.
These assets are statistically cheap vs their recent price history (Z-score < {ZSCORE_BUY_THRESH}).

Candidates (cheapest first):
{table}

Definitions:
- z: price Z-score vs full history. More negative = cheaper vs own history.
- vwap: last/VWAP. <1.0 = trading below volume-weighted average (undervalued vs activity).
- vol_z: volume anomaly. >1.5 = unusual buying interest.
- 24h: 24h price change. Still falling = caution (catching a knife). Stabilising = better entry.
- spread: bid-ask %. >0.15% = flag in risks.

Instructions:
- Pick the best single USDT buy entry or pass.
- Do NOT pick an asset that is still in freefall (very negative 24h + no volume support).
- Rank all candidates. Do not assign equal confidence to all.
- action: accumulate | watch | pass
- thesis and risks: max 120 chars.

Output: one JSON line, blank line, 2-3 sentences prose.
JSON: {{"action": "accumulate"|"watch"|"pass", "top_asset": "<ASSET>", "rank": ["<a1>","<a2>",...], "thesis": "<str>", "risks": "<str>"}}
"""


def format_fiat_trade(asset: str, price: float, spread_pct: float | None,
                      z_score: float, vwap_ratio: float | None,
                      volume_z: float | None, change_24h: float | None,
                      verdict: dict, confidence: int) -> None:
    fees     = fee_breakdown(NOTIONAL_USDT, 1, spread_pct)
    buy_qty  = round(NOTIONAL_USDT / price, 8) if price else 0

    print("\n╔══ FIAT BUY SIGNAL ═══════════════════════════════════════╗")
    print(f"  Mode       : USDT → {asset} (spot buy)")
    print(f"  Action     : Buy {asset}/USDT  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Pair       : {asset}/USDT  on Binance")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Price      : {price:,.4f} USDT")
    print(f"  Qty        : {buy_qty} {asset}  per {fees['notional']} USDT")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['total_fee']} USDT  (0.075% BNB, 1 leg)")
    print(f"  Spread     : {fees['spread_cost']} USDT  ({spread_pct:.4f}%)" if spread_pct else "  Spread     : n/a")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.3f}% of notional)")
    print(f"  Break-even : +{fees['breakeven_pct']:.3f}% price move to cover costs")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Signal     : Z={z_score:+.3f}  vwap={vwap_ratio:.4f}  vol_z={volume_z:+.3f}  24h={change_24h:+.2f}%"
          if all(v is not None for v in [vwap_ratio, volume_z, change_24h])
          else f"  Signal     : Z={z_score:+.3f}")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_fiat(con: duckdb.DuckDBPyConnection) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running fiat entry analysis …")
    candidates = get_fiat_candidates(con)

    if candidates.empty:
        print("  No assets statistically cheap vs history — no entry signal.")
        _log_verdict(con, {
            'action': 'pass', 'top_pair': 'none', 'thesis': 'No fiat entry signal.',
            'risks': '', 'prose': '', 'raw_json': ''
        }, 0, None, 'fiat')
        return

    print("\n── Fiat candidates ──")
    print(candidates[['asset_id', 'z_score', 'vwap_ratio',
                       'volume_z', 'spread_pct', 'change_24h']].to_string(index=False))

    prompt  = build_fiat_prompt(candidates)
    verdict = ask_qwen(prompt)

    # Remap top_asset → top_pair for storage consistency
    top_asset = verdict.get('top_asset', '')
    verdict['top_pair'] = f"{top_asset}/USDT" if top_asset else 'none'

    # Derive confidence from z_score spread between candidates
    if top_asset and top_asset in candidates['asset_id'].values:
        scores = candidates.set_index('asset_id')['z_score']
        winner = scores[top_asset]
        others = scores.drop(top_asset)
        gap    = abs(others.max() - winner) if not others.empty else 0
        conf   = 5 + min(int(gap / 0.25), 4)
        row    = candidates[candidates['asset_id'] == top_asset].iloc[0]
        if row['spread_pct'] is not None and row['spread_pct'] > 0.1:
            conf -= 1
        confidence = max(1, min(conf, 10))
    else:
        confidence = 5

    top_row = None
    if top_asset:
        match = candidates[candidates['asset_id'] == top_asset]
        if not match.empty:
            top_row = match.iloc[0]

    prices = get_current_prices(con)

    if top_row is not None and verdict.get('action') != 'pass':
        format_fiat_trade(
            asset      = top_asset,
            price      = prices.get(top_asset, top_row['price_usd']),
            spread_pct = top_row['spread_pct'],
            z_score    = top_row['z_score'],
            vwap_ratio = top_row['vwap_ratio'],
            volume_z   = top_row['volume_z'],
            change_24h = top_row['change_24h'],
            verdict    = verdict,
            confidence = confidence,
        )
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'fiat')


# ── LLM ───────────────────────────────────────────────────────────────────────

def ask_qwen(prompt: str) -> dict:
    fallback = {
        'action': 'pass', 'top_pair': '', 'top_asset': '',
        'rank': [], 'thesis': '', 'risks': '',
        'prose': 'LLM unavailable.', 'raw_json': ''
    }
    payload = {
        "model":       QWEN_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens":  512,
    }
    try:
        resp = requests.post(QWEN_URL, json=payload, timeout=90)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
    except Exception as e:
        fallback['prose'] = f"LLM request failed: {e}"
        return fallback

    lines, json_line, prose_lines, json_found = content.strip().splitlines(), "", [], False
    for line in lines:
        stripped = line.strip()
        if not json_found and stripped.startswith('{'):
            json_line  = stripped
            json_found = True
        else:
            prose_lines.append(line)

    result             = dict(fallback)
    result['raw_json'] = json_line
    result['prose']    = "\n".join(prose_lines).strip()

    if json_line:
        try:
            parsed = json.loads(json_line)
            for k in ('action', 'top_pair', 'top_asset', 'rank', 'thesis', 'risks'):
                if k in parsed:
                    result[k] = parsed[k]
        except json.JSONDecodeError as e:
            result['prose'] = f"[JSON parse error: {e}]\n" + result['prose']

    return result


def _log_verdict(con: duckdb.DuckDBPyConnection,
                 verdict: dict, confidence: int,
                 top_row, mode: str) -> None:
    ts       = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score  = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'

    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, mode, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ts, mode, top_pair, z_score, confidence,
        verdict.get('action',  'pass'),
        verdict.get('thesis',  ''),
        verdict.get('risks',   ''),
        verdict.get('prose',   ''),
        verdict.get('raw_json',''),
    ])


# ── Status ────────────────────────────────────────────────────────────────────

def status(con: duckdb.DuckDBPyConnection) -> None:
    rows    = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    verdicts = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    latest  = con.execute("SELECT MAX(timestamp) FROM market_history").fetchone()[0]
    print(f"── DB Status ──")
    print(f"  market_history : {rows} rows  (latest: {latest})")
    print(f"  llm_verdicts   : {verdicts} rows")

    print("\n── Sample counts per asset ──")
    print(con.execute("""
        SELECT asset_id, COUNT(*) as samples,
               ROUND(AVG(spread_pct),4) as avg_spread,
               ROUND(AVG(vwap_ratio),4) as avg_vwap
        FROM market_history
        GROUP BY asset_id ORDER BY asset_id
    """).df().to_string(index=False))

    print("\n── Last 5 verdicts ──")
    print(con.execute("""
        SELECT timestamp, mode, top_pair, z_score, confidence_score, action
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals (all time) ──")
    print(con.execute("""
        SELECT mode, top_pair,
               COUNT(*)                       AS hits,
               ROUND(AVG(confidence_score),1) AS avg_conf,
               ROUND(AVG(z_score),3)          AS avg_z
        FROM llm_verdicts
        WHERE action = 'accumulate'
        GROUP BY mode, top_pair
        ORDER BY hits DESC LIMIT 15
    """).df().to_string(index=False))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.4")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',  action='store_true', help='Ingest only')
    grp.add_argument('--analyze', action='store_true', help='Pair signal analysis')
    grp.add_argument('--fiat',    action='store_true', help='Fiat/USDT entry analysis')
    grp.add_argument('--init-db', action='store_true', help='(Re)create DB schema and exit')
    grp.add_argument('--status',  action='store_true', help='Quick DB health summary')
    args = parser.parse_args()

    con = get_conn()
    init_db(con)

    if args.init_db:
        print("DB schema initialized.")
    elif args.status:
        status(con)
    elif args.ingest:
        ingest(con)
    elif args.analyze:
        analyze_pairs(con)
    elif args.fiat:
        analyze_fiat(con)
    else:
        ingest(con)
        analyze_pairs(con)

    con.close()


if __name__ == "__main__":
    main()
