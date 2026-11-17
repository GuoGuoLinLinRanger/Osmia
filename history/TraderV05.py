#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.5
Unified ingest + analyze + paper trading script for the Osmia LLM trading monitor.

Usage:
    python osmia_trader.py              # ingest + analyze (pair mode)
    python osmia_trader.py --ingest     # ingest only
    python osmia_trader.py --analyze    # pair signal analysis
    python osmia_trader.py --fiat       # fiat/USDT entry analysis
    python osmia_trader.py --backfill   # one-shot OHLCV backfill (90 days, run once)
    python osmia_trader.py --outcomes   # score past signals vs actual price moves
    python osmia_trader.py --portfolio  # show fantasy portfolio state
    python osmia_trader.py --init-db    # (re)create DB schema and exit
    python osmia_trader.py --status     # quick DB health summary
    python osmia_trader.py --size 250   # override notional for this run

Changes from v0.4:
    - balances.json: real holdings, notional computed from TRADE_PCT of available balance
    - Fantasy portfolio: paper trades executed automatically on every signal ≥ confidence threshold
    - fantasy_trades table: full trade log with entry price, qty, fees, status
    - fantasy_balances table: running balance per asset after each paper trade
    - --outcomes: retroactive signal scoring — did the price ratio improve after the signal?
    - --backfill: fetch_ohlcv pulls 90 days of 1-min Binance history, populates market_history
      backfilled rows flagged with a 'source' column so live ticks are distinguishable
    - --portfolio: P&L summary of all open/closed fantasy positions
    - Fiat signal suppression warning when DB history < 14 days price range
    - Base dominance detection (mirrors quote dominance — catches BNB-as-base clustering)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import duckdb
import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH       = '/opt/llm_trader/market_data.duckdb'
BALANCES_FILE = '/opt/llm_trader/balances.json'
QWEN_URL      = "http://localhost:8081/v1/chat/completions"
QWEN_MODEL    = "qwen-instruct"

ASSETS  = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP',
           'TRX', 'BNB', 'AVAX', 'LTC', 'XLM', 'APT', 'DOGE', 'SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]

BINANCE_TAKER_FEE = 0.00075   # 0.075% BNB discount

BINANCE_DIRECT_PAIRS = {
    'ETH/BTC', 'BTC/ETH', 'SOL/BTC', 'BTC/SOL', 'SOL/ETH', 'ETH/SOL',
    'BNB/BTC', 'BTC/BNB', 'BNB/ETH', 'ETH/BNB', 'ADA/BTC', 'BTC/ADA',
    'ADA/ETH', 'ETH/ADA', 'XRP/BTC', 'BTC/XRP', 'XRP/ETH', 'ETH/XRP',
    'DOGE/BTC', 'BTC/DOGE', 'LTC/BTC', 'BTC/LTC', 'LTC/ETH', 'ETH/LTC',
    'DOT/BTC', 'BTC/DOT', 'DOT/ETH', 'ETH/DOT', 'AVAX/BTC', 'BTC/AVAX',
    'AVAX/ETH', 'ETH/AVAX',
}

MIN_PAIR_SAMPLES       = 20
VOLUME_WINDOW          = 50
ZSCORE_BUY_THRESH      = -2.0
TOP_N_RAW              = 10
TOP_N_LLM              = 3
MEME_ASSETS            = {'DOGE', 'XLM', 'TRX', 'XRP'}

TRADE_PCT              = 0.10   # fraction of available balance to use per signal
MIN_CONFIDENCE_TRADE   = 6      # minimum confidence to execute a fantasy trade
TRADE_COOLDOWN_MINUTES = 60     # minimum minutes between fantasy trades on same pair
OUTCOME_WINDOW_HOURS   = 24     # hours after signal to measure outcome
BACKFILL_DAYS          = 90
BACKFILL_TIMEFRAME     = '1m'

# ── Balances ──────────────────────────────────────────────────────────────────

DEFAULT_BALANCES = {
    "USDT": 0.0,
    "BTC":  0.0,
    "ETH":  0.0,
    "SOL":  0.0,
    "ADA":  0.0,
    "DOT":  0.0,
    "XRP":  0.0,
    "TRX":  0.0,
    "BNB":  0.0,
    "AVAX": 0.0,
    "LTC":  0.0,
    "XLM":  0.0,
    "APT":  0.0,
    "DOGE": 0.0,
    "SUI":  0.0,
}


def load_balances() -> dict[str, float]:
    p = Path(BALANCES_FILE)
    if not p.exists():
        print(f"  balances.json not found — creating template at {BALANCES_FILE}")
        p.write_text(json.dumps(DEFAULT_BALANCES, indent=2))
        print("  Edit it with your actual holdings, then re-run.")
        return dict(DEFAULT_BALANCES)
    with open(p) as f:
        data = json.load(f)
    # Fill any missing assets with 0
    for a in list(DEFAULT_BALANCES.keys()):
        data.setdefault(a, 0.0)
    return data


def notional_for(asset: str, price: float,
                 balances: dict[str, float],
                 override: float | None = None) -> float:
    """
    Compute trade notional in USDT.
    For pair trades: TRADE_PCT of asset's current USDT value.
    For fiat trades: TRADE_PCT of USDT balance.
    override: explicit USDT amount from --size flag.
    """
    if override is not None:
        return override
    holding = balances.get(asset, 0.0)
    value   = holding * price if asset != 'USDT' else holding
    notional = value * TRADE_PCT
    return max(notional, 1.0)   # floor at 1 USDT to avoid zero


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
    source         VARCHAR DEFAULT 'live',
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
    outcome_pct      DOUBLE,     -- actual price ratio change OUTCOME_WINDOW_HOURS after signal
    outcome_ts       TIMESTAMPTZ,
    PRIMARY KEY (timestamp, top_pair)
);
"""

DDL_FANTASY_TRADES = """
CREATE TABLE IF NOT EXISTS fantasy_trades (
    trade_id        VARCHAR     NOT NULL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    mode            VARCHAR     NOT NULL,
    action          VARCHAR     NOT NULL,   -- pair_reversion | fiat_buy
    sell_asset      VARCHAR,
    buy_asset       VARCHAR,
    sell_qty        DOUBLE,
    buy_qty         DOUBLE,
    sell_price_usd  DOUBLE,
    buy_price_usd   DOUBLE,
    notional_usd    DOUBLE,
    fee_usd         DOUBLE,
    spread_usd      DOUBLE,
    total_cost_usd  DOUBLE,
    signal_z        DOUBLE,
    confidence      INTEGER,
    status          VARCHAR DEFAULT 'open',  -- open | closed | expired
    close_ts        TIMESTAMPTZ,
    close_sell_px   DOUBLE,
    close_buy_px    DOUBLE,
    pnl_usd         DOUBLE,
    pnl_pct         DOUBLE
);
"""

DDL_FANTASY_BALANCES = """
CREATE TABLE IF NOT EXISTS fantasy_balances (
    timestamp   TIMESTAMPTZ NOT NULL,
    asset_id    VARCHAR     NOT NULL,
    qty         DOUBLE,
    price_usd   DOUBLE,
    value_usd   DOUBLE,
    PRIMARY KEY (timestamp, asset_id)
);
"""


# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL_MARKET_HISTORY)
    con.execute(DDL_LLM_VERDICTS)
    con.execute(DDL_FANTASY_TRADES)
    con.execute(DDL_FANTASY_BALANCES)
    # Migrations from v0.4
    _add_col_if_missing(con, 'llm_verdicts',   'outcome_pct',  'DOUBLE')
    _add_col_if_missing(con, 'llm_verdicts',   'outcome_ts',   'TIMESTAMPTZ')
    _add_col_if_missing(con, 'llm_verdicts',   'mode',         "VARCHAR DEFAULT 'pair'")
    _add_col_if_missing(con, 'market_history', 'source',       "VARCHAR DEFAULT 'live'")


def _add_col_if_missing(con, table: str, col: str, typedef: str) -> None:
    cols = con.execute(f"DESCRIBE {table}").df()['column_name'].tolist()
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")


def get_current_prices(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    df = con.execute("""
        SELECT asset_id, price_usd FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df()
    return dict(zip(df['asset_id'], df['price_usd']))


# ── Ingest ────────────────────────────────────────────────────────────────────

def _volume_z_scores(con: duckdb.DuckDBPyConnection,
                     new_df: pd.DataFrame) -> dict[str, float]:
    results = {}
    for _, row in new_df.iterrows():
        asset   = row['asset_id']
        cur_vol = row['volume_24h']
        hist    = con.execute(
            "SELECT volume_24h FROM market_history "
            "WHERE asset_id = ? AND source = 'live' ORDER BY timestamp DESC LIMIT ?",
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

    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
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
            'vwap_ratio':   round(last / vwap,             6) if vwap else None,
            'spread_pct':   round((ask - bid) / mid * 100, 6) if mid  else None,
            'high_low_pct': round((high - low) / low * 100, 6) if low  else None,
            'volume_z':     None,
            'source':       'live',
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
             market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source)
        SELECT timestamp, asset_id, price_usd, volume_24h, change_24h,
               market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source
        FROM df
    """)
    print(f"  Ingested {len(df)} records @ {ts:%Y-%m-%d %H:%M} UTC")
    return len(df)


# ── Backfill ──────────────────────────────────────────────────────────────────

def backfill(con: duckdb.DuckDBPyConnection) -> None:
    """
    One-shot: fetch 90 days of 1-min OHLCV from Binance for all assets.
    Inserts price + volume only (no spread/vwap — not available historically).
    Marks rows source='backfill' so live ticks are distinguishable.
    Safe to re-run — INSERT OR IGNORE skips existing rows.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int((datetime.now(timezone.utc) -
                    timedelta(days=BACKFILL_DAYS)).timestamp() * 1000)

    print(f"Backfilling {BACKFILL_DAYS} days of OHLCV for {len(ASSETS)} assets …")
    print("  This will take several minutes — Binance rate limits apply.\n")

    total = 0
    for symbol in SYMBOLS:
        asset = symbol.split('/')[0]
        print(f"  {asset:>6s} … ", end='', flush=True)
        rows    = []
        cursor  = since_ms
        retries = 0

        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe=BACKFILL_TIMEFRAME,
                    since=cursor, limit=1000
                )
            except Exception as e:
                if retries < 3:
                    retries += 1
                    time.sleep(5)
                    continue
                print(f"failed ({e})")
                break

            if not candles:
                break

            for c in candles:
                ts = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
                ts = ts.replace(second=0, microsecond=0)
                rows.append({
                    'timestamp':    ts,
                    'asset_id':     asset,
                    'price_usd':    float(c[4]),   # close
                    'volume_24h':   float(c[5]),   # volume (candle, not 24h — proxy)
                    'change_24h':   None,
                    'market_cap':   0.0,
                    'vwap_ratio':   None,
                    'volume_z':     None,
                    'spread_pct':   None,
                    'high_low_pct': round((float(c[2]) - float(c[3])) /
                                         float(c[3]) * 100, 6) if c[3] else None,
                    'source':       'backfill',
                })

            cursor = candles[-1][0] + 1
            if len(candles) < 1000:
                break
            time.sleep(0.2)   # be kind to the API

        if rows:
            df = pd.DataFrame(rows)
            con.execute("""
                INSERT OR IGNORE INTO market_history
                    (timestamp, asset_id, price_usd, volume_24h, change_24h,
                     market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source)
                SELECT timestamp, asset_id, price_usd, volume_24h, change_24h,
                       market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source
                FROM df
            """)
            print(f"{len(rows):,} rows")
            total += len(rows)
        else:
            print("0 rows")

    total_db = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    print(f"\nBackfill complete. {total:,} rows inserted. DB total: {total_db:,}")


# ── Fee / trade helpers ───────────────────────────────────────────────────────

def trade_route(base: str, quote: str) -> tuple[str, int]:
    direct = f"{base}/{quote}"
    if direct in BINANCE_DIRECT_PAIRS:
        return f"{base}/{quote} (direct)", 1
    return f"{base}/USDT → {quote}/USDT", 2


def fee_breakdown(notional: float, num_legs: int,
                  spread_pct: float | None) -> dict:
    fee_per_leg   = notional * BINANCE_TAKER_FEE
    total_fee     = fee_per_leg * num_legs
    spread_cost   = notional * (spread_pct / 100) if spread_pct else 0.0
    total_cost    = total_fee + spread_cost
    breakeven_pct = (total_cost / notional) * 100
    return {
        'notional':      round(notional, 4),
        'fee_per_leg':   round(fee_per_leg, 6),
        'num_legs':      num_legs,
        'total_fee':     round(total_fee, 6),
        'spread_cost':   round(spread_cost, 6),
        'total_cost':    round(total_cost, 6),
        'breakeven_pct': round(breakeven_pct, 6),
    }


# ── Signal engine ─────────────────────────────────────────────────────────────

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


def get_pair_persistence(con, pairs):
    if not pairs:
        return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT top_pair, COUNT(*) as hits FROM llm_verdicts
        WHERE top_pair IN ({ph}) AND action = 'accumulate'
          AND timestamp >= NOW() - INTERVAL 24 HOURS
        GROUP BY top_pair
    """, pairs).df()
    return dict(zip(df['top_pair'], df['hits'])) if not df.empty else {}


def get_prev_z_scores(con, pairs):
    if not pairs:
        return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT DISTINCT ON (top_pair) top_pair, z_score FROM llm_verdicts
        WHERE top_pair IN ({ph})
        ORDER BY top_pair, timestamp DESC
    """, pairs).df()
    return dict(zip(df['top_pair'], df['z_score'])) if not df.empty else {}


def detect_dominance(df: pd.DataFrame, col: str) -> str | None:
    counts   = df[col].value_counts()
    dominant = counts[counts >= 3]
    return dominant.index[0] if not dominant.empty else None


def rule_based_score(row, persistence, prev_z) -> float:
    score = row['z_score']
    score -= min(persistence.get(row['pair'], 0) * 0.1, 0.5)
    prev = prev_z.get(row['pair'])
    if prev is not None:
        score -= 0.2 if row['z_score'] < prev else -0.2
    if row['volume_z']   is not None and row['volume_z']   >  1.5: score -= 0.15
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99:  score -= 0.1
    if row['spread_pct'] is not None and row['spread_pct'] >  0.1:  score += 0.3
    if row['quote_asset'] in MEME_ASSETS:                            score += 0.5
    return round(score, 4)


def build_candidates(con) -> tuple[pd.DataFrame, str | None, str | None]:
    """Returns (candidates, quote_dominance, base_dominance)."""
    raw = get_raw_anomalies(con)
    if raw.empty:
        return raw, None, None
    pairs       = raw['pair'].tolist()
    persistence = get_pair_persistence(con, pairs)
    prev_z      = get_prev_z_scores(con, pairs)
    raw['persistence_24h'] = raw['pair'].map(lambda p: persistence.get(p, 0))
    raw['z_velocity']      = raw['pair'].map(
        lambda p: round(raw.loc[raw['pair'] == p, 'z_score'].iloc[0] - prev_z[p], 3)
        if p in prev_z else None
    )
    raw['rule_score'] = raw.apply(lambda r: rule_based_score(r, persistence, prev_z), axis=1)
    quote_warn = detect_dominance(raw, 'quote_asset')
    base_warn  = detect_dominance(raw, 'base_asset')
    top = raw.sort_values('rule_score').head(TOP_N_LLM).reset_index(drop=True)
    return top, quote_warn, base_warn


def derive_confidence(candidates: pd.DataFrame, top_pair: str) -> int:
    if candidates.empty or top_pair not in candidates['pair'].values:
        return 5
    scores    = candidates.set_index('pair')['rule_score']
    winner    = scores[top_pair]
    others    = scores.drop(top_pair)
    gap       = abs(others.min() - winner) if not others.empty else 0
    conf      = 5 + min(int(gap / 0.25), 4)
    row       = candidates[candidates['pair'] == top_pair].iloc[0]
    if row['spread_pct'] is not None and row['spread_pct'] > 0.1: conf -= 1
    if row['quote_asset'] in MEME_ASSETS:                          conf -= 1
    return max(1, min(conf, 10))


# ── Fantasy portfolio ─────────────────────────────────────────────────────────

def _trade_id(ts: datetime, pair: str) -> str:
    return f"{ts.strftime('%Y%m%d%H%M')}_{pair.replace('/', '_')}"


def _snapshot_balances(con: duckdb.DuckDBPyConnection,
                       balances: dict[str, float],
                       prices: dict[str, float]) -> None:
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for asset, qty in balances.items():
        px  = prices.get(asset, 1.0) if asset != 'USDT' else 1.0
        rows.append({'timestamp': ts, 'asset_id': asset,
                     'qty': qty, 'price_usd': px, 'value_usd': qty * px})
    df = pd.DataFrame(rows)
    con.execute("""
        INSERT OR IGNORE INTO fantasy_balances
            (timestamp, asset_id, qty, price_usd, value_usd)
        SELECT timestamp, asset_id, qty, price_usd, value_usd FROM df
    """)


def _pair_on_cooldown(con: duckdb.DuckDBPyConnection,
                      sell_asset: str, buy_asset: str) -> bool:
    """
    Returns True if a fantasy trade on this pair was executed
    within the last TRADE_COOLDOWN_MINUTES. Prevents re-entering
    the same signal every 4 minutes.
    """
    pair = f"{sell_asset}/{buy_asset}"
    result = con.execute(f"""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE sell_asset = ? AND buy_asset = ?
          AND timestamp >= NOW() - INTERVAL {TRADE_COOLDOWN_MINUTES} MINUTES
          AND status = 'open'
    """, [sell_asset, buy_asset]).fetchone()
    return result[0] > 0


def execute_fantasy_pair_trade(con: duckdb.DuckDBPyConnection,
                                sell_asset: str, buy_asset: str,
                                sell_px: float, buy_px: float,
                                notional: float, spread_pct: float | None,
                                z_score: float, confidence: int,
                                balances: dict[str, float]) -> dict | None:
    """
    Paper-executes a pair trade. Updates balances in memory.
    Returns trade record or None if insufficient balance.
    """
    holding_usd = balances.get(sell_asset, 0.0) * sell_px
    if holding_usd < notional * 0.5:
        return None   # not enough to trade — skip silently

    _, legs   = trade_route(sell_asset, buy_asset)
    fees      = fee_breakdown(notional, legs, spread_pct)
    sell_qty  = notional / sell_px  if sell_px  else 0
    buy_qty   = (notional - fees['total_cost']) / buy_px if buy_px else 0

    # Update fantasy balances
    balances[sell_asset] = max(0.0, balances.get(sell_asset, 0.0) - sell_qty)
    balances[buy_asset]  = balances.get(buy_asset, 0.0) + buy_qty

    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{sell_asset}/{buy_asset}"
    return {
        'trade_id':       _trade_id(ts, pair),
        'timestamp':      ts,
        'mode':           'pair',
        'action':         'pair_reversion',
        'sell_asset':     sell_asset,
        'buy_asset':      buy_asset,
        'sell_qty':       round(sell_qty,  8),
        'buy_qty':        round(buy_qty,   8),
        'sell_price_usd': sell_px,
        'buy_price_usd':  buy_px,
        'notional_usd':   fees['notional'],
        'fee_usd':        fees['total_fee'],
        'spread_usd':     fees['spread_cost'],
        'total_cost_usd': fees['total_cost'],
        'signal_z':       z_score,
        'confidence':     confidence,
        'status':         'open',
    }


def execute_fantasy_fiat_trade(con: duckdb.DuckDBPyConnection,
                                buy_asset: str, buy_px: float,
                                notional: float, spread_pct: float | None,
                                z_score: float, confidence: int,
                                balances: dict[str, float]) -> dict | None:
    usdt = balances.get('USDT', 0.0)
    if usdt < notional * 0.5:
        return None

    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = (notional - fees['total_cost']) / buy_px if buy_px else 0

    balances['USDT']    = max(0.0, usdt - notional)
    balances[buy_asset] = balances.get(buy_asset, 0.0) + buy_qty

    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{buy_asset}/USDT"
    return {
        'trade_id':       _trade_id(ts, pair),
        'timestamp':      ts,
        'mode':           'fiat',
        'action':         'fiat_buy',
        'sell_asset':     'USDT',
        'buy_asset':      buy_asset,
        'sell_qty':       round(notional, 6),
        'buy_qty':        round(buy_qty,  8),
        'sell_price_usd': 1.0,
        'buy_price_usd':  buy_px,
        'notional_usd':   fees['notional'],
        'fee_usd':        fees['total_fee'],
        'spread_usd':     fees['spread_cost'],
        'total_cost_usd': fees['total_cost'],
        'signal_z':       z_score,
        'confidence':     confidence,
        'status':         'open',
    }


def _insert_trade(con: duckdb.DuckDBPyConnection, trade: dict) -> None:
    con.execute("""
        INSERT OR IGNORE INTO fantasy_trades
            (trade_id, timestamp, mode, action, sell_asset, buy_asset,
             sell_qty, buy_qty, sell_price_usd, buy_price_usd, notional_usd,
             fee_usd, spread_usd, total_cost_usd, signal_z, confidence, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        trade['trade_id'], trade['timestamp'], trade['mode'], trade['action'],
        trade['sell_asset'], trade['buy_asset'],
        trade['sell_qty'], trade['buy_qty'],
        trade['sell_price_usd'], trade['buy_price_usd'],
        trade['notional_usd'], trade['fee_usd'], trade['spread_usd'],
        trade['total_cost_usd'], trade['signal_z'], trade['confidence'],
        trade['status'],
    ])


# ── Outcome tracking ──────────────────────────────────────────────────────────

def score_outcomes(con: duckdb.DuckDBPyConnection) -> None:
    """
    For each verdict with action=accumulate and no outcome_pct yet,
    check if OUTCOME_WINDOW_HOURS of market_history data now exists.
    If so, compute actual price ratio change and update the verdict.
    Also marks open fantasy_trades as closed/profitable.
    """
    print(f"[{datetime.now():%H:%M:%S}] Scoring outcomes …")

    pending = con.execute(f"""
        SELECT timestamp, top_pair, mode, z_score
        FROM llm_verdicts
        WHERE action = 'accumulate'
          AND outcome_pct IS NULL
          AND timestamp <= NOW() - INTERVAL {OUTCOME_WINDOW_HOURS} HOURS
    """).df()

    if pending.empty:
        print(f"  No pending outcomes (need {OUTCOME_WINDOW_HOURS}h of history after signal).")
        return

    prices_cache = {}
    scored = 0

    for _, row in pending.iterrows():
        pair    = row['top_pair']
        sig_ts  = row['timestamp']
        out_ts  = sig_ts + timedelta(hours=OUTCOME_WINDOW_HOURS)

        if '/' not in str(pair):
            continue

        parts = pair.split('/')
        if len(parts) != 2:
            continue

        base, quote = parts[0], parts[1].replace('USDT', '') or 'USDT'

        # For fiat mode top_pair is like "ETH/USDT" — quote is USDT
        if row['mode'] == 'fiat':
            # Z-score improvement: price went up after buy signal
            entry_px = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
            """, [base, sig_ts]).fetchone()
            exit_px  = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC LIMIT 1
            """, [base, out_ts]).fetchone()
            if not entry_px or not exit_px:
                continue
            outcome_pct = round((exit_px[0] - entry_px[0]) / entry_px[0] * 100, 4)
        else:
            # Pair mode: did the ratio improve (base got cheaper vs quote)?
            entry_base = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
            """, [base, sig_ts]).fetchone()
            entry_quote = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
            """, [quote, sig_ts]).fetchone()
            exit_base = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC LIMIT 1
            """, [base, out_ts]).fetchone()
            exit_quote = con.execute("""
                SELECT price_usd FROM market_history
                WHERE asset_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC LIMIT 1
            """, [quote, out_ts]).fetchone()
            if not all([entry_base, entry_quote, exit_base, exit_quote]):
                continue
            entry_ratio = entry_base[0] / entry_quote[0]
            exit_ratio  = exit_base[0]  / exit_quote[0]
            outcome_pct = round((exit_ratio - entry_ratio) / entry_ratio * 100, 4)

        con.execute("""
            UPDATE llm_verdicts
            SET outcome_pct = ?, outcome_ts = ?
            WHERE timestamp = ? AND top_pair = ?
        """, [outcome_pct, out_ts, sig_ts, pair])
        scored += 1

    print(f"  Scored {scored} outcomes.")

    # Summary
    summary = con.execute("""
        SELECT
            COUNT(*)                                          AS total,
            SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS winners,
            ROUND(AVG(outcome_pct), 3)                        AS avg_outcome_pct,
            ROUND(MAX(outcome_pct), 3)                        AS best,
            ROUND(MIN(outcome_pct), 3)                        AS worst
        FROM llm_verdicts
        WHERE action = 'accumulate' AND outcome_pct IS NOT NULL
    """).fetchone()

    if summary and summary[0] > 0:
        win_rate = round(summary[1] / summary[0] * 100, 1)
        print(f"\n── Signal accuracy ({OUTCOME_WINDOW_HOURS}h window) ──")
        print(f"  Signals scored  : {summary[0]}")
        print(f"  Win rate        : {summary[1]}/{summary[0]} ({win_rate}%)")
        print(f"  Avg outcome     : {summary[2]:+.3f}%")
        print(f"  Best / Worst    : {summary[3]:+.3f}% / {summary[4]:+.3f}%")


# ── Portfolio summary ─────────────────────────────────────────────────────────

def show_portfolio(con: duckdb.DuckDBPyConnection) -> None:
    prices = get_current_prices(con)

    print("── Fantasy Portfolio ──")
    trades = con.execute("""
        SELECT trade_id, timestamp, action, sell_asset, buy_asset,
               sell_qty, buy_qty, sell_price_usd, buy_price_usd,
               notional_usd, total_cost_usd, signal_z, confidence, status
        FROM fantasy_trades
        ORDER BY timestamp DESC
    """).df()

    if trades.empty:
        print("  No fantasy trades executed yet.")
        return

    # Mark each open trade with current P&L
    rows_out = []
    for _, t in trades.iterrows():
        if t['status'] == 'open':
            cur_sell_px = prices.get(t['sell_asset'], t['sell_price_usd'])
            cur_buy_px  = prices.get(t['buy_asset'],  t['buy_price_usd'])
            # Value of what we bought now vs what we sold
            current_val = t['buy_qty'] * cur_buy_px
            cost_basis  = t['notional_usd']
            pnl_usd     = round(current_val - cost_basis, 4)
            pnl_pct     = round(pnl_usd / cost_basis * 100, 3) if cost_basis else 0
        else:
            pnl_usd = t.get('pnl_usd', 0.0)
            pnl_pct = t.get('pnl_pct', 0.0)

        rows_out.append({
            'ts':        str(t['timestamp'])[:16],
            'action':    t['action'],
            'trade':     f"{t['sell_asset']}→{t['buy_asset']}",
            'notional':  f"{t['notional_usd']:.2f}",
            'cost':      f"{t['total_cost_usd']:.4f}",
            'z':         f"{t['signal_z']:+.3f}",
            'conf':      t['confidence'],
            'status':    t['status'],
            'pnl_usd':   f"{pnl_usd:+.4f}",
            'pnl_pct':   f"{pnl_pct:+.3f}%",
        })

    df_out = pd.DataFrame(rows_out)
    print(df_out.to_string(index=False))

    # Totals
    open_trades = [r for r in rows_out if r['status'] == 'open']
    total_pnl   = sum(float(r['pnl_usd']) for r in rows_out)
    print(f"\n  Open positions  : {len(open_trades)}")
    print(f"  Total trades    : {len(rows_out)}")
    print(f"  Total P&L       : {total_pnl:+.4f} USDT (unrealized on open positions)")

    # Latest fantasy balance snapshot
    latest_bal = con.execute("""
        SELECT asset_id, qty, price_usd, value_usd
        FROM fantasy_balances
        WHERE timestamp = (SELECT MAX(timestamp) FROM fantasy_balances)
        ORDER BY value_usd DESC
    """).df()
    if not latest_bal.empty:
        total_val = latest_bal['value_usd'].sum()
        print(f"\n── Fantasy Balances (latest snapshot) ──")
        print(latest_bal[latest_bal['qty'] > 0].to_string(index=False))
        print(f"\n  Total portfolio value : {total_val:.4f} USDT")


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

    lines, json_line, prose_lines, found = content.strip().splitlines(), "", [], False
    for line in lines:
        s = line.strip()
        if not found and s.startswith('{'):
            json_line = s
            found     = True
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


def _log_verdict(con, verdict, confidence, top_row, mode) -> None:
    ts       = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score  = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'
    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, mode, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, [
        ts, mode, top_pair, z_score, confidence,
        verdict.get('action',   'pass'),
        verdict.get('thesis',   ''),
        verdict.get('risks',    ''),
        verdict.get('prose',    ''),
        verdict.get('raw_json', ''),
    ])


# ── Pair analysis ─────────────────────────────────────────────────────────────

def build_pair_prompt(candidates, quote_warn, base_warn) -> str:
    rows = []
    for i, r in candidates.iterrows():
        vel    = f"{r['z_velocity']:+.3f}" if r['z_velocity'] is not None else "n/a"
        spread = f"{r['spread_pct']:.4f}%" if r['spread_pct'] is not None else "n/a"
        vwap   = f"{r['vwap_ratio']:.4f}"  if r['vwap_ratio'] is not None else "n/a"
        vol_z  = f"{r['volume_z']:+.3f}"   if r['volume_z']   is not None else "n/a"
        route, _ = trade_route(r['base_asset'], r['quote_asset'])
        rows.append(
            f"  [rank {i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel} | "
            f"persist={r['persistence_24h']} | vwap={vwap} | vol_z={vol_z} | "
            f"spread={spread} | route={route}"
        )
    table = "\n".join(rows)
    warn  = ""
    if quote_warn:
        warn += (f"\n⚠ QUOTE DOMINANCE: {quote_warn} is the quote in multiple pairs — "
                 f"may mean {quote_warn} is overbought. Address in thesis.\n")
    if base_warn:
        warn += (f"\n⚠ BASE DOMINANCE: {base_warn} is the base in multiple pairs — "
                 f"this is likely one signal ({base_warn} weak), not independent signals.\n")

    return f"""You are a disciplined spot-only crypto trader. Rule-based engine pre-ranked {len(candidates)} candidates. Validate rank 1 or override with justification.

Candidates (best first):
{table}
{warn}
Definitions: z=price-ratio Z-score, vel=Z delta since last analysis (negative=deepening), persist=24h hit count, vwap=last/VWAP, vol_z=volume anomaly, spread=bid-ask %.

Rules:
- Rule rank is authoritative. Override requires explicit justification in thesis.
- Do NOT assign equal confidence to all. Score reflects spread between rule_scores.
- action: accumulate | watch | pass. spread >0.15% forces confidence <=6.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_pair":"<PAIR>","rank":["<p1>",...],"thesis":"<str>","risks":"<str>"}}
"""


def format_pair_trade(top_row, prices, verdict, confidence, notional) -> None:
    base, quote = top_row['base_asset'], top_row['quote_asset']
    base_px     = prices.get(base,  0.0)
    quote_px    = prices.get(quote, 0.0)
    route, legs = trade_route(base, quote)
    fees        = fee_breakdown(notional, legs, top_row['spread_pct'])
    sell_qty    = round(notional / base_px,  8) if base_px  else 0
    buy_qty     = round(notional / quote_px, 8) if quote_px else 0

    print("\n╔══ TRADE SIGNAL ══════════════════════════════════════════╗")
    print(f"  Mode       : PAIR REVERSION (spot)")
    print(f"  Action     : Sell {base} → Buy {quote}  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Route      : {route}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Sell       : {sell_qty} {base}  @ {base_px:,.4f} USDT")
    print(f"  Buy        : {buy_qty} {quote}  @ {quote_px:,.4f} USDT")
    print(f"  Notional   : {fees['notional']} USDT  (10% of {base} holding)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['fee_per_leg']} USDT × {fees['num_legs']} legs = {fees['total_fee']} USDT  (0.075% BNB)")
    if top_row['spread_pct']:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({top_row['spread_pct']:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of notional)")
    print(f"  Break-even : ratio must improve by {fees['breakeven_pct']:.4f}%")
    print(f"  ─────────────────────────────────────────────────────")
    vel = top_row['z_velocity'] if top_row['z_velocity'] is not None else 'n/a'
    print(f"  Signal     : Z={top_row['z_score']:+.3f}  persist={top_row['persistence_24h']}  vel={vel}")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_pairs(con: duckdb.DuckDBPyConnection,
                  size_override: float | None = None) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running pair analysis …")
    candidates, quote_warn, base_warn = build_candidates(con)

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'pair')
        return

    if quote_warn:
        print(f"\n  ⚠ Quote dominance: {quote_warn}")
    if base_warn:
        print(f"  ⚠ Base dominance : {base_warn} — likely one signal dressed as multiple")

    print("\n── Candidates (rule-scored) ──")
    print(candidates[['pair', 'z_score', 'z_velocity', 'persistence_24h',
                       'vwap_ratio', 'volume_z', 'spread_pct', 'rule_score']].to_string(index=False))

    prompt     = build_pair_prompt(candidates, quote_warn, base_warn)
    verdict    = ask_qwen(prompt)
    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair)
    prices     = get_current_prices(con)

    top_row = None
    if top_pair:
        match = candidates[candidates['pair'] == top_pair]
        if not match.empty:
            top_row = match.iloc[0]

    balances = load_balances()

    if top_row is not None and verdict.get('action') != 'pass':
        base    = top_row['base_asset']
        base_px = prices.get(base, 0.0)
        notional = notional_for(base, base_px, balances, size_override)
        format_pair_trade(top_row, prices, verdict, confidence, notional)

        if confidence >= MIN_CONFIDENCE_TRADE:
            quote = top_row['quote_asset']
            if _pair_on_cooldown(con, base, quote):
                print(f"\n  📒 Fantasy trade skipped — {base}/{quote} on {TRADE_COOLDOWN_MINUTES}min cooldown")
            else:
                trade = execute_fantasy_pair_trade(
                    con, base, quote, base_px, prices.get(quote, 0.0),
                    notional, top_row['spread_pct'],
                    top_row['z_score'], confidence, balances
                )
                if trade:
                    _insert_trade(con, trade)
                    _snapshot_balances(con, balances, prices)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient {base} balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'pair')


# ── Fiat analysis ─────────────────────────────────────────────────────────────

def get_fiat_candidates(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, bool]:
    # Warn if history < 14 days
    date_range = con.execute("""
        SELECT DATEDIFF('day', MIN(timestamp), MAX(timestamp))
        FROM market_history WHERE source = 'live'
    """).fetchone()[0] or 0
    short_history = date_range < 14

    query = f"""
    WITH stats AS (
        SELECT asset_id,
               AVG(price_usd)    AS mean_price,
               STDDEV(price_usd) AS std_price,
               COUNT(*)          AS sample_count
        FROM market_history
        GROUP BY asset_id HAVING sample_count > {MIN_PAIR_SAMPLES}
    ),
    latest AS (
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct, change_24h
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    )
    SELECT l.asset_id,
           ROUND(l.price_usd, 6)                                            AS price_usd,
           ROUND(s.mean_price, 6)                                           AS mean_price,
           ROUND((l.price_usd - s.mean_price) / NULLIF(s.std_price, 0), 3) AS z_score,
           ROUND(l.vwap_ratio, 4) AS vwap_ratio,
           ROUND(l.volume_z,   3) AS volume_z,
           ROUND(l.spread_pct, 4) AS spread_pct,
           ROUND(l.change_24h, 3) AS change_24h,
           s.sample_count
    FROM latest l JOIN stats s ON l.asset_id = s.asset_id
    WHERE (l.price_usd - s.mean_price) / NULLIF(s.std_price, 0) < {ZSCORE_BUY_THRESH}
    ORDER BY z_score ASC LIMIT {TOP_N_LLM}
    """
    return con.execute(query).df(), short_history


def build_fiat_prompt(candidates) -> str:
    rows = []
    for i, r in candidates.iterrows():
        rows.append(
            f"  [rank {i+1}] {r['asset_id']:>6s}/USDT | price={r['price_usd']:,.4f} | "
            f"z={r['z_score']:+.3f} | vwap={r['vwap_ratio'] or 'n/a'} | "
            f"vol_z={r['volume_z'] or 'n/a'} | "
            f"24h={r['change_24h']:+.2f}% | spread={r['spread_pct'] or 'n/a'}%"
        )
    table = "\n".join(rows)
    return f"""You are a disciplined spot-only crypto trader advising on USDT buy entries.
Statistically cheap assets (Z < {ZSCORE_BUY_THRESH}):
{table}

Rules: spot only, no leverage. Avoid assets still in freefall (very negative 24h, no volume support).
Rank all. Do not assign equal confidence. action: accumulate|watch|pass. thesis/risks max 120 chars.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_asset":"<ASSET>","rank":["<a1>",...],"thesis":"<str>","risks":"<str>"}}
"""


def format_fiat_trade(asset, price, spread_pct, z_score,
                      vwap_ratio, volume_z, change_24h,
                      verdict, confidence, notional) -> None:
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = round((notional - fees['total_cost']) / price, 8) if price else 0

    print("\n╔══ FIAT BUY SIGNAL ═══════════════════════════════════════╗")
    print(f"  Mode       : USDT → {asset} (spot buy)")
    print(f"  Action     : Buy {asset}/USDT  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Price      : {price:,.4f} USDT")
    print(f"  Qty        : {buy_qty} {asset}  from {fees['notional']} USDT")
    print(f"  Notional   : {fees['notional']} USDT  (10% of USDT balance)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['total_fee']} USDT  (0.075% BNB, 1 leg)")
    if spread_pct:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({spread_pct:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of notional)")
    print(f"  Break-even : +{fees['breakeven_pct']:.4f}% price move")
    print(f"  ─────────────────────────────────────────────────────")
    sig_parts = [f"Z={z_score:+.3f}"]
    if vwap_ratio:  sig_parts.append(f"vwap={vwap_ratio:.4f}")
    if volume_z:    sig_parts.append(f"vol_z={volume_z:+.3f}")
    if change_24h:  sig_parts.append(f"24h={change_24h:+.2f}%")
    print(f"  Signal     : {' '.join(sig_parts)}")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_fiat(con: duckdb.DuckDBPyConnection,
                 size_override: float | None = None) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running fiat entry analysis …")
    candidates, short_history = get_fiat_candidates(con)

    if short_history:
        print("  ⚠ DB history < 14 days — fiat Z-scores may not reflect true price range yet")

    if candidates.empty:
        print("  No assets statistically cheap vs history — no entry signal.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'No fiat entry signal.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'fiat')
        return

    print("\n── Fiat candidates ──")
    print(candidates[['asset_id', 'z_score', 'vwap_ratio',
                       'volume_z', 'spread_pct', 'change_24h']].to_string(index=False))

    prompt    = build_fiat_prompt(candidates)
    verdict   = ask_qwen(prompt)
    top_asset = verdict.get('top_asset', '')
    verdict['top_pair'] = f"{top_asset}/USDT" if top_asset else 'none'

    confidence = 5
    top_row    = None
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
        top_row    = row

    prices   = get_current_prices(con)
    balances = load_balances()

    if top_row is not None and verdict.get('action') != 'pass':
        px       = prices.get(top_asset, top_row['price_usd'])
        notional = notional_for('USDT', 1.0, balances, size_override)
        format_fiat_trade(
            top_asset, px, top_row['spread_pct'], top_row['z_score'],
            top_row['vwap_ratio'], top_row['volume_z'], top_row['change_24h'],
            verdict, confidence, notional
        )
        if confidence >= MIN_CONFIDENCE_TRADE:
            if _pair_on_cooldown(con, 'USDT', top_asset):
                print(f"\n  📒 Fantasy trade skipped — {top_asset}/USDT on {TRADE_COOLDOWN_MINUTES}min cooldown")
            else:
                trade = execute_fantasy_fiat_trade(
                    con, top_asset, px, notional, top_row['spread_pct'],
                    top_row['z_score'], confidence, balances
                )
                if trade:
                    _insert_trade(con, trade)
                    _snapshot_balances(con, balances, prices)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient USDT balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")

    _log_verdict(con, verdict, confidence, top_row, 'fiat')


# ── Status ────────────────────────────────────────────────────────────────────

def status(con: duckdb.DuckDBPyConnection) -> None:
    rows     = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    live     = con.execute("SELECT COUNT(*) FROM market_history WHERE source='live'").fetchone()[0]
    backfill = con.execute("SELECT COUNT(*) FROM market_history WHERE source='backfill'").fetchone()[0]
    verdicts = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    trades   = con.execute("SELECT COUNT(*) FROM fantasy_trades").fetchone()[0]
    latest   = con.execute("SELECT MAX(timestamp) FROM market_history").fetchone()[0]
    print(f"── DB Status ──")
    print(f"  market_history : {rows} rows  (live={live}, backfill={backfill}, latest={latest})")
    print(f"  llm_verdicts   : {verdicts} rows")
    print(f"  fantasy_trades : {trades} rows")

    print("\n── Live sample counts per asset ──")
    print(con.execute("""
        SELECT asset_id, COUNT(*) as samples,
               ROUND(AVG(spread_pct),4) as avg_spread,
               ROUND(AVG(vwap_ratio),4) as avg_vwap
        FROM market_history WHERE source='live'
        GROUP BY asset_id ORDER BY asset_id
    """).df().to_string(index=False))

    print("\n── Last 5 verdicts ──")
    print(con.execute("""
        SELECT timestamp, mode, top_pair, z_score, confidence_score, action, outcome_pct
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals ──")
    print(con.execute("""
        SELECT mode, top_pair, COUNT(*) AS hits,
               ROUND(AVG(confidence_score),1) AS avg_conf,
               ROUND(AVG(z_score),3) AS avg_z,
               ROUND(AVG(outcome_pct),3) AS avg_outcome_pct
        FROM llm_verdicts WHERE action='accumulate'
        GROUP BY mode, top_pair ORDER BY hits DESC LIMIT 15
    """).df().to_string(index=False))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.5")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',    action='store_true')
    grp.add_argument('--analyze',   action='store_true')
    grp.add_argument('--fiat',      action='store_true')
    grp.add_argument('--backfill',  action='store_true', help='One-shot 90-day OHLCV backfill')
    grp.add_argument('--outcomes',  action='store_true', help='Score past signals vs actual price')
    grp.add_argument('--portfolio', action='store_true', help='Fantasy portfolio P&L')
    grp.add_argument('--init-db',   action='store_true')
    grp.add_argument('--status',    action='store_true')
    parser.add_argument('--size', type=float, default=None,
                        help='Override notional USDT size for this run')
    args = parser.parse_args()

    con = get_conn()
    init_db(con)

    if   args.init_db:   print("DB schema initialized.")
    elif args.status:    status(con)
    elif args.backfill:  backfill(con)
    elif args.outcomes:  score_outcomes(con)
    elif args.portfolio: show_portfolio(con)
    elif args.ingest:    ingest(con)
    elif args.analyze:   analyze_pairs(con, args.size)
    elif args.fiat:      analyze_fiat(con, args.size)
    else:
        ingest(con)
        analyze_pairs(con, args.size)

    con.close()


if __name__ == "__main__":
    main()
