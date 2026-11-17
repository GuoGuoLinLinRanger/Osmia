#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.7
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
    python osmia_trader.py --daily-pnl  # send daily P&L summary to Discord
    python osmia_trader.py --size 250   # override notional for this run

Changes from v0.6:
    - Discord messages rewritten in plain English — no raw scores exposed
      Z-score → "X std devs below average", rep → "X% historical win rate",
      velocity → "accelerating/decelerating", conviction → Low/Moderate/Strong/Very Strong
      Mean reversion target price shown, break-even in plain terms
    - Market context agent: Fear & Greed index (alternative.me) + macro headlines
      (CoinDesk RSS) fetched at analysis time, fed to Qwen, stored per verdict
    - Daily P&L Discord: portfolio value delta, best/worst position, signal count
    - DB cleanup: /USDT/USDT dedup on init_db, fix in analyze_fiat root cause
    - Exit strategy: 48h max hold expiry on fantasy trades, Z crossback close
      (Z > -1.0 after entry), min 3 open trades per pair hard limit
    - Signal-weighted position sizing: base 5% + rep bonus ±5% + Z bonus up to 5%
      Floor 3%, ceiling 15%, max 3 open fantasy trades per pair
    - market_context table: F&G + headlines stored per analysis run
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import duckdb
import numpy as np
import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH          = '/opt/llm_trader/market_data.duckdb'
BALANCES_FILE    = '/opt/llm_trader/balances.json'
QWEN_URL         = "http://localhost:8081/v1/chat/completions"
QWEN_MODEL       = "qwen-instruct"
DISCORD_WEBHOOK  = ""   # paste your webhook URL here, leave empty to disable

ASSETS  = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP',
           'TRX', 'BNB', 'AVAX', 'LTC', 'XLM', 'APT', 'DOGE', 'SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]

BINANCE_TAKER_FEE = 0.00075

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
ZSCORE_CLOSE_THRESH    = -1.0   # Z crossback — signal has reverted enough to close
TOP_N_RAW              = 10
TOP_N_LLM              = 3
MEME_ASSETS            = {'DOGE', 'XLM', 'TRX', 'XRP'}

MIN_CONFIDENCE_TRADE   = 6
TRADE_COOLDOWN_MINUTES = 60
MAX_TRADES_PER_PAIR    = 3      # hard limit on open fantasy trades per pair
MAX_HOLD_HOURS         = 48     # auto-expire open trades after this
OUTCOME_WINDOW_HOURS   = 24
BACKFILL_DAYS          = 90
BACKFILL_TIMEFRAME     = '1m'

TREND_SLOPE_THRESHOLD  = 0.00003  # normalised per-min ratio slope; old 0.01 never tripped
REP_MIN_SAMPLES        = 3
MAD_CONSISTENCY        = 1.4826
RECENT_PAIR_SAMPLES    = 120      # minutes; dislocation must still be live on this window
Z_SCORE_CLAMP          = 8.0      # cap |robust z|; a stale long median manufactures z=-20+

# Position sizing
TRADE_PCT_BASE         = 0.05   # floor
TRADE_PCT_REP_BONUS    = 0.05   # max bonus from reputation
TRADE_PCT_Z_BONUS      = 0.05   # max bonus from Z depth
TRADE_PCT_FLOOR        = 0.03
TRADE_PCT_CEIL         = 0.15

# Market context
FNG_URL    = "https://api.alternative.me/fng/?limit=1"
FNG_FEAR   = 25    # extreme fear threshold — boost signals
FNG_GREED  = 75    # extreme greed threshold — suppress signals
NEWS_FEED  = "https://www.coindesk.com/arc/outboundfeeds/rss/"
NEWS_COUNT = 5     # headlines to feed Qwen

# ── Balances ──────────────────────────────────────────────────────────────────

DEFAULT_BALANCES = {a: 0.0 for a in ['USDT'] + ASSETS}


def load_balances() -> dict[str, float]:
    p = Path(BALANCES_FILE)
    if not p.exists():
        print(f"  balances.json not found — creating template at {BALANCES_FILE}")
        p.write_text(json.dumps(DEFAULT_BALANCES, indent=2))
        print("  Edit it with your actual holdings, then re-run.")
        return dict(DEFAULT_BALANCES)
    with open(p) as f:
        data = json.load(f)
    for a in DEFAULT_BALANCES:
        data.setdefault(a, 0.0)
    return data


def signal_weighted_notional(asset: str, price: float,
                              balances: dict[str, float],
                              rep: float | None,
                              z_score: float,
                              override: float | None = None) -> float:
    """
    Dynamic position sizing:
      base_pct    = 5%
      rep_bonus   = rep * 5%   (rep=+1 → +5%, rep=-1 → -5%)
      z_bonus     = min(abs(z)/10, 5%)
    Floor 3%, ceiling 15%.
    """
    if override is not None:
        return override
    holding = balances.get(asset, 0.0)
    value   = holding * price if asset != 'USDT' else holding
    if value <= 0:
        return 1.0

    rep_bonus = (rep if rep is not None else 0.0) * TRADE_PCT_REP_BONUS
    z_bonus   = min(abs(z_score) / 10, TRADE_PCT_Z_BONUS)
    pct       = TRADE_PCT_BASE + rep_bonus + z_bonus
    pct       = max(TRADE_PCT_FLOOR, min(pct, TRADE_PCT_CEIL))
    return max(round(value * pct, 4), 1.0)


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
    outcome_pct      DOUBLE,
    outcome_ts       TIMESTAMPTZ,
    reputation       DOUBLE,
    trend_slope      DOUBLE,
    fng_value        INTEGER,
    fng_label        VARCHAR,
    PRIMARY KEY (timestamp, top_pair)
);
"""

DDL_FANTASY_TRADES = """
CREATE TABLE IF NOT EXISTS fantasy_trades (
    trade_id        VARCHAR     NOT NULL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    mode            VARCHAR     NOT NULL,
    action          VARCHAR     NOT NULL,
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
    status          VARCHAR DEFAULT 'open',
    close_ts        TIMESTAMPTZ,
    close_reason    VARCHAR,
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

DDL_MARKET_CONTEXT = """
CREATE TABLE IF NOT EXISTS market_context (
    timestamp   TIMESTAMPTZ NOT NULL PRIMARY KEY,
    fng_value   INTEGER,
    fng_label   VARCHAR,
    headlines   TEXT
);
"""

# ── DB ────────────────────────────────────────────────────────────────────────

def get_conn(max_retries: int = 10, base_delay: float = 2.0) -> duckdb.DuckDBPyConnection:
    """
    Connect to DuckDB with retry-on-lock. DuckDB is single-writer; if another
    service (ingest/pairs/fiat) holds the lock, wait and retry instead of crashing.
    With 10 retries averaging ~3s each, tolerates ~30s of contention — plenty for
    a ~10s ingest cycle to release.
    """
    import random
    for attempt in range(max_retries):
        try:
            return duckdb.connect(DB_PATH)
        except duckdb.IOException as e:
            if 'lock' not in str(e).lower() or attempt == max_retries - 1:
                raise
            time.sleep(base_delay + random.uniform(0, 2))
    raise RuntimeError("Could not acquire DB lock after retries")


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL_MARKET_HISTORY)
    con.execute(DDL_LLM_VERDICTS)
    con.execute(DDL_FANTASY_TRADES)
    con.execute(DDL_FANTASY_BALANCES)
    con.execute(DDL_MARKET_CONTEXT)

    # Migrations from v0.6
    for col, typedef in [
        ('outcome_pct',  'DOUBLE'),
        ('outcome_ts',   'TIMESTAMPTZ'),
        ('mode',         "VARCHAR DEFAULT 'pair'"),
        ('reputation',   'DOUBLE'),
        ('trend_slope',  'DOUBLE'),
        ('fng_value',    'INTEGER'),
        ('fng_label',    'VARCHAR'),
    ]:
        _add_col_if_missing(con, 'llm_verdicts', col, typedef)

    for col, typedef in [
        ('source',       "VARCHAR DEFAULT 'live'"),
    ]:
        _add_col_if_missing(con, 'market_history', col, typedef)

    _add_col_if_missing(con, 'fantasy_trades', 'close_reason', 'VARCHAR')

    # One-time cleanup: fix /USDT/USDT duplicates.
    # A malformed row (e.g. 'AVAX/USDT/USDT') may share its timestamp with an
    # already-clean twin ('AVAX/USDT'). Blindly rewriting it collides with that
    # row and trips the (timestamp, top_pair) primary key, which crashes init_db
    # on every run. So first DROP the malformed rows that already have a clean
    # twin, then rewrite whatever malformed rows remain (no collision possible).
    bad = con.execute(
        "SELECT COUNT(*) FROM llm_verdicts WHERE top_pair LIKE '%/USDT/USDT%'"
    ).fetchone()[0]
    if bad > 0:
        con.execute(
            "DELETE FROM llm_verdicts WHERE rowid IN ("
            "  SELECT b.rowid FROM llm_verdicts b "
            "  WHERE b.top_pair LIKE '%/USDT/USDT%' "
            "    AND EXISTS (SELECT 1 FROM llm_verdicts g "
            "                WHERE g.timestamp = b.timestamp "
            "                  AND g.top_pair = REPLACE(b.top_pair, '/USDT/USDT', '/USDT'))"
            ")"
        )
        con.execute(
            "UPDATE llm_verdicts SET top_pair = REPLACE(top_pair, '/USDT/USDT', '/USDT') "
            "WHERE top_pair LIKE '%/USDT/USDT%'"
        )
        print(f"  DB cleanup: resolved {bad} /USDT/USDT entries")


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


# ── Market context ────────────────────────────────────────────────────────────

def fetch_market_context() -> dict:
    """
    Fetches Fear & Greed index and top crypto headlines.
    Returns dict with fng_value, fng_label, headlines (list of str).
    Never raises — returns neutral defaults on failure.
    """
    ctx = {'fng_value': 50, 'fng_label': 'Neutral', 'headlines': []}

    # Fear & Greed
    try:
        r = requests.get(FNG_URL, timeout=10)
        data = r.json()['data'][0]
        ctx['fng_value'] = int(data['value'])
        ctx['fng_label'] = data['value_classification']
    except Exception as e:
        print(f"  F&G fetch failed: {e}", file=sys.stderr)

    # Headlines from CoinDesk RSS
    try:
        r = requests.get(NEWS_FEED, timeout=10,
                         headers={'User-Agent': 'Mozilla/5.0'})
        titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', r.text)
        # Skip first (feed title)
        ctx['headlines'] = [t.strip() for t in titles[1:NEWS_COUNT + 1]]
    except Exception as e:
        print(f"  Headlines fetch failed: {e}", file=sys.stderr)

    return ctx


def store_market_context(con: duckdb.DuckDBPyConnection, ctx: dict) -> None:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    con.execute("""
        INSERT OR IGNORE INTO market_context (timestamp, fng_value, fng_label, headlines)
        VALUES (?, ?, ?, ?)
    """, [ts, ctx['fng_value'], ctx['fng_label'],
          json.dumps(ctx['headlines'])])


def fng_modifier(fng_value: int) -> tuple[float, str]:
    """
    Returns (threshold_adjustment, narrative).
    Extreme fear  → lower threshold (easier to trigger), boost confidence
    Extreme greed → raise threshold (harder to trigger), suppress confidence
    """
    if fng_value <= FNG_FEAR:
        adj = -0.3   # easier entry
        narrative = f"Extreme Fear ({fng_value}) — historically good accumulation conditions"
    elif fng_value >= FNG_GREED:
        adj = +0.5   # harder entry
        narrative = f"Extreme Greed ({fng_value}) — elevated risk, tighter entry threshold"
    elif fng_value <= 40:
        adj = -0.1
        narrative = f"Fear ({fng_value}) — market cautious, reversion more likely"
    elif fng_value >= 60:
        adj = +0.2
        narrative = f"Greed ({fng_value}) — market extended, be selective"
    else:
        adj = 0.0
        narrative = f"Neutral ({fng_value}) — no macro bias"
    return adj, narrative


def conviction_label(confidence: int) -> str:
    if confidence >= 8: return "Very Strong"
    if confidence >= 6: return "Strong"
    if confidence >= 4: return "Moderate"
    return "Low"


# ── Discord — plain English ───────────────────────────────────────────────────

def discord_send(message: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    except Exception as e:
        print(f"  Discord send failed: {e}", file=sys.stderr)


def discord_pair_signal(verdict: dict, confidence: int, top_row,
                        prices: dict, notional: float,
                        fng_value: int, fng_label: str) -> None:
    if verdict.get('action') not in ('accumulate', 'watch'):
        return
    if top_row is None:
        return

    base      = top_row['base_asset']
    quote     = top_row['quote_asset']
    base_px   = prices.get(base,  0.0)
    quote_px  = prices.get(quote, 0.0)
    z         = top_row['z_score']
    rep       = top_row.get('reputation')
    vel       = top_row.get('z_velocity')
    _, legs   = trade_route(base, quote)
    fees      = fee_breakdown(notional, legs, top_row['spread_pct'])
    conviction = conviction_label(confidence)

    # Mean reversion target: current ratio → median ratio
    med_ratio  = top_row.get('mean_ratio', 0)
    if med_ratio and quote_px:
        target_base_px = round(med_ratio * quote_px, 4)
        target_str     = f"{target_base_px:,.4f} USDT (+{abs(z)*fees['breakeven_pct']:.2f}% est.)"
    else:
        target_str = "n/a"

    rep_str  = f"{round(rep*100):.0f}% historical win rate" if rep is not None and rep != 0 else "no history yet"
    vel_str  = "deepening ↓" if (vel is not None and vel < 0) else ("recovering ↑" if (vel is not None and vel > 0) else "stable →")
    z_str    = f"{abs(z):.1f} standard deviations below average"
    action_emoji = "🟢" if verdict['action'] == 'accumulate' else "🟡"

    msg = (
        f"{action_emoji} **OSMIA — {conviction} {verdict['action'].upper()} signal**\n"
        f"**Trade:** Sell {base} → Buy {quote}\n"
        f"**Prices:** {base} @ {base_px:,.4f}  |  {quote} @ {quote_px:,.4f} USDT\n"
        f"**Position size:** {notional:.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Signal:** {base} is {z_str}, signal {vel_str}\n"
        f"**Track record:** {rep_str}\n"
        f"**Target:** {target_str}\n"
        f"**Break-even:** price ratio must improve {fees['breakeven_pct']:.3f}% to cover fees\n"
        f"**Est. fee:** {fees['total_cost']:.4f} USDT ({fees['num_legs']} leg{'s' if fees['num_legs']>1 else ''})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Market mood:** {fng_label} (F&G {fng_value})\n"
        f"**Thesis:** {verdict.get('thesis','')}\n"
        f"**Risks:** {verdict.get('risks','')}"
    )
    discord_send(msg)


def discord_fiat_signal(verdict: dict, confidence: int, top_row,
                        prices: dict, notional: float,
                        fng_value: int, fng_label: str) -> None:
    if verdict.get('action') not in ('accumulate', 'watch'):
        return
    if top_row is None:
        return

    asset     = top_row['asset_id']
    px        = prices.get(asset, top_row['price_usd'])
    z         = top_row['z_score']
    rep       = top_row.get('reputation') if hasattr(top_row, 'get') else None
    fees      = fee_breakdown(notional, 1, top_row['spread_pct'])
    buy_qty   = round((notional - fees['total_cost']) / px, 8) if px else 0
    conviction = conviction_label(confidence)
    rep_str   = f"{round(rep*100):.0f}% historical win rate" if rep else "no history yet"
    z_str     = f"{abs(z):.1f} standard deviations below average"
    action_emoji = "🟢" if verdict['action'] == 'accumulate' else "🟡"

    msg = (
        f"{action_emoji} **OSMIA — {conviction} USDT BUY signal**\n"
        f"**Buy:** {asset}/USDT @ {px:,.4f} USDT\n"
        f"**Quantity:** {buy_qty} {asset}  for {notional:.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Signal:** {asset} is {z_str}\n"
        f"**Track record:** {rep_str}\n"
        f"**Break-even:** +{fees['breakeven_pct']:.3f}% price increase needed\n"
        f"**Est. fee:** {fees['total_cost']:.4f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Market mood:** {fng_label} (F&G {fng_value})\n"
        f"**Thesis:** {verdict.get('thesis','')}\n"
        f"**Risks:** {verdict.get('risks','')}"
    )
    discord_send(msg)


def discord_trade_executed(trade: dict, conviction: str) -> None:
    msg = (
        f"📒 **Fantasy trade logged** — {conviction}\n"
        f"{trade['sell_asset']} → {trade['buy_asset']}\n"
        f"Position: {trade['notional_usd']:.2f} USDT  |  Fee: {trade['total_cost_usd']:.4f} USDT\n"
        f"Z-score: {trade['signal_z']:+.3f}  |  Confidence: {trade['confidence']}/10"
    )
    discord_send(msg)


def discord_trade_closed(trade_id: str, buy_asset: str,
                         pnl_usd: float, pnl_pct: float, reason: str) -> None:
    emoji = "✅" if pnl_usd >= 0 else "🔴"
    msg = (
        f"{emoji} **Fantasy trade closed** — {reason}\n"
        f"Asset: {buy_asset}  |  P&L: {pnl_usd:+.4f} USDT ({pnl_pct:+.3f}%)"
    )
    discord_send(msg)


def discord_daily_pnl(con: duckdb.DuckDBPyConnection,
                      prices: dict[str, float]) -> None:
    trades = con.execute("""
        SELECT buy_asset, buy_qty, buy_price_usd, notional_usd, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    total_pnl = 0.0
    best_pnl  = (-999.0, "")
    worst_pnl = (999.0, "")

    for _, t in trades.iterrows():
        cur_px = prices.get(t['buy_asset'], t['buy_price_usd'])
        pnl    = t['buy_qty'] * cur_px - t['notional_usd']
        total_pnl += pnl
        if pnl > best_pnl[0]:  best_pnl  = (pnl, t['buy_asset'])
        if pnl < worst_pnl[0]: worst_pnl = (pnl, t['buy_asset'])

    # Signals today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0,
                                                      second=0, microsecond=0)
    signals_today = con.execute("""
        SELECT COUNT(*) FROM llm_verdicts
        WHERE action = 'accumulate' AND timestamp >= ?
    """, [today_start]).fetchone()[0]

    trades_today = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades WHERE timestamp >= ?
    """, [today_start]).fetchone()[0]

    ctx = con.execute("""
        SELECT fng_value, fng_label FROM market_context
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    mood = f"{ctx[1]} (F&G {ctx[0]})" if ctx else "Unknown"

    emoji = "📈" if total_pnl >= 0 else "📉"
    msg = (
        f"{emoji} **OSMIA Daily Summary**\n"
        f"**Unrealized P&L:** {total_pnl:+.4f} USDT across {len(trades)} open positions\n"
        f"**Best position:** {best_pnl[1]} ({best_pnl[0]:+.4f} USDT)\n"
        f"**Worst position:** {worst_pnl[1]} ({worst_pnl[0]:+.4f} USDT)\n"
        f"**Signals today:** {signals_today} fired  |  {trades_today} trades executed\n"
        f"**Market mood:** {mood}"
    )
    discord_send(msg)


# ── LLM ping ─────────────────────────────────────────────────────────────────

def ping_qwen() -> bool:
    try:
        r = requests.get(QWEN_URL.replace('/v1/chat/completions', '/health'), timeout=5)
        return r.status_code == 200
    except Exception:
        pass
    try:
        r = requests.post(QWEN_URL, json={
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── Ingest ────────────────────────────────────────────────────────────────────

def _volume_mad_zscore(con, new_df: pd.DataFrame) -> dict[str, float]:
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
        v      = hist['volume_24h'].values
        median = np.median(v)
        mad    = np.median(np.abs(v - median))
        denom  = MAD_CONSISTENCY * mad
        results[asset] = float((cur_vol - median) / denom) if denom > 0 else 0.0
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

    df  = pd.DataFrame(rows)
    vzs = _volume_mad_zscore(con, df)
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
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int((datetime.now(timezone.utc) -
                    timedelta(days=BACKFILL_DAYS)).timestamp() * 1000)
    print(f"Backfilling {BACKFILL_DAYS} days of OHLCV for {len(ASSETS)} assets …")
    total = 0
    for symbol in SYMBOLS:
        asset = symbol.split('/')[0]
        print(f"  {asset:>6s} … ", end='', flush=True)
        rows, cursor, retries = [], since_ms, 0
        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe=BACKFILL_TIMEFRAME, since=cursor, limit=1000)
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
                ts = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).replace(second=0, microsecond=0)
                rows.append({
                    'timestamp': ts, 'asset_id': asset,
                    'price_usd': float(c[4]), 'volume_24h': float(c[5]),
                    'change_24h': None, 'market_cap': 0.0,
                    'vwap_ratio': None, 'volume_z': None, 'spread_pct': None,
                    'high_low_pct': round((float(c[2]) - float(c[3])) / float(c[3]) * 100, 6) if c[3] else None,
                    'source': 'backfill',
                })
            cursor = candles[-1][0] + 1
            if len(candles) < 1000:
                break
            time.sleep(0.2)
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
    print(f"\nBackfill complete. {total:,} rows. DB total: "
          f"{con.execute('SELECT COUNT(*) FROM market_history').fetchone()[0]:,}")


# ── Fee helpers ───────────────────────────────────────────────────────────────

def trade_route(base: str, quote: str) -> tuple[str, int]:
    if f"{base}/{quote}" in BINANCE_DIRECT_PAIRS:
        return f"{base}/{quote} (direct)", 1
    return f"{base}/USDT → {quote}/USDT", 2


def fee_breakdown(notional: float, num_legs: int, spread_pct: float | None) -> dict:
    fee_per_leg   = notional * BINANCE_TAKER_FEE
    total_fee     = fee_per_leg * num_legs
    spread_cost   = notional * (spread_pct / 100) if spread_pct else 0.0
    total_cost    = total_fee + spread_cost
    breakeven_pct = (total_cost / notional) * 100
    return {
        'notional':      round(notional,      4),
        'fee_per_leg':   round(fee_per_leg,   6),
        'num_legs':      num_legs,
        'total_fee':     round(total_fee,     6),
        'spread_cost':   round(spread_cost,   6),
        'total_cost':    round(total_cost,    6),
        'breakeven_pct': round(breakeven_pct, 6),
    }


# ── MAD Z-score ───────────────────────────────────────────────────────────────

def mad_zscore(values: np.ndarray, current: float) -> float:
    if len(values) < 5:
        return 0.0
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    denom  = MAD_CONSISTENCY * mad
    return float((current - median) / denom) if denom > 0 else 0.0


# ── Reputation ────────────────────────────────────────────────────────────────

def get_pair_reputations(con, pairs: list[str]) -> dict[str, float]:
    if not pairs:
        return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT top_pair,
               COUNT(*)                                                    AS n,
               AVG(outcome_pct)                                            AS avg_out,
               SUM(CASE WHEN outcome_pct > 0 THEN 1.0 ELSE 0 END) / COUNT(*) AS win_rate
        FROM llm_verdicts
        WHERE top_pair IN ({ph}) AND outcome_pct IS NOT NULL AND action = 'accumulate'
        GROUP BY top_pair
    """, pairs).df()
    result = {}
    for _, r in df.iterrows():
        if r['n'] < REP_MIN_SAMPLES:
            result[r['top_pair']] = 0.0
            continue
        score = math.tanh(r['avg_out'] * r['win_rate'] * math.log(r['n'] + 1))
        result[r['top_pair']] = round(score, 4)
    return result


def reputation_label(rep: float | None) -> str:
    if rep is None or rep == 0.0: return "neutral"
    if rep >= 0.5:  return "strong✅"
    if rep >= 0.2:  return "positive"
    if rep >= 0.0:  return "mild+"
    if rep >= -0.2: return "mild-"
    if rep >= -0.5: return "negative"
    return "poor⚠"


# ── Trend filter ──────────────────────────────────────────────────────────────

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


# ── Signal engine ─────────────────────────────────────────────────────────────

def get_raw_anomalies(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
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
            z = mad_zscore(ratios, current_ratio)
            if z >= ZSCORE_BUY_THRESH:
                continue
            # Stale-median guard: a long window whose median lags a slow grind
            # (e.g. ADA bleeding vs the field for days) manufactures z=-20+ that is
            # a *trend*, not a live reversion spring. Require the deviation to still
            # be anomalous on a short recent window; if it has settled at a new
            # level there is nothing left to revert to — skip it.
            z_recent = mad_zscore(ratios[-RECENT_PAIR_SAMPLES:], current_ratio)
            if z_recent >= ZSCORE_BUY_THRESH:
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


def detect_dominance(df: pd.DataFrame, col: str) -> str | None:
    counts   = df[col].value_counts()
    dominant = counts[counts >= 3]
    return dominant.index[0] if not dominant.empty else None


def rule_based_score(row, persistence, prev_z, reputations, slopes, fng_adj=0.0) -> float:
    score = row['z_score'] + fng_adj   # F&G adjusts entry threshold

    score -= min(persistence.get(row['pair'], 0) * 0.1, 0.5)

    prev = prev_z.get(row['pair'])
    if prev is not None:
        score -= 0.2 if row['z_score'] < prev else -0.2

    if row['volume_z']   is not None and row['volume_z']   >  1.5: score -= 0.15
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99: score -= 0.1
    if row['spread_pct'] is not None and row['spread_pct'] >  0.1: score += 0.3
    if row['quote_asset'] in MEME_ASSETS:                           score += 0.5

    rep = reputations.get(row['pair'], 0.0)
    score -= rep * 0.5

    # Falling-knife guard: a ratio in sustained directional drift is trending,
    # not reverting, so penalise it proportionally — a multi-day grind shouldn't
    # keep winning the board. (Old `slope*10` was a no-op: normalised slopes ~1e-4.)
    slope = abs(slopes.get(row['pair'], 0.0))
    if slope > TREND_SLOPE_THRESHOLD:
        score += min(slope / TREND_SLOPE_THRESHOLD, 4.0) * 0.5

    return round(score, 4)


def build_candidates(con, fng_adj=0.0) -> tuple[pd.DataFrame, str | None, str | None]:
    raw = get_raw_anomalies(con)
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


# ── Fantasy portfolio ─────────────────────────────────────────────────────────

def _trade_id(ts: datetime, pair: str) -> str:
    return f"{ts.strftime('%Y%m%d%H%M')}_{pair.replace('/', '_')}"


def _snapshot_balances(con, balances, prices) -> None:
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = [{'timestamp': ts, 'asset_id': a,
             'qty': q, 'price_usd': prices.get(a, 1.0) if a != 'USDT' else 1.0,
             'value_usd': q * (prices.get(a, 1.0) if a != 'USDT' else 1.0)}
            for a, q in balances.items()]
    df = pd.DataFrame(rows)
    con.execute("""
        INSERT OR IGNORE INTO fantasy_balances (timestamp, asset_id, qty, price_usd, value_usd)
        SELECT timestamp, asset_id, qty, price_usd, value_usd FROM df
    """)


def _pair_on_cooldown(con, sell_asset: str, buy_asset: str) -> bool:
    result = con.execute(f"""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE sell_asset = ? AND buy_asset = ?
          AND timestamp >= NOW() - INTERVAL {TRADE_COOLDOWN_MINUTES} MINUTES
          AND status = 'open'
    """, [sell_asset, buy_asset]).fetchone()
    return result[0] > 0


def _pair_at_max_trades(con, sell_asset: str, buy_asset: str) -> bool:
    result = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE sell_asset = ? AND buy_asset = ? AND status = 'open'
    """, [sell_asset, buy_asset]).fetchone()
    return result[0] >= MAX_TRADES_PER_PAIR


def _close_trade(con, trade_id: str, cur_px: float,
                 notional: float, buy_qty: float,
                 reason: str) -> tuple[float, float]:
    ts          = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    current_val = buy_qty * cur_px
    pnl_usd     = round(current_val - notional, 4)
    pnl_pct     = round(pnl_usd / notional * 100, 4) if notional else 0
    con.execute("""
        UPDATE fantasy_trades
        SET status='closed', close_ts=?, close_reason=?,
            close_buy_px=?, pnl_usd=?, pnl_pct=?
        WHERE trade_id=?
    """, [ts, reason, cur_px, pnl_usd, pnl_pct, trade_id])
    return pnl_usd, pnl_pct


def _auto_close_trades(con: duckdb.DuckDBPyConnection,
                       prices: dict[str, float],
                       current_z_scores: dict[str, float] | None = None) -> int:
    """
    Auto-close open fantasy trades on three conditions:
    1. Mean reversion: P&L >= 1.5x break-even cost
    2. Z crossback: current Z-score for the pair > ZSCORE_CLOSE_THRESH (-1.0)
    3. Expiry: trade has been open > MAX_HOLD_HOURS
    """
    open_trades = con.execute("""
        SELECT trade_id, sell_asset, buy_asset, buy_qty,
               buy_price_usd, notional_usd, total_cost_usd, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    closed = 0
    now    = datetime.now(timezone.utc)

    for _, t in open_trades.iterrows():
        pair    = f"{t['sell_asset']}/{t['buy_asset']}"
        cur_px  = prices.get(t['buy_asset'], t['buy_price_usd'])
        cur_val = t['buy_qty'] * cur_px
        pnl_pct = (cur_val - t['notional_usd']) / t['notional_usd'] * 100 if t['notional_usd'] else 0
        be_pct  = t['total_cost_usd'] / t['notional_usd'] * 100 if t['notional_usd'] else 0

        reason = None

        # 1. Mean reversion
        if pnl_pct >= be_pct * 1.5:
            reason = "reverted"

        # 2. Z crossback
        elif current_z_scores and pair in current_z_scores:
            if current_z_scores[pair] > ZSCORE_CLOSE_THRESH:
                reason = "z_crossback"

        # 3. Expiry
        elif t['timestamp']:
            age_hours = (now - t['timestamp'].to_pydatetime()).total_seconds() / 3600
            if age_hours > MAX_HOLD_HOURS:
                reason = f"expired_{MAX_HOLD_HOURS}h"

        if reason:
            pnl_usd, pnl_pct_final = _close_trade(
                con, t['trade_id'], cur_px,
                t['notional_usd'], t['buy_qty'], reason
            )
            discord_trade_closed(t['trade_id'], t['buy_asset'],
                                  pnl_usd, pnl_pct_final, reason)
            closed += 1

    return closed


def execute_fantasy_pair_trade(con, sell_asset, buy_asset,
                                sell_px, buy_px, notional,
                                spread_pct, z_score, confidence,
                                balances) -> dict | None:
    holding_usd = balances.get(sell_asset, 0.0) * sell_px
    if holding_usd < notional * 0.5:
        return None
    _, legs  = trade_route(sell_asset, buy_asset)
    fees     = fee_breakdown(notional, legs, spread_pct)
    sell_qty = notional / sell_px  if sell_px  else 0
    buy_qty  = (notional - fees['total_cost']) / buy_px if buy_px else 0
    balances[sell_asset] = max(0.0, balances.get(sell_asset, 0.0) - sell_qty)
    balances[buy_asset]  = balances.get(buy_asset, 0.0) + buy_qty
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{sell_asset}/{buy_asset}"
    return {
        'trade_id': _trade_id(ts, pair), 'timestamp': ts,
        'mode': 'pair', 'action': 'pair_reversion',
        'sell_asset': sell_asset, 'buy_asset': buy_asset,
        'sell_qty': round(sell_qty, 8), 'buy_qty': round(buy_qty, 8),
        'sell_price_usd': sell_px, 'buy_price_usd': buy_px,
        'notional_usd': fees['notional'], 'fee_usd': fees['total_fee'],
        'spread_usd': fees['spread_cost'], 'total_cost_usd': fees['total_cost'],
        'signal_z': z_score, 'confidence': confidence, 'status': 'open',
    }


def execute_fantasy_fiat_trade(con, buy_asset, buy_px, notional,
                                spread_pct, z_score, confidence,
                                balances) -> dict | None:
    usdt = balances.get('USDT', 0.0)
    if usdt < notional * 0.5:
        return None
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = (notional - fees['total_cost']) / buy_px if buy_px else 0
    balances['USDT']    = max(0.0, usdt - notional)
    balances[buy_asset] = balances.get(buy_asset, 0.0) + buy_qty
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{buy_asset}/USDT"
    return {
        'trade_id': _trade_id(ts, pair), 'timestamp': ts,
        'mode': 'fiat', 'action': 'fiat_buy',
        'sell_asset': 'USDT', 'buy_asset': buy_asset,
        'sell_qty': round(notional, 6), 'buy_qty': round(buy_qty, 8),
        'sell_price_usd': 1.0, 'buy_price_usd': buy_px,
        'notional_usd': fees['notional'], 'fee_usd': fees['total_fee'],
        'spread_usd': fees['spread_cost'], 'total_cost_usd': fees['total_cost'],
        'signal_z': z_score, 'confidence': confidence, 'status': 'open',
    }


def _insert_trade(con, trade: dict) -> None:
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
    print(f"[{datetime.now():%H:%M:%S}] Scoring outcomes …")

    pending = con.execute(f"""
        SELECT timestamp, top_pair, mode, z_score
        FROM llm_verdicts
        WHERE action = 'accumulate'
          AND outcome_pct IS NULL
          AND timestamp <= NOW() - INTERVAL {OUTCOME_WINDOW_HOURS} HOURS
    """).df()

    if pending.empty:
        print(f"  No pending outcomes (need {OUTCOME_WINDOW_HOURS}h after signal).")
    else:
        scored = 0
        for _, row in pending.iterrows():
            pair   = row['top_pair']
            sig_ts = row['timestamp']
            out_ts = sig_ts + timedelta(hours=OUTCOME_WINDOW_HOURS)
            if '/' not in str(pair):
                continue
            parts = pair.split('/')
            if len(parts) != 2:
                continue
            base, quote = parts[0], parts[1]

            if row['mode'] == 'fiat' or quote == 'USDT':
                ep = con.execute("SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp<=? ORDER BY timestamp DESC LIMIT 1", [base, sig_ts]).fetchone()
                xp = con.execute("SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp>=? ORDER BY timestamp ASC LIMIT 1", [base, out_ts]).fetchone()
                if not ep or not xp: continue
                outcome_pct = round((xp[0] - ep[0]) / ep[0] * 100, 4)
            else:
                def _px(asset, ts, d):
                    op = 'DESC' if d == 'e' else 'ASC'
                    cm = '<=' if d == 'e' else '>='
                    return con.execute(f"SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp{cm}? ORDER BY timestamp {op} LIMIT 1", [asset, ts]).fetchone()
                eb,eq,xb,xq = _px(base,sig_ts,'e'), _px(quote,sig_ts,'e'), _px(base,out_ts,'x'), _px(quote,out_ts,'x')
                if not all([eb,eq,xb,xq]): continue
                outcome_pct = round(((xb[0]/xq[0]) - (eb[0]/eq[0])) / (eb[0]/eq[0]) * 100, 4)

            con.execute("UPDATE llm_verdicts SET outcome_pct=?, outcome_ts=? WHERE timestamp=? AND top_pair=?",
                        [outcome_pct, out_ts, sig_ts, pair])
            scored += 1
        print(f"  Scored {scored} outcomes.")

    summary = con.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS winners,
               ROUND(AVG(outcome_pct), 3) AS avg,
               ROUND(MAX(outcome_pct), 3) AS best,
               ROUND(MIN(outcome_pct), 3) AS worst
        FROM llm_verdicts WHERE action='accumulate' AND outcome_pct IS NOT NULL
    """).fetchone()

    if summary and summary[0] > 0:
        wr = round(summary[1] / summary[0] * 100, 1)
        print(f"\n── Signal accuracy ({OUTCOME_WINDOW_HOURS}h window) ──")
        print(f"  Signals scored : {summary[0]}")
        print(f"  Win rate       : {summary[1]}/{summary[0]} ({wr}%)")
        print(f"  Avg outcome    : {summary[2]:+.3f}%")
        print(f"  Best / Worst   : {summary[3]:+.3f}% / {summary[4]:+.3f}%")

    print(f"\n── Per-pair breakdown ──")
    breakdown = con.execute("""
        SELECT top_pair,
               COUNT(*) AS n,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS wins,
               ROUND(AVG(outcome_pct), 3) AS avg_out,
               ROUND(MIN(outcome_pct), 3) AS worst,
               ROUND(MAX(outcome_pct), 3) AS best
        FROM llm_verdicts
        WHERE action='accumulate' AND outcome_pct IS NOT NULL
        GROUP BY top_pair ORDER BY avg_out DESC
    """).df()
    pairs = breakdown['top_pair'].tolist()
    reps  = get_pair_reputations(con, pairs)
    breakdown['reputation'] = breakdown['top_pair'].map(
        lambda p: f"{reps.get(p, 0.0):+.3f}")
    print(breakdown.to_string(index=False))


# ── LLM ───────────────────────────────────────────────────────────────────────

def ask_qwen(prompt: str) -> dict:
    fallback = {'action': 'pass', 'top_pair': '', 'top_asset': '',
                'rank': [], 'thesis': '', 'risks': '',
                'prose': 'LLM unavailable.', 'raw_json': ''}
    payload = {"model": QWEN_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.2, "max_tokens": 512}
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


def _log_verdict(con, verdict, confidence, top_row, mode,
                 reputation=None, trend_slope=None,
                 fng_value=None, fng_label=None) -> None:
    ts       = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score  = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'
    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, mode, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json,
             reputation, trend_slope, fng_value, fng_label)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [ts, mode, top_pair, z_score, confidence,
          verdict.get('action',   'pass'),
          verdict.get('thesis',   ''),
          verdict.get('risks',    ''),
          verdict.get('prose',    ''),
          verdict.get('raw_json', ''),
          reputation, trend_slope, fng_value, fng_label])


# ── Pair analysis ─────────────────────────────────────────────────────────────

def build_pair_prompt(candidates, quote_warn, base_warn,
                      fng_narrative: str, headlines: list[str]) -> str:
    rows = []
    for i, r in candidates.iterrows():
        vel   = f"{r['z_velocity']:+.3f}"  if r['z_velocity']  is not None else "n/a"
        spread= f"{r['spread_pct']:.4f}%"  if r['spread_pct']  is not None else "n/a"
        vwap  = f"{r['vwap_ratio']:.4f}"   if r['vwap_ratio']  is not None else "n/a"
        vol_z = f"{r['volume_z']:+.3f}"    if r['volume_z']    is not None else "n/a"
        rep   = r.get('reputation')
        rep_s = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
        slope = r.get('trend_slope', 0.0)
        trend_flag = " ⚠TRENDING" if abs(slope) > TREND_SLOPE_THRESHOLD else ""
        route, _ = trade_route(r['base_asset'], r['quote_asset'])
        rows.append(
            f"  [rank {i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel} | "
            f"persist={r['persistence_24h']} | vwap={vwap} | vol_z={vol_z} | "
            f"spread={spread} | rep={rep_s}{trend_flag} | route={route}"
        )
    table = "\n".join(rows)
    warn  = ""
    if quote_warn:
        warn += f"\n⚠ QUOTE DOMINANCE: {quote_warn} is the quote in multiple pairs. Address in thesis.\n"
    if base_warn:
        warn += f"\n⚠ BASE DOMINANCE: {base_warn} — likely one signal, not independent.\n"

    news_block = ""
    if headlines:
        news_block = "\nRecent crypto headlines:\n" + "\n".join(f"  - {h}" for h in headlines) + "\n"

    return f"""You are a disciplined spot-only crypto trader. Rule-based engine pre-ranked {len(candidates)} candidates.

Market context: {fng_narrative}
{news_block}
Candidates (best first):
{table}
{warn}
Column guide: z=robust Z-score, vel=Z delta (negative=deepening), persist=24h hits,
vwap=last/VWAP, vol_z=volume anomaly, spread=bid-ask%, rep=historical score (-1 to +1),
⚠TRENDING=ratio trending not oscillating.

Rules:
- Rule rank is authoritative. Override requires explicit justification.
- Poor rep (< -0.3) + Z > -3.0: watch not accumulate.
- Strong rep (> 0.4) + Z < -1.8: can accumulate.
- ⚠TRENDING: accumulate only if Z < -3.0 and rep > 0.3.
- Factor macro context into thesis — mention F&G if relevant.
- Do NOT assign equal confidence. spread >0.15% caps confidence at 6.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_pair":"<PAIR>","rank":["<p1>",...],"thesis":"<str>","risks":"<str>"}}
"""


def format_pair_trade(top_row, prices, verdict, confidence, notional,
                      fng_value, fng_label) -> None:
    base, quote = top_row['base_asset'], top_row['quote_asset']
    base_px     = prices.get(base,  0.0)
    quote_px    = prices.get(quote, 0.0)
    route, legs = trade_route(base, quote)
    fees        = fee_breakdown(notional, legs, top_row['spread_pct'])
    sell_qty    = round(notional / base_px,  8) if base_px  else 0
    buy_qty     = round(notional / quote_px, 8) if quote_px else 0
    rep         = top_row.get('reputation')
    rep_s       = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
    slope       = top_row.get('trend_slope', 0.0)
    vel         = top_row['z_velocity'] if top_row['z_velocity'] is not None else 'n/a'
    conviction  = conviction_label(confidence)

    print("\n╔══ TRADE SIGNAL ══════════════════════════════════════════╗")
    print(f"  Mode       : PAIR REVERSION (spot)  —  {conviction}")
    print(f"  Action     : Sell {base} → Buy {quote}  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Route      : {route}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Sell       : {sell_qty} {base}  @ {base_px:,.4f} USDT")
    print(f"  Buy        : {buy_qty} {quote}  @ {quote_px:,.4f} USDT")
    print(f"  Position   : {fees['notional']} USDT  (signal-weighted)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['fee_per_leg']} USDT × {fees['num_legs']} legs = {fees['total_fee']} USDT")
    if top_row['spread_pct']:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({top_row['spread_pct']:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of position)")
    print(f"  Break-even : ratio must improve {fees['breakeven_pct']:.4f}%")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Signal     : Z={top_row['z_score']:+.3f}  persist={top_row['persistence_24h']}  vel={vel}")
    print(f"  Reputation : {rep_s}")
    print(f"  Market     : {fng_label} (F&G {fng_value})")
    if abs(slope) > TREND_SLOPE_THRESHOLD:
        print(f"  ⚠ Trend    : slope={slope:.6f} — may be trending not oscillating")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_pairs(con: duckdb.DuckDBPyConnection,
                  size_override: float | None = None) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running pair analysis …")

    if not ping_qwen():
        print(f"  ⚠ WARNING: Qwen unreachable at {QWEN_URL}")

    # Market context
    ctx       = fetch_market_context()
    store_market_context(con, ctx)
    fng_val   = ctx['fng_value']
    fng_lbl   = ctx['fng_label']
    fng_adj, fng_narrative = fng_modifier(fng_val)
    print(f"  Market: {fng_lbl} (F&G {fng_val}) — {fng_narrative}")

    candidates, quote_warn, base_warn = build_candidates(con, fng_adj)

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'pair', fng_value=fng_val, fng_label=fng_lbl)
        return

    if quote_warn: print(f"\n  ⚠ Quote dominance : {quote_warn}")
    if base_warn:  print(f"  ⚠ Base dominance  : {base_warn}")

    print("\n── Candidates (rule-scored) ──")
    disp = candidates[['pair', 'z_score', 'z_velocity', 'persistence_24h',
                        'vwap_ratio', 'volume_z', 'spread_pct',
                        'reputation', 'trend_slope', 'rule_score']].copy()
    disp['reputation'] = disp['reputation'].apply(
        lambda r: f"{r:+.3f}" if r is not None else "n/a")
    print(disp.to_string(index=False))

    prompt     = build_pair_prompt(candidates, quote_warn, base_warn,
                                   fng_narrative, ctx['headlines'])
    # Release the single-writer lock during the multi-minute LLM call so the
    # concurrent ingest/fiat timers aren't blocked (and don't crash on lock).
    con.close()
    verdict    = ask_qwen(prompt)
    con        = get_conn()
    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair, fng_val)
    prices     = get_current_prices(con)

    # Build current Z-score map for crossback detection
    current_zs = {r['pair']: r['z_score'] for _, r in candidates.iterrows()}

    closed = _auto_close_trades(con, prices, current_zs)
    if closed:
        print(f"\n  ✅ Auto-closed {closed} fantasy trade(s)")

    top_row   = None
    rep_val   = None
    slope_val = None
    if top_pair:
        match = candidates[candidates['pair'] == top_pair]
        if not match.empty:
            top_row   = match.iloc[0]
            rep_val   = top_row.get('reputation')
            slope_val = top_row.get('trend_slope')

    balances = load_balances()

    if top_row is not None and verdict.get('action') != 'pass':
        base    = top_row['base_asset']
        base_px = prices.get(base, 0.0)
        notional = signal_weighted_notional(base, base_px, balances,
                                            rep_val, top_row['z_score'], size_override)
        format_pair_trade(top_row, prices, verdict, confidence, notional, fng_val, fng_lbl)
        discord_pair_signal(verdict, confidence, top_row, prices, notional, fng_val, fng_lbl)

        if confidence >= MIN_CONFIDENCE_TRADE:
            quote = top_row['quote_asset']
            if _pair_at_max_trades(con, base, quote):
                print(f"\n  📒 Fantasy trade skipped — {base}/{quote} at max {MAX_TRADES_PER_PAIR} open trades")
            elif _pair_on_cooldown(con, base, quote):
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
                    discord_trade_executed(trade, conviction_label(confidence))
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient {base} balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'pair',
                 rep_val, slope_val, fng_val, fng_lbl)


# ── Fiat analysis ─────────────────────────────────────────────────────────────

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


def build_fiat_prompt(candidates, reputations,
                      fng_narrative: str, headlines: list[str]) -> str:
    rows = []
    for i, r in candidates.iterrows():
        pair  = f"{r['asset_id']}/USDT"
        rep   = reputations.get(pair)
        rep_s = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
        rows.append(
            f"  [rank {i+1}] {r['asset_id']:>6s}/USDT | price={r['price_usd']:,.4f} | "
            f"z={r['z_score']:+.3f} | vwap={r['vwap_ratio'] or 'n/a'} | "
            f"vol_z={r['volume_z'] or 'n/a'} | "
            f"24h={r['change_24h']:+.2f}% | spread={r['spread_pct'] or 'n/a'}% | rep={rep_s}"
        )
    table = "\n".join(rows)

    news_block = ""
    if headlines:
        news_block = "\nRecent crypto headlines:\n" + "\n".join(f"  - {h}" for h in headlines) + "\n"

    return f"""You are a disciplined spot-only crypto trader advising on USDT buy entries.
Market context: {fng_narrative}
{news_block}
Statistically cheap assets (robust Z < {ZSCORE_BUY_THRESH}, median/MAD):
{table}

Rules: spot only, no leverage.
- Poor rep (< -0.3): watch only unless Z < -3.0.
- Strong rep (> 0.4): can accumulate at Z < -1.8.
- Avoid freefall (very negative 24h, no volume support).
- Factor macro context — F&G extreme fear is a buy signal; extreme greed is caution.
- Rank all. Do not assign equal confidence. action: accumulate|watch|pass.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_asset":"<ASSET>","rank":["<a1>",...],"thesis":"<str>","risks":"<str>"}}
"""


def format_fiat_trade(asset, price, spread_pct, z_score,
                      vwap_ratio, volume_z, change_24h,
                      verdict, confidence, notional, rep,
                      fng_value, fng_label) -> None:
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = round((notional - fees['total_cost']) / price, 8) if price else 0
    rep_s   = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
    conviction = conviction_label(confidence)

    print("\n╔══ FIAT BUY SIGNAL ═══════════════════════════════════════╗")
    print(f"  Mode       : USDT → {asset} (spot buy)  —  {conviction}")
    print(f"  Action     : Buy {asset}/USDT  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Price      : {price:,.4f} USDT")
    print(f"  Quantity   : {buy_qty} {asset}  from {fees['notional']} USDT")
    print(f"  Position   : {fees['notional']} USDT  (signal-weighted)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['total_fee']} USDT  (0.075% BNB)")
    if spread_pct:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({spread_pct:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of position)")
    print(f"  Break-even : +{fees['breakeven_pct']:.4f}% price move needed")
    print(f"  ─────────────────────────────────────────────────────")
    sig_parts = [f"Z={z_score:+.3f}"]
    if vwap_ratio:  sig_parts.append(f"vwap={vwap_ratio:.4f}")
    if volume_z:    sig_parts.append(f"vol_z={volume_z:+.3f}")
    if change_24h:  sig_parts.append(f"24h={change_24h:+.2f}%")
    print(f"  Signal     : {' '.join(sig_parts)}")
    print(f"  Reputation : {rep_s}")
    print(f"  Market     : {fng_label} (F&G {fng_value})")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")


def analyze_fiat(con: duckdb.DuckDBPyConnection,
                 size_override: float | None = None) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running fiat entry analysis …")

    if not ping_qwen():
        print(f"  ⚠ WARNING: Qwen unreachable at {QWEN_URL}")

    ctx     = fetch_market_context()
    store_market_context(con, ctx)
    fng_val = ctx['fng_value']
    fng_lbl = ctx['fng_label']
    _, fng_narrative = fng_modifier(fng_val)
    print(f"  Market: {fng_lbl} (F&G {fng_val})")

    candidates, short_history = get_fiat_candidates(con)

    if short_history:
        print("  ⚠ DB history < 14 days — fiat Z-scores may not reflect full price range")

    if candidates.empty:
        print("  No assets statistically cheap vs history — no entry signal.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'No fiat entry signal.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'fiat', fng_value=fng_val, fng_label=fng_lbl)
        return

    fiat_pairs  = [f"{r['asset_id']}/USDT" for _, r in candidates.iterrows()]
    reputations = get_pair_reputations(con, fiat_pairs)

    print("\n── Fiat candidates ──")
    disp = candidates[['asset_id', 'z_score', 'vwap_ratio',
                        'volume_z', 'spread_pct', 'change_24h']].copy()
    disp['reputation'] = [
        f"{reputations.get(r['asset_id'] + '/USDT', 0.0):+.3f}"
        for _, r in candidates.iterrows()
    ]
    print(disp.to_string(index=False))

    prompt    = build_fiat_prompt(candidates, reputations, fng_narrative, ctx['headlines'])
    # Release the single-writer lock during the multi-minute LLM call so the
    # concurrent ingest/pairs timers aren't blocked (and don't crash on lock).
    con.close()
    verdict   = ask_qwen(prompt)
    con       = get_conn()
    top_asset = verdict.get('top_asset', '') or ''
    # Qwen returns the asset inconsistently as "AVAX" or "AVAX/USDT" (sometimes even
    # "AVAX/USDT/USDT"). Take the base symbol (everything before the first '/') so we
    # compose the pair exactly once and never mint a doubled "AVAX/USDT/USDT" top_pair.
    base = top_asset.split('/')[0].strip()
    verdict['top_pair'] = f"{base}/USDT" if base else 'none'

    confidence = 5
    top_row    = None
    rep_val    = None
    if top_asset and top_asset in candidates['asset_id'].values:
        scores = candidates.set_index('asset_id')['z_score']
        winner = scores[top_asset]
        others = scores.drop(top_asset)
        gap    = abs(others.max() - winner) if not others.empty else 0
        conf   = 5 + min(int(gap / 0.25), 4)
        row    = candidates[candidates['asset_id'] == top_asset].iloc[0]
        if row['spread_pct'] is not None and row['spread_pct'] > 0.1: conf -= 1
        rep_val = reputations.get(f"{top_asset}/USDT")
        if rep_val is not None:
            if rep_val >= 0.5:    conf = min(conf + 1, 10)
            elif rep_val <= -0.5: conf = max(conf - 2, 1)
        if fng_val <= FNG_FEAR:    conf = min(conf + 1, 10)
        elif fng_val >= FNG_GREED: conf = max(conf - 1, 1)
        confidence = max(1, min(conf, 10))
        top_row    = row

    prices   = get_current_prices(con)
    balances = load_balances()

    if top_row is not None and verdict.get('action') != 'pass':
        px       = prices.get(top_asset, top_row['price_usd'])
        notional = signal_weighted_notional('USDT', 1.0, balances,
                                            rep_val, top_row['z_score'], size_override)
        format_fiat_trade(
            top_asset, px, top_row['spread_pct'], top_row['z_score'],
            top_row['vwap_ratio'], top_row['volume_z'], top_row['change_24h'],
            verdict, confidence, notional, rep_val, fng_val, fng_lbl
        )
        discord_fiat_signal(verdict, confidence, top_row, prices, notional, fng_val, fng_lbl)

        if confidence >= MIN_CONFIDENCE_TRADE:
            if _pair_at_max_trades(con, 'USDT', top_asset):
                print(f"\n  📒 Fantasy trade skipped — {top_asset}/USDT at max {MAX_TRADES_PER_PAIR} open trades")
            elif _pair_on_cooldown(con, 'USDT', top_asset):
                print(f"\n  📒 Fantasy trade skipped — {top_asset}/USDT on {TRADE_COOLDOWN_MINUTES}min cooldown")
            else:
                trade = execute_fantasy_fiat_trade(
                    con, top_asset, px, notional, top_row['spread_pct'],
                    top_row['z_score'], confidence, balances
                )
                if trade:
                    _insert_trade(con, trade)
                    _snapshot_balances(con, balances, prices)
                    discord_trade_executed(trade, conviction_label(confidence))
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient USDT balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")

    _log_verdict(con, verdict, confidence, top_row, 'fiat',
                 rep_val, None, fng_val, fng_lbl)


# ── Portfolio ─────────────────────────────────────────────────────────────────

def show_portfolio(con: duckdb.DuckDBPyConnection) -> None:
    prices = get_current_prices(con)
    print("── Fantasy Portfolio ──")
    trades = con.execute("""
        SELECT trade_id, timestamp, action, sell_asset, buy_asset,
               sell_qty, buy_qty, sell_price_usd, buy_price_usd,
               notional_usd, total_cost_usd, signal_z, confidence,
               status, close_reason, pnl_usd, pnl_pct
        FROM fantasy_trades ORDER BY timestamp DESC
    """).df()

    if trades.empty:
        print("  No fantasy trades executed yet.")
        return

    all_pairs = [f"{r['sell_asset']}/{r['buy_asset']}" for _, r in trades.iterrows()]
    reps      = get_pair_reputations(con, list(set(all_pairs)))

    rows_out = []
    for _, t in trades.iterrows():
        pair = f"{t['sell_asset']}/{t['buy_asset']}"
        if t['status'] == 'open':
            cur_px      = prices.get(t['buy_asset'], t['buy_price_usd'])
            current_val = t['buy_qty'] * cur_px
            pnl_usd     = round(current_val - t['notional_usd'], 4)
            pnl_pct     = round(pnl_usd / t['notional_usd'] * 100, 3) if t['notional_usd'] else 0
            close_r     = "-"
        else:
            pnl_usd = t['pnl_usd'] or 0.0
            pnl_pct = t['pnl_pct'] or 0.0
            close_r = t['close_reason'] or "-"

        rep = reps.get(pair)
        rows_out.append({
            'ts':       str(t['timestamp'])[:16],
            'trade':    f"{t['sell_asset']}→{t['buy_asset']}",
            'pos':      f"{t['notional_usd']:.2f}",
            'z':        f"{t['signal_z']:+.3f}",
            'conf':     t['confidence'],
            'rep':      f"{rep:+.3f}" if rep is not None else "n/a",
            'status':   t['status'],
            'reason':   close_r,
            'pnl_usd':  f"{pnl_usd:+.4f}",
            'pnl_pct':  f"{pnl_pct:+.3f}%",
        })

    print(pd.DataFrame(rows_out).to_string(index=False))

    open_t   = [r for r in rows_out if r['status'] == 'open']
    closed_t = [r for r in rows_out if r['status'] == 'closed']
    realized = sum(float(r['pnl_usd']) for r in closed_t)
    total    = sum(float(r['pnl_usd']) for r in rows_out)
    print(f"\n  Open positions : {len(open_t)}")
    print(f"  Closed         : {len(closed_t)}")
    print(f"  Realized P&L   : {realized:+.4f} USDT")
    print(f"  Total P&L      : {total:+.4f} USDT (incl. unrealized)")

    latest_bal = con.execute("""
        SELECT asset_id, qty, price_usd, value_usd FROM fantasy_balances
        WHERE timestamp = (SELECT MAX(timestamp) FROM fantasy_balances)
        ORDER BY value_usd DESC
    """).df()
    if not latest_bal.empty:
        total_val = latest_bal['value_usd'].sum()
        print(f"\n── Fantasy Balances ──")
        print(latest_bal[latest_bal['qty'] > 0].to_string(index=False))
        print(f"\n  Total portfolio value : {total_val:.4f} USDT")


# ── Status ────────────────────────────────────────────────────────────────────

def status(con: duckdb.DuckDBPyConnection) -> None:
    rows     = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    live     = con.execute("SELECT COUNT(*) FROM market_history WHERE source='live'").fetchone()[0]
    bf       = con.execute("SELECT COUNT(*) FROM market_history WHERE source='backfill'").fetchone()[0]
    verdicts = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    trades   = con.execute("SELECT COUNT(*) FROM fantasy_trades").fetchone()[0]
    latest   = con.execute("SELECT MAX(timestamp) FROM market_history").fetchone()[0]
    qwen_ok  = ping_qwen()

    ctx = con.execute("""
        SELECT fng_value, fng_label FROM market_context ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    mood = f"{ctx[1]} (F&G {ctx[0]})" if ctx else "not yet fetched"

    print(f"── DB Status ──")
    print(f"  market_history : {rows} rows  (live={live}, backfill={bf}, latest={latest})")
    print(f"  llm_verdicts   : {verdicts} rows")
    print(f"  fantasy_trades : {trades} rows")
    print(f"  Qwen           : {'✅ reachable' if qwen_ok else '⚠ unreachable'} at {QWEN_URL}")
    print(f"  Market mood    : {mood}")

    print("\n── Live sample counts ──")
    print(con.execute("""
        SELECT asset_id, COUNT(*) as samples,
               ROUND(AVG(spread_pct),4) as avg_spread,
               ROUND(AVG(vwap_ratio),4) as avg_vwap
        FROM market_history WHERE source='live'
        GROUP BY asset_id ORDER BY asset_id
    """).df().to_string(index=False))

    print("\n── Last 5 verdicts ──")
    print(con.execute("""
        SELECT timestamp, mode, top_pair, z_score, confidence_score,
               action, outcome_pct, reputation, fng_value
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals ──")
    print(con.execute("""
        SELECT mode, top_pair, COUNT(*) AS hits,
               ROUND(AVG(confidence_score),1) AS avg_conf,
               ROUND(AVG(z_score),3) AS avg_z,
               ROUND(AVG(outcome_pct),3) AS avg_outcome,
               ROUND(AVG(reputation),3) AS avg_rep,
               ROUND(AVG(fng_value),0) AS avg_fng
        FROM llm_verdicts WHERE action='accumulate'
        GROUP BY mode, top_pair ORDER BY hits DESC LIMIT 15
    """).df().to_string(index=False))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.7")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',    action='store_true')
    grp.add_argument('--analyze',   action='store_true')
    grp.add_argument('--fiat',      action='store_true')
    grp.add_argument('--backfill',  action='store_true')
    grp.add_argument('--outcomes',  action='store_true')
    grp.add_argument('--portfolio', action='store_true')
    grp.add_argument('--daily-pnl', action='store_true')
    grp.add_argument('--init-db',   action='store_true')
    grp.add_argument('--status',    action='store_true')
    parser.add_argument('--size', type=float, default=None)
    args = parser.parse_args()

    con = get_conn()
    init_db(con)

    if   args.init_db:   print("DB schema initialized.")
    elif args.status:    status(con)
    elif args.backfill:  backfill(con)
    elif args.outcomes:  score_outcomes(con)
    elif args.portfolio: show_portfolio(con)
    elif args.daily_pnl: discord_daily_pnl(con, get_current_prices(con))
    elif args.ingest:    ingest(con)
    elif args.analyze:   analyze_pairs(con, args.size)
    elif args.fiat:      analyze_fiat(con, args.size)
    else:
        ingest(con)
        analyze_pairs(con, args.size)

    con.close()


if __name__ == "__main__":
    main()
