#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.6
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

Changes from v0.5:
    - Median/MAD replaces mean/stddev for Z-score computation (robust to outliers,
      trend-resistant, 1.4826 consistency factor preserves -2.0 threshold meaning)
    - Pair reputation score: tanh(avg_outcome * win_rate * log(n+1)), bounded -1/+1
      Fed to LLM prompt per candidate, persisted to llm_verdicts
    - Trend filter: linear regression slope on ratio history — pairs with significant
      trend are flagged and penalised in rule_score
    - LLM prompt updated: reputation + trend context, adjusted thresholds
    - Discord webhook: signal alerts on accumulate, fantasy trades, daily P&L
    - QWEN_URL corrected to port 8081, startup ping with warning if unreachable
    - Auto-close fantasy trades when outcome_pct > breakeven_pct
    - --outcomes now prints per-pair breakdown after scoring
    - reputation column added to llm_verdicts
"""

import argparse
import json
import math
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
TOP_N_RAW              = 10
TOP_N_LLM              = 3
MEME_ASSETS            = {'DOGE', 'XLM', 'TRX', 'XRP'}

TRADE_PCT              = 0.10
MIN_CONFIDENCE_TRADE   = 6
TRADE_COOLDOWN_MINUTES = 60
OUTCOME_WINDOW_HOURS   = 24
BACKFILL_DAYS          = 90
BACKFILL_TIMEFRAME     = '1m'

# Trend filter: if abs(slope_zscore) > this, pair is trending not oscillating
TREND_SLOPE_THRESHOLD  = 0.01

# Reputation: minimum scored outcomes before rep score is used
REP_MIN_SAMPLES        = 3

# MAD consistency factor (makes MAD comparable to stddev for normal dist)
MAD_CONSISTENCY        = 1.4826

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


def notional_for(asset: str, price: float,
                 balances: dict[str, float],
                 override: float | None = None) -> float:
    if override is not None:
        return override
    holding  = balances.get(asset, 0.0)
    value    = holding * price if asset != 'USDT' else holding
    notional = value * TRADE_PCT
    return max(notional, 1.0)


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
    for col, typedef in [
        ('outcome_pct',  'DOUBLE'),
        ('outcome_ts',   'TIMESTAMPTZ'),
        ('mode',         "VARCHAR DEFAULT 'pair'"),
        ('reputation',   'DOUBLE'),
        ('trend_slope',  'DOUBLE'),
    ]:
        _add_col_if_missing(con, 'llm_verdicts', col, typedef)
    _add_col_if_missing(con, 'market_history', 'source', "VARCHAR DEFAULT 'live'")


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


# ── Discord ───────────────────────────────────────────────────────────────────

def discord_send(message: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK,
                      json={"content": message},
                      timeout=10)
    except Exception as e:
        print(f"  Discord send failed: {e}", file=sys.stderr)


def discord_signal(verdict: dict, confidence: int,
                   top_row, prices: dict, notional: float,
                   mode: str) -> None:
    if verdict.get('action') not in ('accumulate', 'watch'):
        return
    if mode == 'pair' and top_row is not None:
        base  = top_row['base_asset']
        quote = top_row['quote_asset']
        px_b  = prices.get(base,  0.0)
        px_q  = prices.get(quote, 0.0)
        rep   = top_row.get('reputation')
        rep_s = f"{rep:+.2f}" if rep is not None else "n/a"
        msg = (
            f"**🔔 OSMIA PAIR SIGNAL** `{verdict['action'].upper()}` {confidence}/10\n"
            f"Trade : Sell {base} → Buy {quote}\n"
            f"Z     : {top_row['z_score']:+.3f}  vel={top_row.get('z_velocity') or 'n/a'}  "
            f"rep={rep_s}\n"
            f"Prices: {base}={px_b:,.4f}  {quote}={px_q:,.4f} USDT\n"
            f"Notional: {notional:.2f} USDT\n"
            f"Thesis: {verdict.get('thesis','')}\n"
            f"Risks : {verdict.get('risks','')}"
        )
    elif mode == 'fiat' and top_row is not None:
        asset = top_row['asset_id']
        px    = prices.get(asset, top_row['price_usd'])
        msg = (
            f"**💵 OSMIA FIAT SIGNAL** `{verdict['action'].upper()}` {confidence}/10\n"
            f"Buy   : {asset}/USDT @ {px:,.4f}\n"
            f"Z     : {top_row['z_score']:+.3f}\n"
            f"Notional: {notional:.2f} USDT\n"
            f"Thesis: {verdict.get('thesis','')}\n"
            f"Risks : {verdict.get('risks','')}"
        )
    else:
        return
    discord_send(msg)


def discord_trade(trade: dict) -> None:
    msg = (
        f"**📒 FANTASY TRADE** `{trade['action'].upper()}`\n"
        f"{trade['sell_asset']} → {trade['buy_asset']}\n"
        f"Notional: {trade['notional_usd']:.2f} USDT  "
        f"Fee: {trade['total_cost_usd']:.4f} USDT\n"
        f"Z={trade['signal_z']:+.3f}  conf={trade['confidence']}"
    )
    discord_send(msg)


def discord_daily_pnl(con: duckdb.DuckDBPyConnection,
                      prices: dict[str, float]) -> None:
    trades = con.execute("""
        SELECT buy_asset, buy_qty, buy_price_usd, notional_usd
        FROM fantasy_trades WHERE status = 'open'
    """).df()
    if trades.empty:
        discord_send("**📊 OSMIA DAILY P&L** — No open positions.")
        return
    total_pnl = 0.0
    for _, t in trades.iterrows():
        cur_px  = prices.get(t['buy_asset'], t['buy_price_usd'])
        pnl     = t['buy_qty'] * cur_px - t['notional_usd']
        total_pnl += pnl
    discord_send(
        f"**📊 OSMIA DAILY P&L**\n"
        f"Open positions : {len(trades)}\n"
        f"Unrealized P&L : {total_pnl:+.4f} USDT"
    )


# ── LLM ping ─────────────────────────────────────────────────────────────────

def ping_qwen() -> bool:
    try:
        r = requests.get(QWEN_URL.replace('/v1/chat/completions', '/health'),
                         timeout=5)
        return r.status_code == 200
    except Exception:
        pass
    # Fallback: try a minimal completion
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

def _volume_mad_zscore(con: duckdb.DuckDBPyConnection,
                       new_df: pd.DataFrame) -> dict[str, float]:
    """Volume Z-score using median/MAD instead of mean/stddev."""
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
    print("  This will take several minutes — Binance rate limits apply.\n")

    total = 0
    for symbol in SYMBOLS:
        asset = symbol.split('/')[0]
        print(f"  {asset:>6s} … ", end='', flush=True)
        rows, cursor, retries = [], since_ms, 0

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
                    'price_usd':    float(c[4]),
                    'volume_24h':   float(c[5]),
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

    print(f"\nBackfill complete. {total:,} rows inserted. "
          f"DB total: {con.execute('SELECT COUNT(*) FROM market_history').fetchone()[0]:,}")


# ── Fee helpers ───────────────────────────────────────────────────────────────

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
        'notional':      round(notional,      4),
        'fee_per_leg':   round(fee_per_leg,   6),
        'num_legs':      num_legs,
        'total_fee':     round(total_fee,     6),
        'spread_cost':   round(spread_cost,   6),
        'total_cost':    round(total_cost,    6),
        'breakeven_pct': round(breakeven_pct, 6),
    }


# ── Median/MAD Z-score ────────────────────────────────────────────────────────

def mad_zscore(values: np.ndarray, current: float) -> float:
    """
    Robust Z-score using median and MAD.
    1.4826 consistency factor makes it comparable to stddev for normal data.
    Threshold of -2.0 remains meaningful without recalibration.
    """
    if len(values) < 5:
        return 0.0
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    denom  = MAD_CONSISTENCY * mad
    return float((current - median) / denom) if denom > 0 else 0.0


# ── Reputation score ──────────────────────────────────────────────────────────

def get_pair_reputations(con: duckdb.DuckDBPyConnection,
                         pairs: list[str]) -> dict[str, float]:
    """
    For each pair, compute reputation = tanh(avg_outcome * win_rate * log(n+1)).
    Bounded -1 to +1. Returns 0.0 (neutral) when < REP_MIN_SAMPLES.
    Also returns per-asset reputation for fiat mode (keyed by asset/USDT).
    """
    if not pairs:
        return {}
    ph = ','.join(['?' for _ in pairs])
    df = con.execute(f"""
        SELECT top_pair,
               COUNT(*)                                           AS n,
               AVG(outcome_pct)                                   AS avg_out,
               SUM(CASE WHEN outcome_pct > 0 THEN 1.0 ELSE 0 END)
                 / COUNT(*)                                       AS win_rate
        FROM llm_verdicts
        WHERE top_pair IN ({ph})
          AND outcome_pct IS NOT NULL
          AND action = 'accumulate'
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
    if rep is None or rep == 0.0:
        return "neutral"
    if rep >= 0.5:  return "strong✅"
    if rep >= 0.2:  return "positive"
    if rep >= 0.0:  return "mild+"
    if rep >= -0.2: return "mild-"
    if rep >= -0.5: return "negative"
    return "poor⚠"


# ── Trend filter ──────────────────────────────────────────────────────────────

def get_pair_trend_slopes(con: duckdb.DuckDBPyConnection,
                          pairs: list[str]) -> dict[str, float]:
    """
    For each pair, compute the linear regression slope of the price ratio
    over the last 500 observations. Normalised by median ratio.
    High absolute slope = trending pair, not oscillating.
    """
    result = {}
    for pair in pairs:
        parts = pair.split('/')
        if len(parts) != 2:
            result[pair] = 0.0
            continue
        base, quote = parts

        df = con.execute("""
            SELECT DATE_TRUNC('minute', a.timestamp) AS ts,
                   a.price_usd / NULLIF(b.price_usd, 0) AS ratio
            FROM market_history a
            JOIN market_history b
              ON DATE_TRUNC('minute', a.timestamp) = DATE_TRUNC('minute', b.timestamp)
             AND b.asset_id = ?
            WHERE a.asset_id = ?
            ORDER BY ts DESC
            LIMIT 500
        """, [quote, base]).df()

        if len(df) < 20:
            result[pair] = 0.0
            continue

        y      = df['ratio'].values[::-1]   # chronological order
        x      = np.arange(len(y))
        slope  = float(np.polyfit(x, y, 1)[0])
        median = float(np.median(y))
        # Normalise slope by median ratio so scale is comparable across pairs
        result[pair] = round(slope / median if median != 0 else 0.0, 6)

    return result


# ── Signal engine ─────────────────────────────────────────────────────────────

def get_raw_anomalies(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Cross-pair Z-score using median/MAD computed in Python after fetching ratios.
    DuckDB pulls the ratio history; Python computes robust Z-score.
    """
    # Get latest tick per asset
    latest = con.execute("""
        SELECT asset_id, price_usd, vwap_ratio, volume_z, spread_pct
        FROM market_history
        WHERE timestamp = (SELECT MAX(timestamp) FROM market_history)
    """).df().set_index('asset_id')

    if latest.empty:
        return pd.DataFrame()

    assets = latest.index.tolist()
    rows   = []

    for base in assets:
        for quote in assets:
            if base == quote:
                continue

            base_px  = latest.loc[base,  'price_usd']
            quote_px = latest.loc[quote, 'price_usd']
            if not quote_px or quote_px == 0:
                continue

            current_ratio = base_px / quote_px

            # Fetch historical ratios for this pair
            hist = con.execute("""
                SELECT DATE_TRUNC('minute', a.timestamp) AS ts,
                       a.price_usd / NULLIF(b.price_usd, 0) AS ratio
                FROM market_history a
                JOIN market_history b
                  ON DATE_TRUNC('minute', a.timestamp) = DATE_TRUNC('minute', b.timestamp)
                 AND b.asset_id = ?
                WHERE a.asset_id = ?
                ORDER BY ts DESC
                LIMIT 2000
            """, [quote, base]).df()

            if len(hist) < MIN_PAIR_SAMPLES:
                continue

            ratios    = hist['ratio'].dropna().values
            z         = mad_zscore(ratios, current_ratio)
            med_ratio = float(np.median(ratios))

            if z >= ZSCORE_BUY_THRESH:
                continue

            rows.append({
                'pair':            f"{base}/{quote}",
                'base_asset':      base,
                'quote_asset':     quote,
                'ratio':           round(current_ratio, 6),
                'mean_ratio':      round(med_ratio, 6),   # median, named for compat
                'z_score':         round(z, 3),
                'vwap_ratio':      latest.loc[base, 'vwap_ratio'],
                'volume_z':        latest.loc[base, 'volume_z'],
                'spread_pct':      latest.loc[base, 'spread_pct'],
                'sample_count':    len(ratios),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('z_score').head(TOP_N_RAW).reset_index(drop=True)
    return df


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


def rule_based_score(row, persistence, prev_z, reputations, slopes) -> float:
    score = row['z_score']

    # Persistence
    score -= min(persistence.get(row['pair'], 0) * 0.1, 0.5)

    # Velocity
    prev = prev_z.get(row['pair'])
    if prev is not None:
        score -= 0.2 if row['z_score'] < prev else -0.2

    # Volume / VWAP confirmation
    if row['volume_z']   is not None and row['volume_z']   >  1.5: score -= 0.15
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99: score -= 0.1

    # Liquidity penalty
    if row['spread_pct'] is not None and row['spread_pct'] >  0.1: score += 0.3

    # Meme quote penalty
    if row['quote_asset'] in MEME_ASSETS: score += 0.5

    # Reputation adjustment (good rep = lower score = better rank)
    rep = reputations.get(row['pair'], 0.0)
    score -= rep * 0.5   # rep=+1 gives -0.5 bonus, rep=-1 gives +0.5 penalty

    # Trend penalty (trending pairs are less likely to revert)
    slope = abs(slopes.get(row['pair'], 0.0))
    if slope > TREND_SLOPE_THRESHOLD:
        score += slope * 10   # penalise proportional to trend strength

    return round(score, 4)


def build_candidates(con) -> tuple[pd.DataFrame, str | None, str | None]:
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
        lambda r: rule_based_score(r, persistence, prev_z, reputations, slopes), axis=1
    )

    quote_warn = detect_dominance(raw, 'quote_asset')
    base_warn  = detect_dominance(raw, 'base_asset')
    top = raw.sort_values('rule_score').head(TOP_N_LLM).reset_index(drop=True)
    return top, quote_warn, base_warn


def derive_confidence(candidates: pd.DataFrame, top_pair: str) -> int:
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
    # Reputation modifier
    rep = row.get('reputation')
    if rep is not None:
        if rep >= 0.5:  conf = min(conf + 1, 10)
        elif rep <= -0.5: conf = max(conf - 2, 1)
    # Trend modifier
    slope = abs(row.get('trend_slope', 0.0))
    if slope > TREND_SLOPE_THRESHOLD:
        conf = max(conf - 1, 1)
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
        INSERT OR IGNORE INTO fantasy_balances
            (timestamp, asset_id, qty, price_usd, value_usd)
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


def _auto_close_trades(con: duckdb.DuckDBPyConnection,
                       prices: dict[str, float]) -> int:
    """
    Close open fantasy trades where the position has reverted past break-even.
    Returns number of trades closed.
    """
    open_trades = con.execute("""
        SELECT trade_id, buy_asset, sell_asset, buy_qty,
               buy_price_usd, notional_usd, total_cost_usd
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    closed = 0
    for _, t in open_trades.iterrows():
        cur_px      = prices.get(t['buy_asset'], t['buy_price_usd'])
        current_val = t['buy_qty'] * cur_px
        pnl_usd     = current_val - t['notional_usd']
        pnl_pct     = pnl_usd / t['notional_usd'] * 100 if t['notional_usd'] else 0
        breakeven   = t['total_cost_usd'] / t['notional_usd'] * 100

        # Close if we've recouped cost + 50% more (i.e., meaningfully reverted)
        if pnl_pct >= breakeven * 1.5:
            ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            con.execute("""
                UPDATE fantasy_trades
                SET status = 'closed', close_ts = ?,
                    close_buy_px = ?, pnl_usd = ?, pnl_pct = ?
                WHERE trade_id = ?
            """, [ts, cur_px, round(pnl_usd, 4), round(pnl_pct, 4), t['trade_id']])
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
    balances[buy_asset]  = balances.get(buy_asset,  0.0) + buy_qty

    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{sell_asset}/{buy_asset}"
    return {
        'trade_id':       _trade_id(ts, pair),
        'timestamp':      ts,
        'mode':           'pair',
        'action':         'pair_reversion',
        'sell_asset':     sell_asset,
        'buy_asset':      buy_asset,
        'sell_qty':       round(sell_qty, 8),
        'buy_qty':        round(buy_qty,  8),
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
                entry_px = con.execute(
                    "SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp<=? "
                    "ORDER BY timestamp DESC LIMIT 1", [base, sig_ts]).fetchone()
                exit_px = con.execute(
                    "SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp>=? "
                    "ORDER BY timestamp ASC LIMIT 1", [base, out_ts]).fetchone()
                if not entry_px or not exit_px:
                    continue
                outcome_pct = round((exit_px[0] - entry_px[0]) / entry_px[0] * 100, 4)
            else:
                def _px(asset, ts, direction):
                    op = 'DESC' if direction == 'entry' else 'ASC'
                    cmp = '<=' if direction == 'entry' else '>='
                    return con.execute(
                        f"SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp{cmp}? "
                        f"ORDER BY timestamp {op} LIMIT 1", [asset, ts]).fetchone()
                eb, eq = _px(base, sig_ts, 'entry'), _px(quote, sig_ts, 'entry')
                xb, xq = _px(base, out_ts, 'exit'),  _px(quote, out_ts, 'exit')
                if not all([eb, eq, xb, xq]):
                    continue
                entry_ratio = eb[0] / eq[0]
                exit_ratio  = xb[0] / xq[0]
                outcome_pct = round((exit_ratio - entry_ratio) / entry_ratio * 100, 4)

            con.execute("""
                UPDATE llm_verdicts SET outcome_pct=?, outcome_ts=?
                WHERE timestamp=? AND top_pair=?
            """, [outcome_pct, out_ts, sig_ts, pair])
            scored += 1

        print(f"  Scored {scored} outcomes.")

    # Overall summary
    summary = con.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS winners,
               ROUND(AVG(outcome_pct), 3)  AS avg,
               ROUND(MAX(outcome_pct), 3)  AS best,
               ROUND(MIN(outcome_pct), 3)  AS worst
        FROM llm_verdicts
        WHERE action='accumulate' AND outcome_pct IS NOT NULL
    """).fetchone()

    if summary and summary[0] > 0:
        wr = round(summary[1] / summary[0] * 100, 1)
        print(f"\n── Signal accuracy ({OUTCOME_WINDOW_HOURS}h window) ──")
        print(f"  Signals scored : {summary[0]}")
        print(f"  Win rate       : {summary[1]}/{summary[0]} ({wr}%)")
        print(f"  Avg outcome    : {summary[2]:+.3f}%")
        print(f"  Best / Worst   : {summary[3]:+.3f}% / {summary[4]:+.3f}%")

    # Per-pair breakdown
    print(f"\n── Per-pair breakdown ──")
    breakdown = con.execute("""
        SELECT top_pair,
               COUNT(*)                                           AS n,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS wins,
               ROUND(AVG(outcome_pct), 3)                        AS avg_out,
               ROUND(MIN(outcome_pct), 3)                        AS worst,
               ROUND(MAX(outcome_pct), 3)                        AS best
        FROM llm_verdicts
        WHERE action='accumulate' AND outcome_pct IS NOT NULL
        GROUP BY top_pair
        ORDER BY avg_out DESC
    """).df()
    # Add reputation column
    pairs = breakdown['top_pair'].tolist()
    reps  = get_pair_reputations(con, pairs)
    breakdown['reputation'] = breakdown['top_pair'].map(
        lambda p: f"{reps.get(p, 0.0):+.3f}" if p in reps else "n/a"
    )
    print(breakdown.to_string(index=False))


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


def _log_verdict(con, verdict, confidence, top_row, mode,
                 reputation=None, trend_slope=None) -> None:
    ts       = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score  = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'
    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, mode, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json, reputation, trend_slope)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        ts, mode, top_pair, z_score, confidence,
        verdict.get('action',   'pass'),
        verdict.get('thesis',   ''),
        verdict.get('risks',    ''),
        verdict.get('prose',    ''),
        verdict.get('raw_json', ''),
        reputation,
        trend_slope,
    ])


# ── Pair analysis ─────────────────────────────────────────────────────────────

def build_pair_prompt(candidates, quote_warn, base_warn) -> str:
    rows = []
    for i, r in candidates.iterrows():
        vel    = f"{r['z_velocity']:+.3f}"   if r['z_velocity']   is not None else "n/a"
        spread = f"{r['spread_pct']:.4f}%"   if r['spread_pct']   is not None else "n/a"
        vwap   = f"{r['vwap_ratio']:.4f}"    if r['vwap_ratio']   is not None else "n/a"
        vol_z  = f"{r['volume_z']:+.3f}"     if r['volume_z']     is not None else "n/a"
        rep    = r.get('reputation')
        rep_s  = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral (no data)"
        slope  = r.get('trend_slope', 0.0)
        trend_flag = f" ⚠TRENDING" if abs(slope) > TREND_SLOPE_THRESHOLD else ""
        route, _ = trade_route(r['base_asset'], r['quote_asset'])
        rows.append(
            f"  [rank {i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel} | "
            f"persist={r['persistence_24h']} | vwap={vwap} | vol_z={vol_z} | "
            f"spread={spread} | rep={rep_s}{trend_flag} | route={route}"
        )
    table = "\n".join(rows)
    warn  = ""
    if quote_warn:
        warn += (f"\n⚠ QUOTE DOMINANCE: {quote_warn} is the quote in multiple pairs. "
                 f"May mean {quote_warn} is overbought. Address in thesis.\n")
    if base_warn:
        warn += (f"\n⚠ BASE DOMINANCE: {base_warn} is the base in multiple pairs. "
                 f"Likely one signal, not independent.\n")

    return f"""You are a disciplined spot-only crypto trader. Rule-based engine pre-ranked {len(candidates)} candidates.

Candidates (best first):
{table}
{warn}
Column guide:
- z: robust Z-score (median/MAD). More negative = cheaper vs history.
- vel: Z delta since last analysis. Negative = deepening (good). Positive = recovering.
- persist: 24h accumulate signal count.
- vwap: last/VWAP. <1.0 = below volume-weighted average.
- vol_z: volume anomaly (median/MAD). >1.5 = unusual interest.
- spread: bid-ask %. >0.15% = liquidity concern.
- rep: historical reputation score (-1 to +1). Positive = proven reverter. Negative = historically poor.
- ⚠TRENDING: ratio has significant linear trend — mean reversion less likely.

Rules:
- Rule rank is authoritative. Override requires explicit justification.
- Poor reputation (rep < -0.3) + no strong Z (> -3.0): set action=watch not accumulate.
- Strong reputation (rep > 0.4) + Z < -1.8: can accumulate even at moderate Z.
- ⚠TRENDING pairs: accumulate only if Z < -3.0 and rep > 0.3.
- Do NOT assign equal confidence. spread >0.15% caps confidence at 6.
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
    rep         = top_row.get('reputation')
    rep_s       = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
    slope       = top_row.get('trend_slope', 0.0)

    print("\n╔══ TRADE SIGNAL ══════════════════════════════════════════╗")
    print(f"  Mode       : PAIR REVERSION (spot)")
    print(f"  Action     : Sell {base} → Buy {quote}  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Route      : {route}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Sell       : {sell_qty} {base}  @ {base_px:,.4f} USDT")
    print(f"  Buy        : {buy_qty} {quote}  @ {quote_px:,.4f} USDT")
    print(f"  Notional   : {fees['notional']} USDT  ({TRADE_PCT*100:.0f}% of {base} holding)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['fee_per_leg']} USDT × {fees['num_legs']} legs = {fees['total_fee']} USDT  (0.075% BNB)")
    if top_row['spread_pct']:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({top_row['spread_pct']:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of notional)")
    print(f"  Break-even : ratio must improve by {fees['breakeven_pct']:.4f}%")
    print(f"  ─────────────────────────────────────────────────────")
    vel = top_row['z_velocity'] if top_row['z_velocity'] is not None else 'n/a'
    print(f"  Signal     : Z={top_row['z_score']:+.3f}  persist={top_row['persistence_24h']}  vel={vel}")
    print(f"  Reputation : {rep_s}")
    if abs(slope) > TREND_SLOPE_THRESHOLD:
        print(f"  ⚠ Trend    : slope={slope:.6f} — pair may be trending, not oscillating")
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
        print(f"  ⚠ WARNING: Qwen unreachable at {QWEN_URL} — check port/service")

    candidates, quote_warn, base_warn = build_candidates(con)

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'pair')
        return

    if quote_warn:
        print(f"\n  ⚠ Quote dominance : {quote_warn}")
    if base_warn:
        print(f"  ⚠ Base dominance  : {base_warn} — likely one signal dressed as multiple")

    print("\n── Candidates (rule-scored) ──")
    disp = candidates[['pair', 'z_score', 'z_velocity', 'persistence_24h',
                        'vwap_ratio', 'volume_z', 'spread_pct',
                        'reputation', 'trend_slope', 'rule_score']].copy()
    disp['reputation'] = disp['reputation'].apply(
        lambda r: f"{r:+.3f}" if r is not None else "n/a"
    )
    print(disp.to_string(index=False))

    prompt     = build_pair_prompt(candidates, quote_warn, base_warn)
    verdict    = ask_qwen(prompt)
    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair)
    prices     = get_current_prices(con)

    # Auto-close reverted trades
    closed = _auto_close_trades(con, prices)
    if closed:
        print(f"\n  ✅ Auto-closed {closed} reverted fantasy trade(s)")

    top_row = None
    rep_val = None
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
        notional = notional_for(base, base_px, balances, size_override)
        format_pair_trade(top_row, prices, verdict, confidence, notional)
        discord_signal(verdict, confidence, top_row, prices, notional, 'pair')

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
                    discord_trade(trade)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient {base} balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'pair', rep_val, slope_val)


# ── Fiat analysis ─────────────────────────────────────────────────────────────

def get_fiat_candidates(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, bool]:
    date_range = con.execute("""
        SELECT DATEDIFF('day', MIN(timestamp), MAX(timestamp))
        FROM market_history WHERE source = 'live'
    """).fetchone()[0] or 0
    short_history = date_range < 14

    # Fetch latest prices and history per asset, compute median/MAD Z-score in Python
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
            [asset]
        ).fetchdf()
        if len(hist) < MIN_PAIR_SAMPLES:
            continue
        prices_arr = hist['price_usd'].dropna().values
        z          = mad_zscore(prices_arr, r['price_usd'])
        if z >= ZSCORE_BUY_THRESH:
            continue
        rows.append({
            'asset_id':   asset,
            'price_usd':  round(r['price_usd'], 6),
            'mean_price': round(float(np.median(prices_arr)), 6),
            'z_score':    round(z, 3),
            'vwap_ratio': r['vwap_ratio'],
            'volume_z':   r['volume_z'],
            'spread_pct': r['spread_pct'],
            'change_24h': r['change_24h'],
            'sample_count': len(prices_arr),
        })

    df = pd.DataFrame(rows).sort_values('z_score').head(TOP_N_LLM).reset_index(drop=True) \
        if rows else pd.DataFrame()
    return df, short_history


def build_fiat_prompt(candidates, reputations) -> str:
    rows = []
    for i, r in candidates.iterrows():
        pair = f"{r['asset_id']}/USDT"
        rep  = reputations.get(pair)
        rep_s = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
        rows.append(
            f"  [rank {i+1}] {r['asset_id']:>6s}/USDT | price={r['price_usd']:,.4f} | "
            f"z={r['z_score']:+.3f} | vwap={r['vwap_ratio'] or 'n/a'} | "
            f"vol_z={r['volume_z'] or 'n/a'} | "
            f"24h={r['change_24h']:+.2f}% | spread={r['spread_pct'] or 'n/a'}% | rep={rep_s}"
        )
    table = "\n".join(rows)
    return f"""You are a disciplined spot-only crypto trader advising on USDT buy entries.
Statistically cheap assets (robust Z < {ZSCORE_BUY_THRESH}, median/MAD):
{table}

Rules: spot only, no leverage.
- Poor reputation (rep < -0.3): watch only unless Z < -3.0.
- Strong reputation (rep > 0.4): can accumulate at Z < -1.8.
- Avoid assets in freefall (very negative 24h, no volume support).
- Rank all. Do not assign equal confidence. action: accumulate|watch|pass.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_asset":"<ASSET>","rank":["<a1>",...],"thesis":"<str>","risks":"<str>"}}
"""


def format_fiat_trade(asset, price, spread_pct, z_score,
                      vwap_ratio, volume_z, change_24h,
                      verdict, confidence, notional, rep) -> None:
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = round((notional - fees['total_cost']) / price, 8) if price else 0
    rep_s   = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"

    print("\n╔══ FIAT BUY SIGNAL ═══════════════════════════════════════╗")
    print(f"  Mode       : USDT → {asset} (spot buy)")
    print(f"  Action     : Buy {asset}/USDT  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Price      : {price:,.4f} USDT")
    print(f"  Qty        : {buy_qty} {asset}  from {fees['notional']} USDT")
    print(f"  Notional   : {fees['notional']} USDT  ({TRADE_PCT*100:.0f}% of USDT balance)")
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
    print(f"  Reputation : {rep_s}")
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
        print(f"  ⚠ WARNING: Qwen unreachable at {QWEN_URL} — check port/service")

    candidates, short_history = get_fiat_candidates(con)

    if short_history:
        print("  ⚠ DB history < 14 days — fiat Z-scores may not reflect full price range yet")

    if candidates.empty:
        print("  No assets statistically cheap vs history — no entry signal.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'No fiat entry signal.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'fiat')
        return

    # Get reputations for fiat pairs
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

    prompt    = build_fiat_prompt(candidates, reputations)
    verdict   = ask_qwen(prompt)
    top_asset = verdict.get('top_asset', '')
    verdict['top_pair'] = f"{top_asset}/USDT" if top_asset else 'none'

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
        rep_val    = reputations.get(f"{top_asset}/USDT")
        if rep_val is not None:
            if rep_val >= 0.5:  conf = min(conf + 1, 10)
            elif rep_val <= -0.5: conf = max(conf - 2, 1)
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
            verdict, confidence, notional, rep_val
        )
        discord_signal(verdict, confidence, top_row, prices, notional, 'fiat')

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
                    discord_trade(trade)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient USDT balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")

    _log_verdict(con, verdict, confidence, top_row, 'fiat', rep_val)


# ── Portfolio ─────────────────────────────────────────────────────────────────

def show_portfolio(con: duckdb.DuckDBPyConnection) -> None:
    prices = get_current_prices(con)

    print("── Fantasy Portfolio ──")
    trades = con.execute("""
        SELECT trade_id, timestamp, action, sell_asset, buy_asset,
               sell_qty, buy_qty, sell_price_usd, buy_price_usd,
               notional_usd, total_cost_usd, signal_z, confidence, status,
               pnl_usd, pnl_pct
        FROM fantasy_trades ORDER BY timestamp DESC
    """).df()

    if trades.empty:
        print("  No fantasy trades executed yet.")
        return

    # Fetch reputations for all traded pairs
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
        else:
            pnl_usd = t['pnl_usd'] or 0.0
            pnl_pct = t['pnl_pct'] or 0.0

        rep = reps.get(pair)
        rows_out.append({
            'ts':        str(t['timestamp'])[:16],
            'trade':     f"{t['sell_asset']}→{t['buy_asset']}",
            'notional':  f"{t['notional_usd']:.2f}",
            'z':         f"{t['signal_z']:+.3f}",
            'conf':      t['confidence'],
            'rep':       f"{rep:+.3f}" if rep is not None else "n/a",
            'status':    t['status'],
            'pnl_usd':   f"{pnl_usd:+.4f}",
            'pnl_pct':   f"{pnl_pct:+.3f}%",
        })

    print(pd.DataFrame(rows_out).to_string(index=False))

    open_trades = [r for r in rows_out if r['status'] == 'open']
    closed_trades = [r for r in rows_out if r['status'] == 'closed']
    total_pnl   = sum(float(r['pnl_usd']) for r in rows_out)
    realized    = sum(float(r['pnl_usd']) for r in closed_trades)
    print(f"\n  Open positions  : {len(open_trades)}")
    print(f"  Closed          : {len(closed_trades)}")
    print(f"  Realized P&L    : {realized:+.4f} USDT")
    print(f"  Total P&L       : {total_pnl:+.4f} USDT (incl. unrealized)")

    latest_bal = con.execute("""
        SELECT asset_id, qty, price_usd, value_usd
        FROM fantasy_balances
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

    print(f"── DB Status ──")
    print(f"  market_history : {rows} rows  (live={live}, backfill={bf}, latest={latest})")
    print(f"  llm_verdicts   : {verdicts} rows")
    print(f"  fantasy_trades : {trades} rows")
    print(f"  Qwen           : {'✅ reachable' if qwen_ok else '⚠ unreachable'} at {QWEN_URL}")

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
               action, outcome_pct, reputation
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals ──")
    print(con.execute("""
        SELECT mode, top_pair, COUNT(*) AS hits,
               ROUND(AVG(confidence_score),1) AS avg_conf,
               ROUND(AVG(z_score),3) AS avg_z,
               ROUND(AVG(outcome_pct),3) AS avg_outcome,
               ROUND(AVG(reputation),3) AS avg_rep
        FROM llm_verdicts WHERE action='accumulate'
        GROUP BY mode, top_pair ORDER BY hits DESC LIMIT 15
    """).df().to_string(index=False))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.6")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',      action='store_true')
    grp.add_argument('--analyze',     action='store_true')
    grp.add_argument('--fiat',        action='store_true')
    grp.add_argument('--backfill',    action='store_true')
    grp.add_argument('--outcomes',    action='store_true')
    grp.add_argument('--portfolio',   action='store_true')
    grp.add_argument('--daily-pnl',   action='store_true', help='Send daily P&L to Discord')
    grp.add_argument('--init-db',     action='store_true')
    grp.add_argument('--status',      action='store_true')
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
