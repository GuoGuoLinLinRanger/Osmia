#!/usr/bin/env python3
"""
TraderV10.py  —  v1.0
Unified ingest + analyze + paper trading script for the Osmia LLM trading monitor.

NOTE: run with venv/bin/python3 (the legacy venv has all deps; .venv/uv does NOT).

Usage:
    venv/bin/python3 TraderV10.py                  # ingest + analyze (pair mode)
    venv/bin/python3 TraderV10.py --ingest         # ingest only
    venv/bin/python3 TraderV10.py --analyze        # pair signal analysis
    venv/bin/python3 TraderV10.py --fiat           # fiat/USDT entry analysis
    venv/bin/python3 TraderV10.py --backfill       # one-shot OHLCV backfill (run once)
    venv/bin/python3 TraderV10.py --outcomes       # score past signals vs price moves
    venv/bin/python3 TraderV10.py --portfolio      # show fantasy portfolio state
    venv/bin/python3 TraderV10.py --init-db        # (re)create DB schema and exit
    venv/bin/python3 TraderV10.py --status         # quick DB health summary
    venv/bin/python3 TraderV10.py --daily-pnl      # send daily P&L summary to Discord
    venv/bin/python3 TraderV10.py --scrape-news    # fetch per-asset news -> asset_news
    venv/bin/python3 TraderV10.py --scrape-unlocks # fetch token unlock calendar
    venv/bin/python3 TraderV10.py --size 250       # override notional for this run
    venv/bin/python3 TraderV10.py --mine-proceeds  # inject nightly mine revenue into balance
    venv/bin/python3 TraderV10.py --mine-proceeds --mine-value 15.50  # custom USDT amount

Changes from v0.9:
    - Feature 1: F&G-aware stop-loss. Fires before Buffett exit check every --analyze
      cycle. Thresholds: Extreme Fear -8%, Fear -10%, Neutral -12%, Greed -15%,
      Extreme Greed -18%. Closes immediately, logs stop_loss=True to exit_log.
      Discord: 🛑 Stop-loss triggered. Pair not blacklisted; cooldown still applies.
    - Feature 2: Living context re-evaluation. Every --analyze cycle, targeted
      asset_news refresh for held assets, then one Qwen call per open position with
      fresh macro + news context. EXIT urgency>=7 → force close (news_veto_live).
      EXIT urgency<7 → Discord warning only. HOLD → continue.
    - Feature 3: Nightly mine proceeds injection. --mine-proceeds [--mine-value N]
      flag adds USDT to fantasy balance. 20h guard prevents double injection.
      mine_proceeds table tracks every injection. Posts to Discord.

Changes from v0.8:
    - Feature 1: Symmetric pair collapse — canonical_pair()/pair_id() helpers,
      pair_reputation table stores reputation keyed by canonical (alpha-sorted) pair.
      Position limit (_pair_at_max_trades) uses canonical pair_id so BTC/ETH and
      ETH/BTC share the same 3-trade cap. One-time migration merges existing
      directional reputation into pair_reputation.
    - Feature 2: Three-layer F&G Discord header prepended to every pair/fiat signal:
        📊 Market F&G: 34 (Fear) | Inst: ETF -$400M (Bearish) | Trade: 4/10 (Caution)
      macro_cache table caches F&G value and CoinGlass BTC ETF flows (1h TTL).
      Layer 3 is the Qwen confidence score expressed as Caution/Neutral/Confident.
    - Feature 3: Buffett Inverse F&G exit — replaces hardcoded ZSCORE_CLOSE_THRESH
      with buffett_exit_threshold(fg_value, etf_flow_m). Extreme Fear holds longer
      (z_target=1.5); Extreme Greed exits early (z_target=-0.5). Heavy ETF outflow
      adds +0.3 (contrarian hold); heavy inflow subtracts 0.2. News veto: asset_news
      keywords ['hack','exploit','SEC','ban','delist','insolvency','shutdown','rug']
      in last 6h block auto-close. All exits logged to exit_log table.
    - Feature 4: Top 50 watchlist — ingest() uses CoinGecko /coins/markets (top 50
      by market cap, stablecoins excluded) cached in macro_cache for 24h. Falls back
      to existing ASSETS list. Up to 1,225 canonical pairs.

Changes from v0.7:
    - Layer 1 macro context: BTC 24h%, BTC dominance + total mcap 24h change
      (CoinGecko /global, field market_cap_percentage.btc) combined with F&G into a
      [MACRO CONTEXT] block prepended to every --analyze Qwen prompt.
    - Layer 2 per-asset news: scrape_news() pulls CoinDesk RSS (keyword-filtered) +
      Reddit r/<sub>/new.json + DeFiLlama chain TVL into asset_news (24h-pruned).
      Last-6h headlines for each leg of the scored pair injected into the prompt.
      (Reddit may 403 from datacenter/VPS IPs — degrades gracefully to no rows.)
    - Token unlock calendar: scrape_unlocks() best-effort fetch -> token_unlocks;
      --scrape-unlocks flag. Watchlist UNLOCK_ASSETS (APT, ATOM) — extensible.
    - Heightened scan mode: a pending unlock (<48h) for either leg relaxes the
      Z-buy threshold (-2.0 -> -1.5) for that asset's pairs, injects an UNLOCK ALERT
      into the macro block, and appends HEIGHTENED_PROMPT_SUFFIX. State persisted in
      bot_state (heightened_<ASSET>_until = unlock_at + 6h relief window).
    - --scrape-news runs automatically at the head of --analyze (lock-free network).
    - feedparser added as a dependency (RSS parsing).

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
import feedparser
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

# ── v0.9: canonical pair helpers (Feature 1) ─────────────────────────────────

def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def pair_id(a: str, b: str) -> str:
    ca, cb = canonical_pair(a, b)
    return f"{ca}/{cb}"


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
NEWS_COUNT = 5     # general macro headlines to feed Qwen

# ── v0.8: macro / news / unlocks ────────────────────────────────────────────────
USER_AGENT          = "osmia-trader/0.8"
# CoinGecko /global. NOTE: dominance lives at data.market_cap_percentage.btc — the
# brief's data.bitcoin_dominance does NOT exist on this endpoint (verified live).
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

# Layer 2 per-asset news
NEWS_LOOKBACK_HOURS = 24   # asset_news retention; pruned on each scrape run
NEWS_PROMPT_HOURS   = 6    # window of per-leg headlines injected into the prompt
NEWS_PROMPT_COUNT   = 5    # max headlines per leg in the prompt
REDDIT_LIMIT        = 5

# Full-name + subreddit + DeFiLlama-chain maps for the watchlist (ATOM included
# for unlock coverage even though it isn't a traded pair yet).
ASSET_NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano',
    'DOT': 'Polkadot', 'XRP': 'XRP', 'TRX': 'Tron', 'BNB': 'BNB',
    'AVAX': 'Avalanche', 'LTC': 'Litecoin', 'XLM': 'Stellar', 'APT': 'Aptos',
    'DOGE': 'Dogecoin', 'SUI': 'Sui', 'ATOM': 'Cosmos',
}
SUBREDDITS = {
    'BTC': 'Bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano',
    'DOT': 'Polkadot', 'XRP': 'Ripple', 'TRX': 'Tronix', 'BNB': 'binance',
    'AVAX': 'Avax', 'LTC': 'litecoin', 'XLM': 'Stellar', 'APT': 'Aptos',
    'DOGE': 'dogecoin', 'SUI': 'SuiNetwork', 'ATOM': 'cosmosnetwork',
}
# DeFiLlama /v2/chains name per asset — chains only (BTC/BNB/DOGE/LTC/XLM skipped).
# That endpoint exposes tvl + name only (no tvlChange1d, verified live), so we store
# a TVL snapshot headline rather than a 1d delta.
DEFILLAMA_CHAINS = {
    'ETH': 'Ethereum', 'SOL': 'Solana', 'ADA': 'Cardano', 'DOT': 'Polkadot',
    'AVAX': 'Avalanche', 'APT': 'Aptos', 'SUI': 'Sui', 'TRX': 'Tron',
    'XRP': 'XRP', 'ATOM': 'Cosmos',
}
CHAINS_URL    = "https://api.llama.fi/v2/chains"

# Token unlocks / heightened mode
UNLOCK_ASSETS              = ['APT', 'ATOM']   # watchlist for the unlock calendar
UNLOCK_HORIZON_HOURS       = 48                # pending unlock within this -> heightened
HEIGHTENED_POST_UNLOCK_HRS = 6                 # stay heightened this long after unlock
ZSCORE_BUY_THRESH_HEIGHTENED = -1.5            # relaxed dump threshold under heightened mode
# Token unlocks — DeFiLlama's free emissions dataset host. NOTE: api.llama.fi's
# /emissions + /emission/<slug> are now 402 (paid tier); the datasets host below
# still serves the same data free. Events live at metadata.events; USD values are
# not in the free feed (unlockUsdChart is empty), so amount_usd stays NULL and we
# derive pct_supply from supplyMetrics.maxSupply.
EMISSION_URL   = "https://defillama-datasets.llama.fi/emissions/{slug}"
# Map watchlist asset → DeFiLlama emissions slug. ATOM/Cosmos is NOT tracked by
# DeFiLlama emissions (no slug exists), so it has no source here and is skipped;
# wire a different provider if ATOM unlock coverage is needed.
EMISSION_SLUGS = {'APT': 'aptos'}

# ── v0.9: Feature 2 — CoinGlass ETF flows ──────────────────────────────────────
COINGLASS_ETF_URL   = "https://open-api.coinglass.com/public/v2/etf/bitcoin/flows?limit=1"
ETF_CACHE_SECONDS   = 3600   # 1h TTL for ETF flow in macro_cache

# ── v0.9: Feature 3 — news veto keywords ──────────────────────────────────────
NEWS_VETO_KEYWORDS  = ['hack', 'exploit', 'SEC', 'ban', 'delist',
                       'insolvency', 'shutdown', 'rug']

# ── v0.9: Feature 4 — top-50 watchlist ────────────────────────────────────────
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
STABLECOINS         = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD', 'PYUSD',
                       'USDS', 'USDE', 'USDD', 'GUSD', 'USDJ', 'XUSD', 'CRVUSD', 'FRAX',
                       'USD1', 'USDG', 'RLUSD', 'BFUSD', 'USDF', 'USDY', 'USYC',
                       'USDX', 'DEUSD', 'LISUSD', 'SUSD', 'MKUSD', 'EURC', 'EURI'}
# Tokens known to not be directly tradeable as USDT pairs on Binance:
# tokenized funds/RWA, foreign exchange tokens, low-quality/unverified launches
NON_BINANCE_TOKENS  = {
    'FIGR_HELOC', 'BUIDL', 'XAUT', 'PAXG',          # tokenized gold / funds
    'LEO', 'OKB', 'WBT', 'HTX', 'KCS', 'GT',         # exchange tokens not on Binance
    'RAIN', 'CC', 'LAB', 'M', 'ASTER', 'WLFI', 'PI', # low-quality / unverified
    'HYPE',                                            # Hyperliquid — no Binance USDT pair
}
WATCHLIST_CACHE_HOURS = 24

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

DDL_ASSET_NEWS = """
CREATE TABLE IF NOT EXISTS asset_news (
    fetched_at   TIMESTAMPTZ NOT NULL,
    asset_id     VARCHAR     NOT NULL,
    source       VARCHAR,
    headline     VARCHAR,
    published_at TIMESTAMPTZ,
    PRIMARY KEY (fetched_at, asset_id, source, headline)
);
"""

DDL_TOKEN_UNLOCKS = """
CREATE TABLE IF NOT EXISTS token_unlocks (
    unlock_id     VARCHAR PRIMARY KEY,
    asset_id      VARCHAR     NOT NULL,
    unlock_at     TIMESTAMPTZ NOT NULL,
    amount_tokens DOUBLE,
    amount_usd    DOUBLE,
    pct_supply    DOUBLE,
    unlock_type   VARCHAR,
    fetched_at    TIMESTAMPTZ NOT NULL
);
"""

DDL_BOT_STATE = """
CREATE TABLE IF NOT EXISTS bot_state (
    key        VARCHAR PRIMARY KEY,
    value      VARCHAR,
    updated_at TIMESTAMPTZ
);
"""

# v0.9 new tables

DDL_PAIR_REPUTATION = """
CREATE TABLE IF NOT EXISTS pair_reputation (
    pair_id     VARCHAR PRIMARY KEY,
    n_signals   INTEGER DEFAULT 0,
    n_wins      INTEGER DEFAULT 0,
    avg_outcome DOUBLE,
    win_rate    DOUBLE,
    rep_score   DOUBLE,
    updated_at  TIMESTAMPTZ
);
"""

DDL_MACRO_CACHE = """
CREATE TABLE IF NOT EXISTS macro_cache (
    key        VARCHAR PRIMARY KEY,
    value      VARCHAR,
    fetched_at TIMESTAMPTZ
);
"""

DDL_EXIT_LOG = """
CREATE TABLE IF NOT EXISTS exit_log (
    exit_id       VARCHAR PRIMARY KEY,
    timestamp     TIMESTAMPTZ,
    trade_id      VARCHAR,
    buy_asset     VARCHAR,
    pnl_usd       DOUBLE,
    pnl_pct       DOUBLE,
    reason        VARCHAR,
    fg_value      INTEGER,
    etf_flow_m    DOUBLE,
    z_threshold   DOUBLE,
    news_veto     BOOLEAN,
    stop_loss     BOOLEAN
);
"""

DDL_MINE_PROCEEDS = """
CREATE TABLE IF NOT EXISTS mine_proceeds (
    injected_at    TIMESTAMPTZ PRIMARY KEY,
    amount_usdt    DOUBLE NOT NULL,
    balance_before DOUBLE,
    balance_after  DOUBLE,
    note           VARCHAR DEFAULT 'nightly_mine_injection'
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


def _refresh_pair_reputation(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuild pair_reputation from llm_verdicts using canonical (alpha-sorted) pair_ids."""
    rows = con.execute(f"""
        SELECT cid, n, avg_out, win_rate FROM (
            SELECT
                CASE WHEN SPLIT_PART(top_pair,'/',1) < SPLIT_PART(top_pair,'/',2)
                     THEN top_pair
                     ELSE SPLIT_PART(top_pair,'/',2) || '/' || SPLIT_PART(top_pair,'/',1)
                END AS cid,
                COUNT(*) AS n,
                AVG(outcome_pct) AS avg_out,
                SUM(CASE WHEN outcome_pct > 0 THEN 1.0 ELSE 0 END) / COUNT(*) AS win_rate
            FROM llm_verdicts
            WHERE outcome_pct IS NOT NULL AND action = 'accumulate'
              AND top_pair LIKE '%/%'
            GROUP BY cid
        ) WHERE n >= {REP_MIN_SAMPLES}
    """).fetchall()
    if not rows:
        return
    now = datetime.now(timezone.utc)
    for cid, n, avg_out, win_rate in rows:
        if avg_out is None or win_rate is None:
            continue
        rep_score = round(math.tanh(avg_out * win_rate * math.log(n + 1)), 4)
        n_wins    = int(round(win_rate * n))
        con.execute("""
            INSERT INTO pair_reputation (pair_id, n_signals, n_wins, avg_outcome, win_rate, rep_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (pair_id) DO UPDATE SET
                n_signals   = excluded.n_signals,
                n_wins      = excluded.n_wins,
                avg_outcome = excluded.avg_outcome,
                win_rate    = excluded.win_rate,
                rep_score   = excluded.rep_score,
                updated_at  = excluded.updated_at
        """, [cid, int(n), n_wins, avg_out, win_rate, rep_score, now])


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL_MARKET_HISTORY)
    con.execute(DDL_LLM_VERDICTS)
    con.execute(DDL_FANTASY_TRADES)
    con.execute(DDL_FANTASY_BALANCES)
    con.execute(DDL_MARKET_CONTEXT)
    con.execute(DDL_ASSET_NEWS)          # v0.8
    con.execute(DDL_TOKEN_UNLOCKS)       # v0.8
    con.execute(DDL_BOT_STATE)           # v0.8
    con.execute(DDL_PAIR_REPUTATION)     # v0.9
    con.execute(DDL_MACRO_CACHE)         # v0.9
    con.execute(DDL_EXIT_LOG)            # v0.9
    con.execute(DDL_MINE_PROCEEDS)       # v1.0

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

    _add_col_if_missing(con, 'fantasy_trades', 'close_reason',       'VARCHAR')
    # v0.9
    _add_col_if_missing(con, 'fantasy_trades', 'canonical_pair_id',  'VARCHAR')
    # v1.0: stop_loss flag in exit_log
    _add_col_if_missing(con, 'exit_log',       'stop_loss',          'BOOLEAN')
    # Backfill canonical_pair_id for existing trades
    con.execute("""
        UPDATE fantasy_trades
        SET canonical_pair_id = CASE
            WHEN sell_asset < buy_asset THEN sell_asset || '/' || buy_asset
            ELSE buy_asset || '/' || sell_asset
        END
        WHERE canonical_pair_id IS NULL
    """)

    # v0.9: populate pair_reputation from llm_verdicts (one-time + refresh each run)
    _refresh_pair_reputation(con)

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


# ── v0.8: Layer 1 macro context ─────────────────────────────────────────────────

def fetch_macro_context() -> dict:
    """
    CoinGecko /global → BTC dominance + total market-cap 24h change.
    Pure network, no DB. Never raises — returns Nones on failure.
    """
    macro = {'btc_dominance': None, 'mcap_change_24h': None}
    try:
        r = requests.get(COINGECKO_GLOBAL_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        d = r.json()['data']
        macro['btc_dominance']   = d.get('market_cap_percentage', {}).get('btc')
        macro['mcap_change_24h'] = d.get('market_cap_change_percentage_24h_usd')
    except Exception as e:
        print(f"  CoinGecko /global fetch failed: {e}", file=sys.stderr)
    return macro


def get_btc_change_24h(con) -> float | None:
    row = con.execute("""
        SELECT change_24h FROM market_history
        WHERE asset_id = 'BTC' ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def build_macro_block(ctx: dict, macro: dict, btc_change_24h: float | None,
                      heightened_lines: list[str] | None = None) -> str:
    """Render the [MACRO CONTEXT] block prepended to the Qwen prompt."""
    btc = f"{btc_change_24h:+.1f}% 24h"           if btc_change_24h is not None         else "n/a"
    dom = f"{macro.get('btc_dominance'):.1f}%"    if macro.get('btc_dominance')   is not None else "n/a"
    mc  = f"{macro.get('mcap_change_24h'):+.1f}% 24h" if macro.get('mcap_change_24h') is not None else "n/a"
    lines = [
        "[MACRO CONTEXT]",
        f"BTC: {btc} | Dominance: {dom}",
        f"Market cap: {mc}",
        f"Fear & Greed: {ctx['fng_value']} ({ctx['fng_label']})",
    ]
    if heightened_lines:
        lines.append("")
        lines.extend(heightened_lines)
    return "\n".join(lines)


# ── v0.8: bot_state (heightened-mode persistence) ───────────────────────────────

def set_bot_state(con, key: str, value: str) -> None:
    con.execute("""
        INSERT INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                        updated_at = excluded.updated_at
    """, [key, value, datetime.now(timezone.utc)])


def get_bot_state(con, key: str) -> str | None:
    row = con.execute("SELECT value FROM bot_state WHERE key = ?", [key]).fetchone()
    return row[0] if row else None


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def check_heightened_mode(con, asset_id: str,
                          horizon_hours: int = UNLOCK_HORIZON_HOURS) -> tuple[bool, str]:
    """
    (is_heightened, unlock_context_str) for an asset.

    True when an unlock falls within horizon_hours, OR we are still inside the
    post-unlock relief window persisted in bot_state (heightened_<ASSET>_until =
    unlock_at + HEIGHTENED_POST_UNLOCK_HRS). Refreshes that state on each call.
    """
    now        = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=horizon_hours)

    rows = con.execute("""
        SELECT unlock_at, amount_usd, pct_supply, unlock_type
        FROM token_unlocks
        WHERE asset_id = ? AND unlock_at BETWEEN ? AND ?
        ORDER BY unlock_at ASC
        LIMIT 3
    """, [asset_id, now, window_end]).fetchall()

    if rows:
        lines, latest_until = [], None
        for unlock_at, amount_usd, pct_supply, utype in rows:
            ua  = _as_utc(unlock_at)
            hrs = (ua - now).total_seconds() / 3600
            amt = f"${amount_usd / 1e6:.1f}M" if amount_usd else "$?"
            pct = f"{pct_supply:.2f}% supply" if pct_supply is not None else "?% supply"
            lines.append(f"⚠️ {asset_id} UNLOCK in {hrs:.1f}h: {amt} ({pct}) [{utype or 'unlock'}]")
            until = ua + timedelta(hours=HEIGHTENED_POST_UNLOCK_HRS)
            latest_until = until if (latest_until is None or until > latest_until) else latest_until
        if latest_until:
            set_bot_state(con, f"heightened_{asset_id}_until", latest_until.isoformat())
        return True, "\n".join(lines)

    # No upcoming unlock — check the persisted post-unlock relief window.
    until_s = get_bot_state(con, f"heightened_{asset_id}_until")
    if until_s:
        try:
            until = _as_utc(datetime.fromisoformat(until_s))
            if now <= until:
                return True, (f"⚠️ {asset_id} POST-UNLOCK relief window active "
                              f"(through {until:%Y-%m-%d %H:%M} UTC)")
        except ValueError:
            pass
    return False, ""


# ── v0.8: Layer 2 per-asset news scraper ────────────────────────────────────────

def fetch_all_news(assets: list[str]) -> list[dict]:
    """
    Pure-network scrape (no DB held) → list of {asset_id, source, headline,
    published_at}. Each source degrades independently; never raises.
      • CoinDesk RSS  — fetched once, keyword-filtered per asset
      • Reddit        — r/<sub>/new.json (may 403 from datacenter IPs)
      • DeFiLlama TVL — chains only, snapshot headline
    """
    now  = datetime.now(timezone.utc)
    rows = []

    # CoinDesk RSS — one fetch, filtered per asset below.
    coindesk = []
    try:
        fp = feedparser.parse(NEWS_FEED)
        for e in fp.entries:
            title = (getattr(e, 'title', '') or '').strip()
            summ  = (getattr(e, 'summary', '') or '')
            pub   = None
            if getattr(e, 'published_parsed', None):
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if title:
                coindesk.append((title, summ, pub))
    except Exception as e:
        print(f"  CoinDesk RSS failed: {e}", file=sys.stderr)

    # DeFiLlama chains — one fetch, looked up per asset below.
    chains = {}
    try:
        arr = requests.get(CHAINS_URL, timeout=12,
                           headers={'User-Agent': USER_AGENT}).json()
        chains = {(c.get('name') or '').lower(): c for c in arr}
    except Exception as e:
        print(f"  DeFiLlama chains failed: {e}", file=sys.stderr)

    for asset in assets:
        name = ASSET_NAMES.get(asset, asset)
        kws  = {asset.lower(), name.lower()}

        for title, summ, pub in coindesk:
            blob = f"{title} {summ}".lower()
            if any(k in blob for k in kws):
                rows.append({'asset_id': asset, 'source': 'coindesk',
                             'headline': title[:300], 'published_at': pub})

        sub = SUBREDDITS.get(asset)
        if sub:
            try:
                r = requests.get(f"https://www.reddit.com/r/{sub}/new.json?limit={REDDIT_LIMIT}",
                                 headers={'User-Agent': USER_AGENT}, timeout=12)
                if r.status_code == 200:
                    for child in r.json().get('data', {}).get('children', []):
                        d       = child.get('data', {})
                        title   = (d.get('title') or '').strip()
                        created = d.get('created_utc')
                        pub     = (datetime.fromtimestamp(created, tz=timezone.utc)
                                   if created else None)
                        if title:
                            rows.append({'asset_id': asset, 'source': 'reddit',
                                         'headline': title[:300], 'published_at': pub})
                else:
                    print(f"  Reddit r/{sub} HTTP {r.status_code} (skipped)", file=sys.stderr)
            except Exception as e:
                print(f"  Reddit r/{sub} failed: {e}", file=sys.stderr)

        chain_name = DEFILLAMA_CHAINS.get(asset)
        if chain_name:
            c = chains.get(chain_name.lower())
            if c and c.get('tvl') is not None:
                rows.append({'asset_id': asset, 'source': 'defillama',
                             'headline': f"{chain_name} TVL: ${c['tvl'] / 1e6:.1f}M",
                             'published_at': now})

    return rows


def store_asset_news(con, rows: list[dict]) -> int:
    """Insert scraped news (dedup via PK), then prune anything older than 24h."""
    now = datetime.now(timezone.utc)
    for r in rows:
        con.execute("""
            INSERT OR IGNORE INTO asset_news
                (fetched_at, asset_id, source, headline, published_at)
            VALUES (?, ?, ?, ?, ?)
        """, [now, r['asset_id'], r['source'], r['headline'], r.get('published_at')])
    con.execute("DELETE FROM asset_news WHERE fetched_at < ?",
                [now - timedelta(hours=NEWS_LOOKBACK_HOURS)])
    return len(rows)


def get_asset_news(con, asset: str, hours: int, limit: int) -> list[tuple]:
    """Most-recent distinct headlines for an asset within `hours`. → [(headline, source, when)]."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    raw = con.execute("""
        SELECT headline, source, published_at, fetched_at
        FROM asset_news
        WHERE asset_id = ? AND COALESCE(published_at, fetched_at) >= ?
        ORDER BY COALESCE(published_at, fetched_at) DESC
    """, [asset, cutoff]).fetchall()
    seen, out = set(), []
    for headline, source, pub, fetched in raw:
        if headline in seen:
            continue
        seen.add(headline)
        out.append((headline, source, pub or fetched))
        if len(out) >= limit:
            break
    return out


def format_asset_news_block(con, asset: str,
                            hours: int = NEWS_PROMPT_HOURS,
                            limit: int = NEWS_PROMPT_COUNT) -> str:
    items = get_asset_news(con, asset, hours, limit)
    if not items:
        return ""
    now   = datetime.now(timezone.utc)
    lines = [f"[{asset} NEWS - last {hours}h]"]
    for headline, source, when in items:
        if when:
            ago   = (now - _as_utc(when)).total_seconds() / 3600
            ago_s = f"{ago:.0f}h ago" if ago >= 1 else "<1h ago"
        else:
            ago_s = "recent"
        lines.append(f'- "{headline}" ({source}, {ago_s})')
    return "\n".join(lines)


# ── v0.8: token unlock scraper ──────────────────────────────────────────────────

def fetch_unlocks(assets: list[str]) -> list[dict]:
    """
    Best-effort upcoming-unlock fetch from DeFiLlama's free emissions dataset.
    Pure network, no DB. Returns [] (never raises) if the source is unavailable
    or its shape changes. Only future events are kept; pct_supply is derived from
    supplyMetrics.maxSupply when available; amount_usd is left NULL (not in the
    free feed). Rows that can't be parsed are skipped rather than guessed.
    """
    now = datetime.now(timezone.utc)
    out = []
    for asset in assets:
        slug = EMISSION_SLUGS.get(asset)
        if not slug:
            continue
        try:
            r = requests.get(EMISSION_URL.format(slug=slug), timeout=20,
                             headers={'User-Agent': USER_AGENT})
            if r.status_code != 200:
                print(f"  DeFiLlama emissions/{slug} HTTP {r.status_code}", file=sys.stderr)
                continue
            data = r.json()
        except Exception as e:
            print(f"  DeFiLlama emissions/{slug} failed: {e}", file=sys.stderr)
            continue

        meta      = data.get('metadata', {}) if isinstance(data, dict) else {}
        events    = meta.get('events')
        max_supply = (data.get('supplyMetrics', {}) or {}).get('maxSupply')
        if not isinstance(events, list):
            print(f"  DeFiLlama emissions/{slug}: no metadata.events (skipped)", file=sys.stderr)
            continue

        for ev in events:
            try:
                ts_raw = ev.get('timestamp')
                if ts_raw is None:
                    continue
                unlock_at = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                if unlock_at < now:
                    continue   # only future unlocks
                toks = ev.get('noOfTokens')
                if isinstance(toks, list):
                    toks = sum(t for t in toks if isinstance(t, (int, float)))
                toks = float(toks) if isinstance(toks, (int, float)) else None
                pct  = (round(toks / max_supply * 100, 4)
                        if toks is not None and max_supply else None)
                out.append({
                    'asset_id':      asset,
                    'unlock_at':     unlock_at,
                    'amount_tokens': toks,
                    'amount_usd':    None,   # not in the free emissions feed
                    'pct_supply':    pct,
                    'unlock_type':   (ev.get('unlockType') or ev.get('category') or 'unlock')[:64],
                })
            except (ValueError, TypeError):
                continue
    return out


def store_unlocks(con, rows: list[dict]) -> int:
    now = datetime.now(timezone.utc)
    n = 0
    for r in rows:
        ua = r['unlock_at']
        unlock_id = f"{r['asset_id']}_{ua.strftime('%Y%m%d%H%M')}"
        con.execute("""
            INSERT INTO token_unlocks
                (unlock_id, asset_id, unlock_at, amount_tokens, amount_usd,
                 pct_supply, unlock_type, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (unlock_id) DO UPDATE SET
                unlock_at     = excluded.unlock_at,
                amount_tokens = excluded.amount_tokens,
                amount_usd    = excluded.amount_usd,
                pct_supply    = excluded.pct_supply,
                unlock_type   = excluded.unlock_type,
                fetched_at    = excluded.fetched_at
        """, [unlock_id, r['asset_id'], ua, r.get('amount_tokens'),
              r.get('amount_usd'), r.get('pct_supply'),
              r.get('unlock_type'), now])
        n += 1
    return n


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
                        fng_value: int, fng_label: str,
                        etf_flow_m: float | None = None) -> None:
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

    rep_str  = f"{round(rep*100):.0f}% historical win rate" if rep is not None and not (isinstance(rep, float) and math.isnan(rep)) and rep != 0 else "no history yet"
    vel_str  = "deepening ↓" if (vel is not None and vel < 0) else ("recovering ↑" if (vel is not None and vel > 0) else "stable →")
    z_str    = f"{abs(z):.1f} standard deviations below average"
    action_emoji = "🟢" if verdict['action'] == 'accumulate' else "🟡"
    header   = build_discord_header(fng_value, fng_label, etf_flow_m, confidence)

    msg = (
        f"{header}\n"
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
        f"**Thesis:** {verdict.get('thesis','')}\n"
        f"**Risks:** {verdict.get('risks','')}"
    )
    discord_send(msg)


def discord_fiat_signal(verdict: dict, confidence: int, top_row,
                        prices: dict, notional: float,
                        fng_value: int, fng_label: str,
                        etf_flow_m: float | None = None) -> None:
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
    rep_str   = f"{round(rep*100):.0f}% historical win rate" if rep is not None and not (isinstance(rep, float) and math.isnan(rep)) and rep != 0 else "no history yet"
    z_str     = f"{abs(z):.1f} standard deviations below average"
    action_emoji = "🟢" if verdict['action'] == 'accumulate' else "🟡"
    header    = build_discord_header(fng_value, fng_label, etf_flow_m, confidence)

    msg = (
        f"{header}\n"
        f"{action_emoji} **OSMIA — {conviction} USDT BUY signal**\n"
        f"**Buy:** {asset}/USDT @ {px:,.4f} USDT\n"
        f"**Quantity:** {buy_qty} {asset}  for {notional:.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Signal:** {asset} is {z_str}\n"
        f"**Track record:** {rep_str}\n"
        f"**Break-even:** +{fees['breakeven_pct']:.3f}% price increase needed\n"
        f"**Est. fee:** {fees['total_cost']:.4f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
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
    # v0.9: use top-50 watchlist (24h cached in macro_cache)
    watchlist = get_watchlist(con)
    symbols   = [f"{a}/USDT" for a in watchlist]
    print(f"  Watchlist: {len(watchlist)} assets (top-50 CoinGecko)")
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        tickers = exchange.fetch_tickers(symbols)
    except Exception as e:
        print(f"  ccxt error: {e}", file=sys.stderr)
        return 0

    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for symbol in symbols:
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
    """Look up rep scores for a list of pairs using canonical (alpha-sorted) pair_ids.
    Both 'BTC/ETH' and 'ETH/BTC' resolve to the same canonical entry."""
    if not pairs:
        return {}
    # Map each input pair to its canonical form; query pair_reputation table.
    valid = [(p, pair_id(p.split('/')[0], p.split('/')[1]))
             for p in pairs if '/' in p and len(p.split('/')) == 2]
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
    cid = pair_id(sell_asset, buy_asset)
    result = con.execute(f"""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE canonical_pair_id = ?
          AND timestamp >= NOW() - INTERVAL {TRADE_COOLDOWN_MINUTES} MINUTES
          AND status = 'open'
    """, [cid]).fetchone()
    return result[0] > 0


def _pair_at_max_trades(con, sell_asset: str, buy_asset: str) -> bool:
    cid = pair_id(sell_asset, buy_asset)
    result = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE canonical_pair_id = ? AND status = 'open'
    """, [cid]).fetchone()
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
                       current_z_scores: dict[str, float] | None = None,
                       fg_value: int = 50,
                       etf_flow_m: float | None = None) -> int:
    """
    Auto-close open fantasy trades on four conditions:
    1. Mean reversion: P&L >= 1.5x break-even cost
    2. Z crossback: current Z-score > buffett_exit_threshold(fg_value, etf_flow_m)
    3. Expiry: trade has been open > MAX_HOLD_HOURS
    4. News veto: veto keyword found in buy_asset news last 6h
    Exits logged to exit_log table.
    """
    open_trades = con.execute("""
        SELECT trade_id, sell_asset, buy_asset, buy_qty,
               buy_price_usd, notional_usd, total_cost_usd, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    closed     = 0
    now        = datetime.now(timezone.utc)
    z_thresh   = buffett_exit_threshold(fg_value, etf_flow_m)
    sl_thresh  = stop_loss_threshold(fg_value)

    for _, t in open_trades.iterrows():
        pair    = f"{t['sell_asset']}/{t['buy_asset']}"
        cur_px  = prices.get(t['buy_asset'], t['buy_price_usd'])
        cur_val = t['buy_qty'] * cur_px
        pnl_pct = (cur_val - t['notional_usd']) / t['notional_usd'] * 100 if t['notional_usd'] else 0
        be_pct  = t['total_cost_usd'] / t['notional_usd'] * 100 if t['notional_usd'] else 0

        reason     = None
        news_vetod = False
        is_stop    = False
        cpair      = pair_id(t['sell_asset'], t['buy_asset'])   # canonical for z lookup

        # 0. Stop-loss — first check, overrides everything
        if pnl_pct <= sl_thresh:
            reason  = "stop_loss"
            is_stop = True
            discord_stop_loss(pair, pnl_pct, fg_value, sl_thresh)

        # 1. Mean reversion
        elif pnl_pct >= be_pct * 1.5:
            reason = "reverted"

        # 2. Z crossback — dynamic Buffett threshold
        elif current_z_scores and cpair in current_z_scores:
            if current_z_scores[cpair] > z_thresh:
                reason = "z_crossback"

        # 3. Expiry — independent if: must fire even when pair IS in z_scores but
        #    z hasn't crossed threshold (elif would skip it in that case)
        if reason is None and t['timestamp']:
            age_hours = (now - t['timestamp'].to_pydatetime()).total_seconds() / 3600
            if age_hours > MAX_HOLD_HOURS:
                reason = f"expired_{MAX_HOLD_HOURS}h"

        # 4. News veto
        if reason is None:
            if check_news_veto(con, t['buy_asset']):
                reason     = "news_veto"
                news_vetod = True

        if reason:
            pnl_usd, pnl_pct_final = _close_trade(
                con, t['trade_id'], cur_px,
                t['notional_usd'], t['buy_qty'], reason
            )
            if not is_stop:
                discord_trade_closed(t['trade_id'], t['buy_asset'],
                                     pnl_usd, pnl_pct_final, reason)
            exit_id = f"{t['trade_id']}_exit"
            con.execute("""
                INSERT OR IGNORE INTO exit_log
                    (exit_id, timestamp, trade_id, buy_asset, pnl_usd, pnl_pct,
                     reason, fg_value, etf_flow_m, z_threshold, news_veto, stop_loss)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [exit_id, now, t['trade_id'], t['buy_asset'],
                  pnl_usd, pnl_pct_final, reason, fg_value, etf_flow_m,
                  z_thresh, news_vetod, is_stop])
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
        'canonical_pair_id': pair_id(sell_asset, buy_asset),
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
        'canonical_pair_id': pair_id('USDT', buy_asset),
    }


def _insert_trade(con, trade: dict) -> None:
    con.execute("""
        INSERT OR IGNORE INTO fantasy_trades
            (trade_id, timestamp, mode, action, sell_asset, buy_asset,
             sell_qty, buy_qty, sell_price_usd, buy_price_usd, notional_usd,
             fee_usd, spread_usd, total_cost_usd, signal_z, confidence, status,
             canonical_pair_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        trade['trade_id'], trade['timestamp'], trade['mode'], trade['action'],
        trade['sell_asset'], trade['buy_asset'],
        trade['sell_qty'], trade['buy_qty'],
        trade['sell_price_usd'], trade['buy_price_usd'],
        trade['notional_usd'], trade['fee_usd'], trade['spread_usd'],
        trade['total_cost_usd'], trade['signal_z'], trade['confidence'],
        trade['status'],
        trade.get('canonical_pair_id', pair_id(trade['sell_asset'], trade['buy_asset'])),
    ])


# ── v0.9: Feature 2 — macro cache + ETF flows + Discord header ───────────────

def _set_macro_cache(con, key: str, value: str) -> None:
    con.execute("""
        INSERT INTO macro_cache (key, value, fetched_at) VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value, fetched_at = excluded.fetched_at
    """, [key, value, datetime.now(timezone.utc)])


def _get_macro_cache(con, key: str, max_age_seconds: int) -> str | None:
    row = con.execute(
        "SELECT value, fetched_at FROM macro_cache WHERE key = ?", [key]
    ).fetchone()
    if not row:
        return None
    fetched_at = row[1]
    if fetched_at is None:
        return None
    if hasattr(fetched_at, 'tzinfo') and fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
    return row[0] if age < max_age_seconds else None


def fetch_etf_flows() -> float | None:
    """Fetch latest BTC ETF net flow (USD millions) from CoinGlass. Returns None on failure."""
    try:
        r = requests.get(COINGLASS_ETF_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        data = r.json()
        # CoinGlass response structure varies; try common paths
        items = (data.get('data') or [])
        if isinstance(items, list) and items:
            first = items[0]
            # field is typically 'totalNetFlow' or 'netFlow' in millions USD
            for field in ('totalNetFlow', 'netFlow', 'net_flow', 'total_net_flow'):
                v = first.get(field)
                if v is not None:
                    return float(v)
        # Fallback: try top-level
        for field in ('totalNetFlow', 'netFlow'):
            v = data.get(field)
            if v is not None:
                return float(v)
    except Exception as e:
        print(f"  CoinGlass ETF fetch failed: {e}", file=sys.stderr)
    return None


def get_etf_flow_cached(con) -> float | None:
    """Return cached ETF flow or fetch fresh if older than ETF_CACHE_SECONDS."""
    cached = _get_macro_cache(con, 'etf_flow_m', ETF_CACHE_SECONDS)
    if cached is not None:
        try:
            return float(cached)
        except (ValueError, TypeError):
            pass
    flow = fetch_etf_flows()
    if flow is not None:
        _set_macro_cache(con, 'etf_flow_m', str(flow))
    return flow


def _etf_flow_label(flow_m: float | None) -> str:
    if flow_m is None:
        return "N/A"
    if flow_m < -500:
        return f"-${abs(flow_m):.0f}M (Very Bearish)"
    if flow_m < -100:
        return f"-${abs(flow_m):.0f}M (Bearish)"
    if flow_m < 100:
        return f"${flow_m:+.0f}M (Neutral)"
    if flow_m < 500:
        return f"+${flow_m:.0f}M (Bullish)"
    return f"+${flow_m:.0f}M (Very Bullish)"


def _trade_signal_label(confidence: int) -> str:
    if confidence <= 3:
        return f"{confidence}/10 (Caution)"
    if confidence <= 6:
        return f"{confidence}/10 (Neutral)"
    return f"{confidence}/10 (Confident)"


def build_discord_header(fng_value: int, fng_label: str,
                          etf_flow_m: float | None, confidence: int) -> str:
    return (f"📊 Market F&G: {fng_value} ({fng_label}) | "
            f"Inst: ETF {_etf_flow_label(etf_flow_m)} | "
            f"Trade: {_trade_signal_label(confidence)}")


# ── v0.9: Feature 3 — Buffett Inverse F&G Exit ───────────────────────────────

def buffett_exit_threshold(fg_value: int, etf_flow_m: float | None) -> float:
    """Dynamic Z crossback threshold based on F&G + ETF flows (Buffett inverse logic)."""
    if fg_value <= 20:   z_target = 1.5    # Extreme Fear: hold longer, reversion likely
    elif fg_value <= 40: z_target = 0.8    # Fear
    elif fg_value <= 60: z_target = 0.0    # Neutral
    elif fg_value <= 80: z_target = -0.2   # Greed: exit earlier
    else:                z_target = -0.5   # Extreme Greed: exit early, risk elevated
    # ETF flow modifier
    if etf_flow_m is not None:
        if etf_flow_m < -500:   z_target += 0.3   # heavy outflow: hold (contrarian)
        elif etf_flow_m > 500:  z_target -= 0.2   # heavy inflow: exit sooner (too hot)
    return z_target


def check_news_veto(con, asset_id: str) -> bool:
    """True if buy_asset has a veto keyword in asset_news within the last 6h."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    rows   = con.execute("""
        SELECT headline FROM asset_news
        WHERE asset_id = ? AND fetched_at >= ?
    """, [asset_id, cutoff]).fetchall()
    for (headline,) in rows:
        hl_lower = headline.lower()
        if any(kw.lower() in hl_lower for kw in NEWS_VETO_KEYWORDS):
            return True
    return False


# ── v0.9: Feature 4 — Top-50 watchlist ───────────────────────────────────────

def fetch_top50_watchlist() -> list[str]:
    """Fetch top ~50 non-stablecoin symbols tradeable on Binance as USDT pairs.

    Steps:
    1. Pull top-200 by market cap from CoinGecko.
    2. Strip stablecoins and known non-Binance tokens.
    3. Validate remaining symbols against live Binance USDT markets.
    4. Return up to 50, preserving market-cap rank order.
    """
    try:
        # --- Step 1: fetch 200 candidates from CoinGecko (2 pages of 100) ---
        candidates: list[str] = []
        for page in (1, 2):
            r = requests.get(
                COINGECKO_MARKETS_URL,
                params={'vs_currency': 'usd', 'order': 'market_cap_desc',
                        'per_page': 100, 'page': page, 'sparkline': 'false'},
                timeout=15, headers={'User-Agent': USER_AGENT})
            r.raise_for_status()
            for c in r.json():
                if not isinstance(c, dict) or not c.get('symbol'):
                    continue
                sym = c['symbol'].upper()
                if (sym not in STABLECOINS
                        and sym not in NON_BINANCE_TOKENS
                        and sym.isascii() and sym.isalnum()
                        and 2 <= len(sym) <= 10):
                    candidates.append(sym)

        # --- Step 2: load Binance USDT markets once ---
        exchange = ccxt.binance({'enableRateLimit': True})
        markets  = exchange.load_markets()
        binance_usdt = {s.split('/')[0] for s in markets if s.endswith('/USDT')}

        # --- Step 3: filter to validated symbols, preserving rank order ---
        validated: list[str] = []
        seen: set[str] = set()
        for sym in candidates:
            if sym in binance_usdt and sym not in seen:
                seen.add(sym)
                validated.append(sym)
            if len(validated) >= 50:
                break

        print(f"  Watchlist: {len(candidates)} CoinGecko candidates → "
              f"{len(validated)} Binance-validated USDT pairs")
        return validated

    except Exception as e:
        print(f"  Top-50 watchlist fetch failed: {e}", file=sys.stderr)
        return []


def get_watchlist(con) -> list[str]:
    """Return top-50 watchlist from macro_cache (24h TTL) or fetch fresh."""
    cached = _get_macro_cache(con, 'watchlist_top50', WATCHLIST_CACHE_HOURS * 3600)
    if cached:
        try:
            result = json.loads(cached)
            if isinstance(result, list) and result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    watchlist = fetch_top50_watchlist()
    if watchlist:
        _set_macro_cache(con, 'watchlist_top50', json.dumps(watchlist))
        return watchlist
    return list(ASSETS)   # fallback to hardcoded list


# ── v1.0: Feature 1 — Stop-Loss ──────────────────────────────────────────────

def stop_loss_threshold(fg_value: int) -> float:
    """F&G-aware max loss % before forced exit. Tighter in fear, looser in greed."""
    if fg_value <= 20:   return -8.0    # Extreme Fear — market dangerous, cut fast
    elif fg_value <= 40: return -10.0   # Fear
    elif fg_value <= 60: return -12.0   # Neutral
    elif fg_value <= 80: return -15.0   # Greed — reversion more likely, give room
    else:                return -18.0   # Extreme Greed


def discord_stop_loss(pair: str, pnl_pct: float,
                      fg_value: int, threshold: float) -> None:
    msg = (f"🛑 **Stop-loss triggered** — {pair}\n"
           f"Loss: {pnl_pct:+.3f}%  |  F&G: {fg_value}  |  Threshold: {threshold:.1f}%")
    discord_send(msg)


# ── v1.0: Feature 2 — Living Context Re-evaluation ───────────────────────────

def _get_asset_news_text(con, asset: str, hours: int = 6) -> str:
    """Return recent headlines for an asset as a plain text block."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows   = con.execute("""
        SELECT headline FROM asset_news
        WHERE asset_id = ? AND fetched_at >= ?
        ORDER BY fetched_at DESC LIMIT 5
    """, [asset, cutoff]).fetchall()
    if not rows:
        return "  (no recent headlines)"
    return "\n".join(f"  - {r[0]}" for r in rows)


def _get_entry_thesis(con, sell_asset: str, buy_asset: str) -> str:
    """Look up the original LLM thesis for a pair from llm_verdicts."""
    pair    = f"{sell_asset}/{buy_asset}"
    canon   = pair_id(sell_asset, buy_asset)
    row     = con.execute("""
        SELECT thesis FROM llm_verdicts
        WHERE (top_pair = ? OR top_pair = ?) AND action = 'accumulate'
        ORDER BY timestamp DESC LIMIT 1
    """, [pair, canon]).fetchone()
    return row[0] if row and row[0] else "No original thesis found."


def build_position_review_prompt(pair: str, sell_asset: str, buy_asset: str,
                                  entry_z: float, current_z: float | None,
                                  pnl_pct: float, thesis: str,
                                  fg_value: int, btc_dom: float | None,
                                  etf_flow_m: float | None,
                                  base_news: str, quote_news: str) -> str:
    cur_z_str  = f"{current_z:+.3f}" if current_z is not None else "n/a"
    dom_str    = f"{btc_dom:.1f}%" if btc_dom is not None else "n/a"
    flow_str   = _etf_flow_label(etf_flow_m)
    return f"""[OPEN POSITION REVIEW]
Pair: {pair} | Entry Z: {entry_z:+.3f} | Current Z: {cur_z_str} | PnL: {pnl_pct:+.3f}%
Entry thesis: {thesis}

[CURRENT MACRO]
F&G: {fg_value} | BTC Dom: {dom_str} | ETF Flow: {flow_str}

[FRESH NEWS — {sell_asset}]
{base_news}

[FRESH NEWS — {buy_asset}]
{quote_news}

Should this position be held or closed early?
Respond with HOLD or EXIT on the first line, followed by one sentence reason.
Then on a new line: Urgency: N  (where N is 1-10, 10=close immediately)
"""


def _parse_position_review(content: str) -> tuple[str, int, str]:
    """Parse Qwen response: returns (action, urgency, reason)."""
    action    = 'HOLD'
    urgency   = 0
    reason    = content.strip()
    lines     = content.strip().splitlines()
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith('EXIT'):
            action = 'EXIT'
            break
        elif stripped.startswith('HOLD'):
            action = 'HOLD'
            break
    for line in lines:
        m = re.search(r'urgency[:\s]+(\d+)', line, re.IGNORECASE)
        if m:
            urgency = min(10, max(1, int(m.group(1))))
            break
    # Use first non-action line as reason
    for line in lines[1:]:
        s = line.strip()
        if s and not re.match(r'urgency', s, re.IGNORECASE):
            reason = s
            break
    return action, urgency, reason


def evaluate_open_positions(con: duckdb.DuckDBPyConnection,
                             prices: dict[str, float],
                             fng_val: int,
                             macro: dict,
                             etf_flow_m: float | None,
                             current_zs: dict[str, float]) -> int:
    """
    For every open position: refresh news → Qwen review → force close if EXIT urgency>=7.
    Returns count of positions force-closed.
    """
    open_trades = con.execute("""
        SELECT trade_id, sell_asset, buy_asset, buy_qty,
               buy_price_usd, notional_usd, total_cost_usd, signal_z, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    if open_trades.empty:
        return 0

    # Targeted news refresh — only assets in open positions
    held_assets = list(set(
        open_trades['sell_asset'].tolist() + open_trades['buy_asset'].tolist()
    ))
    held_assets = [a for a in held_assets if a != 'USDT']
    if held_assets:
        print(f"  [living-ctx] Refreshing news for {len(held_assets)} held assets …")
        con.close()
        fresh_rows = fetch_all_news(held_assets)
        con2       = get_conn()
        store_asset_news(con2, fresh_rows)
        # Rebind con for caller — Python can't reassign caller's variable,
        # so we do all DB work through con2 for the rest of this function.
    else:
        con2 = con

    btc_dom   = macro.get('btc_dominance')
    force_closed = 0
    now          = datetime.now(timezone.utc)

    for _, t in open_trades.iterrows():
        pair      = f"{t['sell_asset']}/{t['buy_asset']}"
        cur_px    = prices.get(t['buy_asset'], t['buy_price_usd'])
        cur_val   = t['buy_qty'] * cur_px
        pnl_pct   = (cur_val - t['notional_usd']) / t['notional_usd'] * 100 if t['notional_usd'] else 0
        current_z = current_zs.get(pair)
        thesis    = _get_entry_thesis(con2, t['sell_asset'], t['buy_asset'])
        base_news = _get_asset_news_text(con2, t['sell_asset'])
        quote_news= _get_asset_news_text(con2, t['buy_asset'])

        prompt  = build_position_review_prompt(
            pair, t['sell_asset'], t['buy_asset'],
            float(t['signal_z']), current_z, pnl_pct,
            thesis, fng_val, btc_dom, etf_flow_m,
            base_news, quote_news
        )
        print(f"  [living-ctx] Reviewing {pair} (PnL {pnl_pct:+.2f}%) …")
        con2.close()
        verdict_raw = ask_qwen(prompt)
        con2        = get_conn()

        content  = verdict_raw.get('prose', '') or verdict_raw.get('raw_json', '')
        # ask_qwen parses JSON; for free-form responses fall back to raw content
        if not content:
            continue
        action, urgency, reason = _parse_position_review(content)
        print(f"    → {action} (urgency {urgency}/10): {reason[:80]}")

        if action == 'EXIT' and urgency >= 7:
            # Force close
            cur_px2 = prices.get(t['buy_asset'], t['buy_price_usd'])
            pnl_usd, pnl_pct_final = _close_trade(
                con2, t['trade_id'], cur_px2,
                t['notional_usd'], t['buy_qty'], 'news_veto_live'
            )
            msg = (f"📰 **News veto triggered** — {pair}\n"
                   f"{reason}\n"
                   f"PnL at exit: {pnl_pct_final:+.3f}%")
            discord_send(msg)
            exit_id = f"{t['trade_id']}_livevetoexit"
            con2.execute("""
                INSERT OR IGNORE INTO exit_log
                    (exit_id, timestamp, trade_id, buy_asset, pnl_usd, pnl_pct,
                     reason, fg_value, etf_flow_m, z_threshold, news_veto, stop_loss)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [exit_id, now, t['trade_id'], t['buy_asset'],
                  pnl_usd, pnl_pct_final, 'news_veto_live',
                  fng_val, etf_flow_m, None, True, False])
            force_closed += 1

        elif action == 'EXIT' and urgency < 7:
            msg = (f"⚠️ **Living context warning** — {pair}\n"
                   f"Urgency {urgency}/10 — holding for now.\n"
                   f"{reason}")
            discord_send(msg)

    # Return the reopened connection to be usable by the caller
    # Caller must reassign its own `con` from the return value
    return force_closed, con2


# ── v1.0: Feature 3 — Mine Proceeds ──────────────────────────────────────────

MINE_PROCEEDS_GUARD_HOURS = 20


def inject_mine_proceeds(con: duckdb.DuckDBPyConnection,
                          amount_usdt: float) -> None:
    # Guard: check last injection
    last = con.execute("""
        SELECT MAX(injected_at) FROM mine_proceeds
    """).fetchone()[0]
    if last is not None:
        if hasattr(last, 'tzinfo') and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        if age_hours < MINE_PROCEEDS_GUARD_HOURS:
            print(f"[mine] Already injected {age_hours:.1f}h ago — skipping (guard: {MINE_PROCEEDS_GUARD_HOURS}h)")
            return

    # Read current USDT balance
    balances       = load_balances()
    balance_before = balances.get('USDT', 0.0)
    balance_after  = balance_before + amount_usdt
    balances['USDT'] = balance_after

    # Persist balances
    p = Path(BALANCES_FILE)
    p.write_text(json.dumps(balances, indent=2))

    # Record injection
    now = datetime.now(timezone.utc)
    con.execute("""
        INSERT OR IGNORE INTO mine_proceeds (injected_at, amount_usdt, balance_before, balance_after, note)
        VALUES (?, ?, ?, ?, ?)
    """, [now, amount_usdt, balance_before, balance_after, 'nightly_mine_injection'])

    # Snapshot to fantasy_balances
    prices = get_current_prices(con)
    _snapshot_balances(con, balances, prices)

    print(f"[mine] Injected +{amount_usdt:.2f} USDT  |  {balance_before:.4f} → {balance_after:.4f}")
    discord_send(
        f"⛏ **Mine proceeds injected:** +{amount_usdt:.2f} USDT\n"
        f"New USDT balance: {balance_after:.4f}"
    )


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

HEIGHTENED_PROMPT_SUFFIX = """
⚠️ UNLOCK ALERT ACTIVE
{unlock_context}

You are analyzing a pair where one or both legs have imminent token unlocks.
Historical pattern: token unlocks cause pre-unlock selling (front-running) and
post-unlock volatility.

Bias your analysis toward:
- DUMP signals if the unlocked asset is the base (first leg)
- HOLD signals if the unlocked asset is the quote (second leg, being accumulated)
- Flag any ACCUMULATE signal with HIGH CAUTION if unlock is within 24h

Do not fight the unlock. Assess whether this is a pre-unlock dump opportunity
or a post-unlock recovery entry. State which clearly in your thesis.
"""


def build_pair_prompt(con, candidates, quote_warn, base_warn,
                      fng_narrative: str, headlines: list[str],
                      macro_block: str = "",
                      heightened_ctx: dict[str, str] | None = None) -> str:
    heightened_ctx = heightened_ctx or {}
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

    # Layer 2: per-leg news for every asset appearing in the candidate table.
    leg_assets = pd.unique(candidates[['base_asset', 'quote_asset']].values.ravel())
    leg_blocks = [b for b in (format_asset_news_block(con, a) for a in leg_assets) if b]
    leg_news   = ("\n\n" + "\n\n".join(leg_blocks) + "\n") if leg_blocks else ""

    macro = (macro_block + "\n\n") if macro_block else ""

    # Heightened suffix when any candidate leg has an active unlock alert.
    suffix = ""
    hot = [a for a in leg_assets if a in heightened_ctx]
    if hot:
        unlock_context = "\n".join(heightened_ctx[a] for a in hot)
        suffix = HEIGHTENED_PROMPT_SUFFIX.format(unlock_context=unlock_context)

    return f"""{macro}You are a disciplined spot-only crypto trader. Rule-based engine pre-ranked {len(candidates)} candidates.

Market context: {fng_narrative}
{news_block}{leg_news}
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
{suffix}"""


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

    # Do all network I/O up front with the single-writer lock released, so the
    # concurrent ingest/fiat timers aren't blocked while we scrape (the news
    # scrape can touch ~14 assets × 3 sources). Reopen for the quick writes.
    con.close()
    ctx       = fetch_market_context()   # F&G + macro headlines
    macro     = fetch_macro_context()    # CoinGecko dominance + mcap 24h change
    news_rows = fetch_all_news(ASSETS)   # Layer 2 per-asset news (auto --scrape-news)
    con       = get_conn()
    store_market_context(con, ctx)
    store_asset_news(con, news_rows)

    fng_val   = ctx['fng_value']
    fng_lbl   = ctx['fng_label']
    fng_adj, fng_narrative = fng_modifier(fng_val)
    # v0.9: cache fg_value in macro_cache for header and exit logic
    _set_macro_cache(con, 'fg_value', str(fng_val))

    # Heightened mode: any watchlist asset with an imminent unlock relaxes its
    # pairs' dump threshold and arms the unlock-aware prompt suffix.
    relaxed_assets, heightened_ctx, heightened_lines = set(), {}, []
    for asset in UNLOCK_ASSETS:
        is_h, uctx = check_heightened_mode(con, asset)
        if is_h:
            relaxed_assets.add(asset)
            heightened_ctx[asset] = uctx
            heightened_lines.append(uctx)
            print(f"  🔥 Heightened mode active for {asset}")

    btc_change  = get_btc_change_24h(con)
    macro_block = build_macro_block(ctx, macro, btc_change, heightened_lines)
    print(f"  Market: {fng_lbl} (F&G {fng_val}) — {fng_narrative}")
    print("  " + macro_block.replace("\n", "\n  "))

    candidates, quote_warn, base_warn = build_candidates(con, fng_adj, relaxed_assets)

    # Always run close check regardless of whether there are new signals —
    # the early return below was silently skipping expiry/crossback/veto closes.
    prices     = get_current_prices(con)
    etf_flow_m = get_etf_flow_cached(con)
    _set_macro_cache(con, 'fg_value', str(fng_val))
    empty_zs   = {}
    pre_closed = _auto_close_trades(con, prices, empty_zs, fng_val, etf_flow_m)
    if pre_closed:
        print(f"\n  ✅ Auto-closed {pre_closed} trade(s) (expiry/veto) before signal scan")

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'pair', fng_value=fng_val, fng_label=fng_lbl)
        _refresh_pair_reputation(con)
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

    prompt     = build_pair_prompt(con, candidates, quote_warn, base_warn,
                                   fng_narrative, ctx['headlines'],
                                   macro_block, heightened_ctx)
    # Release the single-writer lock during the multi-minute LLM call so the
    # concurrent ingest/fiat timers aren't blocked (and don't crash on lock).
    con.close()
    verdict    = ask_qwen(prompt)
    con        = get_conn()
    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair, fng_val)

    # Build current Z-score map for crossback detection (prices/etf_flow_m already fetched above)
    current_zs = {r['pair']: r['z_score'] for _, r in candidates.iterrows()}

    closed = _auto_close_trades(con, prices, current_zs, fng_val, etf_flow_m)
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
        discord_pair_signal(verdict, confidence, top_row, prices, notional, fng_val, fng_lbl,
                            etf_flow_m)

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
    _refresh_pair_reputation(con)

    # v1.0: Feature 2 — living context re-evaluation of all open positions
    if ping_qwen():
        try:
            force_closed, con = evaluate_open_positions(
                con, prices, fng_val, macro, etf_flow_m, current_zs
            )
            if force_closed:
                print(f"\n  📰 Living context force-closed {force_closed} position(s)")
        except Exception as e:
            print(f"  [living-ctx] Error during re-evaluation: {e}", file=sys.stderr)
    else:
        print("  [living-ctx] Skipped — Qwen unreachable")


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

    ctx        = fetch_market_context()
    store_market_context(con, ctx)
    fng_val    = ctx['fng_value']
    fng_lbl    = ctx['fng_label']
    etf_flow_m = get_etf_flow_cached(con)
    _set_macro_cache(con, 'fg_value', str(fng_val))
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
    # Qwen returns the asset inconsistently as "AVAX" or "AVAX/USDT" (sometimes even
    # "AVAX/USDT/USDT"). Normalize to the bare base symbol up front and use it
    # everywhere downstream — otherwise the raw "AVAX/USDT" fails the membership test
    # against candidates['asset_id'] (bare symbols), leaving top_row=None and silently
    # skipping both the Discord post and the fantasy trade.
    top_asset = (verdict.get('top_asset', '') or '').split('/')[0].strip().upper()
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
        discord_fiat_signal(verdict, confidence, top_row, prices, notional, fng_val, fng_lbl,
                            etf_flow_m)

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
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.9")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',         action='store_true')
    grp.add_argument('--analyze',        action='store_true')
    grp.add_argument('--fiat',           action='store_true')
    grp.add_argument('--backfill',       action='store_true')
    grp.add_argument('--outcomes',       action='store_true')
    grp.add_argument('--portfolio',      action='store_true')
    grp.add_argument('--daily-pnl',      action='store_true')
    grp.add_argument('--init-db',        action='store_true')
    grp.add_argument('--status',         action='store_true')
    grp.add_argument('--scrape-news',    action='store_true')
    grp.add_argument('--scrape-unlocks', action='store_true')
    grp.add_argument('--mine-proceeds',  action='store_true')
    parser.add_argument('--size',        type=float, default=None)
    parser.add_argument('--mine-value',  type=float, default=10.0)
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
    elif args.scrape_news:
        # Release the lock during the multi-source network scrape, then reopen.
        print(f"[{datetime.now():%H:%M:%S}] Scraping per-asset news …")
        con.close()
        rows = fetch_all_news(ASSETS)
        con  = get_conn()
        print(f"  Stored {store_asset_news(con, rows)} news rows for {len(ASSETS)} assets.")
    elif args.mine_proceeds: inject_mine_proceeds(con, args.mine_value)
    elif args.scrape_unlocks:
        print(f"[{datetime.now():%H:%M:%S}] Scraping token unlocks …")
        con.close()
        rows = fetch_unlocks(UNLOCK_ASSETS)
        con  = get_conn()
        store_unlocks(con, rows)
        total = con.execute("SELECT COUNT(*) FROM token_unlocks WHERE unlock_at > now()").fetchone()[0]
        print(f"  Processed {len(rows)} unlock events; {total} upcoming rows in DB for {UNLOCK_ASSETS}.")
    else:
        ingest(con)
        analyze_pairs(con, args.size)

    con.close()


if __name__ == "__main__":
    main()
