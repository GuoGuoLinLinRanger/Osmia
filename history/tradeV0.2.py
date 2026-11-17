#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.2
Unified ingest + analyze script for the Osmia LLM trading monitor.

Usage:
    python osmia_trader.py              # ingest + analyze
    python osmia_trader.py --ingest     # ingest only
    python osmia_trader.py --analyze    # analyze only (uses existing data)
    python osmia_trader.py --init-db    # (re)create DB schema and exit
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
QWEN_MODEL = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

ASSETS  = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP',
           'TRX', 'BNB', 'AVAX', 'LTC', 'XLM', 'APT', 'DOGE', 'SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]

# Minimum samples before a pair's Z-score is considered statistically valid
MIN_PAIR_SAMPLES   = 10
# Rolling window for per-asset volume Z-score (number of past rows per asset)
VOLUME_WINDOW      = 50
# Z-score threshold for buy-zone flagging
ZSCORE_BUY_THRESH  = -2.0
# Top N pairs sent to LLM
TOP_N_PAIRS        = 5

# ── Schema ────────────────────────────────────────────────────────────────────

DDL_MARKET_HISTORY = """
CREATE TABLE IF NOT EXISTS market_history (
    timestamp      TIMESTAMPTZ NOT NULL,
    asset_id       VARCHAR     NOT NULL,
    price_usd      DOUBLE,
    volume_24h     DOUBLE,
    change_24h     DOUBLE,
    market_cap     DOUBLE,
    vwap_ratio     DOUBLE,   -- last / vwap; >1 overbought vs volume, <1 underbought
    volume_z       DOUBLE,   -- volume_24h Z-score vs rolling VOLUME_WINDOW samples
    spread_pct     DOUBLE,   -- (ask - bid) / mid * 100; liquidity proxy
    high_low_pct   DOUBLE,   -- (high - low) / low * 100; intraday volatility
    PRIMARY KEY (timestamp, asset_id)
);
"""

DDL_LLM_VERDICTS = """
CREATE TABLE IF NOT EXISTS llm_verdicts (
    timestamp        TIMESTAMPTZ NOT NULL,
    top_pair         VARCHAR,
    z_score          DOUBLE,
    confidence_score INTEGER,   -- 1-10 from LLM JSON
    action           VARCHAR,   -- accumulate | watch | pass
    thesis           TEXT,
    risks            TEXT,
    raw_json         TEXT,
    PRIMARY KEY (timestamp, top_pair)
);
"""

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist. Safe to call every run."""
    con.execute(DDL_MARKET_HISTORY)
    con.execute(DDL_LLM_VERDICTS)


# ── Ingest ────────────────────────────────────────────────────────────────────

def _volume_z_scores(con: duckdb.DuckDBPyConnection, new_df: pd.DataFrame) -> dict[str, float]:
    """
    For each asset, compute the Z-score of the current volume_24h
    against the last VOLUME_WINDOW rows already in market_history.
    Returns {asset_id: z_score}.  Falls back to 0.0 when insufficient history.
    """
    results = {}
    for _, row in new_df.iterrows():
        asset = row['asset_id']
        cur_vol = row['volume_24h']
        hist = con.execute(
            """
            SELECT volume_24h FROM market_history
            WHERE asset_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
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

    # Bucket timestamp to the minute to act as natural dedup key
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    rows = []
    for symbol in SYMBOLS:
        t = tickers.get(symbol)
        if not t:
            continue
        asset = symbol.split('/')[0]
        last  = t.get('last') or 0.0
        ask   = t.get('ask')  or 0.0
        bid   = t.get('bid')  or 0.0
        mid   = (ask + bid) / 2 if (ask and bid) else last
        vwap  = t.get('vwap') or 0.0
        high  = t.get('high') or 0.0
        low   = t.get('low')  or 0.0

        rows.append({
            'timestamp':    ts,
            'asset_id':     asset,
            'price_usd':    last,
            'volume_24h':   t.get('quoteVolume') or 0.0,
            'change_24h':   t.get('percentage')  or 0.0,
            'market_cap':   0.0,
            'vwap_ratio':   round(last / vwap, 6)         if vwap  else None,
            'spread_pct':   round((ask - bid) / mid * 100, 6) if mid else None,
            'high_low_pct': round((high - low) / low * 100, 6) if low else None,
            'volume_z':     None,  # filled below after history lookup
        })

    if not rows:
        print("  No ticker data returned.", file=sys.stderr)
        return 0

    df = pd.DataFrame(rows)

    # Compute volume Z-scores against historical data
    vzs = _volume_z_scores(con, df)
    df['volume_z'] = df['asset_id'].map(vzs)

    # INSERT OR IGNORE via explicit column list — safe against schema drift
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


# ── Analysis ──────────────────────────────────────────────────────────────────

def get_anomaly_hotlist(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Cross-pair Z-score on price ratios, joined at minute-bucketed timestamps.
    Returns top N pairs by Z-score ascending (most oversold first).
    Also pulls latest vwap_ratio and volume_z for the base asset as context.
    """
    query = f"""
    WITH bucketed AS (
        SELECT DATE_TRUNC('minute', timestamp) AS ts_min,
               asset_id, price_usd, vwap_ratio, volume_z
        FROM market_history
    ),
    pairs AS (
        SELECT a.ts_min,
               a.asset_id || '/' || b.asset_id AS pair,
               a.price_usd / NULLIF(b.price_usd, 0)  AS ratio,
               a.vwap_ratio                           AS base_vwap_ratio,
               a.volume_z                             AS base_volume_z
        FROM bucketed a
        JOIN bucketed b
          ON a.ts_min = b.ts_min AND a.asset_id != b.asset_id
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
        SELECT pair, ratio, base_vwap_ratio, base_volume_z
        FROM pairs
        WHERE ts_min = (
            SELECT MAX(DATE_TRUNC('minute', timestamp)) FROM market_history
        )
    )
    SELECT  l.pair,
            ROUND(l.ratio, 6)                                          AS ratio,
            ROUND(s.mean_ratio, 6)                                     AS mean_ratio,
            ROUND((l.ratio - s.mean_ratio) / NULLIF(s.std_ratio, 0), 3) AS z_score,
            ROUND(l.base_vwap_ratio, 4)                                AS vwap_ratio,
            ROUND(l.base_volume_z, 3)                                  AS volume_z
    FROM latest l
    JOIN stats  s ON l.pair = s.pair
    WHERE z_score < {ZSCORE_BUY_THRESH}
    ORDER BY z_score ASC
    LIMIT {TOP_N_PAIRS}
    """
    return con.execute(query).df()


def build_prompt(anomaly_df: pd.DataFrame) -> str:
    rows = []
    for _, r in anomaly_df.iterrows():
        vol_flag = ""
        if r['volume_z'] is not None:
            if   r['volume_z'] >  2.0: vol_flag = " [VOLUME SPIKE]"
            elif r['volume_z'] < -2.0: vol_flag = " [VOLUME DRY]"
        vwap_flag = ""
        if r['vwap_ratio'] is not None:
            if   r['vwap_ratio'] < 0.98: vwap_flag = " [BELOW VWAP]"
            elif r['vwap_ratio'] > 1.02: vwap_flag = " [ABOVE VWAP]"
        rows.append(
            f"  {r['pair']:>12s}  z={r['z_score']:+.3f}  "
            f"vwap_ratio={r['vwap_ratio'] if r['vwap_ratio'] else 'n/a'}"
            f"{vwap_flag}{vol_flag}"
        )
    table = "\n".join(rows)

    return f"""You are a growth-oriented spot trader monitoring a basket of crypto assets.

Statistical anomalies detected (price-ratio Z-score < {ZSCORE_BUY_THRESH}, cross-pair):
{table}

Additional context per row:
- vwap_ratio: last_price / VWAP. <1 means trading below volume-weighted average (undervalued vs activity).
- volume_z: 24h volume Z-score vs {VOLUME_WINDOW}-sample rolling window. Spike = unusual interest.

Task:
1. Identify the single best spot accumulation opportunity.
2. Flag any pair where low liquidity (wide spread) or dry volume undercuts the thesis.
3. Respond ONLY with a JSON object on one line, then a blank line, then a prose summary.

JSON schema (strict, no extra keys):
{{"confidence": <int 1-10>, "action": <"accumulate"|"watch"|"pass">, "top_pair": "<PAIR>", "thesis": "<one sentence>", "risks": "<one sentence>"}}

Rules:
- No shorts, no margin, no leverage. Spot only.
- If no pair clears your bar, set action="pass" and confidence<=3.
- thesis and risks must each be under 120 characters.
"""


def ask_qwen(prompt: str) -> dict:
    """
    Returns dict with keys: confidence, action, top_pair, thesis, risks, prose, raw_json.
    Never raises — falls back gracefully.
    """
    fallback = {
        'confidence': 0, 'action': 'pass',
        'top_pair': '', 'thesis': '', 'risks': '',
        'prose': 'LLM unavailable.', 'raw_json': ''
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(QWEN_URL, json=payload, timeout=90)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
    except Exception as e:
        fallback['prose'] = f"LLM request failed: {e}"
        return fallback

    # Split JSON line from prose
    lines = content.strip().splitlines()
    json_line = ""
    prose_lines = []
    json_found = False
    for line in lines:
        stripped = line.strip()
        if not json_found and stripped.startswith('{'):
            json_line = stripped
            json_found = True
        else:
            prose_lines.append(line)

    result = dict(fallback)
    result['raw_json'] = json_line
    result['prose']    = "\n".join(prose_lines).strip()

    if json_line:
        try:
            parsed = json.loads(json_line)
            result.update({k: parsed[k] for k in
                           ('confidence', 'action', 'top_pair', 'thesis', 'risks')
                           if k in parsed})
        except json.JSONDecodeError as e:
            result['prose'] = f"[JSON parse error: {e}]\n" + result['prose']

    return result


def log_verdict(con: duckdb.DuckDBPyConnection,
                verdict: dict, top_row: pd.Series | None) -> None:
    ts      = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score = float(top_row['z_score']) if top_row is not None else None
    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, top_pair, z_score, confidence_score,
             action, thesis, risks, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ts,
        verdict.get('top_pair', ''),
        z_score,
        verdict.get('confidence', 0),
        verdict.get('action', 'pass'),
        verdict.get('thesis', ''),
        verdict.get('risks', ''),
        verdict.get('raw_json', ''),
    ])


def analyze(con: duckdb.DuckDBPyConnection) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running analysis …")
    anomalies = get_anomaly_hotlist(con)

    print("\n── Top Buy-Zone Pairs (Z < {}) ──".format(ZSCORE_BUY_THRESH))
    if anomalies.empty:
        print("  None — market within normal bounds.")
        verdict = {
            'confidence': 0, 'action': 'pass',
            'top_pair': '', 'thesis': 'Market stable.',
            'risks': '', 'prose': 'No anomalies detected.', 'raw_json': ''
        }
        log_verdict(con, verdict, None)
        return

    print(anomalies[['pair', 'z_score', 'vwap_ratio', 'volume_z']].to_string(index=False))

    prompt  = build_prompt(anomalies)
    verdict = ask_qwen(prompt)

    # Match top_pair back to anomaly row for z_score logging
    top_row = None
    if verdict.get('top_pair'):
        match = anomalies[anomalies['pair'] == verdict['top_pair']]
        if not match.empty:
            top_row = match.iloc[0]

    log_verdict(con, verdict, top_row)

    print("\n── Qwen Verdict ──")
    print(f"  Action     : {verdict['action'].upper()}  (confidence {verdict['confidence']}/10)")
    if verdict.get('top_pair'):
        print(f"  Top pair   : {verdict['top_pair']}")
    if verdict.get('thesis'):
        print(f"  Thesis     : {verdict['thesis']}")
    if verdict.get('risks'):
        print(f"  Risks      : {verdict['risks']}")
    if verdict.get('prose'):
        print(f"\n{verdict['prose']}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.2")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',   action='store_true', help='Ingest only')
    grp.add_argument('--analyze',  action='store_true', help='Analyze only')
    grp.add_argument('--init-db',  action='store_true', help='(Re)create DB schema and exit')
    args = parser.parse_args()

    con = get_conn()
    init_db(con)   # idempotent — no-op if tables already exist

    if args.init_db:
        print("DB schema initialized.")
        con.close()
        return

    if args.ingest:
        ingest(con)
    elif args.analyze:
        analyze(con)
    else:
        ingest(con)
        analyze(con)

    con.close()


if __name__ == "__main__":
    main()
