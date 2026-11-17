#!/usr/bin/env python3
"""
osmia_trader.py  —  v0.3
Unified ingest + analyze script for the Osmia LLM trading monitor.

Usage:
    python osmia_trader.py              # ingest + analyze
    python osmia_trader.py --ingest     # ingest only
    python osmia_trader.py --analyze    # analyze only
    python osmia_trader.py --init-db    # (re)create DB schema and exit
    python osmia_trader.py --status     # quick DB health summary

Changes from v0.2:
    - Schema aligned to actual DB (change_24h, high_low_pct added; dead columns removed)
    - DOGE-as-quote detection: collapses DOGE-quoted pairs into a single meta-signal
    - Pair persistence score: consecutive windows in buy zone fed to LLM
    - Z-score velocity: delta vs previous analysis window
    - Real spread_pct fed to LLM for actual liquidity assessment
    - Qwen prompt forces relative ranking of candidates, not absolute scoring
    - prose column now persisted to llm_verdicts
    - Rule-based pre-filter: LLM only sees top 3 candidates after scoring
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

# Stats thresholds
MIN_PAIR_SAMPLES  = 20
VOLUME_WINDOW     = 50
ZSCORE_BUY_THRESH = -2.0
TOP_N_RAW         = 10   # raw pairs pulled from DB before pre-filter
TOP_N_LLM         = 3    # candidates sent to LLM after rule-based scoring

# Assets that frequently appear as quote in spurious pairs
# When one of these dominates the hotlist it's flagged as a meta-signal
MEME_ASSETS = {'DOGE', 'XLM', 'TRX', 'XRP'}

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
            'vwap_ratio':   round(last / vwap,           6) if vwap else None,
            'spread_pct':   round((ask-bid) / mid * 100, 6) if mid  else None,
            'high_low_pct': round((high-low) / low * 100, 6) if low else None,
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

# ── Signal engine ─────────────────────────────────────────────────────────────

def get_raw_anomalies(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Cross-pair Z-score on price ratios.
    Returns TOP_N_RAW pairs with supporting context columns.
    """
    query = f"""
    WITH bucketed AS (
        SELECT DATE_TRUNC('minute', timestamp) AS ts_min,
               asset_id, price_usd, vwap_ratio, volume_z, spread_pct
        FROM market_history
    ),
    pairs AS (
        SELECT a.ts_min,
               a.asset_id || '/' || b.asset_id                    AS pair,
               a.asset_id                                          AS base_asset,
               b.asset_id                                          AS quote_asset,
               a.price_usd / NULLIF(b.price_usd, 0)               AS ratio,
               a.vwap_ratio                                        AS base_vwap_ratio,
               a.volume_z                                          AS base_volume_z,
               a.spread_pct                                        AS base_spread_pct
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
            ROUND(l.ratio, 6)                                            AS ratio,
            ROUND(s.mean_ratio, 6)                                       AS mean_ratio,
            ROUND((l.ratio - s.mean_ratio) / NULLIF(s.std_ratio,0), 3)  AS z_score,
            ROUND(l.base_vwap_ratio, 4)                                  AS vwap_ratio,
            ROUND(l.base_volume_z,   3)                                  AS volume_z,
            ROUND(l.base_spread_pct, 4)                                  AS spread_pct,
            s.sample_count
    FROM latest l
    JOIN stats  s ON l.pair = s.pair
    WHERE (l.ratio - s.mean_ratio) / NULLIF(s.std_ratio,0) < {ZSCORE_BUY_THRESH}
    ORDER BY z_score ASC
    LIMIT {TOP_N_RAW}
    """
    return con.execute(query).df()


def get_pair_persistence(con: duckdb.DuckDBPyConnection,
                         pairs: list[str]) -> dict[str, int]:
    """
    For each pair, count how many of the last 24 llm_verdicts windows
    it appeared as top_pair with action=accumulate.
    Proxy for persistence — not perfect but uses existing data.
    """
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
    """
    Fetch the z_score from the previous llm_verdicts entry for each pair
    to compute velocity (is the signal deepening or recovering?).
    """
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
    """
    If a single quote asset appears in 3+ of the flagged pairs,
    the real signal is that asset being overbought — return it.
    """
    counts = df['quote_asset'].value_counts()
    dominant = counts[counts >= 3]
    if dominant.empty:
        return None
    return dominant.index[0]


def rule_based_score(row: pd.Series,
                     persistence: dict[str, int],
                     prev_z: dict[str, float]) -> float:
    """
    Composite score for pre-filtering before LLM.
    Lower = better (aligns with Z-score direction).
    Components:
      - Z-score magnitude          (weight 1.0)
      - Persistence bonus          (−0.1 per hit in last 24h, max −0.5)
      - Velocity bonus             (−0.2 if deepening, +0.2 if recovering)
      - Volume confirmation        (−0.15 if volume_z > 1.5)
      - VWAP confirmation          (−0.1 if vwap_ratio < 0.99)
      - Liquidity penalty          (+0.3 if spread_pct > 0.1)
      - Meme quote penalty         (+0.5 if quote_asset in MEME_ASSETS)
    """
    score = row['z_score']  # already negative

    # Persistence
    hits = persistence.get(row['pair'], 0)
    score -= min(hits * 0.1, 0.5)

    # Velocity
    prev = prev_z.get(row['pair'])
    if prev is not None:
        if row['z_score'] < prev:   # deepening
            score -= 0.2
        else:                        # recovering
            score += 0.2

    # Volume confirmation
    if row['volume_z'] is not None and row['volume_z'] > 1.5:
        score -= 0.15

    # VWAP confirmation
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99:
        score -= 0.1

    # Liquidity penalty
    if row['spread_pct'] is not None and row['spread_pct'] > 0.1:
        score += 0.3

    # Meme quote penalty
    if row['quote_asset'] in MEME_ASSETS:
        score += 0.5

    return round(score, 4)


def build_candidates(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, str | None]:
    """
    Returns (top_candidates_df, quote_dominance_warning).
    top_candidates_df has at most TOP_N_LLM rows, pre-scored and ranked.
    """
    raw = get_raw_anomalies(con)
    if raw.empty:
        return raw, None

    pairs = raw['pair'].tolist()
    persistence = get_pair_persistence(con, pairs)
    prev_z      = get_prev_z_scores(con, pairs)

    raw['persistence_24h'] = raw['pair'].map(lambda p: persistence.get(p, 0))
    raw['z_velocity']      = raw['pair'].map(
        lambda p: round(raw.loc[raw['pair']==p,'z_score'].iloc[0] - prev_z[p], 3)
        if p in prev_z else None
    )
    raw['rule_score'] = raw.apply(
        lambda r: rule_based_score(r, persistence, prev_z), axis=1
    )

    quote_warn = detect_quote_dominance(raw)
    top = raw.sort_values('rule_score').head(TOP_N_LLM).reset_index(drop=True)
    return top, quote_warn

# ── LLM ───────────────────────────────────────────────────────────────────────

def build_prompt(candidates: pd.DataFrame, quote_warn: str | None) -> str:
    rows = []
    for i, r in candidates.iterrows():
        vel_str  = f"{r['z_velocity']:+.3f}" if r['z_velocity'] is not None else "n/a"
        spread   = f"{r['spread_pct']:.4f}%" if r['spread_pct'] is not None else "n/a"
        vwap     = f"{r['vwap_ratio']:.4f}"  if r['vwap_ratio'] is not None else "n/a"
        vol_z    = f"{r['volume_z']:+.3f}"   if r['volume_z']   is not None else "n/a"
        rows.append(
            f"  [{i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel_str} | "
            f"persist={r['persistence_24h']}h | vwap={vwap} | vol_z={vol_z} | spread={spread}"
        )
    table = "\n".join(rows)

    quote_block = ""
    if quote_warn:
        quote_block = (
            f"\n⚠ QUOTE DOMINANCE DETECTED: {quote_warn} appears as the quote asset in "
            f"multiple pairs. The underlying signal may be {quote_warn} overbought rather "
            f"than independent buy signals on the base assets. Factor this into your analysis.\n"
        )

    return f"""You are a disciplined spot-only crypto trader. You are given {len(candidates)} pre-filtered candidates ranked by a rule-based engine. Your job is to pick the BEST ONE or pass.

Candidates (ranked best-first by rule score):
{table}
{quote_block}
Column definitions:
- z: price-ratio Z-score vs {MIN_PAIR_SAMPLES}+ sample history. More negative = more oversold.
- vel: Z-score delta vs previous analysis. Negative = signal deepening (good). Positive = recovering (caution).
- persist: times this pair appeared as top signal in last 24h.
- vwap: last/VWAP. <1.0 = trading below volume-weighted average.
- vol_z: volume Z-score vs {VOLUME_WINDOW}-sample window. >1.5 = unusual interest.
- spread: bid-ask spread %. >0.1% = liquidity concern.

Rules:
- Spot accumulation only. No shorts, no leverage, no margin.
- If spread > 0.15% on your top pick, flag it explicitly in risks.
- If quote dominance warning is present, your thesis must address whether this changes the signal.
- RANK the candidates 1 to {len(candidates)} in your JSON. Do not give them all the same confidence.
- If no candidate clears your bar, set action=pass and confidence<=3.

Respond with a single JSON object on one line, then a blank line, then 2-3 sentences of prose.

JSON schema (strict):
{{"confidence": <int 1-10>, "action": <"accumulate"|"watch"|"pass">, "top_pair": "<PAIR>", "rank": [<pair1>, <pair2>, ...], "thesis": "<max 120 chars>", "risks": "<max 120 chars>"}}
"""


def ask_qwen(prompt: str) -> dict:
    fallback = {
        'confidence': 0, 'action': 'pass', 'top_pair': '',
        'rank': [], 'thesis': '', 'risks': '',
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

    lines       = content.strip().splitlines()
    json_line   = ""
    prose_lines = []
    json_found  = False
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
            for k in ('confidence', 'action', 'top_pair', 'rank', 'thesis', 'risks'):
                if k in parsed:
                    result[k] = parsed[k]
        except json.JSONDecodeError as e:
            result['prose'] = f"[JSON parse error: {e}]\n" + result['prose']

    return result


def log_verdict(con: duckdb.DuckDBPyConnection,
                verdict: dict,
                candidates: pd.DataFrame) -> None:
    ts      = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    top_row = None
    if verdict.get('top_pair') and not candidates.empty:
        match = candidates[candidates['pair'] == verdict['top_pair']]
        if not match.empty:
            top_row = match.iloc[0]

    z_score = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'

    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        ts, top_pair, z_score,
        verdict.get('confidence', 0),
        verdict.get('action', 'pass'),
        verdict.get('thesis', ''),
        verdict.get('risks', ''),
        verdict.get('prose', ''),
        verdict.get('raw_json', ''),
    ])

# ── Analyze ───────────────────────────────────────────────────────────────────

def analyze(con: duckdb.DuckDBPyConnection) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running analysis …")

    candidates, quote_warn = build_candidates(con)

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        log_verdict(con, {
            'confidence': 0, 'action': 'pass', 'top_pair': 'none',
            'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''
        }, candidates)
        return

    if quote_warn:
        print(f"\n  ⚠ Quote dominance: {quote_warn} is the quote in multiple pairs "
              f"— may be {quote_warn} overbought, not independent signals")

    print("\n── Candidates (rule-scored) ──")
    display_cols = ['pair', 'z_score', 'z_velocity', 'persistence_24h',
                    'vwap_ratio', 'volume_z', 'spread_pct', 'rule_score']
    print(candidates[display_cols].to_string(index=False))

    prompt  = build_prompt(candidates, quote_warn)
    verdict = ask_qwen(prompt)
    log_verdict(con, verdict, candidates)

    print("\n── Qwen Verdict ──")
    print(f"  Action     : {verdict['action'].upper()}  (confidence {verdict['confidence']}/10)")
    if verdict.get('top_pair'):
        print(f"  Top pair   : {verdict['top_pair']}")
    if verdict.get('rank') and len(verdict['rank']) > 1:
        print(f"  Ranking    : {' > '.join(verdict['rank'])}")
    if verdict.get('thesis'):
        print(f"  Thesis     : {verdict['thesis']}")
    if verdict.get('risks'):
        print(f"  Risks      : {verdict['risks']}")
    if verdict.get('prose'):
        print(f"\n{verdict['prose']}")

# ── Status ────────────────────────────────────────────────────────────────────

def status(con: duckdb.DuckDBPyConnection) -> None:
    print("── DB Status ──")
    rows = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    verdicts = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    latest = con.execute(
        "SELECT MAX(timestamp) FROM market_history"
    ).fetchone()[0]
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
        SELECT timestamp, top_pair, z_score, confidence_score, action
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals (all time) ──")
    print(con.execute("""
        SELECT top_pair,
               COUNT(*)                        AS hits,
               ROUND(AVG(confidence_score),1)  AS avg_conf,
               ROUND(AVG(z_score),3)           AS avg_z
        FROM llm_verdicts
        WHERE action = 'accumulate'
        GROUP BY top_pair
        ORDER BY hits DESC
        LIMIT 10
    """).df().to_string(index=False))

# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Osmia LLM Trader v0.3")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',  action='store_true', help='Ingest only')
    grp.add_argument('--analyze', action='store_true', help='Analyze only')
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
        analyze(con)
    else:
        ingest(con)
        analyze(con)

    con.close()

if __name__ == "__main__":
    main()
