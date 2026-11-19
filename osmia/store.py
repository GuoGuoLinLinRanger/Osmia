"""DuckDB schema, connection with lock retry, migrations, small key/value caches."""
import random
import time
from datetime import datetime, timezone

import duckdb

from . import config
from .config import REP_MIN_SAMPLES
from .core import reputation_score


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
    trade_id           VARCHAR     NOT NULL PRIMARY KEY,
    timestamp          TIMESTAMPTZ NOT NULL,
    mode               VARCHAR     NOT NULL,
    action             VARCHAR     NOT NULL,
    sell_asset         VARCHAR,
    buy_asset          VARCHAR,
    sell_qty           DOUBLE,
    buy_qty            DOUBLE,
    sell_price_usd     DOUBLE,
    buy_price_usd      DOUBLE,
    notional_usd       DOUBLE,
    fee_usd            DOUBLE,
    spread_usd         DOUBLE,
    total_cost_usd     DOUBLE,
    signal_z           DOUBLE,
    confidence         INTEGER,
    status             VARCHAR DEFAULT 'open',
    close_ts           TIMESTAMPTZ,
    close_reason       VARCHAR,
    close_sell_px      DOUBLE,
    close_buy_px       DOUBLE,
    pnl_usd            DOUBLE,
    pnl_pct            DOUBLE,
    position_multiplier FLOAT DEFAULT 1.0
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

# v1.4: Feature 2 — unMineable payment dedup log
DDL_MINE_INJECTIONS = """
CREATE TABLE IF NOT EXISTS mine_injections (
    injection_id   VARCHAR PRIMARY KEY,   -- coin + payment_ts yyyymmddHHMMSS
    coin           VARCHAR NOT NULL,
    amount         DOUBLE  NOT NULL,
    payment_ts     TIMESTAMPTZ NOT NULL,
    detected_at    TIMESTAMPTZ NOT NULL,
    injected_usdt  DOUBLE,
    discord_sent   BOOLEAN DEFAULT FALSE
);
"""

# v1.4: Feature 4 — 6h report feedback loop
DDL_REPORT_INSIGHTS_SEQ = "CREATE SEQUENCE IF NOT EXISTS report_insights_id_seq START 1;"

DDL_REPORT_INSIGHTS = """
CREATE TABLE IF NOT EXISTS report_insights (
    id           BIGINT PRIMARY KEY DEFAULT nextval('report_insights_id_seq'),
    generated_at TIMESTAMPTZ NOT NULL,
    pair         VARCHAR     NOT NULL,
    signal       VARCHAR     NOT NULL,   -- 'boost' | 'warn' | 'suppress'
    magnitude    FLOAT       NOT NULL,   -- multiplier delta e.g. +0.1 or -0.2
    reason       VARCHAR,
    expires_at   TIMESTAMPTZ NOT NULL
);
"""

# v1.1: F&G history for Grafana divergence chart
DDL_FG_HISTORY_SEQ = "CREATE SEQUENCE IF NOT EXISTS fg_history_id_seq START 1;"

DDL_FG_HISTORY = """
CREATE TABLE IF NOT EXISTS fg_history (
    id                   BIGINT PRIMARY KEY DEFAULT nextval('fg_history_id_seq'),
    recorded_at          TIMESTAMP DEFAULT current_timestamp,
    fg_official          INTEGER,
    fg_retail            INTEGER,
    fg_institutional     INTEGER,
    fg_onchain           INTEGER,
    fg_trader            INTEGER,
    fg_synthetic         INTEGER,
    etf_flow_usd_m       FLOAT,
    weight_retail        FLOAT,
    weight_institutional FLOAT,
    weight_onchain       FLOAT,
    weight_trader        FLOAT,
    fg_retail_alt_me     INTEGER,
    fg_retail_cmc        INTEGER,
    fg_retail_coingecko  INTEGER
);
"""

# ── v1.6: research layer ─────────────────────────────────────────────────────
DDL_RESEARCH_RUNS = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id         VARCHAR PRIMARY KEY,
    started_at     TIMESTAMPTZ NOT NULL,
    finished_at    TIMESTAMPTZ,
    data_start     TIMESTAMPTZ,
    holdout_start  TIMESTAMPTZ,
    data_end       TIMESTAMPTZ,
    n_assets       INTEGER,
    proposer       VARCHAR,
    summary        VARCHAR          -- JSON
);
"""
DDL_HYPOTHESES = """
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id  VARCHAR PRIMARY KEY,
    run_id         VARCHAR,
    created_at     TIMESTAMPTZ NOT NULL,
    author         VARCHAR,         -- llm | heuristic | baseline
    statement      VARCHAR,
    rationale      VARCHAR,
    spec           VARCHAR,         -- JSON StrategySpec, fixed before testing
    min_trades     INTEGER,
    status         VARCHAR,         -- registered | accepted | rejected | inconclusive | retired
    decided_at     TIMESTAMPTZ,
    result         VARCHAR          -- JSON
);
"""
DDL_STRATEGY_RESULTS = """
CREATE TABLE IF NOT EXISTS strategy_results (
    run_id         VARCHAR NOT NULL,
    hypothesis_id  VARCHAR NOT NULL,
    segment        VARCHAR NOT NULL, -- train | holdout
    n              INTEGER,
    days           INTEGER,
    mean_pct       DOUBLE,
    ci_low         DOUBLE,
    ci_high        DOUBLE,
    win_rate       DOUBLE,
    total_pct      DOUBLE,
    diff_vs_random DOUBLE,
    diff_ci_low    DOUBLE,
    diff_ci_high   DOUBLE,
    equity         VARCHAR,          -- JSON [{ts, cum_pct}]
    PRIMARY KEY (run_id, hypothesis_id, segment)
);
"""
DDL_RESEARCH_JOURNAL = """
CREATE TABLE IF NOT EXISTS research_journal (
    entry_id       VARCHAR PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL,
    run_id         VARCHAR,
    author         VARCHAR,          -- llm | heuristic
    title          VARCHAR,
    body_md        VARCHAR
);
"""

def get_conn(max_retries: int = 10, base_delay: float = 2.0) -> duckdb.DuckDBPyConnection:
    """
    Connect to DuckDB with retry-on-lock. DuckDB is single-writer; if another
    service (ingest/pairs/fiat) holds the lock, wait and retry instead of crashing.
    With 10 retries averaging ~3s each, tolerates ~30s of contention — plenty for
    a ~10s ingest cycle to release.
    """
    for attempt in range(max_retries):
        try:
            return duckdb.connect(config.DB_PATH)
        except duckdb.IOException as e:
            if 'lock' not in str(e).lower() or attempt == max_retries - 1:
                raise
            time.sleep(base_delay + random.uniform(0, 2))
    raise RuntimeError("Could not acquire DB lock after retries")

def _refresh_pair_reputation(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuild pair_reputation from llm_verdicts, keyed by the DIRECTIONAL pair.
    'SOL/BTC' (bet SOL recovers vs BTC) and 'BTC/SOL' are opposite bets and get
    separate reputations; merging them used to average a win and a loss together."""
    rows = con.execute(f"""
        SELECT cid, n, avg_out, win_rate FROM (
            SELECT
                top_pair AS cid,
                COUNT(*) AS n,
                AVG(outcome_pct) AS avg_out,
                SUM(CASE WHEN outcome_pct > 0 THEN 1.0 ELSE 0 END) / COUNT(*) AS win_rate
            FROM llm_verdicts
            WHERE outcome_pct IS NOT NULL AND action = 'accumulate'
              AND top_pair LIKE '%/%'
            GROUP BY cid
        ) WHERE n >= {REP_MIN_SAMPLES}
    """).fetchall()
    con.execute("DELETE FROM pair_reputation")   # full rebuild; drops legacy canonical keys
    if not rows:
        return
    now = datetime.now(timezone.utc)
    for cid, n, avg_out, win_rate in rows:
        if avg_out is None or win_rate is None:
            continue
        rep_score = reputation_score(avg_out, win_rate, n)
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
    con.execute(DDL_FG_HISTORY_SEQ)      # v1.1
    con.execute(DDL_FG_HISTORY)          # v1.1
    con.execute(DDL_MINE_INJECTIONS)     # v1.4
    con.execute(DDL_REPORT_INSIGHTS_SEQ) # v1.4
    con.execute(DDL_REPORT_INSIGHTS)     # v1.4
    con.execute(DDL_RESEARCH_RUNS)       # v1.6
    con.execute(DDL_HYPOTHESES)          # v1.6
    con.execute(DDL_STRATEGY_RESULTS)    # v1.6
    con.execute(DDL_RESEARCH_JOURNAL)    # v1.6
    # v1.1 migration: recreate fg_history if it exists but id has no DEFAULT
    # (happens when table was created before the sequence was added)
    try:
        info  = {r[0]: r for r in con.execute("DESCRIBE fg_history").fetchall()}
        id_row = info.get('id')
        if id_row and (id_row[1].upper() != 'BIGINT' or id_row[4] is None):
            cnt = con.execute("SELECT COUNT(*) FROM fg_history").fetchone()[0]
            if cnt == 0:
                con.execute("DROP TABLE fg_history")
                con.execute(DDL_FG_HISTORY)
    except Exception:
        pass

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
    # v1.2: macro-aware position sizing
    _add_col_if_missing(con, 'fantasy_trades', 'position_multiplier', 'FLOAT DEFAULT 1.0')
    # v1.3: multi-source retail F&G breakdown in fg_history
    _add_col_if_missing(con, 'fg_history', 'fg_retail_alt_me',    'INTEGER')
    _add_col_if_missing(con, 'fg_history', 'fg_retail_cmc',       'INTEGER')
    _add_col_if_missing(con, 'fg_history', 'fg_retail_coingecko', 'INTEGER')
    # v1.4: order book liquidity score per verdict
    _add_col_if_missing(con, 'llm_verdicts', 'liquidity_score', 'FLOAT')
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

def set_bot_state(con, key: str, value: str) -> None:
    con.execute("""
        INSERT INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                        updated_at = excluded.updated_at
    """, [key, value, datetime.now(timezone.utc)])

def get_bot_state(con, key: str) -> str | None:
    row = con.execute("SELECT value FROM bot_state WHERE key = ?", [key]).fetchone()
    return row[0] if row else None

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

def _get_macro_cache_float(con, key: str, max_age_seconds: int = 3600) -> float | None:
    v = _get_macro_cache(con, key, max_age_seconds)
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None
