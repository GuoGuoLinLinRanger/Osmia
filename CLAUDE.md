# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-hosted crypto "paper trading" monitor ("Osmia"/"Qwen Trader"). It ingests live Binance market data, computes statistical anomaly signals, asks a **local** Qwen LLM (OpenAI-compatible endpoint) for a verdict, then logs fantasy/paper trades and pushes Discord alerts. It never places real orders — `balances.json` holds the user's *actual* holdings but is only used for position sizing math.

## Setup

`pyproject.toml` declares every runtime dependency (ccxt, duckdb, feedparser, numpy, pandas, python-dotenv, requests) and supports Python 3.10+. Use any virtualenv with those installed (`uv sync`, or `pip install` the listed deps). Config comes from `.env` in `OSMIA_HOME` — copy `.env.example`.

```bash
python TraderV13.py --init-db    # creates market_data.duckdb in OSMIA_HOME
python TraderV13.py --ingest     # one live ingest
```

The original home server still runs from a legacy `venv/` under `/opt/llm_trader` with `OSMIA_HOME=/opt/llm_trader`.

## Versioning convention (important)

The codebase is a single monolithic script copied to a new file on each iteration: `TraderV04.py` → `…V05` → `…V06` → `…V07` → `…V08` → `…V09` → `…V10` → `…V11` → `…V12` → **`TraderV13.py` (current production, since 2026-06-08)**. Each version is a near-superset rewrite; the module docstring at the top lists what changed from the prior version. Superseded scripts and one-off migration helpers live in `history/`.

**Production points at a specific filename.** The systemd units below hardcode `TraderV13.py`. If you create a new version, you must also repoint the unit `ExecStart=` lines (and reload) — bumping the file alone changes nothing in production. See `deploy/systemd/README.md` for the cutover + rollback procedure.

## Run commands

All subcommands are a mutually-exclusive argparse group on the one script:

```bash
venv/bin/python3 TraderV08.py                 # default: ingest + pair analysis
venv/bin/python3 TraderV08.py --ingest        # fetch tickers → market_history
venv/bin/python3 TraderV08.py --analyze       # crypto-to-crypto (pair) signal + LLM verdict
venv/bin/python3 TraderV08.py --fiat          # USDT-entry signal + LLM verdict
venv/bin/python3 TraderV08.py --backfill      # one-shot 90-day 1m OHLCV backfill (run once)
venv/bin/python3 TraderV08.py --outcomes      # score past verdicts vs actual moves (feeds reputation)
venv/bin/python3 TraderV08.py --portfolio
venv/bin/python3 TraderV08.py --status        # DB health summary
venv/bin/python3 TraderV08.py --daily-pnl
venv/bin/python3 TraderV08.py --scrape-news   # (v0.8) per-asset news → asset_news; auto-runs before --analyze
venv/bin/python3 TraderV08.py --scrape-unlocks# (v0.8) token unlock calendar → token_unlocks
venv/bin/python3 TraderV08.py --init-db       # (re)create schema + run migrations, then exit
venv/bin/python3 TraderV08.py --size 250      # override notional for this run (combine with --analyze/--fiat)
```

Tests live in `tests/` (pytest, in-memory DuckDB, no network): `python -m pytest -q`. The old `history/test_ingest.py` is unmaintained.

**Pair-trade direction:** a candidate `BASE/QUOTE` with z below the threshold means BASE is cheap vs QUOTE, so the trade SELLS QUOTE and BUYS BASE. Verdicts, outcomes and reputation are keyed by that `BASE/QUOTE` signal pair (= `buy_asset/sell_asset` on a trade); P&L is measured against holding the sold leg (`trade_pnl`). `init_db()` runs on every invocation and is idempotent: it `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN` for migrations, so schema changes are made by adding to the `DDL_*` constants / `_add_col_if_missing` lists rather than writing migration scripts.

## Production deployment (systemd)

Nine oneshot services + timers, all running as a dedicated `osmia` user from `/opt/llm_trader` (set `OSMIA_HOME`) via `venv/bin/python3 TraderV13.py`:

| Unit | Command | Timer |
|------|---------|-------|
| `AI-Trader_ingest`  | `--ingest` | every 5 min |
| `AI-Trader_fg`      | `--fetch-fg` | every 5 min |
| `AI-Trader_pairs`   | `--analyze` | every 10 min |
| `AI-Trader_fiat`    | `--fiat` | every 10 min |
| `AI-Trader_news`    | `--scrape-news` | every 30 min |
| `AI-Trader_unlocks` | `--scrape-unlocks` | daily 06:00 |
| `AI-Trader_backup`  | `backup_db.py` | daily 06:00 + 18:00 |
| `AI-Trader_outcomes` | `--outcomes` | hourly |
| `AI-Trader_insights` | `--report --write-insights` | every 6 h |

All units point at `TraderV13.py`.

The `news`/`unlocks` units (v0.8) are defined in `deploy/systemd/`. `--scrape-news` also auto-runs at the head of every `--analyze`, so the news timer is a top-up.

```bash
systemctl status AI-Trader_pairs        # one of ingest/pairs/fiat
journalctl -u AI-Trader_pairs -f
sudo systemctl restart AI-Trader_pairs  # after editing the script
```

## Architecture / data flow

`ingest → market_history → candidate scoring → LLM verdict → fantasy trade + Discord`, all over one DuckDB file.

1. **Ingest** (`ingest`): `ccxt.binance().fetch_tickers()` for the 14 USDT pairs in `ASSETS`, computes per-row `vwap_ratio`, `spread_pct`, `high_low_pct`, and a MAD-based `volume_z`, then `INSERT OR IGNORE` into `market_history` (keyed by `(timestamp, asset_id)`, minute-truncated).
2. **Signal building** (`build_candidates` / `get_fiat_candidates`): pulls recent history, computes a **MAD z-score** of price (mean-reversion signal), z-velocity, persistence, per-pair **reputation** (historical win rate from scored outcomes), and **trend slope**. `rule_based_score` ranks candidates; top N go to the LLM. Pair mode synthesizes crypto-to-crypto pairs and routes them via `BINANCE_DIRECT_PAIRS` (direct vs 2-leg-through-USDT, which changes fee math in `fee_breakdown`).
3. **Market context** (`fetch_market_context` + v0.8 `fetch_macro_context`): Fear & Greed (alternative.me) + CoinDesk RSS headlines, plus (v0.8) BTC dominance / total-mcap 24h change from CoinGecko `/global`, combined by `build_macro_block` into a `[MACRO CONTEXT]` block prepended to the **pair** prompt. v0.8 also scrapes **per-asset news** (`fetch_all_news`: CoinDesk + Reddit + DeFiLlama TVL → `asset_news`, 24h-pruned) and a **token-unlock calendar** (`fetch_unlocks` → `token_unlocks`); a pending unlock <48h triggers **heightened mode** (`check_heightened_mode`): relaxes the Z-buy threshold for that asset's pairs, injects an unlock alert + `HEIGHTENED_PROMPT_SUFFIX`, and persists state in `bot_state`. All of this is wired into `--analyze` (pairs) only — `--fiat` is unchanged. Everything degrades gracefully (Reddit 403s, missing unlock source, unreachable APIs → skipped).
4. **LLM verdict** (`ask_qwen` / `ping_qwen`): POSTs the built prompt to the local Qwen server at `QWEN_URL`. The model is asked to return one JSON line (`action`, `top_pair`, `rank`, `thesis`, `risks`) followed by prose; `ask_qwen` splits the first `{…}` line as JSON and the rest as prose, and **degrades gracefully** to an `action: 'pass'` fallback if the server is unreachable.
5. **Confidence & sizing**: `derive_confidence` produces a 0–10 score; `signal_weighted_notional` sizes the position (base 5% ± reputation ± z-depth, floored 3% / capped 15%) against `balances.json`.
6. **Fantasy execution** (`execute_fantasy_*` / `_insert_trade`): if confidence ≥ `MIN_CONFIDENCE_TRADE` and the pair isn't on cooldown / at `MAX_TRADES_PER_PAIR`, log a paper trade to `fantasy_trades`. `_auto_close_trades` closes on Z crossback (`Z > ZSCORE_CLOSE_THRESH`) or `MAX_HOLD_HOURS` expiry.
7. **Outcome scoring** (`score_outcomes`): grades past `llm_verdicts` against realized price moves after `OUTCOME_WINDOW_HOURS`; these outcomes feed the per-pair reputation used back in step 2 (the system's only learning loop).

**DuckDB is single-writer.** Ingest, pairs, and fiat run on overlapping timers, so `get_conn()` retries on lock (~30s tolerance). Keep any new DB access going through `get_conn()` and avoid holding the connection across slow network calls (LLM/RSS fetches happen before/after, not while writing).

**Tables:** `market_history` (raw ticks), `llm_verdicts` (every analysis run + later outcome score), `fantasy_trades`, `fantasy_balances` (snapshots), `market_context`, and (v0.8) `asset_news` (per-asset headlines, 24h-pruned), `token_unlocks` (unlock calendar), `bot_state` (key/value, e.g. `heightened_<ASSET>_until`).

## Configuration

Paths, the LLM endpoint and Discord come from environment variables (see `.env.example`); everything else is constants near the top of `TraderV13.py`:

- `QWEN_URL` / `QWEN_MODEL` — the LLM endpoint, from `OSMIA_LLM_URL` / `OSMIA_LLM_MODEL` (default `http://localhost:8081`). If analysis is silently "pass"-ing, check this is reachable.
- `DISCORD_WEBHOOK` — from `OSMIA_DISCORD_WEBHOOK`; blank = Discord pushes are no-ops.
- `DB_PATH`, `BALANCES_FILE` — under `OSMIA_HOME` (defaults to the script's folder), overridable via `OSMIA_DB_PATH` / `OSMIA_BALANCES_FILE`. `.env` is loaded from `OSMIA_HOME`.
- `ASSETS` / `SYMBOLS`, `BINANCE_DIRECT_PAIRS`, and the threshold block (z-score buy/close, confidence, cooldowns, position-sizing percentages).
