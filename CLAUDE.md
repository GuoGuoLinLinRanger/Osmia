# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Osmia: a self-hosted crypto **paper-trading** research system. It ingests live exchange data, computes statistical anomaly signals, asks an LLM for a verdict, logs paper trades, scores its own calls against what the market did, and feeds that back into ranking. It never places real orders — `balances.json` only seeds the paper ledger.

## Setup and commands

Python 3.10+. `pip install -e ".[dev]"` (add `anthropic` to the extras for Claude). Config comes from `.env` in `OSMIA_HOME` (defaults to the repo root) — copy `.env.example`.

```bash
python -m osmia --init-db        # create/migrate the DuckDB schema
python -m osmia --ingest         # one ticker ingest
python -m osmia --analyze        # pair cycle: signals -> LLM verdict -> paper trade
python -m osmia --fiat           # USDT-entry cycle
python -m osmia --outcomes       # score past verdicts (learning loop)
python -m osmia --report --write-insights   # LLM report + feedback insights
python -m osmia --status | --portfolio | --backup
python -m osmia --run            # every job on its schedule, one process
python -m pytest -q              # tests: in-memory DuckDB, no network
python -m pyflakes osmia
```

`history/` holds the frozen single-file versions (V0.2 → `TraderV13.py` v1.5) for reference only; never edit or import them.

## Package layout (`osmia/`)

Dependencies only point down this list (no cycles):

| Module | Role |
|---|---|
| `config.py` | env-driven settings + all trading constants |
| `core.py` | pure helpers: pair ids, LLM-output parsing, keyword matching, `trade_pnl`, reputation/threshold math |
| `store.py` | DuckDB DDL, `get_conn` (lock retry), `init_db` migrations, key/value caches |
| `llm.py` | `chat()` for OpenAI-compatible endpoints or Anthropic (SDK); `ask_verdict` / `ask_insights` / `ask_report` / `ping_llm` |
| `notify.py` | Discord (mentions disabled) |
| `sources/market.py` | ccxt ingest/backfill/order book/watchlist; `make_exchange()` honours `OSMIA_EXCHANGE` |
| `sources/news.py` | CoinDesk/Reddit/DeFiLlama news, unlocks, news veto + sentiment |
| `sources/sentiment.py` | macro block, 4-cohort Fear & Greed |
| `signals.py` | robust z-scores, candidates, confidence |
| `portfolio.py` | paper ledger: sizing, open/close/settle, views |
| `learning.py` | outcome scoring, report + insight feedback loop, status |
| `agent.py` | `analyze_pairs` / `analyze_fiat` and open-position LLM review |
| `scheduler.py` | `--run` loop; `backup.py` DB backups; `cli.py` argparse entry |
| `research/` | `data` (price matrix, CSV import), `replay` (StrategySpec + no-lookahead backtest), `stats` (day-block bootstrap, fixed verdict rule), `hypotheses` (LLM/heuristic proposals), `journal`, `loop` (`--research`) |

## Research layer rules

- A hypothesis is a `StrategySpec`; anything from the LLM goes through `sanitize_spec` (bounded params) before it is tested.
- Hypotheses are written to `hypotheses` with status `registered` **before** any replay; the verdict rule (`stats.verdict`) is fixed: accepted only if the holdout 95% CI of (strategy − matched random baseline) is above 0 with ≥ 20 trades over ≥ 3 days.
- The proposer sees training-period numbers only, plus outcomes of past hypotheses whose data ended before the current holdout (`_tested_context`). Never pass holdout results into a proposal prompt.
- Replay must only use bars ≤ t at step t (`test_replay_has_no_lookahead`). Returns are net of round-trip fees.
- `OSMIA_LLM_CACHE=readwrite` records LLM answers; `readonly` serves them only (the public demo).

## Rules that are easy to break

- **Env-driven settings are read at call time as `config.NAME`** (`DB_PATH`, `BALANCES_FILE`, `LLM_*`, `DISCORD_WEBHOOK`, `EXCHANGE_ID`), never `from .config import DB_PATH`, so tests and the scheduler can override them.
- **Pair-trade direction:** candidate `BASE/QUOTE` with z below the threshold means BASE is cheap vs QUOTE, so the trade SELLS QUOTE and BUYS BASE. Verdicts, outcomes and reputation are keyed by that `BASE/QUOTE` signal pair (= `buy_asset/sell_asset` on a trade). P&L is measured against holding the sold leg (`trade_pnl`).
- **Only `action == 'accumulate'` trades.** LLM output goes through `parse_llm_verdict` / `sanitize_insight`; never trust raw model fields.
- **DuckDB is single-writer.** Use `get_conn()`, and close the connection around slow network calls (LLM, RSS, order book). `--run` executes jobs sequentially so it never contends.
- Schema changes: add to the `DDL_*` constants or `_add_col_if_missing` calls in `store.init_db` (idempotent, runs every invocation). No migration scripts.
- Headlines are untrusted text: keep `UNTRUSTED_NOTE` in any prompt that embeds them.

## Deployment

- **Container/cloud:** `python -m osmia --run` (set `OSMIA_EXCHANGE` away from `binance` on US/cloud IPs — Binance returns HTTP 451 there).
- **Home server:** nine systemd oneshot units in `deploy/systemd/` run `python -m osmia --<job>` from `/opt/llm_trader` with `OSMIA_HOME=/opt/llm_trader`; install with `sudo bash deploy/install_systemd.sh`. See `deploy/systemd/README.md`.
