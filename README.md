# LLM Trader ("Osmia")

A self-hosted crypto **paper-trading** monitor. It ingests live Binance market data, computes statistical anomaly signals, asks a **local Qwen LLM** (any OpenAI-compatible endpoint) for a trade verdict, then logs fantasy trades and pushes alerts to Discord.

> **It never places real orders.** `balances.json` holds your holdings (sample values in this repo) and is only used for position-sizing math.

## How it works

```
ingest → market_history → candidate scoring → LLM verdict → fantasy trade + Discord
```

1. **Ingest** — `ccxt` pulls Binance tickers for the configured USDT pairs every 5 minutes and stores per-minute rows (VWAP ratio, spread, volume z-score) in a single DuckDB file.
2. **Signal building** — MAD z-score mean-reversion signal, z-velocity, persistence, per-pair reputation (historical win rate), and trend slope. A rule-based score ranks candidates; the top N go to the LLM.
3. **Market context** — a 4-cohort Fear & Greed system (retail / institutional / on-chain / trader) producing a synthetic F&G alongside the official one, plus CoinDesk/Reddit news, DeFiLlama TVL, BTC dominance, and a token-unlock calendar. An unlock within 48h triggers a "heightened mode" with relaxed thresholds and an alert injected into the prompt.
4. **LLM verdict** — the prompt is POSTed to a local Qwen server; the model returns a JSON verdict (`action`, `top_pair`, `rank`, `thesis`, `risks`). Degrades gracefully to `pass` if the server is unreachable.
5. **Confidence & sizing** — a 0–10 confidence score drives a signal-weighted notional (base 5%, floored/capped), adjusted by a 0.5×–2.0× macro multiplier and order-book liquidity (top-10 depth, spread, slippage estimate). Hard $10 floor per trade; skipped if predicted profit < $1.
6. **Fantasy execution** — paper trades logged to `fantasy_trades`, auto-closed on z-score crossback, F&G-aware stop-loss (−8% in Extreme Fear up to −18% in Extreme Greed), news veto, or max-hold expiry. Proceeds settle to a fantasy USDT balance.
7. **Learning loop** — past verdicts are scored against realized price moves; outcomes feed per-pair reputation, and a Qwen-generated 6h report can write pair-level boost/warn/suppress insights back into the analyze cycle.

## Repository layout

| Path | Purpose |
|------|---------|
| `TraderV13.py` | **Current production script** (v1.4). Single monolithic file; everything lives here. |
| `TraderV12.py` | Previous version, kept for rollback. |
| `backup_db.py` | DuckDB backup helper (runs twice daily in prod). |
| `balances.json` | Holdings used for position-sizing math only (sample values — replace with your own). |
| `deploy/systemd/` | Seven oneshot service + timer units (ingest, F&G, pairs, fiat, news, unlocks, backup). |
| `history/` | Superseded versions (V0.2 → V12) and one-off migration helpers. |
| `CLAUDE.md` | In-depth architecture and operations notes. |

**Versioning convention:** each iteration copies the script to a new file (`TraderV12.py` → `TraderV13.py`); the module docstring lists what changed. The systemd units hardcode the filename, so a new version also requires repointing `ExecStart=` (see `deploy/systemd/README.md`).

## Running

> ⚠️ Use the legacy `venv/` interpreter — it has all runtime deps (`duckdb`, `numpy`, `pandas`, `ccxt`, `requests`, `feedparser`). The uv-managed `.venv/` is incomplete (`pyproject.toml` only declares `ccxt` and `requests`).

```bash
venv/bin/python3 TraderV13.py                  # default: ingest + pair analysis
venv/bin/python3 TraderV13.py --ingest         # fetch tickers → market_history
venv/bin/python3 TraderV13.py --analyze        # crypto-to-crypto signal + LLM verdict
venv/bin/python3 TraderV13.py --fiat           # USDT-entry signal + LLM verdict
venv/bin/python3 TraderV13.py --backfill       # one-shot 90-day 1m OHLCV backfill
venv/bin/python3 TraderV13.py --outcomes       # score past verdicts vs actual moves
venv/bin/python3 TraderV13.py --portfolio      # fantasy portfolio state
venv/bin/python3 TraderV13.py --status         # DB health summary
venv/bin/python3 TraderV13.py --daily-pnl      # daily P&L summary to Discord
venv/bin/python3 TraderV13.py --scrape-news    # per-asset news → asset_news
venv/bin/python3 TraderV13.py --scrape-unlocks # token unlock calendar
venv/bin/python3 TraderV13.py --fetch-fg       # fetch all 4 F&G cohorts (5-min cron)
venv/bin/python3 TraderV13.py --report         # Qwen-powered daily performance report
venv/bin/python3 TraderV13.py --init-db        # (re)create schema + migrations, then exit
```

Sizing overrides: `--size 250`, `--position-pct 1.0`, `--min-position 10`, `--min-profit 1`. Mining income: `--mine-proceeds [--mine-value N]` or `--mine-auto` (unMineable API, address via `UNMINEABLE_XEL_ADDRESS`). Cohort weights: `--set-weights retail=0.3,institutional=0.4,onchain=0.2,trader=0.1`.

## Configuration

Copy `.env.example` to `.env` (in `OSMIA_HOME`, which defaults to the repo folder) and set:

- `OSMIA_LLM_URL` / `OSMIA_LLM_MODEL` — any OpenAI-compatible chat endpoint (local Qwen via llama.cpp, vLLM, LM Studio, Ollama…). If every analysis silently passes, check this is reachable.
- `OSMIA_DISCORD_WEBHOOK` — optional; Discord pushes are no-ops when blank.
- `OSMIA_HOME` / `OSMIA_DB_PATH` / `OSMIA_BALANCES_FILE` — where the DuckDB file and balances live.
- `CMC_API_KEY` — optional CoinMarketCap key for its Fear & Greed feed.

Trading constants stay at the top of `TraderV13.py`:

- `ASSETS` / `SYMBOLS`, `BINANCE_DIRECT_PAIRS`, and the threshold block (z-score buy/close, confidence, cooldowns, sizing percentages).

## Production deployment

Nine systemd oneshot services + timers run from `/opt/llm_trader`:

| Unit | Command | Schedule |
|------|---------|----------|
| `AI-Trader_ingest` | `--ingest` | every 5 min |
| `AI-Trader_fg` | `--fetch-fg` | every 5 min |
| `AI-Trader_pairs` | `--analyze` | every 10 min |
| `AI-Trader_fiat` | `--fiat` | every 10 min |
| `AI-Trader_news` | `--scrape-news` | every 30 min |
| `AI-Trader_unlocks` | `--scrape-unlocks` | daily 06:00 |
| `AI-Trader_backup` | `backup_db.py` | daily 06:00 + 18:00 |
| `AI-Trader_outcomes` | `--outcomes` | hourly |
| `AI-Trader_insights` | `--report --write-insights` | every 6 h |

```bash
systemctl status AI-Trader_pairs
journalctl -u AI-Trader_pairs -f
sudo systemctl restart AI-Trader_pairs   # after editing the script
```

**DuckDB is single-writer** — the overlapping timers rely on `get_conn()` retrying on lock (~30s tolerance). Keep new DB access going through `get_conn()` and don't hold the connection across slow network calls.

## Database tables

`market_history` (raw ticks) · `llm_verdicts` (every analysis + outcome score) · `fantasy_trades` · `fantasy_balances` · `market_context` · `asset_news` · `token_unlocks` · `bot_state` · `fg_history` (Grafana divergence charts) · `mine_injections` · `report_insights` (6h-TTL feedback loop).

Schema is migrated in-place: `--init-db` runs idempotent `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN`, so schema changes go in the `DDL_*` constants rather than migration scripts.
