# LLM Trader ("Osmia")

A self-hosted crypto **paper-trading** monitor. It ingests live Binance market data, computes statistical anomaly signals, asks an **LLM** (a local Qwen or any OpenAI-compatible endpoint, or Claude) for a trade verdict, then logs fantasy trades and pushes alerts to Discord.

> **It never places real orders.** `balances.json` holds your holdings (sample values in this repo) and is only used for position-sizing math.

## How it works

```
ingest → market_history → candidate scoring → LLM verdict → fantasy trade + Discord
```

1. **Ingest** — `ccxt` pulls Binance tickers for the configured USDT pairs every 5 minutes and stores per-minute rows (VWAP ratio, spread, volume z-score) in a single DuckDB file.
2. **Signal building** — MAD z-score mean-reversion signal, z-velocity, persistence, per-pair reputation (historical win rate), and trend slope. A rule-based score ranks candidates; the top N go to the LLM.
3. **Market context** — a 4-cohort Fear & Greed system (retail / institutional / on-chain / trader) producing a synthetic F&G alongside the official one, plus CoinDesk/Reddit news, DeFiLlama TVL, BTC dominance, and a token-unlock calendar. An unlock within 48h triggers a "heightened mode" with relaxed thresholds and an alert injected into the prompt.
4. **LLM verdict** — the prompt goes to the configured LLM (local Qwen by default); the model returns a JSON verdict (`action`, `top_pair`, `rank`, `thesis`, `risks`). Degrades gracefully to `pass` if the server is unreachable.
5. **Confidence & sizing** — a 0–10 confidence score drives a signal-weighted notional (base 5%, floored/capped), adjusted by a 0.5×–2.0× macro multiplier and order-book liquidity (top-10 depth, spread, slippage estimate). Hard $10 floor per trade; skipped if predicted profit < $1.
6. **Fantasy execution** — paper trades logged to `fantasy_trades`, auto-closed on z-score crossback, F&G-aware stop-loss (−8% in Extreme Fear up to −18% in Extreme Greed), news veto, or max-hold expiry. Proceeds settle to a fantasy USDT balance.
7. **Learning loop** — past verdicts are scored against realized price moves; outcomes feed per-pair reputation, and an LLM-written 6h report can write pair-level boost/warn/suppress insights back into the analyze cycle.

## Repository layout

| Path | Purpose |
|------|---------|
| `osmia/` | The package. `config`, `core`, `store`, `llm`, `notify`, `sources/` (market, news, sentiment), `signals`, `portfolio`, `learning`, `agent`, `scheduler`, `backup`, `cli`. |
| `tests/` | pytest suite (in-memory DuckDB, no network). |
| `demo/` | Bundled public price data for the replay demo. |
| `scripts/` | `make_demo_dataset.py`. |
| `balances.json` | Starting holdings for the paper ledger (sample values — replace with your own). |
| `deploy/systemd/` | Nine oneshot service + timer units for a home server. |
| `history/` | The original single-file versions, V0.2 → `TraderV13.py` (v1.5), kept for reference. |
| `CLAUDE.md` | Architecture and the rules that are easy to break. |

## Running

```bash
pip install -e ".[dev]"                # Python 3.10+; add ,anthropic for Claude
cp .env.example .env                    # then set the LLM endpoint (see below)

python -m osmia --init-db               # create/migrate the DuckDB schema
python -m osmia --ingest                # fetch tickers -> market_history
python -m osmia --analyze               # crypto-to-crypto signal + LLM verdict
python -m osmia --fiat                  # USDT-entry signal + LLM verdict
python -m osmia --backfill              # one-shot 90-day 1m OHLCV backfill
python -m osmia --outcomes              # score past verdicts vs actual moves
python -m osmia --report --write-insights   # LLM report + feedback insights
python -m osmia --portfolio | --status | --daily-pnl | --backup
python -m osmia --research              # one autonomous research cycle
python -m osmia --run                   # every job on its schedule, in one process
python -m pytest -q                     # tests
```

Sizing overrides: `--size 250`, `--position-pct 1.0`, `--min-position 10`, `--min-profit 1`. Mining income: `--mine-proceeds [--mine-value N]` or `--mine-auto` (unMineable API, address via `UNMINEABLE_XEL_ADDRESS`). Cohort weights: `--set-weights retail=0.3,institutional=0.4,onchain=0.2,trader=0.1`.

## Configuration

Copy `.env.example` to `.env` (in `OSMIA_HOME`, which defaults to the repo folder) and set:

- **LLM** — `OSMIA_LLM_PROVIDER=openai` (default) talks to any OpenAI-compatible chat endpoint: a local Qwen via llama.cpp / vLLM / LM Studio / Ollama, or a hosted one with `OSMIA_LLM_API_KEY`. `OSMIA_LLM_PROVIDER=anthropic` uses Claude through the official SDK (`pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY`). If every analysis silently passes, the LLM is unreachable — `--status` says so.
- **Exchange** — `OSMIA_EXCHANGE` (any ccxt id, default `binance`). Binance returns HTTP 451 to US and most cloud IPs; use `binanceus`, `kraken` or `coinbase` there.
- `OSMIA_DISCORD_WEBHOOK` — optional; Discord pushes are no-ops when blank.
- `OSMIA_HOME` / `OSMIA_DB_PATH` / `OSMIA_BALANCES_FILE` — where the DuckDB file, backups and balances live.
- `CMC_API_KEY` — optional CoinMarketCap key for its Fear & Greed feed.

Trading constants (`ASSETS`, `BINANCE_DIRECT_PAIRS`, z-score/confidence thresholds, cooldowns, sizing) live in `osmia/config.py`.

## Research loop (autonomous)

`python -m osmia --research` runs one research cycle (the scheduler runs one a day):

1. Split the stored price history in time: **train** | **holdout** (most recent 30%).
2. Measure the default strategy and baselines on train only.
3. The LLM reads that training evidence and proposes new, falsifiable hypotheses (e.g. *"pairs quoted in BTC/ETH revert better than alt-vs-alt"*). Each becomes a bounded `StrategySpec`. With no LLM reachable, a heuristic proposer explores the same space and is labelled as such.
4. Every hypothesis is **pre-registered** (spec frozen, timestamped) before testing.
5. Each is replayed on the holdout — no lookahead, net of fees — against a **matched random baseline** (same timing, filters and exits, random pair, five seeds). The live system's own logged LLM picks are replayed too, as an ablation.
6. Fixed verdict rule: **accepted** only if the 95% confidence interval of (strategy − random) is above zero, with ≥ 20 trades over ≥ 3 days. Intervals come from a **day-block bootstrap**, because trades on the same day are not independent. Previously accepted hypotheses are re-tested and **retired** if they stop holding.
7. The LLM writes a journal entry from the final numbers.

The proposer never sees holdout outcomes that overlap its own holdout, and it never re-tests a spec it has already tried.

### Try it on bundled data (no keys, no LLM needed)

```bash
python -m osmia --demo-load demo/prices_5m.csv.gz   # 21 days of public 5-min closes, 14 assets
python -m osmia --research --no-llm                  # heuristic proposer
python -m osmia --research                           # LLM proposer, if one is configured
```

Regenerate the dataset with `python scripts/make_demo_dataset.py --days 21`.

## Production deployment

On a home server, nine systemd oneshot services + timers run `python -m osmia --<job>` from `/opt/llm_trader` (install with `sudo bash deploy/install_systemd.sh`). On a container or cloud host, run `python -m osmia --run` instead.

| Unit | Command | Schedule |
|------|---------|----------|
| `AI-Trader_ingest` | `--ingest` | every 5 min |
| `AI-Trader_fg` | `--fetch-fg` | every 5 min |
| `AI-Trader_pairs` | `--analyze` | every 10 min |
| `AI-Trader_fiat` | `--fiat` | every 10 min |
| `AI-Trader_news` | `--scrape-news` | every 30 min |
| `AI-Trader_unlocks` | `--scrape-unlocks` | daily 06:00 |
| `AI-Trader_backup` | `--backup` | daily 06:00 + 18:00 |
| `AI-Trader_outcomes` | `--outcomes` | hourly |
| `AI-Trader_insights` | `--report --write-insights` | every 6 h |

```bash
systemctl status AI-Trader_pairs
journalctl -u AI-Trader_pairs -f
sudo systemctl restart AI-Trader_pairs   # after pulling new code
```

**DuckDB is single-writer** — the overlapping timers rely on `get_conn()` retrying on lock (~30s tolerance). Keep new DB access going through `get_conn()` and don't hold the connection across slow network calls.

## Database tables

`market_history` (raw ticks) · `llm_verdicts` (every analysis + outcome score) · `fantasy_trades` · `fantasy_balances` · `market_context` · `asset_news` · `token_unlocks` · `bot_state` · `fg_history` (Grafana divergence charts) · `mine_injections` · `report_insights` (6h-TTL feedback loop).

Schema is migrated in-place: `--init-db` runs idempotent `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN`, so schema changes go in the `DDL_*` constants rather than migration scripts.
