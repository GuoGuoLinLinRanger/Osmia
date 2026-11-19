# Building Osmia (LLM Trader) — Complete Step-by-Step Guide

This is the end-to-end recipe for building the system from nothing to a 24/7 production deployment. The phases mirror the order it was actually built in (v0.2 → v1.4), because each layer depends on data produced by the previous one.

> **Historical note:** this guide describes the single-file era (`TraderV*.py`, now in `history/`). Since v1.5 the code is the `osmia/` package, run as `python -m osmia --<job>`; see the README for current commands. File and constant names below refer to that era.

---

## Phase 0 — Infrastructure prerequisites

You need two machines (or one beefy one):

1. **A Linux server** (the trader host). Runs the Python pipeline and DuckDB. A home server or small VPS is fine — the workload is I/O-bound, not CPU-bound.
2. **A local LLM inference host** serving an OpenAI-compatible API. Osmia uses Qwen behind an endpoint like `http://<llm-host>:8081/v1/chat/completions` (llama.cpp server, vLLM, or LM Studio all work). The model name is passed as `qwen-instruct`. Nothing is sent to a cloud LLM — verdicts stay on the LAN.
3. **A Discord server you control** with a channel webhook (Server Settings → Integrations → Webhooks). This is the entire UI.
4. **No exchange API keys are needed.** All Binance access is public-endpoint only (`fetch_tickers`, `fetch_ohlcv`, `fetch_order_book` via ccxt). The system never places real orders.

## Phase 1 — Project skeleton and environment

```bash
sudo mkdir -p /opt/llm_trader && cd /opt/llm_trader
python3 -m venv venv
venv/bin/pip install ccxt duckdb numpy pandas requests feedparser python-dotenv
```

Create:
- `balances.json` — a flat JSON map of your real holdings (`{"USDT": 25.05, "BTC": 0.016, ...}`). Used **only** for position-sizing math.
- `.env` — secrets loaded via `python-dotenv` (e.g. `UNMINEABLE_XEL_ADDRESS`).

Everything lives in **one script** (`TraderVxx.py`). All configuration is constants at the top of the file (~lines 200–280), not env vars or CLI args:

```python
DB_PATH         = '/opt/llm_trader/market_data.duckdb'
BALANCES_FILE   = '/opt/llm_trader/balances.json'
QWEN_URL        = "http://<llm-host>:8081/v1/chat/completions"
QWEN_MODEL      = "qwen-instruct"
DISCORD_WEBHOOK = ""   # blank = Discord pushes are no-ops
ASSETS  = ['BTC','ETH','SOL','ADA','DOT','XRP','TRX','BNB','AVAX','LTC','XLM','APT','DOGE','SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]
```

**Versioning convention:** each iteration copies the file to a new name (`TraderV12.py` → `TraderV13.py`) and documents the changes in the module docstring. Old versions move to `history/`. Production systemd units hardcode the filename, so a version bump requires repointing the units (Phase 12).

## Phase 2 — Database layer (DuckDB, idempotent migrations)

Build `init_db()` so it runs safely on **every** invocation:

- `CREATE TABLE IF NOT EXISTS` for every table.
- Schema evolution via `_add_col_if_missing` lists (`ALTER TABLE ADD COLUMN`) instead of migration scripts.
- A `get_conn()` helper that **retries on lock for ~30 seconds**, because DuckDB is single-writer and the ingest/pairs/fiat timers overlap.

Rule that keeps it stable: never hold the DB connection across a slow network call. Fetch (LLM, RSS, APIs) first, then open the connection and write.

The final schema is 15 tables: `market_history`, `llm_verdicts`, `fantasy_trades`, `fantasy_balances`, `market_context`, `pair_reputation`, `exit_log`, `asset_news`, `token_unlocks`, `bot_state`, `macro_cache`, `fg_history`, `mine_proceeds`, `mine_injections`, `report_insights`. They are introduced phase by phase below.

## Phase 3 — Ingest (`--ingest`)

One `ccxt.binance().fetch_tickers()` call per run for all 14 symbols. For each ticker compute derived features before storing:

- `vwap_ratio` = last price / VWAP
- `spread_pct`, `high_low_pct`
- `volume_z` — a **MAD-based z-score** over a 50-sample window (`VOLUME_WINDOW = 50`)

Insert with `INSERT OR IGNORE` keyed on `(timestamp, asset_id)` truncated to the minute, so re-runs and overlapping timers are idempotent. This runs every 5 minutes forever; everything downstream is built on this table.

## Phase 4 — Historical backfill (`--backfill`, run once)

Signals need a baseline before live data accumulates. `fetch_ohlcv` pulls **90 days of 1-minute candles** (`BACKFILL_DAYS = 90`, `BACKFILL_TIMEFRAME = '1m'`) per asset into `market_history`, with a `source` column flagging backfilled rows so they're distinguishable from live ticks.

## Phase 5 — Signal engine

This is the statistical core, all on price *ratios* between assets:

1. **Pair synthesis**: from 14 assets, generate crypto-to-crypto candidate pairs. `canonical_pair()` / `pair_id()` alpha-sort the legs so BTC/ETH and ETH/BTC share one identity (reputation, trade caps).
2. **Robust z-score**: median + MAD (× 1.4826 consistency constant) instead of mean/stdev, so a single spike doesn't poison the baseline. Clamp |z| at 8.0 — a stale long median can manufacture fake z = −20.
3. **Freshness check**: the dislocation must still be live on the most recent 120 minutes (`RECENT_PAIR_SAMPLES`), not just present in the long window. Require ≥ 20 samples (`MIN_PAIR_SAMPLES`).
4. **Supporting features**: z-velocity (improving or worsening), persistence (how long below threshold), trend slope (normalised per-minute ratio slope vs `TREND_SLOPE_THRESHOLD = 0.00003` — the original 0.01 literally never fired, a real tuning lesson).
5. **Entry gate**: `ZSCORE_BUY_THRESH = -2.0` (relaxed to −1.5 in heightened mode, Phase 9).
6. **Fee-aware routing**: `BINANCE_DIRECT_PAIRS` (32 entries) marks pairs tradable in one leg; everything else costs two legs through USDT at 0.075% taker each — `fee_breakdown` computes true break-even per route.
7. **Funnel**: `rule_based_score` ranks all candidates; top 10 (`TOP_N_RAW`) are kept, top 3 (`TOP_N_LLM`) go to the LLM. This keeps the expensive step cheap.

A parallel, simpler path (`--fiat`) scores single-asset USDT entry signals.

## Phase 6 — LLM verdict layer

- `ping_qwen()` health-checks the endpoint first.
- The prompt contains: the scored candidates with all statistics, fee/break-even math, macro context block (Phase 9), and recent per-asset headlines.
- **Output contract**: the model must reply with one JSON line — `{action, top_pair, rank, thesis, risks}` — followed by free prose. `ask_qwen()` splits the first `{…}` line as JSON, keeps the rest as commentary.
- **Graceful degradation is non-negotiable**: any unreachable server, timeout, or malformed JSON resolves to `action: "pass"`. The pipeline must never crash because the LLM is down.
- Every verdict (including passes) is stored in `llm_verdicts` for later scoring.

## Phase 7 — Confidence, sizing, and fantasy execution

1. `derive_confidence()` → 0–10 score from the verdict + signal stats. Trade only if ≥ `MIN_CONFIDENCE_TRADE = 6`.
2. `signal_weighted_notional()`: base 12% of available balance (`TRADE_PCT_BASE`), + up to 5% reputation bonus + up to 5% z-depth bonus, floored at 7%, capped at 30% — against real `balances.json` values.
3. **Macro multiplier** (v1.2): 0.5×–2.0× from synthetic F&G level, official-vs-synthetic divergence, institutional cohort direction, and news sentiment.
4. **Order-book check** (v1.4): fetch top-10 bids/asks at signal time → real spread, liquidity score (0–1), slippage estimate for positions > $50. Thin books shrink or kill the trade.
5. **Economics floor** (v1.4): hard `MIN_POSITION_USD = 10` per trade; skip if expected profit (notional × |z| × 1% per z-unit − fees) < `MIN_PROFIT_USD = 1`.
6. **Guards**: 60-minute cooldown per pair, max 3 open trades per canonical pair.
7. Passing trades insert into `fantasy_trades` (entry price, qty, fees, status) and snapshot `fantasy_balances`. On close, proceeds settle into the fantasy USDT balance (`--settle-dust` sweeps sub-$0.10 leftovers).

## Phase 8 — Exits and the outcome loop

Exit mechanisms, checked every analyze cycle:
- **Z crossback**: close when z recovers past the threshold — made dynamic in v0.9 by `buffett_exit_threshold(fg, etf_flow)`: hold longer in Extreme Fear (target z = +1.5), exit early in Extreme Greed (−0.5), ±0.2–0.3 adjustment from ETF flow direction.
- **F&G-adaptive stop-loss** (v1.1): −8% (Extreme Fear) → −10% → −12% → −15% → −18% (Extreme Greed).
- **News veto**: keywords `hack, exploit, SEC, ban, delist, insolvency, shutdown, rug` in the last 6h of `asset_news` force a close.
- **Living re-evaluation** (v1.1): one fresh Qwen call per open position per cycle with updated news; EXIT urgency ≥ 7 → force close, < 7 → Discord warning.
- **Max hold**: 48 hours.

All exits log to `exit_log` with reason. Then the learning loop:

- `--outcomes` grades every verdict ≥ 24h old (`OUTCOME_WINDOW_HOURS`) against the realized ratio move and writes the result back to `llm_verdicts`.
- Graded outcomes roll up into `pair_reputation` (min 3 samples) — the win rate that feeds back into candidate ranking and sizing. **This loop is the system's only form of learning** and is what separates it from a static screener.

## Phase 9 — Market context engine

- **Macro block**: BTC dominance + total-mcap 24h change (CoinGecko `/global`) + Fear & Greed, assembled by `build_macro_block()` and prepended to every pair prompt.
- **Per-asset news** (`--scrape-news`, also auto-runs at the head of `--analyze`): CoinDesk RSS (keyword-filtered) + Reddit `r/<sub>/new.json` + DeFiLlama TVL → `asset_news`, pruned to 24h. Reddit 403s from VPS IPs → degrade to zero rows, never crash.
- **Token unlock calendar** (`--scrape-unlocks` → `token_unlocks`): an unlock < 48h away for either leg triggers **heightened mode** — relaxed z-threshold (−2.0 → −1.5), an UNLOCK ALERT + prompt suffix, state persisted in `bot_state` until unlock + 6h.

## Phase 10 — The 4-cohort Fear & Greed system (v1.0)

Replace the single F&G number with four motivation-based cohorts:

| Cohort | Source | Weight |
|---|---|---|
| Retail | alternative.me F&G | 0.25 |
| Institutional | BTC ETF flows | 0.30 |
| On-chain | CryptoQuant + mempool.space | 0.25 |
| Trader | Qwen's own confidence history | 0.20 |

- Weighted composite = **Synthetic F&G**, computed alongside the Official one; divergence ≥ 15 points fires a ⚡ alert and a prompt note.
- `--fetch-fg` runs on its own 5-minute timer, caches into `macro_cache` and appends to `fg_history` (Grafana-ready) — analysis **never** fetches F&G inline.
- `--set-weights retail=0.3,institutional=0.4,onchain=0.2,trader=0.1` retunes the weights without code edits.

## Phase 11 — Reports and the insight feedback loop (v1.2–v1.4)

- `--report` collects status + outcomes + portfolio data and asks Qwen for a daily recap; `--post-discord` publishes it; `--since DATE` bounds the window.
- `--report --write-insights` asks for **structured pair-level signals** (boost/warn/suppress + magnitude) → `report_insights` with a 6-hour TTL. The analyze cycle reads active insights and applies confidence multipliers, sizing changes, and suppression. Stale rows purge on each report run. The LLM tunes the system that feeds it.
- **Metric honesty fix** (v1.4): the headline win rate is `trade_win_rate` (actioned, closed positions only), not `signal_hit_rate` (which counted unactioned signals and inflated the number). A warning logs if they diverge > 10pp.

## Phase 12 — Discord UX

Iterated three times before it stopped being noise:
- Plain English only — "2.3 std devs below average", "62% historical win rate", conviction levels Low → Very Strong. No raw internals.
- Suppress non-actioned signals entirely; post only opens, closes, and high-confidence blocked signals.
- Close format: P&L in USD first, hold time, entry/exit z, reason emoji (✅/🛑/📰). Wins ≥ 1.5% append a random Ferengi Rule of Acquisition.
- Daily P&L summary: portfolio delta, best/worst position, signal count.

## Phase 13 — Production deployment (systemd)

Oneshot service + timer pairs in `deploy/systemd/` (now `ExecStart=/opt/llm_trader/venv/bin/python3 -m osmia <flag>`):

| Unit | Flag | Schedule |
|---|---|---|
| `AI-Trader_ingest` | `--ingest` | every 5 min |
| `AI-Trader_fg` | `--fetch-fg` | every 5 min |
| `AI-Trader_pairs` | `--analyze` | every 10 min |
| `AI-Trader_fiat` | `--fiat` | every 10 min |
| `AI-Trader_news` | `--scrape-news` | every 30 min |
| `AI-Trader_unlocks` | `--scrape-unlocks` | daily 06:00 |
| `AI-Trader_backup` | `backup_db.py` | 06:00 + 18:00 |

Cutover procedure for a new version: copy script → update `ExecStart=` in all units → `systemctl daemon-reload` → restart units → watch `journalctl -u AI-Trader_pairs -f`. Rollback = repoint to the previous file.

## Phase 14 — Operations

- `--status` — DB health + trade win rate; `--portfolio` — open/closed P&L; `--daily-pnl` — Discord summary.
- `--mine-auto` — polls the unMineable API for new mining payments and injects them into the balance (deduped by coin + payment timestamp, max one per coin per 24h, tracked in `mine_injections`).
- Backups twice daily via `backup_db.py`.

## Build-order gotchas (the things that cost real debugging time)

1. **Backfill before signals** — z-scores on 2 days of data are garbage.
2. **MAD, not stdev** — crypto outliers destroy mean/stdev baselines.
3. **Clamp the z-score** — stale medians manufacture z = −20 phantom signals.
4. **Freshness window** — a dislocation from 6 hours ago is not a signal.
5. **Two-leg fees kill most pair trades** — model routing before trusting any break-even math.
6. **Canonicalize pairs early** — directional duplicates silently split reputation and double trade caps.
7. **Never block analysis on a network fetch** — cache F&G on its own timer; scrape news outside the DB write window.
8. **Everything external WILL go down** — every source needs a degraded path; the LLM fallback is `pass`, not an exception.
9. **Measure the right win rate** — signal hit rate ≠ trade win rate; only the latter belongs in the headline.
