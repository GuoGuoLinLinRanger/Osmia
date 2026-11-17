# Osmia — an LLM-in-the-loop Crypto Trading Monitor
### Presentation / talk track

---

## Slide 1 — What is it?

**A self-hosted system that trades crypto on paper, 24/7, by combining statistical anomaly detection with a locally hosted LLM — and grades its own decisions to get better.**

- Ingests live Binance data, finds statistical dislocations, asks a local Qwen model for a verdict, executes paper trades, and reports everything to Discord.
- 100% self-hosted: the LLM runs on a machine on the LAN; no cloud AI, no exchange API keys, no real orders.
- Co-built, iteratively, in 13 documented versions — from a 423-line prototype to a ~5,000-line production system.

---

## Slide 2 — The problem it solves

- Crypto markets move 24/7; a human can't watch 14 assets and ~90 cross-pairs continuously.
- Pure statistical screeners have no context — they can't read "SEC sues X" or "token unlock tomorrow."
- Pure LLM trading has no statistical discipline — it hallucinates conviction.
- **Osmia's bet: use statistics to find *candidates* and the LLM to *judge* them, with hard-coded risk rails around both.**

---

## Slide 3 — Architecture

```
 Binance (ccxt)      CoinGecko · CoinDesk RSS · Reddit · DeFiLlama
      │               alternative.me · ETF flows · CryptoQuant
      ▼                              │
 ┌─────────┐    ┌──────────────────────────────────┐
 │ INGEST  │    │   CONTEXT ENGINE                 │
 │ 5 min   │    │   4-cohort Fear&Greed, news,     │
 └────┬────┘    │   unlock calendar, macro block   │
      ▼         └──────────────┬───────────────────┘
 ┌──────────────────┐          │
 │ DuckDB (1 file,  │          ▼
 │ 15 tables)       │   ┌──────────────┐   ┌──────────────┐
 └────┬─────────────┘   │ SIGNAL ENGINE │──▶│  LOCAL QWEN  │
      │                 │ MAD z-scores, │   │  (LAN, JSON  │
      │                 │ rank top 3    │   │   verdicts)  │
      │                 └──────┬───────┘   └──────┬───────┘
      │                        ▼                  │
      │                 ┌─────────────────────────▼──────┐
      │                 │ SIZING & RISK RAILS            │
      │                 │ confidence ≥6, macro 0.5–2.0×, │
      │                 │ order-book depth, $10 floor    │
      │                 └──────┬─────────────────────────┘
      │                        ▼
      │                 ┌──────────────┐    ┌─────────┐
      └────────────────▶│ PAPER TRADES │───▶│ DISCORD │
                        │ + exits      │    └─────────┘
                        └──────┬───────┘
                               ▼
                  ┌────────────────────────────┐
                  │ LEARNING LOOPS             │
                  │ outcome scoring → pair     │
                  │ reputation → ranking/sizing│
                  │ LLM report → insights (6h) │
                  └────────────────────────────┘
```

One Python file, one DuckDB file, seven systemd timers.

---

## Slide 4 — How a trade happens (the 10-minute cycle)

1. **Ingest** — tickers for 14 assets every 5 min → minute-keyed rows with VWAP ratio, spread, MAD volume z-score.
2. **Screen** — robust (median/MAD) z-score on every cross-pair ratio; require z ≤ −2.0, still live in the last 2 hours, ≥ 20 samples. Rank by rule-based score; top 3 advance.
3. **Contextualize** — prepend macro block (synthetic F&G, BTC dominance), last-6h headlines per leg, unlock alerts.
4. **Judge** — local Qwen returns one JSON line: `action / top_pair / rank / thesis / risks`. Unreachable model = automatic "pass."
5. **Gate & size** — confidence ≥ 6/10; base 12% of balance ± reputation/z bonuses (7–30%), × macro multiplier (0.5–2.0), order-book liquidity check, $10 minimum, expected profit ≥ $1 after fees (two-leg routing fees modeled).
6. **Execute on paper** — log trade, post a plain-English Discord card.
7. **Manage** — every cycle: sentiment-adaptive stop-loss (−8% to −18%), Buffett inverse-F&G take-profit threshold, news-keyword veto, per-position LLM re-review (exit urgency ≥ 7 = force close), 48h max hold.
8. **Learn** — 24h later the verdict is graded against the realized move; the grade updates pair reputation, which changes future ranking and sizing.

---

## Slide 5 — What makes it interesting (the original parts)

1. **Closed learning loop** — every LLM verdict is scored against reality; win rates feed back into candidate ranking and position sizing. The system's opinion of a pair is earned, not configured.
2. **LLM self-tuning** — a Qwen-written performance report emits structured boost/warn/suppress directives (6h TTL) that the next analysis cycles obey.
3. **4-cohort synthetic Fear & Greed** — Retail (0.25) + Institutional ETF flows (0.30) + On-chain (0.25) + the bot's own trader confidence (0.20); divergence ≥ 15 points from the official index is itself a signal.
4. **Sentiment-adaptive risk** — stop-losses and take-profit thresholds move with the fear/greed regime instead of being fixed numbers.
5. **Defensive engineering as a feature** — ~10 external dependencies, every one with a degraded path; the LLM's failure mode is "pass," never a crash. Uptime is part of the strategy.

---

## Slide 6 — The numbers

**Codebase**
- ~5,000 lines of production Python, 132 functions, one file by design
- 13 versions (v0.2 → v1.4), each with a documented changelog; 12× growth from the 423-line prototype
- 27 CLI subcommands on one argparse interface
- 15-table DuckDB schema, migrated in place (zero migration scripts)

**Scale of operation**
- 14 assets, 32 direct Binance cross-pairs + 2-leg routing for the rest
- ~288 ingest runs + ~288 F&G fetches + ~144 LLM analysis cycles per day
- 90 days × 1-minute OHLCV backfill as the statistical baseline
- ~10 external data sources integrated (Binance, CoinGecko, CoinDesk, Reddit, DeFiLlama, alternative.me, ETF flows, CryptoQuant, mempool.space, unMineable)

**Risk parameters (all enforced in code)**
- Entry: z ≤ −2.0 · confidence ≥ 6/10 · 60-min pair cooldown · max 3 open per pair
- Sizing: 7–30% of balance, macro-multiplied 0.5–2.0×, $10 floor, $1 expected-profit gate
- Exits: stop-loss −8%→−18% by sentiment regime · news veto · LLM exit urgency · 48h max hold

**Production**
- 7 systemd services + timers, twice-daily automated DB backups
- [FILL FROM `--status` / `--report`]: total paper trades, trade win rate, cumulative P&L, market_history row count, days of continuous operation

---

## Slide 7 — Lessons learned

- **Robust statistics or nothing**: mean/stdev z-scores were unusable in crypto; median/MAD with a clamp at |z|=8 fixed phantom −20σ "signals."
- **Fees decide pair trading**: most 2-leg trades are unprofitable before they start; break-even math must include routing.
- **Measure the honest metric**: signal hit-rate flattered the system vs. actual closed-trade win rate — v1.4 made trade win rate the only headline number.
- **The LLM is a judge, not an oracle**: it ranks pre-screened candidates and reads news; the statistics and risk rails make the decisions defensible.
- **Iterate in production**: every version was driven by a deficiency observed live (Discord noise, dust balances, thin-book slippage, inflated metrics) — not speculative features.

---

## Slide 8 — Stack summary

Python 3 · ccxt · DuckDB · numpy/pandas · feedparser · self-hosted Qwen (OpenAI-compatible API) · systemd · Discord webhooks · Grafana-ready F&G history · Linux home-server deployment
