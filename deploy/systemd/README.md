# Osmia trader — systemd units

All units run as a dedicated `osmia` user from the repo checkout at `/opt/llm_trader`,
with `OSMIA_HOME=/opt/llm_trader`, as `venv/bin/python3 -m osmia --<job>`.
Install or update them all with `sudo bash deploy/install_systemd.sh`.

## Unit summary

| Unit                   | Command                  | Schedule         |
|------------------------|--------------------------|------------------|
| `AI-Trader_ingest`     | `--ingest`               | every 5 min      |
| `AI-Trader_fg`         | `--fetch-fg`             | every 5 min      |
| `AI-Trader_pairs`      | `--analyze`              | every 10 min     |
| `AI-Trader_fiat`       | `--fiat`                 | every 10 min     |
| `AI-Trader_news`       | `--scrape-news`          | every 30 min     |
| `AI-Trader_unlocks`    | `--scrape-unlocks`       | daily 06:00      |
| `AI-Trader_backup`     | `--backup`               | daily 06:00 + 18:00 |
| `AI-Trader_outcomes`   | `--outcomes`             | hourly           |
| `AI-Trader_insights`   | `--report --write-insights` | every 6 h     |

`--scrape-news` also auto-runs at the head of every `--analyze`, so the news
timer is a top-up, not a hard dependency.

## Fresh install

```bash
SRC=/opt/llm_trader/deploy/systemd
DEST=/etc/systemd/system

sudo cp "$SRC"/AI-Trader_*.service "$SRC"/AI-Trader_*.timer "$DEST"/
sudo systemctl daemon-reload
sudo systemctl enable --now \
  AI-Trader_ingest.timer \
  AI-Trader_fg.timer \
  AI-Trader_pairs.timer \
  AI-Trader_fiat.timer \
  AI-Trader_news.timer \
  AI-Trader_unlocks.timer \
  AI-Trader_backup.timer \
  AI-Trader_outcomes.timer \
  AI-Trader_insights.timer
```

## Updating to new code

The units run the `osmia` package from the checkout, so an update is a pull plus
a restart; no unit edits are needed:

```bash
cd /opt/llm_trader
sudo -u osmia git pull
sudo -u osmia venv/bin/pip install -e .     # only if dependencies changed
sudo systemctl daemon-reload                # only if unit files changed (re-run install_systemd.sh)
```

Timers pick up the new code on their next tick.

## Rollback

```bash
cd /opt/llm_trader
sudo -u osmia git log --oneline -5          # pick the last good commit
sudo -u osmia git checkout <commit>
```

To return to the pre-package v1.4 script, check out `history/TraderV13.py`'s
parent commit and re-run its own deploy script.

## Notes / known limits

- **Reddit returns HTTP 403** from this IP regardless of User-Agent (datacenter
  block). The Reddit news source degrades to zero rows; CoinDesk + DeFiLlama TVL
  still work.
- **Token unlocks**: DeFiLlama's paid `api.llama.fi/emissions` endpoint (402) is
  avoided; the free `defillama-datasets.llama.fi/emissions/<slug>` dataset is used
  instead. Only **APT** has a slug there — **ATOM** has no free unlock source and
  is skipped silently (Cosmos Hub uses perpetual staking inflation, not vesting
  cliffs, so the signal is weak anyway).
- **CryptoRank vesting** (`GET /v2/currencies/<id>/vesting`, ATOM id `6834`) exists
  but is gated behind a higher tariff (403 on current key). If the plan is upgraded,
  add `ATOM: 6834` to the slug map and parse the `/vesting` payload.
