# Osmia trader — systemd units

All units run as a dedicated `osmia` user from `/opt/llm_trader` using `venv/bin/python3`.
Production currently runs **TraderV13.py**.

## Unit summary

| Unit                   | Command                  | Schedule         |
|------------------------|--------------------------|------------------|
| `AI-Trader_ingest`     | `--ingest`               | every 5 min      |
| `AI-Trader_fg`         | `--fetch-fg`             | every 5 min      |
| `AI-Trader_pairs`      | `--analyze`              | every 10 min     |
| `AI-Trader_fiat`       | `--fiat`                 | every 10 min     |
| `AI-Trader_news`       | `--scrape-news`          | every 30 min     |
| `AI-Trader_unlocks`    | `--scrape-unlocks`       | daily 06:00      |
| `AI-Trader_backup`     | `backup_db.py`           | daily 06:00 + 18:00 |
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
  AI-Trader_backup.timer \n  AI-Trader_outcomes.timer \n  AI-Trader_insights.timer
```

## Bumping to a new script version

The units hardcode the script filename. Bumping the file alone changes nothing
in production. To repoint all units at once:

```bash
NEW=TraderV12.py
OLD=TraderV11.py
sudo sed -i "s#$OLD#$NEW#" /etc/systemd/system/AI-Trader_*.service
sudo systemctl daemon-reload
sudo systemctl restart AI-Trader_ingest AI-Trader_pairs AI-Trader_fiat \
  AI-Trader_news AI-Trader_unlocks AI-Trader_fg
# also update this README and the deploy script
```

Then update `ExecStart` in the source files under `deploy/systemd/` to keep
this directory in sync with production.

## Rollback

```bash
OLD=TraderV10.py
NEW=TraderV11.py
sudo sed -i "s#$NEW#$OLD#" /etc/systemd/system/AI-Trader_*.service
sudo systemctl daemon-reload
sudo systemctl restart AI-Trader_ingest AI-Trader_pairs AI-Trader_fiat \
  AI-Trader_news AI-Trader_unlocks AI-Trader_fg
```

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
