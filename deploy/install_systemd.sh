#!/usr/bin/env bash
# Install or update all Osmia systemd units (home-server deployment).
# Run with sudo from the repo checkout:  sudo bash /opt/llm_trader/deploy/install_systemd.sh
#
# Copies every unit from deploy/systemd/, enables all timers, and fires a first
# --analyze run. Container/cloud hosts should use `python -m osmia --run` instead.
set -euo pipefail

HOME_DIR=/opt/llm_trader
PY=$HOME_DIR/venv/bin/python3
SRC=$HOME_DIR/deploy/systemd
UNITS=/etc/systemd/system

echo "== 0. sanity =="
cd "$HOME_DIR"
OSMIA_HOME=$HOME_DIR "$PY" -c "import osmia.cli" && echo "  osmia package imports OK"

echo "== 1. copy all unit files =="
cp "$SRC"/AI-Trader_*.service "$UNITS"/
cp "$SRC"/AI-Trader_*.timer   "$UNITS"/
echo "  copied $(ls "$SRC"/AI-Trader_*.service | wc -l) services and $(ls "$SRC"/AI-Trader_*.timer | wc -l) timers"

echo "== 2. reload + enable all timers =="
systemctl daemon-reload
systemctl enable --now \
  AI-Trader_ingest.timer \
  AI-Trader_fg.timer \
  AI-Trader_pairs.timer \
  AI-Trader_fiat.timer \
  AI-Trader_news.timer \
  AI-Trader_unlocks.timer \
  AI-Trader_backup.timer \
  AI-Trader_outcomes.timer \
  AI-Trader_insights.timer
echo "  active timers:"
systemctl list-timers 'AI-Trader*' --no-pager | sed 's/^/    /'

echo "== 3. first run (pairs, synchronous) =="
systemctl start AI-Trader_pairs.service || true
echo "  pairs result: $(systemctl show -p Result --value AI-Trader_pairs.service)"

echo "== 4. first-run journal =="
journalctl -u AI-Trader_pairs.service -n 60 --no-pager --since "2 min ago"

echo "== done. rollback steps are in deploy/systemd/README.md =="
