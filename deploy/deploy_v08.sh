#!/usr/bin/env bash
# Full install/repoint of all AI-Trader units.
# Run with sudo:  ! sudo bash /opt/llm_trader/deploy/deploy_v08.sh
#
# Copies all unit files from deploy/systemd/, repoints existing units to the
# current script version, enables any new timers, and fires a first --analyze run.
set -euo pipefail

SCRIPT=TraderV13.py
SRC=/opt/llm_trader/deploy/systemd
UNITS=/etc/systemd/system
STAMP=$(date +%Y%m%d_%H%M%S)

echo "== 0. sanity =="
test -f "/opt/llm_trader/$SCRIPT" || { echo "$SCRIPT missing"; exit 1; }
/opt/llm_trader/venv/bin/python3 -m py_compile "/opt/llm_trader/$SCRIPT" && echo "  $SCRIPT compiles OK"

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
