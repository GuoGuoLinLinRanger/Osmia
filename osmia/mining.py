"""Optional mining-income injection (unMineable)."""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import requests

from . import config
from .config import (
    MINE_INJECT_GUARD_H, MINE_PROCEEDS_GUARD_HOURS, UNMINEABLE_BASE_URL,
    UNMINEABLE_COINS, USER_AGENT,
)
from .notify import discord_send
from .portfolio import _snapshot_balances, load_balances
from .store import get_current_prices


def inject_mine_proceeds(con: duckdb.DuckDBPyConnection,
                          amount_usdt: float) -> None:
    # Guard: check last injection
    last = con.execute("""
        SELECT MAX(injected_at) FROM mine_proceeds
    """).fetchone()[0]
    if last is not None:
        if hasattr(last, 'tzinfo') and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        if age_hours < MINE_PROCEEDS_GUARD_HOURS:
            print(f"[mine] Already injected {age_hours:.1f}h ago — skipping (guard: {MINE_PROCEEDS_GUARD_HOURS}h)")
            return

    # Read current USDT balance
    balances       = load_balances()
    balance_before = balances.get('USDT', 0.0)
    balance_after  = balance_before + amount_usdt
    balances['USDT'] = balance_after

    # Persist balances
    p = Path(config.BALANCES_FILE)
    p.write_text(json.dumps(balances, indent=2))

    # Record injection
    now = datetime.now(timezone.utc)
    con.execute("""
        INSERT OR IGNORE INTO mine_proceeds (injected_at, amount_usdt, balance_before, balance_after, note)
        VALUES (?, ?, ?, ?, ?)
    """, [now, amount_usdt, balance_before, balance_after, 'nightly_mine_injection'])

    # Snapshot to fantasy_balances
    prices = get_current_prices(con)
    _snapshot_balances(con, balances, prices)

    print(f"[mine] Injected +{amount_usdt:.2f} USDT  |  {balance_before:.4f} → {balance_after:.4f}")
    discord_send(
        f"⛏ **Mine proceeds injected:** +{amount_usdt:.2f} USDT\n"
        f"New USDT balance: {balance_after:.4f}"
    )

def _unmineable_address_for(coin: str) -> str:
    """Read address from env: UNMINEABLE_{COIN}_ADDRESS (or UNMINEABLE_ADDRESS for single-coin)."""
    addr = os.getenv(f'UNMINEABLE_{coin}_ADDRESS', '')
    if not addr:
        addr = os.getenv('UNMINEABLE_ADDRESS', '')
    return addr.strip()

def scrape_unmineable(con: duckdb.DuckDBPyConnection) -> None:
    """
    For each configured coin: hit the unMineable API, find new 'success' payments
    since last check, inject into USDT balance (max 1 per coin per 24h).
    Addresses must be set in env: UNMINEABLE_{COIN}_ADDRESS (e.g. UNMINEABLE_XEL_ADDRESS).
    """
    any_found = False
    for coin in UNMINEABLE_COINS:
        address = _unmineable_address_for(coin)
        if not address:
            print(f"  [mine-auto] No address for {coin} — set UNMINEABLE_{coin}_ADDRESS in .env")
            continue

        # Step 1: resolve address → account uuid
        try:
            r = requests.get(
                f"{UNMINEABLE_BASE_URL}/address/{address}",
                params={'coin': coin}, timeout=15,
                headers={'User-Agent': USER_AGENT}
            )
            r.raise_for_status()
            data = r.json()
            uuid = (data.get('data') or {}).get('uuid') or (data.get('data') or {}).get('key')
            if not uuid:
                print(f"  [mine-auto] {coin}: could not extract uuid from response", file=sys.stderr)
                continue
        except Exception as e:
            print(f"  [mine-auto] {coin} address lookup failed: {e}", file=sys.stderr)
            continue

        # Step 2: fetch payments
        try:
            r = requests.get(
                f"{UNMINEABLE_BASE_URL}/account/{uuid}/payments",
                params={'page': 1}, timeout=15,
                headers={'User-Agent': USER_AGENT}
            )
            r.raise_for_status()
            payments = ((r.json().get('data') or {}).get('data')
                        or (r.json().get('data') or {}).get('payments')
                        or [])
        except Exception as e:
            print(f"  [mine-auto] {coin} payments fetch failed: {e}", file=sys.stderr)
            continue

        # Guard: check last injection for this coin
        last_inject = con.execute(
            "SELECT MAX(detected_at) FROM mine_injections WHERE coin = ?", [coin]
        ).fetchone()[0]
        if last_inject is not None:
            if hasattr(last_inject, 'tzinfo') and last_inject.tzinfo is None:
                last_inject = last_inject.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - last_inject).total_seconds() / 3600
            if age_h < MINE_INJECT_GUARD_H:
                print(f"  [mine-auto] {coin}: last injection {age_h:.1f}h ago — guard ({MINE_INJECT_GUARD_H}h) still active")
                continue

        for pmt in (payments or []):
            status = pmt.get('status') or pmt.get('state') or ''
            if status.lower() != 'success':
                continue
            amount = float(pmt.get('amount') or pmt.get('value') or 0)
            if amount <= 0:
                continue
            # Normalise payment timestamp
            ts_raw = pmt.get('created_at') or pmt.get('timestamp') or pmt.get('time')
            try:
                if isinstance(ts_raw, (int, float)):
                    payment_ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                elif ts_raw:
                    payment_ts = datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
                else:
                    payment_ts = datetime.now(timezone.utc)
            except Exception:
                payment_ts = datetime.now(timezone.utc)

            injection_id = f"{coin}_{payment_ts.strftime('%Y%m%d%H%M%S')}"
            existing = con.execute(
                "SELECT injection_id FROM mine_injections WHERE injection_id = ?",
                [injection_id]
            ).fetchone()
            if existing:
                continue   # already processed

            # New payment — inject and record
            now = datetime.now(timezone.utc)
            inject_mine_proceeds(con, amount)
            con.execute("""
                INSERT OR IGNORE INTO mine_injections
                    (injection_id, coin, amount, payment_ts, detected_at, injected_usdt, discord_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [injection_id, coin, amount, payment_ts, now, amount, True])
            print(f"  [mine-auto] {coin}: detected new payment {amount:.4f} → injected")
            any_found = True
            break  # one injection per coin per run (guard enforces 24h anyway)

    if not any_found:
        print("  [mine-auto] No new payments found.")
