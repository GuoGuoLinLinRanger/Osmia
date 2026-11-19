"""Paper-trading ledger: sizing, open/close/settle trades, portfolio views."""
import json
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd

from . import config
from .config import (
    DEFAULT_BALANCES, MACRO_MULT_CEIL, MACRO_MULT_FLOOR, MAX_HOLD_HOURS,
    MAX_TRADES_PER_PAIR, TRADE_COOLDOWN_MINUTES, TRADE_PCT_BASE, TRADE_PCT_CEIL,
    TRADE_PCT_FLOOR, TRADE_PCT_REP_BONUS, TRADE_PCT_Z_BONUS,
)
from .core import (
    _trade_id, asset_price, buffett_exit_threshold, fee_breakdown, pair_id,
    stop_loss_threshold, trade_pnl, trade_route,
)
from .notify import discord_trade_closed
from .signals import get_pair_reputations
from .sources.news import check_news_veto
from .store import get_current_prices


def load_balances() -> dict[str, float]:
    p = Path(config.BALANCES_FILE)
    if not p.exists():
        print(f"  balances.json not found — creating template at {config.BALANCES_FILE}")
        p.write_text(json.dumps(DEFAULT_BALANCES, indent=2))
        print("  Edit it with your actual holdings, then re-run.")
        return dict(DEFAULT_BALANCES)
    with open(p) as f:
        data = json.load(f)
    for a in DEFAULT_BALANCES:
        data.setdefault(a, 0.0)
    return data

def signal_weighted_notional(asset: str, price: float,
                              balances: dict[str, float],
                              rep: float | None,
                              z_score: float,
                              override: float | None = None) -> float:
    """
    Dynamic position sizing:
      base_pct    = 5%
      rep_bonus   = rep * 5%   (rep=+1 → +5%, rep=-1 → -5%)
      z_bonus     = min(abs(z)/10, 5%)
    Floor 3%, ceiling 15%.
    """
    if override is not None:
        return override
    holding = balances.get(asset, 0.0)
    value   = holding * price if asset != 'USDT' else holding
    if value <= 0:
        return 1.0

    rep_bonus = (rep if rep is not None else 0.0) * TRADE_PCT_REP_BONUS
    z_bonus   = min(abs(z_score) / 10, TRADE_PCT_Z_BONUS)
    pct       = TRADE_PCT_BASE + rep_bonus + z_bonus
    pct       = max(TRADE_PCT_FLOOR, min(pct, TRADE_PCT_CEIL))
    return max(round(value * pct, 4), 1.0)

def compute_portfolio_value(balances: dict[str, float],
                             prices: dict[str, float]) -> float:
    """Total USD value of all holdings (USDT at 1:1)."""
    total = 0.0
    for asset, qty in balances.items():
        if qty <= 0:
            continue
        px = 1.0 if asset == 'USDT' else prices.get(asset, 0.0)
        total += qty * px
    return total

def macro_confidence_multiplier(fg_synthetic: int, fg_official: int,
                                  inst_cohort: int,
                                  news_sentiment_score: float) -> float:
    """
    Returns a multiplier 0.5x–2.0x applied to base position size.
    Extreme fear + divergence + institutional accumulation → size up.
    Extreme greed + bad news → pull back.
    """
    multiplier = 1.0

    if fg_synthetic <= 20:
        multiplier += 0.5
    elif fg_synthetic <= 40:
        multiplier += 0.25
    elif fg_synthetic >= 85:
        multiplier -= 0.5
    elif fg_synthetic >= 70:
        multiplier -= 0.25

    divergence = abs(fg_synthetic - fg_official)
    if divergence >= 25:
        multiplier += 0.3
    elif divergence >= 15:
        multiplier += 0.15

    if inst_cohort <= 30:
        multiplier += 0.2
    elif inst_cohort >= 70:
        multiplier -= 0.2

    if news_sentiment_score >= 0.5:
        multiplier += 0.1
    elif news_sentiment_score <= -0.5:
        multiplier -= 0.2

    return max(MACRO_MULT_FLOOR, min(MACRO_MULT_CEIL, multiplier))

def _snapshot_balances(con, balances, prices) -> None:
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = [{'timestamp': ts, 'asset_id': a,
             'qty': q, 'price_usd': prices.get(a, 1.0) if a != 'USDT' else 1.0,
             'value_usd': q * (prices.get(a, 1.0) if a != 'USDT' else 1.0)}
            for a, q in balances.items()]
    df = pd.DataFrame(rows)
    con.execute("""
        INSERT OR IGNORE INTO fantasy_balances (timestamp, asset_id, qty, price_usd, value_usd)
        SELECT timestamp, asset_id, qty, price_usd, value_usd FROM df
    """)

def _pair_on_cooldown(con, sell_asset: str, buy_asset: str) -> bool:
    cid = pair_id(sell_asset, buy_asset)
    result = con.execute(f"""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE canonical_pair_id = ?
          AND timestamp >= NOW() - INTERVAL {TRADE_COOLDOWN_MINUTES} MINUTES
          AND status = 'open'
    """, [cid]).fetchone()
    return result[0] > 0

def _pair_at_max_trades(con, sell_asset: str, buy_asset: str) -> bool:
    cid = pair_id(sell_asset, buy_asset)
    result = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE canonical_pair_id = ? AND status = 'open'
    """, [cid]).fetchone()
    return result[0] >= MAX_TRADES_PER_PAIR

def _close_trade(con, trade_id: str, cur_px: float,
                 notional: float, buy_qty: float,
                 reason: str, sell_qty: float = 0.0,
                 sell_px_now: float = 1.0) -> tuple[float, float]:
    ts               = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pnl_usd, pnl_pct = trade_pnl(sell_qty, sell_px_now, buy_qty, cur_px, notional)
    con.execute("""
        UPDATE fantasy_trades
        SET status='closed', close_ts=?, close_reason=?,
            close_buy_px=?, pnl_usd=?, pnl_pct=?
        WHERE trade_id=?
    """, [ts, reason, cur_px, pnl_usd, pnl_pct, trade_id])
    return pnl_usd, pnl_pct

def _get_latest_fantasy_balances(con) -> dict[str, float]:
    """Return the most recent fantasy_balances snapshot as {asset: qty}."""
    rows = con.execute("""
        SELECT asset_id, qty FROM fantasy_balances
        WHERE timestamp = (SELECT MAX(timestamp) FROM fantasy_balances)
    """).fetchall()
    if rows:
        return {r[0]: float(r[1]) for r in rows}
    return load_balances()

def _snapshot_balances_upsert(con, balances: dict[str, float],
                               prices: dict[str, float]) -> None:
    """Like _snapshot_balances but uses ON CONFLICT DO UPDATE so close-time
    snapshots within the same minute always reflect the latest balance."""
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    for a, q in balances.items():
        px  = 1.0 if a == 'USDT' else prices.get(a, 0.0)
        val = q * px
        con.execute("""
            INSERT INTO fantasy_balances (timestamp, asset_id, qty, price_usd, value_usd)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (timestamp, asset_id) DO UPDATE SET
                qty       = excluded.qty,
                price_usd = excluded.price_usd,
                value_usd = excluded.value_usd
        """, [ts, a, q, px, val])

def _auto_close_trades(con: duckdb.DuckDBPyConnection,
                       prices: dict[str, float],
                       current_z_scores: dict[str, float] | None = None,
                       fg_value: int = 50,
                       etf_flow_m: float | None = None,
                       fg_synthetic: int | None = None) -> int:
    """
    Auto-close open fantasy trades on four conditions:
    1. Mean reversion: P&L >= 1.5x break-even cost
    2. Z crossback: current Z-score > buffett_exit_threshold(fg_value, etf_flow_m, fg_synthetic)
    3. Expiry: trade has been open > MAX_HOLD_HOURS
    4. News veto: veto keyword found in buy_asset news last 6h
    Exits logged to exit_log table.
    """
    open_trades = con.execute("""
        SELECT trade_id, sell_asset, buy_asset, sell_qty, buy_qty,
               sell_price_usd, buy_price_usd, notional_usd, total_cost_usd,
               signal_z, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    closed          = 0
    _closed_details = []   # v1.2: track for USDT settlement
    now             = datetime.now(timezone.utc)
    z_thresh        = buffett_exit_threshold(fg_value, etf_flow_m, fg_synthetic)
    sl_thresh       = stop_loss_threshold(fg_value)

    for _, t in open_trades.iterrows():
        cur_px  = asset_price(prices, t['buy_asset'], t['buy_price_usd'])
        sell_now = asset_price(prices, t['sell_asset'], t['sell_price_usd'])
        _, pnl_pct = trade_pnl(t['sell_qty'], sell_now, t['buy_qty'], cur_px,
                               t['notional_usd'])
        be_pct  = t['total_cost_usd'] / t['notional_usd'] * 100 if t['notional_usd'] else 0

        reason     = None
        news_vetod = False
        is_stop    = False
        # Signal pair is BUY/SELL (the bought asset was the cheap base)
        cpair      = f"{t['buy_asset']}/{t['sell_asset']}"

        # 0. Stop-loss — first check, overrides everything
        if pnl_pct <= sl_thresh:
            reason  = "stop_loss"
            is_stop = True

        # 1. Mean reversion
        elif pnl_pct >= be_pct * 1.5:
            reason = "reverted"

        # 2. Z crossback — dynamic Buffett threshold
        elif current_z_scores and cpair in current_z_scores:
            if current_z_scores[cpair] > z_thresh:
                reason = "z_crossback"

        # 3. Expiry — independent if: must fire even when pair IS in z_scores but
        #    z hasn't crossed threshold (elif would skip it in that case)
        if reason is None and t['timestamp']:
            age_hours = (now - t['timestamp'].to_pydatetime()).total_seconds() / 3600
            if age_hours > MAX_HOLD_HOURS:
                reason = f"expired_{MAX_HOLD_HOURS}h"

        # 4. News veto
        if reason is None:
            if check_news_veto(con, t['buy_asset']):
                reason     = "news_veto"
                news_vetod = True

        if reason:
            pnl_usd, pnl_pct_final = _close_trade(
                con, t['trade_id'], cur_px,
                t['notional_usd'], t['buy_qty'], reason,
                sell_qty=t['sell_qty'], sell_px_now=sell_now,
            )
            entry_z = float(t['signal_z']) if t['signal_z'] is not None else None
            exit_z  = current_z_scores.get(cpair) if current_z_scores else None
            discord_trade_closed(
                str(t['sell_asset']), str(t['buy_asset']),
                pnl_usd, pnl_pct_final, reason,
                open_ts=t['timestamp'], entry_z=entry_z, exit_z=exit_z,
                fg_value=fg_value, sl_thresh=sl_thresh if is_stop else None,
            )
            exit_id = f"{t['trade_id']}_exit"
            con.execute("""
                INSERT OR IGNORE INTO exit_log
                    (exit_id, timestamp, trade_id, buy_asset, pnl_usd, pnl_pct,
                     reason, fg_value, etf_flow_m, z_threshold, news_veto, stop_loss)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [exit_id, now, t['trade_id'], t['buy_asset'],
                  pnl_usd, pnl_pct_final, reason, fg_value, etf_flow_m,
                  z_thresh, news_vetod, is_stop])
            # v1.2: credit USDT proceeds for this closed trade
            _closed_details.append({
                'buy_asset': str(t['buy_asset']),
                'buy_qty':   float(t['buy_qty']),
                'cur_px':    cur_px,
            })
            closed += 1

    _settle_closed_to_usdt(con, _closed_details, prices)
    return closed

def _settle_closed_to_usdt(con, closed_details: list[dict],
                           prices: dict[str, float]) -> None:
    """v1.2: credit each closed position's proceeds to the USDT fantasy balance."""
    if not closed_details:
        return
    balances = _get_latest_fantasy_balances(con)
    for cd in closed_details:
        proceeds = cd['buy_qty'] * cd['cur_px']
        balances['USDT'] = balances.get('USDT', 0.0) + proceeds
        balances[cd['buy_asset']] = max(
            0.0, balances.get(cd['buy_asset'], 0.0) - cd['buy_qty']
        )
        if (cd['cur_px'] > 0
                and balances.get(cd['buy_asset'], 0.0) * cd['cur_px'] < 0.01):
            balances[cd['buy_asset']] = 0.0
    _snapshot_balances_upsert(con, balances, prices)

def execute_fantasy_pair_trade(con, sell_asset, buy_asset,
                                sell_px, buy_px, notional,
                                spread_pct, z_score, confidence,
                                balances,
                                position_multiplier: float = 1.0) -> dict | None:
    holding_usd = balances.get(sell_asset, 0.0) * sell_px
    if holding_usd < notional * 0.5:
        return None
    _, legs  = trade_route(sell_asset, buy_asset)
    fees     = fee_breakdown(notional, legs, spread_pct)
    sell_qty = notional / sell_px  if sell_px  else 0
    buy_qty  = (notional - fees['total_cost']) / buy_px if buy_px else 0
    balances[sell_asset] = max(0.0, balances.get(sell_asset, 0.0) - sell_qty)
    balances[buy_asset]  = balances.get(buy_asset, 0.0) + buy_qty
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{sell_asset}/{buy_asset}"
    return {
        'trade_id': _trade_id(ts, pair), 'timestamp': ts,
        'mode': 'pair', 'action': 'pair_reversion',
        'sell_asset': sell_asset, 'buy_asset': buy_asset,
        'sell_qty': round(sell_qty, 8), 'buy_qty': round(buy_qty, 8),
        'sell_price_usd': sell_px, 'buy_price_usd': buy_px,
        'notional_usd': fees['notional'], 'fee_usd': fees['total_fee'],
        'spread_usd': fees['spread_cost'], 'total_cost_usd': fees['total_cost'],
        'signal_z': z_score, 'confidence': confidence, 'status': 'open',
        'canonical_pair_id': pair_id(sell_asset, buy_asset),
        'position_multiplier': round(position_multiplier, 4),
    }

def execute_fantasy_fiat_trade(con, buy_asset, buy_px, notional,
                                spread_pct, z_score, confidence,
                                balances) -> dict | None:
    usdt = balances.get('USDT', 0.0)
    if usdt < notional * 0.5:
        return None
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = (notional - fees['total_cost']) / buy_px if buy_px else 0
    balances['USDT']    = max(0.0, usdt - notional)
    balances[buy_asset] = balances.get(buy_asset, 0.0) + buy_qty
    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    pair = f"{buy_asset}/USDT"
    return {
        'trade_id': _trade_id(ts, pair), 'timestamp': ts,
        'mode': 'fiat', 'action': 'fiat_buy',
        'sell_asset': 'USDT', 'buy_asset': buy_asset,
        'sell_qty': round(notional, 6), 'buy_qty': round(buy_qty, 8),
        'sell_price_usd': 1.0, 'buy_price_usd': buy_px,
        'notional_usd': fees['notional'], 'fee_usd': fees['total_fee'],
        'spread_usd': fees['spread_cost'], 'total_cost_usd': fees['total_cost'],
        'signal_z': z_score, 'confidence': confidence, 'status': 'open',
        'canonical_pair_id': pair_id('USDT', buy_asset),
        'position_multiplier': 1.0,
    }

def _insert_trade(con, trade: dict) -> None:
    con.execute("""
        INSERT OR IGNORE INTO fantasy_trades
            (trade_id, timestamp, mode, action, sell_asset, buy_asset,
             sell_qty, buy_qty, sell_price_usd, buy_price_usd, notional_usd,
             fee_usd, spread_usd, total_cost_usd, signal_z, confidence, status,
             canonical_pair_id, position_multiplier)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        trade['trade_id'], trade['timestamp'], trade['mode'], trade['action'],
        trade['sell_asset'], trade['buy_asset'],
        trade['sell_qty'], trade['buy_qty'],
        trade['sell_price_usd'], trade['buy_price_usd'],
        trade['notional_usd'], trade['fee_usd'], trade['spread_usd'],
        trade['total_cost_usd'], trade['signal_z'], trade['confidence'],
        trade['status'],
        trade.get('canonical_pair_id', pair_id(trade['sell_asset'], trade['buy_asset'])),
        trade.get('position_multiplier', 1.0),
    ])

def show_portfolio(con: duckdb.DuckDBPyConnection, full: bool = False) -> None:
    prices = get_current_prices(con)
    if full:
        print("── Fantasy Portfolio (full history) ──")
        where = ""
    else:
        print("── Fantasy Portfolio (last 7 days) ──")
        where = "WHERE timestamp >= NOW() - INTERVAL 7 DAY"

    trades = con.execute(f"""
        SELECT trade_id, timestamp, action, sell_asset, buy_asset,
               sell_qty, buy_qty, sell_price_usd, buy_price_usd,
               notional_usd, total_cost_usd, signal_z, confidence,
               status, close_reason, pnl_usd, pnl_pct
        FROM fantasy_trades {where} ORDER BY timestamp DESC
    """).df()

    if trades.empty:
        print("  No fantasy trades executed yet.")
        return

    # Reputation is keyed by signal pair BUY/SELL (the bought asset was the cheap base)
    all_pairs = [f"{r['buy_asset']}/{r['sell_asset']}" for _, r in trades.iterrows()]
    reps      = get_pair_reputations(con, list(set(all_pairs)))

    rows_out = []
    for _, t in trades.iterrows():
        pair = f"{t['sell_asset']}/{t['buy_asset']}"
        signal_pair = f"{t['buy_asset']}/{t['sell_asset']}"
        if t['status'] == 'open':
            pnl_usd, pnl_pct = trade_pnl(
                t['sell_qty'], asset_price(prices, t['sell_asset'], t['sell_price_usd']),
                t['buy_qty'], asset_price(prices, t['buy_asset'], t['buy_price_usd']),
                t['notional_usd'])
            close_r     = "-"
        else:
            pnl_usd = t['pnl_usd'] or 0.0
            pnl_pct = t['pnl_pct'] or 0.0
            close_r = t['close_reason'] or "-"

        rep = reps.get(signal_pair)
        rows_out.append({
            'ts':       str(t['timestamp'])[:16],
            'trade':    f"{t['sell_asset']}→{t['buy_asset']}",
            'pos':      f"{t['notional_usd']:.2f}",
            'z':        f"{t['signal_z']:+.3f}",
            'conf':     t['confidence'],
            'rep':      f"{rep:+.3f}" if rep is not None else "n/a",
            'status':   t['status'],
            'reason':   close_r,
            'pnl_usd':  f"{pnl_usd:+.4f}",
            'pnl_pct':  f"{pnl_pct:+.3f}%",
        })

    print(pd.DataFrame(rows_out).to_string(index=False))

    open_t   = [r for r in rows_out if r['status'] == 'open']
    closed_t = [r for r in rows_out if r['status'] == 'closed']
    realized = sum(float(r['pnl_usd']) for r in closed_t)
    total    = sum(float(r['pnl_usd']) for r in rows_out)
    print(f"\n  Open positions : {len(open_t)}")
    print(f"  Closed         : {len(closed_t)}")
    print(f"  Realized P&L   : {realized:+.4f} USDT")
    print(f"  Total P&L      : {total:+.4f} USDT (incl. unrealized)")

    latest_bal = con.execute("""
        SELECT asset_id, qty, price_usd, value_usd FROM fantasy_balances
        WHERE timestamp = (SELECT MAX(timestamp) FROM fantasy_balances)
        ORDER BY value_usd DESC
    """).df()
    if not latest_bal.empty:
        total_val = latest_bal['value_usd'].sum()
        print(f"\n── Fantasy Balances ──")
        print(latest_bal[latest_bal['qty'] > 0].to_string(index=False))
        print(f"\n  Total portfolio value : {total_val:.4f} USDT")

def run_settle_dust(con: duckdb.DuckDBPyConnection) -> None:
    """Convert all sub-$0.10 non-USDT fantasy balances to USDT at current price."""
    prices   = get_current_prices(con)
    balances = _get_latest_fantasy_balances(con)
    settled  = 0

    for asset in list(balances.keys()):
        if asset == 'USDT':
            continue
        qty = balances.get(asset, 0.0)
        if qty <= 0:
            continue
        px  = prices.get(asset, 0.0)
        if px <= 0:
            continue
        val = qty * px
        if val < 0.10:
            balances['USDT'] = balances.get('USDT', 0.0) + val
            balances[asset]  = 0.0
            settled += 1
            print(f"  Settled dust: {qty:.8f} {asset}  (${val:.4f}) → USDT")

    if settled:
        _snapshot_balances_upsert(con, balances, prices)
        print(f"  Settled {settled} dust position(s). New USDT: {balances.get('USDT', 0.0):.4f}")
    else:
        print("  No dust positions found (all non-USDT balances >= $0.10)")
