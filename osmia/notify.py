"""Discord notifications."""
import math
import random
import sys
from datetime import datetime, timezone

import duckdb
import requests

from . import config
from .config import DIVERGENCE_THRESHOLD, MAX_TRADES_PER_PAIR, TRADE_COOLDOWN_MINUTES
from .core import _trade_signal_label, asset_price, fg_classification, trade_pnl


FERENGI_RULES: list[tuple[int, str]] = [
    (3,   "Never spend more for an acquisition than you have to."),
    (9,   "Opportunity plus instinct equals profit."),
    (41,  "Profit is its own reward."),
    (57,  "Good customers are as rare as latinum — treasure them."),
    (74,  "Knowledge equals profit."),
    (102, "Nature decays, but latinum lasts forever."),
    (109, "Dignity and an empty sack is worth the sack."),
    (125, "You can't make a deal if you're dead."),
    (162, "Even in the worst of times, someone turns a profit."),
    (189, "Let others keep their reputation. You keep their money."),
    (211, "Employees are the rungs on the ladder of success."),
    (239, "Never be afraid to mislabel a product."),
    (284, "Deep down, everyone's a Ferengi."),
    (299, "After you've exploited someone, it never hurts to thank them."),
]

def _ferengi_quote(pnl_pct: float) -> str:
    """Return a Ferengi rule footer when pnl_pct >= 1.5%, else empty string."""
    if pnl_pct < 1.5:
        return ""
    num, rule = random.choice(FERENGI_RULES)
    return f"\n━━━━━━━━━━━━━━━━━━━━━━\n⚖️ Rule of Acquisition #{num}: \"{rule}\""

def discord_send(message: str) -> None:
    if not config.DISCORD_WEBHOOK:
        return
    try:
        # allowed_mentions: model-written text must never ping @everyone/@here/users
        requests.post(config.DISCORD_WEBHOOK,
                      json={"content": message[:2000],
                            "allowed_mentions": {"parse": []}},
                      timeout=10)
    except Exception as e:
        print(f"  Discord send failed: {e}", file=sys.stderr)

def discord_trade_open_pair(trade: dict, top_row,
                             verdict: dict, fg_data: dict,
                             macro_mult: float = 1.0) -> None:
    """Post compact trade-open card for a pair trade. Only called when a trade executes."""
    sell     = trade['sell_asset']
    buy      = trade['buy_asset']
    z        = trade['signal_z']
    conf     = trade['confidence']
    notional = trade['notional_usd']
    sell_px  = trade['sell_price_usd']
    buy_px   = trade['buy_price_usd']
    fee_usd  = trade['total_cost_usd']
    be_pct   = fee_usd / notional * 100 if notional else 0
    target_pct = abs(z) * be_pct

    rep     = top_row.get('reputation')
    rep_str = (f"{rep:+.3f}" if rep is not None
               and not (isinstance(rep, float) and math.isnan(rep))
               else "+0.000 (new)")

    mult_str = f" ({macro_mult:.2f}x mult)" if abs(macro_mult - 1.0) > 0.01 else ""

    fg_off   = fg_data.get('fg_official', 50)
    fg_synth = fg_data.get('fg_synthetic', fg_off)
    bolt     = "⚡" if abs(fg_synth - fg_off) >= DIVERGENCE_THRESHOLD else ""
    fg_line  = (f"📊 F&G: {fg_off}→{fg_synth}{bolt} | "
                f"Retail:{fg_data.get('fg_retail', fg_off)} "
                f"Inst:{fg_data.get('fg_institutional', 50)} "
                f"Chain:{fg_data.get('fg_onchain', 50)} "
                f"Trade:{conf}")

    thesis_raw = verdict.get('thesis') or ''
    thesis = thesis_raw.split('.')[0].strip()

    msg = (
        f"🟢 **TRADE OPEN — {sell} → {buy}**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Size: {notional:.2f} USDT{mult_str} | Z: {z:+.1f} | Conf: {conf}/10\n"
        f"Entry: {sell} @ {sell_px:,.4f} | {buy} @ {buy_px:,.4f}\n"
        f"Target: +{target_pct:.2f}% | Break-even: +{be_pct:.3f}%\n"
        f"Rep: {rep_str} | Est. fee: {fee_usd:.4f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{fg_line}\n"
        f"💡 {thesis}"
    )
    discord_send(msg)

def discord_trade_open_fiat(trade: dict, top_row,
                             verdict: dict, fg_data: dict) -> None:
    """Post compact trade-open card for a fiat (USDT buy) trade."""
    asset    = trade['buy_asset']
    buy_px   = trade['buy_price_usd']
    z        = trade['signal_z']
    conf     = trade['confidence']
    notional = trade['notional_usd']
    fee_usd  = trade['total_cost_usd']
    be_pct   = fee_usd / notional * 100 if notional else 0
    buy_qty  = trade['buy_qty']

    rep     = top_row.get('reputation') if hasattr(top_row, 'get') else None
    rep_str = (f"{rep:+.3f}" if rep is not None
               and not (isinstance(rep, float) and math.isnan(rep))
               else "+0.000 (new)")

    fg_off   = fg_data.get('fg_official', 50)
    fg_synth = fg_data.get('fg_synthetic', fg_off)
    bolt     = "⚡" if abs(fg_synth - fg_off) >= DIVERGENCE_THRESHOLD else ""
    fg_line  = (f"📊 F&G: {fg_off}→{fg_synth}{bolt} | "
                f"Retail:{fg_data.get('fg_retail', fg_off)} "
                f"Inst:{fg_data.get('fg_institutional', 50)} "
                f"Chain:{fg_data.get('fg_onchain', 50)} "
                f"Trade:{conf}")

    thesis_raw = verdict.get('thesis') or ''
    thesis = thesis_raw.split('.')[0].strip()

    msg = (
        f"🟢 **TRADE OPEN — USDT → {asset}**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Size: {notional:.2f} USDT | Z: {z:+.1f} | Conf: {conf}/10\n"
        f"Entry: {asset} @ {buy_px:,.4f} | Qty: {buy_qty:.6f} {asset}\n"
        f"Break-even: +{be_pct:.3f}%\n"
        f"Rep: {rep_str} | Est. fee: {fee_usd:.4f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{fg_line}\n"
        f"💡 {thesis}"
    )
    discord_send(msg)

def discord_signal_suppressed(sell_asset: str, buy_asset: str,
                               confidence: int, open_count: int,
                               reason: str) -> None:
    """Post brief note when a high-confidence signal is blocked by cap or cooldown."""
    if reason == 'cap':
        detail = f"cap reached, {open_count}/{MAX_TRADES_PER_PAIR} positions open"
    else:
        detail = f"cooldown active ({TRADE_COOLDOWN_MINUTES}min)"
    msg = f"⏸ Signal suppressed — {sell_asset}/{buy_asset} ({detail})"
    discord_send(msg)

def discord_trade_closed(sell_asset: str, buy_asset: str,
                          pnl_usd: float, pnl_pct: float, reason: str,
                          open_ts=None, entry_z: float | None = None,
                          exit_z: float | None = None,
                          fg_value: int = 50,
                          sl_thresh: float | None = None) -> None:
    """Unified close notification — handles reverted, z_crossback, expiry, stop-loss, news veto."""
    if reason == 'stop_loss':
        emoji  = "🛑"
        header = f"**STOP-LOSS {pnl_usd:+.2f} USDT ({pnl_pct:+.3f}%) — {sell_asset} → {buy_asset}**"
    elif 'news_veto' in reason:
        emoji  = "📰"
        header = f"**NEWS VETO {pnl_usd:+.2f} USDT ({pnl_pct:+.3f}%) — {sell_asset} → {buy_asset}**"
    elif pnl_usd >= 0:
        emoji  = "✅"
        header = f"**CLOSED +${pnl_usd:.2f} ({pnl_pct:+.3f}%) — {sell_asset} → {buy_asset}**"
    else:
        emoji  = "🔴"
        header = f"**CLOSED ${pnl_usd:.2f} ({pnl_pct:+.3f}%) — {sell_asset} → {buy_asset}**"

    held_str = ""
    if open_ts is not None:
        try:
            open_dt = open_ts.to_pydatetime() if hasattr(open_ts, 'to_pydatetime') else open_ts
            if open_dt.tzinfo is None:
                open_dt = open_dt.replace(tzinfo=timezone.utc)
            delta   = datetime.now(timezone.utc) - open_dt
            hours   = int(delta.total_seconds() // 3600)
            mins    = int((delta.total_seconds() % 3600) // 60)
            held_str = f" | Held: {hours}h {mins:02d}m"
        except Exception:
            pass

    detail = f"Reason: {reason}{held_str}"

    if reason == 'stop_loss':
        thresh_str = f"{sl_thresh:.1f}%" if sl_thresh is not None else "n/a"
        extra = f"Threshold: {thresh_str} (F&G {fg_value})"
    elif 'news_veto' in reason:
        extra = f"Catalyst: negative news detected on {buy_asset}"
    else:
        z_str = ""
        if entry_z is not None:
            z_str = f"Entry Z: {entry_z:+.3f}"
            if exit_z is not None:
                z_str += f" → Exit Z: {exit_z:+.3f}"
        extra = "\n".join(filter(None, [
            z_str,
            f"F&G at close: {fg_value} ({fg_classification(fg_value)})",
        ]))

    ferengi = _ferengi_quote(pnl_pct)

    msg = (
        f"{emoji} {header}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{detail}\n"
        f"{extra}"
        f"{ferengi}"
    )
    discord_send(msg)

def discord_daily_pnl(con: duckdb.DuckDBPyConnection,
                      prices: dict[str, float]) -> None:
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)

    open_trades = con.execute("""
        SELECT sell_asset, buy_asset, sell_qty, buy_qty, sell_price_usd,
               buy_price_usd, notional_usd
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    closed_today = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE status = 'closed' AND close_ts >= ?
    """, [today_start]).fetchone()[0]

    opened_today = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades WHERE timestamp >= ?
    """, [today_start]).fetchone()[0]

    wins_today = con.execute("""
        SELECT COUNT(*) FROM fantasy_trades
        WHERE status = 'closed' AND close_ts >= ? AND pnl_usd >= 0
    """, [today_start]).fetchone()[0]

    clean_pnl = con.execute("""
        SELECT COALESCE(SUM(pnl_usd), 0) FROM fantasy_trades
        WHERE status = 'closed' AND close_ts >= ?
    """, [today_start]).fetchone()[0]

    total_pnl = 0.0
    best_pnl  = (-999.0, "")
    worst_pnl = (999.0, "")
    for _, t in open_trades.iterrows():
        pnl, _ = trade_pnl(t['sell_qty'], asset_price(prices, t['sell_asset'], t['sell_price_usd']),
                           t['buy_qty'], asset_price(prices, t['buy_asset'], t['buy_price_usd']),
                           t['notional_usd'])
        total_pnl += pnl
        if pnl > best_pnl[0]:  best_pnl  = (pnl, t['buy_asset'])
        if pnl < worst_pnl[0]: worst_pnl = (pnl, t['buy_asset'])

    fg_row = con.execute("""
        SELECT fng_value, fng_label FROM market_context
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    fg_str = f"Official {fg_row[0]} ({fg_row[1]})" if fg_row else "n/a"

    win_rate = f"{wins_today}/{closed_today} ({round(wins_today/closed_today*100)}%)" \
               if closed_today else "n/a"
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    num, rule = random.choice(FERENGI_RULES)
    ferengi   = f"\n━━━━━━━━━━━━━━━━━━━━━━\n⚖️ Rule of Acquisition #{num}: \"{rule}\""

    msg = (
        f"📋 **OSMIA Daily — {date_str}**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Trades: {opened_today} opened | {closed_today} closed | {len(open_trades)} open\n"
        f"Win rate: {win_rate} | Avg unrealized: {total_pnl:+.2f} USDT\n"
        f"Clean P&L today: {clean_pnl:+.2f} USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 F&G: {fg_str}"
        f"{ferengi}"
    )
    discord_send(msg)

def build_discord_header(fg_data: dict, confidence: int) -> str:
    """
    Two-line F&G header for Discord:
      Line 1: Official F&G | Synthetic F&G (+ ⚡ if divergence >= threshold)
      Line 2: All 4 cohort values
    """
    fg_off   = fg_data.get('fg_official', 50)
    fg_synth = fg_data.get('fg_synthetic', fg_off)
    lbl_off  = fg_data.get('fg_official_label') or fg_classification(fg_off)
    lbl_syn  = fg_classification(fg_synth)
    divergence = abs(fg_synth - fg_off)
    bolt = " ⚡" if divergence >= DIVERGENCE_THRESHOLD else ""

    line1 = (f"📊 Official F&G: {fg_off} ({lbl_off}) | "
             f"Synthetic: {fg_synth} ({lbl_syn}){bolt}")
    retail_val = fg_data.get('fg_retail', fg_off)
    alt_me     = fg_data.get('fg_retail_alt_me')
    cmc        = fg_data.get('fg_retail_cmc')
    cg         = fg_data.get('fg_retail_coingecko')
    src_vals   = [v for v in (alt_me, cmc, cg) if v is not None]
    if len(src_vals) >= 2 and (max(src_vals) - min(src_vals)) > 10:
        retail_str = f"{retail_val} (alt.me:{alt_me} | CMC:{cmc} | CG:{cg})"
    else:
        retail_str = str(retail_val)
    line2 = (f"🔍 Retail: {retail_str} | "
             f"Inst: {fg_data.get('fg_institutional', 50)} | "
             f"On-chain: {fg_data.get('fg_onchain', 50)} | "
             f"Trader: {fg_data.get('fg_trader', 50)} | "
             f"Trade: {_trade_signal_label(confidence)}")
    return f"{line1}\n{line2}"
