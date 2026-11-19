"""Analysis cycles: build prompts, ask the LLM, gate and execute paper trades."""
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

from . import config
from .config import (
    ASSETS, FNG_FEAR, FNG_GREED, MAX_TRADES_PER_PAIR, MIN_CONFIDENCE_TRADE,
    MIN_POSITION_USD, MIN_PROFIT_USD, ORDER_BOOK_SKIP_SPREAD_PCT,
    TRADE_COOLDOWN_MINUTES, TREND_SLOPE_THRESHOLD, UNLOCK_ASSETS, UNTRUSTED_NOTE,
    ZSCORE_BUY_THRESH, _Z_PROFIT_FACTOR,
)
from .core import (
    _etf_flow_label, _parse_position_review, asset_price, calculate_synthetic_fg,
    conviction_label, fee_breakdown, fg_classification, fng_modifier, pair_id,
    reputation_label, trade_pnl, trade_route,
)
from .learning import apply_report_insights, get_active_report_insights
from .llm import ask_verdict, ping_llm
from .notify import (
    discord_send, discord_signal_suppressed, discord_trade_closed,
    discord_trade_open_fiat, discord_trade_open_pair,
)
from .portfolio import (
    _auto_close_trades, _close_trade, _get_latest_fantasy_balances, _insert_trade,
    _pair_at_max_trades, _pair_on_cooldown, _settle_closed_to_usdt,
    _snapshot_balances_upsert, compute_portfolio_value, execute_fantasy_fiat_trade,
    execute_fantasy_pair_trade, macro_confidence_multiplier,
    signal_weighted_notional,
)
from .signals import (
    build_candidates, derive_confidence, get_fiat_candidates, get_pair_reputations,
    get_pair_zscores,
)
from .sources.market import fetch_order_book_metrics
from .sources.news import (
    _get_asset_news_text, check_heightened_mode, compute_news_sentiment_score,
    fetch_all_news, fetch_market_context, format_asset_news_block, store_asset_news,
    store_market_context,
)
from .sources.sentiment import (
    _build_divergence_note, build_macro_block, fetch_macro_context,
    get_btc_change_24h, get_cached_fg_data, get_cohort_weights, get_fg_trader,
)
from .store import (
    _get_macro_cache_float, _refresh_pair_reputation, _set_macro_cache, get_conn,
    get_current_prices,
)


def _get_entry_thesis(con, sell_asset: str, buy_asset: str) -> str:
    """Look up the original LLM thesis for a pair from llm_verdicts."""
    # Verdicts name the signal pair BASE/QUOTE, and the trade bought BASE.
    signal_pair = f"{buy_asset}/{sell_asset}"
    row     = con.execute("""
        SELECT thesis FROM llm_verdicts
        WHERE top_pair = ? AND action = 'accumulate'
        ORDER BY timestamp DESC LIMIT 1
    """, [signal_pair]).fetchone()
    return row[0] if row and row[0] else "No original thesis found."

def build_position_review_prompt(pair: str, sell_asset: str, buy_asset: str,
                                  entry_z: float, current_z: float | None,
                                  pnl_pct: float, thesis: str,
                                  fg_value: int, btc_dom: float | None,
                                  etf_flow_m: float | None,
                                  base_news: str, quote_news: str) -> str:
    cur_z_str  = f"{current_z:+.3f}" if current_z is not None else "n/a"
    dom_str    = f"{btc_dom:.1f}%" if btc_dom is not None else "n/a"
    flow_str   = _etf_flow_label(etf_flow_m)
    return f"""[OPEN POSITION REVIEW]
Position: sold {sell_asset}, bought {buy_asset} (betting {buy_asset} recovers vs {sell_asset}) | Entry Z: {entry_z:+.3f} | Current Z: {cur_z_str} | PnL: {pnl_pct:+.3f}%
Entry thesis: {thesis}

[CURRENT MACRO]
F&G: {fg_value} | BTC Dom: {dom_str} | ETF Flow: {flow_str}

{UNTRUSTED_NOTE}

[FRESH NEWS — {sell_asset}]
{base_news}

[FRESH NEWS — {buy_asset}]
{quote_news}

Should this position be held or closed early?
Respond with HOLD or EXIT on the first line, followed by one sentence reason.
Then on a new line: Urgency: N  (where N is 1-10, 10=close immediately)
"""

def evaluate_open_positions(con: duckdb.DuckDBPyConnection,
                             prices: dict[str, float],
                             fng_val: int,
                             macro: dict,
                             etf_flow_m: float | None,
                             current_zs: dict[str, float]) -> int:
    """
    For every open position: refresh news → Qwen review → force close if EXIT urgency>=7.
    Returns count of positions force-closed.
    """
    open_trades = con.execute("""
        SELECT trade_id, sell_asset, buy_asset, sell_qty, buy_qty,
               sell_price_usd, buy_price_usd, notional_usd, total_cost_usd,
               signal_z, timestamp
        FROM fantasy_trades WHERE status = 'open'
    """).df()

    if open_trades.empty:
        return 0, con

    # Targeted news refresh — only assets in open positions
    held_assets = list(set(
        open_trades['sell_asset'].tolist() + open_trades['buy_asset'].tolist()
    ))
    held_assets = [a for a in held_assets if a != 'USDT']
    if held_assets:
        print(f"  [living-ctx] Refreshing news for {len(held_assets)} held assets …")
        con.close()
        fresh_rows = fetch_all_news(held_assets)
        con2       = get_conn()
        store_asset_news(con2, fresh_rows)
        # Rebind con for caller — Python can't reassign caller's variable,
        # so we do all DB work through con2 for the rest of this function.
    else:
        con2 = con

    btc_dom   = macro.get('btc_dominance')
    force_closed = 0
    now          = datetime.now(timezone.utc)

    for _, t in open_trades.iterrows():
        pair      = f"{t['sell_asset']}/{t['buy_asset']}"
        cur_px    = asset_price(prices, t['buy_asset'], t['buy_price_usd'])
        sell_now  = asset_price(prices, t['sell_asset'], t['sell_price_usd'])
        _, pnl_pct = trade_pnl(t['sell_qty'], sell_now, t['buy_qty'], cur_px,
                               t['notional_usd'])
        current_z = current_zs.get(f"{t['buy_asset']}/{t['sell_asset']}")
        thesis    = _get_entry_thesis(con2, t['sell_asset'], t['buy_asset'])
        base_news = _get_asset_news_text(con2, t['sell_asset'])
        quote_news= _get_asset_news_text(con2, t['buy_asset'])

        prompt  = build_position_review_prompt(
            pair, t['sell_asset'], t['buy_asset'],
            float(t['signal_z']), current_z, pnl_pct,
            thesis, fng_val, btc_dom, etf_flow_m,
            base_news, quote_news
        )
        print(f"  [living-ctx] Reviewing {pair} (PnL {pnl_pct:+.2f}%) …")
        con2.close()
        verdict_raw = ask_verdict(prompt)
        con2        = get_conn()

        content  = verdict_raw.get('prose', '') or verdict_raw.get('raw_json', '')
        # ask_verdict parses JSON; for free-form responses fall back to raw content
        if not content:
            continue
        action, urgency, reason = _parse_position_review(content)
        print(f"    → {action} (urgency {urgency}/10): {reason[:80]}")

        if action == 'EXIT' and urgency >= 7:
            # Force close
            pnl_usd, pnl_pct_final = _close_trade(
                con2, t['trade_id'], cur_px,
                t['notional_usd'], t['buy_qty'], 'news_veto_live',
                sell_qty=t['sell_qty'], sell_px_now=sell_now,
            )
            discord_trade_closed(
                str(t['sell_asset']), str(t['buy_asset']),
                pnl_usd, pnl_pct_final, 'news_veto_live',
                open_ts=t['timestamp'],
                entry_z=float(t['signal_z']) if t['signal_z'] is not None else None,
                fg_value=fng_val,
            )
            exit_id = f"{t['trade_id']}_livevetoexit"
            con2.execute("""
                INSERT OR IGNORE INTO exit_log
                    (exit_id, timestamp, trade_id, buy_asset, pnl_usd, pnl_pct,
                     reason, fg_value, etf_flow_m, z_threshold, news_veto, stop_loss)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [exit_id, now, t['trade_id'], t['buy_asset'],
                  pnl_usd, pnl_pct_final, 'news_veto_live',
                  fng_val, etf_flow_m, None, True, False])
            _settle_closed_to_usdt(con2, [{'buy_asset': str(t['buy_asset']),
                                           'buy_qty': float(t['buy_qty']),
                                           'cur_px': cur_px}], prices)
            force_closed += 1

        elif action == 'EXIT' and urgency < 7:
            msg = (f"⚠️ **Living context warning** — {pair}\n"
                   f"Urgency {urgency}/10 — holding for now.\n"
                   f"{reason}")
            discord_send(msg)

    # Return the reopened connection to be usable by the caller
    # Caller must reassign its own `con` from the return value
    return force_closed, con2

def _log_verdict(con, verdict, confidence, top_row, mode,
                 reputation=None, trend_slope=None,
                 fng_value=None, fng_label=None,
                 liquidity_score=None) -> None:
    ts       = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    z_score  = float(top_row['z_score']) if top_row is not None else None
    top_pair = verdict.get('top_pair') or 'none'
    con.execute("""
        INSERT OR IGNORE INTO llm_verdicts
            (timestamp, mode, top_pair, z_score, confidence_score,
             action, thesis, risks, prose, raw_json,
             reputation, trend_slope, fng_value, fng_label, liquidity_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [ts, mode, top_pair, z_score, confidence,
          verdict.get('action',   'pass'),
          verdict.get('thesis',   ''),
          verdict.get('risks',    ''),
          verdict.get('prose',    ''),
          verdict.get('raw_json', ''),
          reputation, trend_slope, fng_value, fng_label, liquidity_score])

HEIGHTENED_PROMPT_SUFFIX = """
⚠️ UNLOCK ALERT ACTIVE
{unlock_context}

You are analyzing a pair where one or both legs have imminent token unlocks.
Historical pattern: token unlocks cause pre-unlock selling (front-running) and
post-unlock volatility.

An ACCUMULATE on BASE/QUOTE means: sell QUOTE, buy BASE (betting BASE recovers
against QUOTE). Bias your analysis toward:
- WATCH or PASS if the unlocked asset is the base (first leg): unlock selling
  can keep pushing it down, so the dip may not revert yet
- ACCUMULATE is more defensible if the unlocked asset is the quote (second leg,
  the one being sold)
- Flag any ACCUMULATE with HIGH CAUTION if an unlock is within 24h

Do not fight the unlock. Say clearly in your thesis whether this is a
pre-unlock risk or a post-unlock recovery entry.
"""

def build_pair_prompt(con, candidates, quote_warn, base_warn,
                      fng_narrative: str, headlines: list[str],
                      macro_block: str = "",
                      heightened_ctx: dict[str, str] | None = None,
                      divergence_note: str = "") -> str:
    heightened_ctx = heightened_ctx or {}
    rows = []
    for i, r in candidates.iterrows():
        vel   = f"{r['z_velocity']:+.3f}"  if r['z_velocity']  is not None else "n/a"
        spread= f"{r['spread_pct']:.4f}%"  if r['spread_pct']  is not None else "n/a"
        vwap  = f"{r['vwap_ratio']:.4f}"   if r['vwap_ratio']  is not None else "n/a"
        vol_z = f"{r['volume_z']:+.3f}"    if r['volume_z']    is not None else "n/a"
        rep   = r.get('reputation')
        rep_s = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
        slope = r.get('trend_slope', 0.0)
        trend_flag = " ⚠TRENDING" if abs(slope) > TREND_SLOPE_THRESHOLD else ""
        route, _ = trade_route(r['base_asset'], r['quote_asset'])
        rows.append(
            f"  [rank {i+1}] {r['pair']:>12s} | z={r['z_score']:+.3f} | vel={vel} | "
            f"persist={r['persistence_24h']} | vwap={vwap} | vol_z={vol_z} | "
            f"spread={spread} | rep={rep_s}{trend_flag} | route={route}"
        )
    table = "\n".join(rows)
    warn  = ""
    if quote_warn:
        warn += f"\n⚠ QUOTE DOMINANCE: {quote_warn} is the quote in multiple pairs. Address in thesis.\n"
    if base_warn:
        warn += f"\n⚠ BASE DOMINANCE: {base_warn} — likely one signal, not independent.\n"
    if divergence_note:
        warn += f"\n⚠ {divergence_note}\n"

    news_block = ""
    if headlines:
        news_block = "\nRecent crypto headlines:\n" + "\n".join(f"  - {h}" for h in headlines) + "\n"

    leg_assets = pd.unique(candidates[['base_asset', 'quote_asset']].values.ravel())
    leg_blocks = [b for b in (format_asset_news_block(con, a) for a in leg_assets) if b]
    leg_news   = ("\n\n" + "\n\n".join(leg_blocks) + "\n") if leg_blocks else ""
    if news_block or leg_news:
        news_block = f"\n{UNTRUSTED_NOTE}\n" + news_block

    macro = (macro_block + "\n\n") if macro_block else ""

    suffix = ""
    hot = [a for a in leg_assets if a in heightened_ctx]
    if hot:
        unlock_context = "\n".join(heightened_ctx[a] for a in hot)
        suffix = HEIGHTENED_PROMPT_SUFFIX.format(unlock_context=unlock_context)

    return f"""{macro}You are a disciplined spot-only crypto trader. Rule-based engine pre-ranked {len(candidates)} candidates.

Market context: {fng_narrative}
{news_block}{leg_news}
Candidates (best first):
{table}
{warn}
Column guide: z=robust Z-score, vel=Z delta (negative=deepening), persist=24h hits,
vwap=last/VWAP, vol_z=volume anomaly, spread=bid-ask%, rep=historical score (-1 to +1),
⚠TRENDING=ratio trending not oscillating.

Rules:
- Rule rank is authoritative. Override requires explicit justification.
- Poor rep (< -0.3) + Z > -3.0: watch not accumulate.
- Strong rep (> 0.4) + Z < -1.8: can accumulate.
- ⚠TRENDING: accumulate only if Z < -3.0 and rep > 0.3.
- Factor macro context into thesis — mention F&G if relevant.
- Do NOT assign equal confidence. spread >0.15% caps confidence at 6.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_pair":"<PAIR>","rank":["<p1>",...],"thesis":"<str>","risks":"<str>"}}
{suffix}"""

def format_pair_trade(top_row, prices, verdict, confidence, notional,
                      fg_data: dict, macro_mult: float = 1.0) -> None:
    base, quote = top_row['base_asset'], top_row['quote_asset']
    base_px     = prices.get(base,  0.0)
    quote_px    = prices.get(quote, 0.0)
    route, legs = trade_route(base, quote)
    fees        = fee_breakdown(notional, legs, top_row['spread_pct'])
    sell_qty    = round(notional / quote_px, 8) if quote_px else 0
    buy_qty     = round(notional / base_px,  8) if base_px  else 0
    rep         = top_row.get('reputation')
    rep_s       = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
    slope       = top_row.get('trend_slope', 0.0)
    vel         = top_row['z_velocity'] if top_row['z_velocity'] is not None else 'n/a'
    conviction  = conviction_label(confidence)
    fg_off      = fg_data.get('fg_official', 50)
    fg_synth    = fg_data.get('fg_synthetic', fg_off)
    fg_lbl      = fg_data.get('fg_official_label', fg_classification(fg_off))

    print("\n╔══ TRADE SIGNAL ══════════════════════════════════════════╗")
    print(f"  Mode       : PAIR REVERSION (spot)  —  {conviction}")
    print(f"  Action     : Sell {quote} → Buy {base}  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  Route      : {route}")
    print(f"  ─────────────────────────────────────────────────────")
    mult_str = f"  [{macro_mult:.2f}x macro mult]" if abs(macro_mult - 1.0) > 0.01 else "  [signal-weighted]"
    print(f"  Sell       : {sell_qty} {quote}  @ {quote_px:,.4f} USDT")
    print(f"  Buy        : {buy_qty} {base}  @ {base_px:,.4f} USDT")
    print(f"  Position   : {fees['notional']} USDT{mult_str}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['fee_per_leg']} USDT × {fees['num_legs']} legs = {fees['total_fee']} USDT")
    if top_row['spread_pct']:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({top_row['spread_pct']:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of position)")
    print(f"  Break-even : ratio must improve {fees['breakeven_pct']:.4f}%")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Signal     : Z={top_row['z_score']:+.3f}  persist={top_row['persistence_24h']}  vel={vel}")
    print(f"  Reputation : {rep_s}")
    print(f"  Market     : {fg_lbl} (Official F&G {fg_off} | Synthetic {fg_synth})")
    if abs(slope) > TREND_SLOPE_THRESHOLD:
        print(f"  ⚠ Trend    : slope={slope:.6f} — may be trending not oscillating")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")

def analyze_pairs(con: duckdb.DuckDBPyConnection,
                  size_override: float | None = None,
                  position_pct: float | None = None,
                  min_position: float = MIN_POSITION_USD,
                  min_profit: float = MIN_PROFIT_USD) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running pair analysis …")

    if not ping_llm():
        print(f"  ⚠ WARNING: LLM unreachable at {config.LLM_URL}")

    # V11: Release lock for network I/O. F&G no longer fetched inline — read from
    # macro_cache (populated by --fetch-fg cron). Headlines still fetched inline.
    con.close()
    ctx       = fetch_market_context()   # headlines + official F&G as fallback
    macro     = fetch_macro_context()    # CoinGecko dominance + mcap 24h change
    news_rows = fetch_all_news(ASSETS)   # Layer 2 per-asset news
    con       = get_conn()
    store_market_context(con, ctx)
    store_asset_news(con, news_rows)

    # V11: Primary F&G source is macro_cache (--fetch-fg cron).
    # Falls back to ctx values if cache is empty (e.g., cron not yet configured).
    fg_data  = get_cached_fg_data(con)
    fng_val  = fg_data['fg_official']
    fng_lbl  = fg_data.get('fg_official_label', fg_classification(fng_val))
    fg_synth = fg_data['fg_synthetic']

    # Sync official label into the ctx dict for macro_block rendering
    ctx_for_block = {'fng_value': fng_val, 'fng_label': fng_lbl,
                     'headlines': ctx['headlines']}

    fng_adj, fng_narrative = fng_modifier(fng_val)
    # Update fg_trader in the cohort now that we have a DB connection
    trader_fg = get_fg_trader(con)
    if trader_fg != fg_data.get('fg_trader', 50):
        _set_macro_cache(con, 'fg_trader', str(trader_fg))
        fg_data['fg_trader'] = trader_fg
        # Recompute synthetic with updated trader score
        weights = get_cohort_weights(con)
        fg_synth = calculate_synthetic_fg({
            'retail': fg_data['fg_retail'], 'institutional': fg_data['fg_institutional'],
            'onchain': fg_data['fg_onchain'], 'trader': trader_fg,
        }, weights)
        _set_macro_cache(con, 'fg_synthetic', str(fg_synth))
        fg_data['fg_synthetic'] = fg_synth

    divergence_note = _build_divergence_note(fg_data)
    if divergence_note:
        print(f"  ⚡ {divergence_note}")

    # Heightened mode: any watchlist asset with an imminent unlock relaxes its
    # pairs' dump threshold and arms the unlock-aware prompt suffix.
    relaxed_assets, heightened_ctx, heightened_lines = set(), {}, []
    for asset in UNLOCK_ASSETS:
        is_h, uctx = check_heightened_mode(con, asset)
        if is_h:
            relaxed_assets.add(asset)
            heightened_ctx[asset] = uctx
            heightened_lines.append(uctx)
            print(f"  🔥 Heightened mode active for {asset}")

    btc_change  = get_btc_change_24h(con)
    macro_block = build_macro_block(ctx_for_block, macro, btc_change, heightened_lines, fg_data)
    print(f"  Market: {fng_lbl} (Official F&G {fng_val} | Synthetic {fg_synth}) — {fng_narrative}")
    print("  " + macro_block.replace("\n", "\n  "))

    candidates, quote_warn, base_warn = build_candidates(con, fng_adj, relaxed_assets)

    prices     = get_current_prices(con)
    etf_flow_m = fg_data.get('etf_flow_usd_m') or _get_macro_cache_float(con, 'etf_flow_usd_m')
    empty_zs   = {}
    pre_closed = _auto_close_trades(con, prices, empty_zs, fng_val, etf_flow_m, fg_synth)
    if pre_closed:
        print(f"\n  ✅ Auto-closed {pre_closed} trade(s) (expiry/veto) before signal scan")

    if candidates.empty:
        print("  No anomalies — market within normal bounds.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'Market stable.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'pair', fng_value=fng_val, fng_label=fng_lbl)
        _refresh_pair_reputation(con)
        return

    if quote_warn: print(f"\n  ⚠ Quote dominance : {quote_warn}")
    if base_warn:  print(f"  ⚠ Base dominance  : {base_warn}")

    print("\n── Candidates (rule-scored) ──")
    disp = candidates[['pair', 'z_score', 'z_velocity', 'persistence_24h',
                        'vwap_ratio', 'volume_z', 'spread_pct',
                        'reputation', 'trend_slope', 'rule_score']].copy()
    disp['reputation'] = disp['reputation'].apply(
        lambda r: f"{r:+.3f}" if r is not None else "n/a")
    print(disp.to_string(index=False))

    prompt     = build_pair_prompt(con, candidates, quote_warn, base_warn,
                                   fng_narrative, ctx['headlines'],
                                   macro_block, heightened_ctx, divergence_note)
    con.close()
    verdict    = ask_verdict(prompt)
    con        = get_conn()
    top_pair   = verdict.get('top_pair', '')
    confidence = derive_confidence(candidates, top_pair, fng_val)

    current_zs = {r['pair']: r['z_score'] for _, r in candidates.iterrows()}
    # Open positions need their live z too (z-crossback exit + position review);
    # a recovered pair has dropped out of the candidate list by definition.
    open_pairs = [f"{b}/{s_}" for s_, b in con.execute(
        "SELECT sell_asset, buy_asset FROM fantasy_trades "
        "WHERE status='open' AND mode='pair'").fetchall()]
    current_zs.update({p: z for p, z in get_pair_zscores(con, open_pairs).items()
                       if p not in current_zs})

    closed = _auto_close_trades(con, prices, current_zs, fng_val, etf_flow_m, fg_synth)
    if closed:
        print(f"\n  ✅ Auto-closed {closed} fantasy trade(s)")

    top_row   = None
    rep_val   = None
    slope_val = None
    if top_pair:
        match = candidates[candidates['pair'] == top_pair]
        if not match.empty:
            top_row   = match.iloc[0]
            rep_val   = top_row.get('reputation')
            slope_val = top_row.get('trend_slope')

    balances = _get_latest_fantasy_balances(con)

    # v1.4 Feature 4: read active report insights (6h TTL) before trade decision
    active_insights = get_active_report_insights(con)
    if active_insights:
        print(f"  [insights] {len(active_insights)} active pair insight(s) loaded")

    liquidity_score = None  # logged to llm_verdicts (Feature 3)

    if top_row is not None and verdict.get('action') == 'accumulate':
        # z < threshold on base/quote => base is cheap vs quote. Mean reversion:
        # BUY base, funded by SELLING quote. Outcome scoring (base/quote ratio up
        # = win) and trade P&L now agree on this direction.
        base     = top_row['base_asset']
        quote    = top_row['quote_asset']
        base_px  = prices.get(base, 0.0)
        quote_px = prices.get(quote, 0.0)

        # v1.2 Feature 1: macro-aware position sizing
        news_sent  = compute_news_sentiment_score(con, base)
        macro_mult = macro_confidence_multiplier(
            fg_synth, fng_val, fg_data.get('fg_institutional', 50), news_sent
        )
        if size_override is not None:
            notional   = size_override
            macro_mult = 1.0
        elif position_pct is not None:
            portfolio_val = compute_portfolio_value(balances, prices)
            notional = max(1.0, round(portfolio_val * (position_pct / 100.0) * macro_mult, 4))
        else:
            notional = signal_weighted_notional(quote, quote_px, balances,
                                                rep_val, top_row['z_score'])
            notional = max(1.0, round(notional * macro_mult, 4))

        # v1.4 Feature 4: apply report insights (confidence + sizing adjustments)
        if active_insights:
            confidence, notional, insight_suppressed = apply_report_insights(
                active_insights, top_pair, confidence, notional
            )
            if insight_suppressed:
                print(f"\n  [insights] Signal suppressed for {top_pair} by report feedback loop")
                _log_verdict(con, verdict, confidence, top_row, 'pair',
                             rep_val, slope_val, fng_val, fng_lbl,
                             liquidity_score=liquidity_score)
                _refresh_pair_reputation(con)
                return
            if confidence != derive_confidence(candidates, top_pair, fng_val):
                print(f"  [insights] Confidence adjusted → {confidence}/10 by report insights")

        # v1.4 Feature 1: minimum position floor
        if notional < min_position:
            if confidence >= MIN_CONFIDENCE_TRADE:
                print(f"  [min-pos] Notional {notional:.2f} < floor {min_position:.2f} — clamping")
                notional = min_position
            else:
                print(f"\n  📒 Fantasy trade skipped — notional {notional:.2f} < "
                      f"min_position {min_position:.2f} (conf {confidence} too low to clamp)")
                _log_verdict(con, verdict, confidence, top_row, 'pair',
                             rep_val, slope_val, fng_val, fng_lbl,
                             liquidity_score=liquidity_score)
                _refresh_pair_reputation(con)
                return

        # v1.4 Feature 1: predicted profit check
        _, legs = trade_route(base, quote)
        fees_est = fee_breakdown(notional, legs, top_row.get('spread_pct') or 0.001)
        predicted_profit = notional * abs(top_row['z_score']) * _Z_PROFIT_FACTOR - fees_est['total_cost']
        if predicted_profit < min_profit:
            print(f"\n  📒 Fantasy trade skipped — predicted profit {predicted_profit:.3f} USDT "
                  f"< min_profit {min_profit:.2f}")
            _log_verdict(con, verdict, confidence, top_row, 'pair',
                         rep_val, slope_val, fng_val, fng_lbl,
                         liquidity_score=liquidity_score)
            _refresh_pair_reputation(con)
            return

        # v1.4 Feature 3: order book depth check (network call — outside DB lock)
        con.close()
        ob = fetch_order_book_metrics(base, quote, notional)
        con = get_conn()
        liquidity_score = ob['liquidity_score']

        if ob['skip']:
            spread_pct_real = ob['spread_real'] * 100
            print(f"\n  📒 Fantasy trade skipped — order book spread "
                  f"{spread_pct_real:.3f}% exceeds hard limit {ORDER_BOOK_SKIP_SPREAD_PCT:.2f}%")
            _log_verdict(con, verdict, confidence, top_row, 'pair',
                         rep_val, slope_val, fng_val, fng_lbl,
                         liquidity_score=liquidity_score)
            _refresh_pair_reputation(con)
            return

        if ob['is_thin']:
            # Reduce sizing by liquidity score when book is thin
            old_notional = notional
            notional = max(min_position, round(notional * max(0.5, liquidity_score), 4))
            print(f"  [orderbook] Thin book ({ob['spread_real']*100:.3f}% spread) — "
                  f"sizing {old_notional:.2f} → {notional:.2f} USDT "
                  f"(liquidity score {liquidity_score:.2f})")
        elif ob['est_slippage'] > 0:
            print(f"  [orderbook] Est. slippage {ob['est_slippage']*100:.4f}% for "
                  f"{notional:.2f} USDT fill")

        if ob['is_thin']:
            discord_send(
                f"⚠️ **Thin book** — {base}/{quote}: "
                f"spread {ob['spread_real']*100:.3f}% "
                f"(bid ${ob['bid_depth_usd']:.0f} / ask ${ob['ask_depth_usd']:.0f}). "
                f"Position reduced to {notional:.2f} USDT."
            )

        format_pair_trade(top_row, prices, verdict, confidence, notional, fg_data, macro_mult)

        if confidence >= MIN_CONFIDENCE_TRADE:
            if _pair_at_max_trades(con, base, quote):
                print(f"\n  📒 Fantasy trade skipped — {base}/{quote} at max {MAX_TRADES_PER_PAIR} open trades")
                if confidence >= 8:
                    open_cnt = con.execute(
                        "SELECT COUNT(*) FROM fantasy_trades WHERE canonical_pair_id = ? AND status = 'open'",
                        [pair_id(base, quote)]
                    ).fetchone()[0]
                    discord_signal_suppressed(base, quote, confidence, open_cnt, 'cap')
            elif _pair_on_cooldown(con, base, quote):
                print(f"\n  📒 Fantasy trade skipped — {base}/{quote} on {TRADE_COOLDOWN_MINUTES}min cooldown")
                if confidence >= 8:
                    discord_signal_suppressed(base, quote, confidence, 0, 'cooldown')
            else:
                trade = execute_fantasy_pair_trade(
                    con, quote, base, quote_px, base_px,
                    notional, top_row['spread_pct'],
                    top_row['z_score'], confidence, balances,
                    position_multiplier=macro_mult
                )
                if trade:
                    _insert_trade(con, trade)
                    _snapshot_balances_upsert(con, balances, prices)
                    discord_trade_open_pair(trade, top_row, verdict, fg_data, macro_mult)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient {quote} balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")
        if verdict.get('prose'):
            print(f"  {verdict['prose']}")

    _log_verdict(con, verdict, confidence, top_row, 'pair',
                 rep_val, slope_val, fng_val, fng_lbl,
                 liquidity_score=liquidity_score)
    _refresh_pair_reputation(con)

    # v1.0: Feature 2 — living context re-evaluation of all open positions
    if ping_llm():
        try:
            force_closed, con = evaluate_open_positions(
                con, prices, fng_val, macro, etf_flow_m, current_zs
            )
            if force_closed:
                print(f"\n  📰 Living context force-closed {force_closed} position(s)")
        except Exception as e:
            print(f"  [living-ctx] Error during re-evaluation: {e}", file=sys.stderr)
    else:
        print("  [living-ctx] Skipped — Qwen unreachable")

def build_fiat_prompt(candidates, reputations,
                      fng_narrative: str, headlines: list[str]) -> str:
    rows = []
    for i, r in candidates.iterrows():
        pair  = f"{r['asset_id']}/USDT"
        rep   = reputations.get(pair)
        rep_s = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
        rows.append(
            f"  [rank {i+1}] {r['asset_id']:>6s}/USDT | price={r['price_usd']:,.4f} | "
            f"z={r['z_score']:+.3f} | vwap={r['vwap_ratio'] or 'n/a'} | "
            f"vol_z={r['volume_z'] or 'n/a'} | "
            f"24h={r['change_24h']:+.2f}% | spread={r['spread_pct'] or 'n/a'}% | rep={rep_s}"
        )
    table = "\n".join(rows)

    news_block = ""
    if headlines:
        news_block = "\nRecent crypto headlines:\n" + "\n".join(f"  - {h}" for h in headlines) + "\n"

    return f"""You are a disciplined spot-only crypto trader advising on USDT buy entries.
Market context: {fng_narrative}
{news_block}
Statistically cheap assets (robust Z < {ZSCORE_BUY_THRESH}, median/MAD):
{table}

Rules: spot only, no leverage.
- Poor rep (< -0.3): watch only unless Z < -3.0.
- Strong rep (> 0.4): can accumulate at Z < -1.8.
- Avoid freefall (very negative 24h, no volume support).
- Factor macro context — F&G extreme fear is a buy signal; extreme greed is caution.
- Rank all. Do not assign equal confidence. action: accumulate|watch|pass.
- thesis/risks: max 120 chars each.

One JSON line, blank line, 2-3 sentences prose.
JSON: {{"action":"accumulate"|"watch"|"pass","top_asset":"<ASSET>","rank":["<a1>",...],"thesis":"<str>","risks":"<str>"}}
"""

def format_fiat_trade(asset, price, spread_pct, z_score,
                      vwap_ratio, volume_z, change_24h,
                      verdict, confidence, notional, rep,
                      fng_value, fng_label) -> None:
    fees    = fee_breakdown(notional, 1, spread_pct)
    buy_qty = round((notional - fees['total_cost']) / price, 8) if price else 0
    rep_s   = f"{rep:+.3f} ({reputation_label(rep)})" if rep is not None else "neutral"
    conviction = conviction_label(confidence)

    print("\n╔══ FIAT BUY SIGNAL ═══════════════════════════════════════╗")
    print(f"  Mode       : USDT → {asset} (spot buy)  —  {conviction}")
    print(f"  Action     : Buy {asset}/USDT  [{verdict['action'].upper()}  {confidence}/10]")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Price      : {price:,.4f} USDT")
    print(f"  Quantity   : {buy_qty} {asset}  from {fees['notional']} USDT")
    print(f"  Position   : {fees['notional']} USDT  (signal-weighted)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Fee        : {fees['total_fee']} USDT  (0.075% BNB)")
    if spread_pct:
        print(f"  Spread     : {fees['spread_cost']} USDT  ({spread_pct:.4f}%)")
    print(f"  Total cost : {fees['total_cost']} USDT  ({fees['breakeven_pct']:.4f}% of position)")
    print(f"  Break-even : +{fees['breakeven_pct']:.4f}% price move needed")
    print(f"  ─────────────────────────────────────────────────────")
    sig_parts = [f"Z={z_score:+.3f}"]
    if vwap_ratio:  sig_parts.append(f"vwap={vwap_ratio:.4f}")
    if volume_z:    sig_parts.append(f"vol_z={volume_z:+.3f}")
    if change_24h:  sig_parts.append(f"24h={change_24h:+.2f}%")
    print(f"  Signal     : {' '.join(sig_parts)}")
    print(f"  Reputation : {rep_s}")
    print(f"  Market     : {fng_label} (F&G {fng_value})")
    print(f"  Thesis     : {verdict.get('thesis','')}")
    print(f"  Risks      : {verdict.get('risks','')}")
    if verdict.get('prose'):
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  {verdict['prose']}")
    print("╚══════════════════════════════════════════════════════════╝")

def analyze_fiat(con: duckdb.DuckDBPyConnection,
                 size_override: float | None = None,
                 min_position: float = MIN_POSITION_USD,
                 min_profit: float = MIN_PROFIT_USD) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] Running fiat entry analysis …")

    if not ping_llm():
        print(f"  ⚠ WARNING: LLM unreachable at {config.LLM_URL}")

    # V11: F&G from cache (same as analyze_pairs)
    ctx        = fetch_market_context()
    store_market_context(con, ctx)
    fg_data    = get_cached_fg_data(con)
    fng_val    = fg_data['fg_official']
    fng_lbl    = fg_data.get('fg_official_label', fg_classification(fng_val))
    fg_synth   = fg_data['fg_synthetic']
    etf_flow_m = fg_data.get('etf_flow_usd_m') or _get_macro_cache_float(con, 'etf_flow_usd_m')
    _, fng_narrative = fng_modifier(fng_val)
    print(f"  Market: {fng_lbl} (Official F&G {fng_val} | Synthetic {fg_synth})")

    candidates, short_history = get_fiat_candidates(con)

    if short_history:
        print("  ⚠ DB history < 14 days — fiat Z-scores may not reflect full price range")

    if candidates.empty:
        print("  No assets statistically cheap vs history — no entry signal.")
        _log_verdict(con, {'action': 'pass', 'top_pair': 'none',
                           'thesis': 'No fiat entry signal.', 'risks': '', 'prose': '', 'raw_json': ''},
                     0, None, 'fiat', fng_value=fng_val, fng_label=fng_lbl)
        return

    fiat_pairs  = [f"{r['asset_id']}/USDT" for _, r in candidates.iterrows()]
    reputations = get_pair_reputations(con, fiat_pairs)

    print("\n── Fiat candidates ──")
    disp = candidates[['asset_id', 'z_score', 'vwap_ratio',
                        'volume_z', 'spread_pct', 'change_24h']].copy()
    disp['reputation'] = [
        f"{reputations.get(r['asset_id'] + '/USDT', 0.0):+.3f}"
        for _, r in candidates.iterrows()
    ]
    print(disp.to_string(index=False))

    prompt    = build_fiat_prompt(candidates, reputations, fng_narrative, ctx['headlines'])
    # Release the single-writer lock during the multi-minute LLM call so the
    # concurrent ingest/pairs timers aren't blocked (and don't crash on lock).
    con.close()
    verdict   = ask_verdict(prompt)
    con       = get_conn()
    # Qwen returns the asset inconsistently as "AVAX" or "AVAX/USDT" (sometimes even
    # "AVAX/USDT/USDT"). Normalize to the bare base symbol up front and use it
    # everywhere downstream — otherwise the raw "AVAX/USDT" fails the membership test
    # against candidates['asset_id'] (bare symbols), leaving top_row=None and silently
    # skipping both the Discord post and the fantasy trade.
    top_asset = (verdict.get('top_asset', '') or '').split('/')[0].strip().upper()
    verdict['top_pair'] = f"{top_asset}/USDT" if top_asset else 'none'

    confidence = 5
    top_row    = None
    rep_val    = None
    if top_asset and top_asset in candidates['asset_id'].values:
        scores = candidates.set_index('asset_id')['z_score']
        winner = scores[top_asset]
        others = scores.drop(top_asset)
        gap    = abs(others.max() - winner) if not others.empty else 0
        conf   = 5 + min(int(gap / 0.25), 4)
        row    = candidates[candidates['asset_id'] == top_asset].iloc[0]
        if row['spread_pct'] is not None and row['spread_pct'] > 0.1: conf -= 1
        rep_val = reputations.get(f"{top_asset}/USDT")
        if rep_val is not None:
            if rep_val >= 0.5:    conf = min(conf + 1, 10)
            elif rep_val <= -0.5: conf = max(conf - 2, 1)
        if fng_val <= FNG_FEAR:    conf = min(conf + 1, 10)
        elif fng_val >= FNG_GREED: conf = max(conf - 1, 1)
        confidence = max(1, min(conf, 10))
        top_row    = row

    prices   = get_current_prices(con)
    balances = _get_latest_fantasy_balances(con)

    if top_row is not None and verdict.get('action') == 'accumulate':
        px       = prices.get(top_asset, top_row['price_usd'])
        notional = signal_weighted_notional('USDT', 1.0, balances,
                                            rep_val, top_row['z_score'], size_override)

        # v1.4 Feature 1: minimum position floor
        if notional < min_position:
            if confidence >= MIN_CONFIDENCE_TRADE:
                print(f"  [min-pos] Fiat notional {notional:.2f} < floor {min_position:.2f} — clamping")
                notional = min_position
            else:
                print(f"\n  📒 Fiat trade skipped — notional {notional:.2f} < "
                      f"min_position {min_position:.2f}")
                _log_verdict(con, verdict, confidence, top_row, 'fiat',
                             fng_value=fng_val, fng_label=fng_lbl)
                return

        fees_est_f  = fee_breakdown(notional, 1, top_row.get('spread_pct') or 0.001)
        pred_profit = notional * abs(top_row['z_score']) * _Z_PROFIT_FACTOR - fees_est_f['total_cost']
        if pred_profit < min_profit:
            print(f"\n  📒 Fiat trade skipped — predicted profit {pred_profit:.3f} USDT "
                  f"< min_profit {min_profit:.2f}")
            _log_verdict(con, verdict, confidence, top_row, 'fiat',
                         fng_value=fng_val, fng_label=fng_lbl)
            return

        format_fiat_trade(
            top_asset, px, top_row['spread_pct'], top_row['z_score'],
            top_row['vwap_ratio'], top_row['volume_z'], top_row['change_24h'],
            verdict, confidence, notional, rep_val, fng_val, fng_lbl
        )

        if confidence >= MIN_CONFIDENCE_TRADE:
            if _pair_at_max_trades(con, 'USDT', top_asset):
                print(f"\n  📒 Fantasy trade skipped — {top_asset}/USDT at max {MAX_TRADES_PER_PAIR} open trades")
                if confidence >= 8:
                    open_cnt = con.execute(
                        "SELECT COUNT(*) FROM fantasy_trades WHERE canonical_pair_id = ? AND status = 'open'",
                        [pair_id('USDT', top_asset)]
                    ).fetchone()[0]
                    discord_signal_suppressed('USDT', top_asset, confidence, open_cnt, 'cap')
            elif _pair_on_cooldown(con, 'USDT', top_asset):
                print(f"\n  📒 Fantasy trade skipped — {top_asset}/USDT on {TRADE_COOLDOWN_MINUTES}min cooldown")
                if confidence >= 8:
                    discord_signal_suppressed('USDT', top_asset, confidence, 0, 'cooldown')
            else:
                trade = execute_fantasy_fiat_trade(
                    con, top_asset, px, notional, top_row['spread_pct'],
                    top_row['z_score'], confidence, balances
                )
                if trade:
                    _insert_trade(con, trade)
                    _snapshot_balances_upsert(con, balances, prices)
                    discord_trade_open_fiat(trade, top_row, verdict, fg_data)
                    print(f"\n  📒 Fantasy trade logged: {trade['trade_id']}")
                else:
                    print(f"\n  ⚠ Fantasy trade skipped — insufficient USDT balance")
        else:
            print(f"\n  📒 Fantasy trade skipped — confidence {confidence} < {MIN_CONFIDENCE_TRADE}")
    else:
        print(f"\n  Verdict: {verdict.get('action','pass').upper()}  —  {verdict.get('thesis','')}")

    _log_verdict(con, verdict, confidence, top_row, 'fiat',
                 rep_val, None, fng_val, fng_lbl)
