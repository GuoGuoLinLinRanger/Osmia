"""Learning loop: outcome scoring, reputation-fed reports, LLM insights."""
import sys
from datetime import datetime, timedelta, timezone

import duckdb

from . import config
from .config import DIVERGENCE_THRESHOLD, OUTCOME_WINDOW_HOURS, V08_EPOCH
from .core import asset_price, sanitize_insight, trade_pnl
from .llm import ask_insights, ask_report, ping_llm
from .notify import discord_send
from .signals import get_pair_reputations
from .sources.sentiment import get_cached_fg_data
from .store import get_conn, get_current_prices


def score_outcomes(con: duckdb.DuckDBPyConnection,
                   all_pairs: bool = False) -> None:
    print(f"[{datetime.now():%H:%M:%S}] Scoring outcomes …")

    pending = con.execute(f"""
        SELECT timestamp, top_pair, mode, z_score
        FROM llm_verdicts
        WHERE action = 'accumulate'
          AND outcome_pct IS NULL
          AND timestamp <= NOW() - INTERVAL {OUTCOME_WINDOW_HOURS} HOURS
    """).df()

    if pending.empty:
        print(f"  No pending outcomes (need {OUTCOME_WINDOW_HOURS}h after signal).")
    else:
        scored = 0
        for _, row in pending.iterrows():
            pair   = row['top_pair']
            sig_ts = row['timestamp']
            out_ts = sig_ts + timedelta(hours=OUTCOME_WINDOW_HOURS)
            if '/' not in str(pair):
                continue
            parts = pair.split('/')
            if len(parts) != 2:
                continue
            base, quote = parts[0], parts[1]

            if row['mode'] == 'fiat' or quote == 'USDT':
                ep = con.execute("SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp<=? ORDER BY timestamp DESC LIMIT 1", [base, sig_ts]).fetchone()
                xp = con.execute("SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp>=? ORDER BY timestamp ASC LIMIT 1", [base, out_ts]).fetchone()
                if not ep or not xp: continue
                outcome_pct = round((xp[0] - ep[0]) / ep[0] * 100, 4)
            else:
                def _px(asset, ts, d):
                    op = 'DESC' if d == 'e' else 'ASC'
                    cm = '<=' if d == 'e' else '>='
                    return con.execute(f"SELECT price_usd FROM market_history WHERE asset_id=? AND timestamp{cm}? ORDER BY timestamp {op} LIMIT 1", [asset, ts]).fetchone()
                eb,eq,xb,xq = _px(base,sig_ts,'e'), _px(quote,sig_ts,'e'), _px(base,out_ts,'x'), _px(quote,out_ts,'x')
                if not all([eb,eq,xb,xq]): continue
                outcome_pct = round(((xb[0]/xq[0]) - (eb[0]/eq[0])) / (eb[0]/eq[0]) * 100, 4)

            con.execute("UPDATE llm_verdicts SET outcome_pct=?, outcome_ts=? WHERE timestamp=? AND top_pair=?",
                        [outcome_pct, out_ts, sig_ts, pair])
            scored += 1
        print(f"  Scored {scored} outcomes.")

    # ── Section 1: Signal Accuracy (all scored signals, including unactioned) ──
    sig_row = con.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS winners,
               ROUND(AVG(outcome_pct), 3) AS avg,
               ROUND(MAX(outcome_pct), 3) AS best,
               ROUND(MIN(outcome_pct), 3) AS worst
        FROM llm_verdicts WHERE action='accumulate' AND outcome_pct IS NOT NULL
    """).fetchone()

    print(f"\n── Signal Accuracy ({OUTCOME_WINDOW_HOURS}h window) ──")
    if sig_row and sig_row[0] > 0:
        wr = round(sig_row[1] / sig_row[0] * 100, 1)
        print(f"  Signals scored : {sig_row[0]}")
        print(f"  Signal hit rate: {sig_row[1]}/{sig_row[0]} ({wr}%)")
        print(f"  Avg outcome    : {sig_row[2]:+.3f}%")
        print(f"  Best / Worst   : {sig_row[3]:+.3f}% / {sig_row[4]:+.3f}%")
        print(f"  Note: includes unactioned signals (cooldown/cap filtered)")
    else:
        print(f"  No scored signals yet.")

    # ── Section 2: Trade Performance (actioned positions only) ──
    trade_row = con.execute("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE pnl_pct > 0) AS wins,
               COUNT(*) FILTER (WHERE pnl_pct <= 0) AS losses,
               ROUND(AVG(pnl_pct), 3) AS avg_pct,
               ROUND(MIN(pnl_pct), 3) AS worst,
               ROUND(MAX(pnl_pct), 3) AS best
        FROM fantasy_trades
        WHERE status = 'closed'
    """).fetchone()

    print(f"\n── Trade Performance (actioned positions) ──")
    if trade_row and trade_row[0] > 0:
        twr = round(trade_row[1] / trade_row[0] * 100, 1)
        print(f"  Trades executed: {trade_row[0]}")
        print(f"  Win rate       : {trade_row[1]}/{trade_row[0]} ({twr}%)")
        print(f"  Avg outcome    : {trade_row[3]:+.3f}%")
        print(f"  Best / Worst   : {trade_row[5]:+.3f}% / {trade_row[4]:+.3f}%")
    else:
        print(f"  No closed trades yet.")

    # ── Section 3: Realized P&L ──
    all_pnl   = con.execute(
        "SELECT COALESCE(ROUND(SUM(pnl_usd),4), 0) FROM fantasy_trades WHERE status='closed'"
    ).fetchone()[0]
    clean_pnl = con.execute(
        "SELECT COALESCE(ROUND(SUM(pnl_usd),4), 0) FROM fantasy_trades "
        f"WHERE status='closed' AND timestamp >= '{V08_EPOCH}'"
    ).fetchone()[0]

    print(f"\n── Realized P&L ──")
    print(f"  All-time        : {all_pnl:+.4f} USDT  (includes pre-V08 legacy losses)")
    print(f"  Since V08 clean : {clean_pnl:+.4f} USDT  (from {V08_EPOCH} onward)")

    # ── Section 4: Per-pair breakdown ──
    min_n   = 1 if all_pairs else 3
    label   = "all pairs" if all_pairs else "n≥3"
    print(f"\n── Per-pair breakdown ({label}) ──")
    breakdown = con.execute(f"""
        SELECT top_pair,
               COUNT(*) AS n,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS wins,
               ROUND(AVG(outcome_pct), 3) AS avg_out,
               ROUND(MIN(outcome_pct), 3) AS worst,
               ROUND(MAX(outcome_pct), 3) AS best
        FROM llm_verdicts
        WHERE action='accumulate' AND outcome_pct IS NOT NULL
        GROUP BY top_pair
        HAVING COUNT(*) >= {min_n}
        ORDER BY avg_out DESC
    """).df()
    if not all_pairs:
        print(f"  (use --all-pairs to include pairs with fewer than 3 signals)")
    pairs = breakdown['top_pair'].tolist()
    reps  = get_pair_reputations(con, pairs)
    breakdown['reputation'] = breakdown['top_pair'].map(
        lambda p: f"{reps.get(p, 0.0):+.3f}")
    print(breakdown.to_string(index=False))

def status(con: duckdb.DuckDBPyConnection) -> None:
    rows     = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    live     = con.execute("SELECT COUNT(*) FROM market_history WHERE source='live'").fetchone()[0]
    bf       = con.execute("SELECT COUNT(*) FROM market_history WHERE source='backfill'").fetchone()[0]
    verdicts = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    trades   = con.execute("SELECT COUNT(*) FROM fantasy_trades").fetchone()[0]
    latest   = con.execute("SELECT MAX(timestamp) FROM market_history").fetchone()[0]
    qwen_ok  = ping_llm()

    ctx = con.execute("""
        SELECT fng_value, fng_label FROM market_context ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    mood = f"{ctx[1]} (F&G {ctx[0]})" if ctx else "not yet fetched"

    # v1.4 Feature 5: trade win rate for direct comparison with --report
    wr_row = con.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE pnl_pct > 0)
        FROM fantasy_trades
        WHERE status = 'closed' AND pnl_pct IS NOT NULL
    """).fetchone()
    wr_total = wr_row[0] or 0
    wr_wins  = wr_row[1] or 0
    wr_str   = (f"{wr_wins}/{wr_total} ({round(wr_wins/wr_total*100, 1)}%)"
                if wr_total > 0 else "n/a (no closed trades yet)")

    print(f"── DB Status ──")
    print(f"  market_history : {rows} rows  (live={live}, backfill={bf}, latest={latest})")
    print(f"  llm_verdicts   : {verdicts} rows")
    print(f"  fantasy_trades : {trades} rows")
    print(f"  Trade win rate : {wr_str}  ← actioned positions, pnl non-NULL")
    print(f"  LLM            : {'✅ reachable' if qwen_ok else '⚠ unreachable'} ({config.LLM_PROVIDER}: {config.LLM_MODEL})")
    print(f"  Market mood    : {mood}")

    print("\n── Live sample counts ──")
    print(con.execute("""
        SELECT asset_id, COUNT(*) as samples,
               ROUND(AVG(spread_pct),4) as avg_spread,
               ROUND(AVG(vwap_ratio),4) as avg_vwap
        FROM market_history WHERE source='live'
        GROUP BY asset_id ORDER BY asset_id
    """).df().to_string(index=False))

    print("\n── Last 5 verdicts ──")
    print(con.execute("""
        SELECT timestamp, mode, top_pair, z_score, confidence_score,
               action, outcome_pct, reputation, fng_value
        FROM llm_verdicts ORDER BY timestamp DESC LIMIT 5
    """).df().to_string(index=False))

    print("\n── Top accumulate signals ──")
    print(con.execute("""
        SELECT mode, top_pair, COUNT(*) AS hits,
               ROUND(AVG(confidence_score),1) AS avg_conf,
               ROUND(AVG(z_score),3) AS avg_z,
               ROUND(AVG(outcome_pct),3) AS avg_outcome,
               ROUND(AVG(reputation),3) AS avg_rep,
               ROUND(AVG(fng_value),0) AS avg_fng
        FROM llm_verdicts WHERE action='accumulate'
        GROUP BY mode, top_pair ORDER BY hits DESC LIMIT 15
    """).df().to_string(index=False))

def _purge_expired_insights(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DELETE FROM report_insights WHERE expires_at <= now()")

def get_active_report_insights(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return non-expired report insights as list of dicts."""
    rows = con.execute("""
        SELECT pair, signal, magnitude, reason
        FROM report_insights
        WHERE expires_at > now()
        ORDER BY generated_at DESC
    """).fetchall()
    return [{'pair': r[0], 'signal': r[1], 'magnitude': r[2], 'reason': r[3]}
            for r in rows]

def apply_report_insights(insights: list[dict], pair: str,
                           base_confidence: int,
                           base_notional: float) -> tuple[int, float, bool]:
    """
    Given active insights, return (adjusted_confidence, adjusted_notional, suppressed).
    Matches the exact directional pair or either single leg. Every insight is
    re-sanitized here, so boost can only raise and warn can only lower, each by at
    most INSIGHT_MAX_MAGNITUDE, and a global 'ALL' row is ignored.
    """
    if not insights:
        return base_confidence, base_notional, False

    legs   = pair.split('/') if '/' in pair else [pair]
    conf   = base_confidence
    notional = base_notional
    suppressed = False

    for raw in insights:
        ins = sanitize_insight(raw)
        if ins is None:
            continue
        if not (ins['pair'] == pair or ins['pair'] in legs):
            continue
        sig = ins['signal']
        mag = abs(ins['magnitude'])
        if sig == 'suppress':
            suppressed = True
        elif sig == 'boost':
            conf     = min(10, conf + int(round(mag * 10)))
            notional = round(notional * (1.0 + mag), 4)
        elif sig == 'warn':
            conf     = max(0, conf - int(round(mag * 10)))
            notional = round(notional * max(0.3, 1.0 - mag), 4)

    return conf, notional, suppressed

def _store_report_insights(con: duckdb.DuckDBPyConnection, insights_raw: list[dict]) -> int:
    """Persist a list of parsed insights dicts, each with pair/signal/magnitude/reason."""
    now      = datetime.now(timezone.utc)
    expires  = now + timedelta(hours=6)
    stored   = 0
    for raw in insights_raw if isinstance(insights_raw, list) else []:
        ins = sanitize_insight(raw)
        if ins is None:
            print(f"  [insights] Rejected invalid insight: {str(raw)[:120]}", file=sys.stderr)
            continue
        con.execute("""
            INSERT INTO report_insights (generated_at, pair, signal, magnitude, reason, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [now, ins['pair'], ins['signal'], ins['magnitude'], ins['reason'], expires])
        stored += 1
    return stored

def _collect_report_data(con: duckdb.DuckDBPyConnection,
                          since: str | None = None) -> dict:
    prices  = get_current_prices(con)
    fg_data = get_cached_fg_data(con)

    total_trades  = con.execute("SELECT COUNT(*) FROM fantasy_trades").fetchone()[0]
    open_trades   = con.execute("SELECT COUNT(*) FROM fantasy_trades WHERE status='open'").fetchone()[0]
    closed_trades = con.execute("SELECT COUNT(*) FROM fantasy_trades WHERE status='closed'").fetchone()[0]
    realized_pnl  = con.execute(
        "SELECT COALESCE(SUM(pnl_usd), 0) FROM fantasy_trades WHERE status='closed'"
    ).fetchone()[0] or 0.0

    # Portfolio value from latest balance snapshot
    latest_bal_ts = con.execute("SELECT MAX(timestamp) FROM fantasy_balances").fetchone()[0]
    portfolio_value = 0.0
    if latest_bal_ts:
        for r in con.execute("""
            SELECT asset_id, qty FROM fantasy_balances WHERE timestamp = ?
        """, [latest_bal_ts]).fetchall():
            px = 1.0 if r[0] == 'USDT' else prices.get(r[0], 0.0)
            portfolio_value += (r[1] or 0.0) * px

    # Open positions summary
    open_pos = con.execute("""
        SELECT sell_asset, buy_asset, sell_qty, buy_qty, sell_price_usd,
               buy_price_usd, notional_usd
        FROM fantasy_trades WHERE status='open' ORDER BY timestamp DESC LIMIT 8
    """).fetchall()
    open_pos_lines = []
    for row in open_pos:
        _, pnl_pct = trade_pnl(row[2] or 0.0, asset_price(prices, row[0], row[4] or 0.0),
                               row[3] or 0.0, asset_price(prices, row[1], row[5] or 0.0),
                               row[6] or 1.0)
        open_pos_lines.append(f"{row[0]}→{row[1]}: {pnl_pct:+.2f}%")

    # Signal outcomes (includes unactioned signals)
    where = "WHERE action='accumulate' AND outcome_pct IS NOT NULL"
    if since:
        since = datetime.strptime(str(since)[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
        where += f" AND timestamp >= '{since}'"

    outcomes_row = con.execute(f"""
        SELECT COUNT(*),
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END),
               ROUND(AVG(outcome_pct), 3)
        FROM llm_verdicts {where}
    """).fetchone()
    total_sig       = outcomes_row[0] or 0
    sig_wins        = outcomes_row[1] or 0
    avg_out         = outcomes_row[2] or 0.0
    signal_hit_rate = round(sig_wins / total_sig * 100, 1) if total_sig > 0 else 0.0

    # Trade performance (actioned closed positions only, pnl_pct must be non-NULL)
    # v1.4 Feature 5: require pnl_pct IS NOT NULL to exclude closes with missing data
    trade_since = f"AND timestamp >= '{since}'" if since else ""
    trade_row = con.execute(f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE pnl_pct > 0),
               ROUND(AVG(pnl_pct), 3)
        FROM fantasy_trades
        WHERE status = 'closed' AND pnl_pct IS NOT NULL {trade_since}
    """).fetchone()
    trade_total    = trade_row[0] or 0
    trade_wins     = trade_row[1] or 0
    avg_trade_pct  = trade_row[2] or 0.0
    trade_win_rate = round(trade_wins / trade_total * 100, 1) if trade_total > 0 else 0.0

    # Clean P&L since V08
    clean_pnl = con.execute(
        "SELECT COALESCE(ROUND(SUM(pnl_usd), 4), 0) FROM fantasy_trades "
        f"WHERE status='closed' AND timestamp >= '{V08_EPOCH}'"
    ).fetchone()[0] or 0.0

    top_pairs_df = con.execute(f"""
        SELECT top_pair, COUNT(*) AS n,
               SUM(CASE WHEN outcome_pct > 0 THEN 1 ELSE 0 END) AS w,
               ROUND(AVG(outcome_pct), 3) AS avg_out
        FROM llm_verdicts {where}
        GROUP BY top_pair ORDER BY avg_out DESC LIMIT 5
    """).df()

    worst_pairs_df = con.execute(f"""
        SELECT top_pair, COUNT(*) AS n,
               ROUND(AVG(outcome_pct), 3) AS avg_out
        FROM llm_verdicts {where}
        GROUP BY top_pair ORDER BY avg_out ASC LIMIT 3
    """).df()

    fg_off   = fg_data.get('fg_official', 50)
    fg_synth = fg_data.get('fg_synthetic', fg_off)

    return {
        'total_trades':     total_trades,
        'open_trades':      open_trades,
        'closed_trades':    closed_trades,
        'realized_pnl':     realized_pnl,
        'clean_pnl':        clean_pnl,
        'portfolio_value':  portfolio_value,
        'open_positions':   ', '.join(open_pos_lines) if open_pos_lines else 'None',
        # signal-level (includes unactioned)
        'total_signals':    total_sig,
        'signal_hit_rate':  signal_hit_rate,
        'sig_wins':         sig_wins,
        'avg_outcome':      avg_out,
        # trade-level (actioned positions only)
        'trade_total':      trade_total,
        'trade_wins':       trade_wins,
        'trade_win_rate':   trade_win_rate,
        'avg_trade_pct':    avg_trade_pct,
        'top_pairs_df':     top_pairs_df,
        'worst_pairs_df':   worst_pairs_df,
        'fg_official':      fg_off,
        'fg_synthetic':     fg_synth,
        'fg_retail':        fg_data.get('fg_retail', fg_off),
        'fg_inst':          fg_data.get('fg_institutional', 50),
        'fg_onchain':       fg_data.get('fg_onchain', 50),
        'fg_trader':        fg_data.get('fg_trader', 50),
        'divergence':       abs(fg_synth - fg_off),
    }

def run_report(con: duckdb.DuckDBPyConnection,
               since: str | None = None,
               post_discord: bool = False,
               write_insights: bool = False):
    print(f"[{datetime.now():%H:%M:%S}] Generating Osmia daily report …")
    data = _collect_report_data(con, since)

    # v1.4 Feature 5: cross-check trade win rate vs --status computation
    status_trade_row = con.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE pnl_pct > 0)
        FROM fantasy_trades
        WHERE status = 'closed' AND pnl_pct IS NOT NULL
    """).fetchone()
    status_total = status_trade_row[0] or 0
    status_wins  = status_trade_row[1] or 0
    status_wr    = round(status_wins / status_total * 100, 1) if status_total > 0 else 0.0
    wr_delta = abs(data['trade_win_rate'] - status_wr)
    if wr_delta > 10:
        print(f"  ⚠ Win rate divergence: report={data['trade_win_rate']}% "
              f"vs status={status_wr}% (delta {wr_delta:.1f}pp — check query population)",
              file=sys.stderr)

    top_pairs_str = ""
    for _, r in data['top_pairs_df'].iterrows():
        wr = round(r['w'] / r['n'] * 100, 0) if r['n'] > 0 else 0
        top_pairs_str += f"  {r['top_pair']}: {int(r['n'])} signals, {wr:.0f}% win rate, avg {r['avg_out']:+.3f}%\n"

    worst_pairs_str = ""
    for _, r in data['worst_pairs_df'].iterrows():
        worst_pairs_str += f"  {r['top_pair']}: avg {r['avg_out']:+.3f}% ({int(r['n'])} signals)\n"

    prompt = f"""You are analyzing a cryptocurrency pair trading bot's performance.
Provide a concise, actionable daily recap in plain text.

[SYSTEM STATUS]
Trades: {data['total_trades']} total | {data['open_trades']} open | {data['closed_trades']} closed
All-time realized P&L: {data['realized_pnl']:+.2f} USDT (includes pre-V08 legacy losses)
Clean P&L since V08: {data['clean_pnl']:+.2f} USDT (stop-loss era, meaningful baseline)
Portfolio value: {data['portfolio_value']:.2f} USDT

[SIGNAL ACCURACY — academic, includes unactioned signals]
Signal hit rate: {data['signal_hit_rate']}% ({data['sig_wins']}/{data['total_signals']})
Avg signal outcome: {data['avg_outcome']:+.3f}%

[TRADE PERFORMANCE — actioned positions only]
Trade win rate: {data['trade_win_rate']}% ({data['trade_wins']}/{data['trade_total']})
Avg trade outcome: {data['avg_trade_pct']:+.3f}%

[TOP PERFORMING PAIRS]
{top_pairs_str or '  No scored pairs yet.'}
[PROBLEM PAIRS - negative avg outcome]
{worst_pairs_str or '  None.'}
[OPEN POSITIONS]
{data['open_positions']}

[MACRO CONTEXT]
Official F&G: {data['fg_official']} | Synthetic: {data['fg_synthetic']}
Cohorts — Retail: {data['fg_retail']} | Inst: {data['fg_inst']} | On-chain: {data['fg_onchain']} | Trader: {data['fg_trader']}
Divergence: {data['divergence']} points

Provide:
1. Portfolio health summary (2-3 sentences, use clean P&L not all-time)
2. Top 3 performing pairs and why they work
3. Problem positions requiring attention
4. Trade win rate assessment (use trade win rate, not signal hit rate)
5. Macro context impact on current strategy
6. 2-3 specific actionable recommendations

Be direct and terse. No fluff. Flag anything urgent.
"""

    print("  Asking Qwen for analysis …")
    con.close()
    analysis = ask_report(prompt)
    con = get_conn()

    now_str    = datetime.now().strftime('%Y-%m-%d')
    fg_off     = data['fg_official']
    fg_syn     = data['fg_synthetic']
    bolt       = " ⚡" if data['divergence'] >= DIVERGENCE_THRESHOLD else ""
    sep        = "═" * 38

    print(f"\n{sep}")
    print(f"  OSMIA DAILY REPORT — {now_str}")
    print(sep)
    print()
    print("[STATUS SNAPSHOT]")
    print(f"  Trades: {data['total_trades']} total | {data['open_trades']} open | {data['closed_trades']} closed")
    print(f"  Trade win rate: {data['trade_win_rate']}% ({data['trade_wins']}/{data['trade_total']})  ← actioned positions")
    print(f"  Signal hit rate: {data['signal_hit_rate']}% ({data['sig_wins']}/{data['total_signals']})  ← incl. unactioned")
    print(f"  Realized P&L (all-time): {data['realized_pnl']:+.2f} USDT")
    print(f"  Clean P&L since V08:     {data['clean_pnl']:+.2f} USDT")
    print(f"  Portfolio value: {data['portfolio_value']:.2f} USDT")
    print(f"  Market mood: Official F&G {fg_off} | Synthetic {fg_syn}{bolt}")
    print()
    print("[LLM ANALYSIS]")
    print(analysis)
    print()
    print(sep)

    if post_discord:
        msg = (
            f"📊 **OSMIA Daily Report — {now_str}**\n"
            f"Trades: {data['total_trades']} | Open: {data['open_trades']} "
            f"| Trade win rate: {data['trade_win_rate']}%\n"
            f"Clean P&L (V08+): {data['clean_pnl']:+.2f} USDT  "
            f"| F&G: {fg_off} (Official) / {fg_syn} (Synthetic){bolt}\n\n"
            f"```{analysis[:1500]}```"
        )
        discord_send(msg)
        print("  📨 Report posted to Discord.")

    # v1.4 Feature 4: write structured pair insights to report_insights table
    if write_insights:
        print("  Asking Qwen for pair insights …")
        _purge_expired_insights(con)
        raw_insights = ask_insights(data)
        if raw_insights:
            n = _store_report_insights(con, raw_insights)
            print(f"  Stored {n} pair insight(s) in report_insights (6h TTL).")
        else:
            print("  No actionable pair insights generated.")

    return con
