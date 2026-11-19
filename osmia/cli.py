"""Command-line entry point (python -m osmia ...)."""
import argparse
import sys
from datetime import datetime

from . import __version__
from .agent import analyze_fiat, analyze_pairs
from .config import ASSETS, MIN_POSITION_USD, MIN_PROFIT_USD, UNLOCK_ASSETS
from .learning import run_report, score_outcomes, status
from .mining import inject_mine_proceeds, scrape_unmineable
from .notify import discord_daily_pnl
from .portfolio import run_settle_dust, show_portfolio
from .sources.market import backfill, ingest
from .sources.news import fetch_all_news, fetch_unlocks, store_asset_news, store_unlocks
from .sources.sentiment import run_fetch_fg, set_cohort_weights
from .store import get_conn, get_current_prices, init_db


def _utf8_console() -> None:
    """Output uses emoji and symbols; a Windows cp1252 console would crash on them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or '').lower().replace('-', '') != 'utf8':
                stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = argparse.ArgumentParser(prog='osmia', description=f"Osmia v{__version__}")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--ingest',         action='store_true')
    grp.add_argument('--analyze',        action='store_true')
    grp.add_argument('--fiat',           action='store_true')
    grp.add_argument('--backfill',       action='store_true')
    grp.add_argument('--outcomes',       action='store_true')
    grp.add_argument('--portfolio',      action='store_true')
    grp.add_argument('--daily-pnl',      action='store_true')
    grp.add_argument('--init-db',        action='store_true')
    grp.add_argument('--status',         action='store_true')
    grp.add_argument('--scrape-news',    action='store_true')
    grp.add_argument('--scrape-unlocks', action='store_true')
    grp.add_argument('--mine-proceeds',  action='store_true')
    grp.add_argument('--mine-auto',      action='store_true',
                     help='Auto-scrape unMineable API for new payments and inject')
    grp.add_argument('--fetch-fg',       action='store_true',
                     help='Fetch all 4 F&G cohorts and store to macro_cache + fg_history')
    grp.add_argument('--settle-dust',    action='store_true',
                     help='Convert sub-$0.10 fantasy balances to USDT at current price')
    grp.add_argument('--report',         action='store_true',
                     help='Generate LLM-written performance report')
    grp.add_argument('--run',            action='store_true',
                     help='Run every job on its schedule in this one process (Ctrl-C to stop)')
    grp.add_argument('--backup',         action='store_true',
                     help='Checkpoint and copy the DuckDB file to OSMIA_HOME/backups')
    grp.add_argument('--research',       action='store_true',
                     help='Run one research cycle: propose, pre-register and test hypotheses')
    grp.add_argument('--demo-load',      type=str, default=None, metavar='CSV',
                     help='Import a price CSV(.gz) (e.g. demo/prices_5m.csv.gz) into market_history')
    parser.add_argument('--full',           action='store_true',
                        help='Show full trade history with --portfolio (default: last 7 days)')
    parser.add_argument('--all-pairs',     action='store_true',
                        help='Show all pairs in --outcomes breakdown (default: n≥3 only)')
    parser.add_argument('--size',         type=float, default=None)
    parser.add_argument('--position-pct', type=float, default=None,
                        metavar='PCT',
                        help='Base position size as %% of portfolio (e.g. 1.0 = 1%%)')
    parser.add_argument('--min-position', type=float, default=None,
                        metavar='USD',
                        help=f'Hard floor per trade in USDT (default: {MIN_POSITION_USD})')
    parser.add_argument('--min-profit',   type=float, default=None,
                        metavar='USD',
                        help=f'Skip trade if predicted profit < this (default: {MIN_PROFIT_USD})')
    parser.add_argument('--mine-value',   type=float, default=10.0)
    parser.add_argument('--set-weights',  type=str,   default=None,
                        metavar='retail=0.25,institutional=0.30,onchain=0.25,trader=0.20',
                        help='Set cohort weights (must sum to 1.0)')
    parser.add_argument('--post-discord', action='store_true',
                        help='Post report to Discord (use with --report)')
    parser.add_argument('--since',        type=str,   default=None,
                        metavar='DATE',
                        help='Limit outcomes window (YYYY-MM-DD, use with --report)')
    parser.add_argument('--write-insights', action='store_true',
                        help='Store LLM pair insights to report_insights table (use with --report)')
    parser.add_argument('--hypotheses',   type=int,   default=3,
                        help='Hypotheses per research cycle (use with --research)')
    parser.add_argument('--holdout-frac', type=float, default=0.3,
                        help='Most recent fraction of data held out for verdicts (--research)')
    parser.add_argument('--research-days', type=float, default=None,
                        help='Only use the last N days of data (--research)')
    parser.add_argument('--no-llm',       action='store_true',
                        help='Research with the heuristic proposer only')
    args = parser.parse_args(argv)

    if args.run:
        from .scheduler import run_forever
        run_forever()
        return
    if args.backup:
        from .backup import run_backup
        run_backup()
        return
    if args.research:
        from .research.loop import run_research_cycle
        con = get_conn()
        init_db(con)
        con.close()
        run_research_cycle(n_hypotheses=max(1, min(args.hypotheses, 8)),
                           holdout_frac=min(max(args.holdout_frac, 0.1), 0.6),
                           use_llm=False if args.no_llm else None,
                           days=args.research_days)
        return

    min_pos    = args.min_position if args.min_position is not None else MIN_POSITION_USD
    min_profit = args.min_profit   if args.min_profit   is not None else MIN_PROFIT_USD

    con = get_conn()
    init_db(con)

    # --set-weights can combine with any mode (runs first, then exits)
    if args.set_weights:
        try:
            set_cohort_weights(con, args.set_weights)
        except ValueError as e:
            print(f"  Error: {e}", file=sys.stderr)
            con.close()
            sys.exit(1)
        con.close()
        return

    if   args.init_db:      print("DB schema initialized.")
    elif args.demo_load:
        from .research.data import import_price_csv
        n = import_price_csv(con, args.demo_load)
        print(f"  Imported {n} price rows from {args.demo_load}.")
    elif args.status:       status(con)
    elif args.backfill:     backfill(con)
    elif args.outcomes:     score_outcomes(con, all_pairs=args.all_pairs)
    elif args.portfolio:    show_portfolio(con, full=args.full)
    elif args.daily_pnl:    discord_daily_pnl(con, get_current_prices(con))
    elif args.ingest:       ingest(con)
    elif args.analyze:      analyze_pairs(con, args.size, args.position_pct,
                                          min_pos, min_profit)
    elif args.fiat:         analyze_fiat(con, args.size, min_pos, min_profit)
    elif args.settle_dust:
        print(f"[{datetime.now():%H:%M:%S}] Settling dust balances …")
        run_settle_dust(con)
    elif args.report:
        con = run_report(con, args.since, args.post_discord, args.write_insights)
    elif args.fetch_fg:
        con = run_fetch_fg(con)
    elif args.scrape_news:
        print(f"[{datetime.now():%H:%M:%S}] Scraping per-asset news …")
        con.close()
        rows = fetch_all_news(ASSETS)
        con  = get_conn()
        print(f"  Stored {store_asset_news(con, rows)} news rows for {len(ASSETS)} assets.")
    elif args.mine_proceeds: inject_mine_proceeds(con, args.mine_value)
    elif args.mine_auto:
        print(f"[{datetime.now():%H:%M:%S}] Auto-scraping unMineable payments …")
        scrape_unmineable(con)
    elif args.scrape_unlocks:
        print(f"[{datetime.now():%H:%M:%S}] Scraping token unlocks …")
        con.close()
        rows = fetch_unlocks(UNLOCK_ASSETS)
        con  = get_conn()
        store_unlocks(con, rows)
        total = con.execute("SELECT COUNT(*) FROM token_unlocks WHERE unlock_at > now()").fetchone()[0]
        print(f"  Processed {len(rows)} unlock events; {total} upcoming rows in DB for {UNLOCK_ASSETS}.")
    else:
        ingest(con)
        analyze_pairs(con, args.size, args.position_pct, min_pos, min_profit)

    if con:
        try:
            con.close()
        except Exception:
            pass
