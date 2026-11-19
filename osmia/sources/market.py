"""Exchange data: tickers ingest, OHLCV backfill, order book, watchlist."""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import duckdb
import numpy as np
import pandas as pd
import requests

from .. import config
from ..config import (
    ASSETS, BACKFILL_DAYS, BACKFILL_TIMEFRAME, COINGECKO_MARKETS_URL,
    MAD_CONSISTENCY, NON_BINANCE_TOKENS, ORDER_BOOK_DEPTH,
    ORDER_BOOK_SKIP_SPREAD_PCT, ORDER_BOOK_SLIPPAGE_SIZE,
    ORDER_BOOK_THIN_SPREAD_PCT, STABLECOINS, SYMBOLS, USER_AGENT, VOLUME_WINDOW,
    WATCHLIST_CACHE_HOURS,
)
from ..store import _get_macro_cache, _set_macro_cache


def make_exchange(rate_limit: bool = True):
    """ccxt client for config.EXCHANGE_ID (default 'binance'). Binance returns
    HTTP 451 to US and most cloud IPs; set OSMIA_EXCHANGE=binanceus/kraken/... there."""
    try:
        cls = getattr(ccxt, config.EXCHANGE_ID)
    except AttributeError:
        raise SystemExit(f"Unknown OSMIA_EXCHANGE '{config.EXCHANGE_ID}' (must be a ccxt exchange id)")
    return cls({'enableRateLimit': rate_limit})

def _default_ob_metrics() -> dict:
    return {
        'spread_real': 0.0,
        'liquidity_score': 0.5,
        'est_slippage': 0.0,
        'is_thin': False,
        'bid_depth_usd': 0.0,
        'ask_depth_usd': 0.0,
        'skip': False,
    }

def fetch_order_book_metrics(base: str, quote: str, notional: float) -> dict:
    """
    Fetch live order book for a pair via ccxt Binance (no auth needed for public data).
    Returns: spread_real (ratio), liquidity_score (0-1), est_slippage (ratio),
             is_thin (bool), skip (bool — True when spread > hard threshold).
    Never raises — returns defaults on failure.
    """
    symbol = f"{base}/{quote}" if quote != 'USDT' else f"{base}/USDT"
    # For 2-leg synthetic pairs: use the base asset's USDT book as a proxy
    if quote not in ('USDT', 'BTC', 'ETH', 'BNB'):
        symbol = f"{base}/USDT"
    try:
        exchange = make_exchange(rate_limit=False)
        ob = exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
    except Exception as e:
        print(f"  [orderbook] {symbol} fetch failed: {e}", file=sys.stderr)
        return _default_ob_metrics()

    bids = ob.get('bids') or []
    asks = ob.get('asks') or []
    if not bids or not asks:
        return _default_ob_metrics()

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if best_bid <= 0 or best_ask <= 0:
        return _default_ob_metrics()

    mid_price = (best_bid + best_ask) / 2.0
    spread_real = (best_ask - best_bid) / mid_price

    bid_depth_usd = sum(p * q for p, q in bids)
    ask_depth_usd = sum(p * q for p, q in asks)
    total_depth_usd = bid_depth_usd + ask_depth_usd

    # Liquidity score: 1.0 when book depth >= 20× the notional
    liquidity_score = min(1.0, total_depth_usd / max(1.0, notional * 20))

    # Walk asks to estimate slippage on large fills
    est_slippage = 0.0
    if notional > ORDER_BOOK_SLIPPAGE_SIZE:
        remaining = notional
        filled = 0.0
        fill_cost = 0.0
        for px, qty in asks:
            usd_available = px * qty
            take = min(remaining, usd_available)
            fill_cost += take
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled > 0:
            avg_fill_px = fill_cost / (filled / mid_price) / mid_price if mid_price else 1.0
            est_slippage = max(0.0, (avg_fill_px - 1.0))  # ratio above mid

    is_thin = spread_real > (ORDER_BOOK_THIN_SPREAD_PCT / 100)
    skip    = spread_real > (ORDER_BOOK_SKIP_SPREAD_PCT / 100)

    return {
        'spread_real': spread_real,
        'liquidity_score': round(liquidity_score, 4),
        'est_slippage': round(est_slippage, 6),
        'is_thin': is_thin,
        'skip': skip,
        'bid_depth_usd': round(bid_depth_usd, 2),
        'ask_depth_usd': round(ask_depth_usd, 2),
    }

def _volume_mad_zscore(con, new_df: pd.DataFrame) -> dict[str, float]:
    results = {}
    for _, row in new_df.iterrows():
        asset   = row['asset_id']
        cur_vol = row['volume_24h']
        hist    = con.execute(
            "SELECT volume_24h FROM market_history "
            "WHERE asset_id = ? AND source = 'live' ORDER BY timestamp DESC LIMIT ?",
            [asset, VOLUME_WINDOW]
        ).fetchdf()
        if len(hist) < 5:
            results[asset] = 0.0
            continue
        v      = hist['volume_24h'].values
        median = np.median(v)
        mad    = np.median(np.abs(v - median))
        denom  = MAD_CONSISTENCY * mad
        results[asset] = float((cur_vol - median) / denom) if denom > 0 else 0.0
    return results

def ingest(con: duckdb.DuckDBPyConnection) -> int:
    print(f"[{datetime.now():%H:%M:%S}] Fetching tickers …")
    # v0.9: use top-50 watchlist (24h cached in macro_cache)
    watchlist = get_watchlist(con)
    symbols   = [f"{a}/USDT" for a in watchlist]
    print(f"  Watchlist: {len(watchlist)} assets (top-50 CoinGecko)")
    exchange = make_exchange(rate_limit=True)
    try:
        tickers = exchange.fetch_tickers(symbols)
    except Exception as e:
        print(f"  ccxt error: {e}", file=sys.stderr)
        return 0

    ts   = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for symbol in symbols:
        t = tickers.get(symbol)
        if not t:
            continue
        asset = symbol.split('/')[0]
        last  = t.get('last')  or 0.0
        ask   = t.get('ask')   or 0.0
        bid   = t.get('bid')   or 0.0
        mid   = (ask + bid) / 2 if (ask and bid) else last
        vwap  = t.get('vwap')  or 0.0
        high  = t.get('high')  or 0.0
        low   = t.get('low')   or 0.0
        rows.append({
            'timestamp':    ts,
            'asset_id':     asset,
            'price_usd':    last,
            'volume_24h':   t.get('quoteVolume') or 0.0,
            'change_24h':   t.get('percentage')  or 0.0,
            'market_cap':   0.0,
            'vwap_ratio':   round(last / vwap,             6) if vwap else None,
            'spread_pct':   round((ask - bid) / mid * 100, 6) if mid  else None,
            'high_low_pct': round((high - low) / low * 100, 6) if low  else None,
            'volume_z':     None,
            'source':       'live',
        })

    if not rows:
        print("  No ticker data returned.", file=sys.stderr)
        return 0

    df  = pd.DataFrame(rows)
    vzs = _volume_mad_zscore(con, df)
    df['volume_z'] = df['asset_id'].map(vzs)

    con.execute("""
        INSERT OR IGNORE INTO market_history
            (timestamp, asset_id, price_usd, volume_24h, change_24h,
             market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source)
        SELECT timestamp, asset_id, price_usd, volume_24h, change_24h,
               market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source
        FROM df
    """)
    print(f"  Ingested {len(df)} records @ {ts:%Y-%m-%d %H:%M} UTC")
    return len(df)

def backfill(con: duckdb.DuckDBPyConnection) -> None:
    exchange = make_exchange(rate_limit=True)
    since_ms = int((datetime.now(timezone.utc) -
                    timedelta(days=BACKFILL_DAYS)).timestamp() * 1000)
    print(f"Backfilling {BACKFILL_DAYS} days of OHLCV for {len(ASSETS)} assets …")
    total = 0
    for symbol in SYMBOLS:
        asset = symbol.split('/')[0]
        print(f"  {asset:>6s} … ", end='', flush=True)
        rows, cursor, retries = [], since_ms, 0
        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    symbol, timeframe=BACKFILL_TIMEFRAME, since=cursor, limit=1000)
            except Exception as e:
                if retries < 3:
                    retries += 1
                    time.sleep(5)
                    continue
                print(f"failed ({e})")
                break
            if not candles:
                break
            for c in candles:
                ts = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).replace(second=0, microsecond=0)
                rows.append({
                    'timestamp': ts, 'asset_id': asset,
                    'price_usd': float(c[4]), 'volume_24h': float(c[5]),
                    'change_24h': None, 'market_cap': 0.0,
                    'vwap_ratio': None, 'volume_z': None, 'spread_pct': None,
                    'high_low_pct': round((float(c[2]) - float(c[3])) / float(c[3]) * 100, 6) if c[3] else None,
                    'source': 'backfill',
                })
            cursor = candles[-1][0] + 1
            if len(candles) < 1000:
                break
            time.sleep(0.2)
        if rows:
            df = pd.DataFrame(rows)
            con.execute("""
                INSERT OR IGNORE INTO market_history
                    (timestamp, asset_id, price_usd, volume_24h, change_24h,
                     market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source)
                SELECT timestamp, asset_id, price_usd, volume_24h, change_24h,
                       market_cap, vwap_ratio, volume_z, spread_pct, high_low_pct, source
                FROM df
            """)
            print(f"{len(rows):,} rows")
            total += len(rows)
        else:
            print("0 rows")
    print(f"\nBackfill complete. {total:,} rows. DB total: "
          f"{con.execute('SELECT COUNT(*) FROM market_history').fetchone()[0]:,}")

def fetch_top50_watchlist() -> list[str]:
    """Fetch top ~50 non-stablecoin symbols tradeable on Binance as USDT pairs.

    Steps:
    1. Pull top-200 by market cap from CoinGecko.
    2. Strip stablecoins and known non-Binance tokens.
    3. Validate remaining symbols against live Binance USDT markets.
    4. Return up to 50, preserving market-cap rank order.
    """
    try:
        # --- Step 1: fetch 200 candidates from CoinGecko (2 pages of 100) ---
        candidates: list[str] = []
        for page in (1, 2):
            r = requests.get(
                COINGECKO_MARKETS_URL,
                params={'vs_currency': 'usd', 'order': 'market_cap_desc',
                        'per_page': 100, 'page': page, 'sparkline': 'false'},
                timeout=15, headers={'User-Agent': USER_AGENT})
            r.raise_for_status()
            for c in r.json():
                if not isinstance(c, dict) or not c.get('symbol'):
                    continue
                sym = c['symbol'].upper()
                if (sym not in STABLECOINS
                        and sym not in NON_BINANCE_TOKENS
                        and sym.isascii() and sym.isalnum()
                        and 2 <= len(sym) <= 10):
                    candidates.append(sym)

        # --- Step 2: load Binance USDT markets once ---
        exchange = make_exchange(rate_limit=True)
        markets  = exchange.load_markets()
        exchange_usdt = {s.split('/')[0] for s in markets if s.endswith('/USDT')}

        # --- Step 3: filter to validated symbols, preserving rank order ---
        validated: list[str] = []
        seen: set[str] = set()
        for sym in candidates:
            if sym in exchange_usdt and sym not in seen:
                seen.add(sym)
                validated.append(sym)
            if len(validated) >= 50:
                break

        print(f"  Watchlist: {len(candidates)} CoinGecko candidates → "
              f"{len(validated)} Binance-validated USDT pairs")
        return validated

    except Exception as e:
        print(f"  Top-50 watchlist fetch failed: {e}", file=sys.stderr)
        return []

def get_watchlist(con) -> list[str]:
    """Return top-50 watchlist from macro_cache (24h TTL) or fetch fresh."""
    cached = _get_macro_cache(con, 'watchlist_top50', WATCHLIST_CACHE_HOURS * 3600)
    if cached:
        try:
            result = json.loads(cached)
            if isinstance(result, list) and result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass
    watchlist = fetch_top50_watchlist()
    if watchlist:
        _set_macro_cache(con, 'watchlist_top50', json.dumps(watchlist))
        return watchlist
    return list(ASSETS)   # fallback to hardcoded list
