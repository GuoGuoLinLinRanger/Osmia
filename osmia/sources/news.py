"""News, token unlocks and keyword-based news signals."""
import json
import re
import sys
from datetime import datetime, timedelta, timezone

import duckdb
import feedparser
import requests

from ..config import (
    ASSET_NAMES, CHAINS_URL, DEFILLAMA_CHAINS, EMISSION_SLUGS, EMISSION_URL,
    FNG_URL, HEIGHTENED_POST_UNLOCK_HRS, NEWS_COUNT, NEWS_FEED, NEWS_LOOKBACK_HOURS,
    NEWS_PROMPT_COUNT, NEWS_PROMPT_HOURS, NEWS_VETO_KEYWORDS, REDDIT_LIMIT,
    SUBREDDITS, UNLOCK_HORIZON_HOURS, USER_AGENT, _NEWS_NEGATIVE, _NEWS_POSITIVE,
)
from ..core import _as_utc, count_keywords, mentions_asset
from ..store import get_bot_state, set_bot_state


def compute_news_sentiment_score(con, asset: str, hours: int = 6) -> float:
    """Keyword-based sentiment from recent headlines: -1.0 to +1.0."""
    items = get_asset_news(con, asset, hours, 20)
    if not items:
        return 0.0
    pos = neg = 0
    for headline, _, _ in items:
        pos += count_keywords(headline, _NEWS_POSITIVE)
        neg += count_keywords(headline, _NEWS_NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))

def fetch_market_context() -> dict:
    """
    Fetches Fear & Greed index and top crypto headlines.
    Returns dict with fng_value, fng_label, headlines (list of str).
    Never raises — returns neutral defaults on failure.
    """
    ctx = {'fng_value': 50, 'fng_label': 'Neutral', 'headlines': []}

    # Fear & Greed
    try:
        r = requests.get(FNG_URL, timeout=10)
        data = r.json()['data'][0]
        ctx['fng_value'] = int(data['value'])
        ctx['fng_label'] = data['value_classification']
    except Exception as e:
        print(f"  F&G fetch failed: {e}", file=sys.stderr)

    # Headlines from CoinDesk RSS
    try:
        r = requests.get(NEWS_FEED, timeout=10,
                         headers={'User-Agent': 'Mozilla/5.0'})
        titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', r.text)
        # Skip first (feed title)
        ctx['headlines'] = [t.strip() for t in titles[1:NEWS_COUNT + 1]]
    except Exception as e:
        print(f"  Headlines fetch failed: {e}", file=sys.stderr)

    return ctx

def store_market_context(con: duckdb.DuckDBPyConnection, ctx: dict) -> None:
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    con.execute("""
        INSERT OR IGNORE INTO market_context (timestamp, fng_value, fng_label, headlines)
        VALUES (?, ?, ?, ?)
    """, [ts, ctx['fng_value'], ctx['fng_label'],
          json.dumps(ctx['headlines'])])

def check_heightened_mode(con, asset_id: str,
                          horizon_hours: int = UNLOCK_HORIZON_HOURS) -> tuple[bool, str]:
    """
    (is_heightened, unlock_context_str) for an asset.

    True when an unlock falls within horizon_hours, OR we are still inside the
    post-unlock relief window persisted in bot_state (heightened_<ASSET>_until =
    unlock_at + HEIGHTENED_POST_UNLOCK_HRS). Refreshes that state on each call.
    """
    now        = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=horizon_hours)

    rows = con.execute("""
        SELECT unlock_at, amount_usd, pct_supply, unlock_type
        FROM token_unlocks
        WHERE asset_id = ? AND unlock_at BETWEEN ? AND ?
        ORDER BY unlock_at ASC
        LIMIT 3
    """, [asset_id, now, window_end]).fetchall()

    if rows:
        lines, latest_until = [], None
        for unlock_at, amount_usd, pct_supply, utype in rows:
            ua  = _as_utc(unlock_at)
            hrs = (ua - now).total_seconds() / 3600
            amt = f"${amount_usd / 1e6:.1f}M" if amount_usd else "$?"
            pct = f"{pct_supply:.2f}% supply" if pct_supply is not None else "?% supply"
            lines.append(f"⚠️ {asset_id} UNLOCK in {hrs:.1f}h: {amt} ({pct}) [{utype or 'unlock'}]")
            until = ua + timedelta(hours=HEIGHTENED_POST_UNLOCK_HRS)
            latest_until = until if (latest_until is None or until > latest_until) else latest_until
        if latest_until:
            set_bot_state(con, f"heightened_{asset_id}_until", latest_until.isoformat())
        return True, "\n".join(lines)

    # No upcoming unlock — check the persisted post-unlock relief window.
    until_s = get_bot_state(con, f"heightened_{asset_id}_until")
    if until_s:
        try:
            until = _as_utc(datetime.fromisoformat(until_s))
            if now <= until:
                return True, (f"⚠️ {asset_id} POST-UNLOCK relief window active "
                              f"(through {until:%Y-%m-%d %H:%M} UTC)")
        except ValueError:
            pass
    return False, ""

def fetch_all_news(assets: list[str]) -> list[dict]:
    """
    Pure-network scrape (no DB held) → list of {asset_id, source, headline,
    published_at}. Each source degrades independently; never raises.
      • CoinDesk RSS  — fetched once, keyword-filtered per asset
      • Reddit        — r/<sub>/new.json (may 403 from datacenter IPs)
      • DeFiLlama TVL — chains only, snapshot headline
    """
    now  = datetime.now(timezone.utc)
    rows = []

    # CoinDesk RSS — one fetch, filtered per asset below.
    coindesk = []
    try:
        fp = feedparser.parse(NEWS_FEED)
        for e in fp.entries:
            title = (getattr(e, 'title', '') or '').strip()
            summ  = (getattr(e, 'summary', '') or '')
            pub   = None
            if getattr(e, 'published_parsed', None):
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            if title:
                coindesk.append((title, summ, pub))
    except Exception as e:
        print(f"  CoinDesk RSS failed: {e}", file=sys.stderr)

    # DeFiLlama chains — one fetch, looked up per asset below.
    chains = {}
    try:
        arr = requests.get(CHAINS_URL, timeout=12,
                           headers={'User-Agent': USER_AGENT}).json()
        chains = {(c.get('name') or '').lower(): c for c in arr}
    except Exception as e:
        print(f"  DeFiLlama chains failed: {e}", file=sys.stderr)

    for asset in assets:
        name = ASSET_NAMES.get(asset, asset)

        for title, summ, pub in coindesk:
            if mentions_asset(f"{title} {summ}", asset, name):
                rows.append({'asset_id': asset, 'source': 'coindesk',
                             'headline': title[:300], 'published_at': pub})

        sub = SUBREDDITS.get(asset)
        if sub:
            try:
                r = requests.get(f"https://www.reddit.com/r/{sub}/new.json?limit={REDDIT_LIMIT}",
                                 headers={'User-Agent': USER_AGENT}, timeout=12)
                if r.status_code == 200:
                    for child in r.json().get('data', {}).get('children', []):
                        d       = child.get('data', {})
                        title   = (d.get('title') or '').strip()
                        created = d.get('created_utc')
                        pub     = (datetime.fromtimestamp(created, tz=timezone.utc)
                                   if created else None)
                        if title:
                            rows.append({'asset_id': asset, 'source': 'reddit',
                                         'headline': title[:300], 'published_at': pub})
                else:
                    print(f"  Reddit r/{sub} HTTP {r.status_code} (skipped)", file=sys.stderr)
            except Exception as e:
                print(f"  Reddit r/{sub} failed: {e}", file=sys.stderr)

        chain_name = DEFILLAMA_CHAINS.get(asset)
        if chain_name:
            c = chains.get(chain_name.lower())
            if c and c.get('tvl') is not None:
                rows.append({'asset_id': asset, 'source': 'defillama',
                             'headline': f"{chain_name} TVL: ${c['tvl'] / 1e6:.1f}M",
                             'published_at': now})

    return rows

def store_asset_news(con, rows: list[dict]) -> int:
    """Insert scraped news (dedup via PK), then prune anything older than 24h."""
    now = datetime.now(timezone.utc)
    for r in rows:
        con.execute("""
            INSERT OR IGNORE INTO asset_news
                (fetched_at, asset_id, source, headline, published_at)
            VALUES (?, ?, ?, ?, ?)
        """, [now, r['asset_id'], r['source'], r['headline'], r.get('published_at')])
    con.execute("DELETE FROM asset_news WHERE fetched_at < ?",
                [now - timedelta(hours=NEWS_LOOKBACK_HOURS)])
    return len(rows)

def get_asset_news(con, asset: str, hours: int, limit: int) -> list[tuple]:
    """Most-recent distinct headlines for an asset within `hours`. → [(headline, source, when)]."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    raw = con.execute("""
        SELECT headline, source, published_at, fetched_at
        FROM asset_news
        WHERE asset_id = ? AND COALESCE(published_at, fetched_at) >= ?
        ORDER BY COALESCE(published_at, fetched_at) DESC
    """, [asset, cutoff]).fetchall()
    seen, out = set(), []
    for headline, source, pub, fetched in raw:
        if headline in seen:
            continue
        seen.add(headline)
        out.append((headline, source, pub or fetched))
        if len(out) >= limit:
            break
    return out

def format_asset_news_block(con, asset: str,
                            hours: int = NEWS_PROMPT_HOURS,
                            limit: int = NEWS_PROMPT_COUNT) -> str:
    items = get_asset_news(con, asset, hours, limit)
    if not items:
        return ""
    now   = datetime.now(timezone.utc)
    lines = [f"[{asset} NEWS - last {hours}h]"]
    for headline, source, when in items:
        if when:
            ago   = (now - _as_utc(when)).total_seconds() / 3600
            ago_s = f"{ago:.0f}h ago" if ago >= 1 else "<1h ago"
        else:
            ago_s = "recent"
        lines.append(f'- "{headline}" ({source}, {ago_s})')
    return "\n".join(lines)

def fetch_unlocks(assets: list[str]) -> list[dict]:
    """
    Best-effort upcoming-unlock fetch from DeFiLlama's free emissions dataset.
    Pure network, no DB. Returns [] (never raises) if the source is unavailable
    or its shape changes. Only future events are kept; pct_supply is derived from
    supplyMetrics.maxSupply when available; amount_usd is left NULL (not in the
    free feed). Rows that can't be parsed are skipped rather than guessed.
    """
    now = datetime.now(timezone.utc)
    out = []
    for asset in assets:
        slug = EMISSION_SLUGS.get(asset)
        if not slug:
            continue
        try:
            r = requests.get(EMISSION_URL.format(slug=slug), timeout=20,
                             headers={'User-Agent': USER_AGENT})
            if r.status_code != 200:
                print(f"  DeFiLlama emissions/{slug} HTTP {r.status_code}", file=sys.stderr)
                continue
            data = r.json()
        except Exception as e:
            print(f"  DeFiLlama emissions/{slug} failed: {e}", file=sys.stderr)
            continue

        meta      = data.get('metadata', {}) if isinstance(data, dict) else {}
        events    = meta.get('events')
        max_supply = (data.get('supplyMetrics', {}) or {}).get('maxSupply')
        if not isinstance(events, list):
            print(f"  DeFiLlama emissions/{slug}: no metadata.events (skipped)", file=sys.stderr)
            continue

        for ev in events:
            try:
                ts_raw = ev.get('timestamp')
                if ts_raw is None:
                    continue
                unlock_at = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                if unlock_at < now:
                    continue   # only future unlocks
                toks = ev.get('noOfTokens')
                if isinstance(toks, list):
                    toks = sum(t for t in toks if isinstance(t, (int, float)))
                toks = float(toks) if isinstance(toks, (int, float)) else None
                pct  = (round(toks / max_supply * 100, 4)
                        if toks is not None and max_supply else None)
                out.append({
                    'asset_id':      asset,
                    'unlock_at':     unlock_at,
                    'amount_tokens': toks,
                    'amount_usd':    None,   # not in the free emissions feed
                    'pct_supply':    pct,
                    'unlock_type':   (ev.get('unlockType') or ev.get('category') or 'unlock')[:64],
                })
            except (ValueError, TypeError):
                continue
    return out

def store_unlocks(con, rows: list[dict]) -> int:
    now = datetime.now(timezone.utc)
    n = 0
    for r in rows:
        ua = r['unlock_at']
        unlock_id = f"{r['asset_id']}_{ua.strftime('%Y%m%d%H%M')}"
        con.execute("""
            INSERT INTO token_unlocks
                (unlock_id, asset_id, unlock_at, amount_tokens, amount_usd,
                 pct_supply, unlock_type, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (unlock_id) DO UPDATE SET
                unlock_at     = excluded.unlock_at,
                amount_tokens = excluded.amount_tokens,
                amount_usd    = excluded.amount_usd,
                pct_supply    = excluded.pct_supply,
                unlock_type   = excluded.unlock_type,
                fetched_at    = excluded.fetched_at
        """, [unlock_id, r['asset_id'], ua, r.get('amount_tokens'),
              r.get('amount_usd'), r.get('pct_supply'),
              r.get('unlock_type'), now])
        n += 1
    return n

def check_news_veto(con, asset_id: str) -> bool:
    """True if buy_asset has a veto keyword in asset_news within the last 6h."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
    rows   = con.execute("""
        SELECT headline FROM asset_news
        WHERE asset_id = ? AND fetched_at >= ?
    """, [asset_id, cutoff]).fetchall()
    return any(count_keywords(headline, NEWS_VETO_KEYWORDS) for (headline,) in rows)

def _get_asset_news_text(con, asset: str, hours: int = 6) -> str:
    """Return recent headlines for an asset as a plain text block."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows   = con.execute("""
        SELECT headline FROM asset_news
        WHERE asset_id = ? AND fetched_at >= ?
        ORDER BY fetched_at DESC LIMIT 5
    """, [asset, cutoff]).fetchall()
    if not rows:
        return "  (no recent headlines)"
    return "\n".join(f"  - {r[0]}" for r in rows)
