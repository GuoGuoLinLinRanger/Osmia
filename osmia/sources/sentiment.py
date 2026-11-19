"""Macro context and the 4-cohort Fear & Greed system."""
import json
import sys
from datetime import datetime, timedelta, timezone

import requests

from ..config import (
    CMC_API_KEY, CMC_FNG_URL, COINGECKO_GLOBAL_URL, CRYPTOQUANT_INFLOW_URL,
    DEFAULT_COHORT_WEIGHTS, DIVERGENCE_THRESHOLD, ETF_CACHE_SECONDS,
    FARSIDE_ETF_URL, FG_COHORT_CACHE_SECS, FG_TRADER_LOOKBACK_MINS, FNG_URL,
    MEMPOOL_FEES_URL, MEMPOOL_HASHRATE_URL, USER_AGENT,
)
from ..core import (
    calculate_synthetic_fg, etf_flow_to_fg, fg_classification, onchain_to_fg,
    trader_score_to_fg,
)
from ..store import _get_macro_cache, _set_macro_cache, get_conn


def fetch_macro_context() -> dict:
    """
    CoinGecko /global → BTC dominance + total market-cap 24h change.
    Pure network, no DB. Never raises — returns Nones on failure.
    """
    macro = {'btc_dominance': None, 'mcap_change_24h': None}
    try:
        r = requests.get(COINGECKO_GLOBAL_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        d = r.json()['data']
        macro['btc_dominance']   = d.get('market_cap_percentage', {}).get('btc')
        macro['mcap_change_24h'] = d.get('market_cap_change_percentage_24h_usd')
    except Exception as e:
        print(f"  CoinGecko /global fetch failed: {e}", file=sys.stderr)
    return macro

def get_btc_change_24h(con) -> float | None:
    row = con.execute("""
        SELECT change_24h FROM market_history
        WHERE asset_id = 'BTC' ORDER BY timestamp DESC LIMIT 1
    """).fetchone()
    return float(row[0]) if row and row[0] is not None else None

def build_macro_block(ctx: dict, macro: dict, btc_change_24h: float | None,
                      heightened_lines: list[str] | None = None,
                      fg_data: dict | None = None) -> str:
    """Render the [MACRO CONTEXT] block prepended to the Qwen prompt."""
    btc = f"{btc_change_24h:+.1f}% 24h"           if btc_change_24h is not None         else "n/a"
    dom = f"{macro.get('btc_dominance'):.1f}%"    if macro.get('btc_dominance')   is not None else "n/a"
    mc  = f"{macro.get('mcap_change_24h'):+.1f}% 24h" if macro.get('mcap_change_24h') is not None else "n/a"
    if fg_data:
        fg_off   = fg_data.get('fg_official', ctx['fng_value'])
        fg_lbl   = fg_data.get('fg_official_label', ctx['fng_label'])
        fg_synth = fg_data.get('fg_synthetic', fg_off)
        fg_line  = (f"Fear & Greed (Official): {fg_off} ({fg_lbl}) | "
                    f"Synthetic: {fg_synth} ({fg_classification(fg_synth)})")
    else:
        fg_line = f"Fear & Greed: {ctx['fng_value']} ({ctx['fng_label']})"
    lines = [
        "[MACRO CONTEXT]",
        f"BTC: {btc} | Dominance: {dom}",
        f"Market cap: {mc}",
        fg_line,
    ]
    if heightened_lines:
        lines.append("")
        lines.extend(heightened_lines)
    return "\n".join(lines)

def fetch_etf_flows() -> float | None:
    """Scrape latest BTC ETF total net flow (USD millions) from Farside Investors.

    Returns the Total column of the most recent complete data row.
    Farside uses UK accounting notation: parentheses = negative, e.g. (325.7) = -325.7.
    """
    import re as _re
    try:
        r = requests.get(FARSIDE_ETF_URL, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
        })
        html = r.text
        rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', html, _re.DOTALL)
        data_rows = []
        for row in rows:
            cells = _re.findall(r'<td[^>]*>(.*?)</td>', row, _re.DOTALL)
            clean = [_re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if clean and _re.match(r'\d{2} \w+ \d{4}', clean[0]):
                data_rows.append(clean)
        if not data_rows:
            return None

        def _parse(val: str) -> float | None:
            val = val.replace(',', '').strip()
            if not val or val == '-':
                return None
            m = _re.match(r'^\(([0-9.]+)\)$', val)  # parentheses = negative
            if m:
                return -float(m.group(1))
            try:
                return float(val)
            except ValueError:
                return None

        for row in reversed(data_rows):
            total = _parse(row[-1]) if row else None
            if total is not None:
                return total
    except Exception as e:
        print(f"  Farside ETF fetch failed: {e}", file=sys.stderr)
    return None

def get_etf_flow_cached(con) -> float | None:
    """Return cached ETF flow or fetch fresh if older than ETF_CACHE_SECONDS."""
    cached = _get_macro_cache(con, 'etf_flow_m', ETF_CACHE_SECONDS)
    if cached is not None:
        try:
            return float(cached)
        except (ValueError, TypeError):
            pass
    flow = fetch_etf_flows()
    if flow is not None:
        _set_macro_cache(con, 'etf_flow_m', str(flow))
    return flow

def get_cohort_weights(con) -> dict:
    """Read weights from macro_cache or return defaults."""
    cached = _get_macro_cache(con, 'cohort_weights', 86400 * 365)  # weights don't expire
    if cached:
        try:
            w = json.loads(cached)
            if set(w.keys()) == set(DEFAULT_COHORT_WEIGHTS.keys()):
                return w
        except (json.JSONDecodeError, TypeError):
            pass
    return dict(DEFAULT_COHORT_WEIGHTS)

def _fetch_fg_altme() -> int | None:
    """alternative.me — daily update baseline. Returns None on failure."""
    try:
        r = requests.get(FNG_URL, timeout=10)
        return int(r.json()['data'][0]['value'])
    except Exception as e:
        print(f"  [fg-retail/altme] {e}", file=sys.stderr)
        return None

def _fetch_fg_coinmarketcap() -> int | None:
    """CoinMarketCap — updates every 15 minutes. Returns None if no key or on failure."""
    if not CMC_API_KEY:
        return None
    try:
        r = requests.get(CMC_FNG_URL,
                         headers={'X-CMC_PRO_API_KEY': CMC_API_KEY},
                         timeout=10)
        return int(r.json()['data']['value'])
    except Exception as e:
        print(f"  [fg-retail/cmc] {e}", file=sys.stderr)
        return None

def _fetch_fg_coingecko_proxy() -> int | None:
    """
    CoinGecko /global proxy: market cap change + BTC dominance → F&G estimate.
    Rising total mcap + falling BTC dominance (alt season) → greed.
    Returns None on failure.
    """
    try:
        r = requests.get(COINGECKO_GLOBAL_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        d = r.json()['data']
        btc_dominance     = d['market_cap_percentage']['btc']
        market_cap_change = d['market_cap_change_percentage_24h_usd']
        score = 50 + market_cap_change * 2 + (50 - btc_dominance)
        return max(0, min(100, int(score)))
    except Exception as e:
        print(f"  [fg-retail/coingecko] {e}", file=sys.stderr)
        return None

def fetch_retail_fg() -> tuple[int, str, dict]:
    """
    Average alternative.me + CMC + CoinGecko proxy into a single retail F&G value.
    Returns (averaged_value, label, sources_dict).
    Falls back gracefully if any source is unavailable.
    """
    sources = {
        'alt_me':    _fetch_fg_altme(),
        'cmc':       _fetch_fg_coinmarketcap(),
        'coingecko': _fetch_fg_coingecko_proxy(),
    }
    available = {k: v for k, v in sources.items() if v is not None}
    if not available:
        return 50, 'Neutral', sources
    averaged = round(sum(available.values()) / len(available))
    print(f"  [fg-retail] alt_me={sources['alt_me']} cmc={sources['cmc']} "
          f"cg={sources['coingecko']} → avg={averaged} ({len(available)}/3 sources)")
    return averaged, fg_classification(averaged), sources

def fetch_fg_institutional(con) -> tuple[int, float | None]:
    """
    ETF flows → institutional F&G.
    Returns (fg_value, flow_m_usd). Uses cached flow; neutral on failure.
    """
    flow = get_etf_flow_cached(con)
    if flow is None:
        cached_str = _get_macro_cache(con, 'fg_institutional', 86400)
        if cached_str:
            try:
                return int(cached_str), None
            except (ValueError, TypeError):
                pass
        return 50, None
    return etf_flow_to_fg(flow), flow

def fetch_fg_onchain() -> int:
    """
    Fetch on-chain signals (CryptoQuant + mempool.space) and compute 0-100 F&G.
    All sources degrade independently. Returns 50 (neutral) if all fail.
    """
    exchange_inflow_norm = 0.0   # neutral baseline
    whale_activity       = 0.5   # neutral baseline
    miner_capitulation   = False

    # 1. CryptoQuant exchange inflows (free tier may need API key; degrades gracefully)
    try:
        r = requests.get(CRYPTOQUANT_INFLOW_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        if r.status_code == 200:
            data = r.json()
            inflow = None
            if isinstance(data.get('data'), list) and data['data']:
                item = data['data'][0]
                for field in ('inflow_total', 'value', 'inflow'):
                    inflow = item.get(field)
                    if inflow is not None:
                        break
            if inflow is not None:
                # Normalize: typical 5k-50k BTC/day; > 30k = high selling pressure
                exchange_inflow_norm = min(1.0, max(0.0, (float(inflow) - 5000) / 25000))
    except Exception as e:
        print(f"  [fg-onchain] CryptoQuant inflow failed: {e}", file=sys.stderr)

    # 2. Mempool.space block fees as whale/congestion proxy
    try:
        r = requests.get(MEMPOOL_FEES_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                row = data[0]
                avg_fee = float(row.get('avgFee_50') or row.get('medianFee') or 10)
                whale_activity = min(1.0, max(0.0, avg_fee / 100))
    except Exception as e:
        print(f"  [fg-onchain] Mempool fees failed: {e}", file=sys.stderr)

    # 3. Hashrate hash-ribbon (30d < 60d MA → miner capitulation)
    try:
        r = requests.get(MEMPOOL_HASHRATE_URL, timeout=12,
                         headers={'User-Agent': USER_AGENT})
        if r.status_code == 200:
            hashrates = r.json().get('hashrates', [])
            if len(hashrates) >= 60:
                recent = [h['avgHashrate'] for h in hashrates[-30:]]
                older  = [h['avgHashrate'] for h in hashrates[-60:-30]]
                hr_30d = sum(recent) / len(recent) if recent else 0
                hr_60d = sum(older)  / len(older)  if older  else 0
                miner_capitulation = bool(hr_30d > 0 and hr_30d < hr_60d * 0.95)
    except Exception as e:
        print(f"  [fg-onchain] Hashrate fetch failed: {e}", file=sys.stderr)

    score = onchain_to_fg(exchange_inflow_norm, whale_activity, miner_capitulation)
    print(f"  [fg-onchain] inflow_norm={exchange_inflow_norm:.2f} whale={whale_activity:.2f} "
          f"miner_cap={miner_capitulation} → FG={score}")
    return score

def get_fg_trader(con) -> int:
    """
    Cohort 4: average Qwen confidence for accumulate verdicts in the last
    FG_TRADER_LOOKBACK_MINS minutes, mapped to 0-100 F&G scale. Returns 50 if
    no recent verdicts exist.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=FG_TRADER_LOOKBACK_MINS)
    row = con.execute("""
        SELECT AVG(confidence_score) FROM llm_verdicts
        WHERE action = 'accumulate' AND timestamp >= ?
    """, [cutoff]).fetchone()
    avg = row[0] if row and row[0] is not None else None
    if avg is None:
        return 50
    return trader_score_to_fg(float(avg))

def fetch_all_fg_cohorts(con) -> dict:
    """
    Fetch all 4 cohorts, compute synthetic, return dict with all values.
    Never raises — individual sources fall back to neutral.
    """
    weights = get_cohort_weights(con)

    fg_retail_val, fg_retail_lbl, retail_sources = fetch_retail_fg()
    fg_inst_val,   etf_flow_m                   = fetch_fg_institutional(con)
    fg_onchain_val                              = fetch_fg_onchain()
    fg_trader_val                               = get_fg_trader(con)

    cohort_values = {
        'retail':        fg_retail_val,
        'institutional': fg_inst_val,
        'onchain':       fg_onchain_val,
        'trader':        fg_trader_val,
    }
    fg_synthetic = calculate_synthetic_fg(cohort_values, weights)

    return {
        'fg_official':           fg_retail_val,
        'fg_official_label':     fg_retail_lbl,
        'fg_retail':             fg_retail_val,
        'fg_retail_alt_me':      retail_sources.get('alt_me'),
        'fg_retail_cmc':         retail_sources.get('cmc'),
        'fg_retail_coingecko':   retail_sources.get('coingecko'),
        'fg_retail_sources_count': sum(1 for v in retail_sources.values() if v is not None),
        'fg_institutional':      fg_inst_val,
        'fg_onchain':            fg_onchain_val,
        'fg_trader':             fg_trader_val,
        'fg_synthetic':          fg_synthetic,
        'etf_flow_usd_m':        etf_flow_m,
        'weights':               weights,
    }

def _store_fg_cohorts(con, fg: dict) -> None:
    """Write all cohort values into macro_cache (upsert)."""
    pairs = [
        ('fg_retail',        str(fg['fg_retail'])),
        ('fg_institutional', str(fg['fg_institutional'])),
        ('fg_onchain',       str(fg['fg_onchain'])),
        ('fg_trader',        str(fg['fg_trader'])),
        ('fg_synthetic',     str(fg['fg_synthetic'])),
        ('fg_official',      str(fg['fg_official'])),
        ('fg_official_label', fg['fg_official_label']),
    ]
    if fg.get('etf_flow_usd_m') is not None:
        pairs.append(('etf_flow_usd_m', str(fg['etf_flow_usd_m'])))
    for src_key in ('fg_retail_alt_me', 'fg_retail_cmc', 'fg_retail_coingecko'):
        if fg.get(src_key) is not None:
            pairs.append((src_key, str(fg[src_key])))
    for key, val in pairs:
        _set_macro_cache(con, key, val)

def _insert_fg_history(con, fg: dict) -> None:
    """Insert a timestamped F&G snapshot into fg_history for Grafana."""
    w   = fg.get('weights', DEFAULT_COHORT_WEIGHTS)
    now = datetime.now(timezone.utc)
    # id = epoch seconds; inserts at most once per 5 minutes so no collision risk
    row_id = int(now.timestamp())
    con.execute("""
        INSERT OR IGNORE INTO fg_history
            (id, recorded_at, fg_official, fg_retail, fg_institutional, fg_onchain,
             fg_trader, fg_synthetic, etf_flow_usd_m,
             weight_retail, weight_institutional, weight_onchain, weight_trader,
             fg_retail_alt_me, fg_retail_cmc, fg_retail_coingecko)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        row_id, now,
        fg['fg_official'], fg['fg_retail'], fg['fg_institutional'],
        fg['fg_onchain'],  fg['fg_trader'],  fg['fg_synthetic'],
        fg.get('etf_flow_usd_m'),
        w.get('retail', 0.25), w.get('institutional', 0.30),
        w.get('onchain', 0.25), w.get('trader', 0.20),
        fg.get('fg_retail_alt_me'), fg.get('fg_retail_cmc'), fg.get('fg_retail_coingecko'),
    ])

def run_fetch_fg(con) -> None:
    """
    --fetch-fg: fetch all 4 cohorts, compute synthetic, store to macro_cache
    and fg_history. Silent — no Discord post. Designed for */5 cron.
    """
    print(f"[{datetime.now():%H:%M:%S}] Fetching 4-cohort F&G …")
    # Release lock during network I/O
    con.close()
    fg = fetch_all_fg_cohorts_network_only()
    con = get_conn()
    _store_fg_cohorts(con, fg)
    _insert_fg_history(con, fg)
    w = fg['weights']
    print(
        f"  Official: {fg['fg_official']} ({fg['fg_official_label']}) | "
        f"Synthetic: {fg['fg_synthetic']} ({fg_classification(fg['fg_synthetic'])})"
    )
    print(
        f"  Retail: {fg['fg_retail']} | Inst: {fg['fg_institutional']} | "
        f"On-chain: {fg['fg_onchain']} | Trader: {fg['fg_trader']}"
    )
    divergence = abs(fg['fg_synthetic'] - fg['fg_official'])
    if divergence >= DIVERGENCE_THRESHOLD:
        print(f"  ⚡ Divergence alert: {divergence} pts (synthetic vs official)")
    con.close()
    return con

def fetch_all_fg_cohorts_network_only() -> dict:
    """Network-only wrapper for fetch_all_fg_cohorts (no DB needed for weights here)."""
    # We can't pass con here since lock is released — use defaults for weights
    # (they were already stored in macro_cache by --set-weights; will be reloaded
    # from cache on next DB-connected call).
    weights = dict(DEFAULT_COHORT_WEIGHTS)
    fg_retail_val, fg_retail_lbl, retail_sources = fetch_retail_fg()
    # For institutional, we can't read the cached ETF flow without a DB connection.
    # Call fetch_etf_flows() directly.
    etf_flow_m    = fetch_etf_flows()
    fg_inst_val   = etf_flow_to_fg(etf_flow_m) if etf_flow_m is not None else 50
    fg_onchain_val = fetch_fg_onchain()
    # Trader score needs DB — use neutral (50); will be populated when analyze runs.
    fg_trader_val = 50

    cohort_values = {
        'retail':        fg_retail_val,
        'institutional': fg_inst_val,
        'onchain':       fg_onchain_val,
        'trader':        fg_trader_val,
    }
    fg_synthetic = calculate_synthetic_fg(cohort_values, weights)
    return {
        'fg_official':           fg_retail_val,
        'fg_official_label':     fg_retail_lbl,
        'fg_retail':             fg_retail_val,
        'fg_retail_alt_me':      retail_sources.get('alt_me'),
        'fg_retail_cmc':         retail_sources.get('cmc'),
        'fg_retail_coingecko':   retail_sources.get('coingecko'),
        'fg_retail_sources_count': sum(1 for v in retail_sources.values() if v is not None),
        'fg_institutional':      fg_inst_val,
        'fg_onchain':            fg_onchain_val,
        'fg_trader':             fg_trader_val,
        'fg_synthetic':          fg_synthetic,
        'etf_flow_usd_m':        etf_flow_m,
        'weights':               weights,
    }

def get_cached_fg_data(con) -> dict:
    """
    Read all F&G cohort values from macro_cache. Returns neutral defaults (50)
    for any stale or missing entry. Never raises.
    """
    def _read_int(key, default=50):
        v = _get_macro_cache(con, key, FG_COHORT_CACHE_SECS)
        if v is None:
            # allow slightly stale data rather than full neutral — up to 1h
            v = _get_macro_cache(con, key, 3600)
        try:
            return int(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    def _read_int_opt(key) -> int | None:
        v = _get_macro_cache(con, key, FG_COHORT_CACHE_SECS)
        if v is None:
            v = _get_macro_cache(con, key, 3600)
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    fg_official = _read_int('fg_official', 50)
    lbl_raw     = _get_macro_cache(con, 'fg_official_label', 3600)
    fg_official_label = lbl_raw if lbl_raw else fg_classification(fg_official)

    return {
        'fg_official':          fg_official,
        'fg_official_label':    fg_official_label,
        'fg_retail':            _read_int('fg_retail', fg_official),
        'fg_retail_alt_me':     _read_int_opt('fg_retail_alt_me'),
        'fg_retail_cmc':        _read_int_opt('fg_retail_cmc'),
        'fg_retail_coingecko':  _read_int_opt('fg_retail_coingecko'),
        'fg_institutional':     _read_int('fg_institutional', 50),
        'fg_onchain':           _read_int('fg_onchain', 50),
        'fg_trader':            _read_int('fg_trader', 50),
        'fg_synthetic':         _read_int('fg_synthetic', fg_official),
    }

def set_cohort_weights(con, weights_str: str) -> None:
    """Parse and persist cohort weights from 'retail=0.3,institutional=0.4,...'."""
    parts = {}
    for item in weights_str.split(','):
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        k = k.strip().lower()
        if k in DEFAULT_COHORT_WEIGHTS:
            parts[k] = float(v.strip())
    missing  = set(DEFAULT_COHORT_WEIGHTS.keys()) - set(parts.keys())
    unknown  = set(parts.keys()) - set(DEFAULT_COHORT_WEIGHTS.keys())
    if missing or unknown:
        raise ValueError(f"All 4 cohorts required. Missing: {missing}. Unknown: {unknown}")
    total = sum(parts.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
    _set_macro_cache(con, 'cohort_weights', json.dumps(parts))
    print(f"  Cohort weights updated: {parts}")
    print("  Next --fetch-fg will use these weights.")

def _build_divergence_note(fg_data: dict) -> str:
    """Return divergence context for Qwen when synthetic deviates from official."""
    fg_off   = fg_data.get('fg_official', 50)
    fg_synth = fg_data.get('fg_synthetic', fg_off)
    divergence = abs(fg_synth - fg_off)
    if divergence < DIVERGENCE_THRESHOLD:
        return ""
    direction = "less fear" if fg_synth > fg_off else "more fear"
    return (
        f"NOTE: Synthetic F&G ({fg_synth}) significantly diverges from Official "
        f"({fg_off}) by {divergence} pts — institutional and on-chain cohorts suggest "
        f"{direction} than retail sentiment indicates."
    )
