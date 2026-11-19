"""Pure helpers: pair ids, LLM-output parsing, keyword matching, P&L, scoring math."""
import json
import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import (
    BINANCE_DIRECT_PAIRS, BINANCE_TAKER_FEE, DEFAULT_COHORT_WEIGHTS, FNG_FEAR,
    FNG_GREED, MAD_CONSISTENCY, MEME_ASSETS, TREND_SLOPE_THRESHOLD,
)


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def pair_id(a: str, b: str) -> str:
    ca, cb = canonical_pair(a, b)
    return f"{ca}/{cb}"

VALID_ACTIONS = ('accumulate', 'watch', 'pass')

def normalize_action(raw) -> str:
    """Map an LLM action to one of VALID_ACTIONS. Anything unrecognised is 'pass',
    so only an explicit 'accumulate' can ever open a trade."""
    a = str(raw or '').strip().lower()
    return a if a in VALID_ACTIONS else 'pass'

def _extract_json(content: str, opener: str):
    """Find the first JSON value starting with `opener` ('{' or '[') anywhere in
    LLM text: single-line, pretty-printed or inside ``` fences.
    Returns (value, raw_text, remaining_text) or (None, '', content)."""
    text = (content or '').strip()
    decoder = json.JSONDecoder()
    want = dict if opener == '{' else list
    idx = text.find(opener)
    while idx != -1:
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find(opener, idx + 1)
            continue
        if isinstance(value, want):
            rest = (text[:idx] + text[end:])
            rest = re.sub(r'```[a-zA-Z]*', '', rest).strip()
            return value, text[idx:end], rest
        idx = text.find(opener, end)
    return None, '', text

def parse_llm_verdict(content: str) -> dict:
    """Parse a verdict reply into {action, top_pair, top_asset, rank, thesis,
    risks, prose, raw_json}. Never raises; a reply without a usable JSON object
    becomes action='pass' with the text kept as prose."""
    result = {'action': 'pass', 'top_pair': '', 'top_asset': '',
              'rank': [], 'thesis': '', 'risks': '', 'prose': '', 'raw_json': ''}
    obj, raw, rest = _extract_json(content, '{')
    result['prose'] = rest
    if obj is None:
        return result
    result['raw_json'] = raw
    result['action'] = normalize_action(obj.get('action'))
    for k in ('top_pair', 'top_asset', 'thesis', 'risks'):
        v = obj.get(k)
        if v is not None:
            result[k] = str(v).strip()
    rank = obj.get('rank')
    if isinstance(rank, list):
        result['rank'] = [str(r) for r in rank]
    return result

_KW_CACHE: dict[str, re.Pattern] = {}

def _kw_pattern(keyword: str, case_sensitive: bool = False) -> re.Pattern:
    key = f"{int(case_sensitive)}:{keyword}"
    pat = _KW_CACHE.get(key)
    if pat is None:
        flags = 0 if case_sensitive else re.IGNORECASE
        pat = re.compile(r'(?<![A-Za-z0-9])' + re.escape(keyword) + r'(?![A-Za-z0-9])', flags)
        _KW_CACHE[key] = pat
    return pat

def count_keywords(text: str, keywords) -> int:
    """Number of keywords that appear in text as whole words (case-insensitive).
    'sec' matches 'SEC sues…' but not 'second'; 'ban' does not match 'bank'."""
    return sum(1 for kw in keywords if _kw_pattern(kw).search(text or ''))

def mentions_asset(text: str, ticker: str, name: str | None = None) -> bool:
    """True if text mentions the asset: ticker as an exact-case whole word
    (so 'DOT' but not 'dot com'), or full name case-insensitively."""
    if _kw_pattern(ticker, case_sensitive=True).search(text or ''):
        return True
    return bool(name and name.lower() != ticker.lower()
                and _kw_pattern(name).search(text or ''))

def asset_price(prices: dict[str, float], asset: str, default: float = 0.0) -> float:
    return 1.0 if asset == 'USDT' else float(prices.get(asset, default) or default)

def trade_pnl(sell_qty: float, sell_px_now: float,
              buy_qty: float, buy_px_now: float,
              notional: float) -> tuple[float, float]:
    """P&L of a swap vs. simply having kept what was sold.
    Pair trade: value of the bought leg minus today's value of the sold leg, so it
    measures the ratio move, not overall market drift. Fiat trade: the sold leg is
    USDT at 1.0, so this is ordinary USD P&L. Entry fees are already netted out of
    buy_qty, so a fresh trade starts slightly negative."""
    pnl_usd = float(buy_qty) * float(buy_px_now) - float(sell_qty) * float(sell_px_now)
    pnl_pct = pnl_usd / float(notional) * 100 if notional else 0.0
    return round(pnl_usd, 4), round(pnl_pct, 4)

INSIGHT_SIGNALS       = ('boost', 'warn', 'suppress')

INSIGHT_MAX_MAGNITUDE = 0.5

_INSIGHT_TARGET_RE    = re.compile(r'^[A-Z0-9]{2,12}(/[A-Z0-9]{2,12})?$')

def sanitize_insight(ins) -> dict | None:
    """Validate one LLM-proposed feedback insight. Returns a clean dict or None.
    - signal must be boost/warn/suppress
    - target must be one asset ('SOL') or one directional pair ('SOL/ETH');
      'ALL' is rejected so the LLM can never switch off all trading
    - magnitude is clamped to [0, INSIGHT_MAX_MAGNITUDE]; its sign comes from the
      signal (boost +, warn −), never from the model."""
    if not isinstance(ins, dict):
        return None
    signal = str(ins.get('signal', '')).strip().lower()
    target = str(ins.get('pair', '')).strip().upper()
    if (signal not in INSIGHT_SIGNALS or target == 'ALL'
            or not _INSIGHT_TARGET_RE.match(target)):
        return None
    try:
        mag = abs(float(ins.get('magnitude', 0.0)))
    except (TypeError, ValueError):
        mag = 0.0
    if math.isnan(mag):
        mag = 0.0
    mag = min(mag, INSIGHT_MAX_MAGNITUDE)
    if signal == 'warn':
        mag = -mag
    elif signal == 'suppress':
        mag = 0.0
    return {'pair': target, 'signal': signal, 'magnitude': round(mag, 4),
            'reason': str(ins.get('reason', ''))[:200]}

def reputation_score(avg_out: float, win_rate: float, n: int) -> float:
    """Reputation in (-1, 1): sign from the average outcome, scaled by how
    consistently outcomes agree with that sign and by sample size.
    (The old tanh(avg_out * win_rate * ln(n+1)) gave a pair that lost every
    time a score of 0, because win_rate was 0, so losers looked neutral.)"""
    consistency = win_rate if avg_out >= 0 else (1.0 - win_rate)
    return round(math.tanh(avg_out * consistency * math.log(n + 1)), 4)

def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def fng_modifier(fng_value: int) -> tuple[float, str]:
    """
    Returns (threshold_adjustment, narrative).
    Extreme fear  → lower threshold (easier to trigger), boost confidence
    Extreme greed → raise threshold (harder to trigger), suppress confidence
    """
    if fng_value <= FNG_FEAR:
        adj = -0.3   # easier entry
        narrative = f"Extreme Fear ({fng_value}) — historically good accumulation conditions"
    elif fng_value >= FNG_GREED:
        adj = +0.5   # harder entry
        narrative = f"Extreme Greed ({fng_value}) — elevated risk, tighter entry threshold"
    elif fng_value <= 40:
        adj = -0.1
        narrative = f"Fear ({fng_value}) — market cautious, reversion more likely"
    elif fng_value >= 60:
        adj = +0.2
        narrative = f"Greed ({fng_value}) — market extended, be selective"
    else:
        adj = 0.0
        narrative = f"Neutral ({fng_value}) — no macro bias"
    return adj, narrative

def conviction_label(confidence: int) -> str:
    if confidence >= 8: return "Very Strong"
    if confidence >= 6: return "Strong"
    if confidence >= 4: return "Moderate"
    return "Low"

def trade_route(base: str, quote: str) -> tuple[str, int]:
    if f"{base}/{quote}" in BINANCE_DIRECT_PAIRS:
        return f"{base}/{quote} (direct)", 1
    return f"{base}/USDT → {quote}/USDT", 2

def fee_breakdown(notional: float, num_legs: int, spread_pct: float | None) -> dict:
    fee_per_leg   = notional * BINANCE_TAKER_FEE
    total_fee     = fee_per_leg * num_legs
    spread_cost   = notional * (spread_pct / 100) if spread_pct else 0.0
    total_cost    = total_fee + spread_cost
    breakeven_pct = (total_cost / notional) * 100
    return {
        'notional':      round(notional,      4),
        'fee_per_leg':   round(fee_per_leg,   6),
        'num_legs':      num_legs,
        'total_fee':     round(total_fee,     6),
        'spread_cost':   round(spread_cost,   6),
        'total_cost':    round(total_cost,    6),
        'breakeven_pct': round(breakeven_pct, 6),
    }

def mad_zscore(values: np.ndarray, current: float) -> float:
    if len(values) < 5:
        return 0.0
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    denom  = MAD_CONSISTENCY * mad
    return float((current - median) / denom) if denom > 0 else 0.0

def reputation_label(rep: float | None) -> str:
    if rep is None or rep == 0.0: return "neutral"
    if rep >= 0.5:  return "strong✅"
    if rep >= 0.2:  return "positive"
    if rep >= 0.0:  return "mild+"
    if rep >= -0.2: return "mild-"
    if rep >= -0.5: return "negative"
    return "poor⚠"

def detect_dominance(df: pd.DataFrame, col: str) -> str | None:
    counts   = df[col].value_counts()
    dominant = counts[counts >= 3]
    return dominant.index[0] if not dominant.empty else None

def rule_based_score(row, persistence, prev_z, reputations, slopes, fng_adj=0.0) -> float:
    score = row['z_score'] + fng_adj   # F&G adjusts entry threshold

    score -= min(persistence.get(row['pair'], 0) * 0.1, 0.5)

    prev = prev_z.get(row['pair'])
    if prev is not None:
        score -= 0.2 if row['z_score'] < prev else -0.2

    if row['volume_z']   is not None and row['volume_z']   >  1.5: score -= 0.15
    if row['vwap_ratio'] is not None and row['vwap_ratio'] < 0.99: score -= 0.1
    if row['spread_pct'] is not None and row['spread_pct'] >  0.1: score += 0.3
    if row['quote_asset'] in MEME_ASSETS:                           score += 0.5

    rep = reputations.get(row['pair'], 0.0)
    score -= rep * 0.5

    # Falling-knife guard: a ratio in sustained directional drift is trending,
    # not reverting, so penalise it proportionally — a multi-day grind shouldn't
    # keep winning the board. (Old `slope*10` was a no-op: normalised slopes ~1e-4.)
    slope = abs(slopes.get(row['pair'], 0.0))
    if slope > TREND_SLOPE_THRESHOLD:
        score += min(slope / TREND_SLOPE_THRESHOLD, 4.0) * 0.5

    return round(score, 4)

def _trade_id(ts: datetime, pair: str) -> str:
    return f"{ts.strftime('%Y%m%d%H%M')}_{pair.replace('/', '_')}"

def _etf_flow_label(flow_m: float | None) -> str:
    if flow_m is None:
        return "N/A"
    if flow_m < -500:
        return f"-${abs(flow_m):.0f}M (Very Bearish)"
    if flow_m < -100:
        return f"-${abs(flow_m):.0f}M (Bearish)"
    if flow_m < 100:
        return f"${flow_m:+.0f}M (Neutral)"
    if flow_m < 500:
        return f"+${flow_m:.0f}M (Bullish)"
    return f"+${flow_m:.0f}M (Very Bullish)"

def _trade_signal_label(confidence: int) -> str:
    if confidence <= 3:
        return f"{confidence}/10 (Caution)"
    if confidence <= 6:
        return f"{confidence}/10 (Neutral)"
    return f"{confidence}/10 (Confident)"

def fg_classification(value: int) -> str:
    if value <= 20:  return "Extreme Fear"
    if value <= 40:  return "Fear"
    if value <= 60:  return "Neutral"
    if value <= 80:  return "Greed"
    return "Extreme Greed"

def etf_flow_to_fg(flow_m_usd: float) -> int:
    """Convert BTC ETF daily net flow ($M) to 0-100 F&G scale."""
    if flow_m_usd >= 500:    return 85
    elif flow_m_usd >= 200:  return 70
    elif flow_m_usd >= 50:   return 55
    elif flow_m_usd >= -50:  return 50
    elif flow_m_usd >= -200: return 35
    elif flow_m_usd >= -500: return 20
    else:                    return 10

def onchain_to_fg(exchange_inflow_norm: float, whale_activity: float,
                  miner_capitulation: bool) -> int:
    """
    exchange_inflow_norm: 0-1 (high = bearish)
    whale_activity: 0-1 (high = bullish)
    miner_capitulation: bool (True = bearish)
    """
    score = 50.0
    score -= exchange_inflow_norm * 30
    score += whale_activity * 20
    if miner_capitulation:
        score -= 15
    return int(max(0, min(100, round(score))))

def trader_score_to_fg(avg_confidence: float) -> int:
    """Map Qwen avg_confidence (1-10) to 0-100 F&G scale."""
    return int((avg_confidence / 10) * 100)

def calculate_synthetic_fg(cohort_values: dict, weights: dict = None) -> int:
    weights = weights or DEFAULT_COHORT_WEIGHTS
    return round(sum(cohort_values[k] * weights[k] for k in weights))

def buffett_exit_threshold(fg_official: int, etf_flow_m: float | None,
                           fg_synthetic: int | None = None) -> float:
    """Dynamic Z crossback threshold. Uses synthetic F&G when available."""
    fg = fg_synthetic if fg_synthetic is not None else fg_official
    if fg <= 20:   z_target = 1.5    # Extreme Fear: hold longer, reversion likely
    elif fg <= 40: z_target = 0.8    # Fear
    elif fg <= 60: z_target = 0.0    # Neutral
    elif fg <= 80: z_target = -0.2   # Greed: exit earlier
    else:          z_target = -0.5   # Extreme Greed: exit early, risk elevated
    if etf_flow_m is not None:
        if etf_flow_m < -500:   z_target += 0.3   # heavy outflow: hold (contrarian)
        elif etf_flow_m > 500:  z_target -= 0.2   # heavy inflow: exit sooner (too hot)
    return z_target

def stop_loss_threshold(fg_value: int) -> float:
    """F&G-aware max loss % before forced exit. Tighter in fear, looser in greed."""
    if fg_value <= 20:   return -8.0    # Extreme Fear — market dangerous, cut fast
    elif fg_value <= 40: return -10.0   # Fear
    elif fg_value <= 60: return -12.0   # Neutral
    elif fg_value <= 80: return -15.0   # Greed — reversion more likely, give room
    else:                return -18.0   # Extreme Greed

def _parse_position_review(content: str) -> tuple[str, int, str]:
    """Parse Qwen response: returns (action, urgency, reason)."""
    action    = 'HOLD'
    urgency   = 0
    reason    = content.strip()
    lines     = content.strip().splitlines()
    for line in lines:
        stripped = line.strip().upper()
        if stripped.startswith('EXIT'):
            action = 'EXIT'
            break
        elif stripped.startswith('HOLD'):
            action = 'HOLD'
            break
    for line in lines:
        m = re.search(r'urgency[:\s]+(\d+)', line, re.IGNORECASE)
        if m:
            urgency = min(10, max(1, int(m.group(1))))
            break
    # Use first non-action line as reason
    for line in lines[1:]:
        s = line.strip()
        if s and not re.match(r'urgency', s, re.IGNORECASE):
            reason = s
            break
    return action, urgency, reason
