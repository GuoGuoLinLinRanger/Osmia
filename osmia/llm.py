"""LLM client.

Two providers, picked with OSMIA_LLM_PROVIDER:
  openai     (default) any OpenAI-compatible chat-completions endpoint: a local
             Qwen via llama.cpp / vLLM / LM Studio / Ollama, or a hosted one
             (OpenRouter, Groq, OpenAI...). OSMIA_LLM_URL, OSMIA_LLM_MODEL,
             optional OSMIA_LLM_API_KEY sent as a Bearer token.
  anthropic  Claude through the official `anthropic` SDK (pip install
             'osmia[anthropic]'); key from ANTHROPIC_API_KEY or `ant auth login`.

Every caller goes through chat(); failures raise LLMError and each public helper
turns that into its safe fallback (verdict -> 'pass', insights -> [], ...).
"""
import hashlib
import importlib.util
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone

import requests

from . import config
from .core import _extract_json, parse_llm_verdict


class LLMError(Exception):
    pass


_anthropic_client = None


def _chat_openai(prompt: str, max_tokens: int, temperature: float, timeout: int) -> str:
    headers = {'Authorization': f'Bearer {config.LLM_API_KEY}'} if config.LLM_API_KEY else {}
    payload = {"model": config.LLM_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(config.LLM_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'] or ''
    except Exception as e:
        raise LLMError(str(e)) from e


def _chat_anthropic(prompt: str, timeout: int) -> str:
    global _anthropic_client
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("OSMIA_LLM_PROVIDER=anthropic needs: pip install 'osmia[anthropic]'") from e
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(timeout=timeout)
    try:
        # Thinking is always on for current Claude models and counts toward
        # max_tokens, so leave generous room even for a one-line JSON verdict.
        # Server-side fallbacks re-run a declined request on a fallback model.
        resp = _anthropic_client.beta.messages.create(
            model=config.LLM_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIConnectionError as e:
        raise LLMError(f"network error: {e}") from e
    except anthropic.RateLimitError as e:
        raise LLMError(f"rate limited: {e}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"API error {e.status_code}: {e.message}") from e
    if resp.stop_reason == "refusal":
        raise LLMError("model declined the request")
    return "".join(b.text for b in resp.content if b.type == "text")


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(
        f"{config.LLM_PROVIDER}\n{config.LLM_MODEL}\n{prompt}".encode('utf-8')).hexdigest()


def _cache_db() -> sqlite3.Connection:
    db = sqlite3.connect(config.LLM_CACHE_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS llm_cache (
        key TEXT PRIMARY KEY, provider TEXT, model TEXT, prompt TEXT,
        response TEXT, created_at TEXT)""")
    return db


def cache_get(prompt: str) -> str | None:
    if config.LLM_CACHE not in ('readwrite', 'readonly'):
        return None
    try:
        with closing(_cache_db()) as db:
            row = db.execute("SELECT response FROM llm_cache WHERE key = ?",
                             (_cache_key(prompt),)).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def cache_put(prompt: str, response: str) -> None:
    if config.LLM_CACHE != 'readwrite':
        return
    try:
        with closing(_cache_db()) as db, db:
            db.execute("INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?)",
                       (_cache_key(prompt), config.LLM_PROVIDER, config.LLM_MODEL, prompt,
                        response, datetime.now(timezone.utc).isoformat()))
    except sqlite3.Error as e:
        print(f"  [llm-cache] write failed: {e}", file=sys.stderr)


def chat(prompt: str, max_tokens: int = 512, temperature: float = 0.2,
         timeout: int = 90) -> str:
    """Send one user prompt, return the reply text. Raises LLMError.
    With OSMIA_LLM_CACHE set, answers are reused/recorded (see config)."""
    cached = cache_get(prompt)
    if cached is not None:
        return cached
    if config.LLM_CACHE == 'readonly':
        raise LLMError("no cached answer for this prompt (OSMIA_LLM_CACHE=readonly)")
    if config.LLM_PROVIDER == 'anthropic':
        reply = _chat_anthropic(prompt, timeout)
    else:
        reply = _chat_openai(prompt, max_tokens, temperature, timeout)
    cache_put(prompt, reply)
    return reply


def ping_llm() -> bool:
    """Cheap reachability check; never sends a billed completion."""
    if config.LLM_CACHE == 'readonly':
        return True          # answers come from the cache; misses degrade per call
    if config.LLM_PROVIDER == 'anthropic':
        return importlib.util.find_spec('anthropic') is not None
    base = config.LLM_URL.rsplit('/chat/completions', 1)[0]
    headers = {'Authorization': f'Bearer {config.LLM_API_KEY}'} if config.LLM_API_KEY else {}
    for url in (base + '/models', base.rsplit('/v1', 1)[0] + '/health'):
        try:
            if requests.get(url, headers=headers, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
    return False


def ask_verdict(prompt: str) -> dict:
    """Ask for a trade verdict; any failure becomes action='pass'."""
    try:
        content = chat(prompt, max_tokens=512, temperature=0.2, timeout=90)
    except LLMError as e:
        v = parse_llm_verdict('')
        v['prose'] = f"LLM request failed: {e}"
        return v
    return parse_llm_verdict(content)


def ask_insights(data: dict) -> list[dict]:
    """
    Second LLM call: ask for a JSON array of pair insights for the feedback loop.
    Returns list of {pair, signal, magnitude, reason}; validated by the caller.
    """
    prompt = f"""Based on this trading bot performance data, return ONLY a JSON array of pair-level
insights for the feedback loop. No explanation, no prose. Each object must have:
  pair     — a pair exactly as listed below (e.g. "SOL/ETH") or a single asset (e.g. "SOL")
  signal   — one of: "boost", "warn", "suppress"
  magnitude — strength between 0.0 and 0.5 (direction comes from the signal)
  reason   — one sentence max

Use the trade win rate (not signal hit rate) to drive suppression decisions.
Suppress pairs with < 40% win rate on > 3 trades. Boost pairs with > 70% win rate on > 3 trades.

Performance data:
Trade win rate: {data['trade_win_rate']}% ({data['trade_wins']}/{data['trade_total']} trades)
Avg trade outcome: {data['avg_trade_pct']:+.3f}%
Open P&L: {data['open_positions']}
Official F&G: {data['fg_official']} | Synthetic: {data['fg_synthetic']}

Top pairs by outcome: {', '.join(f"{r['top_pair']}({r['avg_out']:+.3f}%)" for _, r in data['top_pairs_df'].iterrows()) if not data['top_pairs_df'].empty else 'n/a'}
Worst pairs: {', '.join(f"{r['top_pair']}({r['avg_out']:+.3f}%)" for _, r in data['worst_pairs_df'].iterrows()) if not data['worst_pairs_df'].empty else 'n/a'}

Respond with only a JSON array, e.g.:
[{{"pair": "BTC/ETH", "signal": "warn", "magnitude": 0.3, "reason": "Consistent losses past 24h."}}]
If nothing actionable, return an empty array: []
"""
    try:
        content = chat(prompt, max_tokens=400, temperature=0.1, timeout=60)
    except LLMError as e:
        print(f"  [insights] LLM insights call failed: {e}", file=sys.stderr)
        return []
    arr, _, _ = _extract_json(content, '[')
    return arr if isinstance(arr, list) else []


def ask_report(prompt: str) -> str:
    """Ask for a free-text report (no JSON parsing, larger token budget)."""
    try:
        return chat(prompt, max_tokens=800, temperature=0.3, timeout=120)
    except LLMError as e:
        return f"[Report generation failed: {e}]"
