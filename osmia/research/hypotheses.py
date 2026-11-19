"""Hypothesis proposals: the LLM reads training-period evidence and proposes
testable strategy variants; every proposal is clamped into a StrategySpec
before anything is tested (see replay.BOUNDS). When no LLM is reachable a
deterministic heuristic proposer explores the same space, and its entries are
labelled 'heuristic' everywhere so nobody mistakes them for model output.
"""
from __future__ import annotations

import json

import numpy as np

from ..core import _extract_json
from ..llm import LLMError, chat
from .replay import BOUNDS, CHOICES, StrategySpec, sanitize_spec

# Variants the heuristic proposer cycles through, most informative first.
HEURISTIC_TEMPLATES = [
    ("deeper-entry", "Deeper dislocations revert more reliably than marginal ones.",
     {'z_entry': -3.0}),
    ("majors-quote", "Pairs quoted in BTC/ETH revert better than alt-vs-alt pairs.",
     {'quotes': 'majors'}),
    ("btc-up-only", "Mean reversion works better when BTC's 24h trend is up.",
     {'btc_regime': 'up'}),
    ("btc-down-only", "Mean reversion works better when BTC's 24h trend is down.",
     {'btc_regime': 'down'}),
    ("long-window", "A 48h baseline gives a steadier 'normal' ratio than 24h.",
     {'window_hours': 48.0}),
    ("short-hold", "Reversions that matter happen within 6h; holding longer adds noise.",
     {'max_hold_hours': 6.0}),
    ("early-exit", "Exiting at z > -0.5 locks in most of the reversion with less risk.",
     {'z_exit': -0.5}),
    ("tight-stop", "A -4% stop cuts the losers that are trends, not dislocations.",
     {'stop_loss_pct': -4.0}),
    ("single-position", "One position at a time avoids correlated bets.",
     {'max_open': 1}),
]


def _space_description() -> str:
    lines = [f"  {k}: number in [{lo}, {hi}]" for k, (lo, hi) in BOUNDS.items()]
    lines += [f"  {k}: one of {list(v)}" for k, v in CHOICES.items()]
    return "\n".join(lines)


def build_proposal_prompt(context: dict, n: int, allow_llm_selector: bool) -> str:
    tested = context.get('tested', [])
    tested_txt = "\n".join(
        f"  - {t['statement']} -> {t.get('outcome', 'outcome hidden (overlaps current holdout)')}"
        for t in tested[-12:]) or "  (none yet)"
    sel_note = ("" if allow_llm_selector else
                "  (selector 'llm' is unavailable in this run; use 'rule')\n")
    return f"""You are the research lead of an autonomous crypto market research system.
The system paper-trades pair mean reversion: when a pair's price ratio BASE/QUOTE is
unusually low (robust z-score below z_entry over a trailing window), it sells QUOTE and
buys BASE, and exits when z rises above z_exit, after max_hold_hours, or at a stop loss.
Returns below are per trade, net of fees, in percent.

TRAINING-PERIOD EVIDENCE ({context['train_start']} to {context['train_end']}):
{json.dumps(context['train'], indent=1)}

Previously tested hypotheses:
{tested_txt}

Propose {n} NEW, falsifiable hypotheses that could beat the default rule strategy on
unseen data. Each one changes a few parameters of this space:
{_space_description()}
{sel_note}
Each will be pre-registered and judged ONLY on a later holdout period you have not seen,
against a random-pair baseline with the same timing and exits. Prefer hypotheses with a
clear mechanism; avoid fitting noise in the training numbers.

Respond with ONLY a JSON array of {n} objects:
[{{"name": "short-kebab-name", "statement": "one-sentence falsifiable claim",
  "rationale": "why, citing the evidence", "params": {{"z_entry": -3.0}}}}]"""


def propose_llm(context: dict, n: int, allow_llm_selector: bool = False) -> list[dict]:
    """Ask the LLM; return validated proposals (possibly fewer than n). Raises LLMError."""
    content = chat(build_proposal_prompt(context, n, allow_llm_selector),
                   max_tokens=1200, temperature=0.4, timeout=120)
    arr, _, _ = _extract_json(content, '[')
    out = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        params = item.get('params') if isinstance(item.get('params'), dict) else {}
        if not allow_llm_selector and str(params.get('selector', '')).lower() == 'llm':
            params = {**params, 'selector': 'rule'}
        spec = sanitize_spec(params, name=item.get('name'))
        out.append({'author': 'llm', 'spec': spec,
                    'statement': str(item.get('statement', ''))[:300] or spec.name,
                    'rationale': str(item.get('rationale', ''))[:600]})
    return out[:n]


def propose_heuristic(n: int, tested_specs: set, seed: int = 0) -> list[dict]:
    """Untested template variants first, then random combinations of two."""
    out = []
    for name, statement, params in HEURISTIC_TEMPLATES:
        spec = sanitize_spec(params, name=name)
        if spec.params() and _spec_key(spec) not in tested_specs:
            out.append({'author': 'heuristic', 'spec': spec, 'statement': statement,
                        'rationale': 'Heuristic proposer (no LLM available): next untested template.'})
        if len(out) >= n:
            return out
    rng = np.random.default_rng(seed)
    for _ in range(50):
        if len(out) >= n:
            break
        a, b = rng.choice(len(HEURISTIC_TEMPLATES), size=2, replace=False)
        (na, sa, pa), (nb, sb, pb) = HEURISTIC_TEMPLATES[a], HEURISTIC_TEMPLATES[b]
        spec = sanitize_spec({**pa, **pb}, name=f"{na}+{nb}")
        if _spec_key(spec) in tested_specs or any(_spec_key(o['spec']) == _spec_key(spec) for o in out):
            continue
        out.append({'author': 'heuristic', 'spec': spec,
                    'statement': f"{sa.rstrip('.')}; and {sb[0].lower()}{sb[1:]}",
                    'rationale': 'Heuristic proposer (no LLM available): combination of two templates.'})
    return out


def _spec_key(spec: StrategySpec) -> str:
    p = spec.params()
    p.pop('name', None)
    return json.dumps(p, sort_keys=True)


def spec_key(spec: StrategySpec) -> str:
    return _spec_key(spec)


__all__ = ['propose_llm', 'propose_heuristic', 'build_proposal_prompt', 'spec_key', 'LLMError']
