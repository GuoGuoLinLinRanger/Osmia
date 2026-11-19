"""Research journal: one markdown entry per research cycle.

The LLM writes the entry from the run's final numbers (after every verdict is
decided, so it cannot influence them). Without an LLM, a plain factual entry is
generated instead and labelled 'heuristic'.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..llm import LLMError, chat


def _fmt(v, nd=3, pct=True):
    if v is None:
        return 'n/a'
    return f"{v:+.{nd}f}{'%' if pct else ''}"


def results_table(rows: list[dict]) -> str:
    head = ("| Hypothesis | Author | Holdout trades | Mean/trade | 95% CI | vs random (95% CI) | Verdict |\n"
            "|---|---|---|---|---|---|---|")
    body = []
    for r in rows:
        h, c = r['holdout'], r['vs_random']
        ci = f"[{_fmt(h['ci_low'])}, {_fmt(h['ci_high'])}]" if h['ci_low'] is not None else 'n/a'
        vs = (f"{_fmt(c['diff_pct'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]"
              if c['diff_pct'] is not None else 'n/a')
        body.append(f"| {r['name']} | {r['author']} | {h['n']} | {_fmt(h['mean_pct'])} | {ci} | {vs} | **{r['verdict']}** |")
    return head + "\n" + "\n".join(body)


def factual_entry(run: dict, rows: list[dict]) -> str:
    acc = [r['name'] for r in rows if r['verdict'] == 'accepted']
    return (f"## Research cycle {run['run_id']}\n\n"
            f"Data {run['data_start']} → {run['data_end']}; holdout from {run['holdout_start']}. "
            f"Buy-and-hold basket over the holdout: {_fmt(run.get('basket_holdout_pct'))}.\n\n"
            f"{results_table(rows)}\n\n"
            f"**Accepted:** {', '.join(acc) if acc else 'none'}. "
            "Inconclusive means too few trades or a confidence interval that straddles zero; "
            "it is not evidence either way.\n")


def llm_entry(run: dict, rows: list[dict]) -> str:
    """Raises LLMError when no LLM is available."""
    payload = [{'name': r['name'], 'author': r['author'], 'statement': r['statement'],
                'train': r['train'], 'holdout': r['holdout'], 'vs_random': r['vs_random'],
                'verdict': r['verdict']} for r in rows]
    prompt = f"""You keep the lab notebook of an autonomous crypto market research system.
Write this cycle's entry in markdown (max 250 words, no title line). Sections:
**What we tested**, **What the holdout says**, **What we learned**, **Next**.
Be precise and sceptical: the verdict rule is fixed (accepted only if the 95% CI of
mean-return-minus-random is above zero on the holdout); 'inconclusive' is not a win;
training numbers do not count as evidence. Do not invent numbers.

Holdout basket buy-and-hold: {run.get('basket_holdout_pct')}%
Results (returns are % per trade, net of fees):
{json.dumps(payload, indent=1, default=str)}"""
    body = chat(prompt, max_tokens=900, temperature=0.3, timeout=120).strip()
    if not body:
        raise LLMError("empty journal entry")
    return f"## Research cycle {run['run_id']}\n\n{body}\n\n{results_table(rows)}\n"


def write_entry(con, run: dict, rows: list[dict], use_llm: bool) -> dict:
    author, body = 'heuristic', None
    if use_llm:
        try:
            body, author = llm_entry(run, rows), 'llm'
        except LLMError as e:
            print(f"  [journal] LLM unavailable ({e}); writing factual entry")
    if body is None:
        body = factual_entry(run, rows)
    entry = {'entry_id': uuid.uuid4().hex[:12], 'created_at': datetime.now(timezone.utc),
             'run_id': run['run_id'], 'author': author,
             'title': f"Research cycle {run['run_id']}", 'body_md': body}
    con.execute("INSERT INTO research_journal VALUES (?,?,?,?,?,?)",
                [entry[k] for k in ('entry_id', 'created_at', 'run_id', 'author', 'title', 'body_md')])
    return entry
