"""One autonomous research cycle.

  1. Load prices and split them in time: train | holdout (the most recent part).
  2. Measure the default rule strategy and baselines on TRAIN only.
  3. Ask the LLM (or the heuristic proposer) for hypotheses, seeing only train
     evidence and past verdicts whose data ended before this holdout.
  4. Pre-register every hypothesis (spec frozen, timestamped) before testing.
  5. Replay each on train and holdout. Each is judged against a *matched* random
     baseline: same timing, filters and exits, random pair (several seeds).
  6. Apply the fixed verdict rule (stats.verdict); re-test earlier accepted
     hypotheses on the new holdout and retire those that no longer hold.
  7. Write everything to the DB and a journal entry.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from ..llm import LLMError, chat, ping_llm
from ..store import get_conn
from . import stats
from .data import load_price_matrix
from .hypotheses import propose_heuristic, propose_llm, spec_key
from .journal import write_entry
from .replay import StrategySpec, basket_return_pct, replay, sanitize_spec

RANDOM_SEEDS  = (1, 2, 3, 4, 5)
MIN_TRADES    = 20
MIN_BARS      = 288 * 2        # two days of 5-minute bars


def _now():
    return datetime.now(timezone.utc)


def _random_matched(prices, spec: StrategySpec, start=None) -> list[dict]:
    rnd = StrategySpec(**{**spec.params(), 'selector': 'random', 'name': spec.name + '-random'})
    out = []
    for seed in RANDOM_SEEDS:
        out += replay(prices, rnd, seed=seed, start=start)
    return out


def _segment_stats(trades, baseline_trades, seed=0) -> tuple[dict, dict]:
    return stats.summarize(trades, seed=seed), stats.compare(trades, baseline_trades, seed=seed)


def _breakdown(trades: list[dict]) -> dict:
    """Training-period slices the proposer can reason about."""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df['majors_quote'] = df['quote'].isin(['BTC', 'ETH'])
    df['z_bucket'] = pd.cut(df['entry_z'], [-99, -4, -3, -2.5, 0], labels=['<-4', '-4..-3', '-3..-2.5', '>-2.5'])
    out = {}
    for col in ('majors_quote', 'z_bucket', 'reason'):
        g = df.groupby(col, observed=True)['ret_pct'].agg(['count', 'mean'])
        out[col] = {str(k): {'n': int(r['count']), 'mean_pct': round(float(r['mean']), 3)}
                    for k, r in g.iterrows()}
    return out


def _llm_picker(max_calls: int = 150):
    """Selector for 'llm' strategies: one short LLM call per decision (cached)."""
    calls = {'n': 0}

    def pick(table: pd.DataFrame, ts) -> str | None:
        if calls['n'] >= max_calls:
            return None
        calls['n'] += 1
        rows = "\n".join(f"  {r.pair}: z={r.z_score:+.2f}" for r in table.head(8).itertuples())
        prompt = (f"Time {pd.Timestamp(ts).isoformat()}. Pairs BASE/QUOTE whose ratio is unusually low "
                  f"(robust z-score):\n{rows}\nWhich ONE is most likely to revert upward within a day? "
                  'Reply with only JSON: {"pair": "BASE/QUOTE"} or {"pair": null}.')
        try:
            from ..core import _extract_json
            obj, _, _ = _extract_json(chat(prompt, max_tokens=60, temperature=0.0, timeout=60), '{')
        except LLMError:
            return None
        return obj.get('pair') if isinstance(obj, dict) else None
    return pick


def _logged_llm_picker(con, start, end):
    """Replays the live system's own historical LLM picks (llm_verdicts), so the
    real LLM can be compared with rule and random at the same moments for free."""
    rows = con.execute("""
        SELECT timestamp, top_pair FROM llm_verdicts
        WHERE action = 'accumulate' AND mode = 'pair' AND top_pair LIKE '%/%'
          AND timestamp >= ? AND timestamp <= ?
    """, [start, end]).fetchall()
    if len(rows) < MIN_TRADES:
        return None
    picks = pd.Series({pd.Timestamp(t).tz_convert('UTC').floor('10min'): p for t, p in rows})

    def pick(table: pd.DataFrame, ts) -> str | None:
        p = picks.get(pd.Timestamp(ts).floor('10min'))
        return p if p in set(table['pair']) else None
    return pick


def _tested_context(con, holdout_start) -> tuple[list[dict], set]:
    rows = con.execute("""
        SELECT h.statement, h.status, h.spec, r.data_end
        FROM hypotheses h LEFT JOIN research_runs r USING (run_id)
        WHERE h.author != 'baseline' ORDER BY h.created_at
    """).fetchall()
    tested, keys = [], set()
    for statement, status, spec_json, data_end in rows:
        spec = sanitize_spec(json.loads(spec_json))
        keys.add(spec_key(spec))
        item = {'statement': statement}
        # Only reveal outcomes whose data ended before this holdout begins.
        if data_end is not None and pd.Timestamp(data_end) <= pd.Timestamp(holdout_start):
            item['outcome'] = status
        tested.append(item)
    return tested, keys


def run_research_cycle(n_hypotheses: int = 3, holdout_frac: float = 0.3,
                       use_llm: bool | None = None, bar_minutes: int = 5,
                       days: float | None = None) -> dict:
    run_id = _now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
    print(f"[{_now():%H:%M:%S}] Research cycle {run_id}")

    con = get_conn()
    start = None
    if days:
        latest = con.execute("SELECT MAX(timestamp) FROM market_history").fetchone()[0]
        if latest is not None:
            start = pd.Timestamp(latest) - pd.Timedelta(days=days)
    prices = load_price_matrix(con, bar_minutes=bar_minutes, start=start)
    if len(prices) < MIN_BARS:
        con.close()
        msg = f"not enough history ({len(prices)} bars of {bar_minutes}m; need {MIN_BARS})"
        print(f"  {msg}. Ingest longer, or load the demo dataset (--demo-load).")
        return {'run_id': run_id, 'status': 'skipped', 'reason': msg}

    data_start, data_end = prices.index[0], prices.index[-1]
    holdout_start = data_start + (data_end - data_start) * (1 - holdout_frac)
    train = prices[prices.index < holdout_start]
    tested, tested_keys = _tested_context(con, holdout_start)
    logged_picker = _logged_llm_picker(con, data_start, data_end)
    prior_accepted = con.execute("""
        SELECT hypothesis_id, statement, spec, author FROM hypotheses WHERE status = 'accepted'
    """).fetchall()
    con.close()
    print(f"  data {data_start:%Y-%m-%d %H:%M} → {data_end:%Y-%m-%d %H:%M} "
          f"({prices.shape[1]} assets), holdout from {holdout_start:%Y-%m-%d %H:%M}")

    # 2. training evidence for the proposer
    base_spec = StrategySpec(name='baseline-rule')
    base_train = replay(train, base_spec)
    rand_train = _random_matched(train, base_spec)
    context = {
        'train_start': f"{data_start:%Y-%m-%d}", 'train_end': f"{holdout_start:%Y-%m-%d %H:%M}",
        'train': {'default_rule_strategy': {**stats.summarize(base_train), 'spec': base_spec.params()},
                  'random_baseline': stats.summarize(rand_train),
                  'buy_and_hold_basket_pct': basket_return_pct(train),
                  'rule_strategy_breakdown': _breakdown(base_train)},
        'tested': tested,
    }

    # 3. proposals
    llm_ok = ping_llm() if use_llm is None else (use_llm and ping_llm())
    proposals, proposer = [], 'heuristic'
    if llm_ok:
        try:
            proposals = propose_llm(context, n_hypotheses, allow_llm_selector=False)
            proposer = 'llm'
        except LLMError as e:
            print(f"  [propose] LLM unavailable ({e}); using heuristic proposer")
    seen = set(tested_keys)
    proposals = [p for p in proposals if spec_key(p['spec']) not in seen
                 and not seen.add(spec_key(p['spec']))]
    if len(proposals) < n_hypotheses:
        proposals += propose_heuristic(n_hypotheses - len(proposals), seen, seed=len(tested))
    print(f"  proposer: {proposer} ({len(proposals)} hypotheses)")

    # 4. pre-register (frozen spec + timestamp) before any holdout replay
    con = get_conn()
    con.execute("INSERT INTO research_runs (run_id, started_at, data_start, holdout_start, data_end,"
                " n_assets, proposer) VALUES (?,?,?,?,?,?,?)",
                [run_id, _now(), data_start, holdout_start, data_end, int(prices.shape[1]), proposer])
    entries = [{'hypothesis_id': f"{run_id}-base", 'author': 'baseline', 'spec': base_spec,
                'statement': 'Default rule strategy (what the live system trades).', 'rationale': ''}]
    for k, p in enumerate(proposals):
        entries.append({'hypothesis_id': f"{run_id}-h{k + 1}", **p})
    if logged_picker is not None:
        entries.append({'hypothesis_id': f"{run_id}-llm-live", 'author': 'baseline',
                        'spec': StrategySpec(name='live-llm-picks', selector='llm'),
                        'statement': "The live system's own logged LLM picks, replayed with default exits.",
                        'rationale': '', 'picker': logged_picker})
    for e in entries:
        con.execute("INSERT INTO hypotheses (hypothesis_id, run_id, created_at, author, statement,"
                    " rationale, spec, min_trades, status) VALUES (?,?,?,?,?,?,?,?, 'registered')",
                    [e['hypothesis_id'], run_id, _now(), e['author'], e['statement'],
                     e.get('rationale', ''), json.dumps(e['spec'].params()), MIN_TRADES])
    con.close()

    # 5-6. test
    rows = []
    for e in entries:
        spec = e['spec']
        picker = e.get('picker') or (_llm_picker() if spec.selector == 'llm' else None)
        tr = replay(train, spec, picker=picker)
        ho = replay(prices, spec, picker=picker, start=holdout_start)
        tr_s, tr_c = _segment_stats(tr, _random_matched(train, spec))
        ho_s, ho_c = _segment_stats(ho, _random_matched(prices, spec, start=holdout_start))
        v = stats.verdict(ho_s, ho_c, MIN_TRADES)
        rows.append({**e, 'name': spec.name, 'train': tr_s, 'train_vs_random': tr_c,
                     'holdout': ho_s, 'vs_random': ho_c, 'verdict': v,
                     'equity': stats.equity_curve(ho)})
        print(f"  {spec.name:28s} holdout n={ho_s['n']:4d} mean={ho_s['mean_pct']} "
              f"vs random={ho_c['diff_pct']} [{ho_c['ci_low']}, {ho_c['ci_high']}] → {v}")

    retired = []
    for hid, statement, spec_json, author in prior_accepted:
        spec = sanitize_spec(json.loads(spec_json))
        ho = replay(prices, spec, start=holdout_start)
        s, c = _segment_stats(ho, _random_matched(prices, spec, start=holdout_start))
        if stats.verdict(s, c, MIN_TRADES) == 'rejected':
            retired.append((hid, statement, s, c))

    # 7. persist
    basket = basket_return_pct(prices, start=holdout_start)
    run = {'run_id': run_id, 'data_start': f"{data_start:%Y-%m-%d %H:%M}",
           'data_end': f"{data_end:%Y-%m-%d %H:%M}", 'holdout_start': f"{holdout_start:%Y-%m-%d %H:%M}",
           'basket_holdout_pct': basket, 'proposer': proposer}
    con = get_conn()
    for r in rows:
        for seg, s, c, eq in (('train', r['train'], r['train_vs_random'], None),
                              ('holdout', r['holdout'], r['vs_random'], r['equity'])):
            con.execute("INSERT OR REPLACE INTO strategy_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [run_id, r['hypothesis_id'], seg, s['n'], s['days'], s['mean_pct'],
                         s['ci_low'], s['ci_high'], s['win_rate'], s['total_pct'],
                         c['diff_pct'], c['ci_low'], c['ci_high'], json.dumps(eq) if eq else None])
        status = r['verdict'] if r['author'] != 'baseline' else 'baseline'
        con.execute("UPDATE hypotheses SET status=?, decided_at=?, result=? WHERE hypothesis_id=?",
                    [status, _now(), json.dumps({'holdout': r['holdout'], 'vs_random': r['vs_random'],
                                                 'train': r['train']}, default=str), r['hypothesis_id']])
    for hid, *_ in retired:
        con.execute("UPDATE hypotheses SET status='retired', decided_at=? WHERE hypothesis_id=?", [_now(), hid])
    entry = write_entry(con, run, rows, use_llm=llm_ok)
    summary = {'proposer': proposer, 'journal_author': entry['author'],
               'accepted': [r['name'] for r in rows if r['verdict'] == 'accepted' and r['author'] != 'baseline'],
               'retired': [h for h, *_ in retired], 'basket_holdout_pct': basket}
    con.execute("UPDATE research_runs SET finished_at=?, summary=? WHERE run_id=?",
                [_now(), json.dumps(summary), run_id])
    con.close()
    print(f"  accepted: {summary['accepted'] or 'none'}; retired: {summary['retired'] or 'none'}; "
          f"journal by {entry['author']}")
    return {'run_id': run_id, 'status': 'done', **summary, 'rows': rows}

