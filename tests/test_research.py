"""Research layer: replay has no lookahead, stats are honest, the loop pre-registers,
never repeats itself, never leaks holdout outcomes, and the LLM cache works."""
import json
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pytest

from osmia.research import stats
from osmia.research.replay import StrategySpec, replay, sanitize_spec


def ou_prices(n_assets=6, days=6, half_life_h=3.0, sd=0.015, seed=0):
    """Log price = common random walk + per-asset mean-reverting (OU) deviation.
    Pair ratios therefore revert: a rule strategy should beat random entries."""
    rng = np.random.default_rng(seed)
    bars = int(days * 288)
    theta = np.log(2) / (half_life_h * 12)
    sig = sd * np.sqrt(2 * theta)
    market = np.cumsum(rng.normal(0, 0.001, bars))
    x = np.zeros((bars, n_assets))
    for t in range(1, bars):
        x[t] = x[t - 1] * (1 - theta) + rng.normal(0, sig, n_assets)
    idx = pd.date_range('2026-01-01', periods=bars, freq='5min', tz='UTC')
    names = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP'][:n_assets]
    return pd.DataFrame(np.exp(market[:, None] + x) * 100, index=idx, columns=names)


@pytest.fixture(scope='module')
def prices():
    return ou_prices()


# ── replay ───────────────────────────────────────────────────────────────────

def test_replay_has_no_lookahead(prices):
    """Entries before time T must be identical whether or not data after T exists."""
    cut = prices.index[len(prices) // 2]
    spec = StrategySpec(max_hold_hours=2)
    full = replay(prices, spec)
    part = replay(prices[prices.index <= cut], spec)
    key = lambda ts: [(t['pair'], t['entry_ts']) for t in ts if t['entry_ts'] < cut - timedelta(hours=3)]
    assert key(full) == key(part) and len(key(full)) > 5


def test_rule_beats_random_on_mean_reverting_data(prices):
    rule = replay(prices, StrategySpec())
    rnd = []
    for s in (1, 2, 3):
        rnd += replay(prices, StrategySpec(selector='random'), seed=s)
    cmp = stats.compare(rule, rnd)
    assert stats.summarize(rule)['mean_pct'] > 0
    assert cmp['diff_pct'] > 0 and cmp['ci_low'] > 0


def test_returns_are_net_of_fees(prices):
    t = replay(prices, StrategySpec())[0]
    assert t['cost_pct'] > 0
    assert t['ret_pct'] == pytest.approx(t['gross_pct'] - t['cost_pct'], abs=1e-3)


def test_start_skips_warmup_trades(prices):
    start = prices.index[len(prices) * 2 // 3]
    trades = replay(prices, StrategySpec(), start=start)
    assert trades and min(t['entry_ts'] for t in trades) >= start


def test_sanitize_spec_clamps_everything():
    s = sanitize_spec({'z_entry': -50, 'z_exit': 99, 'window_hours': 'lots', 'quotes': 'weird',
                       'selector': 'llm', 'max_open': 9.7, 'recent_hours': 100, 'evil': 1}, name='x' * 99)
    assert s.z_entry == -5.0 and s.z_exit == 1.5 and s.window_hours == 24.0
    assert s.quotes == 'any' and s.selector == 'llm' and s.max_open == 5
    assert s.recent_hours <= s.window_hours / 2 and len(s.name) == 60
    assert sanitize_spec({'z_entry': -1.5, 'z_exit': -2.0}).z_exit == 0.0   # exit must be above entry


# ── stats ────────────────────────────────────────────────────────────────────

def _trades(rets, days):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [{'ret_pct': r, 'entry_ts': t0 + timedelta(days=d), 'exit_ts': t0 + timedelta(days=d, hours=1),
             'hold_hours': 1.0} for r, d in zip(rets, days)]


def test_day_block_ci_is_wider_than_naive_when_days_cluster():
    rng = np.random.default_rng(0)
    day_effect = rng.normal(0, 2, 10)
    rets = [day_effect[d] + rng.normal(0, 0.1) for d in range(10) for _ in range(20)]
    trades = _trades(rets, [d for d in range(10) for _ in range(20)])
    s = stats.summarize(trades)
    naive_half = 1.96 * np.std(rets) / np.sqrt(len(rets))
    assert (s['ci_high'] - s['ci_low']) / 2 > 2 * naive_half


def test_verdict_rules():
    good = {'n': 50, 'days': 5, 'mean_pct': 0.3}
    assert stats.verdict(good, {'ci_low': 0.1, 'ci_high': 0.5}, 20) == 'accepted'
    assert stats.verdict(good, {'ci_low': -0.5, 'ci_high': -0.1}, 20) == 'rejected'
    assert stats.verdict(good, {'ci_low': -0.1, 'ci_high': 0.5}, 20) == 'inconclusive'
    assert stats.verdict({**good, 'n': 10}, {'ci_low': 0.1, 'ci_high': 0.5}, 20) == 'inconclusive'
    assert stats.verdict({**good, 'mean_pct': -0.1}, {'ci_low': 0.1, 'ci_high': 0.5}, 20) == 'inconclusive'


# ── the research loop ────────────────────────────────────────────────────────

@pytest.fixture
def research_db(tv, tmp_path, monkeypatch, prices):
    db = tmp_path / 'r.duckdb'
    monkeypatch.setattr(tv.config, 'DB_PATH', str(db))
    con = duckdb.connect(str(db))
    tv.init_db(con)
    long = prices.stack().reset_index()
    long.columns = ['timestamp', 'asset_id', 'price_usd']
    con.execute("INSERT INTO market_history (timestamp, asset_id, price_usd) SELECT * FROM long")
    con.close()
    return db


def test_research_cycle_end_to_end(tv, research_db):
    from osmia.research.loop import run_research_cycle
    r1 = run_research_cycle(n_hypotheses=3, use_llm=False)
    assert r1['status'] == 'done' and r1['proposer'] == 'heuristic'
    con = duckdb.connect(str(research_db), read_only=True)
    hyps = con.execute("SELECT hypothesis_id, author, status, created_at, decided_at, spec "
                       "FROM hypotheses ORDER BY hypothesis_id").fetchall()
    assert len(hyps) == 4                                         # baseline + 3
    assert all(s in ('baseline', 'accepted', 'rejected', 'inconclusive') for _, _, s, *_ in hyps)
    assert all(created <= decided for *_, created, decided, _ in hyps)   # registered before decided
    assert con.execute("SELECT COUNT(*) FROM strategy_results").fetchone()[0] == 8
    assert con.execute("SELECT author FROM research_journal").fetchone()[0] == 'heuristic'
    base = con.execute("SELECT mean_pct, diff_ci_low FROM strategy_results "
                       "WHERE hypothesis_id LIKE '%-base' AND segment='holdout'").fetchone()
    assert base[0] > 0                                            # OU data: the rule strategy earns
    specs1 = {json.dumps({k: v for k, v in json.loads(s).items() if k != 'name'}, sort_keys=True)
              for *_, s in hyps if 'base' not in _[0]}
    con.close()

    run_research_cycle(n_hypotheses=3, use_llm=False)
    con = duckdb.connect(str(research_db), read_only=True)
    rows = con.execute("SELECT hypothesis_id, spec FROM hypotheses WHERE author='heuristic'").fetchall()
    con.close()
    specs = [json.dumps({k: v for k, v in json.loads(s).items() if k != 'name'}, sort_keys=True)
             for _, s in rows]
    assert len(specs) == 6 and len(set(specs)) == 6              # never re-tests the same spec
    assert specs1 <= set(specs)


def test_holdout_outcomes_are_hidden_from_the_proposer(tv, research_db):
    from osmia.research.loop import _tested_context, run_research_cycle
    run_research_cycle(n_hypotheses=2, use_llm=False)
    con = duckdb.connect(str(research_db), read_only=True)
    holdout_start = con.execute("SELECT holdout_start FROM research_runs").fetchone()[0]
    data_end = con.execute("SELECT data_end FROM research_runs").fetchone()[0]
    hidden, _ = _tested_context(con, holdout_start)        # same data: outcomes overlap -> hidden
    shown, _ = _tested_context(con, data_end + timedelta(days=1))   # new data later -> revealed
    con.close()
    assert hidden and all('outcome' not in t for t in hidden)
    assert all('outcome' in t for t in shown)


def test_llm_proposals_are_clamped_and_cannot_pick_llm_selector(tv, monkeypatch):
    import osmia.research.hypotheses as hyp
    reply = ('Here:\n```json\n[{"name": "wild", "statement": "s", "rationale": "r",'
             ' "params": {"z_entry": -40, "selector": "llm", "max_open": 50}}, "junk"]\n```')
    monkeypatch.setattr(hyp, 'chat', lambda *a, **k: reply)
    ctx = {'train_start': 'a', 'train_end': 'b', 'train': {}, 'tested': []}
    (p,) = hyp.propose_llm(ctx, 3)
    assert p['author'] == 'llm' and p['spec'].z_entry == -5.0
    assert p['spec'].selector == 'rule' and p['spec'].max_open == 5


# ── LLM cache ────────────────────────────────────────────────────────────────

def test_llm_cache_record_then_replay(tv, tmp_path, monkeypatch):
    import osmia.llm as llm
    monkeypatch.setattr(tv.config, 'LLM_CACHE_PATH', str(tmp_path / 'c.sqlite'))
    monkeypatch.setattr(tv.config, 'LLM_PROVIDER', 'openai')
    calls = []
    monkeypatch.setattr(llm, '_chat_openai', lambda *a: calls.append(1) or '{"action":"pass"}')
    monkeypatch.setattr(tv.config, 'LLM_CACHE', 'readwrite')
    assert llm.chat('p1') == '{"action":"pass"}' and llm.chat('p1') == '{"action":"pass"}'
    assert len(calls) == 1                                        # second answer came from cache
    monkeypatch.setattr(tv.config, 'LLM_CACHE', 'readonly')
    assert llm.chat('p1') == '{"action":"pass"}'
    with pytest.raises(llm.LLMError):
        llm.chat('never seen')
    assert llm.ask_verdict('never seen')['action'] == 'pass'
    assert len(calls) == 1                                        # readonly never calls out
