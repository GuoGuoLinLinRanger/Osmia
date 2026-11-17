"""Tests for the v1.5 correctness fixes. Pure functions plus an in-memory DuckDB."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


# ── LLM output parsing ───────────────────────────────────────────────────────

def test_parse_single_line_json(tv):
    v = tv.parse_llm_verdict('{"action":"accumulate","top_pair":"SOL/BTC","rank":["SOL/BTC"],'
                             '"thesis":"t","risks":"r"}\n\nSome prose.')
    assert v["action"] == "accumulate"
    assert v["top_pair"] == "SOL/BTC"
    assert v["prose"] == "Some prose."


def test_parse_pretty_printed_and_fenced_json(tv):
    content = 'Here you go:\n```json\n{\n  "action": "Accumulate",\n  "top_pair": "ETH/BTC"\n}\n```\nDone.'
    v = tv.parse_llm_verdict(content)
    assert v["action"] == "accumulate"          # case-normalized
    assert v["top_pair"] == "ETH/BTC"
    assert "```" not in v["prose"]


def test_parse_without_json_is_pass(tv):
    v = tv.parse_llm_verdict("I think you should buy everything.")
    assert v["action"] == "pass"
    assert v["prose"] == "I think you should buy everything."


@pytest.mark.parametrize("raw,expected", [
    ("accumulate", "accumulate"), ("WATCH", "watch"), (" pass ", "pass"),
    ("hold", "pass"), ("buy", "pass"), (None, "pass"), (3, "pass"),
])
def test_normalize_action(tv, raw, expected):
    assert tv.normalize_action(raw) == expected


def test_extract_json_array(tv):
    arr, _, _ = tv._extract_json('Sure: [{"pair": "SOL", "signal": "warn"}] thanks', "[")
    assert arr == [{"pair": "SOL", "signal": "warn"}]


# ── Report insights: the LLM can't escape its bounds ─────────────────────────

def test_sanitize_insight_clamps_and_signs(tv):
    boost = tv.sanitize_insight({"pair": "sol/eth", "signal": "boost", "magnitude": 9})
    assert boost == {"pair": "SOL/ETH", "signal": "boost", "magnitude": 0.5, "reason": ""}
    warn = tv.sanitize_insight({"pair": "SOL", "signal": "warn", "magnitude": 0.3})
    assert warn["magnitude"] == -0.3            # warn is always negative


@pytest.mark.parametrize("bad", [
    {"pair": "ALL", "signal": "suppress"},       # no global kill switch
    {"pair": "SOL", "signal": "yolo"},
    {"pair": "DROP TABLE", "signal": "warn"},
    "not a dict",
])
def test_sanitize_insight_rejects(tv, bad):
    assert tv.sanitize_insight(bad) is None


def test_sanitize_insight_bad_magnitude_does_not_crash(tv):
    assert tv.sanitize_insight({"pair": "SOL", "signal": "boost", "magnitude": "lots"})["magnitude"] == 0.0


def test_apply_insights_warn_can_only_shrink(tv):
    # Old bug: a "warn" with a positive magnitude grew the position.
    conf, notional, sup = tv.apply_report_insights(
        [{"pair": "SOL/BTC", "signal": "warn", "magnitude": 0.4}], "SOL/BTC", 7, 100.0)
    assert conf < 7 and notional < 100.0 and not sup


def test_apply_insights_ignores_all_and_other_pairs(tv):
    ins = [{"pair": "ALL", "signal": "suppress", "magnitude": 0},
           {"pair": "ETH/ADA", "signal": "suppress", "magnitude": 0}]
    assert tv.apply_report_insights(ins, "SOL/BTC", 7, 100.0) == (7, 100.0, False)


def test_apply_insights_matches_leg(tv):
    _, _, sup = tv.apply_report_insights(
        [{"pair": "SOL", "signal": "suppress", "magnitude": 0}], "SOL/BTC", 7, 100.0)
    assert sup


def test_store_insights_filters_invalid(tv, con):
    n = tv._store_report_insights(con, [
        {"pair": "SOL/BTC", "signal": "boost", "magnitude": 0.2},
        {"pair": "ALL", "signal": "suppress", "magnitude": -1},
        {"pair": "ETH", "signal": "warn", "magnitude": "x"},
    ])
    assert n == 2
    assert {r["pair"] for r in tv.get_active_report_insights(con)} == {"SOL/BTC", "ETH"}


# ── Keyword matching ─────────────────────────────────────────────────────────

def test_keywords_are_whole_words(tv):
    assert tv.count_keywords("SEC sues exchange", ["sec"]) == 1
    assert tv.count_keywords("Bitcoin rallies for a second day", ["sec"]) == 0
    assert tv.count_keywords("Big bank adopts stablecoins", ["ban"]) == 0
    assert tv.count_keywords("India moves to ban crypto", ["ban"]) == 1


def test_mentions_asset(tv):
    assert tv.mentions_asset("Cardano upgrade ships", "ADA", "Cardano")
    assert tv.mentions_asset("ADA jumps 10%", "ADA", "Cardano")
    assert not tv.mentions_asset("Canada passes stablecoin law", "ADA", "Cardano")
    assert not tv.mentions_asset("an apt description", "APT", "Aptos")
    assert not tv.mentions_asset("the dot com bubble", "DOT", "Polkadot")


# ── P&L ──────────────────────────────────────────────────────────────────────

def test_trade_pnl_fiat_is_usd_pnl(tv):
    # Spent 100 USDT for 1 SOL (fees ignored); SOL now 110.
    assert tv.trade_pnl(100, 1.0, 1.0, 110, 100) == (10.0, 10.0)


def test_trade_pnl_pair_ignores_market_drift(tv):
    # Sold 1 BTC (100) for 1 SOL (100). Everything doubles: no relative move, no P&L.
    assert tv.trade_pnl(1, 200, 1, 200, 100) == (0.0, 0.0)
    # SOL +10% vs BTC flat: a 10% win.
    assert tv.trade_pnl(1, 100, 1, 110, 100) == (10.0, 10.0)


def test_mad_zscore(tv):
    vals = np.array([1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0])
    assert tv.mad_zscore(vals, 1.0) == 0.0
    assert tv.mad_zscore(vals, 0.9) < -2
    assert tv.mad_zscore(vals[:3], 0.5) == 0.0   # too few samples


# ── Signal direction + z for exits (in-memory DB) ────────────────────────────

def _seed_prices(tv, con, minutes=60, sol_last=90.0):
    """SOL and BTC flat with small noise; SOL dumps on the last minute."""
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(minutes):
        ts = t0 + timedelta(minutes=i)
        sol = 100 * (1 + rng.normal(0, 0.002)) if i < minutes - 1 else sol_last
        btc = 50000 * (1 + rng.normal(0, 0.002))
        rows += [(ts, "SOL", sol), (ts, "BTC", btc)]
    con.executemany("INSERT INTO market_history (timestamp, asset_id, price_usd) VALUES (?,?,?)", rows)


def test_dump_flags_cheap_base(tv, con):
    _seed_prices(tv, con)
    df = tv.get_raw_anomalies(con)
    assert "SOL/BTC" in set(df["pair"])          # SOL is the cheap base
    assert "BTC/SOL" not in set(df["pair"])
    z = tv.get_pair_zscores(con, ["SOL/BTC", "BTC/SOL"])
    assert z["SOL/BTC"] < -2 and z["BTC/SOL"] > 2


def test_reputation_score_sign(tv):
    assert tv.reputation_score(1.0, 1.0, 5) > 0.8
    assert tv.reputation_score(-1.0, 0.0, 5) < -0.8      # consistent loser is negative
    assert abs(tv.reputation_score(0.1, 0.5, 5)) < 0.2


def test_reputation_is_directional(tv, con):
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = [(t0 + timedelta(minutes=i), "SOL/BTC", "accumulate", 1.0) for i in range(3)]
    rows += [(t0 + timedelta(minutes=10 + i), "BTC/SOL", "accumulate", -1.0) for i in range(3)]
    con.executemany("INSERT INTO llm_verdicts (timestamp, top_pair, action, outcome_pct) "
                    "VALUES (?,?,?,?)", rows)
    tv._refresh_pair_reputation(con)
    reps = tv.get_pair_reputations(con, ["SOL/BTC", "BTC/SOL"])
    assert reps["SOL/BTC"] > 0 > reps["BTC/SOL"]


def _open_pair_trade(tv, con, z_entry=-3.0):
    """Mean reversion on SOL/BTC: sell BTC (quote), buy SOL (cheap base)."""
    balances = {"BTC": 1.0, "SOL": 0.0, "USDT": 0.0}
    trade = tv.execute_fantasy_pair_trade(con, "BTC", "SOL", 50000.0, 90.0, 1000.0,
                                          0.0, z_entry, 7, balances)
    trade["timestamp"] = datetime.now(timezone.utc)
    tv._insert_trade(con, trade)
    return trade


def test_z_crossback_closes_recovered_position(tv, con):
    _open_pair_trade(tv, con)
    prices = {"BTC": 50000.0, "SOL": 90.0}
    # Live z for the signal pair (BUY/SELL = SOL/BTC) has recovered above the
    # neutral-market threshold (0.0); P&L is flat so no other exit fires first.
    closed = tv._auto_close_trades(con, prices, {"SOL/BTC": 0.5}, fg_value=50)
    assert closed == 1
    assert con.execute("SELECT close_reason FROM fantasy_trades").fetchone()[0] == "z_crossback"


def test_no_crossback_while_still_dislocated(tv, con):
    _open_pair_trade(tv, con)
    closed = tv._auto_close_trades(con, {"BTC": 50000.0, "SOL": 90.0},
                                   {"SOL/BTC": -2.5}, fg_value=50)
    assert closed == 0


def test_close_pnl_is_relative_to_sold_leg(tv, con):
    trade = _open_pair_trade(tv, con)
    # Market-wide +20%: SOL and BTC both up 20%. Should be ~flat (minus fees),
    # not a +20% "win" like the old USD-only P&L reported.
    prices = {"BTC": 60000.0, "SOL": 108.0}
    tv._auto_close_trades(con, prices, {"SOL/BTC": 0.5}, fg_value=50)
    pnl_pct = con.execute("SELECT pnl_pct FROM fantasy_trades").fetchone()[0]
    assert -1.0 < pnl_pct < 0.0
    assert trade["sell_asset"] == "BTC" and trade["buy_asset"] == "SOL"


def test_evaluate_open_positions_empty_returns_tuple(tv, con):
    count, c = tv.evaluate_open_positions(con, {}, 50, {}, None, {})
    assert count == 0 and c is con


def test_news_veto_whole_words(tv, con):
    now = datetime.now(timezone.utc)
    con.execute("INSERT INTO asset_news (asset_id, source, headline, published_at, fetched_at) "
                "VALUES ('SOL','t','Solana rallies for a second straight day',?,?)", [now, now])
    assert not tv.check_news_veto(con, "SOL")
    con.execute("INSERT INTO asset_news (asset_id, source, headline, published_at, fetched_at) "
                "VALUES ('SOL','t2','Major Solana DeFi protocol suffers hack',?,?)", [now, now])
    assert tv.check_news_veto(con, "SOL")
