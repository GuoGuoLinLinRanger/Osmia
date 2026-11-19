"""End-to-end analysis cycle with a fake OpenAI-compatible LLM and no internet.

Seeds a temp DuckDB with flat prices and one SOL dump, stubs the network
sources, then runs the real `python -m osmia --analyze` code path.
"""
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import duckdb
import numpy as np
import pytest

PRICES = {"BTC": 90000, "ETH": 3000, "SOL": 150, "ADA": 0.5, "DOT": 4, "XRP": 2,
          "TRX": 0.25, "BNB": 900, "AVAX": 20, "LTC": 90, "XLM": 0.3, "APT": 5,
          "DOGE": 0.15, "SUI": 2}


class _FakeLLM(BaseHTTPRequestHandler):
    prompts: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):                       # /v1/models reachability check
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        _FakeLLM.prompts.append(prompt)
        m = re.search(r"\[rank 1\]\s+(\S+)", prompt)
        if m:   # verdict: pretty-printed JSON in a fence, mixed-case action
            content = ("```json\n" + json.dumps({"action": "Accumulate", "top_pair": m.group(1),
                       "rank": [m.group(1)], "thesis": "Dislocated.", "risks": "Trend."},
                       indent=2) + "\n```\nProse.")
        else:   # open-position review
            content = "HOLD\nStill dislocated.\nUrgency: 2"
        out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def fake_llm(tv):
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = tv.config.LLM_URL, tv.config.LLM_PROVIDER
    tv.config.LLM_URL = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
    tv.config.LLM_PROVIDER = "openai"
    _FakeLLM.prompts = []
    yield _FakeLLM
    tv.config.LLM_URL, tv.config.LLM_PROVIDER = old
    srv.shutdown()


@pytest.fixture
def seeded_db(tv, tmp_path, monkeypatch):
    db = tmp_path / "osmia.duckdb"
    monkeypatch.setattr(tv.config, "DB_PATH", str(db))
    monkeypatch.setattr(tv.config, "BALANCES_FILE", str(tmp_path / "balances.json"))
    (tmp_path / "balances.json").write_text(json.dumps({a: 10.0 for a in PRICES} | {"USDT": 1000.0}))
    rng = np.random.default_rng(1)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for i in range(90):
        ts = now - timedelta(minutes=89 - i)
        for a, p in PRICES.items():
            px = p * 0.92 if (a == "SOL" and i == 89) else p * (1 + rng.normal(0, 0.001))
            rows.append((ts, a, px, 1e6, 1.0, 0.0, 0.0005))
    con = duckdb.connect(str(db))
    tv.init_db(con)
    con.executemany("INSERT INTO market_history (timestamp, asset_id, price_usd, volume_24h,"
                    " vwap_ratio, volume_z, spread_pct) VALUES (?,?,?,?,?,?,?)", rows)
    con.close()
    return db


@pytest.fixture
def offline(monkeypatch):
    """Stub every internet source the cycle touches."""
    import osmia.agent as agent
    from osmia.sources.market import _default_ob_metrics
    monkeypatch.setattr(agent, "fetch_market_context",
                        lambda: {"fng_value": 50, "fng_label": "Neutral", "headlines": []})
    monkeypatch.setattr(agent, "fetch_macro_context", lambda: {})
    monkeypatch.setattr(agent, "fetch_all_news", lambda assets: [])
    monkeypatch.setattr(agent, "fetch_order_book_metrics", lambda b, q, n: _default_ob_metrics())
    monkeypatch.setattr(agent, "MIN_CONFIDENCE_TRADE", 3)   # synthetic data -> low confidence


def test_analyze_cycle_opens_trade_in_signal_direction(tv, fake_llm, seeded_db, offline):
    from osmia.cli import main
    main(["--analyze"])

    con = duckdb.connect(str(seeded_db), read_only=True)
    trades = con.execute("SELECT sell_asset, buy_asset, status FROM fantasy_trades").fetchall()
    verdict = con.execute("SELECT top_pair, action FROM llm_verdicts").fetchone()
    con.close()

    assert verdict[1] == "accumulate"                 # fenced, mixed-case JSON parsed
    base, quote = verdict[0].split("/")
    assert base == "SOL"                              # the dumped asset is the cheap base
    assert trades == [(quote, "SOL", "open")]         # sold quote, bought SOL
    assert any("[OPEN POSITION REVIEW]" in p for p in fake_llm.prompts)


def test_unreachable_llm_means_pass_not_crash(tv, seeded_db, offline, monkeypatch):
    monkeypatch.setattr(tv.config, "LLM_URL", "http://127.0.0.1:9/v1/chat/completions")
    v = tv.ask_verdict("anything")
    assert v["action"] == "pass" and "failed" in v["prose"]


def test_anthropic_provider_without_sdk_falls_back_to_pass(tv, monkeypatch):
    import importlib.util
    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("anthropic SDK installed")
    monkeypatch.setattr(tv.config, "LLM_PROVIDER", "anthropic")
    v = tv.ask_verdict("anything")
    assert v["action"] == "pass" and "osmia[anthropic]" in v["prose"]
    assert tv.ping_llm() is False


def test_make_exchange_honours_config(tv, monkeypatch):
    monkeypatch.setattr(tv.config, "EXCHANGE_ID", "kraken")
    assert tv.make_exchange().id == "kraken"
    monkeypatch.setattr(tv.config, "EXCHANGE_ID", "not-an-exchange")
    with pytest.raises(SystemExit):
        tv.make_exchange()


def test_scheduler_job_failure_does_not_escape(capsys):
    from osmia.scheduler import _run
    _run("bad-args", ["--no-such-flag"])          # argparse exits with code 2
    assert "exited with 2" in capsys.readouterr().out
