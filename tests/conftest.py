import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep tests off any real install: no Discord, no LLM.
os.environ["OSMIA_DISCORD_WEBHOOK"] = ""
os.environ["OSMIA_LLM_URL"] = "http://127.0.0.1:9/unreachable"

MODULES = ["config", "core", "store", "llm", "notify", "sources.market", "sources.news",
           "sources.sentiment", "signals", "portfolio", "mining", "learning", "agent"]


@pytest.fixture(scope="session")
def tv(tmp_path_factory):
    """Every public name of the osmia package in one namespace (tv.init_db, ...)."""
    home = tmp_path_factory.mktemp("osmia_home")
    os.environ["OSMIA_HOME"] = str(home)
    mods = [importlib.import_module(f"osmia.{m}") for m in MODULES]
    cfg = mods[0]
    cfg.DB_PATH = str(home / "test.duckdb")
    cfg.BALANCES_FILE = str(home / "balances.json")
    ns = {}
    for m in mods:
        ns.update({k: v for k, v in vars(m).items() if not k.startswith("__")})
    return SimpleNamespace(**ns)


@pytest.fixture
def con(tv):
    c = duckdb.connect(":memory:")
    tv.init_db(c)
    yield c
    c.close()
