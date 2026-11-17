import os
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep tests off any real install: no Discord, no LLM, temp home.
os.environ["OSMIA_DISCORD_WEBHOOK"] = ""
os.environ["OSMIA_LLM_URL"] = "http://127.0.0.1:9/unreachable"


@pytest.fixture(scope="session")
def tv(tmp_path_factory):
    home = tmp_path_factory.mktemp("osmia_home")
    os.environ["OSMIA_HOME"] = str(home)
    import TraderV13
    TraderV13.DB_PATH = str(home / "test.duckdb")
    TraderV13.BALANCES_FILE = str(home / "balances.json")
    return TraderV13


@pytest.fixture
def con(tv):
    c = duckdb.connect(":memory:")
    tv.init_db(c)
    yield c
    c.close()
