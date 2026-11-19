"""Price data for replay: a wide [bar time x asset] matrix of close prices."""
import gzip
import io
from pathlib import Path

import pandas as pd


def load_price_matrix(con, bar_minutes: int = 5, start=None, end=None,
                      assets: list[str] | None = None) -> pd.DataFrame:
    """Resample market_history into `bar_minutes` bars (last price per bar).
    Rows are bar timestamps (UTC), columns are asset ids. Bars where an asset
    has no data stay NaN; callers drop pairs without enough history."""
    where, params = ["price_usd > 0"], []
    if start is not None:
        where.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        where.append("timestamp < ?")
        params.append(end)
    if assets:
        where.append(f"asset_id IN ({','.join('?' for _ in assets)})")
        params += list(assets)
    long = con.execute(f"""
        SELECT time_bucket(INTERVAL '{int(bar_minutes)} minutes', timestamp) AS ts,
               asset_id, arg_max(price_usd, timestamp) AS px
        FROM market_history
        WHERE {' AND '.join(where)}
        GROUP BY 1, 2
        ORDER BY 1
    """, params).df()
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index='ts', columns='asset_id', values='px').sort_index()
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide


def import_price_csv(con, path: str | Path, source: str = 'demo') -> int:
    """Load a CSV(.gz) with columns timestamp,asset_id,price_usd[,volume_24h]
    into market_history. Existing (timestamp, asset_id) rows are kept."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(io.StringIO(f.read()))
    need = {'timestamp', 'asset_id', 'price_usd'}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if 'volume_24h' not in df.columns:
        df['volume_24h'] = None
    df['source'] = source
    before = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    con.execute("""
        INSERT OR IGNORE INTO market_history (timestamp, asset_id, price_usd, volume_24h, source)
        SELECT timestamp, asset_id, price_usd, volume_24h, source FROM df
    """)
    return con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0] - before
