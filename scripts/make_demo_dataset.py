"""Build the bundled demo dataset: public 5-minute closes for the core assets.

    python scripts/make_demo_dataset.py --days 21 --out demo/prices_5m.csv.gz

Uses the exchange in OSMIA_EXCHANGE (default binance). Public market data only;
no API key. The output loads with `python -m osmia --demo-load <file>`.
"""
import argparse
import gzip
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from osmia.config import ASSETS  # noqa: E402
from osmia.sources.market import make_exchange  # noqa: E402


def fetch(ex, symbol: str, since_ms: int, until_ms: int) -> list[list]:
    rows, cursor = [], since_ms
    while cursor < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe='5m', since=cursor, limit=1000)
        if not batch:
            break
        rows += [r for r in batch if r[0] < until_ms]
        cursor = batch[-1][0] + 5 * 60 * 1000
        time.sleep(ex.rateLimit / 1000)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=float, default=21)
    ap.add_argument('--out', default='demo/prices_5m.csv.gz')
    args = ap.parse_args()

    ex = make_exchange(rate_limit=True)
    ex.load_markets()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    since = end - timedelta(days=args.days)
    since_ms, until_ms = int(since.timestamp() * 1000), int(end.timestamp() * 1000)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out, 'wt', encoding='utf-8', newline='') as f:
        f.write('timestamp,asset_id,price_usd\n')
        for asset in ASSETS:
            symbol = f'{asset}/USDT'
            if symbol not in ex.markets:
                print(f'  skip {symbol}: not listed on {ex.id}')
                continue
            rows = fetch(ex, symbol, since_ms, until_ms)
            for ts, _o, _h, _l, close, _v in rows:
                iso = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                f.write(f'{iso},{asset},{close}\n')
            n += len(rows)
            print(f'  {symbol}: {len(rows)} bars')
    print(f'Wrote {n} rows ({since:%Y-%m-%d} → {end:%Y-%m-%d}, {ex.id}) to {out}')


if __name__ == '__main__':
    main()
