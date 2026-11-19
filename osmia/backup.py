"""DuckDB checkpoint + copy backup with retention pruning.

Run with `python -m osmia --backup` (the scheduler does it at 06:00 and 18:00 UTC).
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

import duckdb

from . import config

RETENTION_DAYS = 5


def run_backup() -> Path | None:
    db_path    = Path(config.DB_PATH)
    backup_dir = config.OSMIA_HOME / 'backups'
    if not db_path.exists():
        print(f'  No database at {db_path}; nothing to back up.')
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts          = datetime.now().strftime('%Y-%m-%d_%H-%M')
    backup_path = backup_dir / f'market_data_{ts}.duckdb'

    # Flush WAL to the main file before copying
    try:
        con = duckdb.connect(str(db_path))
        con.execute('CHECKPOINT')
        con.close()
    except Exception as e:
        print(f'  Checkpoint failed (copying anyway): {e}', file=sys.stderr)

    shutil.copy2(str(db_path), str(backup_path))
    size_mb = backup_path.stat().st_size / 1_000_000
    print(f'[{datetime.now():%H:%M:%S}] Backup written: {backup_path.name}  ({size_mb:.1f} MB)')

    # Prune backups older than RETENTION_DAYS
    cutoff  = datetime.now().timestamp() - RETENTION_DAYS * 86400
    removed = []
    for f in sorted(backup_dir.glob('market_data_*.duckdb')):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed.append(f.name)
    if removed:
        print(f'  Pruned {len(removed)} backup(s): {", ".join(removed)}')

    remaining = sorted(backup_dir.glob('market_data_*.duckdb'))
    print(f'  {len(remaining)} backup(s) on disk '
          f'({sum(f.stat().st_size for f in remaining)/1_000_000:.0f} MB total)')
    return backup_path


if __name__ == '__main__':
    run_backup()
