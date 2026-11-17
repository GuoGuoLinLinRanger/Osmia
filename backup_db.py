#!/usr/bin/env python3
"""
backup_db.py — DuckDB checkpoint + copy backup with retention pruning.
Run via AI-Trader_backup.timer (06:00 and 18:00 daily, 5-day retention).
"""
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import duckdb

OSMIA_HOME   = Path(os.getenv('OSMIA_HOME', Path(__file__).resolve().parent))
DB_PATH      = Path(os.getenv('OSMIA_DB_PATH', OSMIA_HOME / 'market_data.duckdb'))
BACKUP_DIR   = OSMIA_HOME / 'backups'
RETENTION_DAYS = 5

BACKUP_DIR.mkdir(exist_ok=True)

ts          = datetime.now().strftime('%Y-%m-%d_%H-%M')
backup_path = BACKUP_DIR / f'market_data_{ts}.duckdb'

# Flush WAL to the main file before copying
try:
    con = duckdb.connect(str(DB_PATH))
    con.execute('CHECKPOINT')
    con.close()
except Exception as e:
    print(f'  Checkpoint failed (copying anyway): {e}', file=sys.stderr)

shutil.copy2(str(DB_PATH), str(backup_path))
size_mb = backup_path.stat().st_size / 1_000_000
print(f'[{datetime.now():%H:%M:%S}] Backup written: {backup_path.name}  ({size_mb:.1f} MB)')

# Prune backups older than RETENTION_DAYS
cutoff = datetime.now().timestamp() - RETENTION_DAYS * 86400
removed = []
for f in sorted(BACKUP_DIR.glob('market_data_*.duckdb')):
    if f.stat().st_mtime < cutoff:
        f.unlink()
        removed.append(f.name)

if removed:
    print(f'  Pruned {len(removed)} backup(s): {", ".join(removed)}')

remaining = sorted(BACKUP_DIR.glob('market_data_*.duckdb'))
print(f'  {len(remaining)} backup(s) on disk ({sum(f.stat().st_size for f in remaining)/1_000_000:.0f} MB total)')
