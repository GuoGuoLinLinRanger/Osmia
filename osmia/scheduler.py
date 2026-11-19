"""In-process scheduler: every Osmia job on its timer, in one process.

`python -m osmia --run` replaces the nine systemd timers for container and
cloud hosts (no systemd there). Jobs run one at a time, so DuckDB's
single-writer lock is never contended. A failing job is logged and retried on
its next tick; it never stops the loop.
"""
import time
import traceback
from datetime import datetime, timezone

# (name, CLI args, interval seconds, delay before first run seconds)
JOBS = [
    ('ingest',   ['--ingest'],                   300,    0),
    ('fg',       ['--fetch-fg'],                 300,   30),
    ('pairs',    ['--analyze'],                  600,   90),
    ('fiat',     ['--fiat'],                     600,  150),
    ('news',     ['--scrape-news'],             1800,   60),
    ('outcomes', ['--outcomes'],                3600,  300),
    ('insights', ['--report', '--write-insights'], 6 * 3600, 900),
    ('unlocks',  ['--scrape-unlocks'],         24 * 3600, 120),
]
BACKUP_HOURS_UTC = (6, 18)


def _run(name: str, argv: list[str]) -> None:
    from .cli import main
    started = time.monotonic()
    print(f"\n=== [{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {name} ===", flush=True)
    try:
        main(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"  [scheduler] {name} exited with {e.code}", flush=True)
    except Exception:
        print(f"  [scheduler] {name} failed:\n{traceback.format_exc()}", flush=True)
    print(f"  [scheduler] {name} took {time.monotonic() - started:.1f}s", flush=True)


def run_forever(jobs=JOBS) -> None:
    now = time.monotonic()
    due = {name: now + first for name, _, _, first in jobs}
    last_backup_slot = None
    print(f"[scheduler] {len(jobs)} jobs + backups at {BACKUP_HOURS_UTC} UTC. Ctrl-C to stop.",
          flush=True)
    try:
        while True:
            now = time.monotonic()
            for name, argv, every, _ in jobs:
                if now >= due[name]:
                    _run(name, argv)
                    due[name] = max(due[name] + every, time.monotonic() + 1)
            utc = datetime.now(timezone.utc)
            slot = (utc.date(), utc.hour)
            if utc.hour in BACKUP_HOURS_UTC and slot != last_backup_slot:
                _run('backup', ['--backup'])
                last_backup_slot = slot
            time.sleep(max(1.0, min(due.values()) - time.monotonic()) if due else 30)
    except KeyboardInterrupt:
        print("\n[scheduler] stopped.", flush=True)
