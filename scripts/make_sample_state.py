#!/usr/bin/env python3
"""Cut a small sample ledger for the network service from a real replay ledger.

Keeps the derived model outputs (arrival ranges and P(miss) rows) scored in one replay hour, plus
the freshness summary the API reads. No raw Darwin or Train Describer message is copied. The sample
is written under data/sample/ and committed, so `docker compose up --build api` works from a fresh
clone; the compose service uses it only when the real ledger is missing.

    uv run python scripts/make_sample_state.py
    uv run python scripts/make_sample_state.py --hour "2026-10-17 18:00:00"
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from rail_risk.serving.state import SqliteStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="data/interim/serve/state_oot/state.sqlite")
    ap.add_argument("--out", default="data/sample/state_oot")
    ap.add_argument("--hour", default="2026-10-17 18:00:00", help="UTC hour of scored rows to keep")
    args = ap.parse_args()
    start = pd.Timestamp(args.hour, tz="UTC")
    end = start + pd.Timedelta(hours=1)
    lo, hi = start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
    src = sqlite3.connect(f"file:{Path(args.src).resolve()}?mode=ro", uri=True)
    frames = {
        k: pd.read_sql_query(
            f"select * from {k} where snap_time >= ? and snap_time < ?",  # noqa: S608
            src,
            params=(lo, hi),
        )
        for k in ("arrival", "pair")
    }
    src.close()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    store = SqliteStore(out / "state.sqlite")
    clock = str(end)
    last_minute = int(end.timestamp() // 60) - 1
    store.commit_hour(
        str(start), frames, {"clock": clock, "fresh": {"clock": clock, "last_minute": last_minute}}
    )
    store.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.con.execute("PRAGMA journal_mode=DELETE")
    store.con.execute("VACUUM")
    store.close()
    for suffix in ("-wal", "-shm"):
        (out / f"state.sqlite{suffix}").unlink(missing_ok=True)
    n = {k: len(v) for k, v in frames.items()}
    size = (out / "state.sqlite").stat().st_size
    print(f"wrote {out}/state.sqlite rows={n} clock={clock} bytes={size}")


if __name__ == "__main__":
    main()
