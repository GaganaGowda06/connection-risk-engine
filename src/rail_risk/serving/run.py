"""Run the replay worker and the API in one process (the container's command).

    python -m rail_risk.serving.run --split train_pool|oot_week --start 2026-10-17T00:00 --end ...

The worker resumes from the last committed hour in the state store, so a killed container comes
back where it was: the ranges' state and the pending outcomes are in the committed blob.
"""

import argparse
import os
import threading
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import uvicorn

from rail_risk.logging import get_logger
from rail_risk.serving import engine as E
from rail_risk.serving import state as S

log = get_logger("serving_run")
BIND = "0.0.0.0"  # noqa: S104  (inside the container; compose publishes on 127.0.0.1 only)


def replay(args: argparse.Namespace, stop: threading.Event) -> None:
    th = E.Thresholds.load(Path(args.thresholds))
    state_dir = Path(args.state)
    store = S.open_store("sqlite", state_dir)
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    d0, d1 = start.tz_convert("Europe/London").date(), end.tz_convert("Europe/London").date()
    ssds = [d0 + timedelta(days=i) for i in range(-1, (d1 - d0).days + 2)]
    skip = [pd.Timestamp(h, tz="UTC") for h in args.skip]
    eng = E.Engine(
        store, Path(args.source), Path(args.store_root), th, args.split, ssds=ssds, skip=skip
    )
    done = {pd.Timestamp(h) for h in store.hours_done()}
    if not done:
        eng.preload(start, ssds)
    h = start
    while h < end and not stop.is_set():
        if h not in done:
            eng.process_hour(h)
            if args.die_after_hours and len(store.hours_done()) == args.die_after_hours:
                log.warning("simulated crash", hours=len(store.hours_done()))
                os._exit(137)
        h += pd.Timedelta(hours=1)
    log.info("replay finished", hours=len(store.hours_done()), peak_mb=round(eng.peak_mb))
    store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_pool", choices=["train_pool", "oot_week"])
    ap.add_argument("--source", default="data/interim/pport")
    ap.add_argument("--store-root", default="data/interim/serve/store")
    ap.add_argument("--state", default="data/interim/serve/state")
    ap.add_argument("--thresholds", default="reports/serving/guard_threshold.json")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--skip", nargs="*", default=[], help="UTC hours that never arrive")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--die-after-hours", type=int, default=0, help="test: exit hard after N hours")
    args = ap.parse_args()
    os.environ["RAIL_SERVE_STATE"] = args.state
    os.environ["RAIL_GUARD_THRESHOLD"] = args.thresholds
    stop = threading.Event()
    if args.split == "oot_week":
        from rail_risk.ingest.pport_extract import reading_split

        ctx = reading_split("oot_week", "oot_replay")
    else:
        import contextlib

        ctx = contextlib.nullcontext()
    with ctx:
        t = threading.Thread(target=replay, args=(args, stop), daemon=True)
        t.start()
        if args.serve:
            uvicorn.run("rail_risk.serving.app:app", host=BIND, port=args.port, log_level="warning")
        t.join()


if __name__ == "__main__":
    main()


_ = date
