"""The network time split: complete days of train_pool only, in time order with gaps.

A service date d is complete when the archive has every hour file that exists for local days d-1
and d, and the early hours of d+1 (trains run past midnight, schedules are sent the day before).
An hour that never existed (the spring clock change) is not expected. oot_week and winter_test
days are never candidates, and neither are partial or gappy days.
"""

import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from rail_risk.evaluation import Split, SplitConfig, load_split_config, make_split
from rail_risk.ingest.darwin_pport import Stage1Config, load_stage1_config, split_for

LONDON = ZoneInfo("Europe/London")
NEXT_DAY_HOURS = 5  # local hours 00 to 04 of d+1


def expected_labels(day: date) -> set[int]:
    """Local hour labels that occur on a UK local day (23 on the spring change, 24 otherwise)."""
    start = datetime(day.year, day.month, day.day, tzinfo=LONDON).astimezone(UTC)
    labels = set()
    for i in range(26):
        t = (start + timedelta(hours=i)).astimezone(LONDON)
        if t.date() == day:
            labels.add(t.hour)
    return labels


def manifest_files(path: Path) -> set[tuple[date, int]]:
    out = set()
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["gzip_ok"] != "True":
                continue
            d, name = row["path"].split("/")
            out.add((date.fromisoformat(d), int(name[:2])))
    return out


def is_complete(day: date, files: set[tuple[date, int]]) -> bool:
    need = {(day - timedelta(days=1), h) for h in expected_labels(day - timedelta(days=1))}
    need |= {(day, h) for h in expected_labels(day)}
    need |= {(day + timedelta(days=1), h) for h in range(NEXT_DAY_HOURS)}
    return need <= files


def candidate_days(cfg: Stage1Config) -> list[date]:
    days = set(cfg.boundary_days) | set(cfg.clock_change_days)
    for start, end in cfg.windows:
        days |= {start + timedelta(days=i) for i in range((end - start).days + 1)}
    return sorted(days)


def complete_train_pool_days(
    manifest: Path = Path("data/manifests/darwin_archive.csv"),
    stage1: Path = Path("config/stage1.yaml"),
) -> list[date]:
    cfg = load_stage1_config(stage1)
    files = manifest_files(manifest)
    return [
        d
        for d in candidate_days(cfg)
        if split_for(d, cfg) == cfg.default_split and is_complete(d, files)
    ]


def network_split(
    split_file: Path = Path("config/split.yaml"),
    manifest: Path = Path("data/manifests/darwin_archive.csv"),
    stage1: Path = Path("config/stage1.yaml"),
) -> tuple[Split, SplitConfig]:
    cfg = load_split_config(split_file)
    return make_split(complete_train_pool_days(manifest, stage1), cfg), cfg
