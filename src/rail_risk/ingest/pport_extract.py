"""Run the Push Port parser over the mirror, then derive stop_actuals per service date."""

import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from rail_risk.ingest.darwin_pport import (
    PARSER_VERSION,
    SCHEMAS,
    FileStats,
    Stage1Config,
    load_stage1_config,
    parse_file,
    split_for,
)
from rail_risk.logging import get_logger

KEY_COLS = ["rid", "tpl", "event", "wt_arr", "wt_dep", "wt_pass"]


def archive_files(root: Path, days: list[date]) -> list[Path]:
    """Expected hour files for the given UK dates, newest first (missing ones are still listed)."""
    return [
        root / d.isoformat() / f"{h:02d}.pport.gz"
        for d in sorted(days, reverse=True)
        for h in range(24)
    ]


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _parse_one(args: tuple[str, str, bool]) -> dict:
    path, out, force = args
    st = parse_file(Path(path), Path(out), force=force)
    return st.__dict__


def extract(
    files: list[Path], out: Path, workers: int = 4, force: bool = False, log=None
) -> list[FileStats]:
    """One process per file (a fresh worker for each), newest first as given."""
    log = log or get_logger("pport_extract")
    results: list[FileStats] = []
    jobs = [(str(p), str(out), force) for p in files]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        for r in pool.map(_parse_one, jobs, chunksize=1):
            st = FileStats(**r)
            results.append(st)
            log.info(
                "file_done",
                key=st.key,
                status=st.status,
                docs=st.documents,
                rows=sum(st.rows.values()),
                seconds=st.seconds,
                peak_mb=st.peak_rss_mb,
                errors=st.parse_errors,
            )
    return results


def stop_actuals_for_day(out: Path, ssd: date, cfg: Stage1Config | None = None) -> int:
    """Final actual per rid and stop for one service date, from that date's stop_updates.

    The last message carrying an actual (or an actual removal) per stop and event wins; a final
    removal means no actual. Sources are every hour file that contributed to the date.
    """
    cfg = cfg or load_stage1_config()
    src = out / "stop_updates" / f"service_date={ssd.isoformat()}"
    parts = sorted(src.glob("*.parquet"))
    if not parts:
        return 0
    cols = KEY_COLS + [
        "actual",
        "actual_class",
        "src",
        "actual_removed",
        "pt_arr",
        "pt_dep",
        "message_time",
    ]
    table = ds.dataset([str(p) for p in parts], format="parquet").to_table(columns=cols)
    table = table.filter(pc.is_valid(table["event"]))
    n_updates = (
        table.group_by(KEY_COLS)
        .aggregate([("message_time", "count")])
        .rename_columns(KEY_COLS + ["n_updates"])
        .to_pandas()
    )
    keep = pc.or_(pc.is_valid(table["actual"]), pc.fill_null(table["actual_removed"], False))
    got = table.filter(keep).to_pandas().sort_values("message_time")
    del table
    first = (
        got[got["actual"].notna()]
        .drop_duplicates(KEY_COLS, keep="first")[KEY_COLS + ["message_time"]]
        .rename(columns={"message_time": "first_actual_message_time"})
    )
    final = got.drop_duplicates(KEY_COLS, keep="last")
    final = final[final["actual_removed"] != True]  # noqa: E712
    final = final.rename(columns={"message_time": "last_message_time"})
    final = final.merge(n_updates, on=KEY_COLS, how="left").merge(first, on=KEY_COLS, how="left")
    final["ssd"] = ssd
    final["split"] = split_for(ssd, cfg)
    schema = SCHEMAS["stop_actuals"]
    sha = _source_hashes(out, parts)
    table = pa.Table.from_pandas(
        final[[f.name for f in schema]], schema=schema, preserve_index=False
    )
    table = table.replace_schema_metadata(
        {b"source_sha256": json.dumps(sha).encode(), b"parser_version": PARSER_VERSION.encode()}
    )
    dest = out / "stop_actuals" / f"service_date={ssd.isoformat()}" / "part.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(dest) + ".tmp", compression="zstd")
    Path(str(dest) + ".tmp").rename(dest)
    return len(final)


def _source_hashes(out: Path, parts: list[Path]) -> list[str]:
    hashes = set()
    for p in parts:
        stats = out / "_stats" / f"{p.stem}.json"
        if stats.exists():
            hashes.add(json.loads(stats.read_text())["source_sha256"])
    return sorted(h for h in hashes if h)


def build_stop_actuals(out: Path, days: list[date], log=None) -> dict[str, int]:
    log = log or get_logger("pport_extract")
    cfg = load_stage1_config()
    counts: dict[str, int] = {}
    for d in sorted(days, reverse=True):
        started = time.monotonic()
        n = stop_actuals_for_day(out, d, cfg)
        counts[d.isoformat()] = n
        log.info(
            "stop_actuals_done",
            ssd=d.isoformat(),
            rows=n,
            seconds=round(time.monotonic() - started, 1),
        )
    return counts


def split_counts(out: Path) -> dict[str, dict[str, int]]:
    """Rows per split per table, and a hard check that no oot_week or winter_test row is a
    train_pool row and that every row's split matches its service date."""
    cfg = load_stage1_config()
    result: dict[str, dict[str, int]] = {}
    for table in SCHEMAS:
        root = out / table
        if not root.exists():
            continue
        counts: Counter = Counter()
        for d in sorted(root.glob("service_date=*")):
            ssd = date.fromisoformat(d.name.split("=")[1])
            want = split_for(ssd, cfg)
            for part in d.glob("*.parquet"):
                col = pq.read_table(part, columns=["split"]).column("split")
                for label, n in Counter(col.to_pylist()).items():
                    counts[label] += n
                    if label != want:
                        raise ValueError(f"{part}: split {label!r} but service date says {want!r}")
        result[table] = dict(counts)
    return result


def partial_days(stats_dir: Path) -> list[str]:
    """Hour files that did not parse cleanly (missing or truncated)."""
    out = []
    for f in sorted(stats_dir.glob("*.json")):
        st = json.loads(f.read_text())
        if st["status"] != "ok":
            out.append(f"{st['key']}:{st['status']}")
    return out


_ACTIVE_SPLIT = "train_pool"
GATE_THRESHOLD = Path("reports/serving/guard_threshold.json")
GATE_DONE = Path("reports/serving/oot_replay_scored.json")


def replay_gate() -> None:
    """oot_week can be read only by the Stage 6 replay, once: the freshness threshold must already
    be fixed on validation days (its file exists) and the replay must not have been scored."""
    if not GATE_THRESHOLD.exists():
        raise PermissionError(f"oot_week stays closed until {GATE_THRESHOLD} exists")
    if GATE_DONE.exists():
        raise PermissionError(f"oot_week was already replayed and scored: {GATE_DONE}")


@contextmanager
def reading_split(split: str, purpose: str):
    """Let train_pool_table read one other split inside the block. Only the February stress report
    may read winter_test, once; oot_week can never be read through this function."""
    global _ACTIVE_SPLIT
    if (split, purpose) == ("oot_week", "oot_replay"):
        replay_gate()
    elif split != "winter_test" or purpose != "february_stress":
        raise ValueError(f"reading split {split!r} for {purpose!r} is not allowed")
    previous, _ACTIVE_SPLIT = _ACTIVE_SPLIT, split
    try:
        yield
    finally:
        _ACTIVE_SPLIT = previous


def train_pool_table(
    out: Path,
    table: str,
    columns: list[str] | None = None,
    service_dates: list[date] | None = None,
    where: "ds.Expression | None" = None,
) -> pa.Table:
    """The only way a model stage reads a table: train_pool rows, never oot_week or winter_test.

    `service_dates` limits the read to those partitions (a day at a time keeps memory small);
    `where` adds a row filter on top of the split filter."""
    if service_dates is None:
        parts = sorted((out / table).glob("service_date=*/*.parquet"))
    else:
        parts = sorted(
            p for d in service_dates for p in (out / table / f"service_date={d}").glob("*.parquet")
        )
    if not parts:
        return pa.table({})
    dataset = ds.dataset([str(p) for p in parts], format="parquet")
    keep = ds.field("split") == _ACTIVE_SPLIT
    return dataset.to_table(columns=columns, filter=keep if where is None else keep & where)


def train_pool_view(
    con,
    out: Path,
    view: str,
    table: str,
    columns: list[str],
    service_dates: list[date],
) -> bool:
    """Same guard as train_pool_table, for DuckDB: a lazy view over the table's partitions for
    `service_dates`, limited to the active split (train_pool unless reading_split opened another)
    and to `columns`. DuckDB pushes column and row filters of later queries into the Parquet
    scan, so nothing is loaded whole. Returns False when there is no partition."""
    parts = sorted(
        p for d in service_dates for p in (out / table / f"service_date={d}").glob("*.parquet")
    )
    if not parts:
        return False
    files = ", ".join("'" + str(p).replace("'", "''") + "'" for p in parts)
    cols = ", ".join(columns)
    sql = (
        f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT {cols} FROM read_parquet([{files}]) "  # noqa: S608
        f"WHERE split = '{_ACTIVE_SPLIT}'"
    )
    con.execute(sql)
    return True
