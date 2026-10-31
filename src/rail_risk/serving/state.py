# ruff: noqa: S608, E501  (table names come from the fixed KINDS tuple, never from a request)
"""Three state-store designs for the service, one interface.

The messages themselves are always the Parquet files in the store root (the existing feature code
reads them there). The state is what must survive a restart and answer requests: the ledger of
predictions (arrival and pair), the adaptive-range state, the clock, and the list of hours already
processed. Every backend writes one hour atomically (ledger rows and the state blob together).

  memory   dict of DataFrames in the process, pickled to a checkpoint file every `every` hours
  duckdb   one DuckDB file, an hour per transaction
  sqlite   one SQLite file (WAL), an hour per transaction
"""

import json
import pickle
import sqlite3
import time
from pathlib import Path

import duckdb
import pandas as pd

KINDS = ("arrival", "pair")


class StateStore:
    name = "base"

    def commit_hour(self, hour: str, frames: dict, blob: dict) -> None: ...

    def load_blob(self) -> dict | None: ...

    def hours_done(self) -> list[str]: ...

    def risk(self, rid: str) -> pd.DataFrame: ...

    def pair(self, in_rid: str, out_rid: str) -> pd.DataFrame: ...

    def inbound(self, hub: str, start: str, end: str) -> pd.DataFrame: ...

    def all(self, kind: str) -> pd.DataFrame: ...

    def close(self) -> None: ...


class MemoryStore(StateStore):
    name = "memory"

    def __init__(self, path: Path, every: int = 6):
        self.path, self.every = path, every
        self.frames: dict[str, list] = {k: [] for k in KINDS}
        self.blob: dict | None = None
        self.done: list[str] = []
        self._n = 0
        if path.exists():
            d = pickle.loads(path.read_bytes())  # noqa: S301
            self.frames, self.blob, self.done = d["frames"], d["blob"], d["done"]

    def commit_hour(self, hour, frames, blob):
        for k, f in frames.items():
            if len(f):
                self.frames[k].append(f)
        self.blob, self._n = blob, self._n + 1
        self.done.append(hour)
        if self._n % self.every == 0:
            self.checkpoint()

    def checkpoint(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pickle.dumps({"frames": self.frames, "blob": self.blob, "done": self.done}))
        tmp.replace(self.path)

    def load_blob(self):
        return self.blob

    def hours_done(self):
        return list(self.done)

    def all(self, kind):
        fs = self.frames[kind]
        return pd.concat(fs, ignore_index=True) if fs else pd.DataFrame()

    def risk(self, rid):
        a = self.all("arrival")
        return a[a["rid"] == rid] if len(a) else a

    def pair(self, in_rid, out_rid):
        a = self.all("pair")
        return a[(a["in_rid"] == in_rid) & (a["out_rid"] == out_rid)] if len(a) else a

    def inbound(self, hub, start, end):
        a = self.all("pair")
        if a.empty:
            return a
        m = (a["hub"] == hub) & (a["plan_arr_in"] >= start) & (a["plan_arr_in"] < end)
        return a[m]

    def close(self):
        self.checkpoint()


class DuckStore(StateStore):
    name = "duckdb"

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute("SET memory_limit='2GB'")
        self.con.execute("SET threads=2")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (k VARCHAR PRIMARY KEY, v VARCHAR)")
        self.con.execute("CREATE TABLE IF NOT EXISTS hours (h VARCHAR PRIMARY KEY)")
        self._have = {
            r[0]
            for r in self.con.execute("select table_name from information_schema.tables").fetchall()
        }

    def commit_hour(self, hour, frames, blob):
        c = self.con
        c.execute("BEGIN")
        try:
            for k, f in frames.items():
                if not len(f):
                    continue
                c.register("f", f)
                if k not in self._have:
                    c.execute(f"CREATE TABLE {k} AS SELECT * FROM f")
                    self._have.add(k)
                else:
                    c.execute(f"INSERT INTO {k} SELECT * FROM f")
                c.unregister("f")
            c.execute("DELETE FROM meta WHERE k='blob'")
            c.execute("INSERT INTO meta VALUES ('blob', ?)", [json.dumps(blob)])
            c.execute("INSERT INTO hours VALUES (?)", [hour])
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def load_blob(self):
        r = self.con.execute("select v from meta where k='blob'").fetchone()
        return json.loads(r[0]) if r else None

    def hours_done(self):
        return [r[0] for r in self.con.execute("select h from hours order by h").fetchall()]

    def _q(self, sql, params=()):
        return self.con.execute(sql, list(params)).df()

    def all(self, kind):
        return self._q(f"select * from {kind}") if kind in self._have else pd.DataFrame()

    def risk(self, rid):
        return (
            self._q("select * from arrival where rid = ?", [rid])
            if "arrival" in self._have
            else pd.DataFrame()
        )

    def pair(self, in_rid, out_rid):
        if "pair" not in self._have:
            return pd.DataFrame()
        return self._q("select * from pair where in_rid = ? and out_rid = ?", [in_rid, out_rid])

    def inbound(self, hub, start, end):
        if "pair" not in self._have:
            return pd.DataFrame()
        return self._q(
            "select * from pair where hub = ? and plan_arr_in >= ? and plan_arr_in < ?",
            [hub, start, end],
        )

    def close(self):
        self.con.close()


_SERVICE_CACHE: dict = {}


class SqliteStore(StateStore):
    name = "sqlite"

    def __init__(self, path: Path, readonly: bool = False, keep_open: bool = False):
        """`readonly` opens an existing ledger for the API (no write, no pragma, so a worker never
        takes the writer's lock); `keep_open` makes close() a no-op (one connection per thread)."""
        self.keep_open, self.readonly, self.path = keep_open, readonly, path
        if readonly:
            self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
            self.con.execute("PRAGMA query_only=1")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(path), isolation_level=None)
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA synchronous=NORMAL")
            self.con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            self.con.execute("CREATE TABLE IF NOT EXISTS hours (h TEXT PRIMARY KEY)")
        self._have = self._tables()

    def _tables(self) -> set:
        return {r[0] for r in self.con.execute("select name from sqlite_master where type='table'")}

    def _has(self, table: str) -> bool:
        if table not in self._have and self.readonly:  # the replay may have created it since
            self._have = self._tables()
        return table in self._have

    INDEXES = {
        "arrival": ["rid", "snap_time"],
        "pair": ["in_rid, out_rid", "hub, plan_arr_in"],
    }

    def _index(self) -> None:
        """Lookups by service, by pair and by hub and planned arrival must not scan the ledger."""
        for table, specs in self.INDEXES.items():
            if table in self._have:
                for cols in specs:
                    name = f"ix_{table}_{cols.replace(', ', '_')}"
                    self.con.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols})")

    def set_meta(self, k: str, v: dict) -> None:
        """Record a small JSON fact about the ledger (writer only)."""
        self.con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, json.dumps(v)))

    def load_fresh(self) -> dict | None:
        r = self.con.execute("select v from meta where k='fresh'").fetchone()
        return json.loads(r[0]) if r else None

    @staticmethod
    def _typed(f: pd.DataFrame) -> pd.DataFrame:
        """Timestamps as text (SQLite has no time type), every other column keeps its dtype so the
        table is created with INTEGER and REAL columns."""
        g = f.copy()
        for col in g.columns:
            if str(g[col].dtype).startswith("datetime64"):
                g[col] = g[col].astype("datetime64[us, UTC]").dt.strftime("%Y-%m-%d %H:%M:%S")
        return g

    @classmethod
    def _plain(cls, f: pd.DataFrame) -> pd.DataFrame:
        g = cls._typed(f)
        return g.astype(object).where(pd.notna(g), None)

    def commit_hour(self, hour, frames, blob):
        c = self.con
        prepared = {k: self._plain(f) for k, f in frames.items() if len(f)}
        for k in prepared:
            if k not in self._have:  # the table is created first, outside the hour's transaction
                self._typed(frames[k]).head(0).to_sql(k, c, if_exists="append", index=False)
                self._have.add(k)
        c.execute("BEGIN")
        try:
            for k, g in prepared.items():
                marks = ",".join("?" * len(g.columns))
                c.executemany(
                    f"INSERT INTO {k} VALUES ({marks})", g.itertuples(index=False, name=None)
                )
            c.execute("INSERT OR REPLACE INTO meta VALUES ('blob', ?)", (json.dumps(blob),))
            if "fresh" in blob:
                c.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('fresh', ?)", (json.dumps(blob["fresh"]),)
                )
            c.execute("INSERT INTO hours VALUES (?)", (hour,))
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        self._index()

    def load_blob(self):
        r = self.con.execute("select v from meta where k='blob'").fetchone()
        return json.loads(r[0]) if r else None

    def hours_done(self):
        return [r[0] for r in self.con.execute("select h from hours order by h")]

    def hours_count(self) -> int:
        return int(self.con.execute("select count(*) from hours").fetchone()[0])

    def _q(self, sql, params=()):
        return pd.read_sql_query(sql, self.con, params=params)

    def all(self, kind):
        return self._q(f"select * from {kind}") if self._has(kind) else pd.DataFrame()

    def risk(self, rid):
        return (
            self._q("select * from arrival where rid = ?", (rid,))
            if self._has("arrival")
            else pd.DataFrame()
        )

    def pair(self, in_rid, out_rid):
        if not self._has("pair"):
            return pd.DataFrame()
        return self._q("select * from pair where in_rid = ? and out_rid = ?", (in_rid, out_rid))

    def inbound(self, hub, start, end):
        if not self._has("pair"):
            return pd.DataFrame()
        return self._q(
            "select * from pair where hub = ? and plan_arr_in >= ? and plan_arr_in < ?",
            (hub, start, end),
        )

    def inbound_summary(self, hub, start, end, limit) -> pd.DataFrame:
        """Per inbound service in the window: planned arrival, number of scored rows, and the latest
        row's probability and stale flag. Grouped in SQL, so a request touches `limit` services
        instead of loading every scored pair of the hub."""
        if not self._has("pair"):
            return pd.DataFrame()
        head = self._q(
            "select in_rid, plan_arr_in, count(*) as n, max(snap_time) as last_t from pair "
            "where hub = ? and plan_arr_in >= ? and plan_arr_in < ? "
            "group by in_rid, plan_arr_in order by plan_arr_in limit ?",
            (hub, start, end, limit),
        )
        if head.empty:
            return head
        marks = ",".join("?" * len(head))
        last = self._q(
            f"select in_rid, snap_time, p_final, stale from pair where hub = ? "  # noqa: S608
            f"and in_rid in ({marks}) and plan_arr_in >= ? and plan_arr_in < ?",
            (hub, *head["in_rid"], start, end),
        )
        last = last.sort_values("snap_time").groupby("in_rid").tail(1)
        return head.merge(last[["in_rid", "p_final", "stale"]], on="in_rid", how="left")

    # ---- services: who runs which train, derived from rows already in the ledger -------------
    def service_table(self, key: str) -> pd.DataFrame:
        """One row per (service, origin, destination) the ledger knows, cached per ledger clock.
        Times and operator come only from ledger columns (arrival.pt_arr, pair.plan_dep_out,
        pair.plan_arr_in, pair.in_toc, pair.out_toc); a value the ledger never saw is None."""
        ck = (str(self.path), key)
        if ck not in _SERVICE_CACHE:
            _SERVICE_CACHE.clear()
            _SERVICE_CACHE[ck] = self._derive_services()
        return _SERVICE_CACHE[ck]

    def _derive_services(self) -> pd.DataFrame:
        cols = ["rid", "toc", "origin", "dest", "sched_dep", "sched_arr", "has_risk"]
        has_a, has_p = self._has("arrival"), self._has("pair")
        arr = (
            self._q("select rid, route, max(pt_arr) as sched_arr from arrival group by rid, route")
            if has_a
            else pd.DataFrame(columns=["rid", "route", "sched_arr"])
        )
        dep = (
            self._q(
                "select out_rid as rid, hub, min(plan_dep_out) as sched_dep, max(out_toc) as toc from pair group by out_rid, hub"
            )
            if has_p
            else pd.DataFrame(columns=["rid", "hub", "sched_dep", "toc"])
        )
        inn = (
            self._q(
                "select in_rid as rid, hub, min(plan_arr_in) as sched_arr, max(in_toc) as toc from pair group by in_rid, hub"
            )
            if has_p
            else pd.DataFrame(columns=["rid", "hub", "sched_arr", "toc"])
        )
        toc = (
            pd.concat([dep[["rid", "toc"]], inn[["rid", "toc"]]])
            .dropna()
            .drop_duplicates("rid")
            .set_index("rid")["toc"]
            .to_dict()
        )
        risk_rids = set(arr["rid"])
        rows = []
        dep_at = {(r.rid, r.hub): r.sched_dep for r in dep.itertuples()}
        seen_dep, seen_arr = set(), set()
        for r in arr.itertuples():
            o, d = r.route.split("-")[0].upper(), r.route.split("-")[1].upper()
            rows.append((r.rid, toc.get(r.rid), o, d, dep_at.get((r.rid, o)), r.sched_arr, True))
            seen_dep.add((r.rid, o))
            seen_arr.add((r.rid, d))
        for (
            r
        ) in dep.itertuples():  # departures from a hub whose destination the ledger does not know
            if (r.rid, r.hub) not in seen_dep:
                rows.append((r.rid, r.toc, r.hub, None, r.sched_dep, None, r.rid in risk_rids))
        for r in inn.itertuples():  # arrivals at a hub whose origin the ledger does not know
            if (r.rid, r.hub) not in seen_arr:
                rows.append((r.rid, r.toc, None, r.hub, None, r.sched_arr, r.rid in risk_rids))
        return pd.DataFrame(rows, columns=cols)

    def service_toc(self, rid: str, key: str) -> str | None:
        t = self.service_table(key)
        hit = t.loc[t["rid"] == rid, "toc"].dropna()
        return str(hit.iloc[0]) if len(hit) else None

    def services(self, key: str, origin: str | None, dest: str | None, near: str, window_min: int,
                 by: str, limit: int) -> pd.DataFrame:  # fmt: skip
        t = self.service_table(key)
        col = "sched_dep" if by == "dep" else "sched_arr"
        if origin:
            t = t[t["origin"] == origin]
        if dest:
            t = t[t["dest"] == dest]
        t = t[t[col].notna()].copy()
        if t.empty:
            return t
        gap = (pd.to_datetime(t[col]) - pd.Timestamp(near)).abs() / pd.Timedelta(minutes=1)
        t = t.assign(_gap=gap)
        t = t[t["_gap"] <= window_min].sort_values(["_gap", "rid"]).head(limit)
        return t.drop(columns="_gap")

    def close(self):
        if not self.keep_open:
            self.con.close()


def open_store(kind: str, directory: Path) -> StateStore:
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "memory":
        return MemoryStore(directory / "checkpoint.pkl")
    if kind == "duckdb":
        return DuckStore(directory / "state.duckdb")
    if kind == "sqlite":
        return SqliteStore(directory / "state.sqlite")
    raise ValueError(f"unknown state store {kind!r}")


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t
