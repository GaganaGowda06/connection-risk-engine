# ruff: noqa: S608  (table names come from the fixed KINDS tuple, never from a request)
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


class SqliteStore(StateStore):
    name = "sqlite"

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self.con.execute("CREATE TABLE IF NOT EXISTS hours (h TEXT PRIMARY KEY)")
        self._have = {
            r[0] for r in self.con.execute("select name from sqlite_master where type='table'")
        }

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
        return self._q(f"select * from {kind}") if kind in self._have else pd.DataFrame()

    def risk(self, rid):
        return (
            self._q("select * from arrival where rid = ?", (rid,))
            if "arrival" in self._have
            else pd.DataFrame()
        )

    def pair(self, in_rid, out_rid):
        if "pair" not in self._have:
            return pd.DataFrame()
        return self._q("select * from pair where in_rid = ? and out_rid = ?", (in_rid, out_rid))

    def inbound(self, hub, start, end):
        if "pair" not in self._have:
            return pd.DataFrame()
        return self._q(
            "select * from pair where hub = ? and plan_arr_in >= ? and plan_arr_in < ?",
            (hub, start, end),
        )

    def close(self):
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
