# ruff: noqa: E501
"""The serving engine: ingest an hour of messages, then answer what became due in that hour.

Messages live as Parquet files in a store root (hard links of the parsed archive hours, one UTC
hour at a time, in message-time order). After an hour is ingested the engine computes, with the
frozen models and the same feature code as the batch pipeline, every arrival snapshot (15 and 60
minutes before arrival) and every pair snapshot (5, 15, 30, 60 minutes before the outbound's
departure) whose T falls in that hour, applies the adaptive ranges and the freshness guard, and
commits the hour atomically to the state store. A request is answered from that ledger.

Adaptive ranges: one alpha per (route, horizon), starting from the global CQR, gamma 0.05. A row
scored at T uses the outcomes of earlier rows whose arrival was first reported in a message at or
before T. Rows scored under a stale guard do not update the ranges.
"""

import gc
import heapq
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from rail_risk.baselines import models
from rail_risk.clean.run import route_specs
from rail_risk.config import load_routes
from rail_risk.connections import build as CB
from rail_risk.connections import data as DD
from rail_risk.ingest.pport_extract import replay_gate
from rail_risk.logging import get_logger
from rail_risk.network_model import adaptive as A
from rail_risk.network_model import calibrate as C
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.serving import artifacts as AR
from rail_risk.serving.arrival import due_arrival
from rail_risk.serving.state import StateStore

TABLES = ("schedules", "schedule_points", "stop_updates", "associations", "deactivations")
HOUR = pd.Timedelta(hours=1)
TARGET_MISS = 0.2
MAX_ACTUAL = pd.Timedelta(hours=12)
log = get_logger("serving")


def label_to_utc(name: str) -> pd.Timestamp:
    """'2026-10-12_01.parquet' is the local hour 01 of 2026-10-12 (the archive labels local hours)."""
    d, h = name.removesuffix(".parquet").split("_")
    t = pd.Timestamp(f"{d} {h}:00").tz_localize("Europe/London", ambiguous=True)
    return t.tz_convert("UTC")


@dataclass
class Thresholds:
    feed_min: float = 1e9
    hub_min: float = 1e9
    widen_k: float = 0.0  # arrival fallback width = 1 + widen_k * age_hours
    recent_min: int = 120  # a silence longer than the threshold stays a stale signal this long

    @staticmethod
    def load(path: Path) -> "Thresholds":
        if not path.exists():
            return Thresholds()
        d = json.loads(path.read_text())
        return Thresholds(d["feed_min"], d["hub_min"], d["widen_k"], int(d["recent_min"]))


@dataclass
class AciState:
    alpha: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)  # key -> row dict awaiting its outcome
    seen: set = field(default_factory=set)

    def to_json(self) -> dict:
        return {"alpha": self.alpha, "pending": self.pending}

    @staticmethod
    def from_json(d: dict) -> "AciState":
        return AciState(dict(d["alpha"]), dict(d["pending"]))


class Engine:
    def __init__(
        self,
        store: StateStore,
        source_root: Path,
        store_root: Path,
        thresholds: Thresholds | None = None,
        source_split: str = "train_pool",
        memory_limit_gb: float = 4.0,
        ssds: list[date] | None = None,
        skip: list[pd.Timestamp] | None = None,
    ):
        self.store, self.src, self.root = store, source_root, store_root
        self.th = thresholds or Thresholds()
        self.ssds = {f"service_date={d}" for d in ssds} if ssds else None
        self.source_split = source_split
        if source_split == "oot_week":
            replay_gate()
        self.mem_limit_mb = memory_limit_gb * 1000
        self.art = AR.load()
        self.cal = C.Calibrated.load(AR.CAL_DIR / AR.ARRIVAL_RUN)
        self.conn = __import__("joblib").load(AR.MODELS5 / f"{AR.PMISS_RUN}.joblib")
        self.fastest = pd.read_parquet(D.FASTEST_COLD)
        self.holidays = F.bank_holidays()
        routes = load_routes("config/routes.yaml")
        table = [r for r in routes.routes if not r.legacy]
        self.tc = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")
        self.specs = route_specs(type(routes)(routes=table), self.tc)
        self.cfg = CB.load_config()
        self.mct = CB.load_mct_table()
        self.hub_tpl = CB.hub_tiplocs(self.cfg["hubs"], self.tc)
        self.aci = AciState()
        self.minutes: np.ndarray = np.array([], dtype="int64")  # epoch minutes with any message
        self.hub_minutes: dict[str, np.ndarray] = {
            h: np.array([], dtype="int64") for h in self.cfg["hubs"]
        }
        self.clock: pd.Timestamp | None = None
        self.peak_mb = 0.0
        # hours whose files never arrive (a missing archive hour); set before any re-link
        self.skip: set = set(skip or ())
        self._index: dict | None = None
        blob = store.load_blob()
        if blob:
            self.aci = AciState.from_json(blob["aci"])
            self.minutes = np.array(blob["minutes"], dtype="int64")
            self.hub_minutes = {
                h: np.array(v, dtype="int64") for h, v in blob["hub_minutes"].items()
            }
            self.clock = pd.Timestamp(blob["clock"])
            self._relink(store.hours_done())

    # ------------------------------------------------------------------ ingest
    def _file_index(self) -> dict:
        if self._index is None:
            idx: dict = {}
            for t in TABLES:
                for f in sorted((self.src / t).glob("service_date=*/*_??.parquet")):
                    if self.ssds is not None and f.parent.name not in self.ssds:
                        continue
                    idx.setdefault(label_to_utc(f.name), []).append((t, f))
            self._index = idx
        return self._index

    def _link(self, hour: pd.Timestamp) -> list[Path]:
        out = []
        for t, f in self._file_index().get(hour, []):
            dst = self.root / t / f.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:  # another filesystem (a container mount): copy instead
                    shutil.copy2(f, dst)
            out.append(dst)
        return out

    def _relink(self, done: list[str]) -> None:
        for h in done:
            if pd.Timestamp(h) not in self.skip:
                self._link(pd.Timestamp(h))

    def ingest(self, hour: pd.Timestamp) -> int:
        if hour in self.skip:
            return 0
        files = self._link(hour)
        su = [str(f) for f in files if f.parent.parent.name == "stop_updates"]
        if su:
            con = duckdb.connect()
            con.execute("SET TimeZone='UTC'")
            lst = ", ".join(f"'{x}'" for x in su)
            m = con.execute(
                f"SELECT DISTINCT (epoch(date_trunc('minute', message_time))::BIGINT // 60) "  # noqa: S608
                f"FROM read_parquet([{lst}])"
            ).fetchall()
            self.minutes = np.union1d(self.minutes, np.array([r[0] for r in m], dtype="int64"))
            con.register("ht", self.hub_tpl)
            hm = con.execute(
                f"SELECT ht.hub, (epoch(date_trunc('minute', message_time))::BIGINT // 60) "  # noqa: S608
                f"FROM read_parquet([{lst}]) u JOIN ht ON u.tpl = ht.tpl GROUP BY ALL"
            ).fetchall()
            for hub in self.hub_minutes:
                add = np.array([r[1] for r in hm if r[0] == hub], dtype="int64")
                self.hub_minutes[hub] = np.union1d(self.hub_minutes[hub], add)
            con.close()
        return len(files)

    def preload(self, before: pd.Timestamp, ssds: list[date]) -> int:
        """Everything received before the replay starts (the timetable of the first day)."""
        n = 0
        keep = {f"service_date={d}" for d in ssds}
        for hr in sorted(self._file_index()):
            if hr < before and any(f.parent.name in keep for _, f in self._file_index()[hr]):
                n += self.ingest(hr)
        return n

    # ------------------------------------------------------------------ freshness
    @staticmethod
    def _age(minutes: np.ndarray, t: pd.Series) -> np.ndarray:
        tm = (t.astype("datetime64[us, UTC]").astype("int64") // 60_000_000).to_numpy()
        i = np.searchsorted(minutes, tm, side="right") - 1
        age = np.where(i >= 0, tm - minutes[np.clip(i, 0, None)], 1e6)
        return age.astype(float)

    @staticmethod
    def _recent_gap(minutes: np.ndarray, t: pd.Series, window: int) -> np.ndarray:
        """Longest silence (minutes) between consecutive message minutes inside [T - window, T],
        counting the silence still running at T."""
        tm = (t.astype("datetime64[us, UTC]").astype("int64") // 60_000_000).to_numpy()
        out = np.zeros(len(tm))
        for v in np.unique(tm):
            lo = np.searchsorted(minutes, v - window, side="left")
            hi = np.searchsorted(minutes, v, side="right")
            seg = minutes[max(lo - 1, 0) : hi]
            if len(seg) == 0:
                g = float(window)
            else:
                g = float(max(np.diff(seg).max() if len(seg) > 1 else 0, v - seg[-1]))
            out[tm == v] = g
        return out

    def _service_age(self, ssd: date, rids: pd.Series, t: pd.Series) -> np.ndarray:
        """Minutes since the last message about each service at its T (large when none yet)."""
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        files = sorted((self.root / "stop_updates").glob(f"service_date={ssd}/*.parquet"))
        if not files:
            return np.full(len(rids), 1e6)
        lst = ", ".join(f"'{f}'" for f in files)
        q = pd.DataFrame({"rid": rids.to_numpy(), "t": t.to_numpy(), "i": np.arange(len(rids))})
        con.register("q", q)
        r = con.execute(
            f"SELECT q.i, max(u.message_time) AS m FROM q LEFT JOIN "  # noqa: S608
            f"(SELECT rid, message_time FROM read_parquet([{lst}])) u "
            f"ON u.rid = q.rid AND u.message_time <= q.t GROUP BY q.i"
        ).df()
        con.close()
        age = np.full(len(rids), 1e6)
        m = r["m"].notna()
        age[r.loc[m, "i"].to_numpy()] = (
            (t.to_numpy()[r.loc[m, "i"].to_numpy()] - r.loc[m, "m"].to_numpy())
            / np.timedelta64(1, "m")
        ).astype(float)
        return age

    def _anything_due(self, ssd: date, hour: pd.Timestamp, hmin: int, hmax: int) -> bool:
        """Cheap test before a day's tables are loaded: does this service date have a planned
        arrival or departure that could make a snapshot due in the hour?"""
        files = sorted((self.root / "schedule_points").glob(f"service_date={ssd}/*.parquet"))
        if not files:
            return False
        lo, hi = hour + pd.Timedelta(minutes=hmin), hour + HOUR + pd.Timedelta(minutes=hmax)
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        lst = ", ".join(f"'{f}'" for f in files)
        n = con.execute(
            f"SELECT count(*) FROM read_parquet([{lst}]) "  # noqa: S608
            f"WHERE (pt_arr >= TIMESTAMPTZ '{lo}' AND pt_arr < TIMESTAMPTZ '{hi}') "
            f"OR (pt_dep >= TIMESTAMPTZ '{lo}' AND pt_dep < TIMESTAMPTZ '{hi}')"
        ).fetchone()[0]
        con.close()
        return n > 0

    # ------------------------------------------------------------------ arrival
    def _resolve_outcomes(self, ssds: list[date]) -> list[dict]:
        """First report of each pending row's arrival at its destination, from the store."""
        if not self.aci.pending:
            return []
        pend = pd.DataFrame(self.aci.pending.values())
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        files = [
            str(f)
            for d in ssds
            for f in (self.root / "stop_updates").glob(f"service_date={d}/*.parquet")
        ]
        if not files:
            return []
        lst = ", ".join(f"'{x}'" for x in files)
        con.register("p", pend[["key", "rid", "dest_tpl", "pt_arr_s"]])
        r = con.execute(
            f"SELECT p.key, min(u.message_time) AS obs, arg_min(u.actual, u.message_time) AS act "  # noqa: S608
            f"FROM p JOIN (SELECT rid, tpl, message_time, actual FROM read_parquet([{lst}]) "
            f"WHERE event = 'arr' AND actual IS NOT NULL) u ON u.rid = p.rid AND u.tpl = p.dest_tpl "
            f"AND abs(epoch(u.actual) - p.pt_arr_s) <= {int(MAX_ACTUAL.total_seconds())} GROUP BY p.key"
        ).df()
        con.close()
        out = []
        for row in r.itertuples():
            p = self.aci.pending[row.key]
            y = (row.act.timestamp() - p["pt_arr_s"]) / 60.0
            out.append({"key": row.key, "obs": row.obs.timestamp(), "y": y})
        return out

    def _score_aci(self, df: pd.DataFrame, outcomes: list[dict]) -> pd.DataFrame:
        gs, gamma = self.art["gs"], self.art["gamma"]
        heap: dict[tuple, list] = {}
        for o in outcomes:
            p = self.aci.pending[o["key"]]
            heap.setdefault((p["route"], p["h"]), []).append((o["obs"], o["key"], o["y"]))
        for v in heap.values():
            heapq.heapify(v)
        lo = np.empty(len(df))
        hi = np.empty(len(df))
        al = np.empty(len(df))
        order = np.argsort(df["snap_time"].to_numpy(), kind="stable")
        for i in order:
            r = df.iloc[i]
            stream = (r["route"], int(r["horizon_min"]))
            tt = r["snap_time"].timestamp()
            h = heap.get(stream, [])
            while h and h[0][0] <= tt:
                _, key, y = heapq.heappop(h)
                p = self.aci.pending.pop(key)
                miss = float(not (p["lo"] <= y <= p["hi"]))
                a = self.aci.alpha.get(f"{stream[0]}|{stream[1]}", TARGET_MISS)
                self.aci.alpha[f"{stream[0]}|{stream[1]}"] = float(
                    np.clip(a + gamma * (TARGET_MISS - miss), *A.ALPHA_BOUNDS)
                )
            a = self.aci.alpha.get(f"{stream[0]}|{stream[1]}", TARGET_MISS)
            q = gs.q(stream[1], 1.0 - a)
            w = max(r["raw_hi"] - r["raw_lo"], C.EPS_WIDTH)
            lo[i], hi[i], al[i] = r["raw_lo"] - q * w, r["raw_hi"] + q * w, a
        return df.assign(aci_lo=lo, aci_hi=hi, alpha=al)

    def _arrival_hour(self, hour: pd.Timestamp) -> pd.DataFrame:
        end = hour + HOUR
        ssds = sorted({hour.tz_convert("Europe/London").date() - timedelta(days=1),
                       hour.tz_convert("Europe/London").date()})  # fmt: skip
        outcomes = self._resolve_outcomes(ssds + [ssds[0] - timedelta(days=1)])
        parts = []
        for d in ssds:
            if not (self.root / "stop_updates" / f"service_date={d}").exists():
                continue
            if not self._anything_due(d, hour, 15, 60):
                continue
            gap = None
            x = due_arrival(self.root, d, self.specs, hour, end, self.art, self.cal,
                            self.fastest, self.holidays, gap)  # fmt: skip
            if len(x):
                parts.append(x)
            gc.collect()
            pa.default_memory_pool().release_unused()
            self._track_memory()
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        feed = self._age(self.minutes, df["snap_time"])
        df["feed_age_min"] = feed
        for w in (60, 120, 240):
            df[f"feed_gap{w}"] = self._recent_gap(self.minutes, df["snap_time"], w)
        df["svc_age_min"] = 0.0
        for d, ix in df.groupby("ssd").indices.items():
            df.loc[df.index[ix], "svc_age_min"] = self._service_age(
                d, df["rid"].iloc[ix], df["snap_time"].iloc[ix]
            )
        recent = df[f"feed_gap{self.th.recent_min}"]
        df["stale"] = (feed > self.th.feed_min) | (recent > self.th.feed_min)
        worst = np.maximum(feed, recent).round(0).astype(int).astype(str)
        df["stale_reason"] = np.where(df["stale"], "feed silence " + worst + " min", "")
        df = self._score_aci(df, outcomes)
        # rows scored while stale do not teach the ranges; register the others as pending
        for r in df.itertuples():
            if not r.stale:
                self.aci.pending[f"{r.rid}|{r.route}|{r.horizon_min}"] = {
                    "key": f"{r.rid}|{r.route}|{r.horizon_min}", "rid": r.rid, "route": r.route,
                    "h": int(r.horizon_min), "dest_tpl": r.dest_tpl, "pt_arr_s": r.pt_arr.timestamp(),
                    "lo": float(r.aci_lo), "hi": float(r.aci_hi),
                }  # fmt: skip
        hours = (df["feed_age_min"] / 60.0).clip(lower=0)
        f = 1.0 + self.th.widen_k * hours
        df["final_lo"] = np.where(
            df["stale"], df["point"] - f * (df["point"] - df["static_lo"]), df["aci_lo"]
        )
        df["final_hi"] = np.where(
            df["stale"], df["point"] + f * (df["static_hi"] - df["point"]), df["aci_hi"]
        )
        base = {"cold": models.predict(self.art["cold_fit"], df),
                "hist": models.predict(self.art["hist_fit"], df)}  # fmt: skip
        df["pers_point"] = base["hist"]["persistence"]["point"].to_numpy()
        df["pers_p"] = base["hist"]["persistence"]["p_late"].to_numpy()
        df["hist_point"] = base["hist"]["historical"]["point"].to_numpy()
        df["hist_p"] = base["hist"]["historical"]["p_late"].to_numpy()
        df["p_late_final"] = np.where(df["stale"], df["hist_p"], df["p_late"])
        df["ssd"] = df["ssd"].astype(str)
        keep = ["ssd", "rid", "route", "horizon_min", "snap_time", "pt_arr", "dest_tpl", "point",
                "raw_lo", "raw_hi", "static_lo", "static_hi", "aci_lo", "aci_hi", "alpha",
                "final_lo", "final_hi", "p_late", "p_late_final", "pers_point", "pers_p",
                "hist_point", "hist_p", "stale", "stale_reason", "feed_age_min", "feed_gap60",
                "feed_gap120", "feed_gap240", "svc_age_min"]  # fmt: skip
        return df[keep]

    # ------------------------------------------------------------------ pairs
    def _pair_hour(self, hour: pd.Timestamp) -> pd.DataFrame:
        end = hour + HOUR
        parts = []
        loc = hour.tz_convert("Europe/London").date()
        for d in sorted({loc - timedelta(days=1), loc}):
            if not (self.root / "stop_updates" / f"service_date={d}").exists():
                continue
            if not self._anything_due(d, hour, 5, 60):
                continue
            x, _ = CB.build_day(self.root, d, self.cfg, self.mct, self.tc, 1, labels=False,
                                due=(hour, end))  # fmt: skip
            if len(x):
                parts.append(x)
            gc.collect()
            self._track_memory()
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        df = DD.flag_groups(df, self.cfg).reset_index(drop=True)
        pred = self.conn.predict(df)
        df["p_b"] = pred["lgbm_cal"].to_numpy()
        df["p_logit"] = pred["logit_cal"].to_numpy()
        df["p_phys"] = pred["phys_cal"].to_numpy()
        from rail_risk.connections import baseline as BL

        df["p_bar"] = BL.predict(self.art["bar"], df)
        feed = self._age(self.minutes, df["snap_time"])
        hub = np.empty(len(df))
        hub_gap = np.empty(len(df))
        for h, ix in df.groupby("hub").indices.items():
            hub[ix] = self._age(self.hub_minutes[h], df["snap_time"].iloc[ix])
            hub_gap[ix] = self._recent_gap(self.hub_minutes[h], df["snap_time"].iloc[ix], 120)
        df["feed_age_min"], df["hub_age_min"], df["hub_gap120"] = feed, hub, hub_gap
        for w in (60, 120, 240):
            df[f"feed_gap{w}"] = self._recent_gap(self.minutes, df["snap_time"], w)
        for side in ("in", "out"):
            df[f"{side}_age_min"] = 0.0
            for d, ix in df.groupby("ssd").indices.items():
                df.loc[df.index[ix], f"{side}_age_min"] = self._service_age(
                    d, df[f"{side}_rid"].iloc[ix], df["snap_time"].iloc[ix]
                )
        recent = df[f"feed_gap{self.th.recent_min}"]
        feed_stale = (feed > self.th.feed_min) | (recent > self.th.feed_min)
        hub_stale = (hub > self.th.hub_min) | (hub_gap > self.th.hub_min)
        df["stale"] = feed_stale | hub_stale
        worst = np.maximum(feed, recent).round(0).astype(int).astype(str)
        hworst = np.maximum(hub, hub_gap).round(0).astype(int).astype(str)
        df["stale_reason"] = np.where(
            feed_stale, "feed silence " + worst + " min",
            np.where(hub_stale, "hub silence " + hworst + " min", ""),
        )  # fmt: skip
        df["p_final"] = np.where(df["stale"], df["p_bar"], df["p_b"])
        df["nya"] = df["in_arr_late_s"].isna()
        df["ssd"] = df["ssd"].astype(str)
        keep = ["ssd", "hub", "in_rid", "out_rid", "in_toc", "out_toc", "horizon_min", "snap_time",
                "plan_arr_in", "plan_dep_out", "slack_min", "mct_min", "held_group", "nya", "p_b",
                "p_logit", "p_phys", "p_bar", "p_final", "stale", "stale_reason", "feed_age_min",
                "hub_age_min", "hub_gap120", "feed_gap60", "feed_gap120", "feed_gap240",
                "in_age_min", "out_age_min"]  # fmt: skip
        return df[keep]

    # ------------------------------------------------------------------ hour
    def _track_memory(self) -> None:
        import resource

        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        self.peak_mb = max(self.peak_mb, mb)
        if self.peak_mb > self.mem_limit_mb:
            raise MemoryError(
                f"peak memory {self.peak_mb:.0f} MB is over the {self.mem_limit_mb:.0f} MB limit"
            )

    def blob(self) -> dict:
        return {
            "aci": self.aci.to_json(), "minutes": self.minutes.tolist(),
            "hub_minutes": {h: v.tolist() for h, v in self.hub_minutes.items()},
            "clock": str(self.clock),
        }  # fmt: skip

    def process_hour(self, hour: pd.Timestamp) -> dict:
        t0 = time.perf_counter()
        n_files = self.ingest(hour)
        self.clock = hour + HOUR
        t_ing = time.perf_counter() - t0
        arr = self._arrival_hour(hour)
        t_arr = time.perf_counter() - t0 - t_ing
        pairs = self._pair_hour(hour)
        t_pair = time.perf_counter() - t0 - t_ing - t_arr
        t1 = time.perf_counter()
        self.store.commit_hour(str(hour), {"arrival": arr, "pair": pairs}, self.blob())
        t_commit = time.perf_counter() - t1
        self._track_memory()
        stats = {
            "hour": str(hour), "files": n_files, "arrival_rows": len(arr), "pair_rows": len(pairs),
            "stale_arrival": int(arr["stale"].sum()) if len(arr) else 0,
            "stale_pairs": int(pairs["stale"].sum()) if len(pairs) else 0,
            "seconds": {"ingest": round(t_ing, 2), "arrival": round(t_arr, 2),
                        "pairs": round(t_pair, 2), "commit": round(t_commit, 3)},
            "peak_mb": round(self.peak_mb),
        }  # fmt: skip
        log.info("hour", **{k: v for k, v in stats.items() if k != "seconds"}, **stats["seconds"])
        return stats
