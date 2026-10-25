# ruff: noqa: S608
"""Connection pairs and their state at T for every passenger service at the hubs.

A pair is a passenger arrival of one service at a hub and a passenger departure of another from
the same hub on the same service date, with the scheduled gap between the hub MCT and
`max_gap_min`. The timetable is the first schedule message seen for each rid. T is the outbound's
scheduled departure minus the horizon; state uses only messages with message_time <= T. The MCT
comes from data/ref/mct.csv by the real TOC pair (TSI over MSN); a pair with no MCT is dropped.

Outcomes: miss = inbound actual arrival + MCT is later than the outbound's actual departure; an
inbound that was cancelled or never reached the hub is a miss. An outbound that was cancelled or
never left the hub is reported separately and has no label (out_cancel = 1). A service with no
actual and no cancellation is data missing: the pair is dropped. Nothing here names a route.
"""

import hashlib
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from rail_risk.baselines.pairs import MAX_ACTUAL_GAP  # noqa: F401  (same 12 hour rule)
from rail_risk.config import load_routes
from rail_risk.ingest.mct import load_mct_table, mct_lookup  # noqa: F401
from rail_risk.ingest.pport_extract import train_pool_table, train_pool_view  # noqa: F401

CONFIG = Path("config/connections.yaml")
PAIRS_DIR = Path("data/interim/stage5")
MEMORY_LIMIT = "8GB"
THREADS = 4
KEYS = ["ssd", "hub", "in_rid", "out_rid", "horizon_min"]


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def hub_tiplocs(hubs: list[str], tiploc_crs: pd.DataFrame) -> pd.DataFrame:
    t = tiploc_crs.dropna(subset=["crs"])
    t = t[t["crs"].isin(hubs)][["crs", "tiploc"]].drop_duplicates()
    missing = set(hubs) - set(t["crs"])
    if missing:
        raise ValueError(f"no TIPLOC for hub {sorted(missing)}")
    return t.rename(columns={"crs": "hub", "tiploc": "tpl"})


def group_is_held(corridor: str, mod: int, hold: int, routes_held: set[str]) -> bool:
    if corridor in routes_held:
        return True
    return int(hashlib.md5(corridor.encode()).hexdigest(), 16) % mod == hold  # noqa: S324


def held_route_corridors(path: Path = Path("config/routes.yaml")) -> set[str]:
    out = set()
    for r in load_routes(path).routes:
        if r.held_out and not r.legacy:
            out |= {f"{r.origin}|{r.destination}", f"{r.destination}|{r.origin}"}
    return out


def pair_mct(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """MCT by (hub, inbound TOC, outbound TOC); NaN when unknown (the pair is then dropped)."""
    keys = pairs[["hub", "in_toc", "out_toc"]].drop_duplicates()
    vals = {}
    for h, a, d in keys.itertuples(index=False):
        v = mct_lookup(table, h, a, d)
        vals[(h, a, d)] = np.nan if v is None else v
    return pd.Series(
        [vals[k] for k in zip(pairs["hub"], pairs["in_toc"], pairs["out_toc"], strict=True)],
        index=pairs.index,
        dtype="float64",
    )


def _rss_mb() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)  # bytes on macOS


def _cols(root, table, cols, ssd):
    t = train_pool_table(root, table, cols, [ssd])
    return t if t.num_rows else None


ASOF = """
CREATE OR REPLACE TEMP TABLE {name} AS
SELECT pt.pid, pt.h, arg_max(su.{field}, {{'m': su.message_time, 'v': su.{field}}}) AS v
FROM pt JOIN su ON su.rid = pt.{rid} AND su.tpl = pt.tpl AND su.event = '{event}'
  AND su.{field} IS NOT NULL AND su.message_time <= pt.T
  AND abs(epoch(su.{field}) - epoch(pt.{plan})) <= {win}
GROUP BY 1, 2
"""

LAST = """
CREATE OR REPLACE TEMP TABLE {name} AS
SELECT pt.pid, pt.h,
  arg_max({{'late': a.late_s, 't': epoch(a.actual)}},
          {{'t': a.actual, 'm': a.message_time, 'p': a.tpl, 'e': a.event}}) AS x
FROM pt JOIN a ON a.rid = pt.{rid} AND a.message_time <= pt.T
GROUP BY 1, 2
"""


def _build_day(
    root: Path, ssd: date, cfg: dict, mct: pd.DataFrame, tiploc_crs: pd.DataFrame, thin: int
) -> tuple[pd.DataFrame, dict]:
    """Rows for one service date: one per (pair, horizon). thin 1 keeps every pair."""
    win = int(cfg["actual_window_h"]) * 3600
    tpls = hub_tiplocs(cfg["hubs"], tiploc_crs)
    corr = tiploc_crs.dropna(subset=["crs"])[["tiploc", "crs"]].drop_duplicates("tiploc")
    stats_mem: dict = {}
    con = duckdb.connect()
    spill = PAIRS_DIR / "_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET temp_directory='{spill}'")
    con.execute("SET TimeZone='UTC'")
    d = [ssd]
    ok = all(
        (
            train_pool_view(
                con,
                root,
                "sch_v",
                "schedules",
                ["rid", "toc", "train_cat", "is_passenger", "deleted", "is_active", "message_time"],
                d,
            ),  # fmt: skip
            train_pool_view(
                con,
                root,
                "sp_v",
                "schedule_points",
                ["rid", "seq", "tpl", "pt_arr", "pt_dep", "cancelled", "message_time"],
                d,
            ),  # fmt: skip
            train_pool_view(
                con,
                root,
                "raw_su",
                "stop_updates",
                [
                    "rid",
                    "tpl",
                    "event",
                    "pt_arr",
                    "pt_dep",
                    "wt_arr",
                    "wt_dep",
                    "wt_pass",
                    "est",
                    "actual",
                    "message_time",
                ],
                d,
            ),  # fmt: skip
            train_pool_view(
                con, root, "sa_v", "stop_actuals", ["rid", "tpl", "event", "actual"], d
            ),  # fmt: skip
        )
    )
    if not ok:
        con.close()
        return pd.DataFrame(), {"ssd": str(ssd), "pairs": 0}
    con.execute("CREATE TEMP TABLE sch AS SELECT * FROM sch_v")
    con.execute("CREATE TEMP TABLE sp AS SELECT * FROM sp_v")
    stats_mem["tables"] = _rss_mb()
    con.register("ht", tpls)
    con.register("tc", corr)
    con.execute(
        f"""
        CREATE TEMP TABLE ps AS SELECT rid, toc, message_time AS seen FROM (
          SELECT rid, toc, message_time,
                 row_number() OVER (PARTITION BY rid ORDER BY message_time) rn FROM sch
          WHERE coalesce(is_passenger, false) AND coalesce(train_cat, '') NOT IN ('BR','BS','SS'))
        WHERE rn = 1;
        CREATE TEMP TABLE fp AS SELECT sp.* FROM sp JOIN ps
          ON sp.rid = ps.rid AND sp.message_time = ps.seen
          WHERE sp.pt_arr IS NOT NULL OR sp.pt_dep IS NOT NULL;
        CREATE TEMP TABLE ends AS
          SELECT rid,
                 arg_min(tpl, seq) FILTER (WHERE pt_dep IS NOT NULL) AS o_tpl,
                 arg_max(tpl, seq) FILTER (WHERE pt_arr IS NOT NULL) AS d_tpl
          FROM fp GROUP BY rid;
        CREATE TEMP TABLE pr0 AS
        SELECT ht.hub, a.tpl, a.rid AS in_rid, d.rid AS out_rid, pa.toc AS in_toc,
               pd.toc AS out_toc, a.pt_arr AS plan_arr_in, d.pt_dep AS plan_dep_out,
               d.pt_arr AS plan_arr_out, pa.seen AS in_seen, pd.seen AS out_seen,
               (a.pt_dep IS NOT NULL) AS in_through, (d.pt_arr IS NOT NULL) AS out_through,
               coalesce(oi.crs, ei.o_tpl) || '|' || coalesce(di.crs, ei.d_tpl) AS in_corr,
               coalesce(oo.crs, eo.o_tpl) || '|' || coalesce(do_.crs, eo.d_tpl) AS out_corr,
               epoch(d.pt_dep - a.pt_arr) / 60.0 AS gap
        FROM fp a JOIN ht ON a.tpl = ht.tpl AND a.pt_arr IS NOT NULL
        JOIN fp d ON d.tpl = a.tpl AND d.pt_dep IS NOT NULL AND d.rid <> a.rid
          AND d.pt_dep >= a.pt_arr AND d.pt_dep <= a.pt_arr + INTERVAL ({cfg["max_gap_min"]}) MINUTE
        JOIN ps pa ON pa.rid = a.rid JOIN ps pd ON pd.rid = d.rid
        JOIN ends ei ON ei.rid = a.rid JOIN ends eo ON eo.rid = d.rid
        LEFT JOIN tc oi ON oi.tiploc = ei.o_tpl LEFT JOIN tc di ON di.tiploc = ei.d_tpl
        LEFT JOIN tc oo ON oo.tiploc = eo.o_tpl LEFT JOIN tc do_ ON do_.tiploc = eo.d_tpl
        WHERE {thin} = 1 OR hash(a.rid || '|' || d.rid) % {thin} = 0
        """
    )
    pr = con.execute("SELECT * FROM pr0").df()
    stats_mem["pr0"] = _rss_mb()
    stats = {"ssd": str(ssd), "pairs_before_mct": len(pr)}
    if pr.empty:
        return pd.DataFrame(), {**stats, "pairs": 0}
    pr["mct_min"] = pair_mct(pr, mct)
    stats["dropped_unknown_mct"] = int(pr["mct_min"].isna().sum())
    pr = pr[pr["mct_min"].notna() & (pr["gap"] >= pr["mct_min"])].copy()
    pr = pr.reset_index(drop=True)
    pr["pid"] = pr.index
    con.register("pr", pr)
    con.register("hz", pd.DataFrame({"h": cfg["horizons"]}))
    con.execute(
        f"""
        CREATE TEMP TABLE pair_rids AS SELECT in_rid AS rid FROM pr UNION SELECT out_rid FROM pr;
        CREATE TEMP TABLE su AS SELECT * FROM raw_su
          WHERE rid IN (SELECT rid FROM pair_rids) OR tpl IN (SELECT tpl FROM ht);
        CREATE TEMP TABLE sa AS SELECT * FROM sa_v WHERE rid IN (SELECT rid FROM pair_rids);
        CREATE TEMP TABLE pt_all AS
        SELECT pr.*, hz.h, pr.plan_dep_out - hz.h * INTERVAL 1 MINUTE AS T FROM pr, hz
        WHERE pr.in_seen <= pr.plan_dep_out - hz.h * INTERVAL 1 MINUTE
          AND pr.out_seen <= pr.plan_dep_out - hz.h * INTERVAL 1 MINUTE;
        CREATE TEMP TABLE a AS
        SELECT rid, tpl, event, message_time, actual,
               epoch(actual) - epoch(CASE event WHEN 'arr' THEN coalesce(pt_arr, wt_arr)
                 WHEN 'dep' THEN coalesce(pt_dep, wt_dep) ELSE wt_pass END) AS late_s
        FROM su WHERE actual IS NOT NULL AND CASE event WHEN 'arr' THEN coalesce(pt_arr, wt_arr)
                 WHEN 'dep' THEN coalesce(pt_dep, wt_dep) ELSE wt_pass END IS NOT NULL;
        DELETE FROM a WHERE abs(late_s) > {win};
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE spc AS SELECT sp.rid, sp.tpl, sp.message_time, sp.cancelled
          FROM sp WHERE tpl IN (SELECT tpl FROM ht);
        CREATE TEMP TABLE hl AS
          SELECT rid, tpl, min(message_time) AS m1, arg_min(late_s, message_time) AS late_s,
                 arg_min(actual, message_time) AS actual
          FROM a WHERE event = 'arr' AND tpl IN (SELECT tpl FROM ht) GROUP BY rid, tpl;
        CREATE TEMP TABLE hc AS
          SELECT rid, tpl, min(message_time) AS m1 FROM spc WHERE cancelled GROUP BY rid, tpl;
        CREATE TEMP TABLE fp_hub AS SELECT fp.tpl, coalesce(fp.pt_dep, fp.pt_arr) AS tm, ps.seen
          FROM fp JOIN ps ON ps.rid = fp.rid WHERE fp.tpl IN (SELECT tpl FROM ht);
        CREATE TEMP TABLE ut AS SELECT DISTINCT tpl, T, plan_dep_out FROM pt_all;
        CREATE TEMP TABLE cong AS
          SELECT ut.tpl, ut.T, ut.plan_dep_out,
            (SELECT count(*) FROM hl WHERE hl.tpl = ut.tpl AND hl.m1 <= ut.T
               AND hl.actual > ut.T - INTERVAL 30 MINUTE) AS cong_recent_n,
            (SELECT avg(late_s) / 60.0 FROM hl WHERE hl.tpl = ut.tpl AND hl.m1 <= ut.T
               AND hl.actual > ut.T - INTERVAL 30 MINUTE) AS cong_recent_late_min,
            (SELECT count(*) FROM hc WHERE hc.tpl = ut.tpl AND hc.m1 <= ut.T
               AND hc.m1 > ut.T - INTERVAL 60 MINUTE) AS cong_cancel_n,
            (SELECT count(*) FROM fp_hub WHERE fp_hub.tpl = ut.tpl
               AND fp_hub.seen <= ut.T
               AND fp_hub.tm BETWEEN ut.plan_dep_out - INTERVAL 30 MINUTE
                   AND ut.plan_dep_out + INTERVAL 30 MINUTE) AS cong_sched_n
          FROM ut;
        """
    )
    stats_mem["su_a"] = _rss_mb()
    n_slices = int(cfg.get("slices", 8))
    parts = []
    for k in range(n_slices):
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE pt AS SELECT * FROM pt_all WHERE pid % {n_slices} = {k}"
        )
        for name, rid, ev, field, plan in (
            ("in_act", "in_rid", "arr", "actual", "plan_arr_in"),
            ("in_est", "in_rid", "arr", "est", "plan_arr_in"),
            ("out_dact", "out_rid", "dep", "actual", "plan_dep_out"),
            ("out_dest", "out_rid", "dep", "est", "plan_dep_out"),
            ("out_aact", "out_rid", "arr", "actual", "plan_arr_out"),
            ("out_aest", "out_rid", "arr", "est", "plan_arr_out"),
        ):
            con.execute(ASOF.format(name=name, rid=rid, event=ev, field=field, plan=plan, win=win))
        con.execute(LAST.format(name="last_in", rid="in_rid"))
        stats_mem[f"s{k}_asof"] = _rss_mb()
        con.execute(LAST.format(name="last_out", rid="out_rid"))
        stats_mem[f"s{k}_last"] = _rss_mb()
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE canc_in AS SELECT pt.pid, pt.h,
              arg_max(spc.cancelled, spc.message_time) AS v FROM pt JOIN spc
              ON spc.rid = pt.in_rid AND spc.tpl = pt.tpl
              AND spc.message_time <= pt.T GROUP BY 1, 2;
            CREATE OR REPLACE TEMP TABLE canc_out AS SELECT pt.pid, pt.h,
              arg_max(spc.cancelled, spc.message_time) AS v FROM pt JOIN spc
              ON spc.rid = pt.out_rid AND spc.tpl = pt.tpl
              AND spc.message_time <= pt.T GROUP BY 1, 2;
            """
        )
        sql = """
        SELECT pt.hub, pt.in_rid, pt.out_rid, pt.in_toc, pt.out_toc, pt.in_corr,
          pt.out_corr, pt.plan_arr_in, pt.plan_dep_out, pt.in_through, pt.out_through,
          pt.mct_min, pt.gap, pt.h AS horizon_min, pt.T AS snap_time,
          epoch(ia.v) - epoch(pt.plan_arr_in) AS in_arr_late_s,
          epoch(ie.v) - epoch(pt.plan_arr_in) AS in_fc_late_s,
          li.x.late AS in_last_late_s, epoch(pt.T) - li.x.t AS in_last_age_s,
          coalesce(ci.v, false) AS in_cancel_known,
          epoch(od.v) - epoch(pt.plan_dep_out) AS out_dep_late_s,
          epoch(oe.v) - epoch(pt.plan_dep_out) AS out_fc_late_s,
          coalesce(epoch(oa.v), epoch(oae.v)) - epoch(pt.plan_arr_out) AS out_arr_late_s,
          (oa.v IS NOT NULL) AS out_arr_actual,
          lo.x.late AS out_last_late_s, epoch(pt.T) - lo.x.t AS out_last_age_s,
          coalesce(co.v, false) AS out_cancel_known,
          cg.cong_recent_n, cg.cong_recent_late_min, cg.cong_cancel_n, cg.cong_sched_n,
          pt.pid
        FROM pt
        LEFT JOIN in_act ia USING (pid, h) LEFT JOIN in_est ie USING (pid, h)
        LEFT JOIN last_in li USING (pid, h) LEFT JOIN canc_in ci USING (pid, h)
        LEFT JOIN out_dact od USING (pid, h) LEFT JOIN out_dest oe USING (pid, h)
        LEFT JOIN out_aact oa USING (pid, h) LEFT JOIN out_aest oae USING (pid, h)
        LEFT JOIN last_out lo USING (pid, h) LEFT JOIN canc_out co USING (pid, h)
        LEFT JOIN cong cg ON cg.tpl = pt.tpl AND cg.T = pt.T AND cg.plan_dep_out = pt.plan_dep_out
        """
        stats_mem[f"s{k}_cong"] = _rss_mb()
        parts.append(con.execute(sql).df())
        con.execute("DROP TABLE pt")
    feats = pd.concat(parts, ignore_index=True)
    stats_mem["features"] = _rss_mb()
    # outcomes, one per pair
    con.execute(
        f"""
        CREATE TEMP TABLE o_in AS SELECT pid, actual AS act_arr_in FROM (
          SELECT pr.pid, sa.actual, row_number() OVER (PARTITION BY pr.pid
            ORDER BY abs(epoch(sa.actual) - epoch(pr.plan_arr_in))) rn
          FROM pr JOIN sa ON sa.rid = pr.in_rid AND sa.tpl = pr.tpl AND sa.event = 'arr'
            AND sa.actual IS NOT NULL
            AND abs(epoch(sa.actual) - epoch(pr.plan_arr_in)) <= {win}) WHERE rn = 1;
        CREATE TEMP TABLE o_out AS SELECT pid, actual AS act_dep_out FROM (
          SELECT pr.pid, sa.actual, row_number() OVER (PARTITION BY pr.pid
            ORDER BY abs(epoch(sa.actual) - epoch(pr.plan_dep_out))) rn
          FROM pr JOIN sa ON sa.rid = pr.out_rid AND sa.tpl = pr.tpl AND sa.event = 'dep'
            AND sa.actual IS NOT NULL
            AND abs(epoch(sa.actual) - epoch(pr.plan_dep_out)) <= {win}) WHERE rn = 1;
        CREATE TEMP TABLE ran AS SELECT DISTINCT rid FROM sa WHERE actual IS NOT NULL;
        CREATE TEMP TABLE lastsch AS SELECT rid, max(message_time) AS m FROM sch GROUP BY rid;
        CREATE TEMP TABLE cancel_sch AS
          SELECT l.rid, bool_or(coalesce(s.deleted, false) OR NOT coalesce(s.is_active, true)) AS sc
          FROM lastsch l JOIN sch s ON s.rid = l.rid AND s.message_time = l.m GROUP BY l.rid;
        CREATE TEMP TABLE cancel_pt AS
          SELECT spc.rid, bool_or(spc.cancelled) AS pc FROM spc JOIN lastsch l
            ON l.rid = spc.rid AND l.m = spc.message_time GROUP BY spc.rid;
        """
    )
    out = con.execute(
        """
        SELECT pr.pid, oi.act_arr_in, oo.act_dep_out,
          coalesce(cs.sc, false) OR coalesce(cp.pc, false) AS in_cancel_final,
          ri.rid IS NOT NULL AS in_ran,
          coalesce(cs2.sc, false) OR coalesce(cp2.pc, false) AS out_cancel_final,
          ro.rid IS NOT NULL AS out_ran
        FROM pr
        LEFT JOIN o_in oi USING (pid) LEFT JOIN o_out oo USING (pid)
        LEFT JOIN cancel_sch cs ON cs.rid = pr.in_rid LEFT JOIN cancel_pt cp ON cp.rid = pr.in_rid
        LEFT JOIN cancel_sch cs2 ON cs2.rid = pr.out_rid
        LEFT JOIN cancel_pt cp2 ON cp2.rid = pr.out_rid
        LEFT JOIN ran ri ON ri.rid = pr.in_rid LEFT JOIN ran ro ON ro.rid = pr.out_rid
        """
    ).df()
    con.close()
    df = feats.merge(out, on="pid", how="left")
    df["ssd"] = ssd
    stats["rows"] = len(df)
    stats["peak_rss_mb_by_step"] = stats_mem
    return label(df, stats)


def label(df: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, dict]:
    """Outcome rules (see the module docstring). Rows with no usable outcome are dropped."""
    arrived = df["act_arr_in"].notna()
    departed = df["act_dep_out"].notna()
    in_miss_status = ~arrived & (df["in_cancel_final"] | df["in_ran"])
    in_unknown = ~arrived & ~in_miss_status
    out_cancel = ~departed & (df["out_cancel_final"] | df["out_ran"])
    out_unknown = ~departed & ~out_cancel
    df = df.assign(out_cancel=out_cancel.astype(int))
    stats["unknown_inbound_dropped"] = int(in_unknown.sum())
    stats["unknown_outbound_dropped"] = int(out_unknown.sum())
    keep = ~in_unknown & ~out_unknown
    df = df[keep].copy()
    arrived, departed = df["act_arr_in"].notna(), df["act_dep_out"].notna()
    late_arr = df["act_arr_in"] + pd.to_timedelta(df["mct_min"], unit="m")
    miss = (~arrived) | (late_arr > df["act_dep_out"])
    df["miss"] = np.where(df["out_cancel"] == 1, np.nan, miss.astype(float))
    df["in_late_final_min"] = (df["act_arr_in"] - df["plan_arr_in"]).dt.total_seconds() / 60.0
    df["out_late_final_min"] = (df["act_dep_out"] - df["plan_dep_out"]).dt.total_seconds() / 60.0
    df["slack_min"] = df["gap"] - df["mct_min"]
    df["dep_resolved"] = df["act_dep_out"]
    return df, stats


def load_day(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def february_days() -> list[date]:
    from rail_risk.baselines.split import is_complete, manifest_files
    from rail_risk.ingest.darwin_pport import load_stage1_config, split_for

    cfg = load_stage1_config()
    files = manifest_files(Path("data/manifests/darwin_archive.csv"))
    days = [date(2026, 2, d) for d in range(1, 29)]
    return [d for d in days if split_for(d, cfg) == "winter_test" and is_complete(d, files)]


def build_day(root, ssd, cfg, mct, tiploc_crs, thin):
    """One service date; the tables and the DuckDB connection are released before returning."""
    import gc

    try:
        return _build_day(root, ssd, cfg, mct, tiploc_crs, thin)
    finally:
        gc.collect()
