#!/usr/bin/env python3
# ruff: noqa: E501
"""SMART coverage of the 25 routes (step 3): which stations and berths SMART maps, per route.

Route stations come from the calling points (kinds OR, IP, DT) of Darwin schedules that call at both
the route's origin and destination in order, on two days after oot_week (structure only, no delay
labels read). A station is mapped when its TIPLOC has a STANOX in CORPUS and that STANOX has SMART
berth rows. Berth coverage: the share of those stations with both an arrival-side and a
departure-side SMART event (EVENT A or C and B or D).

    uv run python scripts/td_route_coverage.py --smart FILE --corpus FILE --days 2026-10-24 2026-10-25
"""

import argparse
import gzip
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", default="reports/td/route_coverage.json")
    a = ap.parse_args()
    smart = pd.DataFrame(json.load(gzip.open(a.smart, "rt"))["BERTHDATA"])
    corpus = pd.DataFrame(json.load(gzip.open(a.corpus, "rt"))["TIPLOCDATA"])
    corpus = corpus[corpus.TIPLOC.str.strip() != ""]
    tpl2stanox = (
        corpus.drop_duplicates("TIPLOC").set_index("TIPLOC")["STANOX"].str.strip().to_dict()
    )
    ev = smart.groupby("STANOX").agg(
        rows=("EVENT", "size"),
        arr=("EVENT", lambda s: bool(s.isin(["A", "C"]).any())),
        dep=("EVENT", lambda s: bool(s.isin(["B", "D"]).any())),
    )
    crs = pd.read_parquet("data/interim/ref/tiploc_crs.parquet")[["tiploc", "crs"]].dropna()
    con = duckdb.connect()
    con.register("crs", crs)
    globs = [f"data/interim/pport/schedule_points/service_date={d}/*.parquet" for d in a.days]
    con.execute(
        "create view sp as select rid, seq, kind, tpl from read_parquet(["  # noqa: S608
        + ",".join(f"'{g}'" for g in globs)
        + "]) where kind in ('OR','OPOR','IP','OPIP','DT','OPDT')"
    )
    con.execute(
        "create view sc as select sp.rid, sp.seq, sp.tpl, crs.crs from sp left join crs on crs.tiploc = sp.tpl"
    )
    routes = [
        r
        for r in yaml.safe_load(Path("config/routes.yaml").read_text())["routes"]
        if not r.get("legacy")
    ]
    out = []
    for r in routes:
        df = con.execute(
            """
            with o as (select rid, min(seq) so from sc where crs = ? group by rid),
                 d as (select rid, max(seq) sd from sc where crs = ? group by rid),
                 ok as (select o.rid, so, sd from o join d using (rid) where so < sd)
            select sc.tpl, count(distinct sc.rid) n from sc join ok on sc.rid = ok.rid
              where sc.seq between ok.so and ok.sd group by sc.tpl
            """,
            [r["origin"], r["destination"]],
        ).df()
        n_rids = int(df.n.max()) if len(df) else 0
        st = df[df.n >= 0.5 * n_rids] if n_rids else df  # stations most services call at
        rows = []
        for tpl in st.tpl:
            sx = tpl2stanox.get(tpl, "").strip()
            hit = sx in ev.index
            rows.append(
                {
                    "tpl": tpl,
                    "stanox": sx,
                    "smart": hit,
                    "both": bool(hit and ev.loc[sx, "arr"] and ev.loc[sx, "dep"]),
                    "rows": int(ev.loc[sx, "rows"]) if hit else 0,
                }
            )
        t = pd.DataFrame(rows)
        out.append(
            {
                "route": r["id"],
                "length": r["length"],
                "held_out": bool(r.get("held_out")),
                "toc": r["toc"],
                "services": n_rids,
                "stations": len(t),
                "stations_mapped": float(t.smart.mean()) if len(t) else 0.0,
                "stations_arr_and_dep": float(t.both.mean()) if len(t) else 0.0,
                "smart_rows_per_station": float(t.rows.mean()) if len(t) else 0.0,
                "unmapped": t.loc[~t.smart, "tpl"].tolist() if len(t) else [],
            }
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(
        pd.DataFrame(out)
        .drop(columns="unmapped")
        .to_string(index=False, float_format=lambda v: f"{v:.2f}")
    )


if __name__ == "__main__":
    main()
