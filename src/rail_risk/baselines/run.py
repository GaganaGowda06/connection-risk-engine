"""One command for the Stage 2 baselines: build snapshots, fit on the train block, score the test
block, write the tables. Run by `rail-risk evaluate-baselines`."""

import hashlib
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from rail_risk.baselines import build, gap_bucket, models, pairs, scoring
from rail_risk.baselines.split import network_split
from rail_risk.clean.run import route_specs
from rail_risk.config import load_routes
from rail_risk.ingest.darwin_pport import load_stage1_config, split_for
from rail_risk.ingest.mct import MctError, load_mct_table
from rail_risk.logging import get_logger

ROOT = Path("data/interim/pport")
SNAP_DIR = Path("data/interim/stage2") / build.SNAP_DIR
REPORT_DIR = Path("reports/baselines")
HEADLINE = (15, 60)
GAP_HORIZONS = (5, 15, 60)


def git_head() -> str:
    try:
        return subprocess.check_output(  # noqa: S603
            ["/usr/bin/env", "git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assert_train_pool_only(days: list[date]) -> None:
    cfg = load_stage1_config()
    bad = [d for d in days if split_for(d, cfg) != cfg.default_split]
    if bad:
        raise RuntimeError(f"non train_pool service dates reached a fit or score: {bad[:3]}")


def run(
    routes_file: Path = Path("config/routes.yaml"),
    split_file: Path = Path("config/split.yaml"),
    stations_file: Path = Path("data/interim/ref/tiploc_crs.parquet"),
    root: Path = ROOT,
    snap_dir: Path = SNAP_DIR,
    out_dir: Path = REPORT_DIR,
    engine: str = "duckdb",
    rebuild: bool = False,
    routes_filter: list[str] | None = None,
) -> dict:
    log = get_logger("baselines")
    split, cfg = network_split(split_file)
    days = [*split.train, *split.validation, *split.test]
    assert_train_pool_only(days)
    routes = load_routes(routes_file)
    table = [r for r in routes.routes if not r.legacy]
    if routes_filter:
        table = [r for r in table if r.id in routes_filter]
    stations = pd.read_parquet(stations_file)
    specs = route_specs(type(routes)(routes=table), stations)
    built = build.build_range(root, days, specs, snap_dir, engine, rebuild)
    train = build.load_snapshots(snap_dir, list(split.train))
    test = build.load_snapshots(snap_dir, list(split.test))
    if train.empty or test.empty:
        raise RuntimeError("no snapshots for the train or test block; check the extract")
    fitted = models.fit(train)
    preds = models.predict(fitted, test)
    held = {r.id for r in table if r.held_out}
    stats = {m: scoring.row_stats(test, p) for m, p in preds.items()}
    table_df = scoring.score_all(
        stats, held, [r.id for r in table], cfg.resamples, cfg.seed, cfg.confidence
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(out_dir / "stage2_baselines.csv", index=False)
    gap, gap_meta = run_gap_bucket(
        root, split, routes, stations, snap_dir.parent / pairs.PAIRS_DIR, cfg, rebuild
    )
    if gap is not None:
        gap.to_csv(out_dir / "stage2_gap_bucket.csv", index=False)
    run_id = hashlib.sha256(
        f"{git_head()}|{split.train[-1]}|{split.test[0]}|{fitted.fit_rows}|{len(test)}".encode()
    ).hexdigest()[:12]
    meta = {
        "run_id": run_id,
        "git_head": git_head(),
        "engine": engine,
        "split": {k: [str(v[0]), str(v[-1]), len(v)] for k, v in split.blocks().items()},
        "fit_rows": fitted.fit_rows,
        "fit_services": fitted.fit_services,
        "test_rows": int(len(test)),
        "build": {k: (v if not isinstance(v, list) else len(v)) for k, v in built.items()},
        "missing_days": built["missing_days"],
        "gap_bucket": gap_meta,
        "routes_without_test_rows": sorted(
            set(r.id for r in table) - set(test.loc[test["has_target"], "route"])
        ),
        "held_out": sorted(held),
        "oot_week_or_winter_rows_in_fit": 0,
    }
    build.write_json(out_dir / "stage2_baselines.json", meta)
    write_markdown(table_df, meta, out_dir / "stage2_baselines.md", gap)
    log.info("baselines_done", run_id=run_id, test_rows=len(test), fit_rows=fitted.fit_rows)
    return meta


def run_gap_bucket(root, split, routes, stations, pairs_dir, cfg, rebuild):
    """Fit the gap buckets on train pairs and score the test pairs; blocked without an MCT file."""
    try:
        table = load_mct_table()
    except MctError as exc:
        return None, {"status": f"blocked: {exc}"}
    built = pairs.build_pairs_range(
        root, [*split.train, *split.test], routes, stations, pairs_dir, rebuild
    )
    out = {}
    for name, days in (("train", split.train), ("test", split.test)):
        p = pairs.load_pairs(pairs_dir, list(days))
        p = pairs.known_at(p, 15) if name == "train" else p
        p["mct_min"] = gap_bucket.pair_mct(p, table)
        out[name] = (p, int(p["mct_min"].isna().sum()))
    train = gap_bucket.label_pairs(out["train"][0][out["train"][0]["schedule_known"]])
    model = gap_bucket.fit_buckets(train)
    parts = []
    for h in GAP_HORIZONS:
        t = pairs.known_at(out["test"][0].drop(columns=["mct_min"]), h)
        t["mct_min"] = gap_bucket.pair_mct(t, table)
        t = gap_bucket.label_pairs(t[t["schedule_known"]])
        sc = scoring.score_pairs(
            scoring.pair_stats(t, gap_bucket.predict_buckets(model, t)),
            sorted(t["hub"].unique()),
            cfg.resamples,
            cfg.seed,
            cfg.confidence,
        )
        sc.insert(0, "horizon_min", h)
        sc["miss_rate"] = [
            float(
                (t if g == "all_hubs" else t[t["hub"] == g])
                .pipe(lambda x, s=s: x if s == "all_pairs" else x[~x["inbound_arrived_by_t"]])[
                    "miss"
                ]
                .mean()
            )
            for s, g in zip(sc["subset"], sc["group"], strict=True)
        ]
        parts.append(sc)
    mct_used = (
        train.groupby(["hub", "in_toc", "out_toc"])["mct_min"]
        .first()
        .reset_index()
        .to_dict("records")
    )
    meta = {
        "status": "scored",
        "mct_source": str(table["source_file"].iloc[0]),
        "mct_seq": str(table["source_seq"].iloc[0]),
        "mct_generated": str(table["generated_date"].iloc[0]),
        "pairs_built": built,
        "train_pairs": int(len(train)),
        "pairs_without_mct": {k: v[1] for k, v in out.items()},
        "mct_used": mct_used,
    }
    return pd.concat(parts, ignore_index=True), meta


def _cell(row: pd.Series, key: str, digits: int = 2) -> str:
    if row.empty or pd.isna(row.get(key)):
        return "n/a"
    return f"{row[key]:.{digits}f} [{row[f'{key}_lo']:.{digits}f}, {row[f'{key}_hi']:.{digits}f}]"


def write_markdown(
    df: pd.DataFrame, meta: dict, path: Path, gap: pd.DataFrame | None = None
) -> None:
    lines = [
        "# Stage 2 baselines",
        "",
        f"Run {meta['run_id']} at commit {meta['git_head']}. Fitted on the train block "
        f"({meta['split']['train'][0]} to {meta['split']['train'][1]}, {meta['split']['train'][2]} "
        f"days), scored on the test block ({meta['split']['test'][0]} to "
        f"{meta['split']['test'][1]}). Intervals are 95% bootstrap CIs over service dates.",
        "The historical baseline uses each held-out route's own earlier dates, which makes the "
        "bar harder than a route it has never seen. darwin_forecast is a reference, not a GOAL "
        "baseline. Lateness is in minutes. Coverage is of the 10 to 90 percent range.",
        f"Gap-bucket P(miss): {meta['gap_bucket']['status']}.",
        "",
    ]
    for group, title in (
        ("held_out", "Held-out routes together"),
        ("train_routes", "Other routes"),
    ):
        lines += [
            f"## {title}",
            "",
            "| method | minutes before | n | MAE | coverage | Brier | ECE |",
            "|---|---|---|---|---|---|---|",
        ]
        for method in models.METHODS:
            for h in HEADLINE:
                r = df[(df.method == method) & (df.group == group) & (df.horizon_min == h)]
                r = r.iloc[0] if len(r) else pd.Series(dtype=float)
                n = int(r["n"]) if len(r) and not pd.isna(r.get("n")) else 0
                lines.append(
                    f"| {method} | {h} | {n} | {_cell(r, 'mae')} | {_cell(r, 'coverage', 3)} | "
                    f"{_cell(r, 'brier', 3)} | {_cell(r, 'ece', 3)} |"
                )
        lines.append("")
    lines += [
        "## Held-out routes one by one (historical and persistence)",
        "",
        "| route | method | minutes before | n | MAE | coverage | Brier | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for route in meta["held_out"]:
        for method in ("persistence", "historical"):
            for h in HEADLINE:
                r = df[(df.method == method) & (df.group == route) & (df.horizon_min == h)]
                r = r.iloc[0] if len(r) else pd.Series(dtype=float)
                n = int(r["n"]) if len(r) and not pd.isna(r.get("n")) else 0
                lines.append(
                    f"| {route} | {method} | {h} | {n} | {_cell(r, 'mae')} | "
                    f"{_cell(r, 'coverage', 3)} | {_cell(r, 'brier', 3)} | {_cell(r, 'ece', 3)} |"
                )
    if gap is not None:
        lines += gap_markdown(gap, meta["gap_bucket"])
    lines += [
        "",
        "Note: bri-wsm still has about 2 percent of stops with arrival or departure "
        "inversions of two minutes or more (Stage 1), so its actual times carry extra noise.",
    ]
    path.write_text("\n".join(lines) + "\n")


def gap_markdown(gap: pd.DataFrame, meta: dict) -> list[str]:
    lines = [
        "",
        "## Gap-bucket P(miss) on the interchange pairs (test block)",
        "",
        f"MCT from {meta['mct_source']} (sequence {meta['mct_seq']}, generated "
        f"{meta['mct_generated']}), TSI over MSN by the real TOC pair of each pair. Buckets of "
        "planned slack over MCT, fitted on train pairs (including held-out outbound routes' own "
        "earlier dates). 'not arrived' means no actual arrival of the inbound by the forecast "
        "time.",
        "",
        "| minutes before | pairs | hub | n | miss rate | Brier | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in gap.itertuples(index=False):
        row = pd.Series(r._asdict())
        n = int(row["n"]) if not pd.isna(row.get("n")) else 0
        lines.append(
            f"| {r.horizon_min} | {r.subset} | {r.group} | {n} | "
            f"{row['miss_rate']:.3f} | {_cell(row, 'brier', 3)} | {_cell(row, 'ece', 3)} |"
        )
    return lines
