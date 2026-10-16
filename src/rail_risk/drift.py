"""Feature drift: population stability index of live serving features against training.

PSI compares the share of rows in each bin. Bins come from the training distribution, so a
feature that has moved shows up as live rows piling into a few training bins. Missing values are
a bin of their own, because a feed that stops reporting a field is drift too. Numeric features use
10 quantile bins from training, and categorical ones use the training categories plus "other".
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WATCH = 0.1
ACT = 0.25
BINS = 10
EPSILON = 1e-4  # floor on a bin share, so an empty bin does not make PSI infinite
MIN_LIVE_ROWS = 30


def _psi(train_share: np.ndarray, live_share: np.ndarray) -> float:
    t = np.clip(train_share, EPSILON, None)
    v = np.clip(live_share, EPSILON, None)
    return float(np.sum((v - t) * np.log(v / t)))


def psi_numeric(train: pd.Series, live: pd.Series, bins: int = BINS) -> dict[str, Any]:
    """PSI with `bins` quantile bins from `train`, plus one bin for missing values."""
    t = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(live, errors="coerce").to_numpy(dtype=float)
    t_ok, v_ok = t[~np.isnan(t)], v[~np.isnan(v)]
    if len(t_ok) == 0:
        raise ValueError("training column has no values")
    edges = np.unique(np.quantile(t_ok, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]  # values outside the training range fall in the first or last bin
    k = len(inner) + 1
    t_counts = np.bincount(np.searchsorted(inner, t_ok, side="right"), minlength=k)
    v_counts = np.bincount(np.searchsorted(inner, v_ok, side="right"), minlength=k)
    t_all = np.append(t_counts, np.isnan(t).sum()).astype(float)
    v_all = np.append(v_counts, np.isnan(v).sum()).astype(float)
    value = _psi(t_all / t_all.sum(), v_all / max(v_all.sum(), 1.0))
    return {
        "psi": value,
        "kind": "numeric",
        "bins": int(k),
        "train_missing": float(np.isnan(t).mean()),
        "live_missing": float(np.isnan(v).mean()) if len(v) else None,
    }


def psi_categorical(train: pd.Series, live: pd.Series) -> dict[str, Any]:
    """PSI over the training categories, with unseen live categories pooled as "other"."""
    t = train.astype("string").fillna("(missing)")
    v = live.astype("string").fillna("(missing)")
    cats = sorted(t.unique())
    t_counts = np.array([(t == c).sum() for c in cats] + [0], dtype=float)
    v_counts = np.array([(v == c).sum() for c in cats] + [(~v.isin(cats)).sum()], dtype=float)
    value = _psi(t_counts / t_counts.sum(), v_counts / max(v_counts.sum(), 1.0))
    return {
        "psi": value,
        "kind": "categorical",
        "bins": len(cats) + 1,
        "train_missing": float((t == "(missing)").mean()),
        "live_missing": float((v == "(missing)").mean()) if len(v) else None,
    }


def status(psi: float) -> str:
    return "act" if psi >= ACT else "watch" if psi >= WATCH else "ok"


def drift_report(
    train: pd.DataFrame,
    live: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    *,
    window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """PSI per feature, sorted most drifted first. Features missing from either frame are
    listed under `skipped` rather than dropped silently."""
    rows, skipped = [], []
    if len(live) < MIN_LIVE_ROWS:
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "window": window or {},
            "train_rows": int(len(train)),
            "live_rows": int(len(live)),
            "thresholds": {"watch": WATCH, "act": ACT},
            "features": [],
            "skipped": [],
            "summary": {"ok": 0, "watch": 0, "act": 0},
            "note": f"fewer than {MIN_LIVE_ROWS} live rows, so no PSI was computed",
        }
    for name, fn in [(c, psi_numeric) for c in numeric] + [
        (c, psi_categorical) for c in categorical
    ]:
        if name not in train.columns or name not in live.columns:
            skipped.append(name)
            continue
        try:
            result = fn(train[name], live[name])
        except ValueError as exc:
            skipped.append(f"{name} ({exc})")
            continue
        row = {"feature": name, **result, "status": status(result["psi"])}
        if "service_date" in live.columns and result["live_missing"] > 0:
            # where the gaps are: a feed that was not yet captured shows up by service date
            by_day = live.groupby(live["service_date"].astype(str))[name].agg(
                rows="size", missing=lambda v: float(v.isna().mean())
            )
            row["live_missing_by_service_date"] = {
                d: {"rows": int(r["rows"]), "missing": float(r["missing"])}
                for d, r in by_day.iterrows()
            }
        rows.append(row)
    rows.sort(key=lambda r: -r["psi"])
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("ok", "watch", "act")}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": window or {},
        "train_rows": int(len(train)),
        "live_rows": int(len(live)),
        "thresholds": {"watch": WATCH, "act": ACT},
        "features": rows,
        "skipped": skipped,
        "summary": counts,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    w = report["window"]
    lines = [
        "# Feature drift",
        "",
        f"Generated {report['generated_at']}. Live serving features from {w.get('live_from')} to "
        f"{w.get('live_to')} ({report['live_rows']} services, features built {w.get('minutes')} "
        f"minutes before arrival from Darwin) are compared with the training snapshot "
        f"({report['train_rows']} rows at the same horizon). PSI uses 10 quantile bins from "
        f"training, plus a bin for missing values. Under {WATCH} is ok, {WATCH} to {ACT} is "
        f"watch, and {ACT} or more is act.",
        "",
    ]
    if report.get("note"):
        lines += [report["note"], ""]
    else:
        s = report["summary"]
        lines += [
            f"Summary: {s['ok']} ok, {s['watch']} watch, {s['act']} act.",
            "",
            "| Feature | PSI | Status | Training missing | Live missing |",
            "|---|---|---|---|---|",
        ]
        for r in report["features"]:
            lines.append(
                f"| {r['feature']} | {r['psi']:.3f} | {r['status']} | "
                f"{r['train_missing']:.0%} | {r['live_missing']:.0%} |"
            )
        for r in report["features"]:
            by_day = r.get("live_missing_by_service_date")
            if by_day:
                lines += [
                    "",
                    f"Live rows with {r['feature']} missing, by service date:",
                    "",
                    "| Service date | Live services | Missing |",
                    "|---|---|---|",
                    *(
                        f"| {d} | {v['rows']} | {v['missing']:.0%} |"
                        for d, v in sorted(by_day.items())
                    ),
                ]
        if report["skipped"]:
            lines += ["", "Skipped: " + ", ".join(report["skipped"]) + "."]
    if w.get("caveat"):
        lines += ["", w["caveat"]]
    (out_dir / "drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
