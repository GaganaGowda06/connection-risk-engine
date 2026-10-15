"""Evaluation harness: time-based split, point, interval and band metrics, and baselines.

Metrics are ratios of per-service-date sums, so the bootstrap resamples whole service dates
(rows within a day are correlated) cheaply. The baselines are fitted on the training block only.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOG_LOSS_EPS = 1e-15
METRIC_NAMES = ("mae", "rmse", "coverage", "width", "brier", "log_loss")
_SUMS = ("n", "abs", "sq", "n_int", "cov", "width", "n_band", "brier", "ll")


# split


@dataclass(frozen=True)
class SplitConfig:
    test_days: int
    validation_days: int
    gap_days: int
    windows_per_day: int
    resamples: int
    seed: int
    confidence: float


def load_split_config(path: Path) -> SplitConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    boot = raw.get("bootstrap", {})
    cfg = SplitConfig(
        test_days=int(raw["test_days"]),
        validation_days=int(raw["validation_days"]),
        gap_days=int(raw["gap_days"]),
        windows_per_day=int(raw["windows_per_day"]),
        resamples=int(boot.get("resamples", 1000)),
        seed=int(boot.get("seed", 0)),
        confidence=float(boot.get("confidence", 0.95)),
    )
    if min(cfg.test_days, cfg.validation_days, cfg.gap_days, cfg.windows_per_day) < 1:
        raise ValueError("split sizes and the gap must be at least 1")
    return cfg


@dataclass(frozen=True)
class Split:
    train: tuple[date, ...]
    validation: tuple[date, ...]
    test: tuple[date, ...]

    def blocks(self) -> dict[str, tuple[date, ...]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def make_split(complete_days: list[date], cfg: SplitConfig) -> Split:
    days = sorted(set(complete_days))
    need = cfg.test_days + cfg.validation_days + 2 * cfg.gap_days + 1
    if len(days) < need:
        raise ValueError(f"need at least {need} complete days for the split, have {len(days)}")
    test = days[-cfg.test_days :]
    rest = days[: -cfg.test_days - cfg.gap_days]
    validation = rest[-cfg.validation_days :]
    train = rest[: -cfg.validation_days - cfg.gap_days]
    split = Split(tuple(train), tuple(validation), tuple(test))
    check_split(split, cfg.gap_days)
    return split


def check_split(split: Split, gap_days: int = 1) -> None:
    """No date in two blocks, and blocks in time order with at least gap_days between them."""
    blocks = [split.train, split.validation, split.test]
    if not all(blocks):
        raise ValueError("every block must have at least one day")
    seen: set[date] = set()
    for block in blocks:
        overlap = seen & set(block)
        if overlap:
            raise ValueError(f"service dates in two blocks: {sorted(overlap)}")
        seen |= set(block)
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        if not max(earlier) < min(later):
            raise ValueError("blocks are not in time order")
        if (min(later) - max(earlier)).days <= gap_days:
            raise ValueError(f"blocks are closer than the {gap_days} day gap")


def complete_days(manifest: Path, routes: list[str], windows_per_day: int) -> list[date]:
    """Dates for which every given route has all its windows ingested."""
    con = sqlite3.connect(f"file:{manifest}?mode=ro", uri=True)
    try:
        per_route = []
        for route in routes:
            rows = con.execute(
                "SELECT day FROM windows WHERE route=? GROUP BY day HAVING COUNT(*) >= ?",
                (route, windows_per_day),
            ).fetchall()
            per_route.append({date.fromisoformat(r[0]) for r in rows})
    finally:
        con.close()
    return sorted(set.intersection(*per_route)) if per_route else []


# bands


@dataclass(frozen=True)
class Bands:
    lower_edges: tuple[int, ...]
    provisional: bool

    @property
    def k(self) -> int:
        return len(self.lower_edges)

    @property
    def labels(self) -> list[str]:
        out = [
            f"{lo}-{hi - 1}" for lo, hi in zip(self.lower_edges, self.lower_edges[1:], strict=False)
        ]
        return [*out, f"{self.lower_edges[-1]}+"]


def load_bands(path: Path) -> Bands:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    edges = tuple(int(e) for e in raw["lower_edges"])
    if list(edges) != sorted(set(edges)) or len(edges) < 2:
        raise ValueError("band edges must be increasing")
    return Bands(edges, bool(raw.get("provisional", False)))


def band_index(delay: np.ndarray, bands: Bands) -> np.ndarray:
    """Band of each delay in minutes. Early arrivals fall in the first band."""
    return np.digitize(np.asarray(delay, dtype=float), bands.lower_edges[1:]).astype(int)


def one_hot(index: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(index), k))
    out[np.arange(len(index)), index] = 1.0
    return out


# metrics


@dataclass
class Predictions:
    point: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    probs: np.ndarray | None = None


def date_sums(
    dates: np.ndarray, y: np.ndarray, pred: Predictions, bands: Bands
) -> tuple[np.ndarray, np.ndarray]:
    """Per service date sums for every metric. Returns (unique dates, sums array)."""
    n = len(y)
    err = pred.point - y
    cols: dict[str, np.ndarray] = {"n": np.ones(n), "abs": np.abs(err), "sq": err**2}
    if pred.lo is not None and pred.hi is not None:
        cols["n_int"] = np.ones(n)
        cols["cov"] = ((y >= pred.lo) & (y <= pred.hi)).astype(float)
        cols["width"] = pred.hi - pred.lo
    if pred.probs is not None:
        truth = one_hot(band_index(y, bands), bands.k)
        p_true = np.clip((pred.probs * truth).sum(axis=1), LOG_LOSS_EPS, 1.0)
        cols["n_band"] = np.ones(n)
        cols["brier"] = ((pred.probs - truth) ** 2).sum(axis=1)
        cols["ll"] = -np.log(p_true)
    frame = pd.DataFrame({k: cols.get(k, np.zeros(n)) for k in _SUMS})
    frame["date"] = dates
    grouped = frame.groupby("date", sort=True).sum()
    return grouped.index.to_numpy(), grouped[list(_SUMS)].to_numpy()


def metrics_from_sums(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from summed sums, shaped (..., len(_SUMS)). NaN where a metric has no rows."""
    t = {name: totals[..., i] for i, name in enumerate(_SUMS)}

    def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    return {
        "mae": ratio(t["abs"], t["n"]),
        "rmse": np.sqrt(ratio(t["sq"], t["n"])),
        "coverage": ratio(t["cov"], t["n_int"]),
        "width": ratio(t["width"], t["n_int"]),
        "brier": ratio(t["brier"], t["n_band"]),
        "log_loss": ratio(t["ll"], t["n_band"]),
    }


def score(
    dates: np.ndarray,
    y: np.ndarray,
    pred: Predictions,
    bands: Bands,
    *,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | None]]:
    """Each metric with a percentile bootstrap CI that resamples whole service dates."""
    _, sums = date_sums(dates, y, pred, bands)
    point = metrics_from_sums(sums.sum(axis=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(resamples, len(sums)))
    boot = metrics_from_sums(sums[idx].sum(axis=1))
    alpha = (1 - confidence) / 2 * 100
    out: dict[str, dict[str, float | None]] = {}
    for name in METRIC_NAMES:
        value = float(point[name])
        if np.isnan(value):
            out[name] = {"value": None, "ci_low": None, "ci_high": None}
            continue
        lo, hi = np.nanpercentile(boot[name], [alpha, 100 - alpha])
        out[name] = {"value": value, "ci_low": float(lo), "ci_high": float(hi)}
    out["n_rows"] = {"value": float(len(y)), "ci_low": None, "ci_high": None}
    out["n_dates"] = {"value": float(len(sums)), "ci_low": None, "ci_high": None}
    return out


# baselines


def timetable_baseline(df: pd.DataFrame, bands: Bands) -> Predictions:
    """The timetable is on time: delay 0, and probability 1 on the first band."""
    n = len(df)
    return Predictions(np.zeros(n), probs=one_hot(np.zeros(n, dtype=int), bands.k))


def persistence_baseline(df: pd.DataFrame, bands: Bands) -> Predictions:
    """Last known lateness carried to the destination, 0 if the train has not started.

    The band is the band of that point forecast, with probability 1.
    """
    point = df["last_stop_lateness_min"].fillna(0.0).to_numpy(dtype=float)
    return Predictions(point, probs=one_hot(band_index(point, bands), bands.k))


@dataclass
class HistoricalBaseline:
    """Rolling median point, with residual quantiles and band frequencies by route and hour.

    Fitted on the training block only. A (route, hour) with fewer than `min_rows` training rows
    falls back to the route, then to all training rows. Band frequencies carry an additive
    `smoothing` pseudo-count per band so an unseen band is not given probability zero.
    """

    bands: Bands
    min_rows: int = 10
    smoothing: float = 0.5
    median: dict[Any, float] = field(default_factory=dict)
    q_lo: dict[Any, float] = field(default_factory=dict)
    q_hi: dict[Any, float] = field(default_factory=dict)
    freq: dict[Any, np.ndarray] = field(default_factory=dict)

    @staticmethod
    def _levels(route: str, hour: int) -> list[Any]:
        return [(route, int(hour)), (route,), ()]

    def _groups(self, df: pd.DataFrame):
        """(key, rows) for the global, route and route-and-hour levels, thin groups skipped."""
        yield (), df
        for route, g in df.groupby("route_id"):
            if len(g) >= self.min_rows:
                yield (route,), g
        for key, g in df.groupby(["route_id", "dest_hour"]):
            if len(g) >= self.min_rows:
                yield (key[0], int(key[1])), g

    def fit(self, train: pd.DataFrame) -> "HistoricalBaseline":
        df = train[train["label_delay_min"].notna()].copy()
        for key, g in self._groups(df):
            self.median[key] = float(g["label_delay_min"].median())
        # residuals are taken against the point the baseline would issue: the rolling median,
        # or where there is none the training median of the group
        point = np.array(
            [
                self._lookup(self.median, r, h, v)
                for r, h, v in zip(
                    df["route_id"], df["dest_hour"], df["hist_median_28d"], strict=True
                )
            ],
            dtype=float,
        )
        df["_resid"] = df["label_delay_min"].to_numpy(dtype=float) - point
        df["_band"] = band_index(df["label_delay_min"].to_numpy(dtype=float), self.bands)
        for key, g in self._groups(df):
            self.q_lo[key], self.q_hi[key] = (
                float(x) for x in np.quantile(g["_resid"], [0.1, 0.9])
            )
            counts = np.bincount(g["_band"], minlength=self.bands.k)
            self.freq[key] = (counts + self.smoothing) / (
                counts.sum() + self.smoothing * self.bands.k
            )
        return self

    def _lookup(self, table: dict[Any, Any], route: str, hour: int, override: float | None = None):
        if override is not None and not pd.isna(override):
            return override
        for key in self._levels(route, hour):
            if key in table:
                return table[key]
        raise KeyError("baseline has not been fitted")

    def predict(self, df: pd.DataFrame) -> Predictions:
        routes, hours = df["route_id"].to_numpy(), df["dest_hour"].to_numpy()
        point = np.array(
            [
                self._lookup(self.median, r, h, v)
                for r, h, v in zip(routes, hours, df["hist_median_28d"], strict=True)
            ],
            dtype=float,
        )

        def pick(table: dict[Any, Any], r: str, h: int) -> Any:
            return next(table[k] for k in self._levels(r, h) if k in table)

        lo = point + np.array([pick(self.q_lo, r, h) for r, h in zip(routes, hours, strict=True)])
        hi = point + np.array([pick(self.q_hi, r, h) for r, h in zip(routes, hours, strict=True)])
        probs = np.vstack([pick(self.freq, r, h) for r, h in zip(routes, hours, strict=True)])
        return Predictions(point, lo, hi, probs)


# driver


def load_snapshots(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["service_date"] = pd.to_datetime(df["service_date"].astype(str)).dt.date
    return df


def evaluate(
    snapshots: pd.DataFrame, split: Split, bands: Bands, cfg: SplitConfig
) -> dict[str, Any]:
    """Score the three baselines on validation and test, per horizon."""
    block_of = {d: name for name, days in split.blocks().items() for d in days}
    df = snapshots[snapshots["service_date"].isin(block_of)].copy()
    df["block"] = df["service_date"].map(block_of)
    scored = df[df["label_delay_min"].notna()]
    train = scored[scored["block"] == "train"]  # the only rows any baseline is fitted on
    fitted = HistoricalBaseline(bands).fit(train)

    results: dict[str, Any] = {}
    for name in ("timetable", "persistence", "historical"):
        results[name] = {}
        for block in ("validation", "test"):
            part = scored[scored["block"] == block]
            results[name][block] = {}
            for horizon, h in part.groupby("horizon_min"):
                if name == "timetable":
                    pred = timetable_baseline(h, bands)
                elif name == "persistence":
                    pred = persistence_baseline(h, bands)
                else:
                    pred = fitted.predict(h)
                results[name][block][str(horizon)] = score(
                    h["service_date"].to_numpy(),
                    h["label_delay_min"].to_numpy(dtype=float),
                    pred,
                    bands,
                    resamples=cfg.resamples,
                    seed=cfg.seed,
                    confidence=cfg.confidence,
                )
    counts = {
        block: {
            str(h): int(n)
            for h, n in df[df["block"] == block].groupby("horizon_min").size().items()
        }
        for block in ("train", "validation", "test")
    }
    return {
        "split": {k: [d.isoformat() for d in v] for k, v in split.blocks().items()},
        "rows_per_block_and_horizon": counts,
        "scored_rows_excluding_null_labels": int(len(scored)),
        "bands": {
            "labels": bands.labels,
            "provisional": bands.provisional,
            "lower_edges": list(bands.lower_edges),
        },
        "bootstrap": {"resamples": cfg.resamples, "seed": cfg.seed, "confidence": cfg.confidence},
        "results": results,
    }


# reports


def _fmt(cell: dict[str, float | None], digits: int = 2, ci: bool = True) -> str:
    if cell["value"] is None:
        return "n/a"
    text = f"{cell['value']:.{digits}f}"
    if ci and cell["ci_low"] is not None:
        text += f" ({cell['ci_low']:.{digits}f} to {cell['ci_high']:.{digits}f})"
    return text


def write_reports(result: dict[str, Any], out_dir: Path) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    split = result["split"]
    boot = result["bootstrap"]
    horizons = sorted({int(h) for h in result["results"]["timetable"]["test"]}, reverse=True)
    lines = [
        "# Baseline scores",
        "",
        "Delay is the minutes between scheduled and actual arrival at the route destination. "
        f"Training covers {split['train'][0]} to {split['train'][-1]} "
        f"({len(split['train'])} days), validation {split['validation'][0]} to "
        f"{split['validation'][-1]}, and test {split['test'][0]} to {split['test'][-1]}, with one "
        "day dropped between blocks. Intervals are nominal 80% intervals. Brackets are "
        f"{int(boot['confidence'] * 100)}% bootstrap intervals from {boot['resamples']} "
        "resamples of whole service dates. Cancelled services have no delay and are not scored.",
        "",
        "The delay bands are " + ", ".join(result["bands"]["labels"]) + " minutes, and they are "
        "provisional until the per-operator rules are checked in Stage 5. Timetable and "
        "persistence give a hard band with probability 1, so their log loss is dominated by "
        f"the {LOG_LOSS_EPS:g} clip. The historical band frequencies are smoothed by half a count "
        "per band. Only the historical baseline has an interval.",
        "",
    ]
    names = {"timetable": "Timetable", "persistence": "Persistence", "historical": "Historical"}
    for h in horizons:
        lines += [
            f"## T-{h} minutes",
            "",
            "| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier "
            "| Log loss |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key, label in names.items():
            for block in ("validation", "test"):
                r = result["results"][key][block].get(str(h))
                if r is None:
                    continue
                lines.append(
                    f"| {label} | {block} | {int(r['n_rows']['value'])} | {_fmt(r['mae'])} | "
                    f"{_fmt(r['rmse'])} | {_fmt(r['coverage'], 3)} | "
                    f"{_fmt(r['width'], 1, False)} | {_fmt(r['brier'], 3)} | "
                    f"{_fmt(r['log_loss'], 2, False)} |"
                )
        by_val = min(
            names,
            key=lambda k: result["results"][k]["validation"][str(h)]["mae"]["value"],
        )
        test_cell = result["results"][by_val]["test"][str(h)]["mae"]
        by_brier = min(
            names, key=lambda k: result["results"][k]["validation"][str(h)]["brier"]["value"]
        )
        lines += [
            "",
            f"At T-{h} the bar to beat for the point forecast is {names[by_val].lower()}, the "
            f"lowest validation MAE, which scores {_fmt(test_cell)} minutes on test. For the "
            f"delay bands it is {names[by_brier].lower()}, the lowest validation Brier score. "
            "A model has to beat these on the test block with an interval that does not overlap "
            "theirs to count as an improvement.",
            "",
        ]
    (out_dir / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
