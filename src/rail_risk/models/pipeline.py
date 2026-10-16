"""Training pipeline: fit on train, calibrate on validation, score on test, log to MLflow."""

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rail_risk.evaluation import (
    Bands,
    HistoricalBaseline,
    PersistenceIntervalBaseline,
    Predictions,
    Split,
    SplitConfig,
    coverage_by_band,
    coverage_by_group,
    evaluate,
    paired_mae_difference,
    score,
    threshold_reliability,
    timetable_baseline,
)
from rail_risk.horizons import HORIZON_GRID, bucket_labels
from rail_risk.lateness import group_labels, lateness_groups
from rail_risk.models.bands import BandFit, fit_bands
from rail_risk.models.quantile import QuantileModel, fit_models


def load_train_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["min_train_days"] < 1:
        raise ValueError("min_train_days must be at least 1")
    return cfg


def guard(train_days: int, min_days: int, dirty: bool = False) -> str:
    """'reportable' only with enough history and a clean git tree. Anything else is smoke.

    A dirty tree means the logged commit does not describe the code that ran.
    """
    return "reportable" if train_days >= min_days and not dirty else "smoke"


def validation_halves(
    valid: pd.DataFrame, min_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """(tune, calibrate, split). With enough days validation is split in two by date, so the
    conformal offsets are not computed on the rows that tuned the model."""
    dates = sorted(valid["service_date"].unique())
    if len(dates) < min_days:
        return valid, valid, False
    cut = set(dates[: len(dates) // 2])
    first = valid["service_date"].isin(cut)
    return valid[first], valid[~first], True


@dataclass
class Fitted:
    quantile: QuantileModel
    bands: BandFit
    grid: dict[str, list[dict[str, Any]]]
    validation_split: bool
    baseline: Any = None  # HistoricalBaseline fitted on train, the serving fallback


def fit_all(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    bands: Bands,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> Fitted:
    """Everything fitted or tuned here sees train and validation only."""
    tune, calibrate, halved = validation_halves(valid, cfg["split_validation_min_days"])
    quantile, grid = fit_models(train, tune, calibrate, cfg, columns)
    band_fit = fit_bands(train, tune, calibrate, bands, cfg, columns)
    return Fitted(quantile, band_fit, grid, halved, HistoricalBaseline(bands).fit(train))


def model_predictions(
    fitted: Fitted, df: pd.DataFrame, mode: str = "mondrian", threshold: bool = True
) -> Predictions:
    point, lo, hi = fitted.quantile.predict(df, mode)
    probs = fitted.bands.predict(df) if threshold else fitted.bands.predict_pre_threshold(df)
    return Predictions(point, lo, hi, probs)


def score_model(
    fitted: Fitted,
    df: pd.DataFrame,
    bands: Bands,
    cfg: SplitConfig,
    mode: str = "mondrian",
    threshold: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kw = {"resamples": cfg.resamples, "seed": cfg.seed, "confidence": cfg.confidence}
    for horizon, h in df.groupby("horizon_min"):
        pred = model_predictions(fitted, h, mode, threshold)
        dates = h["service_date"].to_numpy()
        y = h["label_delay_min"].to_numpy(dtype=float)
        cell = score(dates, y, pred, bands, **kw)
        cell["coverage_by_band"] = coverage_by_band(dates, y, pred, bands, **kw)
        cell["coverage_by_group"] = coverage_by_group(
            dates, y, pred, bands, lateness_groups(h), group_labels(), **kw
        )
        cell["threshold_reliability"] = threshold_reliability(dates, y, pred.probs, bands, **kw)
        out[str(horizon)] = cell
    return out


def calibration_comparison(
    fitted: Fitted, tune: pd.DataFrame, calibrate: pd.DataFrame, nominal: float, halved: bool
) -> dict[str, Any]:
    """Coverage per route, horizon and lateness group for global and Mondrian CQR on validation.

    Reported on the calibration rows (where each method is calibrated, so close to nominal by
    construction) and, when validation was split, on the earlier tuning rows, which the offsets
    never saw. The test block is not used here.
    """
    labels = group_labels()
    sets = {"calibration_rows": calibrate}
    if halved:
        sets["tuning_rows"] = tune
    out: dict[str, Any] = {"nominal": nominal}
    for name, rows in sets.items():
        y = rows["label_delay_min"].to_numpy(dtype=float)
        groups = lateness_groups(rows)
        horizons = rows["horizon_min"].to_numpy()
        routes = rows["route_id"].to_numpy()
        table: dict[str, Any] = {}
        for mode in ("global", "mondrian"):
            _, lo, hi = fitted.quantile.predict(rows, mode)
            hit, width = (y >= lo) & (y <= hi), hi - lo
            table[mode] = {}
            for route in sorted(set(routes.tolist())):
                table[mode][route] = {}
                for h in sorted(set(horizons.tolist()), reverse=True):
                    table[mode][route][str(h)] = {}
                    for k, label in enumerate(labels):
                        m = (routes == route) & (horizons == h) & (groups == k)
                        table[mode][route][str(h)][label] = {
                            "coverage": float(hit[m].mean()) if m.any() else None,
                            "width": float(width[m].mean()) if m.any() else None,
                            "n": int(m.sum()),
                        }
        out[name] = table
    return out


def fit_for_inference(
    snapshots: pd.DataFrame,
    split: Split,
    bands: Bands,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> Fitted:
    """Fit on the train block and calibrate on validation only, as run_training does."""
    block_of = {d: n for n, days in split.blocks().items() for d in days}
    df = snapshots[snapshots["service_date"].isin(block_of)].copy()
    df["block"] = df["service_date"].map(block_of)
    scored = df[df["label_delay_min"].notna()]
    return fit_all(
        scored[scored["block"] == "train"], scored[scored["block"] == "validation"], bands, cfg,
        columns,
    )  # fmt: skip


def run_training(
    snapshots: pd.DataFrame,
    split: Split,
    bands: Bands,
    eval_cfg: SplitConfig,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> tuple[dict[str, Any], Fitted]:
    """Fit on train, calibrate and tune on validation, then score validation and test once.

    One model is fitted over every route in `snapshots`. It is scored, and its baselines are
    fitted, per route, so each route is compared with its own history.
    """
    block_of = {d: n for n, days in split.blocks().items() for d in days}
    df = snapshots[snapshots["service_date"].isin(block_of)].copy()
    df["block"] = df["service_date"].map(block_of)
    scored = df[df["label_delay_min"].notna()]
    train = scored[scored["block"] == "train"]
    valid = scored[scored["block"] == "validation"]
    fitted = fit_all(train, valid, bands, cfg, columns)  # the test block is not passed in
    tune, calibrate, halved = validation_halves(valid, cfg["split_validation_min_days"])

    routes = sorted(snapshots["route_id"].unique())
    per_route: dict[str, Any] = {}
    for route in routes:
        r_snaps = snapshots[snapshots["route_id"] == route]
        res = evaluate(r_snaps, split, bands, eval_cfg)
        r_scored = scored[scored["route_id"] == route]
        r_train = r_scored[r_scored["block"] == "train"]
        r_valid = r_scored[r_scored["block"] == "validation"]
        r_test = r_scored[r_scored["block"] == "test"]
        blocks = (("validation", r_valid), ("test", r_test))
        for key, kw in (
            ("model", {}),
            ("model_global_cqr", {"mode": "global"}),  # the previous global CQR, same fit
            ("model_pre_threshold_cal", {"threshold": False}),  # before threshold recalibration
        ):
            for name, block in blocks:
                res["results"].setdefault(key, {})[name] = score_model(
                    fitted, block, bands, eval_cfg, **kw
                )
        res["mae_vs_bar"] = _mae_vs_bar(res, fitted, r_train, r_test, bands, eval_cfg)
        per_route[route] = {
            k: res[k]
            for k in ("results", "mae_vs_bar", "rows_per_block_and_horizon",
                      "scored_rows_excluding_null_labels")
        }  # fmt: skip
        common = res
    fits = fitted.quantile.fits
    labels = bucket_labels(fitted.quantile.hedges)
    result: dict[str, Any] = {
        "split": common["split"],
        "bands": common["bands"],
        "bootstrap": common["bootstrap"],
        "routes": per_route,
        "train_days": len(split.train),
        "horizon_buckets": labels,
        "model": {
            "target_by_horizon_bucket": {
                labels[b]: t for b, t in sorted(fitted.quantile.choice.items())
            },
            "tune_pinball_by_target": {
                t: {labels[b]: v for b, v in sorted(per.items())}
                for t, per in fitted.quantile.tune_pinball.items()
            },
            "best_iteration": {t: f.best_iteration for t, f in fits.items()},
            "params": {
                t: {k: v for k, v in f.params.items() if k != "num_threads"}
                for t, f in fits.items()
            },
            "global_offsets": {
                t: {labels[b]: o for b, o in f.global_offsets.items()} for t, f in fits.items()
            },
            "conformal_offsets": {
                t: {f"{r}:{labels[b]}:{g}": o for (r, b, g), o in f.offsets.items()}
                for t, f in fits.items()
            },
            "group_merges": {t: f.merges for t, f in fits.items()},
            "calibration_comparison": calibration_comparison(
                fitted, tune, calibrate, cfg["nominal_coverage"], halved
            ),
            "band_calibration": fitted.bands.calibrator.name,
            "threshold_calibration": {
                "by_horizon_bucket": {
                    labels[b]: {"method": c.method, "tune_log_loss": c.tune_log_loss}
                    for b, c in sorted(fitted.bands.threshold.items())
                },
                "validation_split_in_two": fitted.validation_split,
            },
            "band_calibration_cv_log_loss": fitted.bands.cv_log_loss,
            "validation_split_in_two": fitted.validation_split,
            "grid": fitted.grid,
        },
    }
    return result, fitted


def _mae_vs_bar(
    result: dict[str, Any],
    fitted: Fitted,
    train: pd.DataFrame,
    test: pd.DataFrame,
    bands: Bands,
    cfg: SplitConfig,
) -> dict[str, Any]:
    """Model MAE minus the bar's MAE per test horizon, paired by service date.

    The bar is the baseline with the lowest validation MAE at that horizon, refitted on train.
    `result` and the frames are for one route.
    """
    historical = HistoricalBaseline(bands).fit(train)
    persistence = PersistenceIntervalBaseline(bands).fit(train)
    out: dict[str, Any] = {}
    for horizon, h in test.groupby("horizon_min"):
        bar = min(
            ("timetable", "persistence", "historical"),
            key=lambda k: result["results"][k]["validation"][str(horizon)]["mae"]["value"],
        )
        bar_point = {
            "timetable": lambda d: timetable_baseline(d, bands),
            "persistence": persistence.predict,
            "historical": historical.predict,
        }[bar](h).point
        model_point = fitted.quantile.predict(h)[0]
        diff = paired_mae_difference(
            h["service_date"].to_numpy(),
            h["label_delay_min"].to_numpy(dtype=float),
            model_point,
            bar_point,
            resamples=cfg.resamples,
            seed=cfg.seed,
            confidence=cfg.confidence,
        )
        out[str(horizon)] = {"bar": bar, **diff}
    return out


# tracking


def snapshot_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(path.rglob("*.parquet")):
        digest.update(str(f.relative_to(path)).encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    """(HEAD sha, working tree dirty). Untracked files count as dirty. Unknown sha is dirty."""

    def run(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["/usr/bin/git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    sha = run("rev-parse", "HEAD")
    return sha or "unknown", (not sha) or bool(run("status", "--porcelain"))


def flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for route, res in result["routes"].items():
        for block, horizons in res["results"]["model"].items():
            for h, cell in horizons.items():
                prefix = f"{route}_{block}_h{h}"
                for name in ("mae", "rmse", "coverage", "width", "brier", "log_loss"):
                    value = cell[name]["value"]
                    if value is not None:
                        out[f"{prefix}_{name}"] = float(value)
                for label, g in cell.get("coverage_by_group", {}).items():
                    if g["value"] is not None:
                        out[f"{prefix}_cov_late_{label.replace('+', 'plus')}"] = float(g["value"])
    return out


def log_to_mlflow(
    result: dict[str, Any],
    fitted: Fitted,
    *,
    tracking_dir: Path,
    tag: str,
    snapshot_path: Path,
    train_range: tuple[str, str],
    seed: int,
    git_sha: str,
    git_dirty: bool,
) -> str:
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    # sqlite backend; artifacts sit beside the database. Both live under the git-ignored mlruns/.
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}")
    name = "connection-risk"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name, artifact_location=(tracking_dir / "artifacts").resolve().as_uri()
        )
    mlflow.set_experiment(name)
    chosen = fitted.quantile.chosen
    with mlflow.start_run(run_name=f"{tag}-{train_range[1]}") as run:
        mlflow.log_params(
            {
                **{k: v for k, v in chosen.params.items() if k != "num_threads"},
                "target_by_horizon_bucket": json.dumps(result["model"]["target_by_horizon_bucket"]),
                "horizon_grid": json.dumps(list(HORIZON_GRID)),
                "best_iteration": json.dumps(result["model"]["best_iteration"]),
                "conformal": "mondrian_by_route_horizon_bucket_lateness",
                "band_calibration": result["model"]["band_calibration"],
                "threshold_calibration": json.dumps(
                    {
                        h: c["method"]
                        for h, c in result["model"]["threshold_calibration"][
                            "by_horizon_bucket"
                        ].items()
                    }
                ),
                "validation_split_in_two": result["model"]["validation_split_in_two"],
                "seed": seed,
                "train_days": result["train_days"],
            }
        )
        mlflow.log_metrics(flat_metrics(result))
        mlflow.set_tags(
            {
                "stage": tag,
                "git_commit": git_sha,
                "git_dirty": str(git_dirty).lower(),
                "snapshot_hash": snapshot_hash(snapshot_path),
                "train_start": train_range[0],
                "train_end": train_range[1],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for target, fit in fitted.quantile.fits.items():
                for q, booster in fit.boosters.items():
                    booster.save_model(str(root / f"quantile_{target}_{q}.txt"))
            fitted.bands.booster.save_model(str(root / "bands.txt"))
            (root / "results.json").write_text(json.dumps(result, indent=2, default=str))
            from rail_risk.models.bundle import ModelBundle, save_bundle

            save_bundle(
                ModelBundle(
                    fitted.quantile,
                    fitted.bands,
                    fitted.bands.bands,
                    horizons=HORIZON_GRID,
                    continuous_horizon=True,
                    meta={
                        "git_commit": git_sha,
                        "train_start": train_range[0],
                        "train_end": train_range[1],
                        "tag": tag,
                    },
                ),
                fitted.baseline,
                root,
            )
            mlflow.log_artifacts(tmp, artifact_path="model")
        return run.info.run_id


# report (only written for reportable runs)


def _cell(c: dict[str, Any], key: str, digits: int = 2, ci: bool = True) -> str:
    v = c[key]
    if v["value"] is None:
        return "n/a"
    text = f"{v['value']:.{digits}f}"
    if ci and v["ci_low"] is not None:
        text += f" ({v['ci_low']:.{digits}f} to {v['ci_high']:.{digits}f})"
    return text


def _coverage_table(
    result: dict[str, Any], h: int, key: str, labels: list[str], names: dict[str, str]
) -> list[str]:
    lines = ["| Method | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for method, label in names.items():
        by = result["results"][method]["test"][str(h)].get(key) or {}
        cells = []
        for name in labels:
            b = by.get(name, {})
            if b.get("value") is None:
                cells.append("n/a")
                continue
            cells.append(f"{b['value']:.2f} (n={int(b['n'])}, width {b['width']:.1f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _reliability_cell(c: dict[str, Any], digits: int = 3) -> str:
    if c["value"] is None:
        return "n/a"
    return f"{c['value']:.{digits}f} ({c['ci_low']:.{digits}f} to {c['ci_high']:.{digits}f})"


def _reliability_lines(result: dict[str, Any], h: int) -> list[str]:
    test = result["results"]
    lines = [
        "| Threshold | Share of rows at or above | Method | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for t in ("15", "30", "60"):
        for key, label in (
            ("model", "Model, after threshold calibration"),
            ("model_pre_threshold_cal", "Model, before threshold calibration"),
            ("historical", "Historical"),
        ):
            r = test[key]["test"][str(h)]["threshold_reliability"][t]
            lines.append(
                f"| {t} minutes | {r['base_rate']:.3f} | {label} | "
                f"{_reliability_cell(r['brier'])} | {_reliability_cell(r['ece'])} |"
            )
    for t in ("15", "30", "60"):
        m = test["model"]["test"][str(h)]["threshold_reliability"][t]["table"]
        b = test["historical"]["test"][str(h)]["threshold_reliability"][t]["table"]
        lines += [
            "",
            f"Reliability for P(at least {t} minutes late) at T-{h}, 10 equal-count bins from "
            "the lowest predicted probability to the highest:",
            "",
            "| Bin | Model n | Model predicted | Model observed | Historical predicted "
            "| Historical observed |",
            "|---|---|---|---|---|---|",
        ]
        for mb, hb in zip(m, b, strict=True):
            lines.append(
                f"| {mb['bin']} | {mb['n']} | {_reliability_cell(mb['predicted'], 2)} | "
                f"{_reliability_cell(mb['observed'], 2)} | "
                f"{_reliability_cell(hb['predicted'], 2)} | "
                f"{_reliability_cell(hb['observed'], 2)} |"
            )
    return lines


REPORT_HORIZONS = (120, 60, 15)


def write_model_report(result: dict[str, Any], path: Path, run_id: str, sha: str = "") -> None:
    from rail_risk.lateness import group_labels

    split = result["split"]
    names = {"timetable": "Timetable", "persistence": "Persistence", "historical": "Historical",
             "model_global_cqr": "Model, global CQR", "model": "Model, Mondrian CQR"}  # fmt: skip
    m = result["model"]
    routes = list(result["routes"])
    glabels = group_labels()
    halved = m["validation_split_in_two"]
    target_text = ", ".join(
        f"{label}: {target}" for label, target in m["target_by_horizon_bucket"].items()
    )
    threshold_text = ", ".join(
        f"{label} {c['method']}"
        for label, c in m["threshold_calibration"]["by_horizon_bucket"].items()
    )
    all_days = [*split["train"], *split["validation"], *split["test"]]
    lines = [
        "# Model results",
        "",
        f"MLflow run {run_id}" + (f", commit {sha}" if sha else "") + ". "
        f"One model was trained on {len(split['train'])} complete days "
        f"({split['train'][0]} to {split['train'][-1]}) of {', '.join(routes)}, tuned and "
        f"calibrated on {len(split['validation'])} validation days "
        f"({split['validation'][0]} to {split['validation'][-1]})"
        + (", the earlier half tuning and the later half calibrating" if halved else "")
        + f", and scored once on {len(split['test'])} test days "
        f"({split['test'][0]} to {split['test'][-1]}). Brackets are 95% bootstrap intervals over "
        "service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route "
        "destination, and cancelled services are not scored. Each route is scored against "
        "baselines fitted on that route's own training days.",
        "",
        "The model takes the minutes to scheduled arrival as a number, trained on snapshots at "
        f"{', '.join(str(h) for h in HORIZON_GRID)} minutes before arrival, with the route and "
        "per-route 28 day history as features. The interval comes from LightGBM quantile models "
        "at 0.1, 0.5 and 0.9, widened by a conformal offset computed separately for each route, "
        f"horizon bucket ({', '.join(result['horizon_buckets'])} minutes) and observed lateness "
        "group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness "
        "group uses only what is known at T, and a group with fewer than 30 calibration rows is "
        "merged into its neighbour and logged. The target per horizon bucket was chosen by "
        "tuning pinball loss between the direct delay and the delay minus the lateness at T: "
        f"{target_text}. Band probabilities come from a LightGBM band model with "
        f"{m['band_calibration']} calibration. P(at least 15) and P(at least 30) are then "
        "recalibrated separately per horizon bucket, choosing among isotonic regression, Platt "
        "scaling and no recalibration by log loss on the tuning rows, with the methods fitted on "
        f"the calibration rows. The choice was {threshold_text}. The band probabilities are "
        "rebuilt from the recalibrated exceedances so that they stay monotone and sum to 1. "
        "The delay bands are provisional.",
        "",
        "Two kinds of check are reported. The conformal guarantee is coverage conditional on the "
        "state known at T, so coverage by observed lateness is the interval check. For passengers "
        "and Delay Repay, what matters is whether the probabilities of reaching 15, 30 and 60 "
        "minutes late are calibrated, so threshold reliability is the passenger facing check. "
        "Coverage by true delay band is shown for description only. Results for every grid "
        "horizon are in the run's results.json; this page shows 120, 60 and 15 minutes.",
        "",
    ]
    limits_rows = []
    for route in routes:
        res = result["routes"][route]
        tests = res["results"]["model"]["test"]
        horizons = [h for h in REPORT_HORIZONS if str(h) in tests]
        for h in horizons:
            lines += [
                f"## {route}, T-{h} minutes, test block",
                "",
                "| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |",
                "|---|---|---|---|---|---|",
            ]
            for key, label in names.items():
                c = res["results"][key]["test"][str(h)]
                lines.append(
                    f"| {label} | {_cell(c, 'mae')} | {_cell(c, 'coverage', 3)} | "
                    f"{_cell(c, 'width', 1, False)} | {_cell(c, 'brier', 3)} | "
                    f"{_cell(c, 'log_loss', 2, False)} |"
                )
            d = res["mae_vs_bar"][str(h)]
            verdict = (
                "The interval excludes zero."
                if d["ci_low"] > 0 or d["ci_high"] < 0
                else "The interval includes zero, so the difference is not statistically "
                "significant."
            )
            lines += [
                "",
                f"Against the bar, {d['bar']} (lowest validation MAE), the model's MAE is "
                f"{d['value']:+.2f} minutes ({d['ci_low']:+.2f} to {d['ci_high']:+.2f}, paired "
                f"bootstrap over service dates). {verdict}",
                "",
                "Interval coverage by observed lateness at T, which is what the conformal "
                "guarantee covers:",
                "",
                *_coverage_table(res, h, "coverage_by_group", glabels, names),
                "",
                "Threshold reliability, derived from the band probabilities. Brier is the binary "
                "Brier score and ECE the expected calibration error over 10 equal-count bins. "
                "Historical band frequencies are the bar.",
                "",
                *_reliability_lines(res, h),
                "",
                "Interval coverage by true delay band, descriptive only. A band is selected by "
                "the outcome that the interval was built before knowing, so it cannot reach the "
                "nominal level for any forecast that is not perfect.",
                "",
                *_coverage_table(res, h, "coverage_by_band", result["bands"]["labels"], names),
                "",
            ]
        cal = m["calibration_comparison"]["calibration_rows"]["mondrian"][route]
        limits_rows.append(
            (
                route,
                {h: cal[str(h)]["15+"]["n"] for h in horizons},
                {h: int(tests[str(h)]["coverage_by_group"]["15+"]["n"]) for h in horizons},
                [
                    f"T-{h}"
                    for h in horizons
                    if res["mae_vs_bar"][str(h)]["ci_low"]
                    <= 0
                    <= res["mae_vs_bar"][str(h)]["ci_high"]
                ],
            )
        )
    lines += [
        "## Limitations",
        "",
        f"These results cover {len(routes)} routes: {', '.join(routes)}. BHM to MAN and "
        "AST to EUS via BHM are not in the model, so nothing here says it transfers to them. "
        f"The history runs from {min(all_days)} to {max(all_days)}, which is summer and the "
        "first weeks of autumn, so there is no winter, leaf-fall or storm season, and a seasonal "
        "shift would change both the delays and the calibration. The 15 or more minutes late "
        "lateness group is small. ",
    ]
    for route, n15, t15, sig in limits_rows:
        lines[-1] += (
            f"On {route} it held "
            + ", ".join(f"{n15[h]} calibration rows at T-{h}" for h in n15)
            + " and "
            + ", ".join(f"{t15[h]} test rows at T-{h}" for h in t15)
            + ". "
            + (
                "The model's MAE gain over the bar is not statistically significant at "
                + ", ".join(sig)
                + ". "
                if sig
                else "The model's MAE gain over the bar is statistically significant at "
                "every shown horizon. "
            )
        )
    lines[-1] += (
        "Offsets for a small group are noisy and are merged into a neighbour below 30 rows. "
        f"The test block is {len(split['test'])} days, so the bootstrap intervals are wide, and "
        "the 60 minute and longer delays are too rare for the 60 minute threshold table to say "
        "much. The conformal guarantee holds on average within each cell if the days are "
        "exchangeable, which a time split only approximates. The delay bands are provisional "
        "until the per-operator compensation rules are checked."
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
