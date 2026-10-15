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
    Predictions,
    Split,
    SplitConfig,
    coverage_by_band,
    coverage_by_group,
    evaluate,
    score,
)
from rail_risk.lateness import group_labels, lateness_groups
from rail_risk.models.bands import BandFit, fit_bands
from rail_risk.models.quantile import QuantileModel, fit_models


def load_train_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["min_train_days"] < 1:
        raise ValueError("min_train_days must be at least 1")
    return cfg


def guard(train_days: int, min_days: int) -> str:
    """'reportable' only with enough history. Anything less is a smoke test."""
    return "reportable" if train_days >= min_days else "smoke"


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
    return Fitted(quantile, band_fit, grid, halved)


def model_predictions(fitted: Fitted, df: pd.DataFrame, mode: str = "mondrian") -> Predictions:
    point, lo, hi = fitted.quantile.predict(df, mode)
    return Predictions(point, lo, hi, fitted.bands.predict(df))


def score_model(
    fitted: Fitted, df: pd.DataFrame, bands: Bands, cfg: SplitConfig, mode: str = "mondrian"
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kw = {"resamples": cfg.resamples, "seed": cfg.seed, "confidence": cfg.confidence}
    for horizon, h in df.groupby("horizon_min"):
        pred = model_predictions(fitted, h, mode)
        dates = h["service_date"].to_numpy()
        y = h["label_delay_min"].to_numpy(dtype=float)
        cell = score(dates, y, pred, bands, **kw)
        cell["coverage_by_band"] = coverage_by_band(dates, y, pred, bands, **kw)
        cell["coverage_by_group"] = coverage_by_group(
            dates, y, pred, bands, lateness_groups(h), group_labels(), **kw
        )
        out[str(horizon)] = cell
    return out


def calibration_comparison(
    fitted: Fitted, tune: pd.DataFrame, calibrate: pd.DataFrame, nominal: float, halved: bool
) -> dict[str, Any]:
    """Coverage per horizon and lateness group for global and Mondrian CQR on validation.

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
        table: dict[str, Any] = {}
        for mode in ("global", "mondrian"):
            _, lo, hi = fitted.quantile.predict(rows, mode)
            hit, width = (y >= lo) & (y <= hi), hi - lo
            table[mode] = {}
            for h in sorted(set(horizons.tolist()), reverse=True):
                table[mode][str(h)] = {}
                for k, label in enumerate(labels):
                    m = (horizons == h) & (groups == k)
                    table[mode][str(h)][label] = {
                        "coverage": float(hit[m].mean()) if m.any() else None,
                        "width": float(width[m].mean()) if m.any() else None,
                        "n": int(m.sum()),
                    }
        out[name] = table
    return out


def run_training(
    snapshots: pd.DataFrame,
    split: Split,
    bands: Bands,
    eval_cfg: SplitConfig,
    cfg: dict[str, Any],
    columns: list[str] | None = None,
) -> tuple[dict[str, Any], Fitted]:
    """Fit on train, calibrate and tune on validation, then score validation and test once."""
    block_of = {d: n for n, days in split.blocks().items() for d in days}
    df = snapshots[snapshots["service_date"].isin(block_of)].copy()
    df["block"] = df["service_date"].map(block_of)
    scored = df[df["label_delay_min"].notna()]
    train = scored[scored["block"] == "train"]
    valid = scored[scored["block"] == "validation"]
    fitted = fit_all(train, valid, bands, cfg, columns)  # the test block is not passed in
    tune, calibrate, halved = validation_halves(valid, cfg["split_validation_min_days"])

    result = evaluate(snapshots, split, bands, eval_cfg)
    test = scored[scored["block"] == "test"]
    for name, block in (("validation", valid), ("test", test)):
        result["results"].setdefault("model", {})[name] = score_model(
            fitted, block, bands, eval_cfg
        )
    # the previous global CQR as a comparison column, same fitted models and calibration rows
    for name, block in (("validation", valid), ("test", test)):
        result["results"].setdefault("model_global_cqr", {})[name] = score_model(
            fitted, block, bands, eval_cfg, mode="global"
        )
    result["train_days"] = len(split.train)
    fits = fitted.quantile.fits
    result["model"] = {
        "target_by_horizon": {str(h): t for h, t in fitted.quantile.choice.items()},
        "tune_pinball_by_target": {
            t: {str(h): v for h, v in per.items()}
            for t, per in fitted.quantile.tune_pinball.items()
        },
        "best_iteration": {t: f.best_iteration for t, f in fits.items()},
        "params": {
            t: {k: v for k, v in f.params.items() if k != "num_threads"} for t, f in fits.items()
        },
        "global_offsets": {
            t: {str(h): o for h, o in f.global_offsets.items()} for t, f in fits.items()
        },
        "conformal_offsets": {
            t: {f"{h}:{g}": o for (h, g), o in f.offsets.items()} for t, f in fits.items()
        },
        "group_merges": {t: f.merges for t, f in fits.items()},
        "calibration_comparison": calibration_comparison(
            fitted, tune, calibrate, cfg["nominal_coverage"], halved
        ),
        "band_calibration": fitted.bands.calibrator.name,
        "band_calibration_cv_log_loss": fitted.bands.cv_log_loss,
        "validation_split_in_two": fitted.validation_split,
        "grid": fitted.grid,
    }
    return result, fitted


# tracking


def snapshot_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(path.rglob("*.parquet")):
        digest.update(str(f.relative_to(path)).encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


def git_commit() -> str:
    def run(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            ["/usr/bin/git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = run("rev-parse", "HEAD") or "unknown"
    return commit + ("+dirty" if run("status", "--porcelain") else "")


def flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for block, horizons in result["results"]["model"].items():
        for h, cell in horizons.items():
            for name in ("mae", "rmse", "coverage", "width", "brier", "log_loss"):
                value = cell[name]["value"]
                if value is not None:
                    out[f"{block}_h{h}_{name}"] = float(value)
            for label, g in cell.get("coverage_by_group", {}).items():
                if g["value"] is not None:
                    out[f"{block}_h{h}_cov_late_{label.replace('+', 'plus')}"] = float(g["value"])
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
                "target_by_horizon": json.dumps(result["model"]["target_by_horizon"]),
                "best_iteration": json.dumps(result["model"]["best_iteration"]),
                "conformal": "mondrian_by_lateness",
                "band_calibration": result["model"]["band_calibration"],
                "validation_split_in_two": result["model"]["validation_split_in_two"],
                "seed": seed,
                "train_days": result["train_days"],
            }
        )
        mlflow.log_metrics(flat_metrics(result))
        mlflow.set_tags(
            {
                "stage": tag,
                "git_commit": git_commit(),
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


FLAG_BELOW = 0.70
FLAG_MIN_ROWS = 30


def flagged_bands(result: dict[str, Any], method: str = "model") -> list[dict[str, Any]]:
    """Test delay bands whose interval coverage is below 0.70 on at least 30 rows."""
    flags = []
    for h, cell in result["results"][method]["test"].items():
        for band, b in (cell.get("coverage_by_band") or {}).items():
            if b["value"] is not None and b["n"] >= FLAG_MIN_ROWS and b["value"] < FLAG_BELOW:
                flags.append({"horizon": int(h), "band": band, "coverage": b["value"],
                              "n": int(b["n"])})  # fmt: skip
    return sorted(flags, key=lambda f: (-f["horizon"], f["band"]))


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
            note = f"{b['value']:.2f} (n={int(b['n'])}, width {b['width']:.1f})"
            cells.append(note)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def write_model_report(result: dict[str, Any], path: Path, run_id: str) -> None:
    from rail_risk.lateness import group_labels

    split = result["split"]
    names = {"timetable": "Timetable", "persistence": "Persistence", "historical": "Historical",
             "model_global_cqr": "Model, global CQR", "model": "Model, Mondrian CQR"}  # fmt: skip
    m = result["model"]
    tests = result["results"]["model"]["test"]
    horizons = sorted({int(h) for h in tests}, reverse=True)
    glabels = group_labels()
    target_text = ", ".join(f"T-{h}: {m['target_by_horizon'][str(h)]}" for h in horizons)
    lines = [
        "# Model results",
        "",
        f"MLflow run {run_id}. The model was trained on {len(split['train'])} complete days "
        f"({split['train'][0]} to {split['train'][-1]}), tuned and calibrated on validation "
        f"({split['validation'][0]} to {split['validation'][-1]}), and scored once on test "
        f"({split['test'][0]} to {split['test'][-1]}). Brackets are 95% bootstrap intervals over "
        "service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route "
        "destination, and cancelled services are not scored.",
        "",
        "The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9 with the horizon as "
        "a feature, widened by a conformal offset computed separately for each horizon and each "
        "observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). "
        "The lateness group uses only what is known at T. The target per horizon was chosen by "
        f"validation pinball loss between the direct delay and the delay minus the lateness at T: "
        f"{target_text}. Band probabilities use {m['band_calibration']} calibration. The delay "
        "bands are provisional.",
        "",
    ]
    for h in horizons:
        lines += [
            f"## T-{h} minutes, test block",
            "",
            "| Method | MAE | 80% coverage | Mean width | Brier | Log loss |",
            "|---|---|---|---|---|---|",
        ]
        for key, label in names.items():
            c = result["results"][key]["test"][str(h)]
            lines.append(
                f"| {label} | {_cell(c, 'mae')} | {_cell(c, 'coverage', 3)} | "
                f"{_cell(c, 'width', 1, False)} | {_cell(c, 'brier', 3)} | "
                f"{_cell(c, 'log_loss', 2, False)} |"
            )
        lines += [
            "",
            "Coverage by observed lateness at T, which is what the conformal guarantee covers:",
            "",
            *_coverage_table(result, h, "coverage_by_group", glabels, names),
            "",
            "Coverage by true delay band, which is what passengers feel:",
            "",
            *_coverage_table(result, h, "coverage_by_band", result["bands"]["labels"], names),
            "",
        ]
    flags = flagged_bands(result)
    if flags:
        lines.append(
            "Delay bands where the model interval covers under 70% of at least 30 test rows: "
            + "; ".join(
                f"T-{f['horizon']} band {f['band']} at {f['coverage']:.2f} (n={f['n']})"
                for f in flags
            )
            + "."
        )
    else:
        lines.append(
            "No delay band with at least 30 test rows has model interval coverage below 0.70."
        )
    lines += [
        "",
        "## Limitations",
        "",
        "These results cover one route, BHM to EUS. BHM to MAN and AST to EUS via BHM have not "
        "been ingested yet, so nothing here says the model transfers to them. The history runs "
        "from mid July to mid October, so it covers summer and the start of autumn but no winter, "
        "leaf-fall or storm season, and a seasonal shift would change both the delays and the "
        "calibration. The test block is seven days, so the bootstrap intervals are wide and the "
        "late bands hold few rows; a band with fewer than 30 rows is not flagged and should not "
        "be read as covered. The conformal guarantee holds on average within each observed "
        "lateness group if the days are exchangeable, which a time split only approximates, and "
        "it does not hold within the true delay bands. The delay bands are provisional until the "
        "per-operator compensation rules are checked.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
