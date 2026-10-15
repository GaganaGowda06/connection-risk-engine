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
    evaluate,
    score,
)
from rail_risk.models.bands import BandFit, fit_bands
from rail_risk.models.quantile import QuantileFit, fit_quantile


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
    quantile: QuantileFit
    bands: BandFit
    grid: list[dict[str, Any]]
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
    quantile, grid = fit_quantile(train, tune, calibrate, cfg, columns)
    band_fit = fit_bands(train, tune, calibrate, bands, cfg, columns)
    return Fitted(quantile, band_fit, grid, halved)


def model_predictions(fitted: Fitted, df: pd.DataFrame) -> Predictions:
    point, lo, hi = fitted.quantile.predict(df)
    return Predictions(point, lo, hi, fitted.bands.predict(df))


def score_model(fitted: Fitted, df: pd.DataFrame, bands: Bands, cfg: SplitConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kw = {"resamples": cfg.resamples, "seed": cfg.seed, "confidence": cfg.confidence}
    for horizon, h in df.groupby("horizon_min"):
        pred = model_predictions(fitted, h)
        dates = h["service_date"].to_numpy()
        y = h["label_delay_min"].to_numpy(dtype=float)
        cell = score(dates, y, pred, bands, **kw)
        cell["coverage_by_band"] = coverage_by_band(dates, y, pred, bands, **kw)
        out[str(horizon)] = cell
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

    result = evaluate(snapshots, split, bands, eval_cfg)
    test = scored[scored["block"] == "test"]
    for name, block in (("validation", valid), ("test", test)):
        result["results"].setdefault("model", {})[name] = score_model(
            fitted, block, bands, eval_cfg
        )
    result["train_days"] = len(split.train)
    result["model"] = {
        "best_iteration": fitted.quantile.best_iteration,
        "params": {k: v for k, v in fitted.quantile.params.items() if k != "num_threads"},
        "conformal_offsets": {str(h): o for h, o in fitted.quantile.offsets.items()},
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

    # the file store is in maintenance mode in MLflow 3 and refuses to open unless allowed. It is
    # what the project asked for; migrate-filestore moves it to sqlite later if needed.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(f"file:{tracking_dir.resolve()}")
    mlflow.set_experiment("connection-risk")
    with mlflow.start_run(run_name=f"{tag}-{train_range[1]}") as run:
        mlflow.log_params(
            {
                **{k: v for k, v in result["model"]["params"].items()},
                "best_iteration": result["model"]["best_iteration"],
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
            for q, booster in fitted.quantile.boosters.items():
                booster.save_model(str(root / f"quantile_{q}.txt"))
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


def write_model_report(result: dict[str, Any], path: Path, run_id: str) -> None:
    split = result["split"]
    names = {"timetable": "Timetable", "persistence": "Persistence", "historical": "Historical",
             "model": "Model"}  # fmt: skip
    lines = [
        "# Model results",
        "",
        f"MLflow run {run_id}. Trained on {len(split['train'])} days "
        f"({split['train'][0]} to {split['train'][-1]}), calibrated on validation "
        f"({split['validation'][0]} to {split['validation'][-1]}), and scored once on test "
        f"({split['test'][0]} to {split['test'][-1]}). Brackets are 95% bootstrap intervals over "
        "service dates. Band calibration: "
        f"{result['model']['band_calibration']}. Delay bands are provisional.",
        "",
    ]
    horizons = sorted({int(h) for h in result["results"]["model"]["test"]}, reverse=True)
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
            "Coverage by true delay band:",
            "",
            "| Method | " + " | ".join(result["bands"]["labels"]) + " |",
            "|---|" + "---|" * len(result["bands"]["labels"]),
        ]
        for key, label in names.items():
            by = result["results"][key]["test"][str(h)].get("coverage_by_band") or {}
            if not by:
                continue
            cells = []
            for band in result["bands"]["labels"]:
                b = by.get(band, {})
                cells.append(
                    "n/a" if b.get("value") is None else f"{b['value']:.2f} (n={int(b['n'])})"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
