"""Stage 3 commands: train (tune, fit, gate on validation) and evaluate (score the test block once).

`train` never loads the test block's features. `evaluate` refuses to run twice for a model: the
test block is scored once, and the marker file records which run did it.
"""

import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from rail_risk.baselines import models, scoring
from rail_risk.baselines.run import assert_train_pool_only
from rail_risk.baselines.target_check import assert_targets_ok
from rail_risk.ingest.darwin_pport import load_stage1_config, split_for
from rail_risk.logging import get_logger
from rail_risk.network_model import dataset as D
from rail_risk.network_model import train as T
from rail_risk.network_model.evaluate import paired_table

TEST_MARKER = T.REPORT_DIR / "test_scored.json"
HEADLINE = (15, 60)
BASELINES = ("persistence", "shrunk_persistence", "historical")


def git_state() -> tuple[str, bool]:
    def run(*a: str) -> str:
        return subprocess.check_output(["/usr/bin/env", "git", *a], text=True).strip()  # noqa: S603

    try:
        return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _tracking() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlruns/mlflow.db').resolve()}")
    name = "network_model"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(Path("mlruns/artifacts").resolve()))
    mlflow.set_experiment(name)


def prepare_frames(include_test: bool, cold: bool = False) -> tuple[dict, set[str], object, dict]:
    split, held = D.load_split()
    days = [*split.train, *split.validation, *(split.test if include_test else [])]
    assert_train_pool_only(days)
    fastest = D.ensure_fastest(
        [*split.train, *split.validation, *split.test], list(split.train), cold=cold
    )
    df = D.prepare(D.load(days), fastest)
    rows = D.split_rows(df, split, held)
    assert_targets_ok(None, {k: v for k, v in rows.items() if k != "train_all_routes"})
    assert not rows["train"]["route"].isin(held).any(), "a held-out route reached the fit"
    cfg = load_stage1_config()
    for name in ("train", "validation") + (("test",) if include_test else ()):
        bad = {str(d) for d in rows[name]["ssd"].unique() if split_for(d, cfg) != cfg.default_split}
        assert not bad, f"non train_pool dates in {name}: {sorted(bad)[:3]}"
    return rows, held, split, {"fastest_pairs": len(fastest)}


def train_command(
    config: Path = T.CONFIG,
    mode: str = "history",
    out_dir: Path = T.REPORT_DIR,
    grid: dict | None = None,
    drop: list[str] | None = None,
    warm_base: bool = False,
) -> dict:
    """mode 'history': the Stage 2 base and slack table use every route's own train-date history
    (held-out routes included). mode 'cold': held-out routes get no route-specific history, the
    base is fitted on the training routes only and the slack table leaves the held-out services
    out. `grid` overrides the tuning grid (the widened-grid check)."""
    log = get_logger("network_train")
    if mode not in ("history", "cold"):
        raise ValueError(f"unknown mode {mode!r}")
    cfg = T.load_config(config)
    if grid is not None:
        cfg = {**cfg, "grid": grid}
    if drop:
        cfg = {**cfg, "drop": list(drop)}
    cfg = {**cfg, "warm_base": warm_base}
    cold = mode == "cold"
    rows, held, split, extra = prepare_frames(include_test=False, cold=cold)
    train, val = rows["train"], rows["validation"]
    base_rows = train if cold else rows["train_all_routes"]
    head, dirty = git_state()
    log.info("tuning", train_rows=len(train), folds=cfg["folds"])
    tuned = T.tune(train, cfg, cfg["option"])
    base_fit = models.fit(base_rows)  # history mode: the Stage 2 bar's fit
    base_tr = D.base_point(base_fit, train, cfg.get("warm_base", False))
    val_train_routes = val[~val["route"].isin(held)]
    model = T.fit_final(train, val_train_routes, cfg, tuned, base_tr)
    table = T.gate(model, val, held, base_rows, cfg)
    passed = bool(
        table[table["baseline"].isin(["persistence", "shrunk_persistence"])]["ci_below_zero"].all()
    )
    _tracking()
    with mlflow.start_run(run_name=f"network-{cfg['option']}-{mode}-{head}") as run:
        mlflow.log_params(
            {
                "option": cfg["option"], "seed": cfg["seed"], "n_features": len(model.features),
                "median_params": str(tuned["median"]["params"]),
                "median_rounds": tuned["median"]["rounds"],
                "classifier_params": str(tuned["classifier"]["params"]),
                "classifier_rounds": tuned["classifier"]["rounds"],
                "train_rows": len(train), "train_routes": train["route"].nunique(),
                "train_days": f"{split.train[0]}..{split.train[-1]}",
                "validation_days": f"{split.validation[0]}..{split.validation[-1]}",
            }
        )  # fmt: skip
        mlflow.set_tags(
            {
                "git_head": head, "git_dirty": str(dirty), "gate_passed": str(passed),
                "mode": mode, "dropped": ",".join(drop or []), "warm_base": str(warm_base),
                "features": ",".join(model.features),
                "train_hash": D.data_hash(train, ["rid", "horizon_min", "target_min"]),
                "validation_hash": D.data_hash(val, ["rid", "horizon_min", "target_min"]),
                "held_out_routes": ",".join(sorted(held)),
            }
        )  # fmt: skip
        for r in table.itertuples():
            tag = f"gate_{r.baseline}_{r.horizon_min}"
            mlflow.log_metrics({f"{tag}_diff": r.diff, f"{tag}_hi": r.hi, f"{tag}_lo": r.lo})
        run_id = run.info.run_id
        path = T.MODEL_DIR / run_id
        model.save(path)
        mlflow.log_artifacts(str(path), artifact_path="model")
    out = {
        "run_id": run_id, "git_head": head, "git_dirty": dirty, "gate_passed": passed,
        "mode": mode, "tuned": {k: tuned[k] for k in ("median", "classifier")},
        "tune_days": tuned["tune_days"], "folds": tuned["folds"], "train_rows": len(train),
        "gate": table.round(4).to_dict("records"),
    }  # fmt: skip
    T.write_json(out_dir / "validation_gate.json", out)
    T.write_json(out_dir / "tuning.json", tuned)
    log.info("trained", run_id=run_id, gate_passed=passed)
    return out


def _score(frame: pd.DataFrame, preds: dict[str, pd.DataFrame], held: set[str], cfg: dict):
    stats = {m: scoring.row_stats(frame.assign(schedule_known=True), p) for m, p in preds.items()}
    return scoring.score_all(
        stats, held, sorted(frame["route"].unique()), cfg["resamples"], cfg["seed"], 0.95
    )


def evaluate_command(run_id: str, config: Path = T.CONFIG) -> dict:
    if TEST_MARKER.exists():
        raise RuntimeError(f"the test block was already scored: {TEST_MARKER.read_text()[:200]}")
    cfg = T.load_config(config)
    gate_file = T.REPORT_DIR / "validation_gate.json"
    gate = json.loads(gate_file.read_text())
    if gate["run_id"] != run_id or not gate["gate_passed"]:
        raise RuntimeError("the validation gate has not passed for this run; test stays unscored")
    rows, held, split, _ = prepare_frames(include_test=True)
    test = rows["test"]
    hold = test[test["route"].isin(held)]
    model = T.Booster3.load(T.MODEL_DIR / run_id)
    base = models.predict(models.fit(rows["train_all_routes"]), hold)
    mp = model.predict(hold, base["shrunk_persistence"]["point"].to_numpy())
    preds = {"model": mp[["point", "lo", "hi", "p_late"]], **{k: v for k, v in base.items()}}
    preds.pop("darwin_forecast", None)
    score = _score(hold, preds, held, cfg)
    paired = paired_table(
        hold, mp["point"].to_numpy(),
        {k: base[k]["point"].to_numpy() for k in BASELINES},
        HEADLINE, cfg["resamples"], cfg["seed"],
    )  # fmt: skip
    T.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    score.to_csv(T.REPORT_DIR / "stage3_test_scores.csv", index=False)
    paired.to_csv(T.REPORT_DIR / "stage3_test_paired.csv", index=False)
    hold = hold.assign(pred=mp["point"].to_numpy(), lo=mp["lo"].to_numpy(), hi=mp["hi"].to_numpy())
    hold[
        ["ssd", "rid", "route", "toc", "horizon_min", "target_min", "pred", "lo", "hi"]
    ].to_parquet(T.MODEL_DIR / run_id / "test_predictions.parquet", index=False)
    head, dirty = git_state()
    _tracking()
    with mlflow.start_run(run_id=run_id):
        for r in paired.itertuples():
            tag = f"test_{r.baseline}_{r.horizon_min}"
            mlflow.log_metrics({f"{tag}_diff": r.diff, f"{tag}_hi": r.hi, f"{tag}_lo": r.lo})
        mlflow.set_tag("test_scored_at_head", f"{head}{'+dirty' if dirty else ''}")
    marker = {"run_id": run_id, "git_head": head, "dirty": dirty, "rows": len(hold)}
    TEST_MARKER.write_text(json.dumps(marker) + "\n")
    return {"paired": paired, "score": score, "marker": marker}
