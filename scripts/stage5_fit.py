#!/usr/bin/env python3
"""Stage 5 final fit (run from a clean commit): the chosen option on the train block, plus the three
leave-hub-out models, logged to MLflow (experiment connection_model). Reads train and validation
pairs only. The option comes from the validation comparison (reports/connections/stage5).

    uv run python scripts/stage5_fit.py
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np

from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import data as DD
from rail_risk.connections import pipeline as P
from rail_risk.evaluation import _bin_stats
from rail_risk.network_model.run import git_state

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")


def val_cells(model, pick, va, bar, masks):
    p = model.predict(va)[f"{pick}_cal"].to_numpy()
    y = va["miss"].to_numpy(float)
    out = {}
    for name, m in masks.items():
        b, e, _ = _bin_stats(p[m], y[m], 10)
        bb = float(np.mean((BL.predict(bar, va[m]) - y[m]) ** 2))
        out[name] = {"n": int(m.sum()), "brier": b, "ece": e, "gap_bucket_brier": bb}
    return out


def log_cells(prefix, cells):
    for k, v in cells.items():
        tag = f"{prefix}_{k}"
        mlflow.log_metrics(
            {f"{tag}_brier": v["brier"], f"{tag}_ece": v["ece"],
             f"{tag}_gap_bucket_brier": v["gap_bucket_brier"]}
        )  # fmt: skip


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    head, dirty = git_state()
    cfg = B.load_config()
    pick = json.loads((OUT / "options_validation.json").read_text())["summary"]["pick"]
    eta = json.loads((OUT / "options_online_validation.json").read_text())["eta"]
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    va = DD.labelled(DD.flag_groups(DD.load_block("validation"), cfg))
    tr_ok = tr[~tr.held_group].reset_index(drop=True)
    va_ok = va[~va.held_group].reset_index(drop=True)
    va_held = va[va.held_group].reset_index(drop=True)
    assert not tr_ok["held_group"].any() and not va_ok["held_group"].any()
    mlflow.set_tracking_uri(f"sqlite:///{Path('mlruns/mlflow.db').resolve()}")
    name = "connection_model"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(Path("mlruns/artifacts").resolve()))
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"connections-{pick}-{head}") as run:
        model = P.fit_all(tr_ok, va_ok)
        bar = BL.fit(tr_ok)
        nya = DD.not_arrived(va_held).to_numpy()
        masks = {"held_all": np.ones(len(va_held), bool), "held_not_arrived": nya}
        cells = val_cells(model, pick, va_held, bar, masks)
        mlflow.log_params(
            {"option": pick, "online_eta": eta, "thin": cfg["thin"],
             "max_gap_min": cfg["max_gap_min"], "group_mod": cfg["group_mod"],
             "clf_params": str(model.meta["clf_params"]),
             "train_rows": len(tr_ok), "validation_rows": len(va_ok)}
        )  # fmt: skip
        mlflow.set_tags({"git_head": head, "git_dirty": str(dirty), "test_scored": "False"})
        log_cells("val", cells)
        run_id = run.info.run_id
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS / f"{run_id}.joblib")
        lho = {}
        for hub in cfg["hubs"]:
            tr_h = tr_ok[tr_ok.hub != hub].reset_index(drop=True)
            va_h = va_ok[va_ok.hub != hub].reset_index(drop=True)
            m = P.fit_all(tr_h, va_h)
            joblib.dump(m, MODELS / f"{run_id}_lho_{hub}.joblib")
            vh = va[va.hub == hub].reset_index(drop=True)
            masks = {"all": np.ones(len(vh), bool), "not_arrived": DD.not_arrived(vh).to_numpy()}
            lho[hub] = val_cells(m, pick, vh, BL.fit(tr_h), masks)
            log_cells(f"val_lho_{hub}", lho[hub])
    out = {
        "run_id": run_id, "git_head": head, "git_dirty": dirty, "option": pick,
        "online_eta": eta, "validation": cells, "validation_leave_hub_out": lho,
        "meta": model.meta,
    }  # fmt: skip
    (OUT / "final_fit.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("run_id", "git_head", "git_dirty", "option")}))
    print(json.dumps(lho, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
