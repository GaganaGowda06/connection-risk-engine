"""Train the route-agnostic model: LightGBM quantiles, a calibrated P(delay >= 15), and the
validation gate. No route, station or operator identity reaches the model (features.FEATURES).

Training rows are the train block of the non-held-out routes. Hyperparameters are tuned with
leave-route-group-out folds inside the train dates: a fold's routes are never in its fit, and the
Stage 2 base forecast of a fold is fitted without them. The calibrator for P(delay >= 15) is
fitted on validation rows of the training routes only; held-out routes are never used to fit,
tune or calibrate anything. Validation rows of held-out routes are read once, for the gate.
"""

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression

from rail_risk.baselines import models
from rail_risk.network_model import dataset as D
from rail_risk.network_model import features as F
from rail_risk.network_model.evaluate import paired_table

CONFIG = Path("config/network_model.yaml")
MODEL_DIR = Path("data/interim/stage3/models")
REPORT_DIR = Path("reports/network_model")
RECOVERY_FEATURES = [
    "late_now_min", "horizon_min", "n_actuals", "stops_remaining", "sched_min_remaining",
    "fastest_min_remaining", "slack_min", "late_minus_slack_min", "late_chg_1_min",
    "late_chg_3_min", "fc_late_min",
]  # fmt: skip
MONOTONE_UP = ("late_now_min", "fc_late_min", "late_minus_slack_min")


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if cfg["option"] not in ("residual", "raw", "recovery"):
        raise ValueError(f"unknown option {cfg['option']!r}")
    return cfg


def feature_names(option: str) -> list[str]:
    return RECOVERY_FEATURES if option == "recovery" else F.FEATURES


def target_for_fit(option: str, df: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    y = df["target_min"].to_numpy()
    if option == "residual":
        return y - base
    if option == "recovery":
        return df["late_now_min"].fillna(0.0).to_numpy() - y
    return y


def to_lateness(option: str, df: pd.DataFrame, base: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Back from the fitted target to lateness in minutes."""
    if option == "residual":
        return base + raw
    if option == "recovery":
        return df["late_now_min"].fillna(0.0).to_numpy() - raw
    return raw


@dataclass
class Booster3:
    option: str
    quantiles: dict[float, lgb.Booster]
    classifier: lgb.Booster
    calibrators: dict[int, IsotonicRegression]
    features: list[str]
    class_features: list[str] = field(default_factory=lambda: list(F.FEATURES))

    def predict(self, df: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
        """point, lo, hi (minutes of lateness) and the calibrated P(delay >= 15)."""
        x = D.matrix(df, self.features)
        out = {}
        # a recovery target flips the quantile: the 10% lateness is the 90% recovery
        flip = self.option == "recovery"
        for alpha, key in ((0.1, "lo"), (0.5, "point"), (0.9, "hi")):
            raw = self.quantiles[(1 - alpha) if flip else alpha].predict(x)
            out[key] = to_lateness(self.option, df, base, raw)
        res = pd.DataFrame(out, index=df.index)
        res[["lo", "hi"]] = np.sort(res[["lo", "hi"]].to_numpy(), axis=1)
        p = self.classifier.predict(D.matrix(df, self.class_features))
        h = df["horizon_min"].to_numpy()
        cal = np.array(
            [self.calibrators[int(k)].predict([v])[0] if int(k) in self.calibrators else v
             for k, v in zip(h, p, strict=True)]
        )  # fmt: skip
        res["p_late"] = np.clip(cal, 0.0, 1.0)
        res["p_late_raw"] = p
        return res

    def save(self, path: Path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        for alpha, b in self.quantiles.items():
            b.save_model(str(path / f"q{int(round(alpha * 100)):02d}.txt"))
        self.classifier.save_model(str(path / "classifier.txt"))
        joblib.dump(
            {"option": self.option, "calibrators": self.calibrators, "features": self.features},
            path / "meta.joblib",
        )

    @staticmethod
    def load(path: Path) -> "Booster3":
        import joblib

        meta = joblib.load(path / "meta.joblib")
        q = {
            a: lgb.Booster(model_file=str(path / f"q{int(round(a * 100)):02d}.txt"))
            for a in (0.1, 0.5, 0.9)
        }
        return Booster3(
            meta["option"], q, lgb.Booster(model_file=str(path / "classifier.txt")),
            meta["calibrators"], meta["features"],
        )  # fmt: skip


def route_folds(routes: list[str], k: int) -> list[set[str]]:
    """Leave-route-group-out folds: each route is in exactly one fold."""
    ordered = sorted(routes)
    return [set(ordered[i::k]) for i in range(k)]


def _params(cfg: dict, grid_point: dict, objective: dict, option: str, features: list[str]) -> dict:
    p = {**cfg["fixed"], **grid_point, **objective, "seed": cfg["seed"], "verbose": -1}
    p["num_threads"] = cfg.get("num_threads", 4)
    if objective["objective"] == "binary":
        p["monotone_constraints"] = [1 if c in MONOTONE_UP else 0 for c in features]
    return p


def tune(rows: pd.DataFrame, cfg: dict, option: str) -> dict:
    """Leave-route-group-out tuning on a subsample of train days; returns the best grid point,
    its rounds, and every fold score (the median model, then the late classifier)."""
    days = sorted(rows["ssd"].unique())[:: cfg["tune_day_step"]]
    sub = rows[rows["ssd"].isin(set(days))]
    folds = route_folds(sorted(sub["route"].unique()), cfg["folds"])
    feats = feature_names(option)
    keys = sorted(cfg["grid"])
    points = [
        dict(zip(keys, v, strict=True)) for v in itertools.product(*(cfg["grid"][k] for k in keys))
    ]
    results = {"median": [], "classifier": []}
    for fold in folds:
        va = sub[sub["route"].isin(fold)]
        tr = sub[~sub["route"].isin(fold)]
        fitted = models.fit(tr)  # the base for this fold never saw the fold's routes
        base_tr = models.predict(fitted, tr)["shrunk_persistence"]["point"].to_numpy()
        base_va = models.predict(fitted, va)["shrunk_persistence"]["point"].to_numpy()
        xt, xv = D.matrix(tr, feats), D.matrix(va, feats)
        y_tr = target_for_fit(option, tr, base_tr)
        late_tr = (tr["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
        late_va = (va["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
        cx_t, cx_v = D.matrix(tr), D.matrix(va)
        for gp in points:
            p = _params(cfg, gp, {"objective": "quantile", "alpha": 0.5}, option, feats)
            dv = lgb.Dataset(xv, target_for_fit(option, va, base_va))
            m = lgb.train(
                p, lgb.Dataset(xt, y_tr), cfg["max_rounds"], valid_sets=[dv],
                callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False)],
            )  # fmt: skip
            pred = to_lateness(option, va, base_va, m.predict(xv, num_iteration=m.best_iteration))
            mae = float(np.abs(pred - va["target_min"].to_numpy()).mean())
            results["median"].append(
                {"fold": sorted(fold), **gp, "rounds": m.best_iteration, "mae": mae}
            )
            pc = _params(cfg, gp, {"objective": "binary"}, option, F.FEATURES)
            mc = lgb.train(
                pc, lgb.Dataset(cx_t, late_tr), cfg["max_rounds"],
                valid_sets=[lgb.Dataset(cx_v, late_va)],
                callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False)],
            )  # fmt: skip
            pr = mc.predict(cx_v, num_iteration=mc.best_iteration)
            brier = float(((pr - late_va) ** 2).mean())
            results["classifier"].append(
                {"fold": sorted(fold), **gp, "rounds": mc.best_iteration, "brier": brier}
            )
    out = {"folds": [sorted(f) for f in folds], "tune_days": len(days), "grid_results": results}
    for kind, metric in (("median", "mae"), ("classifier", "brier")):
        df = pd.DataFrame(results[kind])
        agg = (
            df.groupby(keys).agg(score=(metric, "mean"), rounds=("rounds", "median")).reset_index()
        )
        best = agg.sort_values("score").iloc[0]
        out[kind] = {
            "params": {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys},
            "rounds": int(max(best["rounds"], 20)),
            "score": float(best["score"]),
        }
    return out


def fit_final(
    train: pd.DataFrame,
    validation_train_routes: pd.DataFrame,
    cfg: dict,
    tuned: dict,
    base_tr: np.ndarray,
) -> Booster3:
    option = cfg["option"]
    feats = feature_names(option)
    x = D.matrix(train, feats)
    y = target_for_fit(option, train, base_tr)
    qs = {}
    for alpha in cfg["quantiles"]:
        a = (1 - alpha) if option == "recovery" else alpha
        p = _params(
            cfg, tuned["median"]["params"], {"objective": "quantile", "alpha": a}, option, feats
        )
        qs[a] = lgb.train(p, lgb.Dataset(x, y), tuned["median"]["rounds"])
    late = (train["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
    pc = _params(cfg, tuned["classifier"]["params"], {"objective": "binary"}, option, F.FEATURES)
    clf = lgb.train(pc, lgb.Dataset(D.matrix(train), late), tuned["classifier"]["rounds"])
    raw = clf.predict(D.matrix(validation_train_routes))
    yv = (validation_train_routes["target_min"] >= models.LATE_MIN).astype(float).to_numpy()
    h = validation_train_routes["horizon_min"].to_numpy()
    cals = {}
    for k in np.unique(h):
        m = h == k
        cals[int(k)] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            raw[m], yv[m]
        )
    return Booster3(option, qs, clf, cals, feats)


def gate(
    model: Booster3, val: pd.DataFrame, held: set[str], fit_rows: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Paired date-bootstrap on validation, held-out routes, 15 and 60 minutes."""
    hv = val[val["route"].isin(held)]
    preds = models.predict(models.fit(fit_rows), hv)
    base = preds["shrunk_persistence"]["point"].to_numpy()
    mp = model.predict(hv, base)["point"].to_numpy()
    return paired_table(
        hv, mp,
        {
            "persistence": preds["persistence"]["point"].to_numpy(),
            "shrunk_persistence": base,
            "historical": preds["historical"]["point"].to_numpy(),
        },
        (15, 60), cfg["resamples"], cfg["seed"],
    )  # fmt: skip


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
