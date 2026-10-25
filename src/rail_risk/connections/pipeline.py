"""Fit and apply the three P(miss) options and their calibration, from labelled pair frames.

fit_all takes the train-block pairs and the validation-block pairs of groups that were never held
out (the caller filters; leave-hub-out also drops the hub). Folds leave corridor groups out: the
fold-k model never saw a pair that touches a corridor of fold k. The calibration pool is made of
those out-of-fold predictions plus the validation pairs of the same eligible groups.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rail_risk.connections import data as DD
from rail_risk.connections import models as M

NAMES = ("phys", "lgbm", "logit")
GRID_C = [
    {"num_leaves": 15, "min_data_in_leaf": 1000},
    {"num_leaves": 31, "min_data_in_leaf": 500},
    {"num_leaves": 63, "min_data_in_leaf": 200},
]
CROUNDS = 200


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class ConnModel:
    A: M.QuantileSet
    D: M.QuantileSet
    p_cancel: dict
    clf: object
    lr: M.Logistic
    iso: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def raw(self, df: pd.DataFrame) -> pd.DataFrame:
        x = M.prep(df)
        p_phys = M.physical_p(x, self.A, self.D, self.p_cancel)
        xb = x[M.FEATS_B].assign(p_phys=p_phys.astype("float32"))
        return pd.DataFrame(
            {"phys": p_phys, "lgbm": self.clf.predict(xb), "logit": self.lr.predict(x)},
            index=df.index,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        r = self.raw(df)
        h = df["horizon_min"].to_numpy().astype(int)
        out = r.copy()
        for n in NAMES:
            out[f"{n}_cal"] = M.apply_isotonic(self.iso[n], r[n].to_numpy(), h)
        return out


def _targets(df: pd.DataFrame):
    selA = (df["in_arr_late_s"].isna() & df["act_arr_in"].notna()).to_numpy()
    selD = (df["out_dep_late_s"].isna() & df["act_dep_out"].notna()).to_numpy()
    return selA, selD, df["in_late_final_min"].to_numpy(), df["out_late_final_min"].to_numpy()


def fit_all(tr: pd.DataFrame, va: pd.DataFrame, log=print, grid=GRID_C) -> ConnModel:
    t0 = time.time()
    xt, yt = M.prep(tr), tr["miss"].to_numpy(float)
    xv, yv = M.prep(va), va["miss"].to_numpy(float)
    sA, sD, aT, dT = _targets(tr)
    vA, vD, vaT, vdT = _targets(va)
    pc = M.fit_p_cancel(tr)
    folds = [DD.fold_split(tr, k) for k in range(DD.N_FOLDS)]

    def fit_qs(mask):
        a = M.QuantileSet(M.FEATS_A).fit(xt[mask & sA], aT[mask & sA])
        d = M.QuantileSet(M.FEATS_D).fit(xt[mask & sD], dT[mask & sD])
        return a, d

    fold_models = []
    for k, (trm, _tem) in enumerate(folds):
        fold_models.append(fit_qs(trm))
        log(f"fold {k} quantile heads {time.time() - t0:.0f}s")
    finalA, finalD = fit_qs(np.ones(len(tr), bool))
    # one conformal scale per head from the pooled out-of-group rows
    for sel, tgt, vsel, vtgt, ix in (
        (sA, aT, vA, vaT, 0),
        (sD, dT, vD, vdT, 1),
    ):
        raw_oof = np.full((len(tr), len(M.LEVELS)), np.nan)
        for k, (_, tem) in enumerate(folds):
            m = tem & sel
            if m.any():
                raw_oof[m] = fold_models[k][ix].raw(xt[m])
        head = (finalA, finalD)[ix]
        rv = head.raw(xv[vsel])
        ok = ~np.isnan(raw_oof[:, 0]) & sel
        lo = np.concatenate([raw_oof[ok, 1], rv[:, 1]])
        hi = np.concatenate([raw_oof[ok, 5], rv[:, 5]])
        y = np.concatenate([tgt[ok], vtgt[vsel]])
        w = np.maximum(hi - lo, M.EPS_WIDTH)
        s = np.maximum(lo - y, y - hi) / w
        n = len(s)
        scale = float(np.quantile(s, min(1.0, np.ceil((n + 1) * 0.8) / n)))
        for m in (
            head,
            fold_models[0][ix],
            fold_models[1][ix],
            fold_models[2][ix],
            fold_models[3][ix],
        ):
            m.scale = scale
    log(f"scales A {finalA.scale:.3f} D {finalD.scale:.3f}")
    p_oof = np.full(len(tr), np.nan)
    for k, (_, tem) in enumerate(folds):
        p_oof[tem] = M.physical_p(xt[tem], fold_models[k][0], fold_models[k][1], pc)
    # rows whose inbound corridor is in no fold cannot occur: every corridor has a fold
    assert not np.isnan(p_oof).any()
    xb = xt[M.FEATS_B].assign(p_phys=p_oof.astype("float32"))
    # tune the classifier on two folds
    scores = []
    for g in grid:
        ll = []
        for k in (0, 1):
            trm, tem = folds[k]
            c = M.fit_classifier(xb[trm], yt[trm], g, CROUNDS)
            ll.append(_logloss(c.predict(xb[tem]), yt[tem]))
        scores.append(float(np.mean(ll)))
        log(f"classifier grid {g} fold log loss {scores[-1]:.5f} {time.time() - t0:.0f}s")
    best = grid[int(np.argmin(scores))]
    oof = {n: np.full(len(tr), np.nan) for n in NAMES}
    oof["phys"] = p_oof
    for trm, tem in folds:
        c = M.fit_classifier(xb[trm], yt[trm], best, CROUNDS)
        oof["lgbm"][tem] = c.predict(xb[tem])
        oof["logit"][tem] = M.Logistic().fit(xt[trm], yt[trm]).predict(xt[tem])
    clf = M.fit_classifier(xb, yt, best, CROUNDS)
    lr = M.Logistic().fit(xt, yt)
    model = ConnModel(finalA, finalD, pc, clf, lr, meta={"clf_params": best, "grid": scores})
    # isotonic on the pooled out-of-group scores
    rv = model.raw(va)
    hv = va["horizon_min"].to_numpy().astype(int)
    ht = tr["horizon_min"].to_numpy().astype(int)
    for n in NAMES:
        p = np.concatenate([oof[n], rv[n].to_numpy()])
        model.iso[n] = M.fit_isotonic(p, np.concatenate([yt, yv]), np.concatenate([ht, hv]))
    model.meta["oof"] = {n: float(np.mean((oof[n] - yt) ** 2)) for n in NAMES}
    log(f"fit done {time.time() - t0:.0f}s")
    return model
