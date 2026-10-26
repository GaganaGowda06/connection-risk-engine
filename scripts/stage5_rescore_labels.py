#!/usr/bin/env python3
# ruff: noqa: E501
"""RESCORING AFTER A LABEL FIX (my decision, 2026-10-26): the frozen Stage 5 model and the
gap-bucket bar scored again on the test block (or February, as its own line) with the labels of the
adopted rule: an inbound with no arrival at the hub and no cancellation signal (in_cancel_final) is
unknown and excluded. Nothing is retrained or recalibrated: the predictions are the stored ones of
the original scoring (the same frozen run, the same bar fitted on the original train labels), only
the rows whose label became unknown are dropped. A second table refits the bar on the relabelled
train rows, shown as a sensitivity. Each block refuses to be rescored twice; run from a clean commit.

    uv run python scripts/stage5_rescore_labels.py --run-id RUN --block test
    uv run python scripts/stage5_rescore_labels.py --run-id RUN --block february
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rail_risk.connections import baseline as BL
from rail_risk.connections import build as B
from rail_risk.connections import data as DD

OUT = Path("reports/connections/stage5")
MODELS = Path("data/interim/stage5/models")
HORIZONS = (5, 15, 30, 60)
spec = importlib.util.spec_from_file_location(
    "stage5_score", Path(__file__).parent / "stage5_score.py"
)
S5 = importlib.util.module_from_spec(spec)
sys.modules["stage5_score"] = S5
spec.loader.exec_module(S5)


def lines(old: dict, new: dict, hubs) -> list[str]:
    out = [
        "| set | hub | horizon | n old | n new | Brier model old | Brier model new | Brier bar old | Brier bar new | diff old [95% CI] | diff new [95% CI] | ECE old | ECE new |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for subset in ("not_arrived", "all_pairs"):
        for hub in hubs:
            for hz in (*HORIZONS, "all"):
                k = f"held|{subset}|{hub}|{hz}"
                o, n = old.get(k, {}), new.get(k, {})
                if "paired_brier" not in o or "paired_brier" not in n:
                    continue
                po, pn = o["paired_brier"], n["paired_brier"]
                eo = o.get("ece", {}).get("value")
                en = n.get("ece", {}).get("value")
                f = lambda v: "" if v is None else f"{v:.4f}"  # noqa: E731
                out.append(
                    f"| {subset} | {hub} | {hz} | {o['n']:,} | {n['n']:,} | {o['brier_model']:.4f} | {n['brier_model']:.4f} | {o['brier_gap_bucket']:.4f} | {n['brier_gap_bucket']:.4f} "
                    f"| {po['diff']:+.4f} [{po['lo']:+.4f}, {po['hi']:+.4f}] | {pn['diff']:+.4f} [{pn['lo']:+.4f}, {pn['hi']:+.4f}] | {f(eo)} | {f(en)} |"
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--block", choices=("test", "february"), required=True)
    ap.add_argument("--dry", action="store_true", help="two dates, nothing written, any tree")
    args = ap.parse_args()
    marker = OUT / f"{args.block}_relabel_scored.json"
    if marker.exists() and not args.dry:
        raise SystemExit(f"already rescored: {marker.read_text()[:200]}")
    dirty = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty and not args.dry:
        raise SystemExit("run the rescoring from a clean commit")
    head = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
    ).stdout.strip()
    cfg = B.load_config()
    fit = json.loads((OUT / "final_fit.json").read_text())
    if fit["run_id"] != args.run_id:
        raise SystemExit("run id does not match final_fit.json")
    res_n, seed = cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"]
    pred = pd.read_parquet(MODELS / f"{args.run_id}_{args.block}_predictions.parquet")
    if args.dry:
        pred = pred[pred["ssd"].isin(sorted(pred["ssd"].unique())[:2])]
    keys = ["ssd", "hub", "in_rid", "out_rid", "horizon_min"]
    lab = DD.load_block(args.block, sorted(pred["ssd"].unique()))[
        [*keys, "act_arr_in", "in_cancel_final"]
    ]
    lab["in_cancel_final"] = lab["in_cancel_final"].fillna(False).astype(bool)
    df = pred.merge(lab, on=keys, how="left", validate="one_to_one")
    assert df["act_arr_in"].notna().sum() + df["act_arr_in"].isna().sum() == len(df) and len(
        df
    ) == len(pred)
    new = DD.apply_label_rule(df).reset_index(drop=True)
    dropped = df.merge(new[keys], on=keys, how="left", indicator=True).query(
        "_merge == 'left_only'"
    )
    counts = {}
    for hub in cfg["hubs"]:
        m = df["hub"] == hub
        d = int((dropped["hub"] == hub).sum())
        misses = int((df.loc[m, "miss"] == 1).sum())
        counts[hub] = {"rows_before": int(m.sum()), "rows_after": int((new["hub"] == hub).sum()), "unknown": d,
                       "unknown_pct_of_rows": round(100 * d / m.sum(), 3), "misses_before": misses,
                       "unknown_pct_of_misses": round(100 * d / misses, 2), "misses_after": misses - d}  # fmt: skip
    # tables: the original scorer's functions, the frozen predictions, the frozen bar
    held = new[new["held_group"]]
    res = {}
    for subset in ("all_pairs", "not_arrived"):
        res.update(
            S5.table(
                held,
                held["p_main"].to_numpy(),
                held["p_bar"].to_numpy(),
                cfg,
                subset,
                "held",
                res_n,
                seed,
            )
        )
    # sensitivity: the bar refitted on the relabelled train rows
    tr = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    tr_ok = DD.apply_label_rule(tr[~tr.held_group])
    bar2 = BL.fit(tr_ok)
    held = held.assign(p_bar2=BL.predict(bar2, held))
    res2 = {}
    for subset in ("all_pairs", "not_arrived"):
        res2.update(
            S5.table(
                held,
                held["p_main"].to_numpy(),
                held["p_bar2"].to_numpy(),
                cfg,
                subset,
                "held",
                res_n,
                seed,
            )
        )
    # leave-hub-out with the frozen leave-hub-out predictions and bars fitted on the original train labels
    tr_all = DD.labelled(DD.flag_groups(DD.load_block("train"), cfg))
    tr_all_ok = tr_all[~tr_all.held_group]
    lho = {}
    for h in cfg["hubs"]:
        sub = new[new["hub"] == h]
        pb = BL.predict(BL.fit(tr_all_ok[tr_all_ok.hub != h]), sub)
        for subset in ("all_pairs", "not_arrived"):
            part = S5.table(
                sub,
                sub[f"p_lho_{h}"].to_numpy(),
                pb,
                {"hubs": [h]},
                subset,
                f"lho_{h}",
                res_n,
                seed,
            )
            lho.update({k: v for k, v in part.items() if f"|{h}|" in k})
    old = json.loads((OUT / f"{args.block}_results.json").read_text())
    verdict_new, verdict_new2 = S5.verdict(res, cfg), S5.verdict(res2, cfg)
    out = {
        "label": "RESCORING AFTER A LABEL FIX: frozen run, frozen predictions and bar, labels of the 2026-10-26 rule",
        "block": args.block, "run_id": args.run_id, "git_head": head, "rows_before": len(df), "rows_after": len(new),
        "unknown_counts": counts, "verdict_original": old["verdict_held_out_groups"], "verdict_relabelled": verdict_new,
        "verdict_relabelled_bar_refit_on_relabelled_train": verdict_new2,
        "cells_relabelled": res, "cells_relabelled_bar_refit": res2, "cells_leave_hub_out_relabelled": lho,
    }  # fmt: skip
    if args.dry:
        print(json.dumps(counts, indent=1))
        print(json.dumps(verdict_new, indent=1, default=float))
        return
    (OUT / f"{args.block}_relabel_results.json").write_text(
        json.dumps(out, indent=1, default=float)
    )
    md = [
        f"# Stage 5 {args.block} rescored after the label fix ({head[:7]})",
        "",
        "Frozen run " + args.run_id + ", no retraining, no recalibration. Held-out groups.",
        "",
    ]
    md += lines(old["cells"], res, cfg["hubs"])
    md += [
        "",
        "Sensitivity, gap-bucket bar refitted on the relabelled train rows (diff new [95% CI]):",
        "",
    ]
    for hub in cfg["hubs"]:
        for subset in ("not_arrived", "all_pairs"):
            c = res2[f"held|{subset}|{hub}|all"]["paired_brier"]
            md.append(
                f"- {subset} {hub} all horizons: {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
            )
    (OUT / f"{args.block}_relabel_comparison.md").write_text("\n".join(md) + "\n")
    marker.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "git_head": head,
                "rows_before": len(df),
                "rows_after": len(new),
            }
        )
        + "\n"
    )
    print(json.dumps({"unknown_counts": counts, "verdict_original": old["verdict_held_out_groups"], "verdict_relabelled": verdict_new,
                      "verdict_bar_refit": verdict_new2}, indent=1, default=float))  # fmt: skip
    print("\n".join(md[:14]))
    _ = np


if __name__ == "__main__":
    main()
