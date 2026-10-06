"""Step 7: summary table for the ensemble strategy (reads saved results only).

For every experiment x target: AUC, PR-AUC, Brier, log-loss, F1/Acc@0.5, calibration
slope/ECE, paired AUC delta vs baseline_xgb on seed-0 splits and (if run) on unseen
seed-100 splits, plus nested tuned-threshold F1/Acc when available.
usage: python exp_ensemble_summary.py [exp names without the 'ensemble_' prefix...]
"""
import json
import sys

import numpy as np

from harness import RESULTS_DIR, load_xy, paired_auc_delta
from exp_ensemble_common import calibration_stats

_, Y, _ = load_xy()
DEFAULT = ["m_lr_l1", "m_xgbrf_rf", "vote_lxrc", "stack_lxrc", "vote_l1xf", "platt_l1xf",
           "stack3_l1xcxrf", "vote_l1c", "platt_l1c", "iso_l1c", "vote_xrf", "platt_xrf", "iso_xrf",
           "vote_lc6xrf", "platt_lc6xrf", "vote_l1x", "platt_l1x", "vote_l1f", "platt_l1f", "iso_l1f"]


def row(exp, t):
    name = exp if exp == "baseline_xgb" else f"ensemble_{exp}"
    try:
        res = json.load(open(RESULTS_DIR / f"{name}__{t}.json"))
    except FileNotFoundError:
        return None
    m = res["metrics"]
    P = np.load(RESULTS_DIR / f"{name}__{t}.oof.npy")
    cs = calibration_stats(Y[t], P)
    out = {"auc": m["roc_auc"]["mean"], "auc_sd": m["roc_auc"]["std"], "pr": m["pr_auc"]["mean"],
           "brier": m["brier"]["mean"], "ll": m["log_loss"]["mean"], "f1": m["f1"]["mean"],
           "acc": m["accuracy"]["mean"], "slope": cs["cal_slope"], "ece": cs["ece"]}
    if name != "baseline_xgb":
        d = paired_auc_delta(name, "baseline_xgb", t)
        out.update(d0=d["mean"], d0min=d["min"], d0sd=d["std"])
        try:
            d = paired_auc_delta(name, "baseline_xgb", t, seed_offset=100)
            out.update(d100=d["mean"], d100min=d["min"])
        except FileNotFoundError:
            pass
    thr_name = "base" if exp == "baseline_xgb" else exp
    try:
        th = json.load(open(RESULTS_DIR / f"ensemble_thresh_{thr_name}__{t}.json"))
        out.update(tf1=th["f1"], tacc=th["accuracy"], tprec=th["precision"], trec=th["recall"])
    except FileNotFoundError:
        pass
    return out


if __name__ == "__main__":
    exps = ["baseline_xgb"] + (sys.argv[1:] or DEFAULT)
    for t in ["cad", "lad", "lcx", "rca"]:
        print(f"\n=== {t} ===")
        print(f"{'experiment':22} {'AUC':>12} {'PR':>6} {'Brier':>6} {'LL':>6} {'F1.5':>6} {'Acc.5':>6} "
              f"{'slope':>5} {'ECE':>5} {'dAUC s0 (min)':>16} {'dAUC s100 (min)':>16} {'tunedF1/Acc':>12}")
        for e in exps:
            r = row(e, t)
            if r is None:
                continue
            d0 = f"{r['d0']:+.4f}({r['d0min']:+.3f})" if "d0" in r else ""
            d1 = f"{r['d100']:+.4f}({r['d100min']:+.3f})" if "d100" in r else ""
            th = f"{r['tf1']:.3f}/{r['tacc']:.3f}" if "tf1" in r else ""
            print(f"{e:22} {r['auc']:.4f}±{r['auc_sd']:.3f} {r['pr']:.3f} {r['brier']:.4f} {r['ll']:.3f} "
                  f"{r['f1']:.3f} {r['acc']:.3f} {r['slope']:5.2f} {r['ece']:.3f} {d0:>16} {d1:>16} {th:>12}")
