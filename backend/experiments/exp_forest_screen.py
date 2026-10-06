"""forest / stage 1: EXPLORATORY screen of fixed RandomForest / ExtraTrees configs.

Every config is scored with the shared 5x5 protocol (harness.oof_probabilities), but
picking the best of many configs on the same outer folds is selection-biased, so this
stage is only used to (a) understand which knobs matter and (b) shrink the grid that the
honest nested search (exp_forest_nested.py) explores.  Winners are re-checked on unseen
splits (HARNESS_SEED_OFFSET=100) in exp_forest_final.py.

Output: results/forest_screen_summary.json (+ printed table). OOF arrays are not saved
per config to keep results/ tidy; paired deltas vs baseline_xgb are computed here.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from harness import RESULTS_DIR, TARGETS, load_xy, metrics_from_oof, oof_probabilities

N_TREES = 300
ALGOS = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}


def configs():
    out = []
    for algo, mf, msl, cw in itertools.product(
            ["rf", "et"], ["sqrt", 0.33, 0.6], [1, 3, 6, 12], [None, "balanced_subsample"]):
        out.append(dict(algo=algo, max_features=mf, min_samples_leaf=msl, max_depth=None,
                        class_weight=cw))
    # a few depth-limited variants (depth instead of leaf size as the capacity knob)
    for algo, d in itertools.product(["rf", "et"], [3, 5, 8]):
        out.append(dict(algo=algo, max_features="sqrt", min_samples_leaf=1, max_depth=d,
                        class_weight="balanced_subsample"))
    return out


def tag(c):
    return (f"{c['algo']}_mf{c['max_features']}_msl{c['min_samples_leaf']}_d{c['max_depth']}"
            f"_cw{'bs' if c['class_weight'] else 'none'}")


def run_one(c, X, Y, base):
    t0 = time.time()
    res = {"tag": tag(c), "config": c, "targets": {}}
    for t in TARGETS:
        def make():
            return ALGOS[c["algo"]](n_estimators=N_TREES, max_features=c["max_features"],
                                    min_samples_leaf=c["min_samples_leaf"],
                                    max_depth=c["max_depth"], class_weight=c["class_weight"],
                                    n_jobs=1, random_state=42)
        P = oof_probabilities(make, X, Y[t])
        m = metrics_from_oof(Y[t], P)
        d = np.array([roc_auc_score(Y[t], P[r]) - roc_auc_score(Y[t], base[t][r])
                      for r in range(len(P))])
        res["targets"][t] = {
            "auc": m["roc_auc"]["mean"], "auc_std": m["roc_auc"]["std"],
            "brier": m["brier"]["mean"], "log_loss": m["log_loss"]["mean"],
            "f1": m["f1"]["mean"], "acc": m["accuracy"]["mean"],
            "d_mean": float(d.mean()), "d_std": float(d.std()), "d_min": float(d.min()),
        }
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    X, Y, _ = load_xy()
    base = {t: np.load(RESULTS_DIR / f"baseline_xgb__{t}.oof.npy") for t in TARGETS}
    cfgs = configs()
    print(f"screening {len(cfgs)} configs x {len(TARGETS)} targets", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=2, verbose=0)(delayed(run_one)(c, X, Y, base) for c in cfgs)
    print(f"total {time.time() - t0:.0f}s")
    with open(RESULTS_DIR / "forest_screen_summary.json", "w") as f:
        json.dump(out, f, indent=1)

    for t in TARGETS:
        rows = sorted(out, key=lambda r: -r["targets"][t]["auc"])
        print(f"\n=== {t}: top 8 by AUC (baseline_xgb paired delta) ===")
        for r in rows[:8]:
            m = r["targets"][t]
            print(f"  {r['tag']:42s} AUC {m['auc']:.3f}±{m['auc_std']:.3f}  "
                  f"d {m['d_mean']:+.3f} (min {m['d_min']:+.3f})  Brier {m['brier']:.3f}  "
                  f"LL {m['log_loss']:.3f}  F1 {m['f1']:.3f}  Acc {m['acc']:.3f}")
    # average rank across targets -> robust single config
    ranks = {r["tag"]: 0.0 for r in out}
    for t in TARGETS:
        for i, r in enumerate(sorted(out, key=lambda r: -r["targets"][t]["auc"])):
            ranks[r["tag"]] += i / len(TARGETS)
    print("\n=== best average AUC rank across targets ===")
    for k, v in sorted(ranks.items(), key=lambda kv: kv[1])[:10]:
        r = next(x for x in out if x["tag"] == k)
        print(f"  {k:42s} rank {v:5.1f}  " + "  ".join(
            f"{t}:{r['targets'][t]['auc']:.3f}" for t in TARGETS))
    # marginal effect of each knob (mean AUC over the main grid)
    print("\n=== marginal mean AUC per knob value (main grid only) ===")
    main_grid = [r for r in out if r["config"]["max_depth"] is None]
    for knob in ["algo", "max_features", "min_samples_leaf", "class_weight"]:
        vals = sorted({str(r["config"][knob]) for r in main_grid})
        for v in vals:
            sel = [r for r in main_grid if str(r["config"][knob]) == v]
            print(f"  {knob:16s}={v:20s} " + "  ".join(
                f"{t}:{np.mean([r['targets'][t]['auc'] for r in sel]):.3f}/B{np.mean([r['targets'][t]['brier'] for r in sel]):.3f}"
                for t in TARGETS))


if __name__ == "__main__":
    main()
