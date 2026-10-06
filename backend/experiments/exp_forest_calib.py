"""forest / stage 2: effect of class weighting and probability calibration on one base forest.

usage: python exp_forest_calib.py <algo> <max_features> <min_samples_leaf> [targets,comma,sep]

Variants (all refit per outer fold; calibration CV is inner, on the training fold only):
  raw_none     forest, class_weight=None, no calibration
  raw_bs       forest, class_weight=balanced_subsample, no calibration
  sigcv_bs     CalibratedClassifierCV(sigmoid, 5-fold, ensemble=False)  -> 1 forest + Platt
  isocv_bs     CalibratedClassifierCV(isotonic, 5-fold, ensemble=False)
  sigens_bs    CalibratedClassifierCV(sigmoid, 5-fold, ensemble=True)   -> avg of 5 (forest, Platt)
  sigoob_bs    Platt fit on the forest's own out-of-bag probabilities (no extra fits)
  isooob_bs    isotonic on OOB probabilities
  sigcv_none   CalibratedClassifierCV(sigmoid, ensemble=False) on the unweighted forest
"""
from __future__ import annotations

import json
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from exp_forest_common import OOBCalibratedForest, make_forest
from harness import RESULTS_DIR, TARGETS, evaluate, load_xy, paired_auc_delta

ALGO = sys.argv[1]
MF = sys.argv[2] if sys.argv[2] in ("sqrt", "log2") else float(sys.argv[2])
MSL = int(sys.argv[3])
SEL = sys.argv[4].split(",") if len(sys.argv) > 4 else TARGETS
BASE = dict(algo=ALGO, max_features=MF, min_samples_leaf=MSL)


def forest(cw):
    return make_forest(class_weight=cw, **BASE)


def inner():
    return StratifiedKFold(5, shuffle=True, random_state=0)


VARIANTS = {
    "raw_none": lambda: forest(None),
    "raw_bs": lambda: forest("balanced_subsample"),
    "sigcv_bs": lambda: CalibratedClassifierCV(forest("balanced_subsample"), method="sigmoid",
                                               cv=inner(), ensemble=False),
    "isocv_bs": lambda: CalibratedClassifierCV(forest("balanced_subsample"), method="isotonic",
                                               cv=inner(), ensemble=False),
    "sigens_bs": lambda: CalibratedClassifierCV(forest("balanced_subsample"), method="sigmoid",
                                                cv=inner(), ensemble=True),
    "sigoob_bs": lambda: OOBCalibratedForest(ALGO, dict(max_features=MF, min_samples_leaf=MSL,
                                                        class_weight="balanced_subsample"), "sigmoid"),
    "isooob_bs": lambda: OOBCalibratedForest(ALGO, dict(max_features=MF, min_samples_leaf=MSL,
                                                        class_weight="balanced_subsample"), "isotonic"),
    "sigcv_none": lambda: CalibratedClassifierCV(forest(None), method="sigmoid", cv=inner(),
                                                 ensemble=False),
}


def name(v):
    return f"forest_cal_{ALGO}_mf{MF}_msl{MSL}_{v}"


def run(v, t, X, Y):
    return evaluate(name(v), t, VARIANTS[v], X, Y)


def main():
    X, Y, _ = load_xy()
    jobs = [(v, t) for t in SEL for v in VARIANTS]
    res = Parallel(n_jobs=2)(delayed(run)(v, t, X, Y) for v, t in jobs)
    table = {}
    for (v, t), r in zip(jobs, res):
        m = r["metrics"]
        d = paired_auc_delta(name(v), "baseline_xgb", t)
        table.setdefault(t, {})[v] = {k: m[k]["mean"] for k in
                                      ("roc_auc", "brier", "log_loss", "f1", "accuracy", "recall",
                                       "precision")}
        table[t][v].update(auc_std=m["roc_auc"]["std"], d_mean=d["mean"], d_min=d["min"],
                           d_std=d["std"])
    with open(RESULTS_DIR / f"forest_calib_{ALGO}_mf{MF}_msl{MSL}_summary.json", "w") as f:
        json.dump(table, f, indent=1)
    for t in SEL:
        b = json.load(open(RESULTS_DIR / f"baseline_xgb__{t}.json"))["metrics"]
        print(f"\n=== {t}  (baseline_xgb: AUC {b['roc_auc']['mean']:.3f} Brier {b['brier']['mean']:.3f} "
              f"LL {b['log_loss']['mean']:.3f} F1 {b['f1']['mean']:.3f} Acc {b['accuracy']['mean']:.3f})")
        for v, m in table[t].items():
            print(f"  {v:11s} AUC {m['roc_auc']:.3f}±{m['auc_std']:.3f} d {m['d_mean']:+.3f}"
                  f"(min {m['d_min']:+.3f})  Brier {m['brier']:.4f}  LL {m['log_loss']:.4f}  "
                  f"F1 {m['f1']:.3f} Acc {m['accuracy']:.3f} Rec {m['recall']:.3f}")


if __name__ == "__main__":
    main()
