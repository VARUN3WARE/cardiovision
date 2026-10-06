"""forest / stage 4: pre-declared final candidates, run on the standard splits AND on unseen
splits (HARNESS_SEED_OFFSET=100) to check that the screen's selection was not luck.

  forest_final_rf     RandomForest(500, max_features='sqrt', min_samples_leaf=3,
                      class_weight='balanced_subsample') + Platt (CalibratedClassifierCV
                      sigmoid, inner 5-fold, ensemble=False -> one forest + one sigmoid)
  forest_final_et     same with ExtraTrees
  forest_final_blend  mean of the two members' calibrated LOG-ODDS (members fit on the same
                      outer folds, so the combined OOF is a valid OOF of the blend; log-odds
                      averaging keeps attributions exact: 0.5*(phi_rf + phi_et))

usage: [HARNESS_SEED_OFFSET=100] python exp_forest_final.py [--thr]
  --thr also reports nested F1-tuned-threshold metrics (harness.tuned_threshold_metrics) on
  the raw class-weighted forests (Platt is monotone, so the tuned decision rule is the same
  family; skipping the inner calibration keeps the run cheap).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

import harness
from exp_forest_common import make_forest
from harness import (RESULTS_DIR, SEED_OFFSET, TARGETS, evaluate, load_xy, metrics_from_oof,
                     paired_auc_delta, tuned_threshold_metrics)

PARAMS = dict(max_features="sqrt", min_samples_leaf=3, class_weight="balanced_subsample")
N_TREES = 500


def calibrated(algo):
    def make():
        return CalibratedClassifierCV(make_forest(algo, n_estimators=N_TREES, **PARAMS),
                                      method="sigmoid", ensemble=False,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return make


def raw(algo):
    return lambda: make_forest(algo, n_estimators=N_TREES, **PARAMS)


def sfx():
    return f"__seed{SEED_OFFSET}" if SEED_OFFSET else ""


def run(algo, t, X, Y):
    return evaluate(f"forest_final_{algo}", t, calibrated(algo), X, Y)


def thr(algo, t, X, Y):
    return algo, t, tuned_threshold_metrics(f"forest_final_{algo}", t, raw(algo), X, Y)


def main():
    X, Y, _ = load_xy()
    jobs = [(a, t) for t in TARGETS for a in ("rf", "et")]
    Parallel(n_jobs=2)(delayed(run)(a, t, X, Y) for a, t in jobs)

    out = {}
    for t in TARGETS:
        A = np.load(RESULTS_DIR / f"forest_final_rf__{t}{sfx()}.oof.npy")
        B = np.load(RESULTS_DIR / f"forest_final_et__{t}{sfx()}.oof.npy")
        eps = 1e-6
        P = expit(0.5 * (logit(np.clip(A, eps, 1 - eps)) + logit(np.clip(B, eps, 1 - eps))))
        np.save(RESULTS_DIR / f"forest_final_blend__{t}{sfx()}.oof.npy", P)
        m = metrics_from_oof(Y[t], P)
        with open(RESULTS_DIR / f"forest_final_blend__{t}{sfx()}.json", "w") as f:
            json.dump({"experiment": "forest_final_blend", "target": t, "n": int(len(Y[t])),
                       "positives": int(Y[t].sum()), "metrics": m, "seed_offset": SEED_OFFSET},
                      f, indent=2)
        out[t] = {}
        for name in ("forest_final_rf", "forest_final_et", "forest_final_blend"):
            r = json.load(open(RESULTS_DIR / f"{name}__{t}{sfx()}.json"))["metrics"]
            d = paired_auc_delta(name, "baseline_xgb", t, SEED_OFFSET)
            out[t][name] = {"metrics": {k: v["mean"] for k, v in r.items()},
                            "auc_std": r["roc_auc"]["std"], "paired": d}
    b = {t: json.load(open(RESULTS_DIR / f"baseline_xgb__{t}{sfx()}.json"))["metrics"] for t in TARGETS}
    print(f"\n##### seed offset {SEED_OFFSET} #####")
    for t in TARGETS:
        print(f"=== {t}  baseline_xgb AUC {b[t]['roc_auc']['mean']:.3f} Brier {b[t]['brier']['mean']:.3f} "
              f"LL {b[t]['log_loss']['mean']:.3f} F1 {b[t]['f1']['mean']:.3f} Acc {b[t]['accuracy']['mean']:.3f}")
        for name, r in out[t].items():
            m, d = r["metrics"], r["paired"]
            print(f"  {name:20s} AUC {m['roc_auc']:.3f}±{r['auc_std']:.3f}  PR {m['pr_auc']:.3f}  "
                  f"Brier {m['brier']:.3f}  LL {m['log_loss']:.3f}  F1 {m['f1']:.3f}  Acc {m['accuracy']:.3f}  "
                  f"| d {d['mean']:+.4f}±{d['std']:.4f} min {d['min']:+.4f} {d['per_repeat']}")

    if "--thr" in sys.argv:
        res = Parallel(n_jobs=2)(delayed(thr)(a, t, X, Y) for a, t in jobs)
        for a, t, m in res:
            out[t][f"forest_final_{a}"]["tuned_threshold"] = m
            print(f"  tuned-thr {t} {a}: " + "  ".join(f"{k} {v['mean']:.3f}" for k, v in m.items()))
    with open(RESULTS_DIR / f"forest_final_summary{sfx()}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
