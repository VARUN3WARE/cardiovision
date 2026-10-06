"""Adversarial verification of the 'forest' claims (independent re-implementation).

Re-builds the claimed production model from the written recipe (NOT importing the forest
agent's code): two members, each CalibratedClassifierCV(sigmoid, inner StratifiedKFold(5,
shuffle, rs=0), ensemble=False) around a 500-tree RF / ExtraTrees (max_features='sqrt',
min_samples_leaf=3, class_weight='balanced_subsample', random_state=42); blend = mean of the
members' calibrated log-odds.  Both members are refit inside every outer training fold.

The outer loop mirrors harness.oof_probabilities exactly (same StratifiedKFold seeds
r + HARNESS_SEED_OFFSET), so saved OOFs pair with baseline_xgb via harness.paired_auc_delta.
Saves verify_forest_{rf,et,blend}__<target>[__seedN].{oof.npy,json}.

usage: [HARNESS_SEED_OFFSET=100] python exp_verify_forest_run.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from harness import (N_REPEATS, N_SPLITS, RESULTS_DIR, SEED_OFFSET, TARGETS, load_xy,
                     metrics_from_oof)

ALGOS = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}
EPS = 1e-6


def member(algo):
    return CalibratedClassifierCV(
        ALGOS[algo](n_estimators=500, max_features="sqrt", min_samples_leaf=3,
                    class_weight="balanced_subsample", n_jobs=1, random_state=42),
        method="sigmoid", ensemble=False, cv=StratifiedKFold(5, shuffle=True, random_state=0))


def one_repeat(t, r, X, y):
    out = {"rf": np.zeros(len(y)), "et": np.zeros(len(y))}
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=r + SEED_OFFSET)
    for tr, te in cv.split(X, y):
        for a in ("rf", "et"):
            m = member(a).fit(X.iloc[tr], y[tr])
            out[a][te] = m.predict_proba(X.iloc[te])[:, 1]
    return t, r, out


def main():
    X, Y, _ = load_xy()
    t0 = time.time()
    res = Parallel(n_jobs=2)(delayed(one_repeat)(t, r, X, Y[t])
                             for t in TARGETS for r in range(N_REPEATS))
    sfx = f"__seed{SEED_OFFSET}" if SEED_OFFSET else ""
    for t in TARGETS:
        P = {a: np.zeros((N_REPEATS, len(Y[t]))) for a in ("rf", "et")}
        for tt, r, out in res:
            if tt == t:
                for a in P:
                    P[a][r] = out[a]
        P["blend"] = expit(0.5 * (logit(np.clip(P["rf"], EPS, 1 - EPS))
                                  + logit(np.clip(P["et"], EPS, 1 - EPS))))
        for a, arr in P.items():
            name = f"verify_forest_{a}"
            np.save(RESULTS_DIR / f"{name}__{t}{sfx}.oof.npy", arr)
            m = metrics_from_oof(Y[t], arr)
            with open(RESULTS_DIR / f"{name}__{t}{sfx}.json", "w") as f:
                json.dump({"experiment": name, "target": t, "seed_offset": SEED_OFFSET,
                           "metrics": m}, f, indent=2)
            print(f"[{name}] {t} seed{SEED_OFFSET} AUC {m['roc_auc']['mean']:.4f}"
                  f"±{m['roc_auc']['std']:.4f} Brier {m['brier']['mean']:.4f} "
                  f"F1 {m['f1']['mean']:.4f} Acc {m['accuracy']['mean']:.4f}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
