"""Tuned-threshold F1/accuracy/precision/recall for the multitask winners on the harness's
exact outer folds. Stack winners use their own training-fold threshold (MultiTask.threshold_,
chosen from inner-OOF-based stack predictions); indep winners use harness.ThresholdTuned.

usage: python exp_multitask_thresh.py <target> [...]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

import exp_multitask_common as C
from harness import N_REPEATS, N_SPLITS, SEED_OFFSET, ThresholdTuned

WIN = {
    "cad": ("multitask_h2_L_indep", dict(mode="indep", base="lrlog")),
    "lad": ("multitask_h3avg_XLcX", dict(mode="stack", members=("lad:xgb", "lad:lrlog", "cad:lrlog", "cad:xgb"), meta="avg")),
    "lcx": ("multitask_h3avg_XLc", dict(mode="stack", members=("lcx:xgb", "lcx:lrlog", "cad:lrlog"), meta="avg")),
    "rca": ("multitask_h3avg_Lc", dict(mode="stack", members=("rca:lrlog", "cad:lrlog"), meta="avg")),
}

if __name__ == "__main__":
    for t in sys.argv[1:]:
        C._CACHE.clear()
        name, kw = WIN[t]
        y = C.Y_DICT[t]
        X = C.X_ALL
        rows = []
        for r in range(N_REPEATS):
            pred = np.zeros(len(y))
            for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=r + SEED_OFFSET).split(X, y):
                est = C.MultiTask(target=t, **kw)
                if kw["mode"] != "stack":
                    est = ThresholdTuned(est, random_state=r)
                est.fit(X.iloc[tr], y[tr])
                pred[te] = est.predict(X.iloc[te])
            rows.append({"f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred),
                         "precision": precision_score(y, pred), "recall": recall_score(y, pred)})
        df = pd.DataFrame(rows)
        print(f"[{name}] {t} tuned-threshold " + "  ".join(
            f"{k} {df[k].mean():.3f}±{df[k].std(ddof=0):.3f}" for k in df.columns), flush=True)
