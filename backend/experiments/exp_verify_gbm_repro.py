"""Adversarial verification of the 'gbm' strategy picks.

Models are rebuilt from the claimed PRODUCTION RECIPE (not imported from exp_gbm_fixed), so this
also checks that the recipe matches what was evaluated. The baseline is re-run too, so the
saved baseline_xgb files are not trusted blindly.

Usage (from experiments/):
  [HARNESS_SEED_OFFSET=100] python exp_verify_gbm_repro.py harness
      -> harness.evaluate for each pick on its claimed target (name verify_gbm_pick) and the
         baseline re-run (verify_gbm_baseline), all 4 targets for the baseline.
  python exp_verify_gbm_repro.py extra <offset> [<offset> ...]
      -> in-memory paired AUC (pick - baseline) on additional unseen split seeds.
"""
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from harness import evaluate, load_xy

X, Y, _ = load_xy()
assert X.shape == (303, 55), X.shape


def make_pick(target):
    if target == "cad":
        return LGBMClassifier(n_estimators=1000, max_depth=1, num_leaves=2, learning_rate=0.05,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
                              min_child_samples=10, reg_lambda=5.0, reg_alpha=0.5,
                              extra_trees=True, random_state=0, n_jobs=2, verbose=-1)
    if target == "lad":
        return XGBClassifier(n_estimators=1000, max_depth=2, learning_rate=0.01, subsample=0.7,
                             colsample_bytree=0.4, min_child_weight=5, reg_lambda=20.0,
                             reg_alpha=0.5, eval_metric="logloss", tree_method="hist",
                             random_state=0, n_jobs=2)
    if target == "lcx":
        return XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.7,
                             colsample_bytree=0.5, min_child_weight=3, reg_lambda=5.0,
                             reg_alpha=0.5, eval_metric="logloss", tree_method="hist",
                             random_state=0, n_jobs=2)
    if target == "rca":
        return LGBMClassifier(n_estimators=1200, max_depth=2, num_leaves=4, learning_rate=0.02,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
                              min_child_samples=20, reg_lambda=20.0, reg_alpha=0.5,
                              extra_trees=True, random_state=0, n_jobs=2, verbose=-1)
    raise ValueError(target)


def make_baseline(target):
    pos = Y[target].sum()
    spw = (len(Y[target]) - pos) / pos
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, n_jobs=2)


def oof(make, y, offset, repeats=5, splits=5):
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=r + offset)
        for tr, te in cv.split(X, y):
            m = make()
            m.fit(X.iloc[tr], y[tr])
            out[r, te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "harness":
        for t in ["cad", "lad", "lcx", "rca"]:
            evaluate("verify_gbm_pick", t, lambda tt=t: make_pick(tt), X, Y)
            evaluate("verify_gbm_baseline", t, lambda tt=t: make_baseline(tt), X, Y)
    elif mode == "extra":
        for off in map(int, sys.argv[2:]):
            for t in ["cad", "lad", "lcx", "rca"]:
                y = Y[t]
                A = oof(lambda: make_pick(t), y, off)
                B = oof(lambda: make_baseline(t), y, off)
                a = np.array([roc_auc_score(y, A[r]) for r in range(5)])
                b = np.array([roc_auc_score(y, B[r]) for r in range(5)])
                d = a - b
                print(f"offset {off} {t}: pick {a.mean():.4f}±{a.std():.4f}  base {b.mean():.4f}"
                      f"±{b.std():.4f}  delta mean {d.mean():+.4f} std {d.std():.4f} "
                      f"min {d.min():+.4f} max {d.max():+.4f} per_rep {np.round(d, 4).tolist()}",
                      flush=True)
