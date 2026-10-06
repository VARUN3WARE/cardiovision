"""Adversarial checks of the forest agent's production claims.

1. Their RFETLogitBlend (exp_forest_common) vs an independent construction from the recipe:
   identical predict_proba on held-out rows?
2. explain(): base + sum(phi) == logit(predict_proba) exactly?  Includes the extreme rows
   (highest / lowest predicted risk) where the 1e-6 clip could break additivity.
3. Label-permutation sanity check of the full fitting procedure (single member, 1 repeat):
   AUC should be ~0.5 if nothing leaks.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from exp_forest_common import RFETLogitBlend
from exp_verify_forest_run import member
from harness import load_xy


def main():
    X, Y, _ = load_xy()
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    for t in ("rca", "lad"):
        y = Y[t]
        tr, te = next(cv.split(X, y))
        t0 = time.time()
        theirs = RFETLogitBlend().fit(X.iloc[tr], y[tr])
        mine = [member(a).fit(X.iloc[tr], y[tr]) for a in ("rf", "et")]
        p_theirs = theirs.predict_proba(X.iloc[te])[:, 1]
        z = np.mean([logit(np.clip(m.predict_proba(X.iloc[te])[:, 1], 1e-6, 1 - 1e-6))
                     for m in mine], axis=0)
        p_mine = 1 / (1 + np.exp(-z))
        print(f"{t}: max |p_theirs - p_mine| = {np.abs(p_theirs - p_mine).max():.2e}  "
              f"(fit {time.time() - t0:.0f}s)")
        sel = np.r_[np.argsort(p_theirs)[:5], np.argsort(p_theirs)[-5:]]
        Xs = X.iloc[te].iloc[sel]
        t0 = time.time()
        base, phi = theirs.explain(Xs)
        recon = base + phi.sum(1)
        target = logit(theirs.predict_proba(Xs)[:, 1])
        print(f"{t}: explain additivity max err = {np.abs(recon - target).max():.2e}  "
              f"p range [{p_theirs.min():.3f}, {p_theirs.max():.3f}]  "
              f"phi shape {phi.shape}  ({(time.time() - t0) / len(sel):.2f}s/row)")
        for mm in theirs.members_:
            cc = mm.calibrated_classifiers_[0]
            print(f"   member {type(cc.estimator).__name__}: Platt a={cc.calibrators[0].a_:.3f} "
                  f"b={cc.calibrators[0].b_:.3f}")

    # permutation sanity (RF member only, 1 repeat x 5 folds, 2 permutations per target)
    rng = np.random.default_rng(123)
    for t in ("cad", "rca"):
        aucs = []
        for _ in range(2):
            yp = rng.permutation(Y[t])
            oof = np.zeros(len(yp))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, yp):
                oof[te] = member("rf").fit(X.iloc[tr], yp[tr]).predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(yp, oof))
        print(f"permuted-label AUC {t}: {np.round(aucs, 3)}")


if __name__ == "__main__":
    main()
