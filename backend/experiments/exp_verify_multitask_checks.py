"""Side checks for the multitask claims (no result files written).

1. FastRidgeLogitCV == sklearn LogisticRegression at the selected C (pre-Platt coefficients).
2. Test-label probe: flipping the TEST rows' labels (all 4 tasks) in the shared Y_ALL table
   before fitting the authors' MultiTask must not change test predictions.
3. Permutation null: RCA stack [rca:lrlog, cad:lrlog] with permuted RCA labels -> AUC ~ 0.5.
4. Within-fold vs pooled AUC for the RCA winner and baseline (does Platt pooling explain gain?).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import exp_multitask_common as C
from exp_verify_multitask_run import CleanStack
from harness import RESULTS_DIR

X, Y = C.X_ALL, C.Y_DICT

# ---- 1
m = C.make_base("lrlog", Y["rca"]).fit(X, Y["rca"])
pre, lr = m.named_steps["pre"], m.named_steps["lr"]
Z = pre.transform(X)
a, b = lr.platt_
w_fast = lr.coef_[0] / a
sk = LogisticRegression(C=lr.C_, max_iter=20000, tol=1e-10).fit(Z, Y["rca"])
print(f"[1] C_={lr.C_:.4g} platt=({a:.3f},{b:.3f}) max|coef diff| = "
      f"{np.abs(w_fast - sk.coef_[0]).max():.2e}  intercept diff "
      f"{abs((lr.intercept_[0] - b) / a - sk.intercept_[0]):.2e}", flush=True)

# ---- 2
rng = np.random.default_rng(0)
cv = StratifiedKFold(5, shuffle=True, random_state=7)
tr, te = next(cv.split(X, Y["rca"]))
kw = dict(target="rca", mode="stack", members=("rca:lrlog", "cad:lrlog"), meta="avg")
C._CACHE.clear()
p1 = C.MultiTask(**kw).fit(X.iloc[tr], Y["rca"][tr]).predict_proba(X.iloc[te])[:, 1]
saved = C.Y_ALL.copy()
C.Y_ALL.iloc[te] = 1 - C.Y_ALL.iloc[te].to_numpy()
C._CACHE.clear()
p2 = C.MultiTask(**kw).fit(X.iloc[tr], Y["rca"][tr]).predict_proba(X.iloc[te])[:, 1]
C.Y_ALL.loc[:, :] = saved.to_numpy()
C._CACHE.clear()
print(f"[2] max |dp| after flipping test-row labels = {np.abs(p1 - p2).max():.2e}", flush=True)

# ---- 3
aucs = []
for s in range(3):
    yp = rng.permutation(Y["rca"])
    oof = np.zeros(len(yp))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, yp):
        est = CleanStack(target="rca", members=(("rca", "lrlog"), ("cad", "lrlog")),
                         permute_target=True).fit(X.iloc[tr], yp[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    aucs.append(roc_auc_score(yp, oof))
print(f"[3] permuted-RCA stack AUCs: {np.round(aucs, 3)}", flush=True)

# ---- 4 within-fold AUC (mean over 25 folds) vs pooled, from saved OOF
for name, t in [("multitask_h3avg_Lc", "rca"), ("baseline_xgb", "rca"),
                ("multitask_h3avg_XLcX", "lad"), ("baseline_xgb", "lad"),
                ("multitask_h2_L_indep", "cad"), ("baseline_xgb", "cad")]:
    P = np.load(RESULTS_DIR / f"{name}__{t}.oof.npy")
    y = Y[t]
    within = []
    for r in range(P.shape[0]):
        for _, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            within.append(roc_auc_score(y[te], P[r, te]))
    pooled = np.mean([roc_auc_score(y, P[r]) for r in range(P.shape[0])])
    print(f"[4] {name:24s} {t}: pooled {pooled:.4f}  within-fold mean {np.mean(within):.4f}")
