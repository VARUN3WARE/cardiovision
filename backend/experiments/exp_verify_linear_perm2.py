"""Extended label-permutation null for the claimed pipelines and the baseline (8 shuffles each)."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.metrics import roc_auc_score
from harness import load_xy, oof_probabilities
from exp_verify_linear_run import CLAIMED
from exp_linear_blend import xgb_baseline

X, Y, _ = load_xy()
targets = sys.argv[1:] or ["cad", "lad", "lcx", "rca"]
for t in targets:
    name, make = CLAIMED[t]
    for label, mk in [(name, make), ("baseline", lambda t=t: xgb_baseline(t))]:
        rng = np.random.default_rng(777)
        aucs = []
        for k in range(8):
            yp = rng.permutation(Y[t])
            P = oof_probabilities(mk, X, yp, repeats=1)
            aucs.append(roc_auc_score(yp, P[0]))
        print(f"perm2 {t} {label}: mean {np.mean(aucs):.3f} sd {np.std(aucs):.3f} max {np.max(aucs):.3f} {np.round(aucs,3).tolist()}", flush=True)
