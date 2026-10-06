"""Extra adversarial checks for the 'linear' claims.

  perm   : label-permutation test. The claimed pipelines are run through the harness CV
           with shuffled labels; any leakage would show up as AUC well above 0.5.
  perfold: paired delta using the MEAN WITHIN-FOLD AUC (folds rebuilt from the same seeds)
           instead of the harness's pooled-OOF AUC. If the gain only exists pooled, it is a
           cross-fold probability-scale artefact, not better ranking.

    python exp_verify_linear_checks.py perm
    python exp_verify_linear_checks.py perfold <exp_a> <exp_b> <seed_offset>
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from harness import RESULTS_DIR, load_xy, oof_probabilities  # noqa: E402


def perm():
    from exp_verify_linear_run import CLAIMED
    X, Y, _ = load_xy()
    rng = np.random.default_rng(12345)
    for t in ["cad", "lad", "lcx", "rca"]:
        name, make = CLAIMED[t]
        aucs = []
        for k in range(2):
            yp = rng.permutation(Y[t])
            P = oof_probabilities(make, X, yp, repeats=1)
            aucs.append(roc_auc_score(yp, P[0]))
        print(f"perm {t} {name}: shuffled-label AUC {np.round(aucs, 3)}", flush=True)


def perfold(a, b, off):
    X, Y, _ = load_xy()
    sfx = f"__seed{off}" if off else ""
    for t in ["cad", "lad", "lcx", "rca"]:
        try:
            A = np.load(RESULTS_DIR / f"{a}__{t}{sfx}.oof.npy")
            B = np.load(RESULTS_DIR / f"{b}__{t}{sfx}.oof.npy")
        except FileNotFoundError:
            continue
        y = Y[t]
        d_fold, d_pool = [], []
        for r in range(A.shape[0]):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=r + off)
            fa, fb = [], []
            for _, te in cv.split(X, y):
                fa.append(roc_auc_score(y[te], A[r, te]))
                fb.append(roc_auc_score(y[te], B[r, te]))
            d_fold.append(np.mean(fa) - np.mean(fb))
            d_pool.append(roc_auc_score(y, A[r]) - roc_auc_score(y, B[r]))
        d_fold, d_pool = np.array(d_fold), np.array(d_pool)
        print(f"{t} {a} vs {b} seed{off}: pooled d mean {d_pool.mean():+.4f} min {d_pool.min():+.4f} | "
              f"within-fold d mean {d_fold.mean():+.4f} min {d_fold.min():+.4f} "
              f"per-repeat {np.round(d_fold, 4).tolist()}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "perm":
        perm()
    else:
        perfold(sys.argv[2], sys.argv[3], int(sys.argv[4]))
