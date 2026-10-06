"""Step 2: screen fixed equal-weight log-odds votes from the saved member OOF arrays.

sigmoid(mean_k logit(OOF_k)) is exactly the OOF of LogOddsVoting(members) because each
member is deterministic and was fit on the identical training folds. This is only a SCREEN
(choosing among many subsets on the same splits is optimistic); the chosen configs are re-run
as real estimators (exp_ensemble_run.py) and re-checked on unseen splits (seed offset 100).
"""
import itertools
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from harness import RESULTS_DIR, SEED_OFFSET, load_xy
from exp_ensemble_common import logit, sigmoid

_, Y, _ = load_xy()
POOL = sys.argv[1:] or ["lr", "lr_l1", "xgb", "xgb_nospw", "cat", "cat_d6", "rf", "xgbrf"]


def load(m, t):
    name = "baseline_xgb" if m == "base" else f"ensemble_m_{m}"
    sfx = f"__seed{SEED_OFFSET}" if SEED_OFFSET else ""
    return np.load(RESULTS_DIR / f"{name}__{t}{sfx}.oof.npy")


if __name__ == "__main__":
    for t in ["cad", "lad", "lcx", "rca"]:
        y = Y[t]
        B = load("base", t)
        base = np.array([roc_auc_score(y, B[r]) for r in range(len(B))])
        oofs = {}
        for m in POOL:
            try:
                oofs[m] = load(m, t)
            except FileNotFoundError:
                pass
        if "xgb" in oofs:
            print(f"[{t}] xgb member reproduces baseline exactly:", np.allclose(oofs["xgb"], B, atol=1e-6))
        rows = []
        names = list(oofs)
        for k in range(1, len(names) + 1):
            for combo in itertools.combinations(names, k):
                if "xgb" in combo and "xgb_nospw" in combo:
                    continue
                if "cat" in combo and "cat_d6" in combo:
                    continue
                if "lr" in combo and "lr_l1" in combo:
                    continue
                P = sigmoid(np.mean([logit(oofs[m]) for m in combo], axis=0))
                auc = np.array([roc_auc_score(y, P[r]) for r in range(len(P))])
                d = auc - base
                brier = np.mean([np.mean((P[r] - y) ** 2) for r in range(len(P))])
                rows.append((auc.mean(), auc.std(), d.mean(), d.std(), d.min(), brier, "+".join(combo)))
        rows.sort(key=lambda r: -r[0])
        print(f"\n=== {t}  baseline AUC {base.mean():.4f}  (top 15 of {len(rows)}; singles marked *) ===")
        print(f"{'AUC':>7} {'std':>6} {'dMean':>7} {'dStd':>6} {'dMin':>7} {'Brier':>6}  combo")
        for r in rows[:15]:
            print(f"{r[0]:.4f} {r[1]:.4f} {r[2]:+.4f} {r[3]:.4f} {r[4]:+.4f} {r[5]:.4f}  {r[6]}")
        print("singles / spec combos:")
        for r in rows:
            if "+" not in r[6] or r[6] in ("lr+xgb+cat+rf", "lr+xgb+cat+xgbrf", "lr+xgb+cat", "lr+cat", "lr+xgb"):
                print(f"{r[0]:.4f} {r[1]:.4f} {r[2]:+.4f} {r[3]:.4f} {r[4]:+.4f} {r[5]:.4f}  {r[6]}")
