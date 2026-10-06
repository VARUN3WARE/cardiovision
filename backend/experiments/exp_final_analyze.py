"""Analyse the pre-declared fresh-seed head-to-head (exp_final_fresh.py) over seed offsets
400..800 (25 repeats nobody selected on). Also evaluates two pre-declared equal-weight log-odds
combos computed exactly from member OOFs (members are refit on identical folds):
  cad: combo_lrcore_lgbd1 = avg_logit(features_lr_engcore, gbm_lgb_et_d1)
  rca: combo_l1_lgbd2     = avg_logit(features_lr_eng_l1, gbm_lgb_et_d2_reg)

usage: python exp_final_analyze.py [offsets...]
"""
import itertools
import sys
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from harness import RESULTS_DIR, load_xy

warnings.filterwarnings("ignore")
_, Y, _ = load_xy()
OFFS = [int(a) for a in sys.argv[1:]] or [400, 500, 600, 700, 800]
COMBOS = {"cad": {"combo_lrcore_lgbd1": ["features_lr_engcore", "gbm_lgb_et_d1"]},
          "rca": {"combo_l1_lgbd2": ["features_lr_eng_l1", "gbm_lgb_et_d2_reg"]}}
NAMES = {
    "cad": ["features_lr_engcore", "gbm_lgb_et_d1", "linear_cl1_log"],
    "lad": ["features_blend_lrcal_xgb", "features_blend_lr_xgb", "features_xgb_eng", "linear_blendl2_w70"],
    "lcx": ["features_blend_lrcal_xgb", "features_blend_lr_xgb", "features_xgb_eng", "ensemble_vote_l1x"],
    "rca": ["features_lr_eng_l1", "features_lr_engcore", "ensemble_vote_l1f", "gbm_lgb_et_d2_reg"],
}


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(name, t):
    arrs = []
    for o in OFFS:
        f = RESULTS_DIR / f"final_{name}__{t}__seed{o}.oof.npy"
        if not f.exists():
            return None
        arrs.append(np.load(f))
    return np.concatenate(arrs)  # (5*len(OFFS), n)


def met(y, P, fn):
    return np.array([fn(y, np.clip(p, 1e-6, 1 - 1e-6)) for p in P])


for t in ["cad", "lad", "lcx", "rca"]:
    y = Y[t]
    B = load("baseline_xgb", t)
    if B is None:
        print(t, "baseline missing"); continue
    M = {n: load(n, t) for n in NAMES[t]}
    for c, mem in COMBOS.get(t, {}).items():
        if all(M.get(m) is not None for m in mem):
            M[c] = 1 / (1 + np.exp(-np.mean([lg(M[m]) for m in mem], axis=0)))
    M = {k: v for k, v in M.items() if v is not None}
    ba = met(y, B, roc_auc_score)
    print(f"\n==== {t.upper()}  fresh offsets {OFFS}  ({len(ba)} repeats) ====")
    print(f"{'model':26} {'AUC':>6} {'sd':>6} {'PRAUC':>6} {'Brier':>6} {'LL':>6} {'dAUC':>7} {'d_sd':>6} {'dmin':>7} {'pos':>5}")
    print(f"{'baseline_xgb':26} {ba.mean():6.3f} {ba.std():6.3f} {met(y, B, average_precision_score).mean():6.3f} "
          f"{met(y, B, brier_score_loss).mean():6.3f} {met(y, B, log_loss).mean():6.3f}")
    A = {}
    for n, P in M.items():
        a = met(y, P, roc_auc_score)
        A[n] = a
        d = a - ba
        print(f"{n:26} {a.mean():6.3f} {a.std():6.3f} {met(y, P, average_precision_score).mean():6.3f} "
              f"{met(y, P, brier_score_loss).mean():6.3f} {met(y, P, log_loss).mean():6.3f} "
              f"{d.mean():+7.4f} {d.std():6.4f} {d.min():+7.4f} {np.mean(d > 0):5.2f}")
    print("  head-to-head A-B: mean / min / max / frac>0")
    for a, b in itertools.combinations(A, 2):
        d = A[a] - A[b]
        print(f"   {a:26} - {b:26} {d.mean():+.4f} {d.min():+.4f} {d.max():+.4f} {np.mean(d > 0):.2f}")
