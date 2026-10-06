"""Final decision: cross-strategy head-to-head from saved OOF arrays (no refitting).

For every target, loads each verified candidate's OOF on every split set where it exists
(seed offsets 0 / 100 / 200 / 300), pairs it with the baseline on identical folds and reports
AUC, PR-AUC, Brier, log-loss and paired AUC delta (mean / min over the 5 repeats).
Also pairs candidates against each other where both exist.

usage: python exp_final_compare.py [extra_name=file_stem@seed ...]
"""
from __future__ import annotations

import itertools
import sys
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from harness import RESULTS_DIR, load_xy

warnings.filterwarnings("ignore")
_, Y, _ = load_xy()
SEEDS = [0, 100, 200, 300, 400]

BASE = {0: "baseline_xgb", 100: "baseline_xgb", 200: "verify_features_baseline_xgb",
        300: "verify_ensemble_base", 400: "final_baseline_xgb"}

# candidate -> {seed: stem}; a stem may be a list -> try in order
CAND = {
    "cad": {
        "gbm_lgb_et_d1": {0: "gbm_lgb_et_d1", 100: "gbm_lgb_et_d1"},
        "features_lr_engcore": {0: "features_lr_engcore", 100: "features_lr_engcore",
                                200: "verify_features_lr_engcore"},
        "features_blend_lr_xgb": {0: "features_blend_lr_xgb", 100: "features_blend_lr_xgb"},
        "linear_cl1_log": {0: "linear_cl1_log", 100: "linear_cl1_log", 200: "verify_linear_cl1_log"},
        "ensemble_vote_l1c": {0: "ensemble_vote_l1c", 100: "ensemble_vote_l1c",
                              200: "verify_ensemble_cad_l1c", 300: "verify_ensemble_cad_l1c"},
        "multitask_h2_L_indep": {0: "multitask_h2_L_indep", 100: "multitask_h2_L_indep",
                                 200: "verify_multitask_cad_L"},
        "forest_final_blend": {0: "forest_final_blend", 100: "forest_final_blend",
                               200: "verify_forest_blend"},
    },
    "lad": {
        "features_blend_lrcal_xgb": {0: "features_blend_lrcal_xgb", 100: "features_blend_lrcal_xgb",
                                     200: "verify_features_blend_lrcal_xgb"},
        "features_blend_lr_xgb": {0: "features_blend_lr_xgb", 100: "features_blend_lr_xgb"},
        "multitask_h3avg_XLcX": {0: "multitask_h3avg_XLcX", 100: "multitask_h3avg_XLcX",
                                 200: "verify_multitask_lad_XLcX"},
        "linear_blendl2_w70": {0: "linear_blendl2_w70", 100: "linear_blendl2_w70",
                               200: "verify_linear_blendl2_w70"},
        "ensemble_platt_xrf": {0: "ensemble_platt_xrf", 100: "verify_ensemble_lad_plattxrf",
                               200: "verify_ensemble_lad_plattxrf", 300: "verify_ensemble_lad_plattxrf"},
        "forest_final_rf": {0: "forest_final_rf", 100: "forest_final_rf", 200: "verify_forest_rf"},
        "forest_final_blend": {0: "forest_final_blend", 100: "forest_final_blend", 200: "verify_forest_blend"},
        "gbm_xgb_d2_slow_l20": {0: "gbm_xgb_d2_slow_l20", 100: "gbm_xgb_d2_slow_l20"},
    },
    "lcx": {
        "features_blend_lrcal_xgb": {0: "features_blend_lrcal_xgb", 100: "features_blend_lrcal_xgb",
                                     200: "verify_features_blend_lrcal_xgb"},
        "features_blend_lr_xgb": {0: "features_blend_lr_xgb", 100: "features_blend_lr_xgb"},
        "features_xgb_eng": {0: "features_xgb_eng", 100: "features_xgb_eng"},
        "linear_blendl1_w50": {0: "linear_blendl1_w50", 100: "linear_blendl1_w50",
                               200: "verify_linear_blendl1_w50"},
        "ensemble_vote_l1x": {0: "ensemble_vote_l1x", 100: "ensemble_vote_l1x",
                              200: "verify_ensemble_lcx_l1x", 300: "verify_ensemble_lcx_l1x"},
        "gbm_xgb_d3": {0: "gbm_xgb_d3", 100: "gbm_xgb_d3"},
        "multitask_h3avg_XLc": {0: "multitask_h3avg_XLc", 100: "multitask_h3avg_XLc",
                                200: "verify_multitask_lcx_XLc"},
        "forest_final_rf": {0: "forest_final_rf", 100: "forest_final_rf", 200: "verify_forest_rf"},
    },
    "rca": {
        "features_lr_eng_l1": {0: "features_lr_eng_l1", 100: "features_lr_eng_l1",
                               200: "verify_features_lr_eng_l1"},
        "gbm_lgb_et_d2_reg": {0: "gbm_lgb_et_d2_reg", 100: "gbm_lgb_et_d2_reg"},
        "multitask_h3avg_Lc": {0: "multitask_h3avg_Lc", 100: "multitask_h3avg_Lc",
                               200: "verify_multitask_rca_Lc"},
        "ensemble_vote_l1f": {0: "ensemble_vote_l1f", 100: "ensemble_vote_l1f",
                              200: "verify_ensemble_rca_l1f", 300: "verify_ensemble_rca_l1f"},
        "linear_cl2_logord": {0: "linear_cl2_logord", 100: "linear_cl2_logord",
                              200: "verify_linear_cl2_logord"},
        "forest_final_blend": {0: "forest_final_blend", 100: "forest_final_blend", 200: "verify_forest_blend"},
        "features_lr_engcore": {0: "features_lr_engcore", 100: "features_lr_engcore"},
    },
}


def load(stem, t, seed):
    sfx = f"__seed{seed}" if seed else ""
    p = RESULTS_DIR / f"{stem}__{t}{sfx}.oof.npy"
    return np.load(p) if p.exists() else None


def per_rep(y, P, fn):
    return np.array([fn(y, np.clip(P[r], 1e-6, 1 - 1e-6)) for r in range(len(P))])


def summary(t, cands):
    y = Y[t]
    print(f"\n=================== {t.upper()} ===================")
    print(f"{'model':28} {'seed':>4} {'AUC':>6} {'PRAUC':>6} {'Brier':>6} {'LL':>6} {'dAUC':>7} {'dmin':>7}")
    rows = {}
    for seed in SEEDS:
        B = load(BASE[seed], t, seed)
        if B is None:
            continue
        b_auc = per_rep(y, B, roc_auc_score)
        print(f"{'baseline_xgb':28} {seed:>4} {b_auc.mean():6.3f} {per_rep(y, B, average_precision_score).mean():6.3f} "
              f"{per_rep(y, B, brier_score_loss).mean():6.3f} {per_rep(y, B, log_loss).mean():6.3f}")
        for name, m in cands.items():
            if seed not in m:
                continue
            P = load(m[seed], t, seed)
            if P is None:
                continue
            a = per_rep(y, P, roc_auc_score)
            d = a - b_auc
            rows.setdefault(name, {})[seed] = P
            print(f"{name:28} {seed:>4} {a.mean():6.3f} {per_rep(y, P, average_precision_score).mean():6.3f} "
                  f"{per_rep(y, P, brier_score_loss).mean():6.3f} {per_rep(y, P, log_loss).mean():6.3f} "
                  f"{d.mean():+7.4f} {d.min():+7.4f}")
    # pooled over all unseen seeds (>=100) where available
    print(f"-- pooled unseen (seeds>=100) paired delta vs baseline")
    for name, s in rows.items():
        ds = []
        for seed, P in s.items():
            if seed == 0:
                continue
            B = load(BASE[seed], t, seed)
            ds.extend(per_rep(y, P, roc_auc_score) - per_rep(y, B, roc_auc_score))
        if ds:
            ds = np.array(ds)
            print(f"   {name:28} n_rep={len(ds):2d} mean {ds.mean():+.4f} sd {ds.std():.4f} min {ds.min():+.4f} "
                  f"pos {np.mean(ds > 0):.2f}")
    print("-- head-to-head (A - B), all shared seeds incl. 0")
    for a, b in itertools.combinations(rows, 2):
        shared = sorted(set(rows[a]) & set(rows[b]))
        if not shared:
            continue
        ds = np.concatenate([per_rep(y, rows[a][s], roc_auc_score) - per_rep(y, rows[b][s], roc_auc_score)
                             for s in shared])
        if abs(ds.mean()) >= 0.004 or True:
            print(f"   {a:26} - {b:26} seeds {shared}: mean {ds.mean():+.4f} min {ds.min():+.4f} "
                  f"max {ds.max():+.4f} pos {np.mean(ds > 0):.2f}")
    return rows


if __name__ == "__main__":
    targets = sys.argv[1:] or ["cad", "lad", "lcx", "rca"]
    for t in targets:
        summary(t, CAND[t])
