"""gbm strategy, part 1: fixed, strongly-regularised GBM configs per library (no search).

Usage: python exp_gbm_fixed.py [config ...]   (default: all configs)
Experiment names: gbm_<config>. Nothing is learned outside the estimator.
"""
import sys
import time

from harness import evaluate_all, load_xy
from exp_gbm_common import (CatNative, MarginAvg, MarginBag, NestedSelect, make_cat, make_lgb,
                             make_xgb)

_, Y, _ = load_xy()


def spw(t):
    pos = Y[t].sum()
    return (len(Y[t]) - pos) / pos


CONFIGS = {
    # --- XGBoost
    # baseline params minus scale_pos_weight (isolates the calibration effect of spw)
    "xgb_base_nospw": lambda t: make_xgb(n_estimators=400, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                                         reg_lambda=1.0, reg_alpha=0.0, random_state=42, n_jobs=2),
    "xgb_d1": lambda t: make_xgb(max_depth=1, n_estimators=600, learning_rate=0.05, n_jobs=2),
    "xgb_d2": lambda t: make_xgb(max_depth=2, n_jobs=2),
    "xgb_d3": lambda t: make_xgb(max_depth=3, n_jobs=2),
    "xgb_d2_slow": lambda t: make_xgb(max_depth=2, n_estimators=1000, learning_rate=0.01,
                                      colsample_bytree=0.4, n_jobs=2),
    "xgb_d2_spw": lambda t: make_xgb(max_depth=2, scale_pos_weight=spw(t), n_jobs=2),
    # --- LightGBM
    "lgb_d1": lambda t: make_lgb(max_depth=1, num_leaves=2, n_estimators=600, learning_rate=0.05,
                                 n_jobs=2),
    "lgb_d2": lambda t: make_lgb(n_jobs=2),
    "lgb_d3": lambda t: make_lgb(max_depth=3, num_leaves=8, n_jobs=2),
    "lgb_d2_et": lambda t: make_lgb(extra_trees=True, n_estimators=800, n_jobs=2),
    # extra-trees (random split thresholds) variants
    "lgb_et_d1": lambda t: make_lgb(extra_trees=True, max_depth=1, num_leaves=2,
                                    n_estimators=1000, learning_rate=0.05),
    "lgb_et_d3": lambda t: make_lgb(extra_trees=True, max_depth=3, num_leaves=8, n_estimators=800),
    "lgb_et_d2_slow": lambda t: make_lgb(extra_trees=True, n_estimators=1500, learning_rate=0.015),
    "lgb_et_d2_cs03": lambda t: make_lgb(extra_trees=True, n_estimators=800, colsample_bytree=0.3),
    "lgb_et_d2_mcs20": lambda t: make_lgb(extra_trees=True, n_estimators=800, min_child_samples=20),
    "lgb_et_d2_l20": lambda t: make_lgb(extra_trees=True, n_estimators=1000, reg_lambda=20.0),
    "lgb_et_d2_cs08": lambda t: make_lgb(extra_trees=True, n_estimators=800, colsample_bytree=0.8),
    "lgb_et_d2_reg": lambda t: make_lgb(extra_trees=True, n_estimators=1200, learning_rate=0.02,
                                        min_child_samples=20, reg_lambda=20.0),
    "xgb_d3_slow": lambda t: make_xgb(max_depth=3, n_estimators=1000, learning_rate=0.01,
                                      colsample_bytree=0.5),
    "xgb_d2_cs08": lambda t: make_xgb(max_depth=2, colsample_bytree=0.8),
    "xgb_d2_slow_l20": lambda t: make_xgb(max_depth=2, n_estimators=1000, learning_rate=0.01,
                                          colsample_bytree=0.4, reg_lambda=20.0,
                                          min_child_weight=5),
    "xgb_d4_reg": lambda t: make_xgb(max_depth=4, min_child_weight=3, reg_lambda=5.0),
    "lgb_et_d2_reg2": lambda t: make_lgb(extra_trees=True, n_estimators=2000, learning_rate=0.01,
                                         min_child_samples=30, reg_lambda=50.0),
    # log-odds average of the two strongest regularised families
    "avg_lgbet_xgb": lambda t: MarginAvg([CONFIGS["lgb_et_d2_reg"](t),
                                          CONFIGS["xgb_d2_slow_l20"](t)]),
    # honest per-fold model selection among a few regularised configs (inner 2x3-fold CV)
    "select5": lambda t: NestedSelect({k: CONFIGS[k](t) for k in
                                       ["lgb_et_d1", "lgb_d2_et", "lgb_et_d2_reg",
                                        "xgb_d2_slow_l20", "xgb_d3"]}),
    "lgb_et_d2_bag5": lambda t: MarginBag(make_lgb(extra_trees=True, n_estimators=800), 5,
                                          "random_state"),
    # --- CatBoost
    "cat_default": lambda t: make_cat(iterations=1000, depth=6, learning_rate=None,
                                      l2_leaf_reg=3, rsm=None, thread_count=2),
    "cat_d2": lambda t: make_cat(depth=2, thread_count=2),
    "cat_d3": lambda t: make_cat(depth=3, thread_count=2),
    "cat_d4": lambda t: make_cat(depth=4, thread_count=2),
    "catn_d3": lambda t: CatNative(params=dict(depth=3, thread_count=2)),
    "catn_d3_oh": lambda t: CatNative(params=dict(depth=3, thread_count=2, one_hot_max_size=4)),
    "cat_d3_bag5": lambda t: MarginBag(make_cat(depth=3, thread_count=2), 5, "random_seed"),
    "xgb_d2_bag5": lambda t: MarginBag(make_xgb(max_depth=2, n_jobs=2), 5, "random_state"),
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for name in names:
        t0 = time.time()
        builder = CONFIGS[name]
        evaluate_all(f"gbm_{name}", lambda t, b=builder: (lambda: b(t)))
        print(f"== gbm_{name} done in {time.time() - t0:.0f}s", flush=True)
