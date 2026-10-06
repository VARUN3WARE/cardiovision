"""gbm strategy, part 3: verify selected fixed configs on UNSEEN splits and report
tuned-threshold F1 + attribution faithfulness.

Usage:
  HARNESS_SEED_OFFSET=100 python exp_gbm_verify.py rerun <config> [config ...]
      -> re-evaluates exp_gbm_fixed CONFIGS on seeds 100..104 (files ...__seed100)
  python exp_gbm_verify.py thr <config> <target> [<target> ...]
      -> nested tuned-threshold F1/accuracy/precision/recall on seeds 0..4
  python exp_gbm_verify.py shap <config>
      -> fits on all data, checks LightGBM/XGBoost/CatBoost contributions sum to the margin
"""
import json
import sys

import numpy as np

from harness import RESULTS_DIR, SEED_OFFSET, evaluate, load_xy, tuned_threshold_metrics
from exp_gbm_common import CatNative, MarginAvg, MarginBag, NestedSelect
from exp_gbm_fixed import CONFIGS

X, Y, _ = load_xy()


def contribs(model, X):
    """Per-feature log-odds attributions (+ bias as last column) for supported models."""
    if isinstance(model, MarginBag):
        return np.mean([contribs(m, X) for m in model.models_], 0)
    if isinstance(model, MarginAvg):
        w = np.ones(len(model.models_)) if model.weights is None else np.asarray(model.weights, float)
        return np.tensordot(w / w.sum(), np.array([contribs(m, X) for m in model.models_]), 1)
    if isinstance(model, NestedSelect):
        return contribs(model.model_, X)
    if isinstance(model, CatNative):
        return model.shap_values(X)
    name = type(model).__name__
    if name == "LGBMClassifier":
        return model.predict(X, pred_contrib=True)
    if name == "XGBClassifier":
        import xgboost as xgb
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    if name == "CatBoostClassifier":
        from catboost import Pool
        return model.get_feature_importance(Pool(X), type="ShapValues")
    raise TypeError(name)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rerun":
        assert SEED_OFFSET != 0, "set HARNESS_SEED_OFFSET"
        for cfg in sys.argv[2:]:
            for t in ["cad", "lad", "lcx", "rca"]:
                evaluate(f"gbm_{cfg}", t, lambda c=cfg, tt=t: CONFIGS[c](tt), X, Y)
    elif mode == "thr":
        cfg, targets = sys.argv[2], sys.argv[3:]
        out = {}
        for t in targets:
            m = tuned_threshold_metrics(f"gbm_{cfg}", t, lambda: CONFIGS[cfg](t), X, Y)
            out[t] = m
            print(cfg, t, {k: round(v["mean"], 3) for k, v in m.items()}, flush=True)
            with open(RESULTS_DIR / f"gbm_{cfg}__{t}.thr.json", "w") as f:
                json.dump(m, f, indent=2)
    elif mode == "shap":
        cfg = sys.argv[2]
        for t in ["cad", "lad", "lcx", "rca"]:
            m = CONFIGS[cfg](t).fit(X, Y[t])
            C = contribs(m, X)
            p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
            margin = np.log(p / (1 - p))
            err = np.abs(C.sum(1) - margin).max()
            top = np.argsort(-np.abs(C[:, :-1]).mean(0))[:8]
            print(t, cfg, "contrib shape", C.shape, "max|sum-margin|", f"{err:.2e}",
                  "top:", [X.columns[i] for i in top], flush=True)
