"""Sanity check: exact per-feature attributions for the recommended per-target recipes.

Fits each recipe on all 303 patients and checks that
    base_value + sum_j phi_j  ==  model log-odds   (for every patient)
where for CalibratedLinearCV  phi = a * coef * (z - mean z)  (aggregated to the raw columns)
and for LogitBlend            phi = w * phi_linear + (1 - w) * XGBoost pred_contribs.
Also prints the selected C, Platt slope, #non-zero coefficients and the top features.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from exp_linear_blend import LogitBlend, xgb_baseline
from exp_linear_common import CalibratedLinearCV
from harness import load_xy

X, Y, _ = load_xy()
LOG = dict(log_skewed=True)


def recipe(target):
    return {
        "cad": CalibratedLinearCV("l1", prep_kwargs=LOG),
        "lad": LogitBlend(CalibratedLinearCV("l2", prep_kwargs=LOG), xgb_baseline("lad"), w=0.7),
        "lcx": LogitBlend(CalibratedLinearCV("l1", prep_kwargs=LOG), xgb_baseline("lcx"), w=0.5),
        "rca": CalibratedLinearCV("l1", prep_kwargs=LOG),
    }[target]


def linear_explain(m: CalibratedLinearCV, X):
    phi = m.attributions(X, background=X)
    clf = m.pipeline_.named_steps["clf"]
    zbar = m.pipeline_.named_steps["prep"].transform(X).mean(axis=0)
    base = m.a_ * float(clf.intercept_[0] + zbar @ clf.coef_[0]) + m.b_
    return base, phi


for target in ["cad", "lad", "lcx", "rca"]:
    model = recipe(target).fit(X, Y[target])
    p = model.predict_proba(X)[:, 1]
    logit = np.log(p / (1 - p))
    if isinstance(model, LogitBlend):
        lin, w = model.a_, model.w
        b0, phi_lin = linear_explain(lin, X)
        contrib = model.b_.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        phi_xgb = pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)
        base = w * b0 + (1 - w) * contrib[:, -1]
        phi = w * phi_lin.reindex(columns=X.columns, fill_value=0.0) + (1 - w) * phi_xgb
    else:
        lin = model
        base, phi = linear_explain(model, X)
    err = np.abs(base + phi.sum(axis=1).to_numpy() - logit).max()
    coef = lin.pipeline_.named_steps["clf"].coef_[0]
    top = phi.abs().mean().sort_values(ascending=False).head(8)
    print(f"{target}: {type(model).__name__}  C={lin.C_:.4g}  platt_a={lin.a_:.3g}  "
          f"nonzero_coef={int((np.abs(coef) > 1e-10).sum())}/{len(coef)}  "
          f"max|base+sum(phi)-logit|={err:.2e}")
    print("   top mean|phi| (log-odds):", ", ".join(f"{k}={v:.2f}" for k, v in top.items()))
