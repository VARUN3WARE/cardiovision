"""Attribution additivity check on the ACTUALLY CLAIMED recipes (the claimant's
exp_linear_attr_check.py used L1 for RCA instead of the claimed L2+ordinal-VHD model).
Also checks held-out patients explained against a training-set background (production use)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from exp_linear_blend import LogitBlend  # noqa: E402
from exp_linear_common import CalibratedLinearCV  # noqa: E402
from exp_verify_linear_run import CLAIMED  # noqa: E402
from harness import load_xy  # noqa: E402

X, Y, _ = load_xy()
rng = np.random.default_rng(0)
te = rng.choice(len(X), 60, replace=False)
tr = np.setdiff1d(np.arange(len(X)), te)


def lin_explain(m: CalibratedLinearCV, Xe, bg):
    phi = m.attributions(Xe, background=bg)
    clf = m.pipeline_.named_steps["clf"]
    zbar = m.pipeline_.named_steps["prep"].transform(bg).mean(axis=0)
    return m.a_ * float(clf.intercept_[0] + zbar @ clf.coef_[0]) + m.b_, phi


for t in ["cad", "lad", "lcx", "rca"]:
    name, make = CLAIMED[t]
    for label, fit_idx, ex_idx in [("full", np.arange(len(X)), np.arange(len(X))), ("heldout", tr, te)]:
        model = make().fit(X.iloc[fit_idx], Y[t][fit_idx])
        Xe, bg = X.iloc[ex_idx], X.iloc[fit_idx]
        p = model.predict_proba(Xe)[:, 1]
        logit = np.log(p / (1 - p))
        if isinstance(model, LogitBlend):
            lin, w = model.a_, model.w
            b0, phi_lin = lin_explain(lin, Xe, bg)
            c = model.b_.get_booster().predict(xgb.DMatrix(Xe), pred_contribs=True)
            phi = w * phi_lin.reindex(columns=X.columns, fill_value=0.0) + (1 - w) * pd.DataFrame(
                c[:, :-1], columns=X.columns, index=Xe.index)
            base = w * b0 + (1 - w) * c[:, -1]
        else:
            lin = model
            base, phi = lin_explain(model, Xe, bg)
        err = np.abs(base + phi.sum(axis=1).to_numpy() - logit).max()
        coef = lin.pipeline_.named_steps["clf"].coef_[0]
        missing = sorted(set(X.columns) - set(phi.columns) - {"Exertional CP"})
        top = phi.abs().mean().sort_values(ascending=False).head(6)
        print(f"{t} {label:7} {name}: C={lin.C_:.3g} a={lin.a_:.3g} nnz={int((abs(coef) > 1e-10).sum())}/{len(coef)} "
              f"phi_cols={phi.shape[1]} missing={missing} max_err={err:.1e} | "
              + ", ".join(f"{k}={v:.2f}" for k, v in top.items()), flush=True)
