"""Step 6: prove the recommended ensembles give EXACT additive per-feature attributions.

For each target, fit the chosen config on all 303 rows, then compute per-raw-feature
log-odds contributions as the (weighted) sum of member attributions:
  * XGBoost / XGBRF : booster.predict(DMatrix, pred_contribs=True)  (exact TreeSHAP, log-odds)
  * CatBoost        : get_feature_importance(Pool, type="ShapValues") (exact TreeSHAP, log-odds)
  * LR pipeline     : coef_j * z_j (z standardised -> training mean 0), one-hot / log columns
                      summed back to their raw column; base = intercept
and verify base + sum(contribs) == ensemble decision (log-odds) to float precision.

usage: python exp_ensemble_attrib.py <config> [targets...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings

import numpy as np
import pandas as pd

from exp_ensemble_common import NOMINAL, LogOddsVoting, PlattLogit, StackedLogit, logit
from exp_ensemble_run import CONFIGS, X, Y

warnings.filterwarnings("ignore")
RAW = list(X.columns)


def member_attr(model, Xq):
    """(contribs DataFrame n x 55 in log-odds, base array n)."""
    cls = type(model).__name__
    if cls in ("XGBClassifier", "XGBRFClassifier"):
        from xgboost import DMatrix
        c = model.get_booster().predict(DMatrix(Xq), pred_contribs=True)
        return pd.DataFrame(c[:, :-1], columns=RAW, index=Xq.index), c[:, -1]
    if cls == "CatBoostClassifier":
        from catboost import Pool
        c = model.get_feature_importance(Pool(Xq), type="ShapValues")
        return pd.DataFrame(c[:, :-1], columns=RAW, index=Xq.index), c[:, -1]
    if cls == "Pipeline":
        pre, scale, clf = model.named_steps["pre"], model.named_steps["scale"], model.named_steps["clf"]
        z = scale.transform(pre.transform(Xq))
        names = pre.get_feature_names_out()
        contrib = z * clf.coef_[0]
        out = pd.DataFrame(0.0, columns=RAW, index=Xq.index)
        for j, nm in enumerate(names):
            raw = nm if nm in RAW else next(c for c in NOMINAL if nm.startswith(c + "_"))
            out[raw] += contrib[:, j]
        return out, np.full(len(Xq), float(clf.intercept_[0]))
    raise TypeError(f"no exact log-odds attribution for {cls}")


def ensemble_attr(model, Xq):
    if isinstance(model, PlattLogit):
        c, b = ensemble_attr(model.model_, Xq)
        return c * model.a_, b * model.a_ + model.b_
    if isinstance(model, (LogOddsVoting, StackedLogit)):
        w = model.w_
        total, base = None, np.zeros(len(Xq))
        for wk, (_, m) in zip(w, model.fitted_):
            c, b = member_attr(m, Xq)
            total = c * wk if total is None else total + c * wk
            base += wk * b
        if isinstance(model, StackedLogit):
            base += model.b_
        return total, base
    return member_attr(model, Xq)


if __name__ == "__main__":
    cfg = sys.argv[1]
    targets = sys.argv[2:] or ["cad", "lad", "lcx", "rca"]
    for t in targets:
        model = CONFIGS[cfg](t)().fit(X, Y[t])
        contrib, base = ensemble_attr(model, X)
        recon = base + contrib.sum(axis=1).to_numpy()
        z = logit(model.predict_proba(X)[:, 1])
        err = np.abs(recon - z).max()
        extra = ""
        if hasattr(model, "w_"):
            extra = " weights=" + ", ".join(f"{n}:{w:.3f}" for (n, _), w in zip(model.fitted_, model.w_))
        if isinstance(model, PlattLogit):
            extra = f" platt a={model.a_:.3f} b={model.b_:.3f}"
        print(f"\n[{cfg}] {t}: max |base+sum(attr) - logit(p)| = {err:.2e}{extra}")
        top = contrib.abs().mean().sort_values(ascending=False).head(10)
        print("  global mean|attr| top10:", ", ".join(f"{k}={v:.3f}" for k, v in top.items()))
