"""Production-style fit of the multi-task winners + exact per-feature attribution check.

Stack winners (meta="avg"): final_logit(x) = b + sum_k w_k * logit_k(x),  w_k = a/K >= 0
  XGB member   : logit_k(x) = bias_k + sum_j phi_kj(x)          (booster pred_contribs, exact)
  lrlog member : logit_k(x) = icpt + coef . z(x), z = [scaled log1p/numeric cols, one-hot BBB/VHD]
                 phi_kj(x)  = sum over z-columns of raw column j of coef * (z(x) - mean_bg z)
                 base_k     = icpt + coef . mean_bg z                (exact, interventional SHAP)
=> phi_j(x) = sum_k w_k phi_kj(x);  base(x) = b + sum_k w_k base_k;  base + sum_j phi_j = logit.
Exact up to the 1e-4 probability clip on member logits (checked below on unclipped rows).
CAD winner is a single lrlog model (exact linear attribution, weight 1).

usage: python exp_multitask_attrib.py
"""
import numpy as np
import xgboost as xgb

import exp_multitask_common as C

RECIPES = {  # per-target winners (see exp_multitask_report.py)
    "cad": None,  # single-task lrlog
    "lad": ("lad:xgb", "lad:lrlog", "cad:lrlog", "cad:xgb"),
    "lcx": ("lcx:xgb", "lcx:lrlog", "cad:lrlog"),
    "rca": ("rca:lrlog", "cad:lrlog"),
}


def _raw_columns(pre, cols):
    """Raw column for every output column of a fitted ColumnTransformer."""
    out = []
    for name, trans, sel in pre.transformers_:
        if trans == "drop":
            continue
        sel = [cols[i] if isinstance(i, (int, np.integer)) else i for i in sel]
        if hasattr(trans, "categories_"):
            for c, cats in zip(sel, trans.categories_):
                out += [c] * len(cats)
        else:
            out += list(sel)
    return out


def member_contribs(model, X, background):
    """(n, p) per-raw-column contributions and (n,) base, in member log-odds (exact)."""
    cols = list(X.columns)
    if hasattr(model, "get_booster"):
        c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return c[:, :-1], c[:, -1]
    pre, lr = model.named_steps["pre"], model.named_steps["lr"]
    coef, icpt = lr.coef_[0], lr.intercept_[0]
    Z = np.asarray(pre.transform(X), dtype=float)
    Zb = np.asarray(pre.transform(background), dtype=float).mean(axis=0)
    raw = _raw_columns(pre, cols)
    contrib = (Z - Zb) * coef
    phi = np.zeros((len(X), len(cols)))
    for k, c in enumerate(raw):
        phi[:, cols.index(c)] += contrib[:, k]
    return phi, np.full(len(X), icpt + coef @ Zb)


def fit_stack(target, members, X, meta="avg"):
    if members is None:
        return C.MultiTask(target=target, mode="indep", base="lrlog").fit(X, C.Y_DICT[target])
    est = C.MultiTask(target=target, mode="stack", members=members, meta=meta)
    return est.fit(X, C.Y_DICT[target])


def explain(est, X, background):
    if est.mode == "indep":
        return member_contribs(est.model_, X, background)
    w = est.meta_.coef_[0]
    phi = np.zeros(X.shape)
    base = np.full(len(X), est.meta_.intercept_[0])
    for wk, m in zip(w, est.member_models_):
        p, b0 = member_contribs(m, X, background)
        phi += wk * p
        base += wk * b0
    return phi, base


if __name__ == "__main__":
    X = C.X_ALL
    for t, mem in RECIPES.items():
        est = fit_stack(t, mem, X)
        phi, base = explain(est, X, X)
        logit_true = C.logit(est.predict_proba(X)[:, 1])
        models = [est.model_] if est.mode == "indep" else est.member_models_
        member_p = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
        unclipped = ((member_p > 1e-4) & (member_p < 1 - 1e-4)).all(1)
        if est.mode == "indep":
            logit_true = models[0].named_steps["lr"].decision_function(
                models[0].named_steps["pre"].transform(X))
            unclipped[:] = True
        err = np.abs(phi.sum(1) + base - logit_true)[unclipped]
        top = np.argsort(-np.abs(phi).mean(0))[:8]
        w = est.meta_.coef_[0] if est.mode == "stack" else np.array([1.0])
        b = est.meta_.intercept_[0] if est.mode == "stack" else 0.0
        print(f"{t}: meta w={np.round(w, 3)} b={b:.3f} "
              f"| max |sum(phi)+base - logit| = {err.max():.2e} on {unclipped.sum()}/{len(X)} rows")
        if est.mode == "stack":
            print("    member C_:", [getattr(m.named_steps["lr"], "C_", None) if hasattr(m, "named_steps")
                                     else "xgb" for m in est.member_models_],
                  f"threshold_={est.threshold_:.2f}")
        print("    top mean|phi|:", ", ".join(f"{X.columns[j]}={np.abs(phi[:, j]).mean():.3f}" for j in top))
