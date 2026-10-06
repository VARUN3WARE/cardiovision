"""Exact per-feature log-odds attributions for the recommended features_* models, with an
additivity check (base + sum(contributions) == logit(predict_proba)) and a roll-up from
engineered features back to the 55 raw input columns.

usage: python exp_features_attrib.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from exp_features_lib import ECG_ABN, RISK_FACTORS, LogitBlend
from exp_features_run import CONFIGS
from harness import TARGETS, load_xy

warnings.filterwarnings("ignore")

RECOMMENDED = {
    "cad": "features_lr_engcore",
    "lad": "features_blend_lrcal_xgb",
    "lcx": "features_blend_lr_xgb",
    "rca": "features_lr_eng_l1",
}

CP = ["Typical Chest Pain", "Atypical", "Nonanginal"]
SOURCES = {  # engineered feature -> raw columns it is computed from (for the raw roll-up)
    "VHD": ["VHD"], "BBB_LBBB": ["BBB"], "BBB_RBBB": ["BBB"],
    "TG/HDL": ["TG", "HDL"], "LDL/HDL": ["LDL", "HDL"], "AIP log(TG/HDL)": ["TG", "HDL"],
    "Risk factor count": RISK_FACTORS + ["Age", "Sex"], "ECG abnormality count": ECG_ABN + ["BBB"],
    "Chest pain class": CP, "DF pretest prob": ["Age", "Sex"] + CP, "RWMA present": ["Region RWMA"],
    "Age x Male": ["Age", "Sex"], "eGFR": ["CR", "Age", "Sex"], "NLR": ["Neut", "Lymph"],
    "EF below 50": ["EF-TTE"], "FBS 126+": ["FBS"], "Systolic BP 140+": ["BP"],
}


def contrib_lr(p: Pipeline, X):
    fe = p.named_steps["fe"]
    Z = fe.transform(X)
    sc, lr = p.named_steps["m"].named_steps["scale"], p.named_steps["m"].named_steps["lr"]
    C = (Z.to_numpy() - sc.mean_) / sc.scale_ * lr.coef_.ravel()
    return pd.DataFrame(C, columns=fe.feature_names_out_, index=X.index), float(lr.intercept_[0])


def contrib_xgb(p: Pipeline, X):
    Z = p.named_steps["fe"].transform(X)
    out = p.named_steps["m"].get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
    return pd.DataFrame(out[:, :-1], columns=Z.columns, index=X.index), float(out[0, -1])


def contrib_cal(cal: CalibratedClassifierCV, X):
    """ensemble=False -> one base pipeline + one sigmoid: p = expit(-(a*f + b))."""
    cc = cal.calibrated_classifiers_[0]
    C, b0 = contrib_lr(cc.estimator, X)
    a, b = cc.calibrators[0].a_, cc.calibrators[0].b_
    return -a * C, float(-(a * b0 + b))


def contributions(model, X):
    if isinstance(model, LogitBlend):
        w = np.ones(len(model.models_)) / len(model.models_) if model.weights is None else model.weights
        parts = [contributions(m, X) for m in model.models_]
        cols = list(dict.fromkeys(c for cc, _ in parts for c in cc.columns))
        C = sum(wi * cc.reindex(columns=cols, fill_value=0.0) for wi, (cc, _) in zip(w, parts))
        return C, float(sum(wi * b for wi, (_, b) in zip(w, parts)))
    if isinstance(model, CalibratedClassifierCV):
        return contrib_cal(model, X)
    if "xgb" in type(model.named_steps["m"]).__name__.lower():
        return contrib_xgb(model, X)
    return contrib_lr(model, X)


def to_raw(C: pd.DataFrame, raw_cols) -> pd.DataFrame:
    """Sum-preserving roll-up: each engineered feature's contribution is split equally over its
    source raw columns; untouched raw columns keep their own contribution."""
    R = pd.DataFrame(0.0, index=C.index, columns=list(raw_cols))
    for c in C.columns:
        src = SOURCES.get(c, [c])
        for s in src:
            R[s] += C[c] / len(src)
    return R


if __name__ == "__main__":
    X, Y, _ = load_xy()
    for t in TARGETS:
        name = RECOMMENDED[t]
        model = CONFIGS[name](t)().fit(X, Y[t])
        C, base = contributions(model, X)
        p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        err = np.abs(base + C.sum(axis=1).to_numpy() - np.log(p / (1 - p))).max()
        R = to_raw(C, X.columns)
        err_raw = np.abs(R.sum(axis=1).to_numpy() - C.sum(axis=1).to_numpy()).max()
        top = C.abs().mean().sort_values(ascending=False).head(8)
        print(f"\n[{t}] {name}: base={base:+.3f}  max|base+sum-logit|={err:.2e}  raw roll-up err={err_raw:.2e}")
        print("  top mean|contribution| (log-odds):", {k: round(v, 3) for k, v in top.items()})
        print("  top raw roll-up:", {k: round(v, 3) for k, v in R.abs().mean().sort_values(ascending=False).head(8).items()})
