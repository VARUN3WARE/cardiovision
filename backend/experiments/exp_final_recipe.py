"""FINAL production recipe per target + the checks that back it.

Every target is the same shape: an equal-weight blend, in log-odds (margin) space, of
  (1) a logistic regression on stateless clinical features (exp_features_lib.ClinicalFeatures), and
  (2) a shallow gradient-boosted tree model.
  cad: L2-LR on 8 core engineered features (+log labs) + LightGBM extra-trees stumps (additive GAM)
  lad: Platt-calibrated L2-LR (C by inner AUC) on core features + baseline-param XGBoost on engineered
  lcx: same as lad
  rca: L1-LR on all 14 engineered features (+log labs) + LightGBM extra-trees depth-2, heavily regularised

usage:
  HARNESS_SEED_OFFSET=k python exp_final_recipe.py eval          # harness.evaluate as final_pick
  HARNESS_SEED_OFFSET=k python exp_final_recipe.py thresh [base] # nested tuned thresholds (F1-opt + rule-out)
  python exp_final_recipe.py attrib                              # additivity of attributions
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import DMatrix, XGBClassifier  # noqa: E402

from exp_features_lib import ClinicalFeatures  # noqa: E402
import harness  # noqa: E402

warnings.filterwarnings("ignore")

CORE_ENG = ["TG/HDL", "AIP log(TG/HDL)", "Risk factor count", "ECG abnormality count",
            "Chest pain class", "DF pretest prob", "RWMA present", "Age x Male"]


# ----------------------------------------------------------------------------- members
def lr_l2(scoring="neg_log_loss"):
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=np.logspace(-3, 1, 13), scoring=scoring, max_iter=5000,
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=0)))])


def lr_l1():
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=np.logspace(-2.5, 1, 12), penalty="l1", solver="liblinear",
                                                 scoring="neg_log_loss", max_iter=5000, random_state=0,
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=0)))])


def xgb_base(spw):  # identical to baseline_xgb / train.py
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=2)


def lgb_stumps():
    return LGBMClassifier(n_estimators=1000, max_depth=1, num_leaves=2, learning_rate=0.05, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=10, reg_lambda=5.0,
                          reg_alpha=0.5, extra_trees=True, random_state=0, n_jobs=2, verbose=-1)


def lgb_d2_et():
    return LGBMClassifier(n_estimators=1200, max_depth=2, num_leaves=4, learning_rate=0.02, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=20, reg_lambda=20.0,
                          reg_alpha=0.5, extra_trees=True, random_state=0, n_jobs=2, verbose=-1)


def fe(subset=None, log=True, eng=True):
    return ("fe", ClinicalFeatures(fix_encodings=True, engineered=eng, log_labs=log, subset=subset))


# ----------------------------------------------------------------------------- blend
def margin(m, X):
    """Exact log-odds of any member (no probability clipping)."""
    if isinstance(m, (MarginBlend, PlattMargin)):
        return m.decision_function(X)
    if isinstance(m, CalibratedClassifierCV):  # ensemble=False: one base model + one sigmoid
        cc = m.calibrated_classifiers_[0]
        cal = cc.calibrators[0]
        return -(cal.a_ * margin(cc.estimator, X) + cal.b_)
    if isinstance(m, Pipeline):
        return margin(m.steps[-1][1], m[:-1].transform(X))
    if isinstance(m, LGBMClassifier):
        return m.predict(X, raw_score=True)
    if isinstance(m, XGBClassifier):
        return m.predict(X, output_margin=True).astype(float)
    return m.decision_function(X)


class MarginBlend(BaseEstimator, ClassifierMixin):
    """Equal-weight average of member log-odds. Nothing is learned beyond the members."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.models_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.mean([margin(m, X) for m in self.models_], axis=0)

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= getattr(self, "threshold_", 0.5)).astype(int)


class PlattMargin(BaseEstimator, ClassifierMixin):
    """Optional recalibration: logit' = a * margin + b, fit on inner-CV OOF margins of the training
    fold, then the base model is refit on the whole fold. Affine in log-odds, so attributions stay
    exact (multiply by a; base' = a * base + b)."""

    def __init__(self, est, inner=5, random_state=0):
        self.est, self.inner, self.random_state = est, inner, random_state

    def fit(self, X, y):
        y = np.asarray(y)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(self.inner, shuffle=True, random_state=self.random_state).split(X, y):
            oof[te] = margin(clone(self.est).fit(X.iloc[tr], y[tr]), X.iloc[te])
        cal = LogisticRegression(C=1e4, max_iter=5000).fit(oof.reshape(-1, 1), y)
        self.a_, self.b_ = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        self.model_ = clone(self.est).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.a_ * margin(self.model_, X) + self.b_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])


def make_model(target, y):
    y = np.asarray(y)
    spw = (len(y) - y.sum()) / y.sum()
    if target == "cad":
        return MarginBlend([Pipeline([fe(CORE_ENG), ("m", lr_l2())]), lgb_stumps()])
    if target in ("lad", "lcx"):
        lr_cal = CalibratedClassifierCV(Pipeline([fe(CORE_ENG), ("m", lr_l2(scoring="roc_auc"))]),
                                        method="sigmoid", ensemble=False,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=0))
        return MarginBlend([lr_cal, Pipeline([fe(None, log=False), ("m", xgb_base(spw))])])
    if target == "rca":
        return MarginBlend([Pipeline([fe(None), ("m", lr_l1())]), lgb_d2_et()])
    raise ValueError(target)


def make_production_model(target, y):
    """What train.py should fit: make_model, plus a PlattMargin recalibration for LAD/LCX (their
    XGBoost member is over-confident: calibration slope 0.65-0.8 -> 1.0). Within one fitted model the
    ranking is identical (a > 0 affine map), so AUC of the deployed model is unchanged."""
    m = make_model(target, y)
    return PlattMargin(m) if target in ("lad", "lcx") else m


# ----------------------------------------------------------------------------- thresholds
GRID = np.linspace(0.05, 0.95, 91)  # same grid as harness.ThresholdTuned


def pick_thresholds(oof, y, rule_out_sens=0.95):
    f1s = [f1_score(y, (oof >= g).astype(int), zero_division=0) for g in GRID]
    t_f1 = float(GRID[int(np.argmax(f1s))])
    ok = [g for g in GRID if recall_score(y, (oof >= g).astype(int), zero_division=0) >= rule_out_sens]
    t_ro = float(max(ok)) if ok else float(GRID[0])
    bal = [recall_score(y, (oof >= g).astype(int), zero_division=0)
           + recall_score(1 - y, (oof < g).astype(int), zero_division=0) for g in GRID]
    t_bal = float(GRID[int(np.argmax(bal))])
    return t_f1, t_ro, t_bal


def fit_with_thresholds(make, X, y, inner=4, random_state=0):
    """Production training: inner-CV OOF -> thresholds, then refit on all rows."""
    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(inner, shuffle=True, random_state=random_state).split(X, y):
        oof[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    t_f1, t_ro, t_bal = pick_thresholds(oof, y)
    model = make().fit(X, y)
    model.threshold_, model.rule_out_threshold_, model.balanced_threshold_ = t_f1, t_ro, t_bal
    return model


def nested_threshold_metrics(make, X, y):
    """Same protocol as harness.tuned_threshold_metrics (inner 4-fold, random_state=r), plus a
    rule-out operating point (inner-OOF recall >= 0.95) and a balanced-accuracy (Youden) operating
    point, all evaluated on the outer test fold."""
    R, n = harness.N_REPEATS, len(y)
    pf1, pro, pbal = np.zeros((R, n)), np.zeros((R, n)), np.zeros((R, n))
    for r in range(R):
        cv = StratifiedKFold(harness.N_SPLITS, shuffle=True, random_state=r + harness.SEED_OFFSET)
        for tr, te in cv.split(X, y):
            m = fit_with_thresholds(make, X.iloc[tr], y[tr], random_state=r)
            p = m.predict_proba(X.iloc[te])[:, 1]
            pf1[r, te] = p >= m.threshold_
            pro[r, te] = p >= m.rule_out_threshold_
            pbal[r, te] = p >= m.balanced_threshold_
    out = {}
    for tag, P in (("f1opt", pf1), ("ruleout", pro), ("balanced", pbal)):
        rows = [{"f1": f1_score(y, P[r]), "accuracy": accuracy_score(y, P[r]),
                 "precision": precision_score(y, P[r], zero_division=0), "recall": recall_score(y, P[r]),
                 "specificity": recall_score(1 - y, 1 - P[r]),
                 "npv": precision_score(1 - y, 1 - P[r], zero_division=0)} for r in range(R)]
        df = pd.DataFrame(rows)
        out[tag] = {k: {"mean": round(float(df[k].mean()), 4), "std": round(float(df[k].std(ddof=0)), 4)}
                    for k in df.columns}
    return out


# ----------------------------------------------------------------------------- attributions
def _flatten(p):
    steps = []
    for _, s in p.steps:
        steps.extend(_flatten(s) if isinstance(s, Pipeline) else [s])
    return steps


def contributions(m, X):
    """Exact additive log-odds attributions: returns (C DataFrame [n, model features], base [n]).
    base + C.sum(1) == margin(m, X). Columns are the model's own features: raw columns
    (labs possibly log-transformed, BBB split into BBB_LBBB/BBB_RBBB, VHD as ordinal) plus the
    named clinical composites (DF pretest prob, Risk factor count, ...)."""
    X = pd.DataFrame(X)
    if isinstance(m, PlattMargin):
        C, b = contributions(m.model_, X)
        return m.a_ * C, m.a_ * b + m.b_
    if isinstance(m, MarginBlend):
        parts = [contributions(mm, X) for mm in m.models_]
        cols = list(dict.fromkeys(c for C, _ in parts for c in C.columns))
        C = sum(P.reindex(columns=cols, fill_value=0.0) for P, _ in parts) / len(parts)
        return C, sum(b for _, b in parts) / len(parts)
    if isinstance(m, CalibratedClassifierCV):
        cc = m.calibrated_classifiers_[0]
        cal = cc.calibrators[0]
        C, b = contributions(cc.estimator, X)
        return -cal.a_ * C, -(cal.a_ * b + cal.b_)
    steps = _flatten(m) if isinstance(m, Pipeline) else [m]
    Z = X
    for s in steps[:-1]:
        if isinstance(s, StandardScaler):
            break
        Z = s.transform(Z)
    est = steps[-1]
    if isinstance(est, (LogisticRegression, LogisticRegressionCV)):
        sc = steps[-2]
        w = est.coef_.ravel() / sc.scale_
        C = (Z.to_numpy(float) - sc.mean_) * w
        return pd.DataFrame(C, columns=Z.columns, index=X.index), np.full(len(X), float(est.intercept_[0]))
    if isinstance(est, LGBMClassifier):
        out = est.predict(Z, pred_contrib=True)
    elif isinstance(est, XGBClassifier):
        out = est.get_booster().predict(DMatrix(Z), pred_contribs=True).astype(float)
    else:
        raise TypeError(type(est))
    return pd.DataFrame(out[:, :-1], columns=list(Z.columns), index=X.index), out[:, -1]


def display_attributions(C):
    """Merge columns that are functions of ONE raw input back onto it (exact): BBB one-hots -> BBB.
    Composite clinical scores stay as their own named rows."""
    C = C.copy()
    if {"BBB_LBBB", "BBB_RBBB"} & set(C.columns):
        C["BBB"] = C.pop("BBB_LBBB") if "BBB_LBBB" in C else 0.0
        if "BBB_RBBB" in C:
            C["BBB"] = C["BBB"] + C.pop("BBB_RBBB")
    return C


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    mode = sys.argv[1]
    X, Y, _ = harness.load_xy()
    T = ["cad", "lad", "lcx", "rca"]
    if mode == "eval":
        for t in (sys.argv[2:] or T):
            harness.evaluate("final_pick", t, lambda t=t: make_model(t, Y[t]), X, Y)
    elif mode == "eval_platt":
        for t in (sys.argv[2:] or ["lad", "lcx"]):
            harness.evaluate("final_pickplatt", t, lambda t=t: PlattMargin(make_model(t, Y[t])), X, Y)
    elif mode == "thresh":
        which = sys.argv[2] if len(sys.argv) > 2 else "pick"
        res = {}
        for t in T:
            make = (lambda t=t: make_model(t, Y[t])) if which == "pick" else \
                (lambda t=t: xgb_base((len(Y[t]) - Y[t].sum()) / Y[t].sum()))
            res[t] = nested_threshold_metrics(make, X, Y[t])
            print(t, which, json.dumps(res[t]), flush=True)
        sfx = f"__seed{harness.SEED_OFFSET}" if harness.SEED_OFFSET else ""
        with open(harness.RESULTS_DIR / f"final_{which}_thresholds{sfx}.json", "w") as f:
            json.dump(res, f, indent=2)
    elif mode == "attrib":
        rng = np.random.RandomState(0)
        for t in T:
            y = Y[t]
            m = make_model(t, y).fit(X, y)
            C, b = contributions(m, X)
            err = np.abs(b + C.sum(1).to_numpy() - m.decision_function(X)).max()
            p_err = np.abs(expit(b + C.sum(1).to_numpy()) - m.predict_proba(X)[:, 1]).max()
            idx = rng.permutation(len(y))
            tr, te = idx[:240], idx[240:]
            mh = make_model(t, y[tr]).fit(X.iloc[tr], y[tr])
            Ch, bh = contributions(mh, X.iloc[te])
            err_h = np.abs(bh + Ch.sum(1).to_numpy() - mh.decision_function(X.iloc[te])).max()
            D = display_attributions(C)
            top = D.abs().mean().sort_values(ascending=False).head(10)
            print(f"{t}: max|base+sum-logit| full {err:.2e} heldout {err_h:.2e} prob {p_err:.2e}; "
                  f"{D.shape[1]} display features; top mean|phi|: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in top.items()), flush=True)
