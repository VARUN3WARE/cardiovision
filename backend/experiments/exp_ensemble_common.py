"""Shared building blocks for the "ensemble" strategy experiments.

Everything learned from data lives inside the estimators defined here, so they are
refit on every training fold by the harness.

Members (all take the 55 raw label-encoded columns from harness.load_xy()):
  * lr     : one-hot(BBB, VHD) + log1p(skewed labs) + StandardScaler + L2 LogisticRegressionCV
             (C picked by inner 5-fold CV on log-loss). Attribution: coef * z (z standardised,
             training mean 0) summed back per raw column -> exact in log-odds.
  * xgb    : the baseline XGBoost (same params; scale_pos_weight optional).
  * cat    : CatBoost (ShapValues are exact in log-odds).
  * rf     : sklearn RandomForest (probability space; NOT additive in log-odds).
  * xgbrf  : XGBoost random-forest mode (pred_contribs exact in log-odds).

Combiners:
  * LogOddsVoting : sigmoid(sum_k w_k * logit(p_k)), fixed weights (default equal).
                    Attributions = sum_k w_k * member_attr_k (additive).
  * StackedLogit  : inner-CV OOF member logits -> non-negative-weight logistic meta-learner
                    (sigmoid(b + sum_k w_k logit_k)); members refit on the whole training fold.
  * PlattLogit    : inner-CV Platt scaling on the logit: sigmoid(a*logit(p)+b) (keeps additivity).
  * IsotonicCal   : inner-CV isotonic calibration (monotone, breaks additivity; comparison only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

EPS = 1e-6
NOMINAL = ["BBB", "VHD"]                       # label codes are alphabetical, not ordinal
SKEWED = ["FBS", "CR", "TG", "BUN", "ESR", "WBC", "PLT"]
CONSTANT = ["Exertional CP"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


# ----------------------------------------------------------------------------- members
def make_lr_pipeline(columns, C_grid=None, penalty="l2", log_skewed=True, scoring="neg_log_loss"):
    cols = list(columns)
    nominal = [c for c in NOMINAL if c in cols]
    skewed = [c for c in SKEWED if c in cols] if log_skewed else []
    rest = [c for c in cols if c not in nominal and c not in skewed and c not in CONSTANT]
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None), nominal),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), skewed),
        ("num", "passthrough", rest),
    ], verbose_feature_names_out=False)
    Cs = C_grid if C_grid is not None else np.logspace(-3, 1, 13)
    if penalty == "l2":
        clf = LogisticRegressionCV(Cs=Cs, cv=5, scoring=scoring, max_iter=5000, n_jobs=1)
    elif penalty == "l1":
        # random_state fixed: liblinear shuffles coordinates, otherwise fits are not reproducible
        clf = LogisticRegressionCV(Cs=Cs, cv=5, scoring=scoring, penalty="l1",
                                   solver="liblinear", max_iter=5000, n_jobs=1, random_state=0)
    elif penalty == "en":
        clf = LogisticRegressionCV(Cs=Cs, cv=5, scoring=scoring, penalty="elasticnet",
                                   solver="saga", l1_ratios=[0.5], max_iter=20000, n_jobs=1,
                                   random_state=0)
    else:
        raise ValueError(penalty)
    return Pipeline([("pre", pre), ("scale", StandardScaler()), ("clf", clf)])


def spw_for(y):
    y = np.asarray(y)
    pos = y.sum()
    return (len(y) - pos) / pos


def make_xgb(spw=None, **kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=2)
    if spw is not None:
        params["scale_pos_weight"] = spw
    params.update(kw)
    return XGBClassifier(**params)


def make_cat(**kw):
    from catboost import CatBoostClassifier
    params = dict(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                  random_seed=0, thread_count=2, verbose=0, allow_writing_files=False)
    params.update(kw)
    return CatBoostClassifier(**params)


def make_rf(**kw):
    params = dict(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                  n_jobs=2, random_state=0)
    params.update(kw)
    return RandomForestClassifier(**params)


def make_xgbrf(**kw):
    from xgboost import XGBRFClassifier
    params = dict(n_estimators=500, max_depth=6, subsample=0.8, colsample_bynode=0.5,
                  min_child_weight=2, reg_lambda=1.0, n_jobs=2, random_state=0)
    params.update(kw)
    return XGBRFClassifier(**params)


def member_factory(name, columns, y=None):
    """Zero-arg factory for a named member config."""
    spw = spw_for(y) if y is not None else None
    table = {
        "lr": lambda: make_lr_pipeline(columns),
        "lr_nolog": lambda: make_lr_pipeline(columns, log_skewed=False),
        "lr_l1": lambda: make_lr_pipeline(columns, penalty="l1"),
        "lr_auc": lambda: make_lr_pipeline(columns, scoring="roc_auc"),
        "xgb": lambda: make_xgb(spw=spw),               # == baseline_xgb
        "xgb_nospw": lambda: make_xgb(spw=None),
        "cat": lambda: make_cat(),
        "cat_d6": lambda: make_cat(depth=6, iterations=800, learning_rate=0.02),
        "rf": lambda: make_rf(),
        "xgbrf": lambda: make_xgbrf(),
        # RF-like XGBoost forests (sqrt-ish column sampling, deeper trees, bootstrap-sized rows)
        "xgbrf_rf": lambda: make_xgbrf(max_depth=10, colsample_bynode=0.15, subsample=0.632,
                                       min_child_weight=0.5),
        "xgbrf_d8": lambda: make_xgbrf(max_depth=8, colsample_bynode=0.25, subsample=0.7,
                                       min_child_weight=1),
        "xgbrf_d4": lambda: make_xgbrf(max_depth=4, colsample_bynode=0.25, subsample=0.7,
                                       min_child_weight=1),
    }
    return table[name]


# ----------------------------------------------------------------------------- combiners
class LogOddsVoting(BaseEstimator, ClassifierMixin):
    """Average of member log-odds with fixed weights (normalised to sum 1)."""

    def __init__(self, members, weights=None):
        self.members = members          # list of (name, estimator)
        self.weights = weights

    def fit(self, X, y):
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        w = np.ones(len(self.members)) if self.weights is None else np.asarray(self.weights, float)
        self.w_ = w / w.sum()
        self.classes_ = np.array([0, 1])
        return self

    def member_logits(self, X):
        return np.column_stack([logit(m.predict_proba(X)[:, 1]) for _, m in self.fitted_])

    def decision_function(self, X):
        return self.member_logits(X) @ self.w_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _fit_nonneg_logit_meta(Z, y, l2=1e-2):
    """min logloss(y, sigmoid(b + Z w)) + l2*|w|^2, w >= 0. Returns (w, b)."""
    K = Z.shape[1]

    def f(theta):
        w, b = theta[:K], theta[K]
        s = b + Z @ w
        # stable log-loss
        ll = np.mean(np.logaddexp(0, s) - y * s)
        p = sigmoid(s)
        g = np.empty(K + 1)
        g[:K] = Z.T @ (p - y) / len(y) + 2 * l2 * w
        g[K] = np.mean(p - y)
        return ll + l2 * np.sum(w ** 2), g

    x0 = np.r_[np.full(K, 1.0 / K), 0.0]
    bounds = [(0, None)] * K + [(None, None)]
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x[:K], float(res.x[K])


class StackedLogit(BaseEstimator, ClassifierMixin):
    """Stacking with a (non-negative) logistic meta-learner on inner-CV OOF member logits."""

    def __init__(self, members, inner_splits=5, l2=1e-2, nonneg=True, random_state=0):
        self.members = members
        self.inner_splits = inner_splits
        self.l2 = l2
        self.nonneg = nonneg
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        K = len(self.members)
        Z = np.zeros((len(y), K))
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        for tr, te in cv.split(X, y):
            for k, (_, m) in enumerate(self.members):
                mk = clone(m).fit(_rows(X, tr), y[tr])
                Z[te, k] = logit(mk.predict_proba(_rows(X, te))[:, 1])
        if self.nonneg:
            self.w_, self.b_ = _fit_nonneg_logit_meta(Z, y, self.l2)
        else:
            lr = LogisticRegression(C=1.0 / (2 * self.l2 * len(y)), max_iter=5000).fit(Z, y)
            self.w_, self.b_ = lr.coef_[0], float(lr.intercept_[0])
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def member_logits(self, X):
        return np.column_stack([logit(m.predict_proba(X)[:, 1]) for _, m in self.fitted_])

    def decision_function(self, X):
        return self.b_ + self.member_logits(X) @ self.w_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PlattLogit(BaseEstimator, ClassifierMixin):
    """Platt scaling on the base model's logit, fitted on inner-CV OOF predictions:
    p = sigmoid(a * logit(p_base) + b). Linear in log-odds -> attributions scale by a."""

    def __init__(self, estimator, inner_splits=5, random_state=0):
        self.estimator = estimator
        self.inner_splits = inner_splits
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        oof = np.zeros(len(y))
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        for tr, te in cv.split(X, y):
            m = clone(self.estimator).fit(_rows(X, tr), y[tr])
            oof[te] = logit(m.predict_proba(_rows(X, te))[:, 1])
        cal = LogisticRegression(C=1e4, max_iter=5000).fit(oof.reshape(-1, 1), y)
        self.a_, self.b_ = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = sigmoid(self.a_ * logit(self.model_.predict_proba(X)[:, 1]) + self.b_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class IsotonicCal(BaseEstimator, ClassifierMixin):
    """Inner-CV isotonic calibration (comparison only; breaks additivity of attributions)."""

    def __init__(self, estimator, inner_splits=5, random_state=0):
        self.estimator = estimator
        self.inner_splits = inner_splits
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        oof = np.zeros(len(y))
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        for tr, te in cv.split(X, y):
            m = clone(self.estimator).fit(_rows(X, tr), y[tr])
            oof[te] = m.predict_proba(_rows(X, te))[:, 1]
        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof, y)
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(self.iso_.predict(self.model_.predict_proba(X)[:, 1]), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ----------------------------------------------------------------------------- calibration stats
def calibration_stats(y, P):
    """Per-repeat calibration slope/intercept (logistic recalibration of logit p) and 10-bin ECE."""
    rows = []
    for r in range(P.shape[0]):
        z = logit(P[r]).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
        p = np.clip(P[r], EPS, 1 - EPS)
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
                  for b in range(10) if (bins == b).any())
        rows.append({"cal_slope": float(cal.coef_[0, 0]), "cal_intercept": float(cal.intercept_[0]),
                     "ece": float(ece)})
    df = pd.DataFrame(rows)
    return {k: round(float(df[k].mean()), 4) for k in df.columns}
