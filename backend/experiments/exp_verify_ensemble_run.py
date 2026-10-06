"""Adversarial verification of the "ensemble" strategy claims.

Rebuilds every claimed configuration INDEPENDENTLY from the production recipe text (members
written out here; combiners re-implemented here, not imported from exp_ensemble_common), so a
bug in their LogOddsVoting / PlattLogit would show up as an OOF mismatch.

usage: [HARNESS_SEED_OFFSET=k] python exp_verify_ensemble_run.py <config> [targets...]
configs: base, cad_l1c, lad_plattxrf, lad_xrf, lcx_l1x, rca_l1f, univ_l1xf
Results are saved as experiment "verify_ensemble_<config>".
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier

from harness import evaluate, load_xy

warnings.filterwarnings("ignore")
X, Y, _ = load_xy()
assert X.shape[1] == 55 and not {"LAD", "LCX", "RCA", "Cath"} & set(X.columns)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


class Vote(BaseEstimator, ClassifierMixin):
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = np.mean([_logit(m.predict_proba(X)[:, 1]) for m in self.fitted_], axis=0)
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class Platt(BaseEstimator, ClassifierMixin):
    """Inner 5-fold OOF logit -> near-unregularised logistic fit; base refit on full train fold."""

    def __init__(self, est, inner=5, rs=0):
        self.est, self.inner, self.rs = est, inner, rs

    def fit(self, X, y):
        y = np.asarray(y)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(self.inner, shuffle=True, random_state=self.rs).split(X, y):
            oof[te] = _logit(clone(self.est).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1])
        cal = LogisticRegression(C=1e4, max_iter=5000).fit(oof.reshape(-1, 1), y)
        self.a_, self.b_ = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        self.m_ = clone(self.est).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(self.a_ * _logit(self.m_.predict_proba(X)[:, 1]) + self.b_)))
        return np.column_stack([1 - p, p])


# ---- members exactly as in the claimed production_recipe
def lr_l1(cols):
    nominal = ["BBB", "VHD"]; skewed = ["FBS", "CR", "TG", "BUN", "ESR", "WBC", "PLT"]
    rest = [c for c in cols if c not in nominal + skewed + ["Exertional CP"]]
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), skewed),
        ("num", "passthrough", rest)], verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("scale", StandardScaler()),
                     ("clf", LogisticRegressionCV(Cs=np.logspace(-3, 1, 13), cv=5, scoring="neg_log_loss",
                                                  penalty="l1", solver="liblinear", max_iter=5000,
                                                  random_state=0))])


def cat():
    return CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=5, random_seed=0,
                              thread_count=2, verbose=0, allow_writing_files=False)


def xgb(spw):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=2)


def xgbrf():
    return XGBRFClassifier(n_estimators=500, max_depth=6, subsample=0.8, colsample_bynode=0.5,
                           min_child_weight=2, reg_lambda=1.0, n_jobs=2, random_state=0)


def xgbrf_rf():
    return XGBRFClassifier(n_estimators=500, max_depth=10, subsample=0.632, colsample_bynode=0.15,
                           min_child_weight=0.5, reg_lambda=1.0, n_jobs=2, random_state=0)


def spw_of(y):
    return (len(y) - y.sum()) / y.sum()


CONFIGS = {
    "base": lambda t: (lambda: xgb(spw_of(Y[t]))),
    "cad_l1c": lambda t: (lambda: Vote([lr_l1(X.columns), cat()])),
    "lad_plattxrf": lambda t: (lambda: Platt(Vote([xgbrf_rf()]))),
    "lad_xrf": lambda t: (lambda: xgbrf_rf()),
    "lcx_l1x": lambda t: (lambda: Vote([lr_l1(X.columns), xgb(spw_of(Y[t]))])),
    "rca_l1f": lambda t: (lambda: Vote([lr_l1(X.columns), xgbrf()])),
    "univ_l1xf": lambda t: (lambda: Vote([lr_l1(X.columns), xgb(spw_of(Y[t])), xgbrf()])),
}
DEFAULT_T = {"cad_l1c": ["cad"], "lad_plattxrf": ["lad"], "lad_xrf": ["lad"], "lcx_l1x": ["lcx"],
             "rca_l1f": ["rca"]}

if __name__ == "__main__":
    cfg = sys.argv[1]
    targets = sys.argv[2:] or DEFAULT_T.get(cfg, ["cad", "lad", "lcx", "rca"])
    for t in targets:
        evaluate(f"verify_ensemble_{cfg}", t, CONFIGS[cfg](t), X, Y)
