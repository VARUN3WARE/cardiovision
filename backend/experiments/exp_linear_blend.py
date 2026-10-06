"""Log-odds blend of the calibrated L1/L2 logistic regression and the baseline XGBoost.

    python exp_linear_blend.py <l1|l2> [weight ...]

p = sigmoid(w * logit(p_lr) + (1 - w) * logit(p_xgb)), fixed w (no data-driven weight, so
nothing extra to nest). Attributions for the UI stay exact-additive in log-odds:
phi = w * phi_lr (coef * (z - mean z)) + (1 - w) * phi_xgb (pred_contribs).
"""
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier

from exp_linear_common import CalibratedLinearCV
from harness import evaluate_all, load_xy

_, Y, _ = load_xy()
PEN = "l2"


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LogitBlend(BaseEstimator, ClassifierMixin):
    def __init__(self, a, b, w=0.5):
        self.a, self.b, self.w = a, b, w

    def fit(self, X, y):
        self.a_ = clone(self.a).fit(X, y)
        self.b_ = clone(self.b).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = self.w * _logit(self.a_.predict_proba(X)[:, 1]) + \
            (1 - self.w) * _logit(self.b_.predict_proba(X)[:, 1])
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def xgb_baseline(target):
    pos = Y[target].sum()
    spw = (len(Y[target]) - pos) / pos
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, n_jobs=2)


def make_for(w):
    def make_model_for(target):
        def make():
            return LogitBlend(CalibratedLinearCV(PEN, prep_kwargs=dict(log_skewed=True)),
                              xgb_baseline(target), w=w)
        return make
    return make_model_for


if __name__ == "__main__":
    # python exp_linear_blend.py <l1|l2> w1 [w2 ...]
    PEN = sys.argv[1]
    for w in [float(a) for a in sys.argv[2:]] or [0.5]:
        evaluate_all(f"linear_blend{PEN}_w{int(round(w * 100))}", make_for(w))
