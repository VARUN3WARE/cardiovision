"""Pre-declared follow-up for RCA: does adding the CAD model's log-odds (the verified multitask idea)
to the final RCA recipe help at high-sensitivity operating points without costing AUC?

Candidate (fixed before running): logit = a * 0.5 * (margin_rca_pick + margin_cad_pick) + b,
with (a, b) a Platt fit on inner 5-fold OOF margins of the TRAINING fold; both members refit on the
whole training fold. CAD labels are looked up only for training rows (via X.index), never at
prediction time, so this is leakage-free (same structure the multitask verifier probed).

Adopt only if, vs final_pick: fresh-seed AUC delta >= -0.002 AND nested tuned F1 improves >= 0.01
on both seed 0 and seed 400.

usage: HARNESS_SEED_OFFSET=k python exp_final_rcacad.py eval | thresh
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import harness  # noqa: E402
from exp_final_recipe import make_model, margin, nested_threshold_metrics  # noqa: E402

warnings.filterwarnings("ignore")
X, Y, _ = harness.load_xy()
YCAD = Y["cad"]


class RcaWithCad(BaseEstimator, ClassifierMixin):
    def __init__(self, inner=5, random_state=0):
        self.inner, self.random_state = inner, random_state

    def _fit_members(self, Xs, y_rca, y_cad):
        return make_model("rca", y_rca).fit(Xs, y_rca), make_model("cad", y_cad).fit(Xs, y_cad)

    def fit(self, X, y):
        y = np.asarray(y)
        yc = YCAD[np.asarray(X.index)]  # training rows only
        z = np.zeros(len(y))
        for tr, te in StratifiedKFold(self.inner, shuffle=True, random_state=self.random_state).split(X, y):
            r, c = self._fit_members(X.iloc[tr], y[tr], yc[tr])
            z[te] = 0.5 * (margin(r, X.iloc[te]) + margin(c, X.iloc[te]))
        cal = LogisticRegression(C=1e4, max_iter=5000).fit(z.reshape(-1, 1), y)
        self.a_, self.b_ = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        self.rca_, self.cad_ = self._fit_members(X, y, yc)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.a_ * 0.5 * (margin(self.rca_, X) + margin(self.cad_, X)) + self.b_

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "eval":
        harness.evaluate("final_rcacad", "rca", lambda: RcaWithCad(), X, Y)
    elif mode == "thresh":
        res = nested_threshold_metrics(lambda: RcaWithCad(), X, Y["rca"])
        print("rca rcacad", json.dumps(res), flush=True)
        sfx = f"__seed{harness.SEED_OFFSET}" if harness.SEED_OFFSET else ""
        with open(harness.RESULTS_DIR / f"final_rcacad_thresholds{sfx}.json", "w") as f:
            json.dump(res, f, indent=2)
