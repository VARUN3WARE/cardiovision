"""Adversarial re-implementation of the multitask winners (no shared cache, no MultiTask class).

Each config is a fresh estimator; auxiliary-task labels are looked up for the TRAINING rows
only (via X.index), members get inner-CV OOF logits on the training rows, the meta-learner is
"mean of member logits + unpenalised 1-D logistic (Platt)" fitted with sklearn on those OOF
logits. Test rows only ever pass through predict_proba.

usage: [HARNESS_SEED_OFFSET=k] python exp_verify_multitask_run.py <target> [...]
  ONLY=<comma names> restricts configs. Experiment names: verify_multitask_<cfg>.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import exp_multitask_common as C  # only make_base (xgb params / lrlog pipeline) and logit
from harness import evaluate, load_xy, SEED_OFFSET

X, Y, _ = load_xy()
YDF = pd.DataFrame(Y, index=X.index)


class CleanStack(BaseEstimator, ClassifierMixin):
    def __init__(self, target="rca", members=(("rca", "lrlog"),), inner=5, seed=0,
                 permute_target=False):
        self.target = target
        self.members = members
        self.inner = inner
        self.seed = seed
        self.permute_target = permute_target

    def fit(self, Xtr, y):
        y = np.asarray(y)
        if not self.permute_target:
            assert np.array_equal(YDF.loc[Xtr.index, self.target].to_numpy(), y)
        n, K = len(y), len(self.members)
        Z = np.zeros((n, K))
        self.models_ = []
        for k, (task, base) in enumerate(self.members):
            yt = y if task == self.target else YDF.loc[Xtr.index, task].to_numpy()
            for tr, te in StratifiedKFold(self.inner, shuffle=True,
                                          random_state=self.seed).split(Xtr, yt):
                m = C.make_base(base, yt[tr]).fit(Xtr.iloc[tr], yt[tr])
                Z[te, k] = C.logit(m.predict_proba(Xtr.iloc[te])[:, 1])
            self.models_.append(C.make_base(base, yt).fit(Xtr, yt))
        if K == 1 and self.members[0][1] == "lrlog":
            self.meta_ = None  # single lrlog: already Platt-rescaled internally
        else:
            self.meta_ = LogisticRegression(penalty=None, max_iter=1000).fit(
                Z.mean(axis=1, keepdims=True), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, Xte):
        Z = np.column_stack([C.logit(m.predict_proba(Xte)[:, 1]) for m in self.models_])
        if self.meta_ is None:
            return self.models_[0].predict_proba(Xte)
        return self.meta_.predict_proba(Z.mean(axis=1, keepdims=True))


CFG = {
    "cad": {"cad_L": [("cad", "lrlog")]},
    "lad": {"lad_XLcX": [("lad", "xgb"), ("lad", "lrlog"), ("cad", "lrlog"), ("cad", "xgb")],
            "lad_XL": [("lad", "xgb"), ("lad", "lrlog")]},
    "lcx": {"lcx_XLc": [("lcx", "xgb"), ("lcx", "lrlog"), ("cad", "lrlog")],
            "lcx_XL": [("lcx", "xgb"), ("lcx", "lrlog")]},
    "rca": {"rca_Lc": [("rca", "lrlog"), ("cad", "lrlog")],
            "rca_L": [("rca", "lrlog")]},
}


def baseline(target):
    pos = Y[target].sum()
    spw = (len(Y[target]) - pos) / pos
    from xgboost import XGBClassifier
    return lambda: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                 eval_metric="logloss", random_state=42, n_jobs=2)


if __name__ == "__main__":
    only = os.environ.get("ONLY")
    t0 = time.time()
    for t in sys.argv[1:]:
        cfgs = dict(CFG[t])
        if SEED_OFFSET not in (0, 100):  # fresh splits: baseline must be re-run too
            cfgs["baseline"] = None
        for name, mem in cfgs.items():
            if only and name not in only.split(","):
                continue
            make = baseline(t) if mem is None else (
                lambda t=t, mem=tuple(mem): CleanStack(target=t, members=mem))
            evaluate(f"verify_multitask_{name}", t, make, X, Y)
        print(f"--- {t} done {time.time() - t0:.0f}s", flush=True)
