"""Single multi-output models (strategy c) vs independent controls.

  mo_xgbtree : XGBoost multi-label with vector-leaf trees (multi_strategy=multi_output_tree)
  long_xgb   : one XGBoost on "long" data (patient x task rows + one-hot task indicator),
               per-task class balancing via sample weights; native pred_contribs work
  mlp        : sklearn MLP with a shared hidden layer, multi-label output (5-seed average)
  rf_mo      : RandomForest with native multi-output splits;  rf_indep : control

usage: python exp_multitask_mo.py <kind> [targets...]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import exp_multitask_common as C
from harness import evaluate

TASKS = C.TASKS


def _pre():
    num = [c for c in C.X_ALL.columns if c not in C.NOMINAL]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), C.NOMINAL)])


def _long(X, tasks):
    parts = []
    for t in tasks:
        Xt = X.copy()
        for u in TASKS:
            Xt[f"_task_{u}"] = float(u == t)
        parts.append(Xt)
    return pd.concat(parts, axis=0, ignore_index=True)


class MultiOutput(BaseEstimator, ClassifierMixin):
    def __init__(self, target="lad", kind="mo_xgbtree", alpha=1.0, hidden=16):
        self.target = target
        self.kind = kind
        self.alpha = alpha
        self.hidden = hidden

    def fit(self, X, y):
        assert np.array_equal(C._labels(X, self.target), np.asarray(y))
        self.classes_ = np.array([0, 1])
        Y = C.Y_ALL.loc[X.index, TASKS].to_numpy()
        self.j_ = TASKS.index(self.target)
        k = self.kind
        if k == "mo_xgbtree":
            self.m_ = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, tree_method="hist",
                                    multi_strategy="multi_output_tree", eval_metric="logloss",
                                    random_state=42, n_jobs=2).fit(X, Y)
        elif k == "long_xgb":
            XL = _long(X, TASKS)
            yL = Y.T.reshape(-1)  # task-major, matches _long order
            w = np.ones(len(yL))
            n = len(X)
            for i in range(len(TASKS)):
                yi = Y[:, i]
                w[i * n:(i + 1) * n] = np.where(yi == 1, C._spw(yi), 1.0)
            self.m_ = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, eval_metric="logloss", random_state=42,
                                    n_jobs=2).fit(XL, yL, sample_weight=w)
        elif k == "mlp":
            self.m_ = [Pipeline([("pre", _pre()),
                                 ("mlp", MLPClassifier(hidden_layer_sizes=(self.hidden,),
                                                       alpha=self.alpha, solver="lbfgs",
                                                       max_iter=1000, random_state=s))]).fit(X, Y)
                       for s in range(5)]
        elif k == "rf_mo":
            self.m_ = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                             random_state=42, n_jobs=2).fit(X, Y)
        elif k == "rf_indep":
            self.m_ = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                             class_weight="balanced_subsample",
                                             random_state=42, n_jobs=2).fit(X, Y[:, self.j_])
        return self

    def predict_proba(self, X):
        k = self.kind
        if k == "mo_xgbtree":
            p = self.m_.predict_proba(X)[:, self.j_]
        elif k == "long_xgb":
            p = self.m_.predict_proba(_long(X, [self.target]))[:, 1]
        elif k == "mlp":
            p = np.mean([m.predict_proba(X)[:, self.j_] for m in self.m_], axis=0)
        elif k == "rf_mo":
            p = self.m_.predict_proba(X)[self.j_][:, 1]
        else:
            p = self.m_.predict_proba(X)[:, 1]
        return np.column_stack([1 - p, p])


if __name__ == "__main__":
    kind = sys.argv[1]
    targets = sys.argv[2:] or TASKS
    extra = {}
    name = f"multitask_{kind}"
    if kind.startswith("mlp"):  # mlp_a<alpha>
        extra = {"alpha": float(kind.split("_a")[1])} if "_a" in kind else {}
        kind = "mlp"
    for t in targets:
        evaluate(name, t, lambda t=t: MultiOutput(target=t, kind=kind, **extra), C.X_ALL, C.Y_DICT)
