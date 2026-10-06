"""Multi-task estimators (strategy key: multitask).

All estimators take the 55 raw encoded columns as X and the harness target as y.
They additionally look up the *training-fold* labels of the auxiliary targets
(CAD / other vessels) from Y_ALL via X.index -- this is ordinary multi-task
training. At predict time only X is used: auxiliary information enters as
*predicted* probabilities from models fit on the training fold (inner-CV OOF
predictions for the training rows, full-training-fold refits for test rows).

Modes
  indep      : base model on own target (control; spw computed on the training fold)
  hier       : P(v) = P(CAD) * P(v | CAD), conditional model on CAD-positive training rows
  chain_cad  : base model on X + logit(OOF P(CAD))
  chain_all  : base model on X + logits(OOF P(other 3 targets))
  stack      : logistic meta-learner on inner-OOF logits of `members`
               (member spec "task:base" or "cond_<vessel>:base" for P(vessel | CAD)).
               Linear in member log-odds -> member SHAP values combine exactly.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from harness import load_xy

X_ALL, Y_DICT, _ = load_xy()
Y_ALL = pd.DataFrame(Y_DICT, index=X_ALL.index)
TASKS = ["cad", "lad", "lcx", "rca"]
VESSELS = ["lad", "lcx", "rca"]
NOMINAL = ["BBB", "VHD"]  # label-encoded alphabetically, not ordinal


def _spw(y):
    pos = float(np.sum(y))
    return (len(y) - pos) / pos if pos > 0 else 1.0


def make_base(kind: str, y):
    spw = _spw(y)
    if kind == "xgb":  # identical to baseline_xgb params
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                             random_state=42, n_jobs=2)
    if kind == "xgbl":  # shallow, heavily regularised
        return XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.5, min_child_weight=2, reg_lambda=5.0,
                             scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=2)
    if kind == "lr":
        num = [c for c in X_ALL.columns if c not in NOMINAL]
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL)],
                                remainder="passthrough")
        # extra (chain) columns fall through `remainder` unscaled; logits are already O(1)
        return Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.05, class_weight="balanced",
                                                                   max_iter=5000))])
    if kind == "lrlog":
        # fixed log1p of skewed labs, drop constant column, one-hot BBB/VHD, scale,
        # L2 logistic regression with C chosen by inner 5-fold ROC-AUC (refit on all rows)
        from sklearn.preprocessing import FunctionTransformer
        num = [c for c in X_ALL.columns if c not in NOMINAL + ["Exertional CP"]]
        pre = ColumnTransformer([("num", Pipeline([("log", FunctionTransformer(_log_skewed)),
                                                   ("sc", StandardScaler())]), num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), NOMINAL)],
                                remainder="passthrough")
        return Pipeline([("pre", pre), ("lr", FastRidgeLogitCV())])
    raise ValueError(kind)


class FastRidgeLogitCV(BaseEstimator, ClassifierMixin):
    """L2 logistic regression (sklearn objective: C*sum(logloss) + 0.5*||w||^2, free intercept)
    solved by damped Newton along a warm-started C path; C chosen by inner stratified K-fold
    mean ROC-AUC, then refit on all rows. Equivalent to LogisticRegressionCV(scoring='roc_auc')
    but ~50x faster for p~60. Exposes coef_/intercept_ like sklearn's LogisticRegression."""

    def __init__(self, Cs=tuple(np.logspace(-4, 1, 11)), cv=5, random_state=0, calibrate=True):
        self.Cs = Cs
        self.cv = cv
        self.random_state = random_state
        self.calibrate = calibrate

    def _platt(self, d, y):
        A = np.column_stack([d, np.ones(len(d))])
        w = self._solve(A, y, 1e6, np.zeros(2))  # ~unpenalised 1-D logistic fit
        return float(w[0]), float(w[1])

    @staticmethod
    def _loss(A, y, w, C, R):
        z = A @ w
        return C * np.sum(np.logaddexp(0, z) - y * z) + 0.5 * w @ R @ w

    def _solve(self, A, y, C, w):
        R = np.eye(A.shape[1])
        R[-1, -1] = 0.0
        f = self._loss(A, y, w, C, R)
        for _ in range(100):
            pr = 1 / (1 + np.exp(-(A @ w)))
            g = C * A.T @ (pr - y) + R @ w
            H = C * (A.T * (pr * (1 - pr))) @ A + R + 1e-10 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                wn = w - t * step
                fn = self._loss(A, y, wn, C, R)
                if fn <= f + 1e-12 or t < 1e-6:
                    break
                t *= 0.5
            w, df, f = wn, f - fn, fn
            if np.max(np.abs(t * step)) < 1e-7 or df < 1e-10:
                break
        return w

    def _path(self, A, y, Cs):
        w = np.zeros(A.shape[1])
        out = []
        for C in Cs:
            w = self._solve(A, y, C, w)
            out.append(w.copy())
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.column_stack([X, np.ones(len(X))])
        Cs = sorted(self.Cs)
        scores = np.zeros(len(Cs))
        oof = np.zeros((len(Cs), len(y)))
        from scipy.stats import rankdata

        def roc_auc_score(yt, s):
            r = rankdata(s)
            n1 = yt.sum()
            return (r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yt) - n1))
        for tr, te in StratifiedKFold(self.cv, shuffle=True,
                                      random_state=self.random_state).split(X, y):
            for i, w in enumerate(self._path(A[tr], y[tr], Cs)):
                oof[i, te] = A[te] @ w
                scores[i] += roc_auc_score(y[te], oof[i, te])
        best = int(np.argmax(scores))
        self.C_ = Cs[best]
        w = self._path(A, y, Cs[:best + 1])[-1]
        # Platt rescaling of the log-odds, fitted on the inner OOF decision values of the
        # chosen C (training rows only). Small C shrinks log-odds towards 0 by different
        # amounts per fit; without this the scale is not comparable across fits/folds.
        # Result is still exactly linear: a*(x.w + c) + b.
        a, b = self._platt(oof[best], y) if self.calibrate else (1.0, 0.0)
        self.platt_ = (a, b)
        self.coef_ = (a * w[:-1]).reshape(1, -1)
        self.intercept_ = np.array([a * w[-1] + b])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


SKEWED = ["FBS", "TG", "ESR", "CR", "BUN", "WBC", "PLT"]


def _log_skewed(Z):
    Z = Z.copy()
    for c in SKEWED:
        Z[c] = np.log1p(Z[c].clip(lower=0))
    return Z


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- caching
_CACHE: dict = {}


def _key(X) -> str:
    return hashlib.md5(np.asarray(X.index, dtype=np.int64).tobytes()).hexdigest()


def _labels(X, task):
    return Y_ALL.loc[X.index, task].to_numpy()


def fit_full(X, task, base):
    k = ("full", _key(X), task, base)
    if k not in _CACHE:
        _CACHE[k] = make_base(base, _labels(X, task)).fit(X, _labels(X, task))
    return _CACHE[k]


def fit_cond(X, vessel, base):
    """P(vessel | CAD=1) fitted on CAD-positive training rows only."""
    k = ("cond", _key(X), vessel, base)
    if k not in _CACHE:
        m = _labels(X, "cad") == 1
        yv = _labels(X, vessel)[m]
        _CACHE[k] = make_base(base, yv).fit(X[m], yv)
    return _CACHE[k]


def member_oof(X, spec, inner=5, seed=0):
    """Inner-CV OOF probabilities on the training rows for a member spec."""
    k = ("oof", _key(X), spec, inner, seed)
    if k in _CACHE:
        return _CACHE[k]
    task, base = spec.split(":")
    cond = task.startswith("cond_")
    t = task[5:] if cond else task
    yt = _labels(X, t)
    ycad = _labels(X, "cad")
    strat = yt + 2 * ycad if cond else yt
    if np.unique(strat, return_counts=True)[1].min() < inner:  # e.g. the single LAD+/CAD- patient
        strat = np.where(strat == 1, 0, strat) if cond else yt
    oof = np.zeros(len(yt))
    for tr, te in StratifiedKFold(inner, shuffle=True, random_state=seed).split(X, strat):
        if cond:
            tr = tr[ycad[tr] == 1]
        m = make_base(base, yt[tr]).fit(X.iloc[tr], yt[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    _CACHE[k] = oof
    return oof


def member_full(X, spec):
    task, base = spec.split(":")
    if task.startswith("cond_"):
        return fit_cond(X, task[5:], base)
    return fit_full(X, task, base)


class NNLogit:
    """Logistic regression on member logits with non-negative weights (+ small L2).
    Keeps the stack monotone in every member, so attributions keep their sign."""

    def __init__(self, l2=1e-2):
        self.l2 = l2

    def fit(self, Z, y):
        from scipy.optimize import minimize
        n, k = Z.shape

        def f(theta):
            w, b = theta[:k], theta[k]
            z = Z @ w + b
            p = 1 / (1 + np.exp(-z))
            loss = np.mean(np.logaddexp(0, z) - y * z) + self.l2 * np.sum(w ** 2)
            g = Z.T @ (p - y) / n + 2 * self.l2 * w
            return loss, np.append(g, np.mean(p - y))

        x0 = np.append(np.full(k, 1.0 / k), 0.0)
        r = minimize(f, x0, jac=True, method="L-BFGS-B",
                     bounds=[(0, None)] * k + [(None, None)])
        self.coef_ = r.x[:k].reshape(1, -1)
        self.intercept_ = np.array([r.x[k]])
        return self

    def predict_proba(self, Z):
        p = 1 / (1 + np.exp(-(Z @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


class AvgLogit:
    """Equal-weight average of member logits + 1-D Platt (slope, intercept) fitted on the
    inner-OOF average. Only 2 learned parameters -> very stable on ~240 rows.
    Exposes coef_ (= slope/k per member) and intercept_ like a linear meta-learner."""

    def fit(self, Z, y):
        k = Z.shape[1]
        a, b = FastRidgeLogitCV()._platt(Z.mean(axis=1), np.asarray(y, dtype=float))
        self.coef_ = np.full((1, k), a / k)
        self.intercept_ = np.array([b])
        return self

    def predict_proba(self, Z):
        p = 1 / (1 + np.exp(-(Z @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


def _augment(X, Z, names):
    Xa = X.copy()
    for j, n in enumerate(names):
        Xa[f"_logit_{n}"] = Z[:, j]
    return Xa


class MultiTask(BaseEstimator, ClassifierMixin):
    def __init__(self, target="lad", mode="indep", base="xgb", members=None, chain_with=None,
                 meta_C=1.0, meta="lr", inner=5, seed=0):
        self.target = target
        self.mode = mode
        self.base = base
        self.members = members
        self.chain_with = chain_with
        self.meta_C = meta_C
        self.meta = meta
        self.inner = inner
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        assert np.array_equal(_labels(X, self.target), y), "index/label misalignment"
        self.classes_ = np.array([0, 1])
        b = self.base
        if self.mode == "indep":
            self.model_ = fit_full(X, self.target, b)
        elif self.mode == "hier":
            self.cad_ = fit_full(X, "cad", b)
            self.cond_ = fit_cond(X, self.target, b)
        elif self.mode in ("chain_cad", "chain_all"):
            others = (["cad"] if self.mode == "chain_cad"
                      else [t for t in TASKS if t != self.target])
            self.chain_specs_ = [f"{o}:{self.chain_with or b}" for o in others]
            Z = np.column_stack([logit(member_oof(X, s, self.inner, self.seed))
                                 for s in self.chain_specs_])
            self.chain_models_ = [member_full(X, s) for s in self.chain_specs_]
            self.model_ = make_base(b, y).fit(_augment(X, Z, others), y)
            self.chain_names_ = others
        elif self.mode == "stack":
            self.member_specs_ = list(self.members)
            Z = np.column_stack([logit(member_oof(X, s, self.inner, self.seed))
                                 for s in self.member_specs_])
            self.member_models_ = [member_full(X, s) for s in self.member_specs_]
            if self.meta == "nn":
                self.meta_ = NNLogit().fit(Z, y)
            elif self.meta == "avg":
                self.meta_ = AvgLogit().fit(Z, y)
            else:
                self.meta_ = LogisticRegression(C=self.meta_C, max_iter=2000).fit(Z, y)
            # F1-optimal decision threshold from the stack's (inner-OOF-based) training-fold
            # predictions -- training rows only; used by predict(), predict_proba unchanged.
            from sklearn.metrics import f1_score
            p_in = self.meta_.predict_proba(Z)[:, 1]
            grid = np.linspace(0.05, 0.95, 91)
            self.threshold_ = float(grid[int(np.argmax(
                [f1_score(y, (p_in >= g).astype(int), zero_division=0) for g in grid]))])
        else:
            raise ValueError(self.mode)
        return self

    def predict_proba(self, X):
        if self.mode == "indep":
            return self.model_.predict_proba(X)
        if self.mode == "hier":
            p = self.cad_.predict_proba(X)[:, 1] * self.cond_.predict_proba(X)[:, 1]
        elif self.mode in ("chain_cad", "chain_all"):
            Z = np.column_stack([logit(m.predict_proba(X)[:, 1]) for m in self.chain_models_])
            p = self.model_.predict_proba(_augment(X, Z, self.chain_names_))[:, 1]
        else:
            Z = np.column_stack([logit(m.predict_proba(X)[:, 1]) for m in self.member_models_])
            p = self.meta_.predict_proba(Z)[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        thr = getattr(self, "threshold_", 0.5)
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)


def stack_variants(target, base):
    """Named stacking member sets for a target."""
    if target == "cad":
        return {"stack_all": [f"{t}:{base}" for t in TASKS]}
    others = [v for v in VESSELS if v != target]
    return {
        "stack_cad": [f"{target}:{base}", f"cad:{base}"],
        "stack_all": [f"{target}:{base}", f"cad:{base}"] + [f"{o}:{base}" for o in others],
        "stack_soft_hier": [f"cad:{base}", f"cond_{target}:{base}"],
        "stack_hier_plus": [f"{target}:{base}", f"cad:{base}", f"cond_{target}:{base}"],
    }
