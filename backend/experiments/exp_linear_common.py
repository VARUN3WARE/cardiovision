"""Shared building blocks for the regularised linear / kernel experiments (key="linear").

Everything here is either a FIXED, label-free transform (column re-coding, log1p of
skewed labs, dropping the constant column) or a learned step that lives inside the
sklearn Pipeline (StandardScaler, OneHotEncoder, GridSearchCV), so the harness refits
it on every training fold.

Input: the 55 label-encoded columns produced by harness.load_xy() / app.encoders.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CONSTANT = ["Exertional CP"]
# Alphabetical label codes from app.encoders -> clinically ordered codes.
# VHD: {Moderate:0, N:1, Severe:2, mild:3}  ->  N=0 < mild=1 < Moderate=2 < Severe=3
VHD_ORDINAL = {0.0: 2.0, 1.0: 0.0, 2.0: 3.0, 3.0: 1.0}
# Right-skewed labs that benefit from log1p before scaling (fixed transform, no fitting).
SKEWED = ["FBS", "TG", "ESR", "CR", "BUN", "WBC", "PLT"]
# Continuous columns eligible for spline expansion (enough distinct values).
CONTINUOUS = ["Age", "Weight", "Length", "BMI", "BP", "PR", "FBS", "CR", "TG", "LDL", "HDL",
              "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE"]


class Recode(BaseEstimator, TransformerMixin):
    """Fixed (label-free, data-free) re-coding of the raw encoded frame."""

    def __init__(self, vhd: str = "onehot", log_skewed: bool = False):
        self.vhd = vhd
        self.log_skewed = log_skewed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=[c for c in CONSTANT if c in X.columns]).copy()
        if self.vhd == "ordinal":
            X["VHD"] = X["VHD"].map(VHD_ORDINAL).fillna(0.0)
        if self.log_skewed:
            for c in SKEWED:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array([c for c in input_features if c not in CONSTANT])


def make_prep(vhd: str = "onehot", log_skewed: bool = False, splines: bool = False,
              n_knots: int = 4) -> Pipeline:
    """Recode -> ColumnTransformer(one-hot BBB[/VHD], scale numerics[, splines])."""
    onehot = ["BBB"] + (["VHD"] if vhd == "onehot" else [])

    def selector(kind):
        def sel(df):
            cols = [c for c in df.columns if c not in onehot]
            if kind == "spline":
                return [c for c in cols if c in CONTINUOUS]
            if kind == "num" and splines:
                return [c for c in cols if c not in CONTINUOUS]
            return cols
        return sel

    transformers = [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot)]
    if splines:
        transformers.append(("spl", Pipeline([
            ("scale", StandardScaler()),
            ("spline", SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                                         extrapolation="linear")),
            ("scale2", StandardScaler()),
        ]), selector("spline")))
    transformers.append(("num", StandardScaler(), selector("num")))
    return Pipeline([
        ("recode", Recode(vhd=vhd, log_skewed=log_skewed)),
        ("ct", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)),
    ])


C_GRID = list(np.logspace(-3, 1, 9))  # 0.001 .. 10


def inner_cv(seed: int = 0):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def lr_search(penalty: str = "l2", class_weight="grid", l1_ratios=(0.2, 0.5, 0.8),
              scoring: str = "roc_auc", prep_kwargs: dict | None = None,
              c_grid=None) -> GridSearchCV:
    """Pipeline(prep, LR) wrapped in an inner GridSearchCV (refit on the whole training fold)."""
    prep_kwargs = prep_kwargs or {}
    c_grid = c_grid or C_GRID
    if penalty == "l2":
        clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)
        grid = {"clf__C": c_grid}
    elif penalty == "l1":
        clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)
        grid = {"clf__C": [c for c in c_grid if c >= 0.01] + [30.0]}
    elif penalty == "elasticnet":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000, tol=1e-3)
        grid = {"clf__C": [c for c in c_grid if c >= 0.003], "clf__l1_ratio": list(l1_ratios)}
    else:
        raise ValueError(penalty)
    if class_weight == "grid":
        grid["clf__class_weight"] = [None, "balanced"]
    else:
        clf.set_params(class_weight=class_weight)
    pipe = Pipeline([("prep", make_prep(**prep_kwargs)), ("clf", clf)])
    return GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv(), n_jobs=2, refit=True)


def svm_search(class_weight=None, prep_kwargs: dict | None = None) -> GridSearchCV:
    """RBF SVM; inner grid scored on decision_function AUC, Platt-calibrated probabilities
    (SVC(probability=True) runs its own internal 5-fold Platt fit on the training fold)."""
    prep_kwargs = prep_kwargs or {}
    pipe = Pipeline([("prep", make_prep(**prep_kwargs)),
                     ("clf", SVC(kernel="rbf", probability=True, class_weight=class_weight,
                                 random_state=0))])
    grid = {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__gamma": [0.001, 0.003, 0.01, 0.03]}
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv(), n_jobs=2, refit=True)


def linear_attributions(fitted_pipeline: Pipeline, X: pd.DataFrame,
                        background: pd.DataFrame) -> pd.DataFrame:
    """Exact (interventional) SHAP for a fitted Pipeline(prep, LogisticRegression), in log-odds:
    phi_raw_feature = sum over its transformed columns of coef_k * (z_k(x) - mean_bg z_k).
    Rows sum (with the base value) to the model's decision_function."""
    prep, clf = fitted_pipeline.named_steps["prep"], fitted_pipeline.named_steps["clf"]
    Z = prep.transform(X)
    Zbg = prep.transform(background).mean(axis=0)
    names = prep.named_steps["ct"].get_feature_names_out()
    contrib = (Z - Zbg) * clf.coef_[0]
    raw = []
    for n in names:  # map transformed column (e.g. "BBB_2.0", "Age_sp_3") back to its raw feature
        hits = [c for c in X.columns if n == c or n.startswith(c + "_")]
        raw.append(max(hits, key=len) if hits else n)
    out = pd.DataFrame(contrib, columns=names, index=X.index).T.groupby(raw).sum().T
    return out


# ---------------------------------------------------------------------------------------
# Fast nested-C search + Platt recalibration that keeps the model exactly linear.
# ---------------------------------------------------------------------------------------
from sklearn.base import ClassifierMixin, clone  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

C_GRID_WIDE = list(np.logspace(-4, 1, 11))  # 1e-4 .. 10


class CalibratedLinearCV(BaseEstimator, ClassifierMixin):
    """Inner stratified K-fold over C (prep refit per inner fold, reused across the C grid),
    choose C by mean inner-fold ROC-AUC, refit Pipeline(prep, LR) on the whole training fold,
    then Platt-scale its log-odds with a 1-D logistic fit on the inner OOF decision values
    of the chosen C.  Final log-odds = a * (w.z + c) + b  ->  still exactly linear in z, so
    attributions are a * coef * (z - mean z).

    penalty: 'l2' | 'l1' | 'elasticnet' (l1_ratio) ; clf='lda' uses shrinkage LDA instead
    (no C grid; Ledoit-Wolf shrinkage chosen analytically)."""

    def __init__(self, penalty="l2", Cs=None, l1_ratio=0.5, class_weight=None, prep_kwargs=None,
                 inner_splits=5, random_state=0, calibrate=True, select="roc_auc", clf="lr"):
        self.penalty = penalty
        self.Cs = Cs
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.prep_kwargs = prep_kwargs
        self.inner_splits = inner_splits
        self.random_state = random_state
        self.calibrate = calibrate
        self.select = select
        self.clf = clf

    def _make_clf(self, C):
        if self.clf == "lda":
            return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        if self.penalty == "l2":
            return LogisticRegression(C=C, solver="lbfgs", max_iter=5000,
                                      class_weight=self.class_weight)
        if self.penalty == "l1":
            return LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=5000,
                                      class_weight=self.class_weight)
        return LogisticRegression(C=C, penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio,
                                  max_iter=5000, tol=1e-4, class_weight=self.class_weight)

    def fit(self, X, y):
        y = np.asarray(y)
        Cs = [1.0] if self.clf == "lda" else list(self.Cs or C_GRID_WIDE)
        pk = self.prep_kwargs or {}
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        oof = np.zeros((len(Cs), len(y)))
        scores = np.zeros((len(Cs), self.inner_splits))
        for k, (tr, te) in enumerate(cv.split(X, y)):
            prep = make_prep(**pk).fit(X.iloc[tr])
            Ztr, Zte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
            for i, C in enumerate(Cs):
                m = self._make_clf(C).fit(Ztr, y[tr])
                d = m.decision_function(Zte)
                oof[i, te] = d
                scores[i, k] = roc_auc_score(y[te], d)
        if self.select == "roc_auc":
            best = int(np.argmax(scores.mean(axis=1)))
        else:  # pooled OOF log-loss after Platt scaling
            lls = []
            for i in range(len(Cs)):
                pl = LogisticRegression(C=1e6).fit(oof[i].reshape(-1, 1), y)
                lls.append(log_loss(y, pl.predict_proba(oof[i].reshape(-1, 1))[:, 1]))
            best = int(np.argmin(lls))
        self.C_ = Cs[best]
        self.inner_scores_ = scores.mean(axis=1)
        self.pipeline_ = Pipeline([("prep", make_prep(**pk)), ("clf", self._make_clf(self.C_))])
        self.pipeline_.fit(X, y)
        if self.calibrate:
            pl = LogisticRegression(C=1e6).fit(oof[best].reshape(-1, 1), y)
            self.a_, self.b_ = float(pl.coef_[0, 0]), float(pl.intercept_[0])
        else:
            self.a_, self.b_ = 1.0, 0.0
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.a_ * self.pipeline_.decision_function(X) + self.b_

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def attributions(self, X, background):
        """Exact log-odds attributions per raw column; base + row sum == decision_function."""
        phi = linear_attributions(self.pipeline_, X, background) * self.a_
        return phi
