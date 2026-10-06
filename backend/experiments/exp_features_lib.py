"""Clinically-motivated feature engineering + in-CV feature selection (key="features").

Everything here is either stateless (deterministic, label-free column maths on the
55 raw encoded columns) or is fitted inside .fit() on the training fold only, so the
whole object can be handed to the harness and is re-fitted per CV fold.

Input: the 55 encoded columns produced by app.encoders (label-encoded alphabetically):
  BBB {LBBB:0, N:1, RBBB:2}, VHD {Moderate:0, N:1, Severe:2, mild:3}, Sex {Fmale:0, Male:1}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

VHD_ORD = {0.0: 2.0, 1.0: 0.0, 2.0: 3.0, 3.0: 1.0}  # Moderate, N, Severe, mild -> N<mild<Moderate<Severe
CONSTANT = ["Exertional CP"]
RISK_FACTORS = ["DM", "HTN", "Current Smoker", "FH", "DLP", "Obesity"]
ECG_ABN = ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression"]
SKEWED = ["TG", "FBS", "CR", "ESR", "BUN", "WBC", "PLT"]

# Diamond-Forrester pre-test probability of CAD (%), ACC/AHA 2002 stable-angina table.
# rows: age band 30-39, 40-49, 50-59, 60-69 ; (male, female)
_DF = {
    3: [(76, 26), (87, 55), (93, 73), (94, 86)],  # typical angina
    2: [(34, 12), (51, 22), (65, 31), (72, 51)],  # atypical angina
    1: [(4, 2), (13, 3), (20, 7), (27, 14)],      # non-anginal chest pain
    0: [(1.9, 0.3), (5.5, 1.0), (9.7, 3.2), (12.3, 7.5)],  # asymptomatic
}


def _df_pretest(age, male, cp):
    band = np.clip(((np.asarray(age) - 30) // 10).astype(int), 0, 3)
    out = np.empty(len(band))
    for i, (b, m, c) in enumerate(zip(band, np.asarray(male), np.asarray(cp))):
        pm, pf = _DF[int(c)][b]
        out[i] = (pm if m >= 0.5 else pf) / 100.0
    return out


def _egfr_ckdepi2021(cr, age, male):
    cr = np.asarray(cr, float)
    age = np.asarray(age, float)
    male = np.asarray(male) >= 0.5
    kappa = np.where(male, 0.9, 0.7)
    alpha = np.where(male, -0.302, -0.241)
    r = cr / kappa
    return 142 * np.minimum(r, 1) ** alpha * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(male, 1.0, 1.012)


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Stateless clinical feature engineering on the 55 raw encoded columns.

    fix_encodings: VHD -> ordinal severity, BBB -> LBBB/RBBB one-hot, drop constant cols.
    engineered:    add lipid ratios, risk-factor / ECG counts, chest-pain class, DF pre-test
                   probability, RWMA flag, age x sex, eGFR, NLR, simple clinical cut-offs.
    log_labs:      log-transform right-skewed labs (helps linear models only).
    subset:        optional explicit list of engineered names to keep (None = all).
    """

    def __init__(self, fix_encodings=True, engineered=True, log_labs=False, subset=None):
        self.fix_encodings = fix_encodings
        self.engineered = engineered
        self.log_labs = log_labs
        self.subset = subset

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = list(self.transform(X.iloc[:2] if hasattr(X, "iloc") else X[:2]).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.copy()
        if self.fix_encodings:
            out["VHD"] = X["VHD"].map(VHD_ORD).fillna(0.0)
            out["BBB_LBBB"] = (X["BBB"] == 0).astype(float)
            out["BBB_RBBB"] = (X["BBB"] == 2).astype(float)
            out = out.drop(columns=["BBB"] + CONSTANT)
        if self.engineered:
            hdl = X["HDL"].clip(lower=1)
            cp = np.select([X["Typical Chest Pain"] == 1, X["Atypical"] == 1, X["Nonanginal"] == 1],
                           [3, 2, 1], default=0)
            male = X["Sex"]
            eng = {
                "TG/HDL": X["TG"] / hdl,
                "LDL/HDL": X["LDL"] / hdl,
                "AIP log(TG/HDL)": np.log10((X["TG"] / 88.57) / (hdl / 38.67)),
                "Risk factor count": X[RISK_FACTORS].sum(axis=1)
                + ((male == 1) & (X["Age"] >= 45) | (male == 0) & (X["Age"] >= 55)).astype(float),
                "ECG abnormality count": X[ECG_ABN].sum(axis=1) + (X["BBB"] != 1).astype(float),
                "Chest pain class": cp.astype(float),
                "DF pretest prob": _df_pretest(X["Age"], male, cp),
                "RWMA present": (X["Region RWMA"] > 0).astype(float),
                "Age x Male": X["Age"] * male,
                "eGFR": _egfr_ckdepi2021(X["CR"], X["Age"], male),
                "NLR": X["Neut"] / X["Lymph"].clip(lower=1),
                "EF below 50": (X["EF-TTE"] < 50).astype(float),
                "FBS 126+": (X["FBS"] >= 126).astype(float),
                "Systolic BP 140+": (X["BP"] >= 140).astype(float),
            }
            if self.subset is not None:
                eng = {k: v for k, v in eng.items() if k in self.subset}
            for k, v in eng.items():
                out[k] = np.asarray(v, float)
        if self.log_labs:
            for c in SKEWED + ([k for k in ["TG/HDL", "LDL/HDL", "NLR"] if k in out.columns]):
                out[c] = np.log(np.asarray(out[c], float) + 1.0)
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ------------------------------------------------------------------ selectors

def _discrete_mask(X: np.ndarray) -> np.ndarray:
    return np.array([len(np.unique(X[:, j])) <= 5 for j in range(X.shape[1])])


class L1Select(BaseEstimator, TransformerMixin):
    """Keep features with non-zero coefficient in an L1 logistic regression (standardised).
    Falls back to the top `min_features` by |coef| at a larger C if too few survive."""

    def __init__(self, C=0.1, min_features=3, random_state=0):
        self.C = C
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        Xa = np.asarray(X, float)
        Z = StandardScaler().fit_transform(Xa)
        C = self.C
        for _ in range(8):
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=self.random_state,
                                    max_iter=2000).fit(Z, y)
            coef = np.abs(lr.coef_.ravel())
            if (coef > 1e-6).sum() >= self.min_features:
                break
            C *= 2
        self.support_ = coef > 1e-6
        if self.support_.sum() < self.min_features:
            self.support_ = np.zeros_like(self.support_)
            self.support_[np.argsort(-coef)[: self.min_features]] = True
        self.columns_ = list(X.columns[self.support_]) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        return X.loc[:, self.columns_] if hasattr(X, "loc") and self.columns_ is not None \
            else np.asarray(X)[:, self.support_]


class InnerTunedSelect(BaseEstimator, ClassifierMixin):
    """Feature selection with the selection strength tuned by inner CV on the training fold.

    method="mi": rank by mutual information, grid = number of features k.
    method="l1": L1-logistic support, grid = C of the L1 model.
    `base` is the downstream estimator (fresh clone per fit). Scored by inner ROC-AUC.
    """

    def __init__(self, base, method="mi", grid=(5, 10, 15, 20, 30, 45, None), inner_splits=3, random_state=0):
        self.base = base
        self.method = method
        self.grid = grid
        self.inner_splits = inner_splits
        self.random_state = random_state

    def _select(self, X, y, g):
        if self.method == "mi":
            Xa = np.asarray(X, float)
            mi = mutual_info_classif(Xa, y, discrete_features=_discrete_mask(Xa), random_state=self.random_state)
            order = np.argsort(-mi)
            return lambda k: list(X.columns[order[: (k or X.shape[1])]])
        sel = L1Select(C=g, random_state=self.random_state).fit(X, y)
        return sel.columns_

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        scores = np.zeros(len(self.grid))
        for tr, te in cv.split(X, y):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ranker = self._select(Xtr, y[tr], None) if self.method == "mi" else None
            for i, g in enumerate(self.grid):
                cols = ranker(g) if self.method == "mi" else self._select(Xtr, y[tr], g)
                m = clone(self.base).fit(Xtr[cols], y[tr])
                scores[i] += roc_auc_score(y[te], m.predict_proba(Xte[cols])[:, 1])
        self.inner_scores_ = scores / self.inner_splits
        self.best_ = self.grid[int(np.argmax(scores))]
        if self.method == "mi":
            self.columns_ = self._select(X, y, None)(self.best_)
        else:
            self.columns_ = self._select(X, y, self.best_)
        self.model_ = clone(self.base).fit(X[self.columns_], y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(pd.DataFrame(X)[self.columns_])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ------------------------------------------------------------------ models

def make_lr(C=None, class_weight=None, scoring="neg_log_loss"):
    """Standardise + L2 logistic regression. C=None -> C tuned by inner 5-fold CV.
    Default scoring is log-loss: AUC-scored C selection picks very strong shrinkage, which keeps
    AUC but squashes probabilities toward the prior (bad Brier / F1 at 0.5)."""
    from sklearn.linear_model import LogisticRegressionCV
    if C is None:
        lr = LogisticRegressionCV(Cs=np.logspace(-3, 1, 13), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  scoring=scoring, max_iter=5000, class_weight=class_weight)
    else:
        lr = LogisticRegression(C=C, max_iter=5000, class_weight=class_weight)
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


def make_lr_l1(class_weight=None, scoring="neg_log_loss"):
    """Standardise + L1 logistic regression with C tuned by inner CV (embedded selection)."""
    from sklearn.linear_model import LogisticRegressionCV
    lr = LogisticRegressionCV(Cs=np.logspace(-2.5, 1, 12), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              penalty="l1", solver="liblinear", scoring=scoring, max_iter=5000,
                              class_weight=class_weight)
    return Pipeline([("scale", StandardScaler()), ("lr", lr)])


def make_xgb(spw):
    """Identical hyper-parameters to baseline_xgb."""
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, n_jobs=2)


class LogitBlend(BaseEstimator, ClassifierMixin):
    """Weighted average of member log-odds (attributions = weighted sum of member attributions)."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        self.models_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        w = np.ones(len(self.models_)) / len(self.models_) if self.weights is None else np.asarray(self.weights)
        z = sum(wi * _logit(m.predict_proba(X)[:, 1]) for wi, m in zip(w, self.models_))
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
