"""Production models: per-target log-odds blends of a logistic regression on clinical
features and a shallow gradient-boosted tree model, with exact additive attributions.

Chosen by the experiments in backend/experiments/ (see exp_final_recipe.py and the
technical write-up). Verified on split seeds never used for selection (5x5 repeated CV):

    target  model                                              AUC (baseline XGB)
    cad     L2-LR on 8 core clinical features + LightGBM stumps     0.94 (0.90)
    lad     Platt LR(core) + XGBoost(engineered), recalibrated      0.86 (0.83)
    lcx     same as LAD                                             0.76 (0.745)
    rca     L1-LR on engineered features + LightGBM depth-2          0.745 (0.68-0.70)

Everything here is either stateless (ClinicalFeatures) or fitted on training data only.
Input to every model is a DataFrame with the 55 encoded dataset columns (app.encoders).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix, XGBClassifier

# ---------------------------------------------------------------------------
# Clinical feature engineering (stateless, label-free)
# ---------------------------------------------------------------------------

# app.encoders label-encodes categories alphabetically; ClinicalFeatures relies on these codes.
EXPECTED_ENCODING = {
    "BBB": {"LBBB": 0.0, "N": 1.0, "RBBB": 2.0},
    "VHD": {"Moderate": 0.0, "N": 1.0, "Severe": 2.0, "mild": 3.0},
    "Sex": {"Fmale": 0.0, "Male": 1.0},
}
VHD_ORD = {0.0: 2.0, 1.0: 0.0, 2.0: 3.0, 3.0: 1.0}  # Moderate, N, Severe, mild -> N<mild<Moderate<Severe
CONSTANT = ["Exertional CP"]
RISK_FACTORS = ["DM", "HTN", "Current Smoker", "FH", "DLP", "Obesity"]
ECG_ABN = ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression"]
SKEWED = ["TG", "FBS", "CR", "ESR", "BUN", "WBC", "PLT"]
CHEST_PAIN = ["Typical Chest Pain", "Atypical", "Nonanginal"]

# Pre-test probability of CAD (%). Symptomatic rows: ACC/AHA 2002 chronic stable angina guideline,
# Table 4 (combined Diamond-Forrester and CASS data). Asymptomatic row: Diamond & Forrester, NEJM 1979.
# rows: age band 30-39, 40-49, 50-59, 60-69 (ages <30 use the first band, >=70 the last); values (male, female)
_DF = {
    3: [(76, 26), (87, 55), (93, 73), (94, 86)],  # typical angina
    2: [(34, 12), (51, 22), (65, 31), (72, 51)],  # atypical angina
    1: [(4, 2), (13, 3), (20, 7), (27, 14)],      # non-anginal chest pain
    0: [(1.9, 0.3), (5.5, 1.0), (9.7, 3.2), (12.3, 7.5)],  # asymptomatic
}

# Derived clinical scores -> the raw inputs they are computed from (for UI tooltips).
DERIVED_SOURCES = {
    "TG/HDL": ["TG", "HDL"],
    "LDL/HDL": ["LDL", "HDL"],
    "AIP log(TG/HDL)": ["TG", "HDL"],
    "Risk factor count": RISK_FACTORS + ["Age", "Sex"],
    "ECG abnormality count": ECG_ABN + ["BBB"],
    "Chest pain class": CHEST_PAIN,
    "DF pretest prob": ["Age", "Sex"] + CHEST_PAIN,
    "RWMA present": ["Region RWMA"],
    "Age x Male": ["Age", "Sex"],
    "eGFR": ["CR", "Age", "Sex"],
    "NLR": ["Neut", "Lymph"],
    "EF below 50": ["EF-TTE"],
    "FBS 126+": ["FBS"],
    "Systolic BP 140+": ["BP"],
}

CORE_ENG = ["TG/HDL", "AIP log(TG/HDL)", "Risk factor count", "ECG abnormality count",
            "Chest pain class", "DF pretest prob", "RWMA present", "Age x Male"]


def check_encoding(encoding: dict) -> None:
    """Fail loudly if the category codes ClinicalFeatures depends on have changed."""
    for col, expected in EXPECTED_ENCODING.items():
        if encoding.get(col) != expected:
            raise ValueError(f"encoding for {col!r} is {encoding.get(col)}, expected {expected}")


def _df_pretest(age, male, cp):
    band = np.clip(((np.asarray(age, float) - 30) // 10).astype(int), 0, 3)
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

    fix_encodings: VHD -> ordinal severity, BBB -> LBBB/RBBB one-hot, drop constant columns.
    engineered:    lipid ratios, risk-factor / ECG counts, chest-pain class, Diamond-Forrester
                   pre-test probability, RWMA flag, age x sex, eGFR, NLR, clinical cut-offs.
    log_labs:      log-transform right-skewed labs (helps the linear models).
    subset:        explicit list of engineered features to keep (None = all).
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
            for c in SKEWED + [k for k in ("TG/HDL", "LDL/HDL", "NLR") if k in out.columns]:
                out[c] = np.log(np.asarray(out[c], float) + 1.0)
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------

def _lr_l2(scoring="neg_log_loss"):
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=np.logspace(-3, 1, 13), scoring=scoring, max_iter=5000,
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=0)))])


def _lr_l1():
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=np.logspace(-2.5, 1, 12), penalty="l1", solver="liblinear",
                                                 scoring="neg_log_loss", max_iter=5000, random_state=0,
                                                 cv=StratifiedKFold(5, shuffle=True, random_state=0)))])


def _xgb(spw):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=2)


def _lgb_stumps():
    """Depth-1 extra-trees boosting: an additive (GAM-like) model."""
    return LGBMClassifier(n_estimators=1000, max_depth=1, num_leaves=2, learning_rate=0.05, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=10, reg_lambda=5.0,
                          reg_alpha=0.5, extra_trees=True, random_state=0, n_jobs=2, verbose=-1)


def _lgb_d2():
    return LGBMClassifier(n_estimators=1200, max_depth=2, num_leaves=4, learning_rate=0.02, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.5, min_child_samples=20, reg_lambda=20.0,
                          reg_alpha=0.5, extra_trees=True, random_state=0, n_jobs=2, verbose=-1)


def _fe(subset=None, log=True):
    return ("fe", ClinicalFeatures(fix_encodings=True, engineered=True, log_labs=log, subset=subset))


# ---------------------------------------------------------------------------
# Blending in log-odds space
# ---------------------------------------------------------------------------

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
    """Recalibration logit' = a * margin + b, fit on inner-CV out-of-fold margins of the training
    data, then the base model is refit on all of it. Affine in log-odds, so the ranking is unchanged
    and attributions stay exact (multiply by a; base' = a * base + b)."""

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

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= getattr(self, "threshold_", 0.5)).astype(int)


MODEL_DESCRIPTIONS = {
    "cad": "Blend (log-odds) of L2 logistic regression on 8 clinical composites and LightGBM depth-1 trees",
    "lad": "Platt-recalibrated blend of calibrated logistic regression (clinical composites) and XGBoost on engineered features",
    "lcx": "Platt-recalibrated blend of calibrated logistic regression (clinical composites) and XGBoost on engineered features",
    "rca": "Blend (log-odds) of L1 logistic regression on engineered features and LightGBM depth-2 trees",
}


def make_model(target: str, y) -> BaseEstimator:
    """Unfitted production estimator for one target."""
    y = np.asarray(y)
    spw = (len(y) - y.sum()) / y.sum()
    if target == "cad":
        return MarginBlend([Pipeline([_fe(CORE_ENG), ("m", _lr_l2())]), _lgb_stumps()])
    if target in ("lad", "lcx"):
        lr_cal = CalibratedClassifierCV(Pipeline([_fe(CORE_ENG), ("m", _lr_l2(scoring="roc_auc"))]),
                                        method="sigmoid", ensemble=False,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=0))
        blend = MarginBlend([lr_cal, Pipeline([_fe(None, log=False), ("m", _xgb(spw))])])
        return PlattMargin(blend)  # XGBoost member is over-confident; recalibrate the blend
    if target == "rca":
        return MarginBlend([Pipeline([_fe(None), ("m", _lr_l1())]), _lgb_d2()])
    raise ValueError(f"unknown target {target!r}")


# ---------------------------------------------------------------------------
# Decision thresholds (learned on training data only)
# ---------------------------------------------------------------------------

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def pick_thresholds(oof, y, rule_out_sens=0.95):
    """F1-optimal, rule-out (sensitivity >= rule_out_sens) and balanced-accuracy thresholds."""
    y = np.asarray(y)
    f1s = [f1_score(y, (oof >= g).astype(int), zero_division=0) for g in THRESHOLD_GRID]
    t_f1 = float(THRESHOLD_GRID[int(np.argmax(f1s))])
    ok = [g for g in THRESHOLD_GRID if recall_score(y, (oof >= g).astype(int), zero_division=0) >= rule_out_sens]
    t_ro = float(max(ok)) if ok else float(THRESHOLD_GRID[0])
    bal = [recall_score(y, (oof >= g).astype(int), zero_division=0)
           + recall_score(1 - y, (oof < g).astype(int), zero_division=0) for g in THRESHOLD_GRID]
    t_bal = float(THRESHOLD_GRID[int(np.argmax(bal))])
    return round(t_f1, 4), round(t_ro, 4), round(t_bal, 4)


def fit_with_thresholds(make, X, y, inner=4, random_state=0):
    """Inner-CV out-of-fold probabilities -> thresholds, then refit on all rows."""
    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(inner, shuffle=True, random_state=random_state).split(X, y):
        oof[te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    t_f1, t_ro, t_bal = pick_thresholds(oof, y)
    model = make().fit(X, y)
    model.threshold_, model.rule_out_threshold_, model.balanced_threshold_ = t_f1, t_ro, t_bal
    return model


# ---------------------------------------------------------------------------
# Exact additive attributions (log-odds)
# ---------------------------------------------------------------------------

def _flatten(p):
    steps = []
    for _, s in p.steps:
        steps.extend(_flatten(s) if isinstance(s, Pipeline) else [s])
    return steps


def contributions(m, X):
    """(C, base) with base + C.sum(axis=1) == margin(m, X) exactly.

    Columns are the model's own features: raw columns (labs possibly log-transformed, BBB split
    into BBB_LBBB/BBB_RBBB, VHD as ordinal) plus named clinical composites."""
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
        raise TypeError(f"no attribution rule for {type(est).__name__}")
    return pd.DataFrame(out[:, :-1], columns=list(Z.columns), index=X.index), out[:, -1]


def display_attributions(C: pd.DataFrame) -> pd.DataFrame:
    """Merge columns that are functions of ONE raw input back onto it (exact): BBB one-hots -> BBB.
    Composite clinical scores stay as their own named rows."""
    C = C.copy()
    if {"BBB_LBBB", "BBB_RBBB"} & set(C.columns):
        # Add, never overwrite: in a blend, a tree member may attribute to the raw BBB column too.
        merged = C["BBB"] if "BBB" in C else 0.0
        for col in ("BBB_LBBB", "BBB_RBBB"):
            if col in C:
                merged = merged + C.pop(col)
        C["BBB"] = merged
    return C
