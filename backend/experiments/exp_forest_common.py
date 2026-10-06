"""forest: shared estimators for the bagged-tree experiments (RandomForest / ExtraTrees).

All estimators take the 55 raw label-encoded columns and learn everything (config choice,
calibration map) inside fit(), so they are refit per outer training fold by the harness.

Production-relevant property: every calibrated model here is ONE fitted forest followed by
a monotone 1-D map of its probability.  For the sigmoid (Platt) map the calibrated
log-odds are  logit(q) = -(a * p_forest(x) + b),  which is LINEAR in p_forest, so exact
TreeSHAP values of the forest (shap.TreeExplainer works for sklearn forests and is exact /
additive in probability space) convert to exact log-odds attributions: phi_logit = -a*phi_p.
See forest_logodds_attributions().
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ALGOS = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}


def make_forest(algo="rf", n_estimators=300, random_state=42, **params):
    return ALGOS[algo](n_estimators=n_estimators, n_jobs=1, random_state=random_state, **params)


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _fit_predict(cand, X, y, tr, te, n_estimators, random_state):
    m = make_forest(n_estimators=n_estimators, random_state=random_state, **cand)
    m.fit(_rows(X, tr), y[tr])
    return m.predict_proba(_rows(X, te))[:, 1]


def _make_calibrator(method, p, y):
    if method == "sigmoid":
        return _SigmoidCalibration().fit(p, y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return None


class SearchCalibratedForest(BaseEstimator, ClassifierMixin):
    """Nested model selection + calibration in one inner-CV pass.

    fit(): for every candidate config, inner StratifiedKFold OOF probabilities on the
    training fold -> pick the candidate with the best inner score (AUC or log-loss after
    Platt) -> fit the calibrator on THAT candidate's inner OOF probabilities -> refit the
    chosen forest on the whole training fold.  (Equivalent to CalibratedClassifierCV
    (ensemble=False) on the selected config, without paying for a second CV.)"""

    def __init__(self, candidates, inner_splits=4, scoring="auc", calibration="sigmoid",
                 n_estimators=300, random_state=0, n_jobs=2):
        self.candidates = candidates
        self.inner_splits = inner_splits
        self.scoring = scoring
        self.calibration = calibration
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        folds = list(cv.split(np.zeros(len(y)), y))
        jobs = [(ci, fi) for ci in range(len(self.candidates)) for fi in range(len(folds))]
        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict)(self.candidates[ci], X, y, folds[fi][0], folds[fi][1],
                                  self.n_estimators, 42) for ci, fi in jobs)
        oof = np.zeros((len(self.candidates), len(y)))
        for (ci, fi), p in zip(jobs, preds):
            oof[ci, folds[fi][1]] = p
        scores = []
        for ci in range(len(self.candidates)):
            if self.scoring == "auc":
                scores.append(roc_auc_score(y, oof[ci]))
            else:  # negative log-loss of the Platt-calibrated inner OOF probs
                q = _SigmoidCalibration().fit(oof[ci], y).predict(oof[ci])
                scores.append(-log_loss(y, np.clip(q, 1e-6, 1 - 1e-6)))
        self.inner_scores_ = np.array(scores)
        self.best_index_ = int(np.argmax(scores))
        self.best_params_ = dict(self.candidates[self.best_index_])
        self.calibrator_ = _make_calibrator(self.calibration, oof[self.best_index_], y)
        self.model_ = make_forest(n_estimators=self.n_estimators, random_state=42,
                                  **self.best_params_).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            p = np.clip(self.calibrator_.predict(p), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class OOBCalibratedForest(BaseEstimator, ClassifierMixin):
    """Bootstrap forest whose Platt/isotonic map is fit on its own out-of-bag probabilities:
    no extra fits, and the final forest uses all training rows."""

    def __init__(self, algo="rf", params=None, calibration="sigmoid", n_estimators=300,
                 random_state=42):
        self.algo = algo
        self.params = params
        self.calibration = calibration
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        params = dict(self.params or {})
        params.update(bootstrap=True, oob_score=True)
        self.model_ = make_forest(self.algo, n_estimators=self.n_estimators,
                                  random_state=self.random_state, **params).fit(X, y)
        p_oob = self.model_.oob_decision_function_[:, 1]
        ok = np.isfinite(p_oob)
        self.calibrator_ = _make_calibrator(self.calibration, p_oob[ok], y[ok])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            p = np.clip(self.calibrator_.predict(p), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def forest_logodds_attributions(forest, sigmoid_calibrator, X):
    """Exact per-feature log-odds attributions for  q(x) = expit(-(a*p_forest(x) + b)).

    Returns (base_logit, phi) with base_logit + phi.sum(1) == logit(q(x)) exactly.
    Uses shap.TreeExplainer on the sklearn forest (exact TreeSHAP; additive in P space)."""
    import shap
    a, b = float(sigmoid_calibrator.a_), float(sigmoid_calibrator.b_)
    ex = shap.TreeExplainer(forest)
    sv = np.asarray(ex.shap_values(X))[..., 1]          # (n, d) probability-space SHAP
    base_p = float(np.atleast_1d(ex.expected_value)[1])
    return -(a * base_p + b), -a * sv


class RFETLogitBlend(BaseEstimator, ClassifierMixin):
    """Recommended production model (forest_final_blend): mean of the calibrated log-odds of
    a Platt-calibrated RandomForest and a Platt-calibrated ExtraTrees, both
    max_features='sqrt', min_samples_leaf=3, class_weight='balanced_subsample'.
    Each member is CalibratedClassifierCV(sigmoid, inner 5-fold, ensemble=False), i.e. one
    forest fit on all rows + one sigmoid fit on inner-CV probabilities.

    explain(X) -> (base_logit, phi) with base_logit + phi.sum(1) == logit(predict_proba)
    exactly (TreeSHAP per member, scaled by its Platt slope, averaged)."""

    def __init__(self, n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                 class_weight="balanced_subsample", random_state=42):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def _member(self, algo):
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(
            make_forest(algo, n_estimators=self.n_estimators, random_state=self.random_state,
                        max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
                        class_weight=self.class_weight),
            method="sigmoid", ensemble=False,
            cv=StratifiedKFold(5, shuffle=True, random_state=0))

    def fit(self, X, y):
        self.members_ = [self._member(a).fit(X, y) for a in ("rf", "et")]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        eps = 1e-6
        z = [np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
             for p in (m.predict_proba(X)[:, 1] for m in self.members_)]
        return 0.5 * (z[0] + z[1])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def explain(self, X):
        base, phi = 0.0, 0.0
        for m in self.members_:
            cc = m.calibrated_classifiers_[0]
            b, ph = forest_logodds_attributions(cc.estimator, cc.calibrators[0], X)
            base, phi = base + 0.5 * b, phi + 0.5 * ph
        return base, phi
