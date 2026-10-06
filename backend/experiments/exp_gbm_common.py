"""Shared estimators for the 'gbm' strategy (tuned, strongly regularised gradient boosting).

Everything learned from data lives inside these estimators, so the harness re-fits it per
outer training fold:
  * CatNative      - CatBoost with native categorical handling of BBB / VHD (/ Sex).
  * OptunaGBM      - nested hyper-parameter search (Optuna TPE, <=30 trials, 3-fold inner CV)
                     for xgb / lgb / cat; n_estimators is chosen for free from staged
                     predictions of the largest model inside each trial.
  * MarginBag      - averages raw log-odds of several seeds (attributions stay additive:
                     mean of member SHAP rows).
All models take the 55 label-encoded raw columns exactly as harness.load_xy() returns them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

CAT_COLS_DEFAULT = ("BBB", "VHD")


# --------------------------------------------------------------------------- builders
def make_xgb(**kw):
    from xgboost import XGBClassifier
    base = dict(n_estimators=500, max_depth=2, learning_rate=0.03, subsample=0.7,
                colsample_bytree=0.5, min_child_weight=3, reg_lambda=5.0, reg_alpha=0.5,
                eval_metric="logloss", random_state=0, n_jobs=1, tree_method="hist")
    base.update(kw)
    return XGBClassifier(**base)


def make_lgb(**kw):
    from lightgbm import LGBMClassifier
    base = dict(n_estimators=500, max_depth=2, num_leaves=4, learning_rate=0.03,
                subsample=0.7, subsample_freq=1, colsample_bytree=0.5, min_child_samples=10,
                reg_lambda=5.0, reg_alpha=0.5, random_state=0, n_jobs=1, verbose=-1)
    base.update(kw)
    return LGBMClassifier(**base)


def make_cat(**kw):
    from catboost import CatBoostClassifier
    base = dict(iterations=800, depth=3, learning_rate=0.03, l2_leaf_reg=5, rsm=0.5,
                random_seed=0, thread_count=1, verbose=0, allow_writing_files=False)
    base.update(kw)
    return CatBoostClassifier(**base)


class CatNative(BaseEstimator, ClassifierMixin):
    """CatBoost with BBB/VHD (optionally Sex) passed as native categorical features."""

    def __init__(self, params=None, cat_cols=CAT_COLS_DEFAULT):
        self.params = params
        self.cat_cols = cat_cols

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].round().astype(int).astype(str)
        return X

    def fit(self, X, y):
        self.model_ = make_cat(**(self.params or {}))
        self.model_.set_params(cat_features=[c for c in self.cat_cols if c in X.columns])
        self.model_.fit(self._prep(X), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X, ntree_end=0):
        return self.model_.predict_proba(self._prep(X), ntree_end=ntree_end)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def shap_values(self, X):
        from catboost import Pool
        Xp = self._prep(X)
        return self.model_.get_feature_importance(
            Pool(Xp, cat_features=list(self.cat_cols)), type="ShapValues")


# --------------------------------------------------------------------------- staged preds
def _staged_proba(model, X, k, lib):
    if lib == "xgb":
        return model.predict_proba(X, iteration_range=(0, k))[:, 1]
    if lib == "lgb":
        return model.predict_proba(X, num_iteration=k)[:, 1]
    if lib in ("cat", "catn"):
        return model.predict_proba(X, ntree_end=k)[:, 1]
    raise ValueError(lib)


def _build(lib, params, seed):
    if lib == "xgb":
        return make_xgb(**params, random_state=seed)
    if lib == "lgb":
        return make_lgb(**params, random_state=seed)
    if lib == "cat":
        return make_cat(**params, random_seed=seed)
    if lib == "catn":
        p = dict(params)
        p["random_seed"] = seed
        return CatNative(params=p)
    raise ValueError(lib)


def _space(trial, lib):
    if lib == "xgb":
        return dict(
            max_depth=trial.suggest_int("max_depth", 1, 3),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 12, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.25, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
        )
    if lib == "lgb":
        d = trial.suggest_int("max_depth", 1, 3)
        return dict(
            max_depth=d, num_leaves=2 ** d,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
            subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.25, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
            extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
        )
    if lib in ("cat", "catn"):
        return dict(
            depth=trial.suggest_int("depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            rsm=trial.suggest_float("rsm", 0.3, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        )
    raise ValueError(lib)


def _with_n(lib, params, n):
    p = dict(params)
    if lib in ("cat", "catn"):
        p["iterations"] = n
    else:
        p["n_estimators"] = n
    return p


class OptunaGBM(BaseEstimator, ClassifierMixin):
    """Nested HP search. fit() runs an Optuna study on the training data only (inner
    stratified k-fold), picks params + number of trees, then refits on all training data."""

    def __init__(self, lib="xgb", n_trials=25, inner_splits=3, n_max=800, step=50,
                 objective="auc", random_state=0):
        self.lib = lib
        self.n_trials = n_trials
        self.inner_splits = inner_splits
        self.n_max = n_max
        self.step = step
        self.objective = objective
        self.random_state = random_state

    def fit(self, X, y):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        folds = list(cv.split(X, y))
        checkpoints = list(range(self.step, self.n_max + 1, self.step))

        def objective(trial):
            params = _space(trial, self.lib)
            scores = np.zeros((len(folds), len(checkpoints)))
            for i, (tr, te) in enumerate(folds):
                m = _build(self.lib, _with_n(self.lib, params, self.n_max), self.random_state)
                m.fit(X.iloc[tr], y[tr])
                for j, k in enumerate(checkpoints):
                    p = _staged_proba(m, X.iloc[te], k, self.lib)
                    if self.objective == "auc":
                        scores[i, j] = roc_auc_score(y[te], p)
                    else:
                        scores[i, j] = -log_loss(y[te], np.clip(p, 1e-6, 1 - 1e-6))
            mean = scores.mean(0)
            j = int(np.argmax(mean))
            trial.set_user_attr("n", checkpoints[j])
            return float(mean[j])

        sampler = optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=8)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)
        best = study.best_trial
        self.best_params_ = _with_n(self.lib, _space(optuna.trial.FixedTrial(best.params), self.lib),
                                    best.user_attrs["n"])
        self.best_score_ = best.value
        self.model_ = _build(self.lib, self.best_params_, self.random_state)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MarginBag(BaseEstimator, ClassifierMixin):
    """Average raw log-odds of `n_seeds` clones with different seeds."""

    def __init__(self, estimator=None, n_seeds=5, seed_param="random_state"):
        self.estimator = estimator
        self.n_seeds = n_seeds
        self.seed_param = seed_param

    def fit(self, X, y):
        self.models_ = []
        for s in range(self.n_seeds):
            m = clone(self.estimator)
            m.set_params(**{self.seed_param: s})
            self.models_.append(m.fit(X, y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        # average in log-odds space (keeps per-feature attributions additive)
        qs = [np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9) for m in self.models_]
        lo = np.mean([np.log(q / (1 - q)) for q in qs], 0)
        p = 1 / (1 + np.exp(-lo))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MarginAvg(BaseEstimator, ClassifierMixin):
    """Average raw log-odds of several different estimators (attributions: mean of members)."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.models_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        w = np.ones(len(self.models_)) if self.weights is None else np.asarray(self.weights, float)
        w = w / w.sum()
        qs = [np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9) for m in self.models_]
        lo = np.sum([wi * np.log(q / (1 - q)) for wi, q in zip(w, qs)], 0)
        p = 1 / (1 + np.exp(-lo))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class NestedSelect(BaseEstimator, ClassifierMixin):
    """Pick one of a few candidate estimators by repeated inner stratified CV AUC on the
    training fold only, then refit the winner on the whole training fold."""

    def __init__(self, candidates=None, inner_splits=3, inner_repeats=2, random_state=0):
        self.candidates = candidates
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        names = list(self.candidates)
        scores = np.zeros(len(names))
        for r in range(self.inner_repeats):
            cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state + r)
            for tr, te in cv.split(X, y):
                for i, n in enumerate(names):
                    m = clone(self.candidates[n]).fit(X.iloc[tr], y[tr])
                    scores[i] += roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1])
        self.scores_ = dict(zip(names, scores / (self.inner_splits * self.inner_repeats)))
        self.choice_ = names[int(np.argmax(scores))]
        self.model_ = clone(self.candidates[self.choice_]).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
