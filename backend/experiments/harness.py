"""Shared, leakage-safe evaluation harness for model experiments.

Protocol (identical for every experiment so numbers are comparable):
  * 303 patients, 55 input features (LAD/LCX/RCA/Cath excluded), categorical
    columns label-encoded with the fixed, label-free map from app.encoders.
  * Repeated stratified K-fold: 5 folds x 5 repeats (seeds 0..4). Every repeat
    produces a full set of out-of-fold (OOF) probabilities.
  * Anything learned from data (scaling, imputation, feature selection,
    hyper-parameter search, threshold choice, calibration) must live INSIDE the
    estimator passed in, so it is re-fitted on each training fold only.
  * Metrics are computed per repeat on that repeat's OOF predictions and then
    averaged; the std across repeats is reported alongside.

Usage from an experiment script:

    from harness import load_xy, evaluate_all
    def make(target): return Pipeline([...])
    evaluate_all("my_experiment", make)            # all 4 targets
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import LEAKAGE_COLUMNS, TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE  # noqa: E402
from app.encoders import binarize_target, encode_dataframe, learn_encoding  # noqa: E402

DATA_PATH = BACKEND / "data" / "ZAlizadehSani.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_SPLITS = 5
N_REPEATS = 5
# Verification re-runs use unseen splits: HARNESS_SEED_OFFSET=100 -> seeds 100..104.
SEED_OFFSET = int(os.environ.get("HARNESS_SEED_OFFSET", "0"))


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_xy() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Encoded feature frame, binary targets, and the raw frame (for feature engineering)."""
    df = load_raw()
    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X = encode_dataframe(df, learn_encoding(df), features)
    y = {t: binarize_target(df[TARGET_COLUMN_MAP[t]], TARGET_POSITIVE[TARGET_COLUMN_MAP[t]]).to_numpy()
         for t in TARGETS}
    return X, y, df


def oof_probabilities(make_model: Callable[[], BaseEstimator], X: pd.DataFrame, y: np.ndarray,
                      repeats: int = N_REPEATS, splits: int = N_SPLITS) -> np.ndarray:
    """(repeats, n) array of out-of-fold P(y=1)."""
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=r + SEED_OFFSET)
        for tr, te in cv.split(X, y):
            model = make_model()
            model.fit(X.iloc[tr], y[tr])
            out[r, te] = model.predict_proba(X.iloc[te])[:, 1]
    return out


def metrics_from_oof(y: np.ndarray, P: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    """Mean/std of metrics across repeats. thresholds: optional per-repeat decision thresholds."""
    rows = []
    for r in range(P.shape[0]):
        p = np.clip(P[r], 1e-6, 1 - 1e-6)
        thr = 0.5 if thresholds is None else float(thresholds[r])
        pred = (p >= thr).astype(int)
        rows.append({
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": (recall_score(y, pred, zero_division=0)
                                  + recall_score(1 - y, 1 - pred, zero_division=0)) / 2,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=0))} for k in df.columns}


def evaluate(name: str, target: str, make_model: Callable[[], BaseEstimator],
             X: pd.DataFrame | None = None, y: dict | None = None, save: bool = True) -> dict:
    if X is None or y is None:
        X, y, _ = load_xy()
    t0 = time.time()
    P = oof_probabilities(make_model, X, y[target])
    res = {"experiment": name, "target": target, "n": int(len(y[target])),
           "positives": int(y[target].sum()), "seconds": round(time.time() - t0, 1),
           "metrics": metrics_from_oof(y[target], P)}
    res["seed_offset"] = SEED_OFFSET
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{name}__{target}" + (f"__seed{SEED_OFFSET}" if SEED_OFFSET else "")
        np.save(RESULTS_DIR / f"{tag}.oof.npy", P)
        with open(RESULTS_DIR / f"{tag}.json", "w") as f:
            json.dump(res, f, indent=2)
    m = res["metrics"]
    print(f"[{name}] {target:3}  AUC {m['roc_auc']['mean']:.3f}±{m['roc_auc']['std']:.3f}  "
          f"PR-AUC {m['pr_auc']['mean']:.3f}  Brier {m['brier']['mean']:.3f}  "
          f"F1@.5 {m['f1']['mean']:.3f}  Acc@.5 {m['accuracy']['mean']:.3f}  ({res['seconds']}s)", flush=True)
    return res


def evaluate_all(name: str, make_model_for: Callable[[str], Callable[[], BaseEstimator]],
                 targets: list[str] | None = None, X: pd.DataFrame | None = None) -> dict:
    """make_model_for(target) -> zero-arg factory returning a fresh unfitted estimator."""
    Xd, y, _ = load_xy()
    X = Xd if X is None else X
    return {t: evaluate(name, t, make_model_for(t), X, y) for t in (targets or TARGETS)}


class ThresholdTuned(BaseEstimator, ClassifierMixin):
    """Wraps an estimator; picks the F1-optimal threshold from inner-CV OOF predictions
    on the training fold only, then refits on the whole training fold.
    predict() uses the tuned threshold; predict_proba() is unchanged."""

    def __init__(self, estimator: BaseEstimator, inner_splits: int = 4, random_state: int = 0):
        self.estimator = estimator
        self.inner_splits = inner_splits
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        oof = np.zeros(len(y))
        cv = StratifiedKFold(self.inner_splits, shuffle=True, random_state=self.random_state)
        for tr, te in cv.split(X, y):
            m = clone(self.estimator).fit(_rows(X, tr), y[tr])
            oof[te] = m.predict_proba(_rows(X, te))[:, 1]
        grid = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(y, (oof >= g).astype(int), zero_division=0) for g in grid]
        self.threshold_ = float(grid[int(np.argmax(f1s))])
        self.model_ = clone(self.estimator).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def tuned_threshold_metrics(name: str, target: str, make_model: Callable[[], BaseEstimator],
                            X: pd.DataFrame | None = None, y: dict | None = None) -> dict:
    """Like evaluate(), but decisions use a threshold tuned inside each training fold
    (nested), so F1/accuracy are honest for a tuned operating point."""
    if X is None or y is None:
        X, y, _ = load_xy()
    yt = y[target]
    P = np.zeros((N_REPEATS, len(yt)))
    preds = np.zeros((N_REPEATS, len(yt)))
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=r + SEED_OFFSET)
        for tr, te in cv.split(X, yt):
            m = ThresholdTuned(make_model(), random_state=r).fit(X.iloc[tr], yt[tr])
            P[r, te] = m.predict_proba(X.iloc[te])[:, 1]
            preds[r, te] = m.predict(X.iloc[te])
    rows = [{"f1": f1_score(yt, preds[r], zero_division=0),
             "accuracy": accuracy_score(yt, preds[r]),
             "precision": precision_score(yt, preds[r], zero_division=0),
             "recall": recall_score(yt, preds[r], zero_division=0)} for r in range(N_REPEATS)]
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=0))} for k in df.columns}


def paired_auc_delta(name_a: str, name_b: str, target: str, seed_offset: int = 0) -> dict:
    """AUC(a) - AUC(b) per repeat on IDENTICAL folds (paired), from saved OOF arrays.
    A gain is only meaningful if it is consistent across repeats (min_delta > 0)."""
    _, y, _ = load_xy()
    sfx = f"__seed{seed_offset}" if seed_offset else ""
    A = np.load(RESULTS_DIR / f"{name_a}__{target}{sfx}.oof.npy")
    B = np.load(RESULTS_DIR / f"{name_b}__{target}{sfx}.oof.npy")
    d = np.array([roc_auc_score(y[target], A[r]) - roc_auc_score(y[target], B[r]) for r in range(len(A))])
    return {"mean": float(d.mean()), "std": float(d.std(ddof=0)), "min": float(d.min()),
            "max": float(d.max()), "per_repeat": [round(float(x), 4) for x in d]}


if __name__ == "__main__":
    # python harness.py compare <exp_a> <exp_b> [seed_offset]
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        for t in TARGETS:
            try:
                print(t, paired_auc_delta(sys.argv[2], sys.argv[3], t, off))
            except FileNotFoundError as e:
                print(t, "missing:", e)
