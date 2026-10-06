"""Training pipeline for Track A cardiovascular risk prediction.

Fits the production model for each target (CAD, LAD, LCX, RCA; see app/cardio_models.py),
evaluates it with the same leakage-safe protocol as backend/experiments/harness.py, and writes
everything the API and UI need:

  models/<target>.joblib      fitted model with decision thresholds attached
  models/features.json        the 55 input columns (order matters)
  models/encoding.json        category -> code map
  models/metrics.json         cross-validated performance, operating points, calibration bands
  models/feature_info.json    training ranges, imputation defaults, per-feature influence
  models/oof_predictions.npz  out-of-fold probabilities (used by make_presets.py)

Run:  python train.py            (about 4-5 minutes on a 4+ core laptop; targets train in parallel)
"""
from __future__ import annotations

import io
import json
import os
import time
import warnings
import zipfile
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from app.cardio_models import (  # noqa: E402
    MODEL_DESCRIPTIONS, check_encoding, contributions, display_attributions,
    fit_with_thresholds, make_model,
)
from app.config import (  # noqa: E402
    FEATURE_INFO_PATH, FEATURES_PATH, LEAKAGE_COLUMNS, METRICS_PATH, MODELS_DIR,
    TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE,
)
from app.encoders import (  # noqa: E402
    binarize_target, encode_dataframe, learn_encoding, save_encoding,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "ZAlizadehSani.csv"
ZIP_PATH = DATA_DIR / "extention.zip"  # UCI download, committed to the repo
CV_REPEATS = 5
CV_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    """Read the CSV, creating it from the committed UCI zip (xlsx) on a fresh clone."""
    if not DATA_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as z:
            name = next(n for n in z.namelist() if n.endswith(".xlsx"))
            pd.read_excel(io.BytesIO(z.read(name))).to_csv(DATA_PATH, index=False)
        print(f"Extracted {name} -> {DATA_PATH.name}")
    return pd.read_csv(DATA_PATH)


def build_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LEAKAGE_COLUMNS]


def _metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": recall_score(1 - y, 1 - pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def cross_validate(make, X: pd.DataFrame, y: np.ndarray) -> tuple[dict, np.ndarray]:
    """5x5 repeated stratified CV. Returns mean/std metrics and the (repeats, n) OOF matrix."""
    P = np.zeros((CV_REPEATS, len(y)))
    for r in range(CV_REPEATS):
        for tr, te in StratifiedKFold(CV_SPLITS, shuffle=True, random_state=r).split(X, y):
            P[r, te] = make().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    rows = []
    for r in range(CV_REPEATS):
        p = np.clip(P[r], 1e-6, 1 - 1e-6)
        rows.append({"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
                     "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), **_metrics_at(y, p, 0.5)})
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "std": float(df[k].std(ddof=0))} for k in df.columns}, P


def nested_operating_point(make, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Honest metrics for the tuned decision threshold: thresholds are chosen inside each
    training fold (inner CV) and applied to the held-out fold."""
    pred = np.zeros(len(y))
    rule_out = np.zeros(len(y))
    for tr, te in StratifiedKFold(CV_SPLITS, shuffle=True, random_state=0).split(X, y):
        m = fit_with_thresholds(make, X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        pred[te] = p >= m.threshold_
        rule_out[te] = p >= m.rule_out_threshold_
    out = {"f1_threshold": _metrics_at(y, pred, 0.5), "rule_out_threshold": _metrics_at(y, rule_out, 0.5)}
    neg = rule_out == 0
    out["rule_out_threshold"]["npv"] = float((y[neg] == 0).mean()) if neg.any() else None
    return out


def calibration_bands(y: np.ndarray, P: np.ndarray, edges: list[float], labels: list[str]) -> list[dict]:
    """Observed event rate per predicted-probability band, pooled over the CV repeats."""
    p, yy = P.ravel(), np.tile(y, P.shape[0])
    bands = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        n = int(m.sum())
        bands.append({"label": label, "lo": lo, "hi": hi,
                      "share_of_patients": round(n / len(p), 4),
                      "observed_rate": round(float(yy[m].mean()), 4) if n else None,
                      "mean_predicted": round(float(p[m].mean()), 4) if n else None})
    return bands


def feature_info(df: pd.DataFrame, encoding: dict, feature_names: list[str]) -> dict:
    """Training ranges (for out-of-range warnings) and defaults for values the user did not enter.

    Defaults follow the usual form convention that an unrecorded finding is absent: binary and
    Y/N findings default to "no", categories to their "none" level ("N") when they have one, and
    numeric measurements to the cohort median. (Using the most common value instead would assume
    e.g. typical angina and hypertension, present in 54% / 59% of this referral cohort.)"""
    info = {}
    for col in feature_names:
        if col in encoding:
            cats = sorted(df[col].astype(str).unique().tolist())
            default = "N" if "N" in encoding[col] else str(df[col].mode().iloc[0])
            info[col] = {"type": "category", "default": default, "categories": cats}
        else:
            s = df[col].astype(float)
            binary = set(s.unique()) <= {0.0, 1.0}
            info[col] = {
                "type": "binary" if binary else "numeric",
                "default": 0.0 if binary else float(s.median()),
                "min": float(s.min()), "p5": float(s.quantile(0.05)), "median": float(s.median()),
                "p95": float(s.quantile(0.95)), "max": float(s.max()),
                "prevalence": float(s.mean()) if binary else None,
            }
    return info


def train_one(target: str, X: pd.DataFrame, y: np.ndarray) -> tuple[object, dict, np.ndarray]:
    warnings.filterwarnings("ignore")  # worker processes don't inherit the parent's filters
    make = lambda: make_model(target, y)  # noqa: E731
    t0 = time.time()
    cv, P = cross_validate(make, X, y)
    op = nested_operating_point(make, X, y)
    model = fit_with_thresholds(make, X, y)
    if target == "cad":
        edges = [0.0, model.rule_out_threshold_, model.balanced_threshold_, 1.0]
        labels = ["Lower", "Intermediate", "Higher"]
    else:
        edges, labels = [0.0, 1 / 3, 2 / 3, 1.0], ["Lower", "Intermediate", "Higher"]
    pos = int(y.sum())
    metrics = {
        "model": MODEL_DESCRIPTIONS[target],
        "n": int(len(y)), "positive": pos, "negative": int(len(y) - pos), "prevalence": pos / len(y),
        "validation": f"{CV_SPLITS}-fold stratified cross-validation x {CV_REPEATS} repeats",
        "cv": cv,
        # Legacy flat keys (mean over repeats, decisions at 0.5) for older readers.
        **{k: cv[k]["mean"] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "operating_points": op,
        "thresholds": {"f1": model.threshold_, "rule_out": model.rule_out_threshold_,
                       "balanced": model.balanced_threshold_},
        "bands": calibration_bands(y, P, edges, labels),
        "train_seconds": round(time.time() - t0, 1),
    }
    m = cv
    print(f"[{target}] AUC {m['roc_auc']['mean']:.3f}±{m['roc_auc']['std']:.3f}  PR-AUC {m['pr_auc']['mean']:.3f}  "
          f"Brier {m['brier']['mean']:.3f}  F1@.5 {m['f1']['mean']:.3f}  "
          f"F1@tuned {op['f1_threshold']['f1']:.3f}  thr={model.threshold_:.2f}  ({metrics['train_seconds']}s)",
          flush=True)
    return model, metrics, P


def main():
    df = load_dataset()
    feature_names = build_features(df)
    encoding = learn_encoding(df)
    check_encoding(encoding)
    save_encoding(encoding)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    X = encode_dataframe(df, encoding, feature_names)
    print(f"Using {len(feature_names)} input features (leakage columns excluded), "
          f"{CV_SPLITS}x{CV_REPEATS} repeated CV.")

    info = {"features": feature_info(df, encoding, feature_names), "influence": {}}
    ys = {t: binarize_target(df[TARGET_COLUMN_MAP[t]], TARGET_POSITIVE[TARGET_COLUMN_MAP[t]]).to_numpy()
          for t in TARGETS}
    results = Parallel(n_jobs=min(len(TARGETS), os.cpu_count() or 1))(
        delayed(train_one)(t, X, ys[t]) for t in TARGETS)
    all_metrics = {}
    for target, (model, metrics, _) in zip(TARGETS, results):
        joblib.dump(model, MODELS_DIR / f"{target}.joblib")
        all_metrics[target] = metrics
        # Mean |attribution| on the training data, per display feature (log-odds units).
        C, _ = contributions(model, X)
        info["influence"][target] = {k: round(float(v), 4)
                                     for k, v in display_attributions(C).abs().mean().items()}

    # Out-of-fold probabilities (repeats x patients) for make_presets.py's held-out estimates.
    np.savez(MODELS_DIR / "oof_predictions.npz", **{t: P for t, (_, _, P) in zip(TARGETS, results)})
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(FEATURE_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved models, metrics and feature info to {MODELS_DIR}")
    print("Start the API:  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
