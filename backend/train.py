"""Training pipeline for Track A cardiovascular risk prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from app.config import (
    FEATURES_PATH, LEAKAGE_COLUMNS, METRICS_PATH, MODELS_DIR,
    TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE,
)
from app.encoders import (
    binarize_target, encode_dataframe, learn_encoding, save_encoding,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "ZAlizadehSani.csv"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LEAKAGE_COLUMNS]


def train_one(df, feature_names, encoding, target_name, target_col):
    y = binarize_target(df[target_col], TARGET_POSITIVE[target_col])
    X = encode_dataframe(df, encoding, feature_names)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    spw = neg / pos if pos else 1.0
    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=4,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n": int(len(y)), "positive": pos, "negative": neg,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
    print(f"\n=== {target_name} (target={target_col}) ===")
    print(f"  pos={pos} neg={neg} scale_pos_weight={spw:.3f}")
    print(f"  acc={metrics['accuracy']:.3f} prec={metrics['precision']:.3f} "
          f"rec={metrics['recall']:.3f} f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f}")
    model.fit(X, y)
    return model, metrics


def main():
    df = load_dataset()
    feature_names = build_features(df)
    encoding = learn_encoding(df)
    save_encoding(encoding)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Using {len(feature_names)} input features (leakage columns excluded).")
    print(f"Encoded {len(encoding)} categorical columns.")

    all_metrics = {}
    for target in TARGETS:
        target_col = TARGET_COLUMN_MAP[target]
        if target_col not in df.columns:
            print(f"WARNING: target column '{target_col}' missing — skipping {target}.")
            continue
        model, metrics = train_one(df, feature_names, encoding, target, target_col)
        joblib.dump(model, MODELS_DIR / f"{target}.joblib")
        all_metrics[target] = metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")
    print("\nDone. Start the API:  uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
