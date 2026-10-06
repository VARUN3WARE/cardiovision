"""Shared paths and constants (avoids circular imports between model/encoders)."""
from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURES_PATH = MODELS_DIR / "features.json"
ENCODING_PATH = MODELS_DIR / "encoding.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Columns that must NEVER be used as model inputs (target leakage).
LEAKAGE_COLUMNS = ["LAD", "LCX", "RCA", "Cath"]

TARGETS = ["cad", "lad", "lcx", "rca"]

# Internal target name -> dataset column.
TARGET_COLUMN_MAP = {
    "cad": "Cath",
    "lad": "LAD",
    "lcx": "LCX",
    "rca": "RCA",
}

# Dataset target column -> positive class label.
TARGET_POSITIVE = {
    "Cath": "CAD",
    "LAD": "Stenotic",
    "LCX": "Stenotic",
    "RCA": "Stenotic",
}
