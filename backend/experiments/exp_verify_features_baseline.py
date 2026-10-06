"""Baseline XGB under a verify_* name so extra seed offsets (e.g. 200) can be paired against it."""
import warnings

import exp_baseline
from harness import evaluate, load_xy

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    X, Y, _ = load_xy()
    for t in ["cad", "lad", "lcx", "rca"]:
        evaluate("verify_features_baseline_xgb", t, exp_baseline.make_model_for(t), X, Y)
