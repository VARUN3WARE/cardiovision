"""Step 3: run real ensemble estimators through the harness.

usage: python exp_ensemble_run.py <config> [targets...]
Saves results as experiment "ensemble_<config>" (+ "__seed100" suffix when HARNESS_SEED_OFFSET=100)
and a side file <tag>.calib.json with calibration slope/intercept/ECE per target.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import warnings

import numpy as np

from harness import RESULTS_DIR, SEED_OFFSET, evaluate, load_xy
from exp_ensemble_common import (
    IsotonicCal, LogOddsVoting, PlattLogit, StackedLogit, calibration_stats, member_factory,
)

warnings.filterwarnings("ignore")
X, Y, _ = load_xy()


def members(names, t):
    return [(n, member_factory(n, X.columns, Y[t])()) for n in names]


def vote(names):
    return lambda t: (lambda: LogOddsVoting(members(names, t)))


def stack(names, l2=1e-2, inner=5):
    return lambda t: (lambda: StackedLogit(members(names, t), inner_splits=inner, l2=l2))


def stack3(names):
    return stack(names, inner=3)


def platt(names):
    return lambda t: (lambda: PlattLogit(LogOddsVoting(members(names, t)), inner_splits=5))


def iso(names):
    return lambda t: (lambda: IsotonicCal(LogOddsVoting(members(names, t)), inner_splits=5))


ALIASES = {
    "lxrc": ["lr", "xgb", "rf", "cat"],          # the spec ensemble (sklearn RF)
    "lxfc": ["lr", "xgb", "xgbrf", "cat"],       # spec with XGB-RF instead of sklearn RF
    "lxc": ["lr", "xgb", "cat"],
    "lc": ["lr", "cat"],
    "l1xf": ["lr_l1", "xgb", "xgbrf"],           # universal candidate (positive min-delta on all 4)
    "l1xcf": ["lr_l1", "xgb", "cat", "xgbrf"],
    "l1xc6f": ["lr_l1", "xgb", "cat_d6", "xgbrf"],
    "l1c": ["lr_l1", "cat"],                     # CAD candidate
    "l1f": ["lr_l1", "xgbrf"],                   # RCA candidate
    "lc6f": ["lr", "cat_d6", "xgbrf"],           # LAD candidate
    "l1c6f": ["lr_l1", "cat_d6", "xgbrf"],
    "l1x": ["lr_l1", "xgb"],                     # LCX candidate
    "xrf": ["xgbrf_rf"],                         # LAD: RF-like XGBoost forest alone
    "c6xrf": ["cat_d6", "xgbrf_rf"],
    "lc6xrf": ["lr", "cat_d6", "xgbrf_rf"],
    "l1xrf": ["lr_l1", "xgbrf_rf"],
    "l1xcxrf": ["lr_l1", "xgb", "cat", "xgbrf_rf"],   # stacking pool
}
KINDS = {"vote": vote, "stack": stack, "stack3": stack3, "platt": platt, "iso": iso}
CONFIGS = {f"{k}_{a}": KINDS[k](m) for k in KINDS for a, m in ALIASES.items()}

if __name__ == "__main__":
    cfg = sys.argv[1]
    targets = sys.argv[2:] or ["cad", "lad", "lcx", "rca"]
    name = f"ensemble_{cfg}"
    for t in targets:
        evaluate(name, t, CONFIGS[cfg](t), X, Y)
        tag = f"{name}__{t}" + (f"__seed{SEED_OFFSET}" if SEED_OFFSET else "")
        P = np.load(RESULTS_DIR / f"{tag}.oof.npy")
        cs = calibration_stats(Y[t], P)
        with open(RESULTS_DIR / f"{tag}.calib.json", "w") as f:
            json.dump(cs, f)
        print(f"   calib {t}: {cs}", flush=True)
