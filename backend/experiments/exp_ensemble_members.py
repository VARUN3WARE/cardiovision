"""Step 1: evaluate each candidate ensemble member alone (saved as ensemble_m_<member>).

Because every member is deterministic and the harness uses identical folds, the OOF of a
fixed-weight log-odds vote equals sigmoid(mean_k logit(OOF_k)) exactly -> combos can be
screened from these arrays (exp_ensemble_combos.py) and then confirmed with real estimators.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings

import numpy as np

from harness import RESULTS_DIR, evaluate, load_xy
from exp_ensemble_common import member_factory

warnings.filterwarnings("ignore")
X, Y, _ = load_xy()
MEMBERS = sys.argv[1:] or ["xgb", "xgb_nospw", "lr", "lr_nolog", "lr_l1", "lr_auc",
                           "cat", "cat_d6", "rf", "xgbrf"]

if __name__ == "__main__":
    for m in MEMBERS:
        for t in ["cad", "lad", "lcx", "rca"]:
            evaluate(f"ensemble_m_{m}", t, member_factory(m, X.columns, Y[t]), X, Y)
    # determinism check: our 'xgb' member must reproduce baseline_xgb exactly
    if "xgb" in MEMBERS:
        for t in ["cad", "lad", "lcx", "rca"]:
            a = np.load(RESULTS_DIR / f"ensemble_m_xgb__{t}.oof.npy")
            b = np.load(RESULTS_DIR / f"baseline_xgb__{t}.oof.npy")
            print("xgb == baseline_xgb", t, np.allclose(a, b, atol=1e-6))
