"""Adversarial verification of the 'linear' strategy's claimed per-target winners.

Imports the claimant's own classes (no copies) so the exact code is tested, and saves under
verify_linear_* names so the claimant's result files are never overwritten.

    HARNESS_SEED_OFFSET=<0|100> python exp_verify_linear_run.py [baseline] [claimed]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

from harness import evaluate, load_xy  # noqa: E402
from exp_linear_common import CalibratedLinearCV  # noqa: E402
from exp_linear_blend import LogitBlend, xgb_baseline  # noqa: E402

LOG = dict(log_skewed=True)
LOG_ORD = dict(log_skewed=True, vhd="ordinal")

CLAIMED = {
    "cad": ("verify_linear_cl1_log", lambda: CalibratedLinearCV("l1", prep_kwargs=LOG)),
    "lad": ("verify_linear_blendl2_w70",
            lambda: LogitBlend(CalibratedLinearCV("l2", prep_kwargs=LOG), xgb_baseline("lad"), w=0.7)),
    "lcx": ("verify_linear_blendl1_w50",
            lambda: LogitBlend(CalibratedLinearCV("l1", prep_kwargs=LOG), xgb_baseline("lcx"), w=0.5)),
    "rca": ("verify_linear_cl2_logord", lambda: CalibratedLinearCV("l2", prep_kwargs=LOG_ORD)),
}

if __name__ == "__main__":
    what = sys.argv[1:] or ["baseline", "claimed"]
    X, Y, _ = load_xy()
    assert X.shape == (303, 55), X.shape
    assert not {"LAD", "LCX", "RCA", "Cath"} & set(X.columns)
    for t in ["cad", "lad", "lcx", "rca"]:
        if "baseline" in what:
            evaluate("verify_linear_baseline", t, lambda t=t: xgb_baseline(t), X, Y)
        if "claimed" in what:
            name, make = CLAIMED[t]
            evaluate(name, t, make, X, Y)
