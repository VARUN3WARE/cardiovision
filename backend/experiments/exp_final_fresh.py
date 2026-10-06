"""Final decision: PRE-DECLARED head-to-head of the verified candidates on split seeds nobody
has used (HARNESS_SEED_OFFSET=400 and 500). Candidates and the decision rule were fixed before
these runs (see exp_final_compare.py output for seeds 0/100/200/300).

Decision rule (fixed in advance):
  1. Eligible = pooled fresh-seed (10 repeats) paired AUC delta vs baseline with min > 0.
  2. Among eligible, prefer the simplest (single linear < single GBM < 2-member blend < bigger),
     unless a more complex one beats it by >= 0.005 mean AUC with >= 80% of repeats positive.
  3. Ties broken by Brier.

usage: HARNESS_SEED_OFFSET=400 python exp_final_fresh.py [targets...]
Saves results as final_<candidate>__<target>__seed<offset>.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import warnings  # noqa: E402

from harness import evaluate, load_xy  # noqa: E402
from exp_features_run import CONFIGS as FEAT  # noqa: E402
from exp_verify_gbm_repro import make_baseline, make_pick  # noqa: E402
from exp_linear_common import CalibratedLinearCV  # noqa: E402
from exp_verify_ensemble_run import CONFIGS as ENS  # noqa: E402

warnings.filterwarnings("ignore")
X, Y, _ = load_xy()
assert X.shape == (303, 55) and not {"LAD", "LCX", "RCA", "Cath"} & set(X.columns)
LOG = dict(log_skewed=True)


def linear_blendl2_w70(t):
    from exp_linear_blend import LogitBlend, xgb_baseline
    return lambda: LogitBlend(CalibratedLinearCV("l2", prep_kwargs=LOG), xgb_baseline(t), w=0.7)


PLAN = {
    "cad": {
        "baseline_xgb": lambda t: (lambda: make_baseline(t)),
        "features_lr_engcore": FEAT["features_lr_engcore"],
        "gbm_lgb_et_d1": lambda t: (lambda: make_pick("cad")),
        "linear_cl1_log": lambda t: (lambda: CalibratedLinearCV("l1", prep_kwargs=LOG)),
    },
    "lad": {
        "baseline_xgb": lambda t: (lambda: make_baseline(t)),
        "features_blend_lrcal_xgb": FEAT["features_blend_lrcal_xgb"],
        "features_blend_lr_xgb": FEAT["features_blend_lr_xgb"],
        "features_xgb_eng": FEAT["features_xgb_eng"],
        "linear_blendl2_w70": linear_blendl2_w70,
    },
    "lcx": {
        "baseline_xgb": lambda t: (lambda: make_baseline(t)),
        "features_blend_lrcal_xgb": FEAT["features_blend_lrcal_xgb"],
        "features_blend_lr_xgb": FEAT["features_blend_lr_xgb"],
        "features_xgb_eng": FEAT["features_xgb_eng"],
        "ensemble_vote_l1x": ENS["lcx_l1x"],
    },
    "rca": {
        "baseline_xgb": lambda t: (lambda: make_baseline(t)),
        "features_lr_eng_l1": FEAT["features_lr_eng_l1"],
        "features_lr_engcore": FEAT["features_lr_engcore"],
        "ensemble_vote_l1f": ENS["rca_l1f"],
        "gbm_lgb_et_d2_reg": lambda t: (lambda: make_pick("rca")),
    },
}

if __name__ == "__main__":
    targets = sys.argv[1:] or ["cad", "lad", "lcx", "rca"]
    for t in targets:
        for name, factory in PLAN[t].items():
            evaluate(f"final_{name}", t, factory(t), X, Y)
