"""Run regularised linear / kernel configurations through the shared harness.

    python exp_linear_run.py <config> [<config> ...]      (see CONFIGS below)

Every config is a zero-arg factory returning a fresh GridSearchCV(Pipeline(prep, clf)),
so the inner hyper-parameter search, scaling and one-hot encoding are refit per outer
training fold by the harness.
"""
import sys
import time

from harness import evaluate_all
import numpy as np

from exp_linear_common import CalibratedLinearCV, lr_search, svm_search

LOG = dict(log_skewed=True)
LOG_ORD = dict(log_skewed=True, vhd="ordinal")

CONFIGS = {
    # L2 logistic regression
    "linear_l2": lambda: lr_search("l2"),
    "linear_l2_log": lambda: lr_search("l2", prep_kwargs=LOG),
    "linear_l2_logord": lambda: lr_search("l2", prep_kwargs=LOG_ORD),
    "linear_l2_log_none": lambda: lr_search("l2", class_weight=None, prep_kwargs=LOG),
    "linear_l2_log_bal": lambda: lr_search("l2", class_weight="balanced", prep_kwargs=LOG),
    "linear_l2_log_ll": lambda: lr_search("l2", class_weight=None, prep_kwargs=LOG,
                                          scoring="neg_log_loss"),
    # L1 / elastic net
    "linear_l1_log": lambda: lr_search("l1", prep_kwargs=LOG),
    "linear_en_log": lambda: lr_search("elasticnet", prep_kwargs=LOG),
    # GAM-like: cubic splines on continuous columns + L2
    "linear_spl_log": lambda: lr_search("l2", prep_kwargs=dict(log_skewed=True, splines=True)),
    # RBF SVM, Platt-calibrated
    "linear_svm_log": lambda: svm_search(prep_kwargs=LOG),
    # Fast nested-C search + Platt recalibration (stays exactly linear)
    "linear_cl2_log": lambda: CalibratedLinearCV("l2", prep_kwargs=LOG),
    "linear_cl2_log_bal": lambda: CalibratedLinearCV("l2", class_weight="balanced", prep_kwargs=LOG),
    "linear_cl2_logord": lambda: CalibratedLinearCV("l2", prep_kwargs=LOG_ORD),
    "linear_cl2_raw": lambda: CalibratedLinearCV("l2"),
    "linear_cl1_log": lambda: CalibratedLinearCV("l1", prep_kwargs=LOG),
    "linear_cen5_log": lambda: CalibratedLinearCV("elasticnet", l1_ratio=0.5, prep_kwargs=LOG,
                                                  Cs=list(np.logspace(-3, 1, 9))),
    "linear_cl2_log_selll": lambda: CalibratedLinearCV("l2", prep_kwargs=LOG, select="log_loss"),
    "linear_lda_log": lambda: CalibratedLinearCV(clf="lda", prep_kwargs=LOG),
    "linear_cl2_spl": lambda: CalibratedLinearCV("l2", prep_kwargs=dict(log_skewed=True, splines=True)),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CONFIGS)
    for name in names:
        t0 = time.time()
        factory = CONFIGS[name]
        evaluate_all(name, lambda target, f=factory: f)
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
