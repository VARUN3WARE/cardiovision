"""Run feature-engineering / feature-selection experiments through the shared harness.

usage: python exp_features_run.py [config ...] [--targets cad,lad]   (no configs = all configs)
Every config is a factory make_model_for(target) -> () -> unfitted estimator whose first
step is the stateless ClinicalFeatures transformer applied to the 55 raw encoded columns.
"""
import sys
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from exp_features_lib import (
    ClinicalFeatures, InnerTunedSelect, LogitBlend, make_lr, make_lr_l1, make_xgb,
)
from harness import evaluate_all, load_xy

warnings.filterwarnings("ignore")
_, Y, _ = load_xy()
MI_GRID = (5, 10, 15, 20, 30, 45, None)
L1_GRID = (0.02, 0.05, 0.1, 0.2, 0.5)


def spw(t):
    pos = Y[t].sum()
    return (len(Y[t]) - pos) / pos


def fe(fix=True, eng=True, log=False, subset=None):
    return ("fe", ClinicalFeatures(fix_encodings=fix, engineered=eng, log_labs=log, subset=subset))


def pipe(*steps):
    return Pipeline(list(steps))


def lr_cal(subset):
    return CalibratedClassifierCV(pipe(fe(True, True, True, subset), ("m", make_lr(scoring="roc_auc"))),
                                  method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  ensemble=False)


CORE_ENG = ["TG/HDL", "AIP log(TG/HDL)", "Risk factor count", "ECG abnormality count",
            "Chest pain class", "DF pretest prob", "RWMA present", "Age x Male"]

CONFIGS = {
    # ---- logistic regression (L2, C tuned by inner CV) ------------------------------
    "features_lr_raw":      lambda t: lambda: pipe(fe(False, False, False), ("m", make_lr())),
    "features_lr_fixed":    lambda t: lambda: pipe(fe(True, False, False), ("m", make_lr())),
    "features_lr_fixedlog": lambda t: lambda: pipe(fe(True, False, True), ("m", make_lr())),
    "features_lr_eng":      lambda t: lambda: pipe(fe(True, True, True), ("m", make_lr())),
    "features_lr_eng_aucC": lambda t: lambda: pipe(fe(True, True, True), ("m", make_lr(scoring="roc_auc"))),
    "features_lr_engcore":  lambda t: lambda: pipe(fe(True, True, True, CORE_ENG), ("m", make_lr())),
    # embedded L1 selection (C tuned by inner CV)
    "features_lr_eng_l1":   lambda t: lambda: pipe(fe(True, True, True), ("m", make_lr_l1())),
    # explicit selection with strength tuned in inner CV, then L2 LR (fixed C=0.1)
    "features_lr_eng_l1sel": lambda t: lambda: pipe(fe(True, True, True),
                                                    ("m", InnerTunedSelect(make_lr(C=0.1), "l1", L1_GRID))),
    "features_lr_eng_mik":  lambda t: lambda: pipe(fe(True, True, True),
                                                   ("m", InnerTunedSelect(make_lr(C=0.1), "mi", MI_GRID))),
    # ---- baseline XGBoost hyper-parameters, different feature sets ---------------
    "features_xgb_fixed":   lambda t: lambda: pipe(fe(True, False, False), ("m", make_xgb(spw(t)))),
    "features_xgb_eng":     lambda t: lambda: pipe(fe(True, True, False), ("m", make_xgb(spw(t)))),
    "features_xgb_engcore": lambda t: lambda: pipe(fe(True, True, False, CORE_ENG), ("m", make_xgb(spw(t)))),
    "features_xgb_eng_l1sel": lambda t: lambda: pipe(fe(True, True, False),
                                                     ("m", InnerTunedSelect(make_xgb(spw(t)), "l1", L1_GRID))),
    "features_xgb_eng_mik": lambda t: lambda: pipe(fe(True, True, False),
                                                   ("m", InnerTunedSelect(make_xgb(spw(t)), "mi", MI_GRID))),
    # ---- AUC-selected (strong) shrinkage + Platt recalibration (single linear model,
    #      ensemble=False -> calibrated logit = a*(w.x+b)+c, so attributions stay exact) ---
    "features_lr_engcore_cal": lambda t: lambda: lr_cal(CORE_ENG),
    "features_lr_eng_cal": lambda t: lambda: lr_cal(None),
    # ---- log-odds blends of engineered LR and engineered XGB -----------------------
    "features_blend_lr_xgb": lambda t: lambda: LogitBlend([
        pipe(fe(True, True, True, CORE_ENG), ("m", make_lr())),
        pipe(fe(True, True, False), ("m", make_xgb(spw(t))))]),
    "features_blend_lrcal_xgb": lambda t: lambda: LogitBlend([
        lr_cal(CORE_ENG),
        pipe(fe(True, True, False), ("m", make_xgb(spw(t))))]),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    targets = None
    if "--targets" in args:  # e.g. --targets lad,lcx
        i = args.index("--targets")
        targets = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    for n in args or list(CONFIGS):
        evaluate_all(n, CONFIGS[n], targets=targets)
