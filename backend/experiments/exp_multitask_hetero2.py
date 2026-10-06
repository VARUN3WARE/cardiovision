"""Multi-task stacks built on the stronger linear member `lrlog` (log1p skewed labs,
one-hot BBB/VHD, L2 logistic regression with nested-CV C) and the baseline XGB member.
Non-negative logistic meta-learner on inner-OOF member logits (refit per outer fold).

vessel v:  h2_L_indep  = lrlog alone (control)
           h2_Lc       = [v:lrlog, cad:lrlog]                 (+CAD signal)
           h2_XL       = [v:xgb, v:lrlog]                     (control, base diversity only)
           h2_XLc      = [v:xgb, v:lrlog, cad:lrlog]          (+CAD signal)
CAD:       h2_L_indep, h2_XL = [cad:xgb, cad:lrlog],
           h2_Lv  = [cad:lrlog, lad:lrlog, lcx:lrlog, rca:lrlog],
           h2_XLv = [cad:xgb, cad:lrlog, lad:lrlog, lcx:lrlog, rca:lrlog]

usage: python exp_multitask_hetero2.py [targets...]
"""
import sys
import time

import exp_multitask_common as C
from harness import evaluate


def runs(t):
    if t == "cad":
        return {"XL": ["cad:xgb", "cad:lrlog"],
                "Lv": ["cad:lrlog", "lad:lrlog", "lcx:lrlog", "rca:lrlog"],
                "XLv": ["cad:xgb", "cad:lrlog", "lad:lrlog", "lcx:lrlog", "rca:lrlog"]}
    return {"Lc": [f"{t}:lrlog", "cad:lrlog"],
            "XL": [f"{t}:xgb", f"{t}:lrlog"],
            "XLc": [f"{t}:xgb", f"{t}:lrlog", "cad:lrlog"]}


if __name__ == "__main__":
    t0 = time.time()
    for t in sys.argv[1:] or C.TASKS:
        C._CACHE.clear()
        evaluate("multitask_h2_L_indep", t,
                 lambda t=t: C.MultiTask(target=t, mode="indep", base="lrlog"), C.X_ALL, C.Y_DICT)
        for name, mem in runs(t).items():
            evaluate(f"multitask_h2_{name}", t,
                     lambda t=t, mem=tuple(mem): C.MultiTask(target=t, mode="stack", members=mem,
                                                             meta="nn"),
                     C.X_ALL, C.Y_DICT)
        print(f"--- {t} done {time.time() - t0:.0f}s", flush=True)
