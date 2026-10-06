"""Same member sets as exp_multitask_hetero2.py but with the equal-weight meta-learner
(average of member logits + 1-D Platt fitted on inner-OOF). Names: multitask_h3avg_<set>.
Controls: h3avg_XL (no CAD member) for h3avg_XLc / h3avg_XLcX; h2_L_indep for h3avg_Lc.

usage: python exp_multitask_hetero3.py [targets...]
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
            "XLc": [f"{t}:xgb", f"{t}:lrlog", "cad:lrlog"],
            "XLcX": [f"{t}:xgb", f"{t}:lrlog", "cad:lrlog", "cad:xgb"]}


if __name__ == "__main__":
    t0 = time.time()
    for t in sys.argv[1:] or C.TASKS:
        C._CACHE.clear()
        for name, mem in runs(t).items():
            evaluate(f"multitask_h3avg_{name}", t,
                     lambda t=t, mem=tuple(mem): C.MultiTask(target=t, mode="stack", members=mem,
                                                             meta="avg"),
                     C.X_ALL, C.Y_DICT)
        print(f"--- {t} done {time.time() - t0:.0f}s", flush=True)
