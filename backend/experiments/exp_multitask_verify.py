"""Re-run selected multitask winners + their single-task controls on unseen splits.

usage: HARNESS_SEED_OFFSET=100 python exp_multitask_verify.py <target> [<target> ...]
Experiment names are identical to the seed-0 runs; the harness appends __seed100.
"""
import os
import sys

import exp_multitask_common as C
from harness import evaluate

assert int(os.environ.get("HARNESS_SEED_OFFSET", "0")) != 0, "set HARNESS_SEED_OFFSET"

V = {  # target -> [(experiment name, mode, members, base)]
    "cad": [("multitask_indep_lr", "indep", None, "lr"),
            ("multitask_h2_L_indep", "indep", None, "lrlog"),
            ("multitask_h3avg_XL", "stack", ("cad:xgb", "cad:lrlog"), None),
            ("multitask_h3avg_XLv", "stack", ("cad:xgb", "cad:lrlog", "lad:lrlog", "lcx:lrlog", "rca:lrlog"), None),
            ("multitask_h2_XLv", "stack", ("cad:xgb", "cad:lrlog", "lad:lrlog", "lcx:lrlog", "rca:lrlog"), None)],
    "lad": [("multitask_h2_L_indep", "indep", None, "lrlog"),
            ("multitask_h2_XL", "stack", ("lad:xgb", "lad:lrlog"), None),
            ("multitask_h2_XLc", "stack", ("lad:xgb", "lad:lrlog", "cad:lrlog"), None),
            ("multitask_h3avg_XL", "stack", ("lad:xgb", "lad:lrlog"), None),
            ("multitask_h3avg_XLc", "stack", ("lad:xgb", "lad:lrlog", "cad:lrlog"), None),
            ("multitask_h3avg_XLcX", "stack", ("lad:xgb", "lad:lrlog", "cad:lrlog", "cad:xgb"), None)],
    "lcx": [("multitask_h3avg_XL", "stack", ("lcx:xgb", "lcx:lrlog"), None),
            ("multitask_h3avg_XLc", "stack", ("lcx:xgb", "lcx:lrlog", "cad:lrlog"), None),
            ("multitask_h2_XLc", "stack", ("lcx:xgb", "lcx:lrlog", "cad:lrlog"), None)],
    "rca": [("multitask_h2_L_indep", "indep", None, "lrlog"),
            ("multitask_h3avg_Lc", "stack", ("rca:lrlog", "cad:lrlog"), None),
            ("multitask_h2_Lc", "stack", ("rca:lrlog", "cad:lrlog"), None)],
}

if __name__ == "__main__":
    only = os.environ.get("ONLY")
    for t in sys.argv[1:]:
        C._CACHE.clear()
        for name, mode, mem, base in V[t]:
            if only and name not in only.split(","):
                continue
            meta = ("nn" if name.startswith("multitask_h2") else
                    "avg" if name.startswith("multitask_h3avg") else "lr")
            kw = dict(target=t, mode=mode, meta=meta)
            if mem:
                kw["members"] = mem
            if base:
                kw["base"] = base
            evaluate(name, t, lambda kw=kw: C.MultiTask(**kw), C.X_ALL, C.Y_DICT)
