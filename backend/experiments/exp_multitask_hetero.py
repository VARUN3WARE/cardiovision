"""Heterogeneous multi-task stacks: own-target XGB + LR members, plus CAD / other-vessel
members, combined by a logistic meta-learner on inner-OOF logits (refit per outer fold).

  het_own_*   : [own:xgb, own:lr]                         (control: base diversity only)
  het_cad_*   : [own:xgb, own:lr, cad:xgb, cad:lr]        (vessel targets)
  het_cadlr_* : [own:xgb, own:lr, cad:lr]                 (vessel targets)
  het_veslr_* : [cad:xgb, cad:lr, lad:lr, lcx:lr, rca:lr]  (CAD target)
  suffix _lr = unconstrained LogisticRegression(C=1) meta; _nn = non-negative weights.

usage: python exp_multitask_hetero.py [targets...]
"""
import sys
import time

import exp_multitask_common as C
from harness import evaluate


def member_sets(t):
    if t == "cad":
        return {"own": ["cad:xgb", "cad:lr"],
                "veslr": ["cad:xgb", "cad:lr", "lad:lr", "lcx:lr", "rca:lr"]}
    return {"own": [f"{t}:xgb", f"{t}:lr"],
            "cad": [f"{t}:xgb", f"{t}:lr", "cad:xgb", "cad:lr"],
            "cadlr": [f"{t}:xgb", f"{t}:lr", "cad:lr"]}


if __name__ == "__main__":
    t0 = time.time()
    for t in sys.argv[1:] or C.TASKS:
        C._CACHE.clear()
        for setname, mem in member_sets(t).items():
            for meta in ("lr", "nn"):
                evaluate(f"multitask_het_{setname}_{meta}", t,
                         lambda t=t, mem=tuple(mem), meta=meta:
                         C.MultiTask(target=t, mode="stack", members=mem, meta=meta),
                         C.X_ALL, C.Y_DICT)
        print(f"--- {t} done {time.time() - t0:.0f}s", flush=True)
