"""Structural multi-task experiments: independent vs hierarchical vs chain vs stacked.

usage: python exp_multitask_struct.py <base> [targets...]
  base in {xgb, xgbl, lr}; experiment names are multitask_<mode>_<base>.
"""
import sys
import time

import exp_multitask_common as C
from harness import evaluate

base = sys.argv[1]
targets = sys.argv[2:] or C.TASKS
modes_only = None  # set via env if needed

t0 = time.time()
for target in targets:
    C._CACHE.clear()
    runs = [("indep", {})]
    if target == "cad":
        runs += [("chain_all", {})]
    else:
        runs += [("hier", {}), ("chain_cad", {}), ("chain_all", {})]
    runs += [(name, {"members": tuple(m)}) for name, m in C.stack_variants(target, base).items()]
    for mode, kw in runs:
        est_mode = "stack" if mode.startswith("stack") else mode

        def make(target=target, est_mode=est_mode, kw=kw):
            return C.MultiTask(target=target, mode=est_mode, base=base, **kw)

        evaluate(f"multitask_{mode}_{base}", target, make, C.X_ALL, C.Y_DICT)
    print(f"--- {target} done, elapsed {time.time() - t0:.0f}s", flush=True)
