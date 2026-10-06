"""gbm strategy, part 2: NESTED hyper-parameter search per library.

Each outer training fold runs its own Optuna TPE study (n_trials, 3-fold inner CV, AUC
objective, n_estimators picked from staged predictions) and refits the winner on the whole
outer training fold - so the reported OOF metrics are honest for "tune-then-train".

Usage: python exp_gbm_nested.py <lib: xgb|lgb|cat|catn> <n_trials> <target> [target ...]
Experiment name: gbm_nested_<lib>
"""
import json
import sys
import time

from harness import RESULTS_DIR, evaluate, load_xy
from exp_gbm_common import OptunaGBM

lib = sys.argv[1]
n_trials = int(sys.argv[2])
targets = sys.argv[3:]
n_max = {"xgb": 600, "lgb": 600, "cat": 600, "catn": 600}[lib]

X, Y, _ = load_xy()
chosen = []


class Logged(OptunaGBM):
    def fit(self, X, y):
        super().fit(X, y)
        chosen.append({"score": round(self.best_score_, 4), **self.best_params_})
        return self


for t in targets:
    t0 = time.time()
    chosen.clear()
    evaluate(f"gbm_nested_{lib}", t,
             lambda: Logged(lib=lib, n_trials=n_trials, inner_splits=3, n_max=n_max, step=50), X, Y)
    with open(RESULTS_DIR / f"gbm_nested_{lib}__{t}.params.json", "w") as f:
        json.dump(chosen, f, indent=1, default=float)
    print(f"== {lib} {t} {time.time() - t0:.0f}s", flush=True)
