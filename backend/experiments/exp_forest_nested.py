"""forest / stage 3: HONEST nested selection of the forest config + Platt calibration.

usage: python exp_forest_nested.py <auc|logloss> [n_estimators] [inner_splits]

For each outer training fold, SearchCalibratedForest evaluates a small RF/ET grid with an
inner stratified CV, picks the best config by inner AUC (or Platt-calibrated log-loss),
fits the sigmoid on that config's inner OOF probabilities and refits on the whole fold.
Nothing about the outer test fold is seen, so the numbers are unbiased for the
"search + calibrate" procedure.  Also logs which configs get picked.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys

from exp_forest_common import SearchCalibratedForest
from harness import RESULTS_DIR, TARGETS, evaluate, load_xy, paired_auc_delta

SCORING = sys.argv[1] if len(sys.argv) > 1 else "auc"
N_TREES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
INNER = int(sys.argv[3]) if len(sys.argv) > 3 else 3

GRID = [dict(algo=a, max_features=mf, min_samples_leaf=msl, class_weight="balanced_subsample")
        for a, mf, msl in itertools.product(["rf", "et"], ["sqrt", 0.33], [1, 3, 8])]

PICKS: dict = collections.defaultdict(collections.Counter)


class Logged(SearchCalibratedForest):
    """Records the chosen config per outer fit (for the report)."""

    def fit(self, X, y):
        super().fit(X, y)
        PICKS[self._target][str(self.best_params_)] += 1
        return self


def main():
    X, Y, _ = load_xy()
    name = f"forest_nested_{SCORING}_sig"
    summary = {}
    for t in TARGETS:
        def make(t=t):
            m = Logged(GRID, inner_splits=INNER, scoring=SCORING, calibration="sigmoid",
                       n_estimators=N_TREES, random_state=0, n_jobs=2)
            m._target = t
            return m
        r = evaluate(name, t, make, X, Y)
        d = paired_auc_delta(name, "baseline_xgb", t)
        summary[t] = {"metrics": {k: v["mean"] for k, v in r["metrics"].items()},
                      "auc_std": r["metrics"]["roc_auc"]["std"], "paired_vs_baseline": d,
                      "picks": PICKS[t].most_common()}
        print(f"   paired vs baseline_xgb: mean {d['mean']:+.4f} std {d['std']:.4f} "
              f"min {d['min']:+.4f}  per-repeat {d['per_repeat']}")
        print("   picks:", PICKS[t].most_common(4), flush=True)
    with open(RESULTS_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
