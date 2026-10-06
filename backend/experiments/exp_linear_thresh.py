"""Nested F1-optimal threshold metrics (harness.tuned_threshold_metrics) for linear configs.

    python exp_linear_thresh.py <config> [targets...]

Saves results/<config>__<target>.thresh.json (and __seed<off> when HARNESS_SEED_OFFSET set).
"""
import json
import sys

from exp_linear_blend import LogitBlend, xgb_baseline
from exp_linear_common import CalibratedLinearCV
from exp_linear_run import CONFIGS
from harness import RESULTS_DIR, SEED_OFFSET, load_xy, tuned_threshold_metrics


def factory(name, target):
    if name == "baseline_xgb":  # reference point; saved as baseline_xgb__<t>.thresh.json
        return lambda: xgb_baseline(target)
    if name.startswith("linear_blend"):  # linear_blend<l1|l2>_w<pct>
        pen, w = name[len("linear_blend"):].split("_w")
        return lambda: LogitBlend(CalibratedLinearCV(pen, prep_kwargs=dict(log_skewed=True)),
                                  xgb_baseline(target), w=int(w) / 100)
    return CONFIGS[name]

if __name__ == "__main__":
    name = sys.argv[1]
    targets = sys.argv[2:] or ["cad", "lad", "lcx", "rca"]
    X, Y, _ = load_xy()
    for t in targets:
        res = tuned_threshold_metrics(name, t, factory(name, t), X, Y)
        tag = f"{name}__{t}" + (f"__seed{SEED_OFFSET}" if SEED_OFFSET else "")
        with open(RESULTS_DIR / f"{tag}.thresh.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"[{name}] {t:3} tuned-threshold  " + "  ".join(
            f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in res.items()), flush=True)
