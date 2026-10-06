"""Nested tuned-threshold (F1-optimal, chosen inside each training fold) metrics for the
recommended features_* models and the baseline, so F1/accuracy are honest at a tuned cut-off.

usage: python exp_features_thresh.py [target:config ...]
"""
import json
import sys
import warnings

import exp_baseline
from exp_features_run import CONFIGS
from harness import RESULTS_DIR, SEED_OFFSET, tuned_threshold_metrics

warnings.filterwarnings("ignore")

DEFAULT = ["cad:features_lr_engcore", "lad:features_blend_lr_xgb", "lcx:features_blend_lr_xgb",
           "rca:features_lr_eng_l1", "cad:baseline_xgb", "lad:baseline_xgb", "lcx:baseline_xgb",
           "rca:baseline_xgb"]

if __name__ == "__main__":
    for item in sys.argv[1:] or DEFAULT:
        t, name = item.split(":")
        factory = exp_baseline.make_model_for(t) if name == "baseline_xgb" else CONFIGS[name](t)
        m = tuned_threshold_metrics(name, t, factory)
        tag = f"{name}__{t}" + (f"__seed{SEED_OFFSET}" if SEED_OFFSET else "")
        with open(RESULTS_DIR / f"{tag}.tuned_threshold.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"[tuned-thr] {t:3} {name:28} F1 {m['f1']['mean']:.3f}±{m['f1']['std']:.3f}  "
              f"Acc {m['accuracy']['mean']:.3f}  Prec {m['precision']['mean']:.3f}  "
              f"Rec {m['recall']['mean']:.3f}", flush=True)
