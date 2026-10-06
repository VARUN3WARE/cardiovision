"""Nested tuned-threshold F1/accuracy for the claimed CAD/RCA models vs baseline (prints only)."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from harness import SEED_OFFSET, load_xy, tuned_threshold_metrics
from exp_verify_linear_run import CLAIMED
from exp_linear_blend import xgb_baseline
X, Y, _ = load_xy()
for t in sys.argv[1:]:
    name, make = CLAIMED[t]
    for label, mk in [(name, make), ("baseline", lambda t=t: xgb_baseline(t))]:
        r = tuned_threshold_metrics(label, t, mk, X, Y)
        print(f"seed{SEED_OFFSET} {t} {label}: " + "  ".join(f"{k} {v['mean']:.3f}" for k, v in r.items()), flush=True)
