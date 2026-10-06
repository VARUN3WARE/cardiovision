"""Step 5: honest nested tuned-threshold F1/accuracy (harness.tuned_threshold_metrics).

usage: python exp_ensemble_thresh.py <config|base> [targets...]
The F1-optimal threshold is chosen from inner-CV OOF predictions inside each outer training
fold (ThresholdTuned), so the reported F1/accuracy are not optimistic.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time
import warnings

from harness import RESULTS_DIR, tuned_threshold_metrics
from exp_ensemble_common import member_factory
from exp_ensemble_run import CONFIGS, X, Y

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cfg = sys.argv[1]
    targets = sys.argv[2:] or ["cad", "lad", "lcx", "rca"]
    for t in targets:
        t0 = time.time()
        if cfg == "base":
            make = member_factory("xgb", X.columns, Y[t])
            name = "baseline_xgb"
        else:
            make = CONFIGS[cfg](t)
            name = f"ensemble_{cfg}"
        res = tuned_threshold_metrics(name, t, make, X, Y)
        out = {k: round(v["mean"], 4) for k, v in res.items()}
        out.update({f"{k}_std": round(v["std"], 4) for k, v in res.items()})
        with open(RESULTS_DIR / f"ensemble_thresh_{cfg}__{t}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"[thresh {cfg}] {t}: F1 {out['f1']:.3f}±{out['f1_std']:.3f}  Acc {out['accuracy']:.3f}  "
              f"Prec {out['precision']:.3f}  Rec {out['recall']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
