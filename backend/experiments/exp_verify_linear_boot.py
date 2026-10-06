"""Patient-level bootstrap of the paired, repeat-averaged AUC delta (claimed vs baseline).
Repeats re-use the same 303 patients, so per-repeat signs overstate certainty; this resamples patients."""
import sys
import numpy as np
from scipy.stats import rankdata
from harness import RESULTS_DIR, load_xy

_, Y, _ = load_xy()
PAIRS = {"cad": "verify_linear_cl1_log", "lad": "verify_linear_blendl2_w70",
         "lcx": "verify_linear_blendl1_w50", "rca": "verify_linear_cl2_logord"}

def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

rng = np.random.default_rng(0)
for t, name in PAIRS.items():
    y = Y[t]
    for off in [0, 100, 200]:
        sfx = f"__seed{off}" if off else ""
        A = np.load(RESULTS_DIR / f"{name}__{t}{sfx}.oof.npy"); B = np.load(RESULTS_DIR / f"verify_linear_baseline__{t}{sfx}.oof.npy")
        point = np.mean([auc(y, A[r]) - auc(y, B[r]) for r in range(5)])
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            bs.append(np.mean([auc(y[i], A[r, i]) - auc(y[i], B[r, i]) for r in range(5)]))
        bs = np.array(bs)
        print(f"{t} seed{off}: delta {point:+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  P(delta<=0)={np.mean(bs<=0):.3f}", flush=True)
