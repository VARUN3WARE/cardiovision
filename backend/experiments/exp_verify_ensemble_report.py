"""Report for the adversarial verification of the "ensemble" claims (reads saved OOF arrays)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from harness import RESULTS_DIR, load_xy
from exp_ensemble_common import calibration_stats

_, Y, _ = load_xy()
CLAIMS = [("cad", "cad_l1c", "ensemble_vote_l1c"), ("lad", "lad_plattxrf", "ensemble_platt_xrf"),
          ("lad", "lad_xrf", "ensemble_vote_xrf"), ("lcx", "lcx_l1x", "ensemble_vote_l1x"),
          ("rca", "rca_l1f", "ensemble_vote_l1f")] + [(t, "univ_l1xf", "ensemble_vote_l1xf")
                                                       for t in ["cad", "lad", "lcx", "rca"]]


def load(name, t, s):
    return np.load(RESULTS_DIR / f"{name}__{t}{'__seed%d' % s if s else ''}.oof.npy")


def aucs(y, P):
    return np.array([roc_auc_score(y, p) for p in P])


for t, cfg, theirs in CLAIMS:
    y = Y[t]
    mine0 = load(f"verify_ensemble_{cfg}", t, 0)
    th0 = load(theirs, t, 0)
    print(f"\n##### {t.upper()}  {cfg}  (claimed as {theirs})")
    print(f"  seed0 OOF identical to theirs: {np.allclose(mine0, th0, atol=1e-6)}  "
          f"max|diff|={np.abs(mine0 - th0).max():.2e}")
    pooled = []
    for s in [0, 100, 200, 300]:
        A = load(f"verify_ensemble_{cfg}", t, s)
        Bv = load("verify_ensemble_base", t, s)
        if s in (0, 100):
            Bo = load("baseline_xgb", t, s)
            assert np.allclose(Bv, Bo, atol=1e-6), "my baseline != baseline_xgb"
        a, b = aucs(y, A), aucs(y, Bv)
        d = a - b
        if s:
            pooled.extend(d)
        cs = calibration_stats(y, A)
        bri = np.mean([np.mean((p - y) ** 2) for p in A])
        brb = np.mean([np.mean((p - y) ** 2) for p in Bv])
        print(f"  seed{s:<3} AUC {a.mean():.4f}±{a.std():.4f} (base {b.mean():.4f})  "
              f"delta mean {d.mean():+.4f} std {d.std():.4f} min {d.min():+.4f}  "
              f"per-rep {np.round(d, 4).tolist()}  Brier {bri:.4f} vs {brb:.4f}  "
              f"slope {cs['cal_slope']:.2f} ECE {cs['ece']:.3f}")
    pooled = np.array(pooled)
    print(f"  UNSEEN pooled (seeds 100/200/300, 15 repeats): mean {pooled.mean():+.4f} "
          f"min {pooled.min():+.4f}  positive {int((pooled > 0).sum())}/15  "
          f"t={pooled.mean() / (pooled.std(ddof=1) / np.sqrt(len(pooled))):.1f}")
