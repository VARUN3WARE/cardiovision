"""Report for the final production recipe (experiment final_pick) vs baseline_xgb on every split
set: 0 (original), 100/200/300 (verifier sets), 400-800 (fresh, pre-declared head-to-head).
Includes a patient-level bootstrap CI of the paired AUC delta (resampling patients, AUC averaged
over the 5 repeats of a split set) on the seed-0 and pooled fresh sets."""
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from harness import RESULTS_DIR, load_xy

warnings.filterwarnings("ignore")
_, Y, _ = load_xy()
BASE = {0: "baseline_xgb", 100: "baseline_xgb", 200: "verify_features_baseline_xgb", 300: "verify_ensemble_base",
        400: "final_baseline_xgb", 500: "final_baseline_xgb", 600: "final_baseline_xgb",
        700: "final_baseline_xgb", 800: "final_baseline_xgb"}


def L(n, t, s):
    return np.load(RESULTS_DIR / f"{n}__{t}{'__seed%d' % s if s else ''}.oof.npy")


def met(y, P, fn):
    return np.array([fn(y, np.clip(p, 1e-6, 1 - 1e-6)) for p in P])


def boot(y, A, B, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(np.mean([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for a, b in zip(A, B)]))
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), float(np.mean(out <= 0))


for t in ["cad", "lad", "lcx", "rca"]:
    y = Y[t]
    print(f"\n==== {t.upper()} ====")
    print(f"{'seed':>4} {'AUC':>6} {'base':>6} {'dAUC':>7} {'dmin':>7} {'PRAUC':>6} {'bPRAUC':>6} {'Brier':>6} "
          f"{'bBrier':>6} {'LL':>6} {'bLL':>6} {'F1@.5':>6} {'Acc@.5':>6}")
    fresh_d, fresh_A, fresh_B = [], [], []
    for s, bn in BASE.items():
        try:
            A, B = L("final_pick", t, s), L(bn, t, s)
        except FileNotFoundError:
            continue
        a, b = met(y, A, roc_auc_score), met(y, B, roc_auc_score)
        f1 = np.mean([((p >= .5) & (y == 1)).sum() * 2 / ((p >= .5).sum() + y.sum()) for p in A])
        acc = np.mean([((p >= .5) == y).mean() for p in A])
        print(f"{s:>4} {a.mean():6.3f} {b.mean():6.3f} {(a - b).mean():+7.4f} {(a - b).min():+7.4f} "
              f"{met(y, A, average_precision_score).mean():6.3f} {met(y, B, average_precision_score).mean():6.3f} "
              f"{met(y, A, brier_score_loss).mean():6.3f} {met(y, B, brier_score_loss).mean():6.3f} "
              f"{met(y, A, log_loss).mean():6.3f} {met(y, B, log_loss).mean():6.3f} {f1:6.3f} {acc:6.3f}")
        if s == 0:
            lo, hi, p0 = boot(y, A, B)
            print(f"      seed-0 patient bootstrap 95% CI of dAUC [{lo:+.4f}, {hi:+.4f}]  P(d<=0)={p0:.3f}")
        if s >= 100:
            fresh_d.extend(a - b)
        if s >= 400:
            fresh_A.append(A); fresh_B.append(B)
    d = np.array(fresh_d)
    print(f"  all unseen split sets (100..800): {len(d)} repeats, dAUC mean {d.mean():+.4f} sd {d.std():.4f} "
          f"min {d.min():+.4f} frac>0 {np.mean(d > 0):.2f}")
    if fresh_A:
        lo, hi, p0 = boot(y, np.concatenate(fresh_A), np.concatenate(fresh_B), n=500)
        print(f"  fresh 400-800 patient bootstrap 95% CI of dAUC [{lo:+.4f}, {hi:+.4f}]  P(d<=0)={p0:.3f}")
