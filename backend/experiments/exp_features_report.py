"""Summarise features_* experiments: metrics + paired AUC delta vs a reference (default baseline_xgb).

usage: python exp_features_report.py [seed_offset] [reference]
"""
import json
import sys

from harness import RESULTS_DIR, TARGETS, paired_auc_delta

off = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ref = sys.argv[2] if len(sys.argv) > 2 else "baseline_xgb"
sfx = f"__seed{off}" if off else ""

names = sorted({p.name.split("__")[0] for p in RESULTS_DIR.glob(f"features_*{sfx}.json")
                if (off and "__seed" in p.name) or (not off and "__seed" not in p.name)})
for t in TARGETS:
    print(f"\n=== {t} (seed_offset={off}, ref={ref}) ===")
    print(f"{'experiment':28} {'AUC':>6} {'±':>5} {'PR':>6} {'Brier':>6} {'F1':>6} {'Acc':>6} "
          f"{'dMean':>7} {'dStd':>6} {'dMin':>7}")
    rows = []
    for n in names + [ref]:
        f = RESULTS_DIR / f"{n}__{t}{sfx}.json"
        if not f.exists():
            continue
        m = json.load(open(f))["metrics"]
        d = paired_auc_delta(n, ref, t, off)
        rows.append((m["roc_auc"]["mean"], n, m, d))
    for auc, n, m, d in sorted(rows, reverse=True):
        print(f"{n:28} {auc:6.3f} {m['roc_auc']['std']:5.3f} {m['pr_auc']['mean']:6.3f} "
              f"{m['brier']['mean']:6.3f} {m['f1']['mean']:6.3f} {m['accuracy']['mean']:6.3f} "
              f"{d['mean']:+7.3f} {d['std']:6.3f} {d['min']:+7.3f}")
