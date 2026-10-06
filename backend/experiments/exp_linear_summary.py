"""Tabulate every saved linear_* experiment against baseline_xgb (paired, identical folds).

    python exp_linear_summary.py [seed_offset]
"""
import json
import sys

from harness import RESULTS_DIR, paired_auc_delta

OFF = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SFX = f"__seed{OFF}" if OFF else ""

rows = []
for target in ["cad", "lad", "lcx", "rca"]:
    for f in sorted(RESULTS_DIR.glob(f"linear_*__{target}{SFX}.json")):
        name = f.name.split("__")[0]
        if not OFF and "__seed" in f.name:
            continue
        m = json.load(open(f))["metrics"]
        try:
            d = paired_auc_delta(name, "baseline_xgb", target, OFF)
        except FileNotFoundError:
            continue
        rows.append((target, name, m, d))

for target in ["cad", "lad", "lcx", "rca"]:
    print(f"\n=== {target} (seed offset {OFF}) ===")
    print(f"{'experiment':26} {'AUC':>12} {'PR-AUC':>7} {'Brier':>6} {'F1@.5':>6} {'Acc@.5':>6} "
          f"{'dAUC mean':>9} {'std':>6} {'min':>7}")
    for t, name, m, d in sorted([r for r in rows if r[0] == target],
                                key=lambda r: -r[2]["roc_auc"]["mean"]):
        print(f"{name:26} {m['roc_auc']['mean']:.3f}±{m['roc_auc']['std']:.3f} "
              f"{m['pr_auc']['mean']:7.3f} {m['brier']['mean']:6.3f} {m['f1']['mean']:6.3f} "
              f"{m['accuracy']['mean']:6.3f} {d['mean']:+9.4f} {d['std']:6.4f} {d['min']:+7.4f}")
