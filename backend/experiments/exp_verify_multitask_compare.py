"""Paired AUC deltas of verify_multitask_* configs vs baseline (seed 0/100: baseline_xgb;
seed 200: verify_multitask_baseline re-run on the same fresh splits) and vs controls."""
import sys
from harness import paired_auc_delta

PAIRS = {"cad": [("cad_L", None)], "lad": [("lad_XLcX", "lad_XL")],
         "lcx": [("lcx_XLc", "lcx_XL")], "rca": [("rca_Lc", "rca_L")]}
for t in sys.argv[1:] or PAIRS:
    for cfg, ctrl in PAIRS[t]:
        for off in (0, 100, 200):
            base = "verify_multitask_baseline" if off == 200 else "baseline_xgb"
            try:
                d = paired_auc_delta(f"verify_multitask_{cfg}", base, t, off)
                s = f"{t} {cfg:9s} seed{off:3d} vs base {d['mean']:+.4f} sd {d['std']:.4f} min {d['min']:+.4f} {d['per_repeat']}"
                if ctrl:
                    c = paired_auc_delta(f"verify_multitask_{cfg}", f"verify_multitask_{ctrl}", t, off)
                    s += f" | vs {ctrl} {c['mean']:+.4f} min {c['min']:+.4f}"
                print(s)
            except FileNotFoundError as e:
                print(t, cfg, off, "missing", e.filename.split('/')[-1])
