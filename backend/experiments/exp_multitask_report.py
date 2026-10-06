"""Summarise multitask_* experiments: AUC, paired delta vs baseline_xgb, and paired delta
vs the matching single-task control (isolates the multi-task contribution).

usage: python exp_multitask_report.py [seed_offset]
"""
import json
import sys

from harness import RESULTS_DIR, paired_auc_delta

OFF = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SFX = f"__seed{OFF}" if OFF else ""

CONTROL = {  # experiment -> its single-task control
    "multitask_hier_{b}": "multitask_indep_{b}", "multitask_chain_cad_{b}": "multitask_indep_{b}",
    "multitask_chain_all_{b}": "multitask_indep_{b}", "multitask_stack_cad_{b}": "multitask_indep_{b}",
    "multitask_stack_all_{b}": "multitask_indep_{b}", "multitask_stack_soft_hier_{b}": "multitask_indep_{b}",
    "multitask_stack_hier_plus_{b}": "multitask_indep_{b}",
}


def control_of(name):
    for b in ("xgb", "lr", "xgbl"):
        for k, v in CONTROL.items():
            if name == k.format(b=b):
                return v.format(b=b)
    if name.startswith("multitask_het_") and "_own_" not in name:
        return name.replace("_cadlr_", "_own_").replace("_cad_", "_own_").replace("_veslr_", "_own_")
    if name in ("multitask_h2_Lc",):
        return "multitask_h2_L_indep"
    if name in ("multitask_h2_XLc", "multitask_h2_XLv"):
        return "multitask_h2_XL"
    if name in ("multitask_h2_Lv", "multitask_h3avg_Lc", "multitask_h3avg_Lv"):
        return "multitask_h2_L_indep"
    if name in ("multitask_h3avg_XLc", "multitask_h3avg_XLcX", "multitask_h3avg_XLv"):
        return "multitask_h3avg_XL"
    return None


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob(f"multitask_*{SFX}.json")):
        stem = f.name[:-5]
        if SFX:
            stem = stem[: -len(SFX)]
        elif "__seed" in stem:
            continue
        name, target = stem.rsplit("__", 1)
        m = json.load(open(f))["metrics"]
        try:
            d = paired_auc_delta(name, "baseline_xgb", target, OFF)
        except FileNotFoundError:
            continue
        c = control_of(name)
        dc = None
        if c and (RESULTS_DIR / f"{c}__{target}{SFX}.oof.npy").exists():
            dc = paired_auc_delta(name, c, target, OFF)
        rows.append((target, m["roc_auc"]["mean"], name, m, d, c, dc))
    for t in ("cad", "lad", "lcx", "rca"):
        print(f"\n=== {t} (seed offset {OFF})")
        for target, auc, name, m, d, c, dc in sorted([r for r in rows if r[0] == t], key=lambda r: -r[1]):
            s = (f"{name:34s} AUC {auc:.3f}±{m['roc_auc']['std']:.3f} PR {m['pr_auc']['mean']:.3f} "
                 f"Brier {m['brier']['mean']:.3f} F1 {m['f1']['mean']:.3f} Acc {m['accuracy']['mean']:.3f} | "
                 f"vs base {d['mean']:+.3f} (min {d['min']:+.3f}, sd {d['std']:.3f})")
            if dc:
                s += f" | vs ctrl {dc['mean']:+.3f} (min {dc['min']:+.3f})"
            print(s)


if __name__ == "__main__":
    main()
