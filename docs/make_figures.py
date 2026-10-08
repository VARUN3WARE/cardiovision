"""Figures for the technical documentation (docs/technical-documentation.md).

Run from the repo root after `python backend/train.py`:
    backend/.venv/bin/python docs/make_figures.py

Writes docs/figures/evaluation.png (ROC + calibration from the out-of-fold predictions saved by
train.py) and docs/figures/architecture.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.config import TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE  # noqa: E402
from app.encoders import binarize_target  # noqa: E402
from train import load_dataset  # noqa: E402

OUT = ROOT / "docs" / "figures"
# Validated categorical palette (dataviz reference, light mode): slots 1-4 in fixed order.
COLORS = {"cad": "#2a78d6", "lad": "#eb6834", "lcx": "#1baf7a", "rca": "#eda100"}
NAMES = {"cad": "CAD", "lad": "LAD", "lcx": "LCX", "rca": "RCA"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, 
})


def evaluation_figure():
    df = load_dataset()
    oof = np.load(ROOT / "backend" / "models" / "oof_predictions.npz")
    metrics = json.loads((ROOT / "backend" / "models" / "metrics.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    for t in TARGETS:
        y = binarize_target(df[TARGET_COLUMN_MAP[t]], TARGET_POSITIVE[TARGET_COLUMN_MAP[t]]).to_numpy()
        P = oof[t]
        fpr, tpr, _ = roc_curve(y, P[0])  # one full repeat of out-of-fold predictions
        ax1.plot(fpr, tpr, color=COLORS[t], lw=1.6, solid_capstyle="round")
        # Calibration: pooled over the 5 repeats, 10 quantile bins.
        p, yy = P.ravel(), np.tile(y, P.shape[0])
        edges = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
        mp = np.array([p[idx == b].mean() for b in range(10)])
        ob = np.array([yy[idx == b].mean() for b in range(10)])
        ax2.plot(mp, ob, color=COLORS[t], lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8)

    # ROC panel: legend with AUC in the empty area under the diagonal (labels on curves collide).
    ax1.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)))
    roc_handles = [plt.Line2D([], [], color=COLORS[t], lw=1.8) for t in TARGETS]
    roc_labels = [f"{NAMES[t]}   AUC {metrics[t]['cv']['roc_auc']['mean']:.2f} ± {metrics[t]['cv']['roc_auc']['std']:.2f}"
                  for t in TARGETS]
    ax1.legend(roc_handles, roc_labels, loc="lower right", frameon=False, fontsize=7.2, handlelength=1.6,
               title="5x5 CV mean ± sd", title_fontsize=6.6)
    ax1.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")
    ax1.set_title("ROC (out-of-fold, one CV repeat)", fontsize=8.5, color=INK, loc="left")

    # Calibration panel
    ax2.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax2.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Mean predicted probability (decile)", ylabel="Observed rate")
    ax2.set_title("Calibration (pooled over 5 repeats)", fontsize=8.5, color=INK, loc="left")

    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], color=COLORS[t], lw=1.8, marker="o", ms=4) for t in TARGETS]
    ax2.legend(handles, [NAMES[t] for t in TARGETS], loc="lower right", frameon=False, fontsize=7.2)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "evaluation.png", dpi=300)
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set(xlim=(0, 100), ylim=(0, 36))
    ax.axis("off")

    def box(x, y, w, h, title, lines, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc="#eef4fc" if accent else "#f4f4f2", ec="#2a78d6" if accent else "#c9c8c2", lw=0.8))
        ax.text(x + 1.0, y + h - 1.3, title, fontsize=6.0, weight="bold", color=INK, va="top")
        ax.text(x + 1.0, y + h - 4.2, "\n".join(lines), fontsize=5.0, color=INK2, va="top", linespacing=1.5)

    def arrow(x1, y1, x2, y2, label=None, below=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color="#2a78d6", lw=1.0))
        if label:
            ax.text((x1 + x2) / 2, y1 + (-1.0 if below else 1.0), label, fontsize=5.0, color=INK2, ha="center",
                    va="top" if below else "bottom", linespacing=1.2)

    box(0.5, 20, 28, 15.5, "Training  (backend/train.py)", [
        "UCI Z-Alizadeh Sani: 303 x 55 inputs",
        "Clinical composites (stateless)",
        "4 blends: logistic reg. + boosted trees",
        "5x5 repeated CV, nested thresholds,",
        "calibration bands"])
    box(0.5, 0.5, 28, 15.5, "Artefacts  (backend/models/)", [
        "cad / lad / lcx / rca .joblib",
        "metrics.json (CV, operating points)",
        "feature_info.json (ranges, defaults)",
        "oof_predictions.npz (example patients)"])
    box(36, 0.5, 28, 35, "FastAPI  (backend/app/)", [
        "POST /explain, /predict",
        "  1. validate input (pydantic)",
        "  2. fill unrecorded values, encode",
        "  3. 4 models -> probability, threshold,",
        "      risk band, range warnings",
        "  4. exact attributions per model",
        "      (linear terms + TreeSHAP)",
        "GET /metrics  /features  /health",
        "",
        "Safety disclaimer in every response"], accent=True)
    box(71.5, 20, 28, 15.5, "React app  (frontend/src/)", [
        "Clinical form: quick / full mode,",
        "validation, live re-estimation",
        "Risk dashboard, calibration bands",
        "Plain-language SHAP + breakdown"])
    box(71.5, 0.5, 28, 15.5, "3D view  (React Three Fiber)", [
        "Realistic CC BY heart; coronary tree",
        "snapped into its grooves (BVH)",
        "Risk colours, territories, selection,",
        "camera focus, pins, schematic fallback"])
    arrow(14.5, 19.6, 14.5, 16.4)
    arrow(28.9, 8.0, 35.6, 8.0, "load")
    arrow(71.1, 30.0, 64.4, 30.0, "patient\nfeatures")
    arrow(64.4, 24.0, 71.1, 24.0, "estimates +\nattributions", below=True)
    arrow(85.5, 19.6, 85.5, 16.4)
    fig.savefig(OUT / "architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation_figure()
    architecture_figure()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
