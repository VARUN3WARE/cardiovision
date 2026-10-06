"""Verify additivity of the claimed attribution method for the CLAIMED configs
(LCX uses features_blend_lrcal_xgb here, unlike exp_features_attrib.RECOMMENDED), both
in-sample and on held-out rows (fit on 80%, explain the other 20%)."""
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from exp_features_attrib import contributions, to_raw
from exp_features_run import CONFIGS
from harness import load_xy

warnings.filterwarnings("ignore")
CLAIMED = {"cad": "features_lr_engcore", "lad": "features_blend_lrcal_xgb",
           "lcx": "features_blend_lrcal_xgb", "rca": "features_lr_eng_l1"}

if __name__ == "__main__":
    X, Y, _ = load_xy()
    for t, name in CLAIMED.items():
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=Y[t], random_state=7)
        for label, fit_idx, ex_idx in [("full", np.arange(len(X)), np.arange(len(X))), ("heldout", tr, te)]:
            m = CONFIGS[name](t)().fit(X.iloc[fit_idx], Y[t][fit_idx])
            Xe = X.iloc[ex_idx]
            C, base = contributions(m, Xe)
            p = np.clip(m.predict_proba(Xe)[:, 1], 1e-12, 1 - 1e-12)
            err = np.abs(base + C.sum(axis=1).to_numpy() - np.log(p / (1 - p))).max()
            R = to_raw(C, X.columns)
            err_raw = np.abs(R.sum(axis=1).to_numpy() - C.sum(axis=1).to_numpy()).max()
            nz = int((C.abs().mean() > 1e-9).sum())
            print(f"[{t}] {name:26} {label:8} max|base+sum-logit|={err:.2e} rollup_err={err_raw:.2e} "
                  f"nonzero_feats={nz}/{C.shape[1]} p_range=({p.min():.3f},{p.max():.3f})", flush=True)
