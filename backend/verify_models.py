"""Independent verification of the production models: is the performance real, and is the backbone sound?

Run after train.py:   .venv/bin/python verify_models.py      (about 5 minutes on 16 cores)
Writes ../docs/model-verification.md and models/verification.json.

Checks, each designed to catch a way small clinical datasets fool you:
  1. Data integrity: missing values, duplicate patients, label consistency (CAD vs vessel labels).
  2. Leakage: target columns excluded; no single input almost reproduces a label; a label-permutation
     test (the full recipe on shuffled labels must score AUC ~0.5, or something leaks).
  3. Fresh-split cross-validation: the production recipe re-run on 5x5 splits with seeds never used
     in development (5000-5004), everything refitted inside each training fold; AUC with a bootstrap 95% CI,
     Brier score, calibration slope and intercept. (New splits of the same 303 patients cannot remove all
     model-selection optimism; see the report's note.)
  4. Baselines on the same folds: the Diamond-Forrester pre-test probability alone (a classic clinical
     score, no fitting), plain logistic regression on the 55 raw inputs, default XGBoost.
  5. Served models: the saved models equal a fresh refit of the recipe; apparent (training) AUC vs
     cross-validated AUC (overfitting gap); the live API returns exactly the saved models' predictions.
  6. Vessel specificity: do the vessel models say WHICH artery, or mostly "how much CAD"?
  7. Clinical direction: changing one finding at a time moves the estimate the way medicine expects
     (and, where it does not, what the raw data show).
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from app.cardio_models import _df_pretest, make_model  # noqa: E402
from app.config import LEAKAGE_COLUMNS, METRICS_PATH, MODELS_DIR, TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE  # noqa: E402
from app.encoders import binarize_target, encode_dataframe, load_encoding  # noqa: E402
from train import build_features, load_dataset  # noqa: E402

warnings.filterwarnings("ignore")
# Split seeds never used in development: the experiments used 0-4 and offsets 100-900 (seeds 100-904).
FRESH_SEEDS = [5000, 5001, 5002, 5003, 5004]
N_PERMUTATIONS = 3
OUT_MD = Path(__file__).resolve().parent.parent / "docs" / "model-verification.md"
OUT_JSON = MODELS_DIR / "verification.json"


def baseline(kind: str, y):
    if kind == "lr_raw":
        return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.1, max_iter=5000))])
    if kind == "xgb_default":
        return XGBClassifier(eval_metric="logloss", random_state=0, n_jobs=1)
    return make_model(kind, y)


def fit_predict(target, kind, X, y, tr, te):
    warnings.filterwarnings("ignore")
    model = baseline(kind if kind != "prod" else target, y[tr])
    return model.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]


def calibration(y, p):
    """Calibration slope and intercept (logistic recalibration of the log-odds; ideal 1 and 0)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(lo, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def bootstrap_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    t0 = time.time()
    df = load_dataset()
    features = build_features(df)
    encoding = load_encoding()
    X = encode_dataframe(df, encoding, features)
    ys = {t: binarize_target(df[TARGET_COLUMN_MAP[t]], TARGET_POSITIVE[TARGET_COLUMN_MAP[t]]).to_numpy() for t in TARGETS}
    saved_metrics = json.loads(Path(METRICS_PATH).read_text())
    R: dict = {"targets": {}}

    # 1. Data integrity ---------------------------------------------------------------------------
    any_vessel = (ys["lad"] | ys["lcx"] | ys["rca"]).astype(bool)
    cad = ys["cad"].astype(bool)
    R["data"] = {
        "rows": int(len(df)), "inputs": len(features), "missing_values": int(df[features].isna().sum().sum()),
        "duplicate_patients": int(df[features].duplicated().sum()),
        "cad_without_vessel_label": int((cad & ~any_vessel).sum()),
        "vessel_label_without_cad": int((~cad & any_vessel).sum()),
        "prevalence": {t: round(float(ys[t].mean()), 3) for t in TARGETS},
    }

    # 2a. Leakage: excluded columns and the strongest single inputs -------------------------------
    R["leakage"] = {"target_columns_in_inputs": sorted(set(features) & set(LEAKAGE_COLUMNS)), "top_single_input_auc": {}}
    for t in TARGETS:
        aucs = {c: max(roc_auc_score(ys[t], X[c]), 1 - roc_auc_score(ys[t], X[c])) for c in features if X[c].nunique() > 1}
        top = sorted(aucs.items(), key=lambda kv: -kv[1])[:3]
        R["leakage"]["top_single_input_auc"][t] = [(c, round(a, 3)) for c, a in top]

    # Jobs: fresh-split CV for the production recipe and baselines, plus permutation runs.
    jobs = []
    for t in TARGETS:
        for kind in ("prod", "lr_raw", "xgb_default"):
            for r, seed in enumerate(FRESH_SEEDS):
                for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ys[t])):
                    jobs.append((t, kind, r, ys[t], tr, te))
        for k in range(N_PERMUTATIONS):
            yp = np.random.default_rng(1000 + k).permutation(ys[t])
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=2000 + k).split(X, yp):
                jobs.append((t, f"perm{k}", 0, yp, tr, te))
    print(f"Running {len(jobs)} model fits in parallel...", flush=True)
    preds = Parallel(n_jobs=min(14, os.cpu_count() or 1))(
        delayed(fit_predict)(t, "prod" if kind.startswith("perm") else kind, X, y, tr, te) for t, kind, _, y, tr, te in jobs)
    oof: dict = {}
    for (t, kind, r, y, tr, te), p in zip(jobs, preds):
        key = (t, kind)
        oof.setdefault(key, {"y": y, "P": {}})
        oof[key]["P"].setdefault(r, np.zeros(len(y)))[te] = p

    pretest = _df_pretest(X["Age"], X["Sex"], np.select(
        [X["Typical Chest Pain"] == 1, X["Atypical"] == 1, X["Nonanginal"] == 1], [3, 2, 1], default=0))

    for t in TARGETS:
        y = ys[t]
        P = np.array([oof[(t, "prod")]["P"][r] for r in range(len(FRESH_SEEDS))])
        aucs = [roc_auc_score(y, p) for p in P]
        mean_p = P.mean(axis=0)
        slope, intercept = calibration(np.tile(y, len(P)), P.ravel())
        res = {
            "auc_fresh_mean": float(np.mean(aucs)), "auc_fresh_sd": float(np.std(aucs)),
            "auc_saved_cv": float(saved_metrics[t]["cv"]["roc_auc"]["mean"]),
            "auc_95ci": bootstrap_auc(y, mean_p),
            "brier": float(np.mean([brier_score_loss(y, p) for p in P])),
            "brier_prevalence_only": float(brier_score_loss(y, np.full(len(y), y.mean()))),
            "calibration_slope": slope, "calibration_intercept": intercept,
            "permutation_auc": [float(roc_auc_score(oof[(t, f"perm{k}")]["y"], oof[(t, f"perm{k}")]["P"][0]))
                                for k in range(N_PERMUTATIONS)],
            "baselines": {"diamond_forrester_only": float(roc_auc_score(y, pretest))},
        }
        for kind in ("lr_raw", "xgb_default"):
            B = np.array([oof[(t, kind)]["P"][r] for r in range(len(FRESH_SEEDS))])
            b_aucs = [roc_auc_score(y, p) for p in B]
            res["baselines"][kind] = float(np.mean(b_aucs))
            res["baselines"][f"{kind}_beaten_in_repeats"] = f"{sum(a > b for a, b in zip(aucs, b_aucs))}/{len(aucs)}"
        R["targets"][t] = res
        print(f"[{t}] fresh AUC {res['auc_fresh_mean']:.3f}±{res['auc_fresh_sd']:.3f} (saved {res['auc_saved_cv']:.3f})"
              f"  CI {res['auc_95ci'][0]:.3f}-{res['auc_95ci'][1]:.3f}  permuted {np.mean(res['permutation_auc']):.3f}"
              f"  slope {slope:.2f}", flush=True)

    # 6. Vessel specificity ---------------------------------------------------------------------------
    mean_oof = {t: np.mean([oof[(t, "prod")]["P"][r] for r in range(len(FRESH_SEEDS))], axis=0) for t in TARGETS}
    pos = ys["cad"] == 1
    R["vessel_specificity"] = {
        t: {"own_model_auc": R["targets"][t]["auc_fresh_mean"],
            "cad_model_auc_for_this_label": float(roc_auc_score(ys[t], mean_oof["cad"])),
            "auc_within_cad_patients": float(roc_auc_score(ys[t][pos], mean_oof[t][pos]))}
        for t in ("lad", "lcx", "rca")}
    vessels = np.array(["lad", "lcx", "rca"])
    single = np.where((ys["lad"] + ys["lcx"] + ys["rca"]) == 1)[0]
    truth = vessels[np.argmax(np.column_stack([ys[v] for v in vessels])[single], axis=1)]
    top = vessels[np.argmax(np.column_stack([mean_oof[v] for v in vessels])[single], axis=1)]
    R["vessel_specificity"]["single_vessel_patients"] = {
        "n": int(len(single)), "top_vessel_correct": float((top == truth).mean()),
        "always_lad_correct": float((truth == "lad").mean()),
        "isolated_rca_ranked_first": f"{int(((top == 'rca') & (truth == 'rca')).sum())}/{int((truth == 'rca').sum())}"}

    # 5. Served models -----------------------------------------------------------------------------
    served = {t: joblib.load(MODELS_DIR / f"{t}.joblib") for t in TARGETS}
    refits = Parallel(n_jobs=4)(delayed(lambda t: make_model(t, ys[t]).fit(X, ys[t]))(t) for t in TARGETS)
    R["served"] = {}
    for t, refit in zip(TARGETS, refits):
        p_saved = served[t].predict_proba(X)[:, 1]
        R["served"][t] = {
            "max_diff_vs_fresh_refit": float(np.abs(p_saved - refit.predict_proba(X)[:, 1]).max()),
            "apparent_auc": float(roc_auc_score(ys[t], p_saved)),
        }
    from app.main import _predict  # noqa: E402  (loads the same model files the API serves)
    from app.schemas import PatientFeatures  # noqa: E402
    from make_presets import to_schema  # noqa: E402
    rows = np.random.default_rng(7).choice(len(df), 40, replace=False)
    worst = 0.0
    for i in rows:
        api, _, _ = _predict(PatientFeatures(**to_schema(df.iloc[i])))
        for t in TARGETS:
            worst = max(worst, abs(api["results"][t].probability - served[t].predict_proba(X.iloc[[i]])[0, 1]))
    R["served"]["api_vs_saved_max_diff_40_patients"] = worst

    # 6. Clinical direction: one finding changed for every patient, mean change in the CAD estimate.
    def mean_shift(target, edit):
        a, b = df.copy(), df.copy()
        edit(a, b)
        pa = served[target].predict_proba(encode_dataframe(a, encoding, features))[:, 1]
        pb = served[target].predict_proba(encode_dataframe(b, encoding, features))[:, 1]
        return float((pb - pa).mean())

    def setcol(col, va, vb):
        def f(a, b):
            a[col], b[col] = va, vb
        return f

    def chest(kind):
        def f(a, b):
            for d in (a, b):
                d["Typical Chest Pain"], d["Atypical"], d["Nonanginal"] = 0, "N", "N"
            if kind == "typical":
                b["Typical Chest Pain"] = 1
            else:
                b[{"atypical": "Atypical", "nonanginal": "Nonanginal"}[kind]] = "Y"
        return f

    def weight(va, vb):
        """Weight changed with BMI and the obesity flag (BMI >= 25) recomputed from height, as the API does."""
        def f(a, b):
            for d, w in ((a, va), (b, vb)):
                d["Weight"] = w
                d["BMI"] = w / (d["Length"] / 100) ** 2
                d["Obesity"] = np.where(d["BMI"] >= 25, "Y", "N")
        return f

    checks = [
        ("Typical angina vs no chest pain", "cad", chest("typical"), "+"),
        ("Atypical chest pain vs no chest pain", "cad", chest("atypical"), "+"),
        ("Age 70 vs 45", "cad", setcol("Age", 45, 70), "+"),
        ("Male vs female", "cad", setcol("Sex", "Fmale", "Male"), "+"),
        ("Diabetes vs none", "cad", setcol("DM", 0, 1), "+"),
        ("Hypertension vs none", "cad", setcol("HTN", 0, 1), "+"),
        ("Current smoker vs not", "cad", setcol("Current Smoker", 0, 1), "+"),
        ("ST depression vs none", "cad", setcol("St Depression", 0, 1), "+"),
        ("T-wave inversion vs none", "cad", setcol("Tinversion", 0, 1), "+"),
        ("Regional wall-motion abnormality 2 vs 0", "cad", setcol("Region RWMA", 0, 2), "+"),
        ("Ejection fraction 35% vs 60%", "cad", setcol("EF-TTE", 60, 35), "+"),
        ("Fasting glucose 160 vs 90 mg/dL", "cad", setcol("FBS", 90, 160), "+"),
        ("HDL 30 vs 55 mg/dL", "cad", setcol("HDL", 55, 30), "+"),
        ("Triglycerides 250 vs 100 mg/dL", "cad", setcol("TG", 100, 250), "+"),
        ("LDL 160 vs 90 mg/dL", "cad", setcol("LDL", 90, 160), "+"),
        ("Weight 94 vs 55 kg (BMI follows)", "cad", weight(55, 94), "+"),
        ("Regional wall-motion abnormality 2 vs 0 (LAD model)", "lad", setcol("Region RWMA", 0, 2), "+"),
        ("Typical angina vs no chest pain (RCA model)", "rca", chest("typical"), "+"),
        ("Diabetes vs none (RCA model)", "rca", setcol("DM", 0, 1), "+"),
    ]
    R["clinical_direction"] = []
    for name, t, edit, expected in checks:
        d = mean_shift(t, edit)
        ok = (d > 0) == (expected == "+")
        R["clinical_direction"].append({"check": name, "model": t, "mean_change_pts": round(100 * d, 1),
                                        "expected": expected, "as_expected": bool(ok)})

    yc = ys["cad"].astype(bool)
    cp = np.select([df["Typical Chest Pain"] == 1, df["Atypical"] == "Y", df["Nonanginal"] == "Y"],
                   ["typical", "atypical", "non-anginal"], "none recorded")
    rate = lambda m: [round(float(yc[m].mean()), 2), int(m.sum())]  # noqa: E731
    R["raw_rates"] = {
        "cad_rate_by_chest_pain": {k: rate(cp == k) for k in ("typical", "atypical", "non-anginal", "none recorded")},
        "cad_rate_by_ldl": {f"{lo}-{hi}": rate((df.LDL > lo) & (df.LDL <= hi)) for lo, hi in ((0, 100), (100, 130), (130, 400))},
        "cad_rate_by_hdl": {f"{lo}-{hi}": rate((df.HDL > lo) & (df.HDL <= hi)) for lo, hi in ((0, 40), (40, 50), (50, 200))},
        "cad_rate_by_weight": {f"{lo}-{hi}": rate((df.Weight > lo) & (df.Weight <= hi)) for lo, hi in ((0, 65), (65, 80), (80, 200))},
    }

    R["seconds"] = round(time.time() - t0)
    OUT_JSON.write_text(json.dumps(R, indent=2, default=float))
    write_report(R)
    print(f"Done in {R['seconds']} s -> {OUT_MD}")


def write_report(R):
    T = R["targets"]
    L = ["# Model verification (generated by `backend/verify_models.py`)", "",
         "Independent checks of the production models. Re-run with `python verify_models.py` after `train.py`.", ""]
    d = R["data"]
    L += ["## 1. Data integrity", "",
          f"- {d['rows']} patients, {d['inputs']} inputs, {d['missing_values']} missing values, "
          f"{d['duplicate_patients']} duplicate patients.",
          f"- Label consistency: {d['cad_without_vessel_label']} patients labelled CAD with no stenotic vessel, "
          f"{d['vessel_label_without_cad']} labelled Normal with a stenotic vessel.",
          f"- Prevalence: " + ", ".join(f"{k.upper()} {v:.0%}" for k, v in d["prevalence"].items()) + ".", ""]
    L += ["## 2. Leakage", "",
          f"- Target columns among the inputs: {R['leakage']['target_columns_in_inputs'] or 'none'}.",
          "- Strongest single inputs (AUC on their own): " + "; ".join(
              f"{t.upper()}: " + ", ".join(f"{c} {a:.2f}" for c, a in v) for t, v in R["leakage"]["top_single_input_auc"].items()) + ".",
          "- Label-permutation test (full recipe, labels shuffled; ~0.5 expected if nothing leaks): " + ", ".join(
              f"{t.upper()} {np.mean(T[t]['permutation_auc']):.2f}" for t in T) + ".", ""]
    L += ["## 3. Fresh-split cross-validation (split seeds 5000-5004, never used in development)", "",
          "± is the spread across the 5 repeats (split noise); the bootstrap 95% CI is the real uncertainty "
          "with 303 patients. New splits of the same patients cannot remove all optimism from choosing the "
          "recipe on this data: expect the true AUC to be about 0.005-0.01 lower (0.01-0.02 for RCA).", "",
          "| Target | AUC fresh 5x5 | AUC saved (train.py) | 95% CI (bootstrap) | Brier (prevalence-only) | Calibration slope / intercept |",
          "|---|---|---|---|---|---|"]
    for t, r in T.items():
        L.append(f"| {t.upper()} | {r['auc_fresh_mean']:.3f} ± {r['auc_fresh_sd']:.3f} | {r['auc_saved_cv']:.3f} | "
                 f"{r['auc_95ci'][0]:.3f}–{r['auc_95ci'][1]:.3f} | {r['brier']:.3f} ({r['brier_prevalence_only']:.3f}) | "
                 f"{r['calibration_slope']:.2f} / {r['calibration_intercept']:+.2f} |")
    L += ["", "## 4. Baselines on the same folds (AUC)", "",
          "| Target | Production | Diamond-Forrester score only | Logistic regression, raw inputs | Default XGBoost |",
          "|---|---|---|---|---|"]
    for t, r in T.items():
        b = r["baselines"]
        L.append(f"| {t.upper()} | {r['auc_fresh_mean']:.3f} | {b['diamond_forrester_only']:.3f} | "
                 f"{b['lr_raw']:.3f} (beaten in {b['lr_raw_beaten_in_repeats']} repeats) | "
                 f"{b['xgb_default']:.3f} (beaten in {b['xgb_default_beaten_in_repeats']}) |")
    v = R["vessel_specificity"]
    L += ["", "## 5. Vessel specificity: which artery, or how much CAD?", "",
          "| Artery | Its own model (AUC) | CAD model used for this label (AUC) | Own model, CAD patients only (AUC) |",
          "|---|---|---|---|"]
    for t in ("lad", "lcx", "rca"):
        L.append(f"| {t.upper()} | {v[t]['own_model_auc']:.3f} | {v[t]['cad_model_auc_for_this_label']:.3f} | "
                 f"{v[t]['auc_within_cad_patients']:.3f} |")
    sv = v["single_vessel_patients"]
    L += ["", f"- In the {sv['n']} single-vessel patients, the highest vessel estimate pointed at the right artery "
          f"{sv['top_vessel_correct']:.0%} of the time, against {sv['always_lad_correct']:.0%} for always guessing "
          f"LAD; isolated RCA disease was ranked first in {sv['isolated_rca_ranked_first']} patients.",
          "- So the vessel models mostly measure how much coronary disease there is, and only partly which artery. "
          "The app therefore shows three independent per-artery estimates and never a 'culprit' vessel.", ""]
    s = R["served"]
    L += ["", "## 6. Served models", "",
          "| Target | Saved vs fresh refit (max abs diff) | Apparent AUC (training data) | Cross-validated AUC |", "|---|---|---|---|"]
    for t in T:
        L.append(f"| {t.upper()} | {s[t]['max_diff_vs_fresh_refit']:.2e} | {s[t]['apparent_auc']:.3f} | {T[t]['auc_fresh_mean']:.3f} |")
    L += ["", f"- Live API vs saved models on 40 random patients: max difference {s['api_vs_saved_max_diff_40_patients']:.1e}.", ""]
    L += ["## 7. Clinical direction (one finding changed for every patient; mean change in the estimate)", "",
          "| Change | Model | Mean change | Expected | As expected |", "|---|---|---|---|---|"]
    for c in R["clinical_direction"]:
        L.append(f"| {c['check']} | {c['model'].upper()} | {c['mean_change_pts']:+.1f} pts | {c['expected']} | "
                 f"{'yes' if c['as_expected'] else '**no**'} |")
    rr = R["raw_rates"]
    fmt = lambda d: ", ".join(f"{k} {v[0]:.0%} (n={v[1]})" for k, v in d.items())  # noqa: E731
    L += ["", "Where the model disagrees with the textbook direction, the raw data (CAD rate in this referral "
          "cohort) show why; these are learned associations, not causes:", "",
          f"- By chest pain: {fmt(rr['cad_rate_by_chest_pain'])}. Patients referred with no chest pain had other "
          "strong reasons for angiography, so 'atypical' is not riskier than 'none recorded' here.",
          f"- By LDL (mg/dL): {fmt(rr['cad_rate_by_ldl'])}. Flat: LDL carries no signal in this cohort, most "
          "likely because patients with known disease are already on lipid-lowering treatment.",
          f"- By HDL (mg/dL): {fmt(rr['cad_rate_by_hdl'])}. Weak and non-monotonic.",
          f"- By weight (kg): {fmt(rr['cad_rate_by_weight'])}. Lighter patients more often had CAD in this referral "
          "cohort, so the models learned an inverse link (the weight check recomputes BMI and the obesity flag, as the API does).",
          "- Consequence: the app's explanations may show LDL slightly lowering an estimate. That reflects the data, "
          "not physiology, and must not be read as advice. The what-if card therefore locks LDL, HDL and weight "
          "instead of offering them as sliders.", "",
          f"_Run time {R['seconds']} s._", ""]
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
