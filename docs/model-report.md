# CardioVision model report

How the production models were chosen, how well they perform, and where they fall short. All figures come from `backend/models/metrics.json`, produced by `python train.py`, unless stated otherwise. The experiments behind the choice are in `backend/experiments/`.

## 1. Task and data

- **Data:** Extension of the Z-Alizadeh Sani dataset (UCI id 411). 303 patients referred for coronary angiography at one centre. 55 inputs: demographics, history, symptoms, examination, ECG, laboratory and echocardiography.
- **Targets:**
  - overall CAD (`Cath`): 216 positive (71.3%)
  - LAD stenosis: 177 (58.4%)
  - LCX stenosis: 119 (39.3%)
  - RCA stenosis: 114 (37.6%)

  Stenosis means at least 50% narrowing on angiography.
- **Leakage guard:** `LAD`, `LCX`, `RCA` and `Cath` are never model inputs.
- **Baselines from always guessing the larger class:** accuracy of 71.3% for CAD, 58.4% for LAD, 60.7% for LCX and 62.4% for RCA. Accuracy should always be read against these.

## 2. Evaluation protocol

`backend/experiments/harness.py` is shared by every experiment, and `train.py` follows the same protocol.

- **Repeated stratified cross-validation:** 5 folds × 5 repeats. Each repeat gives out-of-fold (OOF) probabilities for all 303 patients. Metrics are averaged over repeats, and ± is the standard deviation across repeats (split noise, not a confidence interval; bootstrap CIs are in the limitations section).
- **Independent verification:** `backend/verify_models.py` re-checks leakage (label permutation), re-runs the recipe on never-used split seeds, compares baselines on the same folds, tests vessel specificity, confirms the served models and API, and checks clinical direction. Results: `docs/model-verification.md`.
- **Everything learned from data happens inside each training fold:** scaling, regularisation strength (inner CV), calibration (inner CV), and decision thresholds (inner CV). The engineered clinical features are stateless formulas that use no labels.
- **Thresholds are honest:** the reported tuned-threshold metrics are *nested*. The threshold is chosen inside each training fold and applied to the held-out fold.
- **How the final model was chosen, to limit selection bias:**
  - Six strategies, about 250 configurations, were explored on split seeds 0–4.
  - Each strategy's claims were re-run by an adversarial verifier on unseen splits (seeds 100+), with a leakage audit. No leakage was found; models trained on shuffled labels score AUC 0.43–0.55.
  - The final pick used seeds 400–800, with the decision rule fixed in advance.
  - An independent re-check on seeds 900–904 found a paired AUC gain over the old model in **every** repeat (table in §4).

## 3. Production models (`backend/app/cardio_models.py`)

Every target is an equal-weight blend, in log-odds, of two members:
1. a regularised **logistic regression** on clinical composite features, and
2. a **shallow gradient-boosted tree** model.

| Target | Logistic member | Tree member | Recalibration |
|---|---|---|---|
| CAD | L2, 8 core composites + raw inputs (log labs) | LightGBM depth-1 "stumps" (an additive model) | none needed (calibration slope 1.13) |
| LAD | L2 (C by inner AUC), Platt-calibrated, core composites | XGBoost depth-4 on all engineered features | Platt on the blend (slope 0.78 → 1.02) |
| LCX | as LAD | as LAD | Platt on the blend (slope 0.65 → 1.00) |
| RCA | L1, all 14 composites (log labs) | LightGBM depth-2, heavily regularised | none needed (slope 0.96) |

**Clinical composites:**
- Diamond-Forrester pre-test probability (from age, sex and chest-pain type)
- number of classic risk factors
- number of ECG abnormalities
- chest-pain class
- any wall-motion abnormality
- TG/HDL ratio, LDL/HDL ratio and the atherogenic index of plasma
- age × male
- CKD-EPI 2021 eGFR
- neutrophil/lymphocyte ratio
- EF < 50%, fasting glucose ≥ 126, systolic BP ≥ 140

VHD is mapped to ordinal severity and BBB is one-hot encoded.

**Where the gain comes from.** Roughly half comes from the model family: the old depth-4, class-weighted XGBoost overfitted and was over-confident. The other half comes from the composites, which are the only change that moved LCX.

**Explanations are exact.** Linear members contribute `coef·(x − mean)`. Tree members use native TreeSHAP from LightGBM/XGBoost. Contributions are averaged across the blend and passed through the Platt map. As a result, `base + Σ contributions = model log-odds`; the API checks this on every request. The `shap` package is not used, because `shap.TreeExplainer` is incompatible with xgboost 3.x.

## 4. Results

### Discrimination and calibration (5×5 repeated CV, decisions at 0.5)

| Target | ROC-AUC | PR-AUC | Brier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| CAD | **0.943 ± 0.005** | 0.976 | 0.083 | 0.883 | 0.910 | 0.928 | 0.919 |
| LAD | **0.856 ± 0.011** | 0.883 | 0.148 | 0.799 | 0.797 | 0.880 | 0.836 |
| LCX | **0.761 ± 0.013** | 0.625 | 0.196 | 0.700 | 0.656 | 0.497 | 0.566 |
| RCA | **0.739 ± 0.006** | 0.594 | 0.199 | 0.682 | 0.596 | 0.479 | 0.530 |

### Operating points (nested: threshold chosen inside each training fold)

| Target | F1-optimal threshold | Accuracy | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|
| CAD | 0.44 | 0.878 | 0.905 | 0.926 | 0.759 | 0.915 |
| LAD | 0.37 | 0.795 | 0.778 | 0.910 | 0.635 | 0.839 |
| LCX | 0.39 | 0.660 | 0.543 | 0.840 | 0.543 | 0.660 |
| RCA | 0.31 | 0.620 | 0.497 | 0.798 | 0.513 | 0.613 |

**CAD rule-out point:** threshold 0.42, sensitivity 0.97, negative predictive value 0.89, specificity 0.66.

### Against the old model: paired on identical folds, split seeds 900–904 (never used for selection)

| Target | Old XGBoost AUC | New AUC | Paired ΔAUC (min across repeats) |
|---|---|---|---|
| CAD | 0.900 | 0.940 | +0.040 (+0.032) |
| LAD | 0.828 | 0.862 | +0.034 (+0.027) |
| LCX | 0.745 | 0.758 | +0.014 (+0.003) |
| RCA | 0.678 | 0.746 | +0.068 (+0.048) |

Brier and log-loss also improve for every target. For example, RCA Brier drops from 0.25 to 0.20, and the calibration slope rises from 0.31 to 0.96.

### Calibration bands shown in the UI (observed rate in cross-validation, pooled over repeats)

| Target | Lower | Intermediate | Higher |
|---|---|---|---|
| CAD (bands at the rule-out 0.42 / balanced 0.58 thresholds) | 15% had CAD | 48% | 93% |
| LAD (bands at 1/3 and 2/3) | 13% | 52% | 84% |
| LCX | 17% | 52% | 69% |
| RCA | 20% | 47% | 71% |

## 5. Comparison with published results

Published accuracies on this dataset reach 83–98%. Most of those figures come from protocols that leak information:
- feature selection or weighting fitted on all 303 patients before cross-validation;
- genetic-algorithm or swarm selection scored on the same CV;
- SMOTE applied before splitting;
- single test sets of about 31–76 patients.

Studies that report a clean protocol agree with our numbers:
- **CAD:** logistic regression under 10-fold CV reaches AUC 0.932 and accuracy 0.90, while the same paper's single 8:2 split reports 0.981 ([Zhang et al. 2024, PLoS One](https://doi.org/10.1371/journal.pone.0307952)). One paper's own consistent ablation gives about 0.885 accuracy against a 97.7% headline ([Baratpur et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12738877/)). **Ours:** AUC 0.94, accuracy 0.88.
- **Vessels:** early 10-fold work reports accuracies of LAD 74–80%, LCX 59–65% and RCA 66–69% ([Alizadehsani et al. 2013](https://pubmed.ncbi.nlm.nih.gov/23717807/); [2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4253773/)). [Jin & Li 2022](https://www.aimspress.com/article/doi/10.3934/mbe.2022211) give a simple baseline of AUC 0.79 / 0.79 / 0.73 on a held-out set of about 31 patients; their 0.90+ stacking numbers are the best of many test-set runs. **Ours:** AUC 0.86 / 0.76 / 0.74, accuracy 0.80 / 0.70 / 0.68.

## 6. Limitations

- **One centre, 303 referred patients, no external validation.** Bootstrap 95% intervals on AUC: CAD 0.91–0.97, LAD 0.82–0.91, LCX 0.71–0.81, RCA 0.70–0.81 (about ±0.03 for CAD and LAD, ±0.05 for LCX and RCA). The cohort is 71% CAD-positive, so the models are not validated for screening populations.
- **LCX and RCA are hard targets.** Their AUC is 0.74–0.76, and at 0.5 they miss about half of stenoses. The UI marks them as having limited discrimination.
- **The vessel models are not very vessel-specific.** They mostly learn "how much CAD" rather than "which artery": the CAD model's own estimate predicts the LAD label as well as the LAD model does (AUC 0.86 vs 0.86), and among CAD patients the vessel AUCs fall to 0.69 (LAD), 0.63 (LCX) and 0.62 (RCA). In the 87 single-vessel patients the highest vessel estimate picks the right artery 61% of the time, against 66% for always guessing LAD, and isolated RCA disease is ranked first in 0 of 17. The UI therefore presents three independent per-artery estimates, never a culprit vessel, and shows the LAD false alarm for the RCA example patient.
- **The four models are trained independently.** For about 20% of patients the highest vessel probability exceeds the CAD probability. The UI flags strong disagreement rather than forcing consistency.
- **Missing inputs** are filled with conventional defaults: an unrecorded finding counts as absent, and measurements take the cohort median. The response lists every filled field. Quick mode (19 inputs, everything else defaulted) was measured with the production models on held-out folds (5-fold, seed 0): CAD AUC 0.935 → 0.937, LAD 0.859 → 0.846, LCX 0.774 → 0.756, RCA 0.744 → 0.728.
- **Selection bias:** the recipe was chosen among many candidates on these 303 patients. New split seeds cannot remove patient-level selection optimism; an independent review estimates about 0.005–0.01 (0.01–0.02 for RCA). On never-used split seeds the CAD AUC is 0.937–0.939 rather than 0.943.
- **Associations, not causes:** in this referral cohort LDL and HDL carry no protective signal (CAD rate is flat across LDL levels), and "no chest pain recorded" carries more risk than atypical pain (63% vs 43% CAD), because those patients were referred for other strong reasons. The models learn these patterns; explanations must not be read as physiology or advice. See `docs/model-verification.md` section 7.

## 7. Reproducing

```bash
cd backend
python train.py                    # ~4-5 min; writes models/ and metrics.json
python make_presets.py             # example patients from held-out predictions
python test_pipeline.py            # end-to-end checks
python experiments/exp_baseline.py # the old model under the same protocol
HARNESS_SEED_OFFSET=900 python experiments/exp_final_recipe.py eval   # re-check on fresh splits
```
