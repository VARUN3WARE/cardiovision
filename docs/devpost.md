# Devpost submission text: CardioVision

Paste each section into the matching Devpost field. Links to add: GitHub repository, YouTube demo
(unlisted), and the PDF `docs/CardioVision-Technical-Documentation.pdf`.

**Tagline (≤ 200 characters):** Predicts coronary artery disease and LAD/LCX/RCA stenosis from routine
clinical data, maps each risk onto its artery on a 3D heart, and explains every estimate.

---

## Inspiration

A risk score of "72%" does not tell a clinician *where* coronary disease is likely, or *why* the model thinks
so. Track A asked for predictions that become spatial and inspectable. We also wanted the numbers to be
trustworthy: this dataset has only 303 patients, and many published results on it are inflated by evaluation
leakage. So we set ourselves two goals: make the risk visible on the anatomy, and make every number honest.

## What it does

- **Predicts four things** from 55 clinical inputs (demographics, history, symptoms, examination, ECG, labs,
  echo): overall CAD, and ≥50% stenosis of the LAD, LCX and RCA.
- **Maps each prediction onto its own artery** on an interactive 3D heart: a sculpted, textured open-licence
  model with our coronary tree fitted into its grooves. Each vessel is coloured green → amber → red by its
  own model's probability. A *Territories* mode shades the heart muscle supplied by each artery; clicking an
  artery or the heart wall swings the camera to that vessel and opens its explanation. Anatomy pins, a
  heartbeat at the patient's pulse, and a schematic heart (with epicardial fat and cardiac veins) as a toggle
  and fallback.
- **Explains every estimate exactly**: a one-sentence summary, the top drivers with odds multipliers, a
  waterfall from the model's starting point to the patient, and a physiological breakdown of all 55
  measurements by body system.
- **Is honest about uncertainty**: likelihood bands with the event rate observed in validation, each model's
  cross-validated AUC (weaker vessel models marked), held-out estimates against the real angiography for
  example patients, including false alarms, and a permanent safety disclaimer.
- **Is usable**: quick (19 inputs) or full (55) entry, validation as you type, live updates, and clear error
  handling.

## How we built it

- **Models (Python, scikit-learn, XGBoost, LightGBM).** We built a leakage-safe evaluation harness first (5×5
  repeated stratified CV, everything learned refitted inside each fold, paired comparisons on identical folds).
  Six modelling strategies were explored, each claim independently re-run on unseen splits with a leakage
  audit, and the final design was chosen on fresh splits with a rule fixed in advance. Each target is a
  log-odds blend of a regularised logistic regression on clinical composite features (Diamond-Forrester
  pre-test probability, risk-factor and ECG-abnormality counts, lipid ratios, eGFR, …) and a shallow boosted
  tree model. Decision thresholds and calibration are learned inside training folds.
- **Explanations.** Exact additive attributions (linear terms plus native TreeSHAP, averaged across the blend)
  that sum to the model output, checked on every request.
- **API (FastAPI).** Validation, imputation contract, four models, thresholds, calibration bands,
  out-of-range warnings and attributions in about 130 ms per request.
- **Front end (React, TypeScript, Tailwind).** One field specification drives the form; live, debounced
  re-estimation with stale-result detection.
- **3D (three.js / React Three Fiber, drei, three-mesh-bvh, postprocessing).** We traced the coronary tree on
  the model's sculpted grooves; at load time every path is snapped to the surface with a BVH closest-point
  query, lifted along the normal, smoothed and rendered as a tapered tube bound to exactly one model output.
  Perfusion territories come from nearest-artery proximity, masked below the AV groove.

## Results

| | CAD | LAD | LCX | RCA |
|---|---|---|---|---|
| ROC-AUC (5×5 CV) | **0.943** | **0.856** | **0.761** | **0.739** |
| Accuracy at 0.5 | 0.883 | 0.799 | 0.700 | 0.682 |
| F1 at learned threshold (nested) | 0.915 | 0.839 | 0.660 | 0.613 |

On splits never used for selection, the final models beat our first model in every repeat (AUC +0.040 /
+0.034 / +0.014 / +0.068) with far better calibration. These results are at the top of what protocol-clean
studies on this dataset report.

## Challenges we ran into

- **Small data, optimistic literature.** With 303 patients, split noise alone moves RCA AUC by about 0.02.
  Hyper-parameter searches inside folds, stacking and resampling all looked good on one split and failed on
  fresh ones; only paired, repeated, pre-registered comparisons separated real gains from noise.
- **Silent data bugs.** Platelets are stored in thousands, so a 250,000 entry was wildly out of distribution,
  and one input was never reaching the model because of a name mismatch. Both are now caught by validation and
  a test that the API schema and the model features match one-to-one.
- **Imputation that lies.** Filling missing yes/no findings with the cohort's most common value would have
  assumed typical angina and hypertension (majority-positive in this referral cohort). Unrecorded findings now
  count as absent, and every filled value is shown.
- **Fitting arteries onto a real mesh.** The model's coronaries are sculpted grooves, not objects. We traced
  them on the surface and snapped tubes into the grooves with a BVH, then derived territories from the result.

## Accomplishments that we're proud of

- An evaluation we can defend line by line, with every gain re-verified on unseen data.
- Explanations that are mathematically exact, not approximations.
- A 3D view where each coloured vessel *is* one model's output, on a realistic heart, running on an
  integrated GPU.
- Showing the model's own false alarms in the demo instead of hiding them.

## What we learned

Honest validation is most of the work on small clinical data. Calibration matters as much as AUC once
probabilities are shown to people. Clinically motivated features (the Diamond-Forrester pre-test table, for
example) beat model tuning.

## What's next

External validation on a second cohort; vessel-specific modelling (the current vessel models partly learn
"how much CAD" rather than "which artery"); a printable patient report; and segment-level lesion display if
imaging-derived labels become available.

## Built with

python · fastapi · pydantic · scikit-learn · xgboost · lightgbm · pandas · react · typescript · tailwindcss ·
vite · three.js · react-three-fiber · drei · three-mesh-bvh · postprocessing

## Credits

Dataset: Extension of Z-Alizadeh Sani dataset, UCI Machine Learning Repository. 3D heart model: "Realistic
Human Heart" by neshallads (Sketchfab), CC BY 4.0.

**Safety:** decision support / educational use only; not a diagnosis and not a substitute for angiography or
professional medical judgement.
