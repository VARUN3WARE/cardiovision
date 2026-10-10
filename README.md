# CardioVision — Track A: Cardiovascular Risk Visualization & Prediction

Interactive 3D cardiovascular risk prediction system for the **Multimodal AI Hackathon 2026** (Track A).

Predicts overall **Coronary Artery Disease (CAD)** and vessel-specific **stenosis** (≥50% narrowing) of the **LAD**, **LCX** and **RCA** coronary arteries from 55 clinical, ECG, laboratory and echo features, then maps the risk onto an interactive 3D heart with exact per-feature explanations. An interactive twin layer goes further: the heart beats with the patient's recorded pulse, ejection fraction, blood pressure and ECG findings; the models' evidence flows into it by data type; a what-if card shows what moves each estimate; a teaching lab shows what a narrowed artery does; the most similar real patients are shown with their angiograms; and one button plays a captioned tour of it all.

| Validated performance (5×5 repeated CV, n=303) | CAD | LAD | LCX | RCA |
|---|---|---|---|---|
| ROC-AUC | **0.943** | **0.856** | **0.761** | **0.739** |
| 95% CI (bootstrap, independent re-run) | 0.91–0.97 | 0.82–0.91 | 0.71–0.81 | 0.70–0.81 |
| Accuracy (at 0.5) | 0.883 | 0.799 | 0.700 | 0.682 |

**Documentation:** [technical documentation (PDF, 5 pages)](docs/CardioVision-Technical-Documentation.pdf) ·
[source](docs/technical-documentation.md) · [model report](docs/model-report.md) (methods, operating points,
calibration, literature comparison, limitations) · [model verification](docs/model-verification.md) (leakage,
fresh splits, baselines, vessel specificity) · [demo video script](docs/demo-script.md) ·
[Devpost text](docs/devpost.md).

> ⚠️ **Clinical Safety Disclaimer:** This system is for **decision support and educational purposes only**. It was trained on 303 patients referred for angiography at a single centre and is **not** validated for screening. It is **not** a diagnosis and does not replace angiography, formal diagnostic imaging or professional medical judgement.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Frontend (React + Vite + TS + Tailwind + React Three Fiber)      │
│                                                                  │
│ FeatureInput → App → HeartScene: Realistic, Atlas or Schematic   │
│ (clinical form)  │     heart; function twin + ECG monitor;       │
│                  │     reasoning flow; ischaemia lab (sandbox);  │
│                  │     story mode (guided tour)                  │
│                  ├→ ClinicalDashboard, bullseye, What-if card    │
│                  ├→ ExplanationPanel + evidence by data type     │
│                  └→ Similar patients: kNN in the browser over    │
│                     cohort.json, 3D cohort map                   │
└───────────┬──────────────────────────────────┬───────────────────┘
            │ POST /explain (every estimate)   │ POST /predict (what-if drags)
            ▼                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ Backend (FastAPI)                                                │
│ /predict  →  4 models: probability, threshold, risk band         │
│ /explain  →  + exact per-feature attributions (log-odds)         │
│ /metrics, /features, /health  →  validation + metadata           │
└────────────────────────────────┬─────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ Models (app/cardio_models.py)                                    │
│ Log-odds blend per target: logistic regression on clinical       │
│ composites + shallow boosted trees (LightGBM / XGBoost);         │
│ leakage-safe (LAD/LCX/RCA/Cath never inputs)                     │
└──────────────────────────────────────────────────────────────────┘
Offline: train.py → models/ · make_presets.py → presets.json ·
         make_cohort.py → cohort.json · verify_models.py → docs/model-verification.md
```

## Features

### Interactive twin and simulations (standout layer)

Deep links open each view directly (append to `http://localhost:5173/`). `?patient=low|lad|rca|triple|uncertain`
loads a dataset patient and `?heart=realistic|atlas|schematic` picks the heart.

- **Function twin + ECG monitor**: the heart beats at the patient's recorded pulse, with the squeeze set by the
  ejection fraction, the aortic pulse by systolic BP and a late lateral wall in left bundle branch block; a
  two-lead illustrative ECG (II, V1), in step with the beat, shows the recorded ECG findings. All three hearts.
  `?patient=lad&heart=atlas`
- **Reasoning flow**: each model's exact attributions summed by data type (history & demographics, symptoms &
  exam, ECG, echo, labs, pre-test score) stream from towers around the heart into the selected artery, or into
  a halo round the whole heart for overall CAD (width = effect, red pushes up, blue pulls down). The explanation
  panel shows the same numbers as an "Evidence by data type" strip. `?patient=lad&heart=atlas&reasoning=1`
  (add `&model=lad|lcx|rca` for one artery)
- **What-if**: sliders for systolic BP, fasting glucose and triglycerides, a quit-smoking switch and "flip one
  recorded finding" switches (typical angina, ST depression, wall motion, EF) re-run the models live; a delta
  strip shows each change. The form is never edited, Reset returns to the patient exactly, and LDL, HDL and
  weight are locked (no learned benefit in this cohort, [verification §7](docs/model-verification.md)).
  `?patient=rca&whatif=1&contrast=angina` (also `bp=`, `fbs=`, `tg=`, `quit=1`, `contrast=angina,st,rwma,ef`)
- **Ischaemia lab** (teaching sandbox, Atlas heart): drag a generic narrowing (0–95%) and exercise (rest → peak)
  and watch flow pulses (mostly in diastole) thin past the SANDBOX marker, the territory turn ischaemic and
  contract less, the ischaemic cascade light up beside the dataset inputs it corresponds to, and the flow-reserve
  curve show why ≥50% is the threshold. Textbook coronary physiology, unit-tested against the published curve;
  exercise starts from the patient's recorded pulse, BP and age. `?patient=triple&lab=lad&narrowing=90&exercise=1`
- **Similar patients + cohort map**: the 10 most similar of the 303 real patients (found in the browser, inputs
  weighted by what the CAD model uses, inputs not entered left out), their held-out estimates against their
  angiograms, and a 3D t-SNE map of the cohort coloured by outcome. `?patient=triple&cohort=1`
- **Story mode**: the **▶ Story** button plays a captioned 92-second tour of the current patient on the Atlas
  heart in six steps: the chest fly-in; the function twin; the reasoning streams for overall CAD; the artery in
  the highest band with its inspector ("independent estimates, not a culprit ranking"); the ischaemia lab
  narrowing that artery 0 → 85% and then exercising rest → peak; and a closing card (reference anatomy, referral
  cohort, cross-validated AUCs with bootstrap CIs, decision support only). Play/pause, previous/next, progress
  dots, arrow keys, Esc; Exit gives back your exact view (heart, modes, lab, selection, camera) and never changes
  the patient's data or a what-if scenario; reduced motion gives static steps.
  `?patient=triple&story=1` (plays from the start), `&step=1..6` (that step as it ends, paused), `&play=1`
  (plays from the start of that step)

Each layer says what it is: the anatomy is a reference model, not the patient's imaging; the twin uses the
recorded values; attributions are associations in a referral cohort, not causes; the what-if shows model
sensitivity, not the effect of treatment; the lab is a labelled sandbox, never the patient's lesion.

### Core

- **Validated models**: per target, an equal-weight log-odds blend of a regularised logistic regression on
  clinical composites (Diamond-Forrester pre-test probability, risk-factor and ECG-abnormality counts, lipid
  ratios, eGFR, ...) and a shallow gradient-boosted tree model. Chosen from six strategies by a leakage-safe
  5×5 repeated-CV protocol with adversarial re-verification on unseen splits (`backend/experiments/`).
  Decision thresholds and calibration are learned inside training folds only.
- **Usable clinical form**: quick mode (19 key inputs) or full mode (55); units, ranges and typical values
  inline; chest pain and smoking as single choices; BMI calculated; validation as you type; any field can be
  left empty (unrecorded findings count as absent, measurements take the cohort median) and the result lists
  what was filled in; out-of-range values are flagged.
- **Live results**: estimates update as you type, stale results are clearly marked, and API errors are shown
  in place (field-level validation, offline / untrained-model messages, status indicator).
- **Honest results display**: whole-percent probabilities, risk bands with the event rate actually observed in
  validation, each model's cross-validated AUC, weaker vessel models marked, disagreement between the CAD and
  vessel models flagged, safety disclaimer always visible and included in every API response.
- **Realistic 3D heart** (default): a sculpted, textured heart model (credit below) with our own coronary tree
  fitted onto it. The LAD, diagonals, LCX, obtuse marginals, RCA, RV branch and PDA were traced on the model's
  sculpted grooves; at load time each path is snapped onto the surface (BVH closest point), lifted along the
  normal and smoothed, so the risk-coloured vessels lie in the real grooves. Perfusion territories are computed
  per vertex from the nearest artery tree and masked to the ventricles below the AV groove; clicking the wall,
  camera focus, labels, anatomy pins and the heartbeat all work on this model.
- **Atlas 3D heart in the chest** (toggle): built from **BodyParts3D**, the anatomy database named in the
  brief. Every coronary branch is a named anatomical structure (59 BodyParts3D meshes with FMA ids: LAD trunk,
  diagonals, septals, conus and right anterior branches; circumflex; RCA trunk, conus, marginal, ventricular and
  posterior branches, PDA and its septals; left main) and takes its colour from exactly one model output; the
  left main is neutral because the dataset does not label it. Clicking a branch names it ("Diagonal branch of
  anterior descending branch of left coronary artery, FMA3860 → LAD → coloured by the LAD model"). The heart
  opens inside a ghosted chest (skin, ribs, costal cartilage, sternum, spine, clavicles, scapulae, diaphragm,
  trachea) and the camera flies in; a camera-facing cutaway keeps the heart visible through the chest wall from
  any angle. X-ray mode shows the vessels on the back of the heart from the front; A/P/L/R/S/I buttons snap to
  the standard views. Territories come from each wall vertex's distance to the LAD/LCX/RCA trees (right
  dominance, as in this reference heart); epicardial fat follows the groove vessels. Both GLBs together are
  under 0.8 MB. Rebuild from the official archive with `npm run assets:bp3d` (fetches only the 181 listed
  meshes over HTTP Range, CRC-checked, never the whole 143 MB zip).
- **Schematic 3D heart** (toggle, and automatic fallback if the model file is missing or fails to load;
  procedural, no external assets): ventricles, crinkled atrial
  appendages, aortic root and arch with its three branches, pulmonary trunk/arteries, venae cavae and pulmonary
  veins as cut stumps. Coronary arteries run in their real grooves: LAD + diagonals in the anterior
  interventricular groove, LCX + obtuse marginals in the left AV groove, RCA + conus/RV/acute marginal/PDA/
  posterolateral branches in the right AV and posterior interventricular grooves. They taper and wander
  slightly, sit half-embedded in lobular **epicardial fat**, and are paired with the cardiac veins (great,
  middle, small cardiac veins, coronary sinus). Each artery takes its model's band colour (green, amber, red).
- **Realistic rendering**: wet-tissue physical materials (clearcoat + reddish sheen), procedural muscle-fibre
  normal map, studio environment lighting built from light-formers (no HDR download), contact shadow, and an
  "HD" post-processing pass (N8AO ambient occlusion, bloom on the selected vessel, ACES tone mapping,
  vignette) that can be switched off on slow GPUs.
- **One band colour scale** across the 3D arteries, cards and legend (lower / undecided / higher, with stripes
  for undecided); the last estimate stays on screen while you type, so the arteries never flash grey.
- **Perfusion territories**: myocardium shaded by the risk of the vessel that supplies it (LAD: anterior wall,
  septum, apex; LCX: lateral wall; RCA: RV and inferior wall).
- **Click to inspect**: clicking an artery, its label, or a region of heart wall (which selects the artery
  perfusing it) swings the camera to that territory, highlights the supplied myocardium, switches the
  explanation panel to that vessel's model and opens the **vessel inspector**: band, the rate observed in
  validation, the decision threshold with its measured sensitivity and specificity, the model's AUC, the main
  drivers, the territory with its AHA segments, and what it does not show (Esc closes).
- **AHA 17-segment bullseye** in the dashboard, linked to the 3D heart: hover a segment to find its artery and
  tint its territory, click to inspect it.
- **Plain-language explanations per model**: a summary sentence, the top drivers with strength and odds
  multipliers, derived clinical scores labelled with their inputs, a waterfall in percentage points and a
  physiological breakdown of all 55 measurements by body system. The attributions are exact (linear terms +
  native TreeSHAP, base + Σ = model log-odds, checked per request).
- **Example patients**: five real dataset patients (normal, LAD, RCA, three-vessel, and an honest uncertain
  case), shown with their angiography result and the *held-out* estimate (from models that never saw them),
  including false alarms.
- **Anatomy pins** (Realistic and Schematic hearts): numbered annotations with short clinical notes.

### 3D performance without a dedicated GPU

Full-frame render time (scene plus post-processing), measured on a laptop's integrated GPU (Intel Alder Lake
GT2, Mesa, Chrome) with the 3D panel at 676×757 px, device pixel ratio 1. Frames were timed synchronously
(render, then a 1-pixel GPU read-back), median and range over repeated runs:

| Heart model | HD on (AO, bloom, vignette) | HD off |
|---|---|---|
| Realistic (Sketchfab, textured) | 11 ms | 6 ms |
| Atlas (BodyParts3D heart) | 14 ms (12–17) | 11 ms (4–11) |
| Atlas in the chest (heart + 74k-triangle chest) | 12 ms (10–18) | 9 ms (7–9) |
| Schematic (procedural) | 14 ms | 11 ms |

Every configuration stays under 18 ms per frame (above 55 fps). The device pixel ratio is capped at 1.5,
contact shadows are rendered once per model, and HD switches itself off if the frame rate stays below about
24 fps. The asset sizes are small: Realistic heart 1.6 MB, Atlas heart 0.39 MB, chest 0.35 MB. The table was
measured before the twin layers were added; they allocate nothing per frame and use instanced particles, but
have not yet been re-timed on the integrated GPU.

## Quickstart

### Prerequisites
- Python 3.10–3.13
- Node 18+
- The dataset ships with the repo (`backend/data/extention.zip`, the UCI download); `train.py` extracts it.

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (one-time, ~4-5 min on a 4+ core laptop; weights are not committed)
python train.py

# Optional: regenerate the frontend's example patients (frontend/src/presets.json)
python make_presets.py

# Optional: regenerate the "Similar patients" data (frontend/src/cohort/cohort.json, a few seconds,
# byte-identical on re-runs). Re-run it after any retrain, so its weights and held-out estimates match.
python make_cohort.py

# Sanity check (no server needed)
python test_pipeline.py

# Optional: independent model verification (a few minutes; writes docs/model-verification.md)
python verify_models.py

# Serve API
uvicorn app.main:app --reload --port 8000
```
API docs at http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
npm test        # unit tests (vitest): twin, coronary physiology, lab, reasoning groups, what-if, cohort kNN, story
```
App at http://localhost:5173 (Vite proxies API calls to :8000, so start the backend first).
To point a production build at another API host, set `VITE_API_URL` at build time (CORS is enabled).

### Troubleshooting
- **"API offline"** in the header: start the backend (`uvicorn app.main:app --port 8000`), then click *retry*.
- **"Models not trained"**: run `python train.py` in `backend/`.
- **Slow or blank 3D view**: switch *HD* off in the 3D toolbar; without WebGL the app shows a 2D fallback and
  everything else keeps working.

---

## Project structure
```
backend/
  app/
    main.py            # FastAPI app: /predict, /explain, /metrics, /features, /health
    cardio_models.py   # production models, clinical feature engineering, exact attributions
    model.py           # model loading + inference (thresholds, risk bands)
    shap_explainer.py  # per-feature attributions for the API (additivity checked)
    schemas.py         # pydantic request/response models, validation, imputation contract
    encoders.py        # categorical encoding
    config.py          # shared paths/constants
  train.py             # training: 5x5 CV metrics, thresholds, calibration bands, feature info
  make_presets.py      # picks real example patients from held-out predictions
  make_cohort.py       # similarity space + 3D t-SNE map of the 303 patients -> frontend/src/cohort/cohort.json
  verify_models.py     # independent verification -> docs/model-verification.md, models/verification.json
  test_pipeline.py     # in-process end-to-end test
  experiments/         # leakage-safe harness + the model-selection experiments
  requirements.txt     # runtime; requirements-dev.txt adds experiment-only packages
  data/                # UCI zip (committed); extracted CSV (gitignored)
  models/              # metrics, features, encoding, feature_info, verification JSON (committed); weights (gitignored)
frontend/
  public/models/       # heart.glb (Realistic), bp3d_heart.glb + bp3d_thorax.glb (Atlas), ATTRIBUTION.md
  scripts/             # build-bp3d.mjs (Atlas GLBs), compress-heart.mjs
  src/
    App.tsx
    anatomy/registry.ts    # one artery registry (VESSELS, territories, AHA segments) drives every view
    components/
      HeartScene.tsx       # R3F scene: heart toggle/fallback, toolbar, lighting, post-processing, selection,
                           #   camera, measured overlay layout, story playback, schematic heart
      SpecimenHeart.tsx    # realistic heart: territories overlay, vessels, pins, heartbeat
      specimenAnatomy.ts   # coronary tree traced on the model, surface snapping, territories, views
      AtlasHeart.tsx       # BodyParts3D heart + chest: named branches, X-ray, lab pulses and marker
      atlasAnatomy.ts      # atlas geometry, territories, views, distance from the ostium (lab)
      atlasMaterials.ts    # shader patches: tissue bump, chest cutaway, stripes, flow pulses, hypokinesis
      heartAnatomy.ts      # procedural (schematic) heart, fat, arteries/veins, territories, normal maps
      ArteryTube.tsx       # risk-coloured, clickable vessel tube with label (shared material)
      VesselInspector.tsx  # one artery's estimate, reliability, drivers and territory
      Bullseye.tsx         # AHA 17-segment map linked to the 3D heart
      ClinicalDashboard.tsx # risk bands, validation AUC, held-out vs angiography, notices
      WhatIfPanel.tsx      # what-if card: sliders, locked inputs, finding switches, delta strip
      FeatureInput.tsx     # clinical form: quick/full mode, validation, example patients
      ExplanationPanel.tsx # plain-language per-model explanations
      ErrorBoundary.tsx    # keeps the app usable if a panel (e.g. WebGL) fails
    twin/                  # function twin: heart clock, beat timing, mechanics, ECG (twin.ts, EcgMonitor.tsx,
                           #   useHeartMotion.ts) and the lab's coronary physiology (coronaryPhysiology.ts)
    reasoning/             # data-type groups (dataGroups.ts), 3D streams (ReasoningFlow.tsx), anchors,
                           #   "Evidence by data type" strip
    lab/                   # ischaemia lab panel, flow-reserve curve, lab settings and deep link
    whatif/                # scenario logic (scenario.ts), API wiring (useScenario.ts), strip above the 3D view
    cohort/                # cohort.json (generated), kNN (knn.ts), Similar patients card, 3D cohort map
    story/                 # story mode: the six steps and captions (storySteps.ts), controls and closing
                           #   card (StoryOverlay.tsx), step clock, AUC CIs copied from verification.json
    fieldSpec.ts           # single source for form fields, units, ranges, payload/validation
    featureLabels.ts       # clinical labels for model features and derived scores
    risk.ts                # probability / band / reliability formatting, band colours
    presets.json           # generated by backend/make_presets.py
    api/client.ts
  package.json
scripts/bp3d/              # BodyParts3D part list and range fetcher (npm run assets:bp3d)
```

## Evaluation (per rubric)
- **Predictive Performance (30%)** — ROC-AUC, accuracy, precision, recall, F1 for CAD + LAD/LCX/RCA under
  5×5 repeated CV; independent verification (no leakage, fresh splits, bootstrap CIs, baselines)
- **3D Visualization (25%)** — three hearts (Realistic, BodyParts3D Atlas in the chest, Schematic), artery risk
  colours, territories, inspector, bullseye; the function twin with ECG, reasoning streams, the ischaemia lab,
  the 3D cohort map and the guided story mode
- **Clinical Interpretability (20%)** — exact attributions, plain-language panel, physiological breakdown,
  evidence by data type, what-if sensitivity, the ischaemic cascade, similar patients with their angiograms
- **System Integration (15%)** — data pipeline, model↔dashboard↔3D, live updates, what-if re-runs while dragging
- **Technical Implementation (10%)** — architecture, reproducibility (deterministic generators, pinned deps),
  unit tests, public datasets and meshes

## Credits
- 3D heart model: ["Realistic Human Heart"](https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089)
  by **neshallads**, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (`frontend/public/models/heart.glb`; geometry unmodified, textures re-encoded as WebP to cut the file from 7.55 to
  1.6 MB; attribution also shown in the app).
- Atlas heart and chest: **BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0
  International** ([licence](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)); derived files
  `frontend/public/models/bp3d_*.glb` are distributed under CC BY-SA 4.0. Changes and sources:
  [`frontend/public/models/ATTRIBUTION.md`](frontend/public/models/ATTRIBUTION.md).
- Ischaemia lab physiology: published models and values, cited in `frontend/src/twin/coronaryPhysiology.ts` and
  in the lab's Sources list (Gould 1974 and 1980, Pijls 1993, Uren 1994, Dodge 1992, Gobel 1978, Nesto &
  Kowalchuk 1987, Gallagher 1984, Seligman 2022, ACC/AHA 2002, Fletcher 2013).
- Dataset: Z-Alizadeh Sani et al., UCI Machine Learning Repository (see below).

## Dataset
- **Primary**: UCI ML Repository — *Extension of Z-Alizadeh Sani Dataset* (also on Kaggle)
- 303 patient records, demographics + vitals + labs + ECG + echo
- Targets: overall CAD + LAD/LCX/RCA stenosis status
- **Leakage guard**: `LAD`, `LCX`, `RCA`, `Cath` columns are excluded from model input features.

## Model performance (5×5 repeated stratified CV, 303 patients)

| Target | ROC-AUC | PR-AUC | Brier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| CAD | 0.943 ± 0.005 | 0.976 | 0.083 | 0.883 | 0.910 | 0.928 | 0.919 |
| LAD | 0.856 ± 0.011 | 0.883 | 0.148 | 0.799 | 0.797 | 0.880 | 0.836 |
| LCX | 0.761 ± 0.013 | 0.625 | 0.196 | 0.700 | 0.656 | 0.497 | 0.566 |
| RCA | 0.739 ± 0.006 | 0.594 | 0.199 | 0.682 | 0.596 | 0.479 | 0.530 |

± is the spread across the 5 CV repeats (split noise), not a confidence interval. Bootstrap 95% CIs are wide
with 303 patients: CAD 0.91–0.97, LAD 0.82–0.91, LCX 0.71–0.81, RCA 0.70–0.81.

**Independently verified** (`backend/verify_models.py` → [`docs/model-verification.md`](docs/model-verification.md)):
no target leakage (the full recipe on shuffled labels scores AUC 0.48–0.52); re-run on split seeds never used in
development it gives AUC 0.939 / 0.859 / 0.745 / 0.754 (CAD / LAD / LCX / RCA), so LCX and RCA move about ±0.02
between split sets; it beats plain logistic regression, default XGBoost and the Diamond-Forrester score alone on the
same folds; calibration slopes 0.85–1.11; the served models equal a fresh refit and the API returns exactly their
predictions. Two honest limits it confirms: the vessel models mostly measure how much CAD there is, not which
artery (within CAD patients their AUC falls to 0.62–0.69), and LDL/HDL carry no protective signal in this
referral cohort (so the what-if card locks LDL, HDL and weight). Expect the true AUCs to be about 0.005–0.01
lower than shown (0.01–0.02 for RCA), because the recipe was chosen on these 303 patients.

Decisions at 0.5 shown. The app uses the F1-optimal threshold learned in training (nested estimates: LCX
F1 0.66, RCA F1 0.61, CAD rule-out sensitivity 0.97 with NPV 0.89); see the model report. The previous XGBoost
model scored AUC 0.90 / 0.83 / 0.75 / 0.68 under the same protocol. Source: `backend/models/metrics.json`
(regenerated by `train.py`).
