# CardioVision — Track A: Cardiovascular Risk Visualization & Prediction

Interactive 3D cardiovascular risk prediction system for the **Multimodal AI Hackathon 2026** (Track A).

Predicts overall **Coronary Artery Disease (CAD)** and vessel-specific **stenosis** (≥50% narrowing) of the **LAD**, **LCX** and **RCA** coronary arteries from 55 clinical, ECG, laboratory and echo features, then maps the risk onto an interactive 3D heart with exact per-feature explanations.

| Validated performance (5×5 repeated CV, n=303) | CAD | LAD | LCX | RCA |
|---|---|---|---|---|
| ROC-AUC | **0.943** | **0.856** | **0.761** | **0.739** |
| Accuracy (at 0.5) | 0.883 | 0.799 | 0.700 | 0.682 |

**Documentation:** [technical documentation (PDF, 5 pages)](docs/CardioVision-Technical-Documentation.pdf) ·
[source](docs/technical-documentation.md) · [model report](docs/model-report.md) (methods, operating points,
calibration, literature comparison, limitations) · [demo video script](docs/demo-script.md) ·
[Devpost text](docs/devpost.md).

> ⚠️ **Clinical Safety Disclaimer:** This system is for **decision support and educational purposes only**. It was trained on 303 patients referred for angiography at a single centre and is **not** validated for screening. It is **not** a diagnosis and does not replace angiography, formal diagnostic imaging or professional medical judgement.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Frontend  (React + Vite + TS + React Three Fiber)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ FeatureInput │  │ HeartScene   │  │ ExplanationPanel │  │
│  │ (clinical    │  │ (3D heart,   │  │ (SHAP feature    │  │
│  │  form)       │  │  artery      │  │  attributions)   │  │
│  │              │  │  color-code) │  │                  │  │
│  └──────┬───────┘  └──────▲───────┘  └────────▲─────────┘  │
│         │                 │                   │            │
│         └─────────────────┴───────────────────┘            │
│                          │ POST /explain                    │
└──────────────────────────┼─────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Backend  (FastAPI)                                       │
│  /predict  →  4 models: probability, threshold, risk band │
│  /explain  →  + exact per-feature attributions (log-odds)  │
│  /metrics, /features, /health  →  validation + metadata    │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Models  (app/cardio_models.py)                           │
│  Log-odds blend per target: logistic regression on        │
│  clinical composites + shallow boosted trees (LightGBM /  │
│  XGBoost); leakage-safe (LAD/LCX/RCA/Cath never inputs)   │
└──────────────────────────────────────────────────────────┘
```

## Features

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
  middle, small cardiac veins, coronary sinus). Each artery is colored green → amber → red by its predicted
  stenosis probability.
- **Realistic rendering**: wet-tissue physical materials (clearcoat + reddish sheen), procedural muscle-fibre
  normal map, studio environment lighting built from light-formers (no HDR download), contact shadow, and an
  "HD" post-processing pass (N8AO ambient occlusion, bloom on the selected vessel, ACES tone mapping,
  vignette) that can be switched off on slow GPUs. Ventricles and atria contract separately at the patient's
  pulse rate.
- **Perfusion territories**: myocardium shaded by the risk of the vessel that supplies it (LAD: anterior wall,
  septum, apex; LCX: lateral wall; RCA: RV and inferior wall).
- **Click to inspect**: clicking an artery, its label, or a region of heart wall (which selects the artery
  perfusing it) swings the camera to that territory, highlights the supplied myocardium and switches the SHAP
  panel to that vessel's model.
- **Plain-language explanations per model**: a summary sentence, the top drivers with strength and odds
  multipliers, derived clinical scores labelled with their inputs, and a waterfall in percentage points. The
  attributions are exact (linear terms + native TreeSHAP, base + Σ = model log-odds, checked per request).
- **Example patients**: five real dataset patients (normal, LAD, RCA, three-vessel, and an honest uncertain
  case), shown with their angiography result and the *held-out* estimate (from models that never saw them),
  including false alarms.
- **Anatomy pins**: numbered annotations with short clinical notes.
- Heartbeat animation at the patient's pulse rate.

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

# Sanity check (no server needed)
python test_pipeline.py

# Serve API
uvicorn app.main:app --reload --port 8000
```
API docs at http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev
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
  test_pipeline.py     # in-process end-to-end test
  experiments/         # leakage-safe harness + the model-selection experiments
  requirements.txt     # runtime; requirements-dev.txt adds experiment-only packages
  data/                # UCI zip (committed); extracted CSV (gitignored)
  models/              # metrics.json, feature_info.json (committed); trained weights (gitignored)
frontend/
  src/
    App.tsx
    components/
      heartAnatomy.ts      # procedural heart surface, epicardial fat, coronary arteries/veins, great vessels,
                           #   atria, territories, noise + generated normal maps
      HeartScene.tsx       # R3F scene: model toggle/fallback, lighting, post-processing, selection,
                           #   camera focus, schematic heart
      SpecimenHeart.tsx    # realistic heart model: territories overlay, vessels, pins, heartbeat
      specimenAnatomy.ts   # coronary tree traced on the model, surface snapping, territories, views
      ArteryTube.tsx       # risk-coloured, clickable vessel tube with label (shared by both hearts)
      ClinicalDashboard.tsx # risk bands, validation AUC, held-out vs angiography, notices
      FeatureInput.tsx     # clinical form: quick/full mode, validation, example patients
      ExplanationPanel.tsx  # plain-language per-model explanations
      ErrorBoundary.tsx    # keeps the app usable if a panel (e.g. WebGL) fails
    fieldSpec.ts           # single source for form fields, units, ranges, payload/validation
    featureLabels.ts       # clinical labels for model features and derived scores
    risk.ts                # probability / band / reliability formatting
    presets.json           # generated by backend/make_presets.py
    api/client.ts
  package.json
```

## Evaluation (per rubric)
- **Predictive Performance (30%)** — accuracy, precision, recall, F1, ROC-AUC for CAD + LAD/LCX/RCA
- **3D Visualization (25%)** — interactive heart, artery risk color-mapping, rotate/zoom/select
- **Clinical Interpretability (20%)** — SHAP feature attributions, physiological breakdown
- **System Integration (15%)** — data pipeline, model↔dashboard, real-time updates
- **Technical Implementation (10%)** — architecture, reproducibility, public datasets/meshes

## Credits
- 3D heart model: ["Realistic Human Heart"](https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089)
  by **neshallads**, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (`frontend/public/models/heart.glb`; geometry unmodified, textures re-encoded as WebP to cut the file from 7.55 to
  1.6 MB; attribution also shown in the app).
- Atlas heart and chest: **BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0
  International** ([licence](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)); derived files
  `frontend/public/models/bp3d_*.glb` are distributed under CC BY-SA 4.0. Changes and sources:
  [`frontend/public/models/ATTRIBUTION.md`](frontend/public/models/ATTRIBUTION.md).
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

Decisions at 0.5 shown. The app uses the F1-optimal threshold learned in training (nested estimates: LCX
F1 0.66, RCA F1 0.61, CAD rule-out sensitivity 0.97 with NPV 0.89); see the model report. The previous XGBoost
model scored AUC 0.90 / 0.83 / 0.75 / 0.68 under the same protocol. Source: `backend/models/metrics.json`
(regenerated by `train.py`).
