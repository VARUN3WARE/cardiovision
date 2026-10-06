# CardioVision — Track A: Cardiovascular Risk Visualization & Prediction

Interactive 3D cardiovascular risk prediction system for the **Multimodal AI Hackathon 2026** (Track A).

Predicts overall **Coronary Artery Disease (CAD)** status and vessel-specific **stenosis** for the **LAD**, **LCX**, and **RCA** coronary arteries from clinical/ECG/echo features, then maps the predicted risk onto an interactive 3D heart model with SHAP-based clinical interpretability.

> ⚠️ **Clinical Safety Disclaimer:** This system is for **decision support and educational purposes only**. It is **not** a substitute for formal diagnostic imaging or professional medical advice.

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
│  /predict  →  CAD classifier + LAD/LCX/RCA classifiers     │
│  /explain  →  predictions + SHAP values for all 4 models   │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Models  (scikit-learn / XGBoost tree ensembles)         │
│  Trained on Z-Alizadeh Sani extension (UCI / Kaggle)      │
│  Leakage-safe: LAD, LCX, RCA, Cath excluded from inputs   │
└──────────────────────────────────────────────────────────┘
```

## Features

- **Vessel-level prediction**: four XGBoost classifiers (overall CAD, LAD, LCX, RCA stenosis), 5-fold CV.
- **Anatomical 3D heart** (procedural, no external mesh or image assets): ventricles, crinkled atrial
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
- **Per-vessel SHAP**: diverging bar chart of top drivers with clinical labels and units, plus a
  baseline → patient probability reconciliation.
- **Example patients**: four real dataset patients (normal, isolated LAD, isolated RCA, three-vessel disease),
  chosen by out-of-fold predictions, shown next to their angiography ground truth.
- Heartbeat animation at the patient's pulse rate.

## Quickstart

### Prerequisites
- Python 3.10+
- Node 18+
- The Z-Alizadeh Sani extension dataset as `backend/data/ZAlizadehSani.csv`
  (UCI id 411; download manually, `ucimlrepo` cannot import this dataset)

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (one-time, ~10 s, writes to backend/models/; weights are not committed)
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

---

## Project structure
```
backend/
  app/
    main.py            # FastAPI app, /predict and /explain endpoints
    model.py           # model loading + inference
    shap_explainer.py  # SHAP via XGBoost native pred_contribs
    schemas.py         # pydantic request/response models
    encoders.py        # categorical encoding
    config.py          # shared paths/constants
  train.py             # training pipeline (CAD + 3 vessels)
  make_presets.py      # picks real example patients by out-of-fold prediction
  test_pipeline.py     # in-process end-to-end test
  requirements.txt
  data/                # dataset (gitignored)
  models/              # trained weights (gitignored)
frontend/
  src/
    App.tsx
    components/
      heartAnatomy.ts      # procedural heart surface, epicardial fat, coronary arteries/veins, great vessels,
                           #   atria, territories, noise + generated normal maps
      HeartScene.tsx       # R3F scene: materials, lighting, post-processing, risk coloring, selection,
                           #   camera focus, heartbeat
      ClinicalDashboard.tsx # overall CAD + vessel probabilities (+ ground truth for presets)
      FeatureInput.tsx     # clinical feature form + example patients
      ExplanationPanel.tsx  # per-vessel SHAP breakdown
    featureLabels.ts       # clinical labels/units for features
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

## Dataset
- **Primary**: UCI ML Repository — *Extension of Z-Alizadeh Sani Dataset* (also on Kaggle)
- 303 patient records, demographics + vitals + labs + ECG + echo
- Targets: overall CAD + LAD/LCX/RCA stenosis status
- **Leakage guard**: `LAD`, `LCX`, `RCA`, `Cath` columns are excluded from model input features.

## Model performance (5-fold stratified CV, 303 patients)

| Target | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CAD | 0.871 | 0.900 | 0.921 | 0.911 | 0.906 |
| LAD | 0.795 | 0.811 | 0.847 | 0.829 | 0.846 |
| LCX | 0.710 | 0.640 | 0.597 | 0.617 | 0.744 |
| RCA | 0.657 | 0.551 | 0.474 | 0.509 | 0.678 |

Source: `backend/models/metrics.json` (regenerated by `train.py`).
