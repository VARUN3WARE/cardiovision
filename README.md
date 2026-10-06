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
│                          │ POST /predict                    │
└──────────────────────────┼─────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Backend  (FastAPI)                                       │
│  /predict  →  CAD classifier + LAD/LCX/RCA classifiers     │
│  /explain  →  SHAP values for the input feature vector     │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Models  (scikit-learn / XGBoost tree ensembles)         │
│  Trained on Z-Alizadeh Sani extension (UCI / Kaggle)      │
│  Leakage-safe: LAD, LCX, RCA, Cath excluded from inputs   │
└──────────────────────────────────────────────────────────┘
```

## Quickstart

### Prerequisites
- Python 3.10+
- Node 18+
- The Z-Alizadeh Sani extension dataset (place CSV in `backend/data/`)

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (one-time, writes to backend/models/)
python train.py

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
App at http://localhost:5173

---

## Project structure
```
backend/
  app/
    main.py            # FastAPI app, /predict and /explain endpoints
    model.py           # model loading + inference
    shap_explainer.py  # SHAP attribution
    schemas.py         # pydantic request/response models
  train.py             # training pipeline (CAD + 3 vessels)
  requirements.txt
  data/                # dataset (gitignored)
  models/              # trained weights (gitignored)
frontend/
  src/
    App.tsx
    components/
      HeartScene.tsx       # R3F 3D heart + artery color-coding
      ClinicalDashboard.tsx # overall CAD + vessel probabilities
      FeatureInput.tsx     # clinical feature form
      ExplanationPanel.tsx  # SHAP breakdown
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
