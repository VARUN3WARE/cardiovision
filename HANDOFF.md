# HANDOFF -- CardioVision (Track A)

Handoff doc for the next agent (Claude) picking up this project.
Last updated: 2026-10-06, ~midnight. Day 1 complete. Day 2+ pending.

## 1. What this project is

Multimodal AI Hackathon 2026 -- Track A: Cardiovascular Risk Visualization and Prediction.

Build an interactive 3D system that predicts Coronary Artery Disease (CAD) and vessel-specific stenosis for the LAD, LCX, RCA coronary arteries from clinical/ECG/echo features, then maps the predicted risk onto an interactive 3D heart model with SHAP-based clinical interpretability.

- Hackathon dev window: 30 Sep to 14 Oct EOD (started 6 Oct, ~8 days left)
- Team: 2 people
- Goal: place 1st/2nd (INR 15k / 10k)
- Submission: Devpost page + public GitHub repo + 3-10 min YouTube demo + team info
- Repo: https://github.com/VARUN3WARE/cardiovision.git (branch main)

Track C (speech analytics) was evaluated and rejected. See ../hackathon_details.md, ../Track A.pdf, ../Track C.pdf in the parent dir. Do not switch tracks.

## 2. Rubric (what judges score)

| Criterion | Weight | Status |
|---|---|---|
| Predictive Performance | 30% | strong baseline done |
| 3D Visualization | 25% | placeholder scene, needs real mesh |
| Clinical Interpretability | 20% | SHAP working |
| System Integration | 15% | backend+frontend built separately, not yet run together in browser |
| Technical Implementation | 10% | clean arch, reproducible |

Biggest remaining win = 3D Visualization (25%). Currently a stylized sphere + tube arteries. Sourcing a real heart .glb and mapping LAD/LCX/RCA to it is the highest-leverage Day 2 task.

## 3. How to run it

Backend:
- cd cardiovision/backend
- source .venv/bin/activate
- python train.py  (already run; retrains 4 models in ~8s, writes models/)
- uvicorn app.main:app --reload --port 8000  (API at :8000, docs at /docs)
- .venv/bin/python test_pipeline.py  (in-process test, no server needed)

Frontend:
- cd cardiovision/frontend
- npm install  (already done, 208 packages)
- npm run dev  (Vite at :5173, proxies API to :8000)
- npm run build  (tsc + vite build, currently passes)

Run both: start uvicorn on :8000, then npm run dev on :5173, open http://localhost:5173. The form has an example high-risk patient preloaded. Click "Predict Risk" to see probabilities + 3D color update + SHAP.

## 4. Dataset facts

- Source: UCI ML Repository id=411, "Extension of Z-Alizadeh Sani dataset"
- Size: 303 rows x 59 columns, no missing values
- Targets (already clean string labels):
  - Cath: CAD (216) / Normal (87) -- overall CAD label, 71.3% prevalence (imbalanced)
  - LAD: Stenotic (177) / Normal (126)
  - LCX: Stenotic (119) / Normal (184)
  - RCA: Stenotic (114) / Normal (189)
- LEAKAGE COLUMNS (never use as inputs): LAD, LCX, RCA, Cath. 55 input features remain.
- Categorical encoding: 21 object columns. Y/N to 1/0; Sex Male/Fmale (typo "Fmale" is in the data); BBB N/LBBB/RBBB; VHD N/mild/Moderate/Severe. Auto-learned in train.py, saved to models/encoding.json.
- The ucimlrepo Python API CANNOT import this dataset (download-only). The xlsx is already in backend/data/. Don't try fetch_ucirepo(id=411).

## 5. Model results (5-fold CV, honest numbers)

| Target | Acc | Precision | Recall | F1 | AUC | Notes |
|---|---|---|---|---|---|---|
| CAD | 0.871 | 0.900 | 0.921 | 0.911 | 0.906 | strong |
| LAD | 0.795 | 0.811 | 0.847 | 0.829 | 0.846 | good |
| LCX | 0.710 | 0.640 | 0.597 | 0.617 | 0.744 | moderate, tunable |
| RCA | 0.657 | 0.551 | 0.474 | 0.509 | 0.678 | weak, hard target on 303 rows |

XGBoost: n_estimators=400, max_depth=4, lr=0.05, subsample=0.8, scale_pos_weight=neg/pos.
CAD/LAD are judge-pleasing strong. LCX/RCA are genuinely hard. Don't overfit the validation chasing RCA. Report CV honestly.
Example patient (preloaded) predicts CAD=1.000, LAD=0.998, LCX=0.697, RCA=0.972. SHAP top contributors: Typical Chest Pain (+1.83), TG, Region RWMA, FBS, EF-TTE, T-inversion, DM, Age.

## 6. CRITICAL gotchas (read before touching anything)

1. shap.TreeExplainer is BROKEN with xgboost 3.x. It throws "could not convert string to float: '[5E-1]'" because xgboost changed base_score storage format. We use XGBoost's native booster.predict(dmatrix, pred_contribs=True) instead. See app/shap_explainer.py. Do NOT revert to shap.TreeExplainer or downgrade xgboost. The native path works and is faster.

2. Circular import trap: app/model.py imports from app/encoders.py, and both need MODELS_DIR. Shared constants live in app/config.py to break the cycle. If you add a module needing paths/constants, import from config.py, not from model.py or encoders.py.

3. Schema to dataset column name mapping: the frontend/pydantic uses camel-ish field names (CurrentSmoker, EF_TTE, RegionRWMA); the dataset uses spaced names (Current Smoker, EF-TTE, Region RWMA). SCHEMA_TO_COLUMN in schemas.py + to_dataset_columns() in main.py handle translation. If you add/rename a feature, update BOTH the schema field AND SCHEMA_TO_COLUMN, then retrain.

4. Sandbox network isolation: in this Cursor environment, each Shell command runs in its own network namespace. A uvicorn server started in one shell is NOT reachable from a later shell's curl (connection refused). To test the HTTP API: (a) run server + curl in the SAME Shell command, or (b) use the in-process test_pipeline.py (no socket), or (c) run the Shell with required_permissions: ["all"] to escape the sandbox. DNS for github.com / pypi / npm is also blocked inside the sandbox -- use required_permissions: ["all"] for any network install or git push.

5. Machine: Dell G15, i5-12500H, 16GB RAM (~5.7GB free), RTX 3050 but NVIDIA driver is broken (nvidia-smi fails). Doesn't matter -- Track A needs no GPU. Disk at 89% (28GB free). The venv pulled in a 351MB nvidia-nccl-cu12 as an xgboost dep (harmless but big).

6. Models are gitignored (*.joblib). The repo ships features.json/encoding.json/metrics.json but NOT the trained weights. A fresh clone must run python train.py first. This is intentional -- document it in the README.

## 7. What's DONE (Day 1)

- Track selection (A over C)
- Repo created, git-managed, pushed to GitHub (3 commits on main)
- Dataset downloaded + EDA (no missing values, targets clean, imbalance noted)
- Categorical encoding pipeline (21 cols, saved map)
- 4 XGBoost classifiers trained, leakage guard, class-imbalance weighting
- 5-fold CV metrics computed and saved
- FastAPI: /health, /features, /predict, /explain
- SHAP via native XGBoost pred_contribs (clinically sensible output)
- In-process pipeline test passing
- Frontend scaffold: React + Vite + TS + R3F + Tailwind, deps installed
- 55-field clinical form wired to real schema, example patient preloaded
- 3D scene (placeholder), dashboard, SHAP panel components
- tsc --noEmit + vite build pass

## 8. What's NEXT (Day 2+), in priority order

### Day 2 -- 3D mesh + integration (highest leverage)
1. Source a real heart .glb/.gltf (CC0 / open license): NIH 3D Print Exchange, BodyParts3D, Sketchfab (filter CC0), Open Anatomy Project. Place in frontend/public/heart.glb, load with useGLTF in HeartScene.tsx.
2. Map LAD/LCX/RCA to the mesh: either named nodes in the glb, or overlay artery tubes (current ArteryTube approach) positioned along the real heart surface. Color each by predicted probability (green to yellow to red; probToColor already exists).
3. Run backend + frontend together in a browser and verify the full loop: form to POST /predict to arteries recolor to SHAP panel renders. (Use required_permissions: ["all"] for the shell, or run uvicorn + vite in one command.)
4. Click-an-artery interaction to highlight + show vessel-specific risk + its SHAP drivers.

### Day 3 -- API wiring polish + SHAP per-vessel
- Currently /explain returns SHAP for the CAD model only. Extend to return SHAP for the selected vessel model (LAD/LCX/RCA) so the explanation panel reflects which artery the user clicked. (Frontend already passes selected; backend needs a per-vessel field.)
- Add a few preset patient buttons (low-risk / high-risk / borderline) for the demo.

### Day 4 -- Model tuning (optional, bounded)
- LCX/RCA are weak. Try: probability calibration (Platt/isotonic), threshold tuning for F1, light hyperparameter search. Timebox to half a day. Report whatever you get honestly.
- Consider a small ensemble (XGB + RandomForest) only if it clearly helps CV AUC.

### Day 5 -- Dashboard + disclaimer polish
- Persistent, visible safety disclaimer (rubric requirement -- currently a banner, make sure it's unmistakable).
- Patient profile panel: show entered measurements alongside their SHAP contribution.
- Loading/error states, responsive layout.

### Day 6 -- Docs + demo video
- 6-page technical doc: dataset preprocessing, model architecture, 3D pipeline, usage, eval results (use models/metrics.json).
- 3-10 min YouTube demo (English audio/subtitles): feature input to 3D interactions to explanation to technical walkthrough.
- Finalize README, push, create Devpost submission page, add all teammates.

### Buffer -- 14 Oct
- Final Devpost submission before EOD. All 4 components: project description, public GitHub repo (w/ README), YouTube video, team info.

## 9. Key files to read first

1. ../Track A.pdf -- the actual problem statement (authoritative)
2. PLAN.md -- day-by-day plan + risk register
3. backend/app/config.py -- constants
4. backend/train.py -- how models are made
5. backend/app/shap_explainer.py -- the native-SHAP workaround (gotcha #1)
6. frontend/src/components/HeartScene.tsx -- where the 3D work happens next
7. backend/models/metrics.json -- current model performance

## 10. TL;DR for Claude

Track A is locked. Backend ML + API + SHAP are done and working (CAD AUC .906). Frontend builds and is wired to the real schema. The single highest-value next step is sourcing a real heart mesh and mapping the 3 arteries to it (25% of the rubric, currently a placeholder). Watch the gotchas in section 6 -- especially the shap/xgboost incompat (use native pred_contribs) and the sandbox network isolation (use required_permissions: ["all"] for installs/push). Don't switch tracks.
