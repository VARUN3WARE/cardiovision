# 8-Day Build Plan — Track A (Cardiovascular)

**Team**: 2 people · **Window**: 6 Oct → 14 Oct EOD · **Goal**: place 1st/2nd

Person 1 = **ML/Backend** (M) · Person 2 = **3D/Frontend** (F)

## Day-by-day

### Day 1 — Tue 6 Oct (tonight)
- **M**: Download Z-Alizadeh Sani extension (UCI/Kaggle) → `backend/data/`. EDA notebook: class balance, missingness, leakage columns (`LAD`,`LCX`,`RCA`,`Cath`). Define feature list + 4 targets.
- **F**: Vite + React + TS scaffold. Install R3F/drei/Tailwind. Find a CC0 heart `.glb` (NIH 3D Print Exchange / Sketchfab), load with `useGLTF`, render static scene with OrbitControls.
- **Milestone**: heart visible in browser; dataset understood.

### Day 2 — Wed 7 Oct
- **M**: Baseline classifiers (XGBoost/GBM) for CAD + LAD + LCX + RCA. 5-fold CV, report Acc/Prec/Recall/F1/AUC. Save to `backend/models/`.
- **F**: Identify LAD/LCX/RCA mesh nodes (name them / tag with userData). Wire a mock prop `arteryProbs: {lad, lcx, rca}` → artery color (green→yellow→red gradient).

### Day 3 — Thu 8 Oct
- **M**: FastAPI `/predict` endpoint returning `{cad_prob, lad_prob, lcx_prob, rca_prob}`. Pydantic schema matching dataset features.
- **F**: `FeatureInput.tsx` form — clinical fields (age, BP, pulse, cholesterol, ST elevation, T inversion, LVH, RWMA region, etc.). POST to `/predict`. Show probabilities in `ClinicalDashboard.tsx`.

### Day 4 — Fri 9 Oct
- **M**: SHAP explainer (`shap.TreeExplainer`). `/explain` endpoint returns per-feature SHAP values + base value.
- **F**: `ExplanationPanel.tsx` — SHAP waterfall/bar chart (use a lib or hand-roll). Show top contributing features with physiological labels.

### Day 5 — Sat 10 Oct
- **Both**: Integration pass. Real-time artery color update on predict. Click an artery → highlight + show vessel-specific risk + its SHAP drivers. Rotate/zoom/select polish.
- **M**: Hyperparameter tuning, calibrate probabilities (Platt/isotonic), cross-val report table for docs.

### Day 6 — Sun 11 Oct
- **F**: Clinical dashboard layout — patient profile panel, measurements alongside contribution, safety disclaimer banner (visible, persistent).
- **M**: Robustness — handle missing fields, input validation, edge cases. Add a few example patients (preloaded buttons).

### Day 7 — Mon 12 Oct
- **Both**: End-to-end QA. Performance pass (3D stays responsive, no GPU). Write the 6-page technical doc (dataset preprocessing, model arch, 3D pipeline, usage, eval results).
- **F**: UI polish, loading states, error states, responsive layout.

### Day 8 — Tue 13 Oct
- **Both**: Record 3–10 min demo video (English audio/subtitles): feature input workflow → 3D viz interactions → explanation panel → technical walkthrough.
- Finalize README, push to public GitHub, create Devpost submission page.

### Buffer — Wed 14 Oct
- Final submission on Devpost before EOD. All 4 components: project description, public GitHub repo (w/ README), YouTube demo video, team info.

## Risk register
| Risk | Mitigation |
|---|---|
| Heart mesh has no named artery nodes | Use a separate artery-line mesh or overlay tubes via R3F; worst case color a proxy region |
| 303 rows → overfitting | tree ensembles + CV + calibrate; report CV not train acc |
| SHAP slow on every request | precompute expected values; cache explainer |
| 3D not responsive in browser | low-poly mesh, instancing, cap DPR |
| Dataset licensing | UCI/Kaggle are public for research; cite in README |

## Division of labor (clean split)
- **M owns**: `backend/`, `train.py`, models, SHAP, API, eval tables, doc §1–3,5
- **F owns**: `frontend/`, 3D scene, dashboard, explanation viz, doc §4, demo video edit
- **Shared**: integration, README, Devpost page, final QA
