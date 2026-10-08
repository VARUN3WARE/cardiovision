# HANDOFF -- CardioVision (Track A)

Handoff doc for the next agent (Claude) picking up this project.
Last updated: 2026-10-08. Days 1-3 complete: models (7c), usability (7c), realistic 3D heart (7d). Remaining: docs, demo video, Devpost (section 8).

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
| Predictive Performance | 30% | verified blends: AUC CAD .943, LAD .856, LCX .761, RCA .739 (5x5 CV) |
| 3D Visualization | 25% | realistic CC BY heart with fitted coronaries + territories; schematic toggle/fallback |
| Clinical Interpretability | 20% | exact per-model attributions, plain-language panel, calibration bands |
| System Integration | 15% | live updates, validated form, quick mode, error handling, verified in browser |
| Technical Implementation | 10% | harness + experiments + model report, pinned deps, fresh-clone path |

Biggest remaining wins: the 6-page technical doc (most content is in docs/model-report.md), the demo video, Devpost.

## 3. How to run it

Backend:
- cd cardiovision/backend
- source .venv/bin/activate
- python train.py  (~4-5 min: 5x5 CV + nested thresholds, 4 targets in parallel; writes models/)
- python make_presets.py  (example patients from held-out predictions -> frontend/src/presets.json)
- uvicorn app.main:app --reload --port 8000  (API at :8000, docs at /docs)
- .venv/bin/python test_pipeline.py  (in-process test, no server needed)

Frontend:
- cd cardiovision/frontend
- npm install  (already done, 208 packages)
- npm run dev  (Vite at :5173, proxies API to :8000)
- npm run build  (tsc + vite build, currently passes)

Run both: start uvicorn on :8000, then npm run dev on :5173, open http://localhost:5173 (in the Claude desktop app: ../.claude/launch.json defines `api` and `web`). A labelled demo patient is preloaded and estimated automatically (live mode).

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

## 5. Model results (5x5 repeated CV, honest numbers)

| Target | ROC-AUC | Acc@0.5 | F1@0.5 | Nested tuned F1 | Old XGB AUC |
|---|---|---|---|---|---|
| CAD | 0.943 | 0.883 | 0.919 | 0.915 | 0.900 |
| LAD | 0.856 | 0.799 | 0.836 | 0.839 | 0.828 |
| LCX | 0.761 | 0.700 | 0.566 | 0.660 | 0.745 |
| RCA | 0.739 | 0.682 | 0.530 | 0.613 | 0.678 |

Models: per-target log-odds blend of logistic regression on clinical composites + shallow boosted trees
(backend/app/cardio_models.py). Full methods, calibration, literature comparison and limitations:
docs/model-report.md. Don't chase RCA/LCX further without the harness (backend/experiments/harness.py);
split noise is ~0.02 AUC, so compare candidates with paired deltas on identical folds.

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

## 7b. What's DONE (Day 2, 7 Oct)

- **Bug fixes (affected every prediction before):**
  - PLT is stored in thousands in the dataset (25-742). The schema required >= 10000 and the example sent
    250000, wildly out of distribution. Schema now 10-1500 (x10^3/uL); 24 real rows previously failed validation.
  - `DiastolicMurmur` was missing from `SCHEMA_TO_COLUMN`, so it was silently encoded as 0 ("N") for every request.
    `test_pipeline.py` now asserts the schema and feature list map 1:1.
  - RegionRWMA is 0-4 (region), not 0/1; FunctionClass 0-4; binary flags now bounded 0/1; TG max raised to 2000.
- `/explain` returns `explanations: {cad, lad, lcx, rca}` (SHAP per model). `explanation` kept for compatibility.
- `make_presets.py`: picks 4 real patients (normal / isolated LAD / isolated RCA / 3-vessel) using OUT-OF-FOLD
  probabilities -> `frontend/src/presets.json`. The UI shows their angiography ground truth.
- 3D: decided against an external .glb (licensing + no named coronary nodes). `heartAnatomy.ts` builds an
  analytic heart surface P(u, v) so arteries are projected exactly into the grooves. Includes atria, aorta,
  pulmonary trunk + PAs, SVC/IVC, 4 pulmonary veins, LAD+D1/D2, LCX+OM1/OM2, RCA+AM+PDA, left main.
- `HeartScene.tsx`: risk colors, hover/click selection, labels with %, perfusion-territory mode, selected
  vessel's territory highlight, camera swings to the selected vessel, Reset, heartbeat at patient's PR.
- `ExplanationPanel`: tabs Overall/LAD/LCX/RCA (synced with 3D selection), diverging SHAP bars, clinical labels
  + units (`featureLabels.ts`), baseline -> patient probability reconciliation.
- Dashboard: risk level, clickable vessel cards, ground truth for presets. Disclaimer strengthened.
- Running in the Claude desktop app: `../.claude/launch.json` (parent dir) defines `api` (:8000) and `web` (:5173).

Realism pass (also 7 Oct): researched realism cues from cadaveric specimens / atlases and existing models.
Downloadable heart meshes with coronaries exist (Sketchfab "Cardiac Anatomy: coronary arteries", Get3DModels)
but are mostly CC-BY-NC and have no separable LAD/LCX/RCA, so we kept the procedural model and added:
lobular epicardial fat in the grooves (arteries half-embedded), cardiac veins + coronary sinus, tapering
tortuous arteries with more branches, aortic arch branches, crinkled auricles, cut vessel stumps, generated
muscle-fibre normal map, clearcoat + sheen tissue, Lightformer environment, ContactShadows, and
@react-three/postprocessing@2 (N8AO, Bloom, ToneMapping, Vignette; "HD" toggle). Mid-risk color is now amber
(yellow vanished into fat). Clicking heart wall selects the perfusing artery; a `<Bvh>` wrapper keeps raycasts
cheap. First load takes a few seconds (shader compilation) behind a "Building 3D heart model" overlay.

Gotchas found on Day 2:
- R3F only raycasts objects that have handlers, so opaque meshes without handlers do NOT block clicks:
  arteries on the back were clickable through the heart. Myocardium/atria/great vessels now stop propagation.
- @react-three/postprocessing v3 needs R3F 9; we're on R3F 8, so stay on v2.x.
- R3F listens for clicks on the canvas wrapper, so DOM overlays inside `<Html>` must `stopPropagation`
  or `onPointerMissed` immediately clears the selection.
- `useThree().size` can be 0 on the first layout pass; guard before dividing (it NaN'd the camera once).
- `fastapi.testclient` needs httpx2, which isn't installed; test HTTP with a real uvicorn + curl instead.

## 7c. What's DONE (ML + usability, 7 Oct)

- Leakage-safe harness (backend/experiments/harness.py) and 6 experiment strategies, each adversarially
  re-verified on unseen splits; final pick on seeds 400-800 with a pre-declared rule; independent re-check
  on seeds 900-904 beat the old model in every repeat. Production models in app/cardio_models.py.
- train.py: CSV rebuilt from the committed UCI zip on a fresh clone; parallel per-target training; learned
  thresholds (F1 / rule-out / balanced); calibration bands; feature_info.json (ranges, defaults, influence).
- API: /metrics, richer /features, /health; responses carry threshold, band + observed rate, imputed fields,
  range warnings, model version, disclaimer. Schema: only Age/Sex required; unrecorded findings default to
  absent (NOT the cohort mode: typical angina/HTN are majority-positive in this referral cohort).
- UI: field spec, quick (19) / full (55) mode, inline validation, live updates with stale detection,
  structured API errors + status pill, honest dashboard (bands, AUC, held-out vs angiography for presets),
  plain-language explanations, error boundaries, responsive layout.
- Finding for the write-up: the vessel models are not very vessel-specific (no isolated-RCA patient is singled
  out by held-out predictions; LAD is over-called), shown honestly in the RCA preset.

## 7d. What's DONE (realistic 3D heart, 8 Oct)

- frontend/public/models/heart.glb: "Realistic Human Heart" by neshallads, CC BY 4.0 (credited in the UI, README
  and public/models/ATTRIBUTION.md). Single mesh, ~12k verts, Y-up anatomical anterior pose, ~0.5 scale after its
  node transforms; texture metalness channel is 0 (metalness forced to 0 anyway).
- Coronary tree traced on the model's sculpted grooves (surface picks, model coordinates) in
  specimenAnatomy.ts; at load: BVH closest-point snapping + normal lift + smoothing. Territories by nearest
  artery tree (soft-min), masked below an AV-groove height ring. Camera views from territory centroids.
- SpecimenHeart.tsx renders it (accelerated raycast, translucent RGBA overlay for territories, clickable wall,
  pins lifted off the surface, heartbeat). HeartScene: Realistic/Schematic toggle; HEAD probe rejects the dev
  server's HTML fallback; ErrorBoundary inside the Canvas falls back to the schematic heart.
- Gotchas: pins/labels exactly on the surface get hidden by Html occlusion (lift them); vite.config dedupes
  three (three-mesh-bvh otherwise pulled a second copy); the model's low roughness glitters under bloom
  (roughness scaled x1.45).

## 7e. What's DONE (BodyParts3D atlas heart + chest, 8 Oct; plan items B1-B3)

The approved plan is docs/plan-3d-and-explanations.md. Work so far covers its BodyParts3D workstream.
- Assets: scripts/bp3d/make_parts.py writes parts.json from the official index files (FMA concept -> FJ files;
  LAD = FMA3862 + septals FMA3892, LCX = FMA3895, RCA = FMA50039 + PDA septals FMA3845, LM = FMA3855).
  scripts/bp3d/fetch_bp3d.py range-fetches only those 181 OBJ members (CRC32-checked) into the gitignored
  scripts/bp3d/cache/ (the archive server is slow: ~10 min). frontend/scripts/build-bp3d.mjs converts
  (x, z, -y) mm -> m, welds, drops fragments <2%, simplifies (meshoptimizer), merges one mesh per part with a
  per-vertex `_BRANCH` index (node extras list each branch's FJ id + FMA id + name), bakes `_VDIST` on the
  ventricular wall (0.1 mm distance to LAD/LCX/RCA trees and to the groove vessels), meshopt-compresses.
  `npm run assets:bp3d` does both. Output: public/models/bp3d_heart.glb (72k tris, 0.39 MB) and
  bp3d_thorax.glb (74k tris, 0.35 MB); derived files CC BY-SA 4.0 (see ATTRIBUTION.md for why).
- BP3D v4 has no lung surfaces (only bronchial/vascular trees), so the chest is skin (cropped z 1000-1450 mm),
  ribs, costal cartilage, sternum, T1-T12, clavicles, scapulae, diaphragm, trachea + left main bronchus.
- Runtime: atlasAnatomy.ts (dequantise + bake transforms, centre on ventricles+atria, territories =
  softmin over baked distances, fat from groove distance, views from territory centroids, label anchors on
  the trunks), atlasMaterials.ts (onBeforeCompile: simplex-noise derivative bump for UV-less tissue,
  camera-facing cylinder cutaway through the chest, fresnel ghosting), AtlasHeart.tsx (wall with vertex
  colours, tissue, veins, one eased-colour material per coronary tree, branch picking + hover read-outs,
  X-ray, Thorax with layer toggles; Lambert shading for the chest to keep it cheap).
- HeartScene: Realistic / Atlas / Schematic toggle (Realistic is still the default; plan gate G2 decides),
  Chest (+ layer menu) and X-ray in Atlas mode, A/P/L/R/S/I view buttons, chest-to-heart fly-in (skipped
  with prefers-reduced-motion), "Clicked structure" read-out, territory caveat caption, BodyParts3D credit.
  Dev-only `window.__cv` exposes the R3F state for scripted checks (not in production builds).
- Gotchas: rAF is throttled while the browser pane is hidden, so FPS measured there is meaningless (measure
  with the page visible); the coronary OBJs are open tubes (DoubleSide); the cutaway assumes the heart sits
  at the world origin.

## 8. What's NEXT, in priority order

Done on 8 Oct: physiological breakdown (requirement 3c), technical documentation
(docs/CardioVision-Technical-Documentation.pdf, 5 of 6 allowed pages; rebuild with docs/build_pdf.sh after
docs/make_figures.py), demo script (docs/demo-script.md), Devpost text (docs/devpost.md).

1. Record the demo video from docs/demo-script.md (~7 min, 1080p), upload to YouTube as unlisted, add
   English subtitles.
2. Devpost: paste docs/devpost.md, add the GitHub link, YouTube link and PDF, and add every teammate with
   real full names (each needs a Devpost account). Team names are not in the docs yet; add them to the PDF
   title block if wanted (docs/technical-documentation.md front matter) and rebuild.
3. Final push before 14 Oct EOD.
4. Optional: the multi-lens code review of the 7 Oct ML/UI changes was interrupted before reporting
   (workflow run wf_a4f512a2-65c); printable patient report export.

## 9. Key files to read first

1. ../Track A.pdf -- the actual problem statement (authoritative)
2. PLAN.md -- day-by-day plan + risk register
3. backend/app/config.py -- constants
4. backend/train.py -- how models are made
5. backend/app/shap_explainer.py -- the native-SHAP workaround (gotcha #1)
6. backend/app/cardio_models.py -- production models + exact attributions
7. docs/model-report.md -- results, calibration, limitations
8. frontend/src/components/specimenAnatomy.ts + SpecimenHeart.tsx -- realistic heart integration;
   atlasAnatomy.ts + AtlasHeart.tsx + atlasMaterials.ts -- BodyParts3D atlas heart and chest
9. backend/models/metrics.json -- current model performance

## 10. TL;DR for Claude

Track A is locked. Models are verified log-odds blends (AUC .943/.856/.761/.739) with exact attributions and
honest calibration; the UI is a validated, live-updating clinical form with plain-language explanations; the
3D view is a CC BY realistic heart with our coronary tree fitted into its grooves, plus a BodyParts3D atlas heart
in a ghosted chest whose 59 named coronary branches are coloured by their vessel's model (schematic heart as
toggle and fallback). The approved week plan is docs/plan-3d-and-explanations.md. Docs done (5-page PDF, demo script, Devpost text). Commit + push to main after every verified
change. Remaining: record the video, fill Devpost, team names.
Gotchas in sections 6, 7b and 7d. Don't switch tracks.
