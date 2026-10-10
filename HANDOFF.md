# HANDOFF -- CardioVision (Track A)

Handoff doc for the next agent (Claude) picking up this project.
Last updated: 2026-10-10. Done: models (7c), usability (7c), realistic heart (7d), BodyParts3D atlas (7e), 3D encoding and inspection (7f), medical audit (7g), function twin (7h) and the standout layer T1-T6, story mode included (7i). Remaining: final regression, demo video, Devpost (section 8).

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
| Predictive Performance | 30% | verified blends: AUC CAD .943, LAD .856, LCX .761, RCA .739 (5x5 CV); independent re-run (docs/model-verification.md): no leakage, fresh-split AUC .939/.859/.745/.754 with bootstrap CIs |
| 3D Visualization | 25% | Realistic, BodyParts3D Atlas (in the chest) and Schematic hearts; function twin + ECG, reasoning streams, ischaemia lab, 3D cohort map, story mode |
| Clinical Interpretability | 20% | exact per-model attributions, plain-language panel, calibration bands, evidence by data type, what-if sensitivity, ischaemic cascade, similar patients |
| System Integration | 15% | live updates, validated form, quick mode, error handling; what-if re-runs the models while dragging; the twin follows the patient's recorded values |
| Technical Implementation | 10% | harness + experiments + model report + verification script, deterministic generators, 111 vitest tests, pinned deps, fresh-clone path |

Biggest remaining wins: the demo video, Devpost.

## 3. How to run it

Backend:
- cd cardiovision/backend
- source .venv/bin/activate
- python train.py  (~4-5 min: 5x5 CV + nested thresholds, 4 targets in parallel; writes models/)
- python make_presets.py  (example patients from held-out predictions -> frontend/src/presets.json)
- python make_cohort.py  (Similar patients data -> frontend/src/cohort/cohort.json; a few seconds, byte-identical
  reruns; re-run after ANY retrain, then npm test)
- uvicorn app.main:app --reload --port 8000  (API at :8000, docs at /docs)
- .venv/bin/python test_pipeline.py  (in-process test, no server needed)
- .venv/bin/python verify_models.py  (independent verification, ~2-5 min -> docs/model-verification.md and
  models/verification.json; no server needed)

Frontend:
- cd cardiovision/frontend
- npm install  (already done)
- npm run dev  (Vite at :5173, proxies API to :8000)
- npm test  (vitest: twin, coronary physiology, lab settings, reasoning groups, what-if, cohort kNN, story script)
- npm run build  (tsc + vite build, currently passes)
- Before every commit: npx tsc -b && npm test && npm run build

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

## 7f. What's DONE (3D encoding and inspection, 8 Oct; plan items E1-E7)

- E1: risk.ts BAND_COLOR + configureBands(/metrics) + vesselBand; heartAnatomy.riskColor is now the band
  colour (soft 1.5-point edges) and `undecided(p)` drives world-space stripes (atlasMaterials `stripes`
  patch) on every artery; cards use the same hex; labels "LAD · 70% chance"; left main neutral;
  ArteryTube.useVesselMaterial is the one shared, eased vessel material; App keeps the last estimate while
  typing (HeartScene `pending` chip) so arteries never flash grey.
- E2: MapSemantics caption + 6 rules + wall-motion chip (never drawn on the heart).
- E3: frontend/src/anatomy/registry.ts (VESSELS, UNMODELLED, MODEL_TABS, SEGMENT_VESSEL) drives every view;
  atlas GLB distance attributes are per artery (_D_LAD/_D_LCX/_D_RCA/_D_GROOVE).
- E4: dpr [1, 1.5], ContactShadows frames=1 keyed on model+ready, PerformanceMonitor turns HD off below
  ~24 fps; frame-time table in README.
- E5: VesselInspector.tsx (band, observed rate, screening threshold, AUC + "read independently", top 3
  drivers, territory + AHA segments, what is not shown; Esc closes).
- E6: Bullseye.tsx in the dashboard; hover highlights the artery and tints its territory in all 3 models.
- E7: heart.glb textures -> WebP (scripts/compress-heart.mjs): 7.55 -> 1.60 MB, visually unchanged.
- Backend fix (256332f): display_attributions overwrote BBB instead of adding the tree member's share;
  /explain returned 500 for the Low-risk preset. test_pipeline.py now checks N/LBBB/RBBB + all presets.

## 7g. Medical-statement audit (9 Oct)

Every clinical statement in the UI, API descriptions, presets and demo script was checked. Verified correct:
Diamond-Forrester values (= ACC/AHA 2002 Table 4; asymptomatic row from Diamond & Forrester 1979), CKD-EPI
2021 eGFR, AIP unit conversions, ATP III age risk factor, ADA FBS >= 126, Cerqueira 2002 AHA segment map,
vessel territories, anatomy-pin notes, CAD = >=50% narrowing, 216/303 = 71% CAD. Corrected: LV pin omitted
the RCA's inferior wall; "Obesity" is the dataset's BMI >= 25 flag (WHO: overweight); "Reduced EF (<50%)"
-> "EF below 50%"; "NYHA class" -> "Functional class" (dataset term, NYHA assumed); "Wall-motion score" ->
"Regional wall-motion abnormality" grade 0-4; summary sentence now uses the card's band and attribution
wording ("driven mainly by", not "because of"); angiography verdicts say "<50% / >=50% narrowing" (not
"normal"); inspector states the threshold's measured sensitivity/specificity; "premature" removed from FH;
demo script no longer says the LCX card is dashed (AUC 0.761 >= 0.75).

## 7h. Function twin (T1 of docs/plan-twin-and-simulation.md, 9 Oct)

- frontend/src/twin/twin.ts: `twinFromResult` reads the patient's pulse, EF, BP and ECG findings from the CAD
  explanation (values as the model saw them; imputed ones flagged), `beatTiming` (PR 0.16 s, QRS 0.09/0.14 s,
  Bazett QT), a shared `heartClock` advanced at 500 Hz ticks by <TwinClock> inside the Canvas, `mechanics`
  (atrial kick after P, ventricular contraction from QRS+0.04 s to end of T scaled by EF, aortic distension by
  BP, LBBB apical rocking) and `ecgSample` (sum-of-Gaussians leads II and V1 with ST/T, Q, LVH, LBBB, RBBB).
- useHeartMotion drives all three hearts (Atlas: ventricles, atria and great vessels each pivot about their own
  centre). twin/EcgMonitor.tsx is a sweeping two-lead monitor under the 3D view (toggle "ECG").
- Deep links: ?patient=low|lad|rca|triple|uncertain and ?heart=atlas|realistic|schematic.
- Hidden browser pane = throttled rAF, so check visuals with headless Chrome:
  google-chrome --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --window-size=1440,900
  --virtual-time-budget=20000 --screenshot=out.png "http://localhost:5173/?patient=low&heart=atlas"
  (fine for a quick look; for 3D label alignment use the CDP capture in 7i)

## 7i. Standout layer (10 Oct; docs/plan-twin-and-simulation.md T1-T6 done)

T1-T5 were each reviewed independently and all their review fixes are committed (the ischaemia lab's last, in
6d80f98). Deep links append to http://localhost:5173/ and combine with ?patient=... and ?heart=... (7h).

**T1 Function twin + ECG** (3538b91, tests 176ba8d): see 7h. Since 089958e HeartScene also takes
`recorded?: ExplainResponse | null` (App passes the patient's own result next to `result={shown}`). The heartbeat,
ECG monitor, the lab's starting values and the wall-motion note use `recorded ?? result`, so a what-if scenario
only changes colours, labels and probabilities; a chip says "What-if: colours show the scenario; heartbeat and ECG
show recorded values".

**T3 Reasoning flow** (abf0039, review fixes 089958e)
- Files: frontend/src/reasoning/dataGroups.ts (+ test, fixture __fixtures__/presetExplanations.json),
  ReasoningFlow.tsx, anchors.tsx (each heart registers per-artery label anchors and a ventricles+atria hull),
  hoverStore.ts, EvidenceStrip.tsx, featureInfo.ts (HeartScene fetches /features itself); toolbar "Reasoning".
- Deep link: ?reasoning=1, optionally &model=lad|lcx|rca (selects that artery); ignored when ?lab= is present.
- Decisions: six groups (History & demographics, Symptoms & exam, ECG, Echo, Labs, Pre-test score) with a stated
  rule for derived composites; base + group sums = model log-odds exactly (vitest). Towers sit around the heart,
  never on it. Overall CAD streams end on a camera-facing halo fitted to the projected heart outline, with the
  "Overall CAD xx%" badge outside it (the first version's single target point read as "the LAD"); vessel models
  end at the artery label. Badge and caption use the explanation's own log-odds; streams dim while pending or
  while a what-if drag runs ahead of its explanation. Reduced motion freezes the particles.
- Layout is measured, not hard-coded: legend, view buttons, credit, toolbar bottom and notices are measured
  rectangles; towers avoid the heart outline and artery labels and clamp inside the canvas; below 520 px (or when
  they do not fit) towers become one-line chips in rows above and below the heart; the legend shrinks to the band
  scale while Reasoning is on. Notices stack under the toolbar in one column: pending, what-if chip, reasoning
  caption, vessel chip ("details" closes Reasoning and opens the inspector). CameraRig runs at useFrame priority
  -0.75 and ReasoningFlow calls updateMatrixWorld() first (no one-frame lag). Reasoning and Lab are mutually
  exclusive; each hides the chest and restores it on exit.

**T5 What-if** (ff40c77, review fixes 78a1976; verification weight row 7ba62cf)
- Files: frontend/src/whatif/scenario.ts (pure and tested: levers, contrasts, reducer, normaliseScenario,
  scenarioView, scenarioRequest, singleFlight, throttle, deep link), useScenario.ts, ScenarioStrip.tsx;
  components/WhatIfPanel.tsx; App.tsx; api/client.ts predict(); ClinicalDashboard.tsx notice.
- Deep link: ?whatif=1 (required) &bp= &fbs= &tg= (clamped to the slider range once the patient loads) &quit=1
  (current smokers only) &contrast=angina,st,rwma,ef. Showcase: ?patient=rca&whatif=1&contrast=angina (CAD 90% ->
  40%, LAD 33% -> 9%, RCA 53% -> 30%, checked against the API).
- Decisions: the form is never edited, and a scenario starts from the request body that produced the shown result
  (resultBody, stored in run()), never from unsubmitted or invalid edits. Dragging sends POST /predict throttled to
  120 ms with one request in flight and the newest waiting (singleFlight), then one /explain on release. Sliders:
  systolic BP, fasting glucose, triglycerides (cohort p5-p95, widened to include the recorded value; outside p5-p95
  dashed). LDL, HDL and weight are locked with notes quoting docs/model-verification.md section 7 (no learned
  benefit; weight even inverse). Every preset records RWMA 0 and EF 55, so finding switches work both ways
  ("Present -> if absent/normal", "Absent -> if present (hypothetical)"). The indicator is a one-line strip ABOVE
  the 3D view, not an overlay; the strip, dashboard notice and Why panel say what changed and "model sensitivity,
  not treatment effect". Views switch to the scenario only when its reply for these values arrives (Calculating /
  error with Retry; never a what-if label on the patient's numbers).

**T2 Ischaemia lab** (eea1b71, wording a1cca8a, review fixes 6d80f98)
- Files: frontend/src/twin/coronaryPhysiology.ts (+ test reproducing the science-spec tables), lab/labState.ts
  (+ test: deep link, starting values), lab/LabPanel.tsx, lab/FlowReserveCurve.tsx; changes in HeartScene,
  AtlasHeart (pulses, SandboxMarker), atlasAnatomy (distance from the ostium by shortest paths over each tree,
  computed on first open), atlasMaterials (pulse stripe, territory tint, hypokinesis), twin.ts, EcgMonitor ("lab"
  chips).
- Deep link: ?lab=lad|lcx|rca&narrowing=0..95&exercise=0..1 (opens Atlas; values clamped; unknown artery -> first
  registry artery; lab=0 or off ignored). Demo: ?patient=triple&lab=lad&narrowing=0, drag to 90, exercise to peak
  (at 90% and peak: maximal flow 7%, resting 26%, demand 23,260 vs supply 2,660, 89% short, full cascade).
- Decisions: the closing line deliberately differs from the plan. By mean |attribution| (models/feature_info.json
  influence) typical chest pain ranks 1st and Region RWMA 3rd of 63 inputs in the CAD model, but ST depression only
  24th (37th of 68 in the LAD model), so the panel does not call ST depression one of the strongest inputs. Cascade
  order from Nesto 1987, thresholds (5/15/25/40% of demand unmet) illustrative. Exercise: heart rate to 85% of
  220 - age, systolic BP +50 (at most 210), never below rest, starting from the patient's recorded pulse, BP and
  age as the model saw them (named; the cohort median when not entered; dataset medians 70/130/58 before any
  estimate; never a what-if scenario's). Markers 10/12/14 mm along the LAD/LCX/RCA
  trunks. The lab's ST depression appears on the ECG only as a "lab" chip (not readable in LBBB) and is never
  mapped to the artery. Lab on: Atlas, chest off (restored on exit), atria and great arteries see-through, risk
  colours and inspector replaced by the sandbox; clicking an artery or wall moves the narrowing; Realistic or
  Schematic closes the lab; lab off restores the twin exactly.
- Review findings, fixed in 6d80f98: the SANDBOX band, ring and label show and hide together and hide only when the
  ventricular wall is more than 3 vessel radii in front of the marker for several frames (own BVH ray test), so no
  flicker when the lab opens from the toolbar; the banner wraps inside the free area and the heart is fitted to the
  space the panel, banner and legend leave (no clipping at 1366 px); narrow views get a bottom sheet capped at ~40%
  that leads with the territory bar and the cascade; the closing line names the overall CAD model and inputs
  recorded at rest (ST depression lasts only while the ischaemia does); the reserve hint explains 4 at a resting
  pulse x BP of 9,100 and gives the patient's own value (e.g. 3.5 at 10,400); first open prepares the lab in
  idle-time steps with its shaders compiled ahead (no pause).

**T4 Similar patients + cohort map** (4120ac2, review fixes eae5f5b)
- Files: backend/make_cohort.py (offline, deterministic) -> frontend/src/cohort/cohort.json (~80 KB, fetched
  lazily); cohort/knn.ts (+ knn.test.ts, fixtures __fixtures__/cadExplanations.json incl. Quick-mode replies),
  CohortPanel.tsx, CohortGalaxy.tsx, palette.ts; App.tsx card below the risk estimate.
- Deep link: ?cohort=1 (opens the card and scrolls to it), e.g. ?patient=triple&cohort=1 (recognised as dataset
  patient #173; 10 of 10 similar patients had CAD; LAD 5, LCX 6, RCA 5).
- Decisions: the 55 encoded inputs (VHD by severity) z-scored and weighted by the CAD model's mean |attribution|,
  no derived composites (no double counting). The browser kNN reproduces make_cohort.py exactly (review: 0
  mismatches on 703 queries). The 3D map is t-SNE (trustworthiness 0.974 vs PCA 0.901; PCA fallback) and keeps ~65%
  of true 10-neighbourhoods, so the list uses the full distance and the caption says so. Inputs not entered
  (result.imputed mapped through cohort.json schema_to_column) are left out of the distance and of the identity
  test. Identity guard: same values on every compared input (0.02 SD), or this record with up to 3 inputs changed;
  the agreeing inputs must carry >= 20 bits (32 for a changed record) and no other record may qualify. "No close
  matches" when the 10th neighbour is farther than for 95% of dataset patients. The card always uses the recorded
  result, never the what-if. Text: CAD-weighted similarity, so artery counts say little about which artery; label
  inconsistency (#93) flagged. Colours: magenta = CAD, blue = no CAD on angiography (not band colours).
- Gotchas: re-run make_cohort.py after any retrain (weights and held-out estimates come from feature_info.json and
  models/oof_predictions.npz), then npm test (knn.test.ts checks the browser finds the Python neighbours); recapture
  the fixtures if the presets change. A failed dynamic import is cached by the browser, hence the fetch with retry.

Also on 10 Oct: d36a974 shared obesity_flag() with a 1e-9 tolerance (64 kg / 1.60 m gave BMI 24.999...); 67790c7
"not entered" wording everywhere (findings count as absent, measurements take the cohort median).

**T6 Story mode** (fea1a35, screen-reader captions 23d6f4c, review fixes 59587f8)
- Files: frontend/src/story/storySteps.ts (+ test: script, deep link, topArtery, labDemo, captions), StoryOverlay.tsx
  (bar, caption, closing card), useStoryClock.ts, verification.ts (+ test: the AUC CIs, 303 patients and 71% CAD
  checked against backend/models/verification.json); HeartScene (toolbar "▶ Story", step views, snapshot/restore);
  public/favicon.svg.
- Deep links: ?story=1 (plays from the start), ?story=1&step=N (step N as it ends, paused: for screenshots),
  &play=1 (plays from the start of step N). The story starts once the Atlas heart is known to be available, e.g.
  ?patient=triple&story=1&step=4.
- Steps (Atlas heart, 92 s): 1 A heart in the chest (fly-in, 12 s); 2 The function twin (caption with the recorded
  pulse, EF, BP and ECG findings, 15 s); 3 What moved the estimate (reasoning for overall CAD, 20 s); 4 One estimate
  per artery (inspector on the artery in the highest band, ties by probability, "independent, not a culprit
  ranking", 15 s); 5 Ischaemia lab (sandbox) on that artery (narrowing 0 -> 85%, then exercise rest -> peak, 18 s);
  6 What this is, and is not (reference anatomy; single-centre referral cohort, 303 patients, 71% CAD; ROC AUC from
  /metrics with the bootstrap 95% CIs; associations not causes, lab not the patient; decision support only; 12 s).
  The plan's order changed: the inspector comes before the lab, so the lab demonstrates on the inspected artery.
- Decisions: captions use only the result, the twin's recorded values, the lab model, /metrics and the verification
  copy; with a what-if on, the colours, streams and captions say "(what-if scenario)" while the twin keeps the
  recorded values. Play/pause, previous/next, progress dots, arrow keys, Esc; Replay at the end. The clock waits
  while a heart is being built. Exit restores heart model, view mode, chest and layers, X-ray, pins, heartbeat, ECG,
  lab and its settings, reasoning, selection and inspector, and the exact camera position; the story never changes
  the patient's inputs, estimates or a what-if scenario. Reduced motion: no flights or camera swings, static steps.
  Narrow views: two-row bar, captions clamped to three lines until tapped, legend collapsed, closing card in the
  legend's place. Verified on 10 Oct: ?patient=triple&story=1 plays step 1, Esc returns to the Realistic heart the
  tab opened with, console clean.
- Independent review (read-only, 10 Oct; 12 starting states x 3 exits restored exactly, every caption number matched
  the app): 2 major + 7 minor findings, all fixed in 59587f8. The lab step's caption said "resting flow holds" at 85%
  narrowing, but the panel showed resting flow already short for 4 of 5 presets (resting flow starts to fall at ~84%
  even at the reference pulse x BP of 9,100, earlier at higher resting demand): the caption now reports this heart's
  rest flow, maximal flow and unmet share with the panel's rounding (unit test at 85% and rest). The 3-line clamp on
  narrow views hid the honesty qualifiers: every caption now leads with its qualifier (tested: within the first 90
  characters). Also fixed: "(highest band)" only when the artery is alone in its band (else "highest chance; all
  three are in the higher band" or "all three read <5% in the lower band"); stale states named on every step
  ("previous estimate: inputs changed", "updating"); Pause (waiting) while the heart loads and the fly-in waits for
  play; arrow keys stop at the ends; deep links wait for the camera so Exit restores it; Replay only at the end; the
  progress line follows the story clock; focus moves to the controls on start and back to Story on exit; the closing
  card is announced, leads with the decision-support line and keeps its buttons in view; the flow curve's
  screen-reader label gives this heart's resting-flow onset.

**Working conventions and gotchas**
- Lanes (two agents, one checkout): lane A owns HeartScene, AtlasHeart, SpecimenHeart, atlasAnatomy, atlasMaterials,
  ArteryTube, ExplanationPanel, VesselInspector, twin/, reasoning/, lab/, story/; lane B owns App.tsx, api/client.ts,
  ClinicalDashboard, WhatIfPanel, whatif/, cohort/ and backend offline scripts. Commit only your own paths
  (git commit -m ... -- <paths>); never git add -A.
- Checks before every commit: (cd frontend && npx tsc -b && npm test && npm run build); 111 tests in 8 files at
  59587f8. Never hard-code ["lad","lcx","rca"]: use anatomy/registry.ts (VESSELS, VESSEL_IDS, VESSEL).
- Toolbar: 2 rows at 1440 px, 3 at 1024, 2 at 390; on narrow views X-ray, HD, Reset and the A/P/L/R/S/I views move
  into the "⋯" menu.
- Dev-only globals for scripted checks: window.__cv (R3F state), window.__reasoning, window.__anchors.
- Accurate headless screenshots: the plain `--screenshot --window-size=W,H` recipe loads the page at a shorter
  viewport and stretches the canvas at capture, so drei Html labels drift 17-60 px off their 3D anchors. For
  anything with 3D labels use a CDP capture: Chrome with --remote-debugging-port and the SwiftShader flags,
  Emulation.setDeviceMetricsOverride (width, height, deviceScaleFactor 1) BEFORE navigating, wait ~15 s, then
  Page.captureScreenshot (the 10 Oct session used a small shot.py doing exactly this).
- Headless-only artefacts: in SwiftShader, ?heart=schematic deep links lose the first /explain (net::ERR_ABORTED;
  the app says it cannot reach the API); a real browser is fine (explain 200 in ~160 ms). Click "Estimate risk"
  after load or switch hearts. Under load, some headless runs also see the API briefly offline.
- Vite's watcher occasionally serves a stale module after rapid successive edits; touching the file again fixes it.
- Frame times were not re-measured on the integrated GPU after this layer (the README table predates it).

## 8. What's NEXT, in priority order

Done on 8 Oct: physiological breakdown (requirement 3c), technical documentation
(docs/CardioVision-Technical-Documentation.pdf, 5 of 6 allowed pages; rebuild with docs/build_pdf.sh after
docs/make_figures.py), demo script, Devpost text. Done on 10 Oct: T1-T6 and the review fixes of T1-T5 (7i); demo
script, Devpost text, README and plan updated for the standout layer, story mode included.

1. Story mode (T6): done (fea1a35, 23d6f4c; review fixes 59587f8); README bullet and deep links, docs/demo-script.md Scene 9 (4:20-4:45)
   and its tab, docs/devpost.md "What it does", docs/plan-twin-and-simulation.md T6 and 7i are updated.
2. Ischaemia-lab review fixes: done (6d80f98). Next: a final regression of every deep link at 1440x900 and
   1024x768 with the CDP capture (5 presets x 3 hearts with labels matching the cards; ?reasoning, ?whatif, ?lab,
   ?cohort, ?story=1&step=1..6), console clean, then npm test, npm run build, test_pipeline.py and
   verify_models.py (numbers unchanged).
3. Technical PDF refresh: done on 10 Oct (6 of 6 allowed pages, about 6 lines free on page 6). Section 7 covers the
   twin and simulation layer, Section 4 the independent verification (table: fresh-split AUC with CIs, baselines,
   calibration slopes, AUC within CAD patients; leakage, vessel specificity, clinical direction), Section 5 the
   atlas heart, and Section 1 the anatomy registry and the measured latency (/explain median 140 ms, p90 147;
   /predict 40 ms). New figures from CDP captures: figures/app-overview.jpg (atlas heart in the chest, patient
   #173) and twin-reasoning.jpg, twin-lab.jpg, twin-story.jpg; architecture.png redrawn by docs/make_figures.py.
   Rebuild with docs/build_pdf.sh; any addition needs a matching cut.
4. Re-time 3D frames on the integrated GPU with Reasoning and Lab on; update the README table.
5. Decide on the plan's "model facts tab" (listed as never-cut, not built): in the app the cards and inspector
   show cross-validated AUCs, while the verification (CIs, permutation, baselines, vessel specificity) is only in
   docs/model-verification.md, which the demo script shows directly.
6. Record the demo video from docs/demo-script.md (5-6 min, 1080p), upload to YouTube as unlisted, add English
   subtitles.
7. Devpost: paste docs/devpost.md, add the GitHub link, YouTube link and PDF, and add every teammate with real
   full names (each needs a Devpost account). Team names are not in the docs yet; add them to the PDF title block
   if wanted (docs/technical-documentation.md front matter) and rebuild.
8. Fresh-clone test (train.py, make_presets.py, make_cohort.py, test_pipeline.py, npm test, npm run build), then
   the final push before 14 Oct EOD.
9. Optional: printable patient report export; click a similar patient to load it. (Done: App passes `info` to
   HeartScene, so the 3D view no longer makes its own /features request, d2eda6e.)

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
9. backend/models/metrics.json -- current model performance; docs/model-verification.md -- independent checks
10. docs/plan-twin-and-simulation.md + section 7i -- the standout layer; frontend/src/anatomy/registry.ts, then
    frontend/src/twin, reasoning, lab, whatif, cohort, story

## 10. TL;DR for Claude

Track A is locked. Models are verified log-odds blends (AUC .943/.856/.761/.739; independent re-run: no leakage,
fresh-split AUC .939/.859/.745/.754, bootstrap CIs, vessel models mostly measure how much CAD) with exact
attributions and honest calibration; the UI is a validated, live-updating clinical form with plain-language
explanations; the 3D view is a CC BY realistic heart with our coronary tree fitted into its grooves, plus a
BodyParts3D atlas heart in a ghosted chest whose 59 named coronary branches are coloured by their vessel's model
(schematic heart as toggle and fallback). One band colour scale everywhere, vessel inspector, bullseye,
registry-driven views. Standout layer (docs/plan-twin-and-simulation.md, section 7i): function twin + ECG,
reasoning flow, what-if, ischaemia lab, similar patients and story mode are done.
Commit + push to main after every verified change, only your own paths. Remaining: final regression, record the
video, fill Devpost, team names.
Gotchas in sections 6, 7b, 7d and 7i. Don't switch tracks.
