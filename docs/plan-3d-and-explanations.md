> **Status: proposed, awaiting approval (8 Oct 2026).** Produced by a research + judge-panel planning workflow (4 researchers, 3 planners, 2 judges, synthesis). Decisions in section 6 are open.

# CardioVision: final plan to win Track A (8 to 14 Oct)

Nothing in the repo has been changed. Everything below needs your approval before work starts. The decisions only you can make are in section 6.

**Who does what.**
- **F** builds the 3D view and the frontend.
- **M** builds the ML backend and the explanations.

You decide which of you is F and which is M.

---

## 1. Goal and the story for the judges

**What the 3D view will show.**
- The demo opens on a see-through human chest built from BodyParts3D, the anatomy resource the organisers name.
- The camera flies in to an atlas heart. Every coronary branch in it is a named structure with an ID from the FMA anatomy ontology, and each branch takes its colour from exactly one model output, through one lookup table.

**How a judge can inspect it.**
- The same three colour bands appear in the 3D view, on the cards and on a 17-segment bullseye.
- Clicking a vessel opens an inspector with:
  - the model's call;
  - how reliable that model is;
  - the findings that drove the estimate;
  - the heart muscle that artery supplies.
- Explanations read like a lab report.
- A bar per data type (history, symptoms, ECG, echo, labs) makes the "multimodal" fusion visible.
- Dragging a what-if slider recolours the heart live.

**How we stay honest.**
- Validation is shown inside the app: ROC and calibration curves, and real angiography outcomes with confidence intervals.
- The middle colour band is drawn striped and labelled "undecided".
- One finding we verified, which other teams probably won't show: our vessel models do not reliably tell which artery is diseased. So the UI presents independent per-artery estimates and never names a "culprit" vessel.
- The brief's caveats are visible on screen.

**What stays fixed.**
- None of this changes the verified models (CAD AUC .943, LAD .856, LCX .761, RCA .739).
- The trained weights get committed, the app runs with one command, and frame rate is measured on an integrated GPU.

---

## 2. What the provided resources change

**The brief (Track A.pdf)**
- It lists model weights as a deliverable, but `.gitignore` currently excludes `*.joblib` and `*.npz`. **We will commit them.**
- It says clinical and ECG features "should not be treated as direct anatomical coordinates", and that Region RWMA "does not provide a direct… 3D anatomical lesion map". So:
  - each artery tree gets one colour;
  - RWMA is shown only as a count chip;
  - we show no lesion position and no culprit vessel.
- It asks for a "torso/heart" model and names BodyParts3D, so we add a chest context layer.
- It asks for "responsive without a dedicated GPU", so we will publish a measured frame-rate table.

**BodyParts3D** (all verified against the official archive)
- **Licence**
  - It is **CC BY 4.0** since 2025-02-27 ([licence page](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)).
  - Required credit line: *"BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International"*. We also link the licence and list what we changed.
  - The 2013 OBJ file headers still say CC BY-SA 2.1 JP. This is decision 6.2.
  - The audit's note that "share-alike applies" is out of date.
- **Getting the meshes**
  - The only current release is v4.0 at 99% polygon reduction.
  - The IS-A zip is 136 MB with 2,234 OBJ files ([download page](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html)). It is the right zip because it has the ventricular wall FJ2428 and the septal branches.
  - The server supports byte-range requests, so we fetch only the files we need and check their CRC32 checksums.
  - Research has already pulled 62 files this way: all 59 coronary meshes plus the walls.
- **Coordinates**
  - Units are millimetres, in the LPS frame ([diagram](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20130619/coordinate_system.png)).
  - glTF frame = (x, z, −y) × 0.001. This matches our current heart's frame: +X is patient-left, +Y up, +Z anterior.
- **Which mesh belongs to which artery**

| Model output | FMA concepts | Mesh files |
|---|---|---|
| LAD | FMA3862, plus septal branches FMA3892 | 18 meshes + FJ2732–34 |
| LCX | FMA3895 | FJ2649–2654 |
| RCA | FMA50039, including the posterior descending artery FMA3840 | FJ2692–2700 |
| Left main | FMA3855 | FJ2737 (not modelled, shown neutral) |

  - Because the posterior descending artery comes off the RCA, this reference heart is **right-dominant**.
- **Mesh quality and size**
  - The vessels sit 1.0–1.2 mm from the wall surface, so the parts are already aligned with each other.
  - The tubes are open-ended and some are fragmented.
  - There are no 17-segment meshes; BodyParts3D's "myocardial zone" concepts point at the wrong files.
  - A full heart with every coronary branch compresses to about 0.3–0.6 MB, against 7.55 MB for today's `heart.glb`.
- **Runtime choices**
  - Use the meshopt decoder. drei's default Draco decoder loads from the gstatic CDN.
  - Do not use Z-Anatomy (CC BY-SA).
  - Do not depend on the Anatomography API at runtime.

**The dataset** ([UCI 411](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset), [JMSS 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3660711), [RCM 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4253773))
- **Cohort and labels**
  - 303 patients with 55 predictors and 4 targets: a single-centre referral cohort, Tehran 2011–12, 71% with CAD.
  - Labels say only whether each artery has ≥50% narrowing. Left main is not coded.
- **Fields that are currently mislabelled in the app**
  - Region RWMA appears to be a *count of regions* (0–4). This is inferred from the data and the wording will say so.
  - Function Class is coded 0–3, NYHA-style, where 0 means "not recorded / not limited".
  - PR is pulse rate, not the ECG PR interval.
  - The "Obesity" flag means BMI ≥25.
  - FH means heart disease in a first-degree relative.
  - Exertional chest pain is constant ("No" for every patient).
  - EF is recorded in 5% steps and never above 60.
- **Things to disclose**
  - Row 93 has Cath = Normal but LAD = Stenotic.
  - The papers and the file disagree on several points: DLP counts are swapped, the papers list Rhythm but the file has BBB, Function Class is 1–4 in the papers and 0–3 in the file, and one paper says 500 patients against 303.
  - The models learned **no benefit** from lowering LDL or weight; both move the risk the wrong way, probably because of treatment and selection effects. The what-if panel therefore locks those two sliders.

**Facts verified from the current models** (`metrics.json` and the out-of-fold predictions)
- **Band cut-offs:** CAD bands are cut at **0.42 / 0.58**; vessel bands at 1/3 and 2/3.
- **Screening thresholds** (tuned for F1): CAD .44, LAD .37, LCX .39, RCA .31. The RCA threshold sits *inside* its Lower band, so the wording must say "screening flag", never "likely".
- **Undecided zones vs the middle band.** The 90% split-conformal undecided zones closely match the middle band, so a single encoding is justified:

| Target | Undecided zone | Middle band |
|---|---|---|
| CAD | .44–.56 | .42–.58 |
| LAD | .29–.71 | .33–.67 |
| LCX | .33–.67 | .33–.67 |
| RCA | .29–.71 | .33–.67 |

- **The vessel models barely tell arteries apart:**
  - Within CAD patients, the AUC for picking the right artery is LAD .69, LCX .65, RCA .60.
  - In the 87 single-vessel patients, the highest-probability vessel was correct **61%** of the time, against **66% for always guessing LAD**.
  - Isolated RCA disease was ranked first in **0 of 17** cases.
  - The CAD model predicts the LAD label exactly as well as the LAD model does (.864 vs .864).

**Code defects to fix first**
- `App.tsx:166` makes the arteries flash grey on every edit.
- `riskColor` is centred at 0.5 and ignores the bands.
- The left main is silently coloured as max(LAD, LCX) (`ArteryTube.tsx:41`).
- Percentages are formatted three different ways.
- The physiological breakdown is collapsed by default.
- The vessel list is hard-coded in 9 or more places.
- Contact shadows re-render every frame, and the device pixel ratio can go up to 2.

---

## 3. Workstreams and items

Each item lists what the judge sees, the rubric line it serves, the build steps, the risk and fallback, and the acceptance test. Effort is in hours.

### WS0: Safety net and reproducibility

**S0. Baseline tag, check script, backup recording** (P0, 1.5 h, both)
- **Judge sees:** nothing directly. This protects the submission.
- **Rubric:** all criteria; Devpost requires a video.
- **Build:**
  - Tag `v0.9-baseline` on `699a6a8`.
  - Add `scripts/check.sh`, which runs `tsc --noEmit`, `npm run build` and `backend/test_pipeline.py`. Run it before every push to main, which keeps the push-continuously rule safe.
  - Record a rough 4–5 minute take of the current build and keep it locally.
- **Risk:** low.
- **Done when:** the tag is on the remote, `check.sh` exits 0, and a backup MP4 longer than 4 minutes exists.

**S1. Commit the weights, add a manifest and a one-command run** (P0, 1.5 h, M)
- **Judge sees:** a fresh clone runs without the 4–5 minute `train.py` step.
- **Rubric:** Technical Implementation 10% (reproducibility). Weights are an explicit deliverable in the brief.
- **Build:**
  - Remove `backend/models/*.joblib` and `*.npz` from `.gitignore`. Commit the four models (2.2 MB) and `oof_predictions.npz` (49 KB).
  - Add `backend/models/MANIFEST.json` with sha256 hashes and library versions; `/health` reports the manifest hash.
  - Add `scripts/dev.sh` to create the venv, install, and start uvicorn on :8000 and Vite on :5173.
  - Add a "Quick start" section to the README and update HANDOFF §6.6.
- **Risk:** pickles break across library versions. Fallback: versions are pinned, the test fails loudly on a mismatch, and `train.py` stays documented.
- **Done when:**
  - a clone into a temp folder followed by `scripts/dev.sh` gives `/health models_loaded=true`;
  - the probabilities for all 5 presets equal those in `presets.json`;
  - the sha256 hashes match the manifest.

### WS1: Honest, consistent 3D encoding

**E1. Smooth updates and one band colour scale** (P0, 3.5 h, F)
- **Judge sees:**
  - Arteries never flash grey while typing; colours glide to the new value.
  - Green, amber and red mean the same bands everywhere. Amber is striped and labelled "undecided".
  - Labels read **"LAD · 70% chance"**, never a bare "LAD 70%", which a cardiologist would read as 70% narrowing.
  - The legend reads "Chance of ≥50% narrowing in this artery (not % blocked)". It shows the observed rate for each band and screening-threshold ticks at 37, 39 and 31%.
  - The left main is neutral, with the note "not labelled in this dataset".
- **Rubric:** System Integration 15% (real-time updates); 3D Visualization 25% (spatial risk mapping); the brief's "consistent correspondence".
- **Build:**
  - `App.tsx:166`: always pass the last probabilities plus a `pending` flag.
  - `ArteryTube.tsx`: replace the `useMemo` colour with a target colour, eased in `useFrame` by `1-exp(-8dt)`. Use `formatPct` in the labels (l.111). Make the left main neutral (l.41).
  - Add `bandColor(p, bands)` in `risk.ts`, with cut-offs read from `/metrics` and soft edges. Use it in ArteryTube, the territory overlay in `SpecimenHeart.tsx`, the schematic myocardium and the dashboard cards.
  - Stripes:
    - Tubes: an `onBeforeCompile` chunk using the existing `uv.x`.
    - Atlas meshes: stripes from world position.
    - Set `customProgramCacheKey`.
  - Rewrite the `HeartScene.tsx` legend (~l.250); line 234 uses `formatPct`.
  - Keep the current palette, and check it under Chrome's deuteranopia emulation.
- **Risk:** shader cache conflicts. Fallback: lower emissive glow plus a "?" on the label instead of stripes.
- **Done when:**
  - typing 10 characters in Age, screen-recorded, shows no grey frame;
  - the 3D label, card and inspector strings match on all 5 presets;
  - an RCA at p=0.35 is amber in both 3D and its card;
  - no label matches `/^\w+ \d+%$/`;
  - `check.sh` is green.

**E2. "What this map shows" caption and RWMA extent chip** (P0, 1.5 h, F)
- **Judge sees:**
  - A caption that is always visible: "Vessel-level estimates; position along the artery is not predicted".
  - An info popover with 6 rules:
    1. Labels are per artery.
    2. Each artery tree has one uniform colour.
    3. Branches take their parent artery's colour (an assumption).
    4. Territories assume standard right dominance.
    5. Findings are model inputs, not locations.
    6. The anatomy is a reference model, not this patient's imaging.
  - A chip: "Wall motion: 2 region(s) abnormal; location not recorded".
- **Rubric:** 3D Visualization 25%; Clinical Interpretability 20%; the brief's primer caveats.
- **Build:** new `components/MapSemantics.tsx` inside the HeartScene overlay; pass `RegionRWMA` to it. Quote the primer verbatim.
- **Risk:** clutter on small screens. Fallback: an icon only below 768 px.
- **Done when:**
  - the caption is visible at 1366×768;
  - changing RWMA from 0 to 3 changes only the chip, and grep shows RWMA reaches no mesh or vertex code.

**E3. Anatomy and vessel lookup table** (P0, 3 h, F)
- **Judge sees:** in the PDF and the video, "adding an artery, a structure or a model is one table entry". Each clickable part names the output it shows.
- **Rubric:** the brief's "add structures/models without redesign" and "consistent correspondence"; Technical Implementation 10%.
- **Build:**
  - New `frontend/src/anatomy/registry.ts`, with an entry per vessel: id, labels, territory text, AHA segments, model target, FMA IDs and BodyParts3D file IDs.
    - AHA segments follow Cerqueira 2002 ([doi](https://doi.org/10.1161/hc0402.102975)): LAD 1, 2, 7, 8, 13, 14, 17; LCX 5, 6, 11, 12, 16; RCA 3, 4, 9, 10, 15.
    - The left main has `target: null`.
  - Replace the hard-coded lists at:
    - `App.tsx:233`
    - `ClinicalDashboard.tsx:52`
    - `ExplanationPanel.tsx:74`
    - `HeartScene.tsx:49-62` and `:430`
    - `SpecimenHeart.tsx:24`
    - `specimenAnatomy.ts:243` (the territory stride becomes the vessel count + 1)
  - `/metrics` gains a `targets` array. This is purely additive; `VesselProbability` stays unchanged.
  - Add vitest. Installing it needs network access once.
- **Risk:** it touches about 9 files. Fallback: land it alone, first thing Friday, run a 5-preset regression, and revert to the tag if anything breaks.
- **Done when:**
  - `grep '"lad", "lcx", "rca"'` matches only `registry.ts`;
  - a test entry with `target: null` renders neutral and can't be selected, with no other file edited;
  - screenshots of the 5 presets are unchanged from the baseline.

**E4. Performance budget and a measured frame rate** (P0, 2.5 h, F; independent of BodyParts3D)
- **Judge sees:** smooth rotation on a laptop's integrated GPU, an FPS table in the README and PDF, and HD switching itself off when frames drop.
- **Rubric:** the brief's "no dedicated GPU" requirement; 3D Visualization 25%; Technical Implementation 10%.
- **Build:**
  - `HeartScene.tsx`:
    - set `dpr={[1,1.5]}`;
    - add `PerformanceMonitor` with `flipflops={3}` and `AdaptiveDpr`;
    - turn HD off by default when `detect-gpu` reports tier ≤1;
    - set `ContactShadows frames={1}`.
  - Tubes go from 180×14 to 96×10 segments, about 31k fewer triangles (use 120×12 if the LAD looks faceted).
  - The schematic grid goes from 170×240 to 110×160.
  - Add `useGLTF.preload`.
  - The `?stats` flag mounts StatsGl and logs `renderer.info`.
- **Risk:** low.
- **Done when:**
  - on the i5-12500H integrated GPU at 1080p, Realistic mode runs ≥45 fps with HD off and ≥30 fps with HD on;
  - a 6× CPU throttle turns HD off automatically;
  - the table is committed.

**E5. Vessel inspector** (P0, 3 h, F)
- **Judge sees:** clicking a tube, label, wall or bullseye segment opens a card showing:
  - the name, plus the branch name and FMA ID in Atlas mode;
  - "Chance of ≥50% narrowing: 34%" with its band, the band's confidence interval and an "undecided" chip;
  - "above this model's screening threshold (37%, tuned for sensitivity)";
  - reliability: "detects narrowing in this artery, AUC .86; telling which artery among patients with CAD, AUC .69";
  - the top 3 drivers with odds multipliers;
  - the territory and AHA segments it supplies;
  - "Not shown: lesion position, severity, left main".
- **Rubric:** requirement 2c (select regions to inspect vessel-specific risk); Clinical Interpretability 20%.
- **Build:** new `components/VesselInspector.tsx`, replacing the 3-line box (`HeartScene.tsx` ~228–240). Data comes from `results[v]`, `explanations[v]`, `/metrics`, `/model-facts` (U1) and the lookup table. It closes on Esc and stops pointer propagation.
- **Risk:** crowding. Fallback: a bottom sheet below 1024 px.
- **Done when:**
  - all 5 ways of opening it show identical content, matching the dashboard;
  - grep finds no hard-coded `.69` or `0/17` in the TSX;
  - Esc closes it;
  - there is no horizontal scroll at 375 px.

**E6. AHA 17-segment bullseye, linked to the 3D view** (P1, 4 h, F)
- **Judge sees:**
  - An SVG polar map titled "Muscle each artery usually supplies (standard AHA assignment, right dominance assumed), shaded by that artery's chance of narrowing. Not a perfusion or wall-motion map."
  - Each territory is one flat colour, striped when undecided.
  - Hovering a segment highlights its artery and territory in 3D; clicking opens the inspector.
  - The RWMA chip sits beside the map and is never painted onto it.
- **Rubric:** 3D Visualization 25% (spatial mapping, region selection); Clinical Interpretability 20%.
- **Build:** `components/Bullseye.tsx` with a pure `segmentColor(seg, probs, bands)` function. Segment-to-vessel mapping comes from the lookup table. The hover state moves up to `App`. Works in every heart mode at vessel and territory level.
- **Risk:** being misread as a perfusion map. Mitigation: the title, flat colours and the separate RWMA chip.
- **Done when:**
  - a unit test shows `segmentColor` depends only on the probabilities and bands, and the mapping equals Cerqueira's;
  - hovering segment 4 highlights the RCA;
  - changing RWMA alters segment colours only through the model probabilities.

**E7. Compress heart.glb textures** (P1, 1.5 h, F)
- **Judge sees:** a faster first load, down from 7.55 MB to an estimated 1–2 MB.
- **Build:** `gltf-transform` resize plus WebP: 2k base colour, 1k normal map.
- **Fallback:** keep the normal map as PNG if it bands.
- **Done when:** the file is ≤2 MB and both of you accept a side-by-side screenshot.

### WS2: BodyParts3D anatomy (P1, behind gates; Realistic stays the default heart until gate G2)

**B1. Asset pipeline** (P1, 5 h hard timebox, F)
- **Judge sees:** one command rebuilds every anatomical asset from the source the organisers named. The credit line appears in the footer and the README.
- **Rubric:** Technical Implementation 10% (reproducibility, public anatomical resources); System Integration 15%.
- **Build:**
  - `scripts/bp3d/fetch_bp3d.py` (standard library only):
    - reads the IS-A zip's central directory over HTTP Range;
    - extracts only the files listed in `scripts/bp3d/parts.json`, which maps FMA concept → file ID → layer or vessel;
    - checks CRC32 and caches into a gitignored folder;
    - fetches heart files first, and torso files only after your OK (decision 6.1).
  - `frontend/scripts/build-bp3d.mjs` (`@gltf-transform/core` + `functions`; meshoptimizer is already installed):
    1. Parse the OBJ files and convert coordinates with (x, z, −y) × 0.001.
    2. Drop connected pieces with less than 2% of the vertices.
    3. Push coronaries about 0.7 mm outward.
    4. Weld, simplify, quantize, meshopt.
    5. Name nodes like `lad|FMA3860|FJ2633`, with extras `{fma, fj, name, vessel}`.
    6. Bake a `_territory` attribute on FJ2428, splitting LV and RV by distance to the cavity meshes FJ2422 and FJ2423.
  - Outputs, both committed:
    - `bp3d_heart.glb`: ≤60k triangles, ≤0.8 MB;
    - `bp3d_thorax.glb`: ≤80k triangles, ≤1.5 MB; the skin is cropped to z 1000–1450 mm before simplifying.
  - `public/models/ATTRIBUTION.md` holds the exact credit line, the licence link, the list of changes, the file IDs and the Mitsuhashi 2009 citation (PMID 18835852). Add `npm run assets:bp3d`.
- **Risk:** the archive server may be slow; the meshes have debris and open ends. **Gate G1 (Fri 20:00):** if the GLB doesn't load correctly oriented with tagged coronaries, drop B2 and B3 and move F to E6 and E7.
- **Done when:**
  - fetch plus build reproduce identical node names and triangle counts;
  - `gltf-transform inspect` shows ≥55 coronary nodes with `extras.vessel` in {lad, lcx, rca, lm};
  - the posterior descending artery (FJ2692–2700) is tagged rca, and FJ2633 and FJ2732 are tagged lad;
  - sizes are within budget.

**B2. Atlas heart mode** (P1, 7 h, F)
- **Judge sees:**
  - A model toggle with three options: Realistic, Atlas and Schematic.
  - Every named branch, each in its parent vessel's band colour: diagonals, septals, conus, LCX branches, RV marginals, the posterior descending artery and the posterior branches.
  - Clicking a branch shows, for example, "Posterior interventricular branch of RCA (FMA3840) → part of RCA (FMA50039) → model output: RCA, 34% chance, undecided".
  - An X-ray toggle reveals the back-wall LCX and posterior descending artery from the front.
  - Territories are computed from the geometry, flat-coloured, and labelled "standard supply, right dominance assumed; illustrative".
  - A caption: "Reference anatomy (BodyParts3D), not this patient's imaging".
- **Rubric:** 3D Visualization 25%; requirement 2b ("colour-code individual coronary artery nodes"); consistent correspondence.
- **Build:**
  - New `components/AtlasHeart.tsx` and `atlasAnatomy.ts`, loading with `useGLTF(url, false, true)`.
  - Coronaries are double-sided, with one shared, eased material per vessel looked up from the table.
  - The left main and veins are neutral and can't be clicked.
  - Wall material: MeshPhysical with sheen.
  - X-ray: wall opacity .25 with depthWrite off.
  - Scale ×≈29 to match the current scene; camera views come from territory centroids.
- **Risk:** it is less photoreal than Sketchfab. **Gate G2 (Sun 13:00):** you choose the default from screenshots (decision 6.3); otherwise it stays a toggle.
- **Done when:**
  - ≥50 branches are coloured and none is grey except the left main;
  - clicking a diagonal, a septal, an LCX branch, the posterior descending artery and an RV marginal selects LAD, LAD, LCX, RCA and RCA;
  - colour hex values equal the dashboard's on all 5 presets;
  - the posterior descending artery is visible in the anterior view with X-ray on;
  - the atria get no territory tint;
  - ≥45 fps with HD off.

**B3. Chest context, fly-in, view cube** (P1, 5 h, F)
- **Judge sees:**
  - Atlas mode opens on a ghosted chest (skin, ribs, sternum, spine, translucent lungs, diaphragm), with the heart behind the sternum and the apex to the patient's left.
  - The camera flies in to the heart.
  - Layer toggles for Skin, Skeleton and Lungs.
  - A cube with L/R/S/I/A/P faces snaps to standard views.
  - The "reference anatomy" caption.
- **Rubric:** requirement 2a ("torso/heart model"); 3D Visualization 25% (interactivity).
- **Build:**
  - New `components/ThoraxContext.tsx`:
    - fresnel ghost skin with depthWrite off and an explicit renderOrder;
    - opaque bones with `raycast={() => null}`, so clicks pass through;
    - lungs at 0.2 opacity.
  - drei `CameraControls` **in Atlas mode only**; Realistic and Schematic keep OrbitControls. Transitions: overview → `fitToSphere(heart)` → vessel views.
  - Add `GizmoViewcube`.
  - Respect reduced-motion settings.
  - On GPUs of tier ≤1, skin and lungs start off.
- **Risk:** transparency sorting, frame rate. **Gate G3 (Sun 19:00):** below 30 fps with all layers on, ship skeleton-only by default. If the fly-in is buggy, ship static Body and Heart buttons instead.
- **Done when:**
  - the apex appears on the viewer's right in the A view;
  - the fly-in takes ≤2 s;
  - clicking the LAD through the skin selects the LAD;
  - ≥30 fps with all layers on;
  - the whole scene is ≤200k triangles and ≤60 draw calls.

### WS3: Explanations a cardiologist would sign off on

**X1. Dataset-faithful labels and a feature dictionary** (P0, 2.5 h, M)
- **Judge sees:** correct terms everywhere:
  - "Regions with abnormal wall motion (0–4)", shown as "2 regions";
  - "Functional class (NYHA-style)", where 0 means not limited / not recorded;
  - pulse rate (not the PR interval);
  - "BMI ≥25 (dataset flag)";
  - heart disease in a first-degree relative;
  - the three-criteria definitions of chest pain;
  - Exertional CP marked "constant, unused";
  - EF hint: "5% steps, max 60";
  - "rare in training data (n=…)" badges for yes/no features with ≤16 positives.
- **Rubric:** Clinical Interpretability 20%; Technical Implementation 10% (use of public datasets).
- **Build:**
  - `featureLabels.ts` and `fieldSpec.ts`.
  - The Field descriptions in `backend/app/schemas.py`, which feed `/docs`.
  - New `frontend/src/featureDictionary.ts`: meaning, unit, reference range with source URL, modifiable class (fixed / lifestyle / state / finding), data group, and rare-n.
- **Risk:** the RWMA meaning is inferred, so the text says "appears to be (inferred from data)".
- **Done when:**
  - grep finds none of the old strings;
  - API payloads for the 5 presets are byte-identical;
  - `/docs` shows the new text;
  - a test confirms every dictionary entry has a definition and a source URL where a range is given.

**X2. Physiological breakdown as a lab sheet** (P0, 4 h, M)
- **Judge sees:** the breakdown is open by default. For each measurement:
  - the value;
  - a typical adult reference range with an H/L flag, sex-specific for HDL, Hb, creatinine and ESR, with a source link;
  - the patient's percentile in the cohort;
  - a cohort fact, e.g. "FBS ≥126: 87% had CAD vs 66% otherwise (n=82)";
  - its contribution and odds multiplier.

  Range sources: [ADA](https://diabetes.org/about-diabetes/diagnosis), MedlinePlus lab pages, [WHO BMI](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight), ASE 2015 for EF.
- **Rubric:** requirement 3c (physiological measurements alongside their contribution); Clinical Interpretability 20%.
- **Build:**
  - New `backend/make_cohort_stats.py`, run on the CSV with no retraining, writes `models/cohort_stats.json`.
  - `/features` gets an additive `cohort` key.
  - `PhysiologicalBreakdown` in `ExplanationPanel.tsx` (l.177) changes from `<details>` to an always-open table with a Compact/All toggle.
- **Risk:** reference ranges vary by lab. Labels say "typical adult reference" and "cohort statistic, not this patient".
- **Done when:**
  - FBS 160 shows H, about the 88th percentile, and an ADA tooltip;
  - HDL 45 is flagged for a woman but not for a man;
  - the table is visible without a click;
  - every old `/features` key is still present.

**X3. Contribution by data type, evidence tags, wording** (P0, 3 h, M)
- **Judge sees:**
  - Exactly additive bars per model, split by data type: Demographics & history, Symptoms & exam, ECG, Echo, Labs, plus a separate row "Pre-test score (age + sex + chest-pain type)". This makes the multimodal fusion visible.
  - Each driver is tagged as a disease finding, a modifiable risk factor, background, or lab/physiology.
  - The summary reads "driven mainly by…" plus "(associations in this referral cohort, not causes)".
  - Hovering a driver shows its contribution in each of the 4 models as small bars in the panel. **The arteries do not glow.**
- **Rubric:** Clinical Interpretability 20% (attribution quality, clarity); the hackathon's multimodal theme.
- **Build:** helpers in `src/explain.ts`. Each derived composite is assigned to one group as a stated convention: risk-factor count → history, ECG count → ECG, eGFR/NLR/lipid ratios → labs. No splitting, so sums stay exact.
- **Risk:** group boundaries are a judgement call; they are documented in the PDF.
- **Done when:**
  - for every preset and target, base value plus the group sums equals the model's log-odds within 1e-6;
  - the summary contains no "because";
  - each driver row has exactly one tag.

**X4. Template clinical summary with a fact validator** (P1, 2.5 h, M)
- **Judge sees:** two short summaries, one for clinicians and one in plain language, where every number matches the panels. Example: "High likelihood of CAD (>95%), driven by typical angina and 3 regions of abnormal wall motion; LAD above its screening threshold; RCA undecided; FBS 160 mg/dL is in the diabetic range."
- **Build:**
  - `frontend/src/narrative.ts`:
    - `buildFactSheet()` gives each fact an ID;
    - `renderSummary()` extends the current `summarySentence`;
    - `validateText()` rejects stray numbers, "% blocked", diagnosis or treatment wording, and localisation.
  - `NarrativeCard` sits above "Why this estimate?".
- **Done when:** a test over the 5 presets finds every number in the fact sheet, and the corrupted string "LAD 99% blocked" fails validation.

**X5. Optional LLM rewrite** (P2, 3.5 h, M; decision 6.4)
- **Judge sees:** a more fluent summary with the badge "AI-written, verified k/k facts". Without an API key, the template appears unchanged.
- **Build:**
  - `POST /narrative`, temperature 0, sending only the fact sheet. Follows the Explingo narrator pattern ([arXiv 2412.05145](https://arxiv.org/abs/2412.05145)).
  - The validator is ported to Python, with fallback to the template.
  - Outputs for the 5 presets are cached in the repo, so the demo works offline.
- **Done when:** with no key there are no errors; an invented number is rejected; the cached outputs pass validation.

### WS4: Validation and uncertainty shown in the product

**U1. Observed outcomes with confidence intervals; "undecided" = the middle band** (P0, 3 h, M)
- **Judge sees:**
  - A 10×10 icon array per card: "Of N patients whose held-out estimate fell in this band, k had ≥50% narrowing at angiography (95% CI …); referral cohort". With the real 0.58 cut, the CAD Higher band is about 196/210.
  - An "undecided" chip on vessels in the middle band, with the tooltip "matches the 90% conformal undecided zone; approximate, computed from fold models".
- **Rubric:** Predictive Performance 30% (risk estimation quality, validation methodology); Clinical Interpretability 20%.
- **Build:**
  - New `backend/model_facts.py` (no retraining):
    1. Load the out-of-fold predictions and labels.
    2. Take each patient's **mean out-of-fold probability across the 5 repeats** (unique patients, not pooled).
    3. Assign bands using the cut-offs in `metrics.json`; compute n, k and a Wilson CI per band.
    4. Compute conformal q̂, ROC points, calibration deciles and slope, the confusion matrix at threshold, the cross-vessel AUC matrix, within-CAD AUCs and the single-vessel test.
    5. Write `models/model_facts.json`.
  - Add `GET /model-facts` and optional, backward-compatible fields `band_n/k/ci` on `TargetResult`.
  - New `IconArray.tsx`.
- **Risk:** switching to unique-patient rates changes figures already quoted in the README and PDF; D1 updates them.
- **Done when:**
  - the JSON is deterministic;
  - band n sums to 303 for each target;
  - printed conformal coverage is 0.90 ± 0.02;
  - old clients still parse `/explain`;
  - icon counts equal round(rate × 100).

**U2. Model facts tab** (P0, 3 h, M)
- **Judge sees:**
  - ROC and calibration curves (with slope) from held-out predictions, the confusion matrix at threshold, and CV mean ± SD.
  - A **localisation section**: the cross-vessel AUC matrix, within-CAD AUCs .69/.65/.60, and the single-vessel test (61% vs 66% for always-LAD; RCA 0/17), ending with the rule "Vessel probabilities are independent per-artery estimates, not a culprit ranking".
  - Intended use and "not for" uses.
- **Rubric:** Predictive Performance 30%; Technical Implementation 10%.
- **Build:** `components/ModelFacts.tsx` with hand-rolled SVG (no new dependency). Load the dataviz skill before drawing the charts.
- **Done when:**
  - the pooled out-of-fold AUC shown is within 0.02 of the CV mean in `metrics.json`;
  - the TSX has no numeric literals except axis ticks;
  - the "not a culprit ranking" text is present;
  - it is readable at 380 px.

**U3. Overall drivers and agreement with the original investigators** (P1, 2 h, M)
- **Judge sees:**
  - A heatmap of the top 15 features against CAD, LAD, LCX and RCA, built from the already-loaded `info.influence` and labelled "in-sample".
  - The original authors' Gini rankings from [RCM 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4253773) beside it, with ticks where they agree:
    - LAD: RWMA, EF, Age, ESR, Lymph;
    - RCA: DM, Age, Lymph, ESR, Neut, FBS.
- **Done when:** the cell values equal `feature_info` influence to 2 decimals.

**U4. "About the data" panel** (P1, 1.5 h, M)
- **Judge sees:** cohort provenance; the ≥50% definition; left main not coded; RWMA as a count rather than a location; row 93; the paper vs file discrepancies; constant and rare features; all with links.
- **Done when:** it opens from the header, and every claim carries a working link.

### WS5: Live what-if

**W1. What-if panel that recolours the heart live** (P0, 5 h, M)
- **Judge sees:**
  - Sliders for BP, FBS, TG and HDL, and a "quit smoking" toggle.
  - LDL and weight are **locked**, with the note "no learned benefit in this referral cohort (likely treatment/selection effects)".
  - A separate "Diagnostic contrast: if this finding were absent" group: typical angina, ST depression, RWMA, EF.
  - The 3D view, cards and bullseye recolour while you drag; a delta strip shows e.g. "LAD 63% → 41%".
  - A permanent banner: "How the model's estimate responds (sensitivity), not the effect of treatment".
  - Reset returns to the patient.
- **Rubric:** System Integration 15% (real-time updates, model-dashboard integration); Clinical Interpretability 20%.
- **Build:**
  - `components/WhatIfPanel.tsx`; levers come from the dictionary's modifiable class.
  - Slider ranges are the cohort p5–p95, and values outside are shaded.
  - `App` keeps scenario overrides separate from the patient; the form is never edited.
  - `api/client.ts` gains `predict()` on the existing `POST /predict` (about 36 ms), throttled to 120 ms with an AbortController.
  - Releasing a slider calls `/explain`.
  - Demo it on the "rca" or "uncertain" preset. On "rca", RWMA 0→3 gives LAD +31.
- **Risk:** saturated presets barely move; the panel suggests switching preset.
- **Done when:**
  - on "rca", the RWMA contrast raises LAD by about 31 points;
  - dragging TG recolours with no grey frame, at ≤10 requests/s with no pile-up;
  - after Reset the form is byte-identical;
  - the banner is present.

**W2. Response-curve sparklines** (P2, 3 h, M)
- **Build:** `POST /whatif` returns 25-point curves from one batched call per target (about 45 ms). The client interpolates, so dragging makes zero requests.
- **Done when:** the curve value at the current setting equals `/predict` ± 0.005.

### WS6: Ship

**D1. Docs** (P0, 5 h, M)
- PDF page 6, "Anatomy, encoding and honesty":
  - the BodyParts3D pipeline and licence;
  - the lookup-table correspondence table;
  - encoding rules and primer compliance;
  - the uncertainty method;
  - the localisation result;
  - the FPS table.
- Also add: provenance and caveats, refreshed band figures, and credits (BodyParts3D CC BY 4.0, Sketchfab CC BY 4.0, UCI CC BY 4.0).
- Rebuild with `docs/make_figures.py` and `docs/build_pdf.sh`, keeping it ≤6 pages.
- Update the README quick start and credits, `docs/demo-script.md` (with a rubric → timestamp checklist), `docs/devpost.md` and `HANDOFF.md`.

**D2. Video and Devpost** (P0, about 10 h, both). Target 5–6 minutes, recorded from tag `v1.0-demo`:

| Time | Scene | Rubric line |
|---|---|---|
| 0:00 | Chest fly-in to the atlas heart (or Realistic, if B is cut) | Anatomical representation |
| 0:25 | Patient entry, colours easing, legend | Pipeline / real-time |
| 1:00 | Click a branch → inspector; bullseye hover; X-ray; view cube | 2c / spatial mapping |
| 2:00 | Why: summary, data-type bars, tagged drivers, lab sheet | 3b / 3c |
| 2:50 | How sure: icon arrays, undecided stripes, Model facts and localisation honesty | Predictive 30% |
| 3:30 | What-if drag on the "rca" preset | Real-time |
| 4:15 | Under the hood: lookup table, `npm run assets:bp3d`, CV protocol, committed weights, FPS | Technical 10% |
| 5:00 | Limits and disclaimer | Requirement 5 |

- **Recording:** OBS at 1080p30, after pre-warming every mode. HD on only if `?stats` shows ≥30 fps while recording.
- **Publishing:** fix the English subtitles (LAD, LCX, RCA, SHAP, FMA), upload as unlisted, and complete Devpost with repo, video, PDF and everyone's full names.

**T1. Guided tour** (P2, 4 h, F)
- **Chapters:** where the heart sits; the three arteries; *arteries flagged at their threshold* (not "highest-risk" or "culprit"); why; how sure; what-if.
- **Condition:** only if every P0 and P1 item is green by Monday 12:00. Otherwise record from the shot list above.

**Effort summary**

| Priority | Hours |
|---|---|
| P0 | about 37 dev hours + 15 for docs and video |
| P1 | about 28.5 |
| P2 | about 10.5 |

Capacity from Thursday evening to Monday is about 88 hours, so P0 + P1 + docs (about 70 h) leaves roughly 20% buffer.

---

## 4. Day-by-day schedule

| Day | F (3D / frontend) | M (ML / backend / explanations) | Checkpoint |
|---|---|---|---|
| **Thu 8, evening (~4 h each)** | S0 backup recording of the current build; E1 encoding and flicker fix | S0 tag `v0.9-baseline` and `check.sh`; S1 weights, manifest, `dev.sh`; X1 labels | `check.sh` green, push. You answer the section 6 decisions. |
| **Fri 9** | E3 lookup table first, in its own commit, followed by a 5-preset regression; E2 caption; E4 performance and FPS baseline; B1 pipeline (5 h timebox; torso files only after your OK) | U1 `model_facts.py`, `/model-facts`, icon arrays, band CIs; start U2 | **G1 at 20:00:** atlas GLB valid? If not, drop B2 and B3. |
| **Sat 10** | E5 inspector; B2 atlas heart behind a toggle | U2 Model facts tab; X2 lab sheet; start X3 | Evening: 5 presets × all modes in the browser |
| **Sun 11** | Finish B2 (**G2 at 13:00:** which heart is the default); B3 chest and fly-in (**G3 at 19:00:** ≥30 fps?) | Finish X3; W1 what-if | **20:00 scope check:** apply the cut order if behind |
| **Mon 12** | E6 bullseye; E7 WebP; final performance pass and FPS numbers | X4 template summary; U3; U4; D1 docs | 12:00: P2 only if everything is green. 14:00: full regression plus a fresh clone. **18:00: feature freeze, tag `v1.0-demo`.** Evening: finish D1. |
| **Tue 13** | Rehearse, record from the tag (2–3 takes per scene), edit | Subtitles, upload unlisted by 20:00, Devpost draft | Bug fixes only; re-tag if one is needed |
| **Wed 14** | Fresh-clone test from GitHub; tag `v1.0` | Check the video plays logged-out; rebuild the PDF if anything changed | **Submit by 14:00 IST.** The rest of the day is buffer. Update HANDOFF. |

**Rules for the week**
- Commit and push to main after each change that passes `check.sh`.
- New 3D modes stay behind toggles that default to off until they pass their gate.

**Cut order if behind** (first to go first)
1. The P2 items: LLM rewrite, sparklines, tour.
2. Skin and lungs in B3 (keep the skeleton).
3. U3.
4. E6 bullseye.
5. E7.
6. B3 entirely.
7. B1 and B2 entirely: Realistic heart only, and we don't claim BodyParts3D.
8. X4 (the current summary sentence stays).

P0 items are never cut.

---

## 5. What we will NOT do, and why

- **No retraining or model changes.** `metrics.json`, the presets, the README tables and the PDF numbers stay frozen. All new statistics come from the existing out-of-fold predictions.
- **No suggestion of where a lesion is.**
  - No colour gradients along a vessel and no per-branch probabilities.
  - No SYNTAX, CAD-RADS or FFR-style display.
  - Labels are per artery, ≥50% narrowing, and nothing finer.
- **No painting RWMA, ST/T changes, Q waves or LVH onto the 3D heart or the bullseye.** The brief's primer forbids it.
- **No "culprit" or "highest-risk vessel" framing.** That means no automatic fly-to and no artery glow when hovering a driver. The data contradicts it: 61% correct vs 66% for always guessing LAD, and RCA 0/17.
- **No "narrowing likely" wording.** The thresholds are sensitivity-tuned screening flags, and the RCA flag at 31% sits inside its Lower band.
- **No left main colour borrowed from max(LAD, LCX).** It is shown neutral and "not labelled".
- **No second uncertainty encoding.** "Undecided" means the middle band, and the conformal analysis is its justification.
- **No causal or treatment claims, and no counterfactual search (DiCE).** LDL and weight are locked in the what-if panel.
- **No patient-specific anatomy and no dominance toggle.** "Reference anatomy, right dominance assumed" is stated.
- **No global camera refactor.** CameraControls are used in Atlas mode only.
- **No uncuttable BodyParts3D work.** No per-segment 3D highlighting or short-axis cutaway, and no fitting the Sketchfab heart into the chest.
- **Not built this week:** a printable report, a similar-patients panel, beeswarm or dependence plots, an agentic chat or Q&A, hosting, Docker, or bootstrap ensembles at inference (25× slower).
- **No palette swap, no major dependency upgrades** (R3F 9, postprocessing v3), and **no return to `shap.TreeExplainer`**.
- **No Z-Anatomy assets, no Anatomography API at runtime, and no Draco decoder loaded from a CDN.**
- **No 7–9 minute video.** 5–6 minutes, with every rubric line given a timestamp.

---

## 6. Decisions you need to make

1. **BodyParts3D download.**
   - May we range-fetch the heart files (a few MB) now?
   - May we fetch the chest files later? They are about 55 MB of OBJ uncompressed (the skin alone is 14.5 MB), probably about 15–20 MB over the network; that last figure is an estimate.
   - Either way, only the listed files are fetched, never the full 136 MB zip.
2. **Licence for the BodyParts3D-derived GLBs.**
   - Option (a): CC BY 4.0, which is the current official licence.
   - Option (b, recommended): CC BY-SA 4.0 for those GLB files only, because the old file headers say BY-SA 2.1 JP.
   - The app code is unaffected either way, and the files stay separate from the CC BY Sketchfab heart.
3. **Default heart.** Realistic (photoreal, current) or Atlas (every named branch, lighter, in the chest). We recommend deciding at gate G2 on Sunday from screenshots; Realistic stays the default until then.
4. **LLM summary** (P2).
   - It needs an `ANTHROPIC_API_KEY`, sends only the fact sheet (dataset values, no identifiers), and is cached for the demo.
   - The Track A rubric doesn't score it; it only matches the hackathon's general GenAI theme.
   - Yes or no?
5. **Committing the trained weights** (2.2 MB plus a 49 KB file) to the public repo. Recommended, because the brief lists weights as a deliverable.
6. **What-if lever set.** BP, FBS, TG, HDL and smoking adjustable; LDL and weight locked with a note; findings only as "if absent" contrasts. OK?
7. **Who is F and who is M,** who narrates the video, and both full names for Devpost.
8. **Hosted demo.** We recommend none: judges get the video, the PDF and a one-command local run, and this avoids free-tier cold starts during judging (15–20 Oct).

---

**Sources:**
- BodyParts3D: [licence](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html), [download](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html), [README](https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/README_e.html)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Dataset: [UCI 411](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset), [JMSS 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3660711), [RCM 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4253773)
- AHA 17-segment model: [Cerqueira 2002](https://doi.org/10.1161/hc0402.102975), [Radiopaedia](https://radiopaedia.org/articles/cardiac-segmentation-model-1)
- Pipeline tools and reference project: [glTF-Transform](https://gltf-transform.dev), [human-atlas (reference implementation)](https://github.com/ashemag/human-atlas)

Repo facts were re-checked read-only today: HEAD is `699a6a8`; the band cuts and thresholds in `backend/models/metrics.json` are as stated; `.gitignore` lines 22 and 35 exclude `*.joblib` and `*.npz`; `frontend/public/models/heart.glb` is 7,555,476 bytes.