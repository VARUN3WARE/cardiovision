> **Status (10 Oct 2026): T1–T6 done.** T1 function twin 3538b91 (tests 176ba8d); T3 reasoning flow abf0039
> (review fixes 089958e); T5 what-if ff40c77 (review fixes 78a1976); T4 twin cohort 4120ac2 (review fixes
> eae5f5b); T2 ischaemia lab eea1b71 + a1cca8a (review fixes 6d80f98); T6 story mode fea1a35 + 23d6f4c. T1–T5
> were each reviewed independently. Files, deep links, design decisions and gotchas: HANDOFF.md section 7i; "As
> built" notes below record where the build differs from this plan. Builds on `docs/plan-3d-and-explanations.md`
> (BodyParts3D atlas, encoding, inspector, bullseye and registry are done).

# CardioVision Twin: plan for a stand-out final version (9 to 14 Oct)

## 1. The idea in one sentence

Turn the 3D view from "a heart coloured by risk" into **a heart that behaves like this patient's**, a **physiology lab
that shows what a narrowed artery actually does**, and **the AI's reasoning visualised as data flowing into the
anatomy**: three layers, each honest about what it is.

| Layer | What it answers for a judge | Built from | Status (10 Oct) |
|---|---|---|---|
| **T1 Function twin** | "Is this *this* patient?" Their pulse, pumping strength and ECG findings drive the heart. | The patient's own measurements | Done: 3538b91, 176ba8d |
| **T2 Ischaemia lab** | "Why does ≥50% narrowing matter, and why do angina and wall-motion abnormality weigh so much in the model?" | Textbook coronary physiology (published models) | Done: eea1b71, a1cca8a; review fixes 6d80f98 |
| **T3 Reasoning flow** | "How did the AI combine the data types to reach this?" | The models' exact attributions | Done: abf0039, 089958e |
| T4 Twin cohort | "Which real patients look like this one, and what did their angiograms show?" | The 303 patients, held-out predictions | Done: 4120ac2, eae5f5b |
| T5 What-if | "What moves this estimate?" | Live model re-runs | Done: ff40c77, 78a1976 |
| T6 Story mode | One button that tells the whole story in 90 seconds (also the video's backbone) | All of the above | Done: fea1a35, 23d6f4c |

**Why this wins the rubric:**
- 3D Visualization (25%): animated, patient-driven, interactive simulation, not a static coloured model.
- Clinical Interpretability (20%): the ischaemic cascade explains *why* the top drivers predict disease.
- System Integration (15%): every slider and input re-drives the models and the twin live.
- Predictive Performance (30%): the verified numbers, shown in-app (model facts tab, already planned).
- It matches the hackathon's multimodal theme: history, symptoms, ECG, echo and labs visibly fuse into one estimate.

## 2. Honest framing of "digital twin" (this is what makes it credible)

- A cardiac digital twin combines **mechanistic models** (physiology) and **statistical models** (here, the ML),
  personalised with the patient's data (Corral-Acero et al., *Eur Heart J* 2020;41:4556).
- Ours is a **function-driven twin on reference anatomy**. It uses the patient's recorded pulse, ejection
  fraction, blood pressure and ECG findings. It does **not** have their anatomy or coronary images.
- Every twin screen carries the caption: *"Driven by this patient's recorded values on a reference heart. Not their
  imaging; artery positions and lesions are not known."*
- The ischaemia lab is a **sandbox** for one generic narrowing chosen by the user. It is never drawn as the
  patient's lesion, which keeps us inside the brief's rule that findings are not anatomical coordinates.

## 3. Modules

### T1. Function twin: the heart beats like this patient's (P0, about 1 day) — done (3538b91, 176ba8d)

**As built:** `twin/twin.ts` (shared heart clock, beat timing, mechanics, `ecgSample`), `twin/EcgMonitor.tsx`
(sweeping two-lead monitor, toggle "ECG") and `twin/useHeartMotion.ts`, on all three hearts; vitest for beat order,
QT/QRS and the EF squeeze. Deep links `?patient=low|lad|rca|triple|uncertain`, `?heart=atlas|realistic|schematic`.
Since 089958e the twin always shows the patient's recorded values, also while a what-if scenario colours the
arteries (HeartScene `recorded` prop).

**What a judge sees:**
- **Heart rate** from the patient's pulse (already done).
- **Pumping strength** from EF: contraction amplitude scales with ejection fraction, so EF 60% vs 35% is visibly
  different, and the label says *"Squeeze driven by EF 35%"*.
- **Aortic pulse** amplitude scales with blood pressure.
- **A live ECG strip** under the heart, synchronised with the beat:
  - P wave → atrial contraction; QRS → ventricular contraction; T wave → relaxation.
  - It shows the patient's recorded findings: ST depression, T-wave inversion, pathological Q waves, LVH (tall
    QRS), and LBBB/RBBB (wide QRS ≥120 ms with typical morphology).
  - Caption: *"Illustrative waveform built from the recorded findings, not the patient's tracing."*
- **Bundle branch block** delays contraction of the affected side: LBBB makes the lateral LV wall contract late
  ("dyssynchrony"), a real, recognisable sign.

**Science:**
- Synthetic ECG with the dynamical model of McSharry et al. (*IEEE TBME* 2003;50:289), a sum of Gaussian P, Q, R,
  S and T waves.
- Normal intervals: PR 120–200 ms, QRS < 120 ms (≥ 120 ms in bundle branch block), QT rate-corrected (Bazett).

**Build:**
- `twin/ecg.ts`: wave synthesis.
- `EcgStrip.tsx`: canvas trace, phase-locked to the existing `phase` clock.
- Per-vertex contraction in the atlas wall shader, using the territory and septum-to-lateral weights we already
  have.

**Done when:**
- Five presets each produce an ECG matching their findings.
- EF 35 vs 60 is visibly different.
- An LBBB patient shows late lateral contraction.
- Frame time stays under 18 ms.

### T2. Ischaemia lab: what a narrowed artery does (P0, about 1.5 days; the "wow" module) — done (eea1b71, a1cca8a)

**As built:** `twin/coronaryPhysiology.ts` (tests reproduce the curve: maximal flow first below 95% at 50.5%
narrowing, resting flow flat to about 84%, normal reserve 4), `lab/LabPanel.tsx`, `lab/FlowReserveCurve.tsx`,
`lab/labState.ts`. Distance from the ostium is computed in the browser on first open (shortest paths over each
artery tree), not baked into the GLB. Deep link `?lab=lad|lcx|rca&narrowing=0..95&exercise=0..1`. The closing line
below was changed: ST depression ranks only 24th of 63 inputs in the CAD model by mean |attribution| (typical chest
pain 1st, wall motion 3rd), so the panel does not call it one of the strongest inputs. Review fixes (steady SANDBOX
marker, layouts at every width, wording, no first-open pause) landed in 6d80f98; frame rate on an integrated GPU is
not yet measured.

**What a judge sees.** A "Physiology lab" toggle. The user picks an artery, drags **Narrowing 0–95%** and
**Exercise rest → peak**, and watches:

1. **Coronary flow** as glowing pulses travelling down the arteries, mostly in **diastole**: the LV squeezes its own
   arteries shut in systole, so the RCA is less affected. Downstream of the narrowing, pulses thin out.
2. **Supply vs demand** for that artery's territory. Demand rises with heart rate and blood pressure (the rate-pressure
   product); supply is capped by the narrowing. When supply falls short, the territory turns ischaemic (dark blue).
3. **The ischaemic cascade** lights up in order, each step tied to a dataset feature:
   - perfusion defect →
   - **wall-motion abnormality** (the territory visibly stops contracting; the RWMA input) →
   - **ST depression** on the ECG strip (the ST depression input) →
   - **angina** (the typical chest pain input).
   The order follows Nesto & Kowalchuk, *Am J Cardiol* 1987;59:23C.
4. A small **flow-reserve curve** marks where the slider sits:
   - Resting flow holds until about 85% narrowing.
   - Maximal (exercise) flow starts falling at about **50%**: *that is why the dataset, and clinicians, use ≥50%.*
   - Source: Gould et al., *Am J Cardiol* 1974;33:87.
5. Closing line (planned): *"This is why typical angina, wall-motion abnormality and ST depression are the model's
   strongest inputs: they are the late steps of this cascade."* As built it names only typical angina and
   wall-motion abnormality, because ST depression ranks 24th of 63 inputs in the CAD model.

**Science:**
- Stenosis pressure drop ΔP = F·Q + S·Q², with a viscous and a separation term (Gould).
- Resting autoregulation, coronary flow reserve about 3–4× in normal arteries.
- Demand ∝ heart rate × systolic BP.
- All constants come from textbook values, are documented in the code, and are unit-tested against the published
  curve.

**Build:**
- `twin/coronaryPhysiology.ts`: a pure, tested model.
- Flow pulses as a shader stripe moving along a per-vertex *distance-from-ostium* attribute, baked into the atlas
  GLB (graph distance over each artery tree; the Realistic/Schematic tubes already have it as `uv.x`).
- Territory hypokinesis in the wall shader.
- Cascade checklist UI.

**Guardrails:**
- The lab runs only on the generic narrowing the user chose. Its banner says *"Teaching sandbox, not this patient"*.
- Patient mode never shows a lesion.

**Done when:**
- The model reproduces Gould's curve: maximal flow at 50% and 80% narrowing within ±10% of the published values;
  resting flow flat to about 85%.
- The cascade order is fixed.
- 60 fps with HD off.

### T3. Reasoning flow: the AI's evidence flowing into the heart (P0, about 1 day) — done (abf0039, 089958e)

**As built:** `reasoning/` (dataGroups.ts with tests, ReasoningFlow.tsx, anchors.tsx, EvidenceStrip.tsx). Groups:
History & demographics, Symptoms & exam, ECG, Echo, Labs and Pre-test score. For overall CAD the streams end on a
halo around the whole heart (a single target point read as a spot on the LAD). Towers and labels are laid out from
measured overlays and become chips on narrow views. Deep link `?reasoning=1` (`&model=lad|lcx|rca`).

**What a judge sees:**
- Five **modality towers** around the heart: History & risk factors, Symptoms & exam, ECG, Echo, Labs, plus a
  "clinical scores" node.
- Animated **streams** run from each tower into the artery (or the whole heart, for CAD) whose estimate it moves.
  - Stream thickness is that data type's exact contribution (summed attributions).
  - Red streams push the estimate up; blue ones pull it down.
- Hovering a stream lists its findings with their odds multipliers.
- This makes "multimodal fusion" literally visible: *"ECG and echo carried most of this LAD estimate."*

**Honesty:**
- Towers sit around the heart, never on anatomy, so streams are data, not locations.
- Sums are exact, the same numbers as the explanation panel.
- Composite scores are assigned to one group by a stated rule.

**Build:**
- `ReasoningFlow.tsx`: instanced particles along curved paths, driven by `result.explanations`.
- Group sums come from the X3 helper already planned in the earlier plan.

**Done when:**
- For every preset and model, stream widths are proportional to the group sums.
- The group sums plus the base equal the model's log-odds within 1e-6.

### T4. Twin cohort: the most similar real patients (P1, about 0.75 day) — done (4120ac2, eae5f5b)

**As built:** no API change. `backend/make_cohort.py` writes a static `frontend/src/cohort/cohort.json`
(deterministic) and the browser finds the 10 nearest neighbours (`cohort/knn.ts`, identical to the script). 3D
t-SNE map; inputs not entered are left out of the comparison; a dataset patient is recognised and left out of its
own neighbours; "no close matches" when nothing is close. Deep link `?cohort=1`.

**What a judge sees:**
- A 3D "cohort galaxy" of the 303 patients, placed by similarity of their model inputs, coloured by their real
  angiography result.
- The current patient is a glowing point among them.
- A panel lists the 10 most similar patients: *"8 of 10 had CAD; 6 had LAD narrowing"*, with their held-out
  estimates.
- This is case-based evidence, which clinicians trust.

**Build:**
- An offline embedding (t-SNE or UMAP on standardised inputs) and a nearest-neighbour index, computed by a script
  and served by the API.
- Instanced points in a second canvas or tab.

**Honesty:** "similar in recorded findings", not identical patients; outcomes are from this single referral cohort.

### T5. What-if (P0, already planned as W1, about 0.75 day) — done (ff40c77, 78a1976)

**As built:** sliders for BP, fasting glucose and triglycerides plus quit smoking; **HDL is locked too** (with LDL
and weight: no learned benefit, `docs/model-verification.md` section 7). Finding switches flip a recorded finding
either way (present → absent, absent → present, marked hypothetical). A strip above the 3D view shows the scenario;
the heartbeat and ECG keep the recorded values. Deep link `?whatif=1&bp=&fbs=&tg=&quit=1&contrast=angina,st,rwma,ef`.

- Sliders: BP, fasting glucose, triglycerides, HDL, smoking. LDL and weight are locked, because the verification
  showed no learned benefit.
- The twin, arteries, bullseye and reasoning streams all update live.
- Banner: *"How the model's estimate responds, not the effect of treatment."*

### T6. Story mode (P1, about 0.5 day) — done (fea1a35, 23d6f4c)

**As built:** a toolbar **▶ Story** button plays six captioned steps on the Atlas heart, 92 s in all: chest fly-in
(12 s); the function twin (15 s); reasoning streams for the overall model (20 s); the inspector on the artery in the
highest band, ties broken by probability, captioned "independent, not a culprit ranking" (15 s); the ischaemia lab on
that artery, narrowing 0 → 85% and then exercise rest → peak (18 s); a closing card (12 s: reference anatomy, the
303-patient referral cohort, cross-validated AUCs with bootstrap 95% CIs from the verification, decision support
only). The inspector now comes before the lab, so the lab demonstrates on the inspected artery, and "the arteries
colour" is part of that step. Play/pause, previous/next, progress dots, arrow keys, Esc; Exit restores the user's
exact view (heart, modes, chest, lab settings, reasoning, selection, camera) and never changes the patient's data or
a what-if scenario; reduced motion gives static steps. Deep links `?story=1`, `?story=1&step=N` (step N as it ends,
paused) and `&play=1` (plays from the start of step N). The plan as written:

One **"Tell the story"** button plays a 90-second camera choreography:

1. chest fly-in;
2. heart beating at the patient's rate, with the ECG;
3. reasoning streams flow in;
4. the arteries colour;
5. the ischaemia lab demonstrates the cascade;
6. the inspector opens on the highest-band artery;
7. closing honesty card.

It doubles as the backbone of the demo video and can be paused or escaped at any time.

## 4. Schedule (what remains, including the deliverables)

| Day | 3D / frontend (F) | ML / backend / docs (M) | Checkpoint |
|---|---|---|---|
| Fri 9 | T1 function twin + ECG strip | Model facts tab (verified numbers, ROC, calibration, vessel specificity) | ECG matches all 5 presets |
| Sat 10 | T3 reasoning flow | T5 what-if; data-type sums (X3) | Streams sum exactly |
| Sun 11 | T2 ischaemia lab: flow pulses, hypokinesis, cascade UI | T2 physiology model + tests against Gould; bake distance-from-ostium into the GLB | Gould curve reproduced |
| Mon 12 | T6 story mode; performance pass | T4 twin cohort (if on time); docs and PDF refresh | **18:00 feature freeze** |
| Tue 13 | Record the video (5–6 min) from story mode + manual scenes | Devpost text, subtitles, upload unlisted | Video uploaded |
| Wed 14 | Fresh-clone test | Submit by 14:00 | **Submitted** |

**Actual (10 Oct):** T1 landed on 9 Oct; T3, T5, T4 and T2 on 10 Oct (ahead of the table), each independently
reviewed, with every review fix committed (T2's last, in 6d80f98). T6 landed the same day (fea1a35, 23d6f4c). The
model facts tab is not built: in the app the cards and the inspector show each model's cross-validated AUC, and the
independent verification (CIs, permutation test, baselines, vessel specificity) lives in
`docs/model-verification.md`; the story's closing card adds the bootstrap 95% CIs.

**Cut order if behind:**
1. T4 cohort.
2. T6 auto-play (record the story manually instead).
3. The exercise slider in T2 (keep the narrowing slider).
4. BBB dyssynchrony in T1.

Never cut: T1's ECG, T3, the model facts tab, video, Devpost.

## 5. What we will not do, and why

- **No "risk over time" or disease-progression forecast:** the dataset has no follow-up, so any curve would be invented.
- **No patient-specific lesion, FFR or stenosis location:** we have no imaging, and the brief forbids it.
- **No claims that the lab's numbers are this patient's:** it is a teaching sandbox with published physiology.
- **No anatomy changed by ECG findings:** LVH on the ECG is a voltage sign; we do not thicken the wall.
- **No VR/AR or heavy new dependencies:** judges watch a video; frame time on a laptop matters more.

## 6. Decisions for you

1. **Modules:** which of T1–T6 do we build? Recommended: T1, T2, T3, T5 and T6; T4 if time allows.
   *Outcome: all six, done by 10 Oct.*
2. **Default heart:** Atlas (needed for the lab's flow pulses and territories, and for the chest fly-in) or Realistic?
   Recommended: **Atlas as default**. The lab runs on Atlas only; T1 and T3 run on all three hearts.
   *As of 10 Oct the default is still Realistic; the lab switches to Atlas by itself and `?heart=atlas` opens it.*
3. **Who does what:** confirm F and M, the narrator, and both full names for Devpost.
4. **Optional AI-written summary** (earlier decision 6.4): still yes or no?

## 7. References

- Corral-Acero J, et al. The 'Digital Twin' to enable the vision of precision cardiology. *Eur Heart J* 2020;41:4556–64.
- Gould KL, Lipscomb K, Hamilton GW. Physiologic basis for assessing critical coronary stenosis. *Am J Cardiol* 1974;33:87–94.
- Nesto RW, Kowalchuk GJ. The ischemic cascade. *Am J Cardiol* 1987;59:23C–30C.
- McSharry PE, et al. A dynamical model for generating synthetic electrocardiogram signals. *IEEE Trans Biomed Eng* 2003;50:289–94.
- Cerqueira MD, et al. Standardized myocardial segmentation (AHA 17 segments). *Circulation* 2002;105:539–42.
