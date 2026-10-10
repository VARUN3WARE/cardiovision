# CardioVision demo video script (target 5:50)

Requirements (Track A): 3–10 minutes, on YouTube (unlisted is fine), English audio or subtitles, showing the
working system, the feature-input workflow, the 3D visualisation interactions and the technical
implementation. Narration below is about 790 words: roughly 5 minutes 30 seconds of speech at a calm pace, about
5:50 with the pauses. Every number in it was checked against the running app or `docs/model-verification.md`
on 10 Oct.

## Before recording

- Start both servers (`uvicorn app.main:app --port 8000` in `backend/`, `npm run dev` in `frontend/`). Chrome,
  full screen, 1920×1080, browser zoom 100%, developer tools and notifications closed.
- Open the scene links below in tabs, in order, and let each load once: that warms the 3D models, shaders and
  the API. Background tabs pause their animation. If a tab ever shows "Cannot reach the prediction API", click
  **Estimate risk**.
- In each 3D tab the toolbar should read **HD on** (HD switches itself off if the frame rate stays low).
- Turn off the operating system's "reduce motion" setting: with it, the chest fly-in is skipped and the
  particles freeze.
- Record with OBS (or similar) at 1080p, 30 fps, with a separate microphone track. Keep the mouse slow.
- Add English subtitles on YouTube (auto-captions, then correct "LAD", "LCX", "RCA", "ECG", "BodyParts3D",
  "Diamond-Forrester", "ischaemia").

| Scene | Tab (append to `http://localhost:5173/`) |
|---|---|
| 1, 11 | `?patient=triple&heart=atlas` |
| 2 | `/` (the demo patient) |
| 3 | `?patient=lad` |
| 4 | `?patient=lad&heart=atlas` |
| 5 | `?patient=lad&heart=atlas&reasoning=1` |
| 6 | `?patient=rca&heart=atlas&whatif=1&contrast=angina` |
| 7 | `?patient=triple&lab=lad&narrowing=0` |
| 8 | `?patient=triple&cohort=1` |
| 9 | Story mode: link to be added when it lands (planned `?story=1`) |
| 10 | `docs/model-verification.md` on GitHub (and the PDF's architecture figure) |

## Scene 1: the problem (0:00–0:20)

**Open:** `?patient=triple&heart=atlas`. **Do:** reload the tab so the camera flies in through the chest (the
fly-in plays on load); let the heart beat; rotate slowly.

> Coronary artery disease is the world's leading cause of death. CardioVision predicts overall disease and
> significant narrowing of each main coronary artery from 55 routine findings, and shows every estimate on an
> interactive 3D heart. It is decision support for education, trained on 303 patients from one centre: not a
> diagnosis.

## Scene 2: entering a patient (0:20–0:55)

**Open:** `/`. **Do:** click **New patient**, then **Quick (19 key inputs)**. Type Age 58, Sex Female, Weight
72, Height 160; Chest pain **Typical angina**; Diabetes **Yes**; Systolic BP 150; Ejection fraction 45;
Regional wall-motion abnormality 2. Point at the BMI and the live update. Type Age **5**: the field error
appears; correct it to 58. Point at the "values not entered" line in the results.

> Entering a patient is quick. Quick mode asks for the nineteen most informative inputs; full mode has all
> fifty-five. Units, ranges and BMI are built in, and input is validated as you type, so an implausible age never
> reaches the model. Estimates update live. Anything left empty is listed with the result: an unrecorded finding
> counts as absent and a missing measurement takes the cohort median, so nothing positive is invented.

## Scene 3: reading the result (0:55–1:20)

**Open:** `?patient=lad`. **Do:** point at the CAD band text ("In validation, 93% of patients in this band had
CAD on angiography"), then the three vessel cards with their AUC lines and the dashed RCA card; point at the
held-out lines (LAD match, LCX match, RCA false alarm).

> Each estimate has a likelihood band with the rate actually observed in validation, and each artery shows its
> model's cross-validated AUC; the weaker RCA model is dashed. This is a real patient with isolated LAD disease.
> Models that never saw her match her angiogram for the LAD and LCX, and raise a false alarm for the RCA.

## Scene 4: the heart behaves like hers (1:20–2:00)

**Open:** `?patient=lad&heart=atlas`. **Do:** point at the ECG monitor (♥ 100 bpm, BP 180, chips "ST
depression" and "T-wave inversion"). Click the **LAD** label: the camera turns to its territory and the
inspector opens; hover a bullseye segment in the results column. Then click **Low risk** under "Real patients
from the dataset" and point at the wide QRS on the monitor and the late lateral wall.

> The heart behaves like hers. Her recorded pulse of 100 sets the rate, her ejection fraction the squeeze and
> her blood pressure the aortic pulse, and the monitor draws an illustrative ECG from her recorded findings: ST
> depression and T-wave inversion. The anatomy is a BodyParts3D reference heart, not her imaging. Clicking an
> artery opens its inspector: band, observed rate, threshold, AUC, drivers and the segments it usually supplies,
> linked to the bullseye. This other patient has left bundle branch block: a wide QRS, and the lateral wall
> contracts late.

## Scene 5: how the model combined the data (2:00–2:30)

**Open:** `?patient=lad&heart=atlas&reasoning=1`. **Do:** let the streams run into the halo (Symptoms ×7.3 ↑,
ECG ×2.9 ↑, Echo ×0.52 ↓); hover the Symptoms tower. Click the **LAD** label: the streams retarget. Scroll the
results column to **Why this estimate?** and point at the "Evidence by data type" strip.

> How did the model combine the data? Reasoning mode sums each model's exact attributions by data type. For
> overall disease the streams flow into a halo round the whole heart: her symptoms multiply the odds about
> seven-fold, her ECG about three-fold, and her echo pulls the estimate down, in blue. Click the LAD and they
> retarget to its model. The panel shows the same numbers: associations in this cohort, not causes.

## Scene 6: what moves the estimate (2:30–3:05)

**Open:** `?patient=rca&heart=atlas&whatif=1&contrast=angina`. **Do:** point at the strip above the 3D view
("What-if: typical angina → no chest pain · model sensitivity, not treatment effect") and the recoloured
arteries. Scroll to the **What-if** card: delta strip CAD 90% → 40%, LAD 33% → 9%, RCA 53% → 30%; point at the
locked LDL, HDL and weight. Click **Reset to patient**, drag **Triglycerides** slowly up to about 250 (LAD
climbs to about 50%), then **Reset to patient** again.

> What moves an estimate? The what-if card re-runs the models on a scenario without touching the record. This
> patient has typical angina. Without it, overall disease falls from 90 to 40%, the LAD from 33 to 9 and the RCA
> from 53 to 30%: the model's sensitivity, not the effect of treatment. Dragging a slider re-runs all four models
> several times a second. LDL, HDL and weight are locked: this cohort taught the models no benefit from lowering
> them.

## Scene 7: what a narrowed artery does (3:05–3:55)

**Open:** `?patient=triple&lab=lad&narrowing=0`. **Do:** point at the amber banner ("Teaching sandbox with
textbook physiology, not this patient's arteries") and the pulses. Drag **Narrowing** slowly to 50% (pause on
the curve marker), then to 80% (rest flow still 100%). Drag **Exercise** to peak (HR 80 → 129, BP 130 → 180):
the territory turns blue, the cascade lights to ST depression and the monitor shows "ST depression · lab".
Finally drag **Narrowing** to 90%: angina lights; read-outs max flow 7%, rest flow 26%, demand 23,260 vs supply
2,660, 89% short.

> Why does fifty percent narrowing matter? The ischaemia lab is a teaching sandbox with textbook physiology, not
> this patient's arteries. Pulses show coronary flow, mostly while the heart relaxes. As I narrow the LAD, nothing
> changes at rest. At fifty percent, maximal flow is just starting to fall, which is why it is the clinical and
> dataset threshold. Even at eighty, resting flow is normal. Now exercise: demand outstrips what the artery can
> deliver, the territory turns blue and contracts less, and the cascade lights in order: perfusion, wall motion,
> then ST depression on the monitor. At ninety, angina. These late steps help explain why typical angina and
> wall-motion abnormality weigh so heavily in the overall model.

## Scene 8: similar real patients (3:55–4:20)

**Open:** `?patient=triple&cohort=1`. **Do:** point at "10 of 10 similar patients had CAD; LAD ≥50% narrowing
in 5, LCX in 6, RCA in 5" and "This is dataset patient #173 … left out"; drag the cohort map and hover a
ringed neighbour; scroll the list of held-out estimates against angiography.

> Clinicians also reason from similar cases. These are the ten most similar of our 303 real patients, weighted by
> what the model uses: all ten had coronary disease, shown with their held-out estimates and angiograms. She is in
> the dataset, so she is recognised and left out. On the map, the two clusters are patients with and without
> typical angina.

## Scene 9: story mode (4:20–4:40) — placeholder, update when Story mode lands

**Open:** the Story-mode link (planned `?story=1`). **Do:** press the **Story** button and let two or three
captioned steps play; show pause and Esc. If Story mode is not ready at recording time, cut this scene.

> One button tells the whole story on the current patient: the fly-in, the heartbeat and ECG, the reasoning
> streams, the inspector on the highest-band artery, the lab's cascade, and a closing honesty card.

## Scene 10: how it was built and verified (4:40–5:30)

**Open:** the architecture figure in `docs/CardioVision-Technical-Documentation.pdf`, then
`docs/model-verification.md` on GitHub: sections 2 (leakage), 3 (fresh splits with 95% CIs) and 4 (baselines).

> Under the hood, a FastAPI service runs four models, each a blend of a regularised logistic regression on
> clinical composites and shallow boosted trees, with exact attributions checked on every request. We verified
> them independently. With shuffled labels the full recipe scores an AUC of 0.48 to 0.52, so nothing leaks. On
> fresh splits never used in development, AUC is 0.94 for overall disease, 95% interval 0.91 to 0.97; 0.86 for
> the LAD, 0.82 to 0.91; and about 0.75 for the LCX and RCA, 0.70 to 0.81. They beat the Diamond-Forrester score
> and plain baselines on the same folds, and the browser code, including the physiology, is unit-tested.

## Scene 11: limitations and close (5:30–5:50)

**Open:** `?patient=triple&heart=atlas`, rotating slowly.

> The limits are real: one centre, 303 referred patients, no external validation, and the vessel models mostly
> measure how much disease there is, not which artery, so we show three independent estimates and never a
> culprit. CardioVision: coronary risk you can see, question and test. Thanks for watching.

## Rubric checklist

| Criterion (weight) | Where in the video | What the judge sees |
|---|---|---|
| Predictive Performance (30%) | 0:55, 4:40 | bands with observed rates, AUC per artery, held-out estimates vs angiography (including a false alarm); permutation test, fresh-split AUCs with 95% CIs, baselines |
| 3D Visualization (25%) | 0:00, 1:20, 2:00, 3:05, 3:55 | Atlas heart in the chest; function twin + ECG, inspector, bullseye, LBBB; reasoning streams; lab pulses, ischaemic territory and hypokinesis; 3D cohort map |
| Clinical Interpretability (20%) | 1:20, 2:00, 2:30, 3:05, 3:55 | inspector drivers; evidence by data type; what-if sensitivity with locked inputs explained; ischaemic cascade tied to dataset inputs and the flow-reserve curve; similar patients with angiograms |
| System Integration (15%) | 0:20, 1:20, 2:30, 4:40 | validated form with live updates and listed fill-ins; the twin follows the patient's values; models re-run while dragging; architecture |
| Technical Implementation (10%) | 4:40 | FastAPI + four blended models, exact attributions, independent verification, unit tests |

## YouTube details

- **Title:** CardioVision: coronary artery disease and vessel-level risk, explained in 3D (Multimodal AI Hackathon 2026, Track A)
- **Visibility:** Unlisted (not private).
- **Description:** one-paragraph summary, GitHub link, the safety disclaimer, chapters (0:00 Problem, 0:20 Input,
  0:55 Results, 1:20 Function twin, 2:00 Reasoning flow, 2:30 What-if, 3:05 Ischaemia lab, 3:55 Similar patients,
  4:20 Story mode, 4:40 How it was built and verified, 5:30 Limitations; adjust to the final cut) and the credits
  "3D heart model: Realistic Human Heart by neshallads, CC BY 4.0" and "Atlas heart and chest: BodyParts3D,
  © The Database Center for Life Science, CC BY 4.0".
