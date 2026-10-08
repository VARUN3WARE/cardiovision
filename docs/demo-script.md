# CardioVision demo video script (target 7 minutes)

Requirements (Track A): 3–10 minutes, on YouTube (unlisted is fine), English audio or subtitles, showing the
working system, the feature-input workflow, the 3D visualisation interactions and the technical
implementation. Narration below is about 950 words, roughly 7 minutes at a calm pace.

## Before recording

- Start both servers (`uvicorn app.main:app --port 8000` in `backend/`, `npm run dev` in `frontend/`) and
  open http://localhost:5173 in Chrome, full screen, 1920×1080, browser zoom 100%. Load the page once
  beforehand so the 3D model and shaders are cached.
- 3D toolbar: **Realistic**, **Arteries**, **HD on**. Close developer tools and notifications.
- Record with OBS (or similar) at 1080p, 30 fps, with a separate microphone track. Keep the mouse slow.
- Add English subtitles on YouTube (auto-captions, then correct "LAD", "LCX", "RCA", "SHAP").

## Scene 1: the problem and the result (0:00–0:40)

**Screen:** the app on the demo patient; slowly rotate the heart.

> Coronary artery disease is the leading cause of death worldwide, but a risk percentage on its own doesn't
> tell a clinician *where* disease is likely. CardioVision predicts overall coronary artery disease and
> significant narrowing of each of the three main coronary arteries (the LAD, LCX and RCA) from 55 routine
> clinical findings, and maps each prediction onto the artery it belongs to on an interactive 3D heart, with
> an explanation of every estimate.
>
> Note the banner: this is decision support for education only, trained on 303 patients from a single
> centre. It is not a diagnosis.

## Scene 2: entering a patient (0:40–1:55)

**Screen:** click **New patient**, then **Quick (19 key inputs)**. Type Age 58, Sex Female, Weight 72,
Height 160; choose Chest pain **Typical angina**; Diabetes **Yes**; Systolic BP 150; Ejection fraction 45;
Wall-motion score 2. Point at the BMI, the units and the live update.

> Let's enter a new patient. Quick mode asks only for the nineteen most informative inputs; anything not
> entered is filled with a typical cohort value and listed in the results. Units and plausible ranges are
> built in, BMI is calculated, and chest pain and smoking are single choices, as they are clinically.
> Estimates update live as I type: there is no separate submit step.

**Screen:** type Age **5**: the inline error appears and the results show "Fix the highlighted fields".
Correct it to 58.

> Input is validated as you type. An implausible age is flagged at the field and never reaches the model.

**Screen:** switch to **Full (all 55)** briefly and scroll through the sections.

> Full mode exposes all 55 inputs, grouped as a clinician would record them: symptoms, risk factors, vitals,
> echocardiography, ECG and laboratory values.

## Scene 3: reading the result (1:55–2:55)

**Screen:** the Risk estimate panel. Hover the CAD band text, then the vessel cards and their AUC lines.

> The overall estimate is shown with a likelihood band, and next to it the rate actually observed in
> validation, for example "In validation, 93% of patients in this band had CAD on angiography." Each vessel
> has its own probability and band, plus its cross-validated AUC. The LCX and RCA models are weaker (AUC 0.76
> and 0.74); the RCA card, below 0.75, is dashed and labelled as indicative. We would rather show that than hide it.

**Screen:** click **Single-vessel (LAD)** under "Real patients from the dataset".

> These are real dataset patients. This one had isolated LAD disease on angiography. Under each card is the
> *held-out* estimate, from models that never saw this patient, with a match, false-alarm or missed badge.
> Here LAD and LCX match, and RCA is a false alarm: an honest picture of what the models can and can't do.

## Scene 4: the 3D heart (2:55–4:25)

**Screen:** rotate and zoom the realistic heart. Point at the LAD (red), RCA and LCX.

> The heart is an open-licence sculpted model. We traced the coronary tree along its own grooves: the LAD
> with its diagonal branches, the circumflex with its marginal branches, and the right coronary artery down to
> the posterior descending artery. Each vessel is coloured by its own model's probability, here a red LAD
> and green-to-amber circumflex and RCA. The heart beats at this patient's pulse rate.

**Screen:** click **Territories**.

> Territories mode shades the heart muscle by the risk of the artery that supplies it: the anterior wall and
> apex for the LAD, the lateral wall for the circumflex, and the right ventricle and inferior wall for the RCA.

**Screen:** click **Arteries**, then click the **RCA** label; the camera swings to the inferior wall.

> Selecting an artery, by clicking it, its label, or simply the heart wall, turns the camera to that
> vessel's territory, makes it pulse, and switches the explanation panel to that vessel's model.

**Screen:** click **Anatomy**, open pin 3 (pulmonary trunk). Then toggle **Schematic** and back to
**Realistic**.

> Numbered pins add anatomy notes for teaching, and a schematic heart, with epicardial fat and the cardiac
> veins, is one click away; it is also the automatic fallback if the 3D model can't load. Everything runs in
> the browser on an integrated GPU, and an HD switch turns off post-processing on slower machines.

## Scene 5: why this estimate? (4:25–5:35)

**Screen:** the "Why this estimate?" panel on the **LAD** tab. Then open **Physiological breakdown**.

> Every estimate is explained. A summary sentence names the main reasons. Each bar shows how much a finding
> pushed the estimate up or down, with its odds multiplier, and the list below walks from the model's
> starting point to this patient's estimate in percentage points. Derived clinical scores, such as the
> Diamond-Forrester pre-test probability, are labelled with the inputs they come from.
>
> These attributions are exact: for each model the contributions add up precisely to the model's output, and
> the server checks that on every request. The physiological breakdown lists all 55 measurements, grouped by
> body system, with the patient's value and its contribution.

## Scene 6: how it was built (5:35–6:45)

**Screen:** `docs/CardioVision-Technical-Documentation.pdf`: Figure 2 (architecture), the results table and
Figure 3 (ROC and calibration). Optionally the API docs at http://localhost:8000/docs.

> Under the hood, a FastAPI service validates the input, runs four models and returns probabilities,
> thresholds, calibrated bands and attributions in about 130 milliseconds.
>
> We were careful about evaluation, because this dataset is small and published results are often
> optimistic. Every experiment used the same leakage-safe protocol: five-fold cross-validation repeated five
> times, with everything learned from the data refitted inside each training fold. Six modelling strategies
> were tried; each result was independently re-run on unseen splits with a leakage audit, and the final
> choice was made on fresh splits with a rule fixed in advance.
>
> The winning design is, for every target, a blend of a regularised logistic regression on clinical composite
> features and a shallow boosted-tree model. Cross-validated AUC is 0.94 for overall CAD, 0.86 for the LAD,
> 0.76 for the circumflex and 0.74 for the RCA, beating our first model on every repeat, with much better
> calibration. These numbers sit at the top of what protocol-clean studies on this dataset report.

## Scene 7: limitations and close (6:45–7:15)

**Screen:** back to the app, slowly rotating the heart.

> The limits are real: one centre, 303 patients, no external validation, and the vessel models are only
> partly vessel-specific. That is why the interface shows validation numbers, held-out results and the safety
> notice alongside every estimate.
>
> CardioVision: coronary risk you can see, inspect and question. The code, the trained-model recipe and the
> documentation are on GitHub. Thanks for watching.

## YouTube details

- **Title:** CardioVision: coronary artery disease and vessel-level risk, explained in 3D (Multimodal AI Hackathon 2026, Track A)
- **Visibility:** Unlisted (not private).
- **Description:** one-paragraph summary, GitHub link, the safety disclaimer, and the credit "3D heart model:
  Realistic Human Heart by neshallads, CC BY 4.0".
