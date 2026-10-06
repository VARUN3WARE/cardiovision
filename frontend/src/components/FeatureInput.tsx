import { useState } from "react";

// Field names match the backend PatientFeatures schema exactly.
// Categorical fields use raw dataset values (Sex: "Male"|"Fmale", Y/N, etc.).
export interface FeatureValues {
  Age: number; Weight: number; Length: number; Sex: "Male" | "Fmale"; BMI: number;
  DM: number; HTN: number; CurrentSmoker: number; EX_Smoker: number; FH: number;
  Obesity: "Y" | "N"; CRF: "Y" | "N"; CVA: "Y" | "N"; AirwayDisease: "Y" | "N";
  ThyroidDisease: "Y" | "N"; CHF: "Y" | "N"; DLP: "Y" | "N";
  BP: number; PR: number; Edema: number;
  WeakPeripheralPulse: "Y" | "N"; LungRales: "Y" | "N";
  SystolicMurmur: "Y" | "N"; DiastolicMurmur: "Y" | "N";
  TypicalChestPain: number; Dyspnea: "Y" | "N"; FunctionClass: number;
  Atypical: "Y" | "N"; Nonanginal: "Y" | "N"; ExertionalCP: "Y" | "N"; LowTHAng: "Y" | "N";
  QWave: number; StElevation: number; StDepression: number; Tinversion: number;
  LVH: "Y" | "N"; PoorRProgression: "Y" | "N"; BBB: "N" | "LBBB" | "RBBB";
  FBS: number; CR: number; TG: number; LDL: number; HDL: number; BUN: number; ESR: number;
  HB: number; K: number; Na: number; WBC: number; Lymph: number; Neut: number; PLT: number;
  EF_TTE: number; RegionRWMA: number; VHD: "N" | "mild" | "Moderate" | "Severe";
}

// A high-risk example patient (matches the backend test case).
const EXAMPLE: FeatureValues = {
  Age: 60, Weight: 78, Length: 170, Sex: "Male", BMI: 27.0,
  DM: 1, HTN: 1, CurrentSmoker: 1, EX_Smoker: 0, FH: 1,
  Obesity: "Y", CRF: "N", CVA: "N", AirwayDisease: "N", ThyroidDisease: "N",
  CHF: "N", DLP: "Y",
  BP: 145, PR: 80, Edema: 0, WeakPeripheralPulse: "N", LungRales: "N",
  SystolicMurmur: "N", DiastolicMurmur: "N", TypicalChestPain: 1,
  Dyspnea: "Y", FunctionClass: 2, Atypical: "N", Nonanginal: "N",
  ExertionalCP: "N", LowTHAng: "Y",
  QWave: 0, StElevation: 1, StDepression: 1, Tinversion: 1,
  LVH: "N", PoorRProgression: "N", BBB: "N",
  FBS: 120, CR: 1.1, TG: 180, LDL: 150, HDL: 38, BUN: 16, ESR: 12,
  HB: 13.5, K: 4.2, Na: 140, WBC: 7500, Lymph: 30, Neut: 65, PLT: 250000,
  EF_TTE: 50, RegionRWMA: 1, VHD: "N",
};

interface Props {
  onSubmit: (features: Record<string, unknown>) => void;
  loading: boolean;
}

export default function FeatureInput({ onSubmit, loading }: Props) {
  const [v, setV] = useState<FeatureValues>(EXAMPLE);
  const setNum = (k: keyof FeatureValues, val: string) =>
    setV((s) => ({ ...s, [k]: Number(val) }));
  const setStr = (k: keyof FeatureValues, val: string) =>
    setV((s) => ({ ...s, [k]: val }));

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(v as unknown as Record<string, unknown>);
      }}
    >
      <Section title="Demographic">
        <Num label="Age" value={v.Age} onChange={(x) => setNum("Age", x)} />
        <Num label="Weight (kg)" value={v.Weight} onChange={(x) => setNum("Weight", x)} />
        <Num label="Height (cm)" value={v.Length} onChange={(x) => setNum("Length", x)} />
        <Sel label="Sex" value={v.Sex} onChange={(x) => setStr("Sex", x)}
          opts={[["Male", "Male"], ["Fmale", "Fmale"]]} />
        <Num label="BMI" value={v.BMI} onChange={(x) => setNum("BMI", x)} />
      </Section>

      <Section title="History / Risk Factors">
        <Bin label="Diabetes (DM)" value={v.DM} onChange={(x) => setNum("DM", x)} />
        <Bin label="Hypertension" value={v.HTN} onChange={(x) => setNum("HTN", x)} />
        <Bin label="Current Smoker" value={v.CurrentSmoker} onChange={(x) => setNum("CurrentSmoker", x)} />
        <Bin label="Ex-Smoker" value={v.EX_Smoker} onChange={(x) => setNum("EX_Smoker", x)} />
        <Bin label="Family History" value={v.FH} onChange={(x) => setNum("FH", x)} />
        <YN label="Obesity" value={v.Obesity} onChange={(x) => setStr("Obesity", x)} />
        <YN label="CRF" value={v.CRF} onChange={(x) => setStr("CRF", x)} />
        <YN label="CVA" value={v.CVA} onChange={(x) => setStr("CVA", x)} />
        <YN label="Airway Disease" value={v.AirwayDisease} onChange={(x) => setStr("AirwayDisease", x)} />
        <YN label="Thyroid Disease" value={v.ThyroidDisease} onChange={(x) => setStr("ThyroidDisease", x)} />
        <YN label="CHF" value={v.CHF} onChange={(x) => setStr("CHF", x)} />
        <YN label="DLP" value={v.DLP} onChange={(x) => setStr("DLP", x)} />
      </Section>

      <Section title="Examination">
        <Num label="BP (mmHg)" value={v.BP} onChange={(x) => setNum("BP", x)} />
        <Num label="Pulse (bpm)" value={v.PR} onChange={(x) => setNum("PR", x)} />
        <Bin label="Edema" value={v.Edema} onChange={(x) => setNum("Edema", x)} />
        <YN label="Weak Peripheral Pulse" value={v.WeakPeripheralPulse} onChange={(x) => setStr("WeakPeripheralPulse", x)} />
        <YN label="Lung Rales" value={v.LungRales} onChange={(x) => setStr("LungRales", x)} />
        <YN label="Systolic Murmur" value={v.SystolicMurmur} onChange={(x) => setStr("SystolicMurmur", x)} />
        <YN label="Diastolic Murmur" value={v.DiastolicMurmur} onChange={(x) => setStr("DiastolicMurmur", x)} />
        <Bin label="Typical Chest Pain" value={v.TypicalChestPain} onChange={(x) => setNum("TypicalChestPain", x)} />
        <YN label="Dyspnea" value={v.Dyspnea} onChange={(x) => setStr("Dyspnea", x)} />
        <Num label="Function Class" value={v.FunctionClass} onChange={(x) => setNum("FunctionClass", x)} />
        <YN label="Atypical" value={v.Atypical} onChange={(x) => setStr("Atypical", x)} />
        <YN label="Nonanginal" value={v.Nonanginal} onChange={(x) => setStr("Nonanginal", x)} />
        <YN label="Exertional CP" value={v.ExertionalCP} onChange={(x) => setStr("ExertionalCP", x)} />
        <YN label="LowTH Ang" value={v.LowTHAng} onChange={(x) => setStr("LowTHAng", x)} />
      </Section>

      <Section title="ECG">
        <Bin label="Q Wave" value={v.QWave} onChange={(x) => setNum("QWave", x)} />
        <Bin label="ST Elevation" value={v.StElevation} onChange={(x) => setNum("StElevation", x)} />
        <Bin label="ST Depression" value={v.StDepression} onChange={(x) => setNum("StDepression", x)} />
        <Bin label="T Inversion" value={v.Tinversion} onChange={(x) => setNum("Tinversion", x)} />
        <YN label="LVH" value={v.LVH} onChange={(x) => setStr("LVH", x)} />
        <YN label="Poor R Progression" value={v.PoorRProgression} onChange={(x) => setStr("PoorRProgression", x)} />
        <Sel label="BBB" value={v.BBB} onChange={(x) => setStr("BBB", x)}
          opts={[["N", "None"], ["LBBB", "LBBB"], ["RBBB", "RBBB"]]} />
      </Section>

      <Section title="Laboratory">
        <Num label="FBS" value={v.FBS} onChange={(x) => setNum("FBS", x)} />
        <Num label="Creatinine" value={v.CR} onChange={(x) => setNum("CR", x)} />
        <Num label="Triglycerides" value={v.TG} onChange={(x) => setNum("TG", x)} />
        <Num label="LDL" value={v.LDL} onChange={(x) => setNum("LDL", x)} />
        <Num label="HDL" value={v.HDL} onChange={(x) => setNum("HDL", x)} />
        <Num label="BUN" value={v.BUN} onChange={(x) => setNum("BUN", x)} />
        <Num label="ESR" value={v.ESR} onChange={(x) => setNum("ESR", x)} />
        <Num label="Hemoglobin" value={v.HB} onChange={(x) => setNum("HB", x)} />
        <Num label="Potassium" value={v.K} onChange={(x) => setNum("K", x)} />
        <Num label="Sodium" value={v.Na} onChange={(x) => setNum("Na", x)} />
        <Num label="WBC" value={v.WBC} onChange={(x) => setNum("WBC", x)} />
        <Num label="Lymph %" value={v.Lymph} onChange={(x) => setNum("Lymph", x)} />
        <Num label="Neut %" value={v.Neut} onChange={(x) => setNum("Neut", x)} />
        <Num label="Platelets" value={v.PLT} onChange={(x) => setNum("PLT", x)} />
      </Section>

      <Section title="Echocardiogram">
        <Num label="Ejection Fraction" value={v.EF_TTE} onChange={(x) => setNum("EF_TTE", x)} />
        <Bin label="Region RWMA" value={v.RegionRWMA} onChange={(x) => setNum("RegionRWMA", x)} />
        <Sel label="Valvular HD" value={v.VHD} onChange={(x) => setStr("VHD", x)}
          opts={[["N", "None"], ["mild", "Mild"], ["Moderate", "Moderate"], ["Severe", "Severe"]]} />
      </Section>

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded bg-red-700 px-4 py-2 font-semibold text-white hover:bg-red-800 disabled:opacity-50"
      >
        {loading ? "Predicting…" : "Predict Risk"}
      </button>
    </form>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-red-300">{title}</div>
      <div className="grid grid-cols-2 gap-2">{children}</div>
    </div>
  );
}

function Num({ label, value, onChange }: { label: string; value: number; onChange: (v: string) => void }) {
  return (
    <label className="block text-xs">
      <span className="text-gray-300">{label}</span>
      <input type="number" value={value} onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white" />
    </label>
  );
}

function Bin({ label, value, onChange }: { label: string; value: number; onChange: (v: string) => void }) {
  return (
    <label className="block text-xs">
      <span className="text-gray-300">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white">
        <option value={0}>No</option><option value={1}>Yes</option>
      </select>
    </label>
  );
}

function YN({ label, value, onChange }: { label: string; value: "Y" | "N"; onChange: (v: string) => void }) {
  return (
    <label className="block text-xs">
      <span className="text-gray-300">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white">
        <option value="N">No</option><option value="Y">Yes</option>
      </select>
    </label>
  );
}

function Sel({ label, value, onChange, opts }: {
  label: string; value: string; onChange: (v: string) => void; opts: [string, string][];
}) {
  return (
    <label className="block text-xs">
      <span className="text-gray-300">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white">
        {opts.map(([val, lbl]) => <option key={val} value={val}>{lbl}</option>)}
      </select>
    </label>
  );
}
