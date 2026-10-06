import { useState } from "react";

export interface FeatureValues {
  age: number;
  sex: number;
  bp: number;
  pulse: number;
  cholesterol: number;
  st_elevation: number;
  t_inversion: number;
  lvh: number;
  region_rwma: number;
}

const DEFAULTS: FeatureValues = {
  age: 55,
  sex: 1,
  bp: 130,
  pulse: 75,
  cholesterol: 220,
  st_elevation: 0,
  t_inversion: 0,
  lvh: 0,
  region_rwma: 0,
};

interface Props {
  onSubmit: (features: Record<string, number>) => void;
  loading: boolean;
}

export default function FeatureInput({ onSubmit, loading }: Props) {
  const [v, setV] = useState<FeatureValues>(DEFAULTS);
  const set = (k: keyof FeatureValues, val: string) =>
    setV((s) => ({ ...s, [k]: Number(val) }));

  return (
    <form
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(v as unknown as Record<string, number>);
      }}
    >
      <div className="grid grid-cols-2 gap-3">
        <Field label="Age" value={v.age} onChange={(x) => set("age", x)} />
        <SelectField
          label="Sex"
          value={v.sex}
          onChange={(x) => set("sex", x)}
          options={[{ value: 1, label: "Male" }, { value: 0, label: "Female" }]}
        />
        <Field label="Blood Pressure (mmHg)" value={v.bp} onChange={(x) => set("bp", x)} />
        <Field label="Pulse (bpm)" value={v.pulse} onChange={(x) => set("pulse", x)} />
        <Field label="Cholesterol (mg/dL)" value={v.cholesterol} onChange={(x) => set("cholesterol", x)} />
        <ToggleField label="ST Elevation" value={v.st_elevation} onChange={(x) => set("st_elevation", x)} />
        <ToggleField label="T Inversion" value={v.t_inversion} onChange={(x) => set("t_inversion", x)} />
        <ToggleField label="LVH" value={v.lvh} onChange={(x) => set("lvh", x)} />
        <ToggleField label="Region RWMA" value={v.region_rwma} onChange={(x) => set("region_rwma", x)} />
      </div>
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

function Field({ label, value, onChange }: { label: string; value: number; onChange: (v: string) => void }) {
  return (
    <label className="block text-sm">
      <span className="text-gray-300">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded bg-slate-800 px-2 py-1 text-white"
      />
    </label>
  );
}

function SelectField({
  label, value, onChange, options,
}: { label: string; value: number; onChange: (v: string) => void; options: { value: number; label: string }[] }) {
  return (
    <label className="block text-sm">
      <span className="text-gray-300">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded bg-slate-800 px-2 py-1 text-white"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

function ToggleField({ label, value, onChange }: { label: string; value: number; onChange: (v: string) => void }) {
  return (
    <label className="block text-sm">
      <span className="text-gray-300">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full rounded bg-slate-800 px-2 py-1 text-white"
      >
        <option value={0}>No</option>
        <option value={1}>Yes</option>
      </select>
    </label>
  );
}
