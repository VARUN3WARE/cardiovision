/**
 * Ischaemia lab panel: artery picker, narrowing and exercise sliders, supply vs demand read-outs, the
 * ischaemic cascade tied to the dataset inputs it corresponds to, and Gould's flow-reserve curve.
 * Every number comes from the textbook model in twin/coronaryPhysiology.ts (sources listed below);
 * the patient contributes only the resting pulse, systolic pressure and age the exercise starts from.
 */
import { useState, type CSSProperties } from "react";
import { VESSEL, VESSELS } from "../anatomy/registry";
import { FEATURE_LABELS } from "../featureLabels";
import { CASCADE, MAX_NARROWING, type LabPatient, type LabState } from "../twin/coronaryPhysiology";
import FlowReserveCurve from "./FlowReserveCurve";
import { exerciseLabel, type LabSettings } from "./labState";

export const LAB_BANNER = "Teaching sandbox with textbook physiology, not this patient's arteries.";

interface Props {
  settings: LabSettings;
  onChange: (s: LabSettings) => void;
  state: LabState;
  patient: LabPatient;
  onClose: () => void;
  className?: string;
  style?: CSSProperties;
  /** Show the sandbox banner at the top of the panel (narrow views, where it cannot sit over the heart). */
  banner?: boolean;
  /** Where the starting heart rate, pressure and age come from (labState.baselineText). */
  baseline: string;
  /** False with left bundle branch block: ST depression cannot be read, so the monitor does not draw it. */
  stReadable?: boolean;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const round10 = (x: number) => Math.round(x / 10) * 10;

export default function LabPanel({
  settings, onChange, state: s, patient, onClose, className = "", style, banner = false, baseline, stReadable = true,
}: Props) {
  const artery = VESSEL[settings.vessel];
  const [sources, setSources] = useState(false);
  const set = (patch: Partial<LabSettings>) => onChange({ ...settings, ...patch });
  return (
    <aside
      aria-label="Ischaemia lab"
      style={style}
      className={`flex flex-col gap-1 overflow-y-auto rounded border border-amber-600/60 bg-slate-950/90 px-2.5 py-1.5 text-[11px] leading-tight text-gray-300 shadow-lg [&>*]:shrink-0 ${className}`}
    >
      {banner && <LabBanner className="sticky top-0 z-10 -mx-1" />}
      <header className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 whitespace-nowrap">
          <h3 className="text-xs font-semibold text-white">Ischaemia lab</h3>
          <span className="rounded border border-amber-500/80 px-1 text-[8px] font-bold tracking-wide text-amber-300">SANDBOX</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setSources((o) => !o)}
            aria-expanded={sources}
            className="rounded px-1 text-[9px] text-gray-400 underline decoration-dotted hover:text-white"
            title="Where every number comes from, and the assumptions"
          >
            Sources
          </button>
          <button
            onClick={onClose}
            aria-label="Close the ischaemia lab"
            title="Close the lab: back to the patient's own twin"
            className="rounded px-1.5 text-gray-400 hover:bg-slate-700 hover:text-white"
          >
            ×
          </button>
        </div>
      </header>

      <div role="radiogroup" aria-label="Artery with the sandbox narrowing" className="flex overflow-hidden rounded">
        {VESSELS.map((v) => (
          <button
            key={v.id}
            role="radio"
            aria-checked={settings.vessel === v.id}
            onClick={() => set({ vessel: v.id })}
            title={`${v.name}: ${v.territory}`}
            className={`flex-1 py-1 text-[11px] font-semibold ${
              settings.vessel === v.id ? "bg-amber-500 text-slate-950" : "bg-slate-800 text-gray-300 hover:bg-slate-700"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      <label className="block">
        <span className="flex justify-between">
          <span>Narrowing <span className="text-gray-500">(diameter)</span></span>
          <b className="text-white">{settings.narrowing}%</b>
        </span>
        <input
          type="range"
          min={0}
          max={Math.round(MAX_NARROWING * 100)}
          step={1}
          value={settings.narrowing}
          onChange={(e) => set({ narrowing: Number(e.target.value) })}
          className="h-4 w-full accent-amber-500"
        />
      </label>
      <label className="block" title="Rest to peak: heart rate rises to 85% of the age-predicted maximum (220 − age), systolic BP by 50 mmHg (at most 210)">
        <span className="flex justify-between">
          <span>Exercise</span>
          <b className="text-white">
            {exerciseLabel(settings.exercise)} · HR {Math.round(s.hr)} · BP {Math.round(s.sbp)}
          </b>
        </span>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(settings.exercise * 100)}
          onChange={(e) => set({ exercise: Number(e.target.value) / 100 })}
          className="h-4 w-full accent-amber-500"
        />
        <span className="block text-[9px] text-gray-500">{baseline}</span>
      </label>

      <dl className="grid grid-cols-3 gap-x-2 gap-y-1 rounded bg-slate-900/80 p-1.5">
        <Stat label="Max flow" value={pct(s.qmaxFrac)} hint="Maximal (exercise) flow, as a share of a normal artery's" />
        <Stat label="Rest flow" value={pct(s.restFrac)} hint="Resting flow, as a share of a normal artery's" />
        <Stat label="Reserve" value={`${s.cfr.toFixed(1)}×`} hint="Coronary flow reserve: maximal ÷ resting flow (normal about 4)" />
        <Stat label="Pd/Pa max" value={s.ffr.toFixed(2)} hint="Pressure beyond the narrowing ÷ aortic pressure at maximal flow (what FFR measures)" />
        <Stat label="Demand" value={round10(s.rpp).toLocaleString("en-GB")} hint="Heart rate × systolic BP (rate-pressure product), which tracks the heart's oxygen use" />
        <Stat label="Supply max" value={round10(s.supplyRpp).toLocaleString("en-GB")} hint="The highest rate-pressure product this artery can feed: above it, its territory is short of blood" />
      </dl>

      <div title={`${artery.label} territory: ${artery.territory} (standard anatomy, right dominance)`}>
        <div className="flex justify-between">
          <span>{artery.label} territory short of supply</span>
          <b className={s.ischaemia > 0 ? "text-sky-300" : "text-white"}>{pct(s.ischaemia)}</b>
        </div>
        <div className="mt-0.5 h-1.5 overflow-hidden rounded bg-slate-800">
          <div className="h-full rounded bg-[#3155d8] transition-[width]" style={{ width: pct(s.ischaemia) }} />
        </div>
      </div>

      <section aria-label="Ischaemic cascade">
        <div className="flex items-baseline justify-between" title="Order from Nesto & Kowalchuk, Am J Cardiol 1987; the thresholds (5, 15, 25, 40% of demand unmet) are illustrative">
          <span className="font-semibold text-gray-100">Ischaemic cascade</span>
          <span className="text-[9px] text-gray-500">dataset input</span>
        </div>
        <ol className="mt-0.5 space-y-0.5">
          {CASCADE.map((c, i) => {
            const on = s.stage > i;
            const input = c.input ? FEATURE_LABELS[c.input]?.label ?? c.input : null;
            return (
              <li key={c.key} className={`flex items-center gap-1.5 rounded px-1 py-[1px] text-[10px] ${on ? "bg-sky-950/70" : ""}`}>
                <span
                  aria-hidden
                  className={`h-2 w-2 shrink-0 rounded-full ${on ? "bg-sky-300 shadow-[0_0_6px_#7dd3fc]" : "border border-slate-600"}`}
                />
                <span className={`whitespace-nowrap ${on ? "text-white" : "text-gray-500"}`}>
                  {i + 1}. {c.label}
                  {c.key === "st" && on && (stReadable
                    ? <span className="text-amber-300" title="Shown on the ECG monitor, marked lab"> · ECG</span>
                    : <span className="text-gray-400" title="ST depression cannot be read with left bundle branch block"> · not readable (LBBB)</span>)}
                </span>
                <span className={`ml-auto shrink-0 whitespace-nowrap text-[9px] ${on ? "text-sky-200" : "text-gray-600"}`} title={input ? `Dataset input: ${input}` : "Seen on perfusion imaging; not in the dataset"}>
                  {input ? shortInput(c.key) : "not an input"}
                </span>
              </li>
            );
          })}
        </ol>
        <div className="mt-0.5 text-[9px] leading-snug text-gray-500">
          Order: Nesto 1987 · thresholds illustrative
        </div>
      </section>

      <FlowReserveCurve
        vessel={settings.vessel}
        patient={patient}
        narrowing={s.narrowing}
        demand={s.demand}
        onPick={(n) => set({ narrowing: Math.round(n * 100) })}
      />

      <p className="rounded bg-slate-900/80 p-1.5 text-[10px] leading-snug text-gray-200">
        This is why typical angina and wall-motion abnormality are among the model&apos;s strongest inputs: they are late
        steps of this cascade. ST depression is a late step too but weighs less in the model; an ECG shows it only while
        ischaemia is present.
      </p>

      {sources && (
        <ul className="list-disc space-y-0.5 pl-3 text-[9px] leading-snug text-gray-500">
          <li>Pressure loss across the narrowing ΔP = F·Q + S·Q² (Gould, Cleve Clin Q 1980); curve shape as Gould, Am J Cardiol 1974, and Uren, N Engl J Med 1994.</li>
          <li>Normal flow reserve 4; Pd/Pa with venous pressure 0 (Pijls, Circulation 1993); artery sizes from Dodge, Circulation 1992.</li>
          <li>Demand ∝ heart rate × systolic BP (Gobel, Circulation 1978). Peak: 85% of 220 − age (ACC/AHA 2002), BP +50 up to 210 (Fletcher, Circulation 2013).</li>
          <li>Wall thickening vs flow: Gallagher, Am J Physiol 1984. Diastolic flow pattern: Seligman, EuroIntervention 2022.</li>
          <li>Cascade order: Nesto &amp; Kowalchuk, Am J Cardiol 1987; its thresholds (5, 15, 25, 40% of demand unmet) are illustrative.</li>
          <li>Assumed: 5 mm focal narrowing, resting velocity 8 cm/s, mean aortic pressure 100 mmHg. Territories assume standard anatomy, right dominance.</li>
        </ul>
      )}
    </aside>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div title={hint}>
      <dt className="text-[9px] text-gray-500">{label}</dt>
      <dd className="text-[12px] font-semibold text-white">{value}</dd>
    </div>
  );
}

function shortInput(key: (typeof CASCADE)[number]["key"]) {
  return key === "wallMotion" ? "Region RWMA" : key === "st" ? "ST depression" : key === "angina" ? "Typical chest pain" : "";
}

/** Always visible while the lab is on. */
export function LabBanner({ className = "", style }: { className?: string; style?: CSSProperties }) {
  return (
    <div
      role="note"
      style={style}
      className={`pointer-events-none rounded border border-amber-500/80 bg-amber-950/90 px-2.5 py-1 text-center text-[11px] font-semibold leading-snug text-amber-100 shadow ${className}`}
    >
      {LAB_BANNER}
    </div>
  );
}

/** What the lab view's colours mean (replaces the risk legend while the lab is on). */
export function LabLegend() {
  return (
    <div className="space-y-1 rounded bg-slate-900/85 p-1.5 text-[10px] leading-snug text-gray-300">
      <div className="flex items-center gap-1.5">
        <span className="h-1.5 w-4 shrink-0 rounded-full bg-[#ffe9b8] shadow-[0_0_6px_#ffe9b8]" /> Pulses: coronary blood flow, mostly while the heart relaxes
      </div>
      <div className="flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 shrink-0 rounded-full border-2 border-amber-500" /> Sandbox narrowing; fewer pulses get past it when flow falls short
      </div>
      <div className="flex items-center gap-1.5">
        <span className="h-2.5 w-4 shrink-0 rounded bg-[#1c3394]" /> Territory short of supply (darker = more); it squeezes less
      </div>
      <div className="text-gray-500">Arteries here show the sandbox, not the patient&apos;s estimates.</div>
    </div>
  );
}
