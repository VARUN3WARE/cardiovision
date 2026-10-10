/**
 * Ischaemia lab panel: artery picker, narrowing and exercise sliders, supply vs demand read-outs, the
 * ischaemic cascade tied to the dataset inputs it corresponds to, and Gould's flow-reserve curve.
 * Every number comes from the textbook model in twin/coronaryPhysiology.ts (sources listed below);
 * the patient contributes only the recorded resting pulse, systolic pressure and age the exercise starts from.
 *
 * Two layouts: docked at the side of the 3D view, or a bottom sheet (narrow views) that leads with what the
 * lab teaches (the territory short of supply and the cascade), is capped in height and can be collapsed.
 */
import { useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { VESSEL, VESSELS } from "../anatomy/registry";
import { FEATURE_LABELS } from "../featureLabels";
import { CASCADE, MAX_NARROWING, RPP_REF, labState, type LabPatient, type LabState } from "../twin/coronaryPhysiology";
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
  /** "side": docked beside the heart; "sheet": a bottom sheet (banner first, then the cascade, collapsible). */
  layout?: "side" | "sheet";
  /** Sheet only: collapsed to the banner, the header and the territory bar. */
  collapsed?: boolean;
  onCollapse?: (collapsed: boolean) => void;
  /** Where the starting heart rate, pressure and age come from (labState.baselineText). */
  baseline: string;
  /** False with left bundle branch block: ST depression cannot be read, so the monitor does not draw it. */
  stReadable?: boolean;
  /** Shown at the end of the panel (the sheet carries the model credit there). */
  footer?: ReactNode;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const round10 = (x: number) => Math.round(x / 10) * 10;
const GROUPED = new Intl.NumberFormat("en-GB"); // one formatter (toLocaleString builds one per call)
const thousands = (x: number) => GROUPED.format(round10(x));

export default function LabPanel({
  settings, onChange, state: s, patient, onClose, className = "", style, layout = "side", collapsed = false, onCollapse,
  baseline, stReadable = true, footer,
}: Props) {
  const artery = VESSEL[settings.vessel];
  const [sources, setSources] = useState(false);
  const set = (patch: Partial<LabSettings>) => onChange({ ...settings, ...patch });
  const sheet = layout === "sheet";
  // Reserve with no narrowing depends on resting demand: 4 at the reference pulse × BP, less when it is higher.
  const open = useMemo(() => labState(settings.vessel, 0, 0, patient), [settings.vessel, patient]);
  const restRpp = patient.hr * patient.sbp;

  const header = (
    <header className="flex items-center justify-between gap-2">
      <div className="flex min-w-0 items-center gap-1.5 whitespace-nowrap">
        <h3 className="truncate text-xs font-semibold text-white">Ischaemia lab</h3>
        {/* The sheet's banner, right above, already says it. */}
        {!sheet && <span className="rounded border border-amber-500/80 px-1 text-[8px] font-bold tracking-wide text-amber-300">SANDBOX</span>}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        {!(sheet && collapsed) && (
          <button
            onClick={() => setSources((o) => !o)}
            aria-expanded={sources}
            className="rounded px-1 text-[9px] text-gray-400 underline decoration-dotted hover:text-white"
            title="Where every number comes from, and the assumptions"
          >
            Sources
          </button>
        )}
        {sheet && onCollapse && (
          <button
            onClick={() => onCollapse(!collapsed)}
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Expand the lab panel" : "Collapse the lab panel"}
            title={collapsed ? "Show the lab's controls and read-outs" : "Collapse: more room for the heart"}
            className="rounded px-1.5 text-gray-300 hover:bg-slate-700 hover:text-white"
          >
            {collapsed ? "▴" : "▾"}
          </button>
        )}
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
  );

  const supply = (
    <div title={`${artery.label} territory: ${artery.territory} (standard anatomy, right dominance)`}>
      <div className="flex justify-between gap-2">
        <span>{artery.label} territory short of supply</span>
        <b className={s.ischaemia > 0 ? "text-sky-300" : "text-white"}>{pct(s.ischaemia)}</b>
      </div>
      <div className="mt-0.5 h-1.5 overflow-hidden rounded bg-slate-800">
        <div className="h-full rounded bg-[#3155d8] transition-[width]" style={{ width: pct(s.ischaemia) }} />
      </div>
    </div>
  );

  const cascade = (
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
            <li key={c.key} className={`flex items-start gap-1.5 rounded px-1 py-[1px] text-[10px] ${on ? "bg-sky-950/70" : ""}`}>
              <span
                aria-hidden
                className={`mt-[3px] h-2 w-2 shrink-0 rounded-full ${on ? "bg-sky-300 shadow-[0_0_6px_#7dd3fc]" : "border border-slate-600"}`}
              />
              <span className={`min-w-0 flex-1 ${on ? "text-white" : "text-gray-500"}`}>
                {i + 1}. {c.label}
                {c.key === "st" && on && stReadable && (
                  <span className="text-amber-300" title="Shown on the ECG monitor, marked lab"> · ECG</span>
                )}
                {"note" in c && <span className={`text-[9px] ${on ? "text-gray-300" : "text-gray-600"}`}> · {c.note}</span>}
                {c.key === "st" && on && !stReadable && (
                  <span className="block text-[9px] text-gray-400" title="ST depression cannot be read with left bundle branch block">
                    not readable with LBBB
                  </span>
                )}
              </span>
              <span className={`shrink-0 whitespace-nowrap text-[9px] ${on ? "text-sky-200" : "text-gray-600"}`} title={input ? `Dataset input: ${input}` : "Seen on perfusion imaging; not in the dataset"}>
                {input ? shortInput(c.key) : "not an input"}
              </span>
            </li>
          );
        })}
      </ol>
      <div className="mt-0.5 text-[9px] leading-snug text-gray-500">Order: Nesto 1987 · thresholds illustrative</div>
    </section>
  );

  const controls = (
    <>
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
        <span className="flex justify-between gap-2">
          <span>Exercise</span>
          <b className="text-right text-white">
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

      <div className="rounded bg-slate-900/80 p-1.5">
        <dl className="grid grid-cols-3 gap-x-2 gap-y-1">
          <Stat label="Max flow" value={pct(s.qmaxFrac)} hint="Maximal flow (full vasodilation), as a share of a normal artery's" />
          <Stat label="Rest flow" value={pct(s.restFrac)} hint="Resting flow, as a share of a normal artery's" />
          <Stat
            label="Reserve"
            value={`${s.cfr.toFixed(1)}×`}
            hint={`Coronary flow reserve: maximal ÷ resting flow. With no narrowing it is about 4 at a resting pulse × BP of ${thousands(RPP_REF)}; a higher resting demand lowers it.`}
          />
          <Stat label="Pd/Pa max" value={s.ffr.toFixed(2)} hint="Pressure beyond the narrowing ÷ aortic pressure at maximal flow (what FFR measures)" />
          <Stat label="Demand" value={thousands(s.rpp)} hint="Heart rate × systolic BP (rate-pressure product), which tracks the heart's oxygen use" />
          <Stat label="Supply max" value={thousands(s.supplyRpp)} hint="The highest rate-pressure product this artery can feed: above it, its territory is short of blood" />
        </dl>
        <p className="mt-1 text-[9px] leading-snug text-gray-500" title="Reserve is maximal ÷ resting flow, and resting flow follows the resting pulse × BP">
          {Math.abs(restRpp - RPP_REF) < 50
            ? `No narrowing: reserve about 4 (normal) at this resting pulse × BP of ${thousands(restRpp)}.`
            : `No narrowing: reserve 4 at a resting pulse × BP of ${thousands(RPP_REF)}, ${open.cfr.toFixed(1)} at this heart's ${thousands(restRpp)}.`}
        </p>
      </div>
    </>
  );

  const curve = (
    <FlowReserveCurve
      vessel={settings.vessel}
      patient={patient}
      narrowing={s.narrowing}
      demand={s.demand}
      onPick={(n) => set({ narrowing: Math.round(n * 100) })}
    />
  );

  const closing = (
    <p className="rounded bg-slate-900/80 p-1.5 text-[10px] leading-snug text-gray-200">
      This helps explain why typical angina and wall-motion abnormality weigh heavily in the overall CAD model
      (recorded at rest, where a wall-motion abnormality often means earlier damage). Ischaemic ST depression lasts
      only while the ischaemia does, so a resting ECG often misses it.
    </p>
  );

  const sourceList = sources && (
    <ul className="list-disc space-y-0.5 pl-3 text-[9px] leading-snug text-gray-500">
      <li>Pressure loss across the narrowing ΔP = F·Q + S·Q² (Gould, Cleve Clin Q 1980); curve shape as Gould, Am J Cardiol 1974, and Uren, N Engl J Med 1994.</li>
      <li>Normal flow reserve 4 at the cohort&apos;s median resting pulse × BP ({thousands(RPP_REF)}); Pd/Pa with venous pressure 0 (Pijls, Circulation 1993); artery sizes from Dodge, Circulation 1992.</li>
      <li>Demand ∝ heart rate × systolic BP (Gobel, Circulation 1978). Peak: 85% of 220 − age (ACC/AHA 2002), BP +50 up to 210 (Fletcher, Circulation 2013).</li>
      <li>Wall thickening vs flow: Gallagher, Am J Physiol 1984. Diastolic flow pattern: Seligman, EuroIntervention 2022.</li>
      <li>Cascade order: Nesto &amp; Kowalchuk, Am J Cardiol 1987; its thresholds (5, 15, 25, 40% of demand unmet) are illustrative. Angina is its least consistent step: ischaemia can be silent.</li>
      <li>Assumed: 5 mm focal narrowing, resting velocity 8 cm/s, mean aortic pressure 100 mmHg. Territories assume standard anatomy, right dominance.</li>
      <li>No collateral flow: collaterals can feed part of a territory beyond a severe narrowing; this model leaves them out.</li>
    </ul>
  );

  return (
    <aside
      aria-label="Ischaemia lab"
      style={style}
      className={`flex flex-col gap-1 overflow-y-auto rounded border border-amber-600/60 bg-slate-950/90 px-2.5 py-1.5 text-[11px] leading-tight text-gray-300 shadow-lg [&>*]:shrink-0 ${className}`}
    >
      {sheet ? (
        <>
          <LabBanner className="sticky top-0 z-10 -mx-1" compact />
          {header}
          {supply}
          {!collapsed && (
            <>
              {cascade}
              {controls}
              {curve}
              {closing}
              {footer}
              {sourceList}
            </>
          )}
        </>
      ) : (
        <>
          {header}
          {controls}
          {supply}
          {cascade}
          {curve}
          {closing}
          {footer}
          {sourceList}
        </>
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

/** Always visible while the lab is on (over the heart beside the docked panel, or the sheet's first line). */
export function LabBanner({ className = "", style, compact = false }: { className?: string; style?: CSSProperties; compact?: boolean }) {
  return (
    <div
      role="note"
      style={style}
      className={`pointer-events-none rounded border border-amber-500/80 bg-amber-950/95 text-center font-semibold leading-snug text-amber-100 shadow ${
        compact ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-[11px]"} ${className}`}
    >
      {LAB_BANNER}
    </div>
  );
}

/**
 * What the lab view's colours mean (replaces the risk legend while the lab is on): a block beside the docked
 * panel (with the model credit as `children`), or one line above the bottom sheet.
 */
export function LabLegend({ line = false, children }: { line?: boolean; children?: ReactNode }) {
  if (line) {
    return (
      <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-0.5 rounded bg-slate-900/85 px-1.5 py-0.5 text-[9px] leading-tight text-gray-300">
        <span className="inline-flex items-center gap-1"><PulseKey />blood-flow pulses</span>
        <span className="inline-flex items-center gap-1"><RingKey />sandbox narrowing</span>
        <span className="inline-flex items-center gap-1"><TerritoryKey />short of supply</span>
      </div>
    );
  }
  return (
    <div className="space-y-1 rounded bg-slate-900/85 p-1.5 text-[10px] leading-snug text-gray-300">
      <div className="flex items-center gap-1.5">
        <PulseKey /> Pulses: coronary blood flow, mostly while the heart relaxes
      </div>
      <div className="flex items-center gap-1.5">
        <RingKey /> Sandbox narrowing: when flow falls short, the whole artery carries fewer pulses
      </div>
      <div className="flex items-center gap-1.5">
        <TerritoryKey /> Territory short of supply (darker = more); it squeezes less
      </div>
      <div className="text-gray-500">Arteries here show the sandbox, not the patient&apos;s estimates.</div>
      {children}
    </div>
  );
}

const PulseKey = () => <span aria-hidden className="h-1.5 w-4 shrink-0 rounded-full bg-[#ffe9b8] shadow-[0_0_6px_#ffe9b8]" />;
const RingKey = () => <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-full border-2 border-amber-500" />;
const TerritoryKey = () => <span aria-hidden className="h-2.5 w-4 shrink-0 rounded bg-[#1c3394]" />;
