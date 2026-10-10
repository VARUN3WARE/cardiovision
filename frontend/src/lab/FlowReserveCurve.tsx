import { useMemo, useState } from "react";
import type { Vessel } from "../anatomy/registry";
import { MAX_NARROWING, RPP_REF, flowReserveCurve, type LabPatient } from "../twin/coronaryPhysiology";

// Two series on the dark panel surface (validated for colour-vision deficiency against #0f172a).
const MAX_COLOR = "#3987e5";
const REST_COLOR = "#d95926";
const W = 224;
const H = 72;
const M = { l: 20, r: 6, t: 6, b: 16 };

interface Props {
  vessel: Vessel;
  patient: LabPatient;
  narrowing: number; // 0..0.95
  demand: number; // demand at the current exercise level, in reference resting flows
  onPick?: (narrowing: number) => void;
}

/**
 * Gould's curve for the sandbox artery: resting and maximal flow (in units of this heart's resting flow
 * through a normal artery) against narrowing, the demand at the current exercise level, and where the
 * slider sits. Where maximal flow falls below demand, the territory is ischaemic (shaded).
 */
export default function FlowReserveCurve({ vessel, patient, narrowing, demand, onPick }: Props) {
  const curve = useMemo(() => flowReserveCurve(vessel, patient), [vessel, patient]);
  const [hover, setHover] = useState<number | null>(null);
  const d0 = (patient.hr * patient.sbp) / RPP_REF;
  const demandY = demand / d0;
  const yMax = Math.max(2, Math.ceil(Math.max(curve[0].max, demandY) * 1.08));
  const x = (n: number) => M.l + (n / MAX_NARROWING) * (W - M.l - M.r);
  const y = (v: number) => H - M.b - (Math.min(v, yMax) / yMax) * (H - M.t - M.b);
  const path = (key: "rest" | "max") => curve.map((c, i) => `${i ? "L" : "M"}${x(c.narrowing).toFixed(1)},${y(c[key]).toFixed(1)}`).join("");
  // Ischaemic zone: between the demand line and maximal flow, wherever maximal flow is the lower.
  const short = curve.filter((c) => c.max < demandY);
  const zone = short.length
    ? `M${x(short[0].narrowing)},${y(demandY)}` + short.map((c) => `L${x(c.narrowing).toFixed(1)},${y(c.max).toFixed(1)}`).join("") +
      `L${x(short[short.length - 1].narrowing)},${y(demandY)}Z`
    : "";
  const at = (n: number) => curve[Math.round((n / MAX_NARROWING) * (curve.length - 1))];
  const cur = at(narrowing);
  const read = hover === null ? null : at(hover);
  const fromPointer = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    return Math.min(MAX_NARROWING, Math.max(0, ((px - M.l) / (W - M.l - M.r)) * MAX_NARROWING));
  };
  const ticks = Array.from({ length: yMax + 1 }, (_, i) => i).filter((i) => yMax <= 5 || i % 2 === 0);

  return (
    <figure className="space-y-0.5">
      <div className="flex flex-wrap gap-x-2 text-[9px] text-gray-400" aria-hidden>
        <span className="inline-flex items-center gap-1"><LineKey color={MAX_COLOR} />maximal</span>
        <span className="inline-flex items-center gap-1"><LineKey color={REST_COLOR} />resting flow</span>
        <span className="inline-flex items-center gap-1"><LineKey color="#94a3b8" dashed />demand now</span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full cursor-crosshair select-none"
        role="img"
        aria-label={`Resting and maximal coronary flow against narrowing for this artery. Maximal flow starts to fall at about 50% narrowing; resting flow holds until about 85%. Current narrowing ${Math.round(narrowing * 100)}%: maximal flow ${cur.max.toFixed(1)} times resting.`}
        onPointerMove={(e) => setHover(fromPointer(e))}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => onPick?.(Math.round(fromPointer(e) * 100) / 100)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="#1e293b" strokeWidth={1} />
            <text x={M.l - 4} y={y(t) + 3} textAnchor="end" fontSize={8} fill="#64748b">{t}×</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 0.95].map((n) => (
          <text key={n} x={x(n)} y={H - 4} textAnchor={n === 0 ? "start" : n === 0.95 ? "end" : "middle"} fontSize={8} fill={n === 0.5 ? "#cbd5e1" : "#64748b"}>
            {Math.round(n * 100)}%
          </text>
        ))}
        {/* The ≥50% threshold used by clinicians and by the dataset's labels. */}
        <line x1={x(0.5)} x2={x(0.5)} y1={M.t} y2={H - M.b} stroke="#475569" strokeWidth={1} />
        {zone && <path d={zone} fill="#1d3a9e" opacity={0.45} />}
        <line x1={M.l} x2={W - M.r} y1={y(demandY)} y2={y(demandY)} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 2" />
        <path d={path("rest")} fill="none" stroke={REST_COLOR} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <path d={path("max")} fill="none" stroke={MAX_COLOR} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {read && <line x1={x(read.narrowing)} x2={x(read.narrowing)} y1={M.t} y2={H - M.b} stroke="#e2e8f0" strokeOpacity={0.35} strokeWidth={1} />}
        <line x1={x(cur.narrowing)} x2={x(cur.narrowing)} y1={M.t} y2={H - M.b} stroke="#f8fafc" strokeOpacity={0.75} strokeWidth={1} />
        <circle cx={x(cur.narrowing)} cy={y(cur.rest)} r={3.5} fill={REST_COLOR} stroke="#0f172a" strokeWidth={2} />
        <circle cx={x(cur.narrowing)} cy={y(cur.max)} r={3.5} fill={MAX_COLOR} stroke="#0f172a" strokeWidth={2} />
      </svg>
      <div className="text-[9px] leading-snug text-gray-400" aria-live="polite" title="In units of this heart's resting flow through a normal artery">
        At <b className="text-gray-100">{Math.round((read ?? cur).narrowing * 100)}%</b>: maximal{" "}
        <b className="text-gray-100">{(read ?? cur).max.toFixed(1)}×</b>, resting <b className="text-gray-100">{(read ?? cur).rest.toFixed(2)}×</b>
        <span className="text-gray-500">{read && onPick ? " · click to set" : ""}</span>
      </div>
      <figcaption className="text-[10px] leading-snug text-gray-200">
        Maximal flow starts to fall at about 50%: why ≥50% narrowing is the clinical (and dataset) threshold
      </figcaption>
    </figure>
  );
}

function LineKey({ color, dashed = false }: { color: string; dashed?: boolean }) {
  return (
    <svg width={12} height={6} className="shrink-0" aria-hidden>
      <line x1={0} x2={12} y1={3} y2={3} stroke={color} strokeWidth={2} strokeDasharray={dashed ? "3 2" : undefined} />
    </svg>
  );
}
