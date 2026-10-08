import { BAND_COLOR, vesselBand } from "../risk";
import type { ArteryProbs } from "./ArteryTube";
import { SEGMENT_VESSEL, VESSEL, VESSELS, type Vessel } from "../anatomy/registry";

/**
 * AHA 17-segment polar map ("bullseye"), shaded by the chance of narrowing in the artery that usually
 * supplies each segment. Standard assignment (Cerqueira et al., Circulation 2002;105:539-42), right
 * dominance assumed. It is a map of supply, not of perfusion or wall motion: one flat colour per artery.
 */

const SEGMENT_NAME: Record<number, string> = {
  1: "basal anterior", 2: "basal anteroseptal", 3: "basal inferoseptal", 4: "basal inferior",
  5: "basal inferolateral", 6: "basal anterolateral", 7: "mid anterior", 8: "mid anteroseptal",
  9: "mid inferoseptal", 10: "mid inferior", 11: "mid inferolateral", 12: "mid anterolateral",
  13: "apical anterior", 14: "apical septal", 15: "apical inferior", 16: "apical lateral", 17: "apex",
};

/** Colour of one segment: its supplying artery's band colour (pure function of probabilities and bands). */
export function segmentColor(segment: number, probs: ArteryProbs | null): string {
  if (!probs) return "#475569";
  const band = vesselBand(probs[SEGMENT_VESSEL[segment]]);
  return band === "Intermediate" ? "url(#bullseye-undecided)" : BAND_COLOR[band];
}

// Rings (outer radius fractions) and segment layout: anterior at the top, septum on the left.
const R = { apex: 0.2, apical: 0.45, mid: 0.7, basal: 0.95 };

interface Sector {
  seg: number;
  r0: number;
  r1: number;
  a0: number; // degrees, counter-clockwise from 3 o'clock
  a1: number;
}

function sectors(): Sector[] {
  const out: Sector[] = [];
  // Six 60° segments for the basal and mid rings, centred on anterior (90°) and going counter-clockwise.
  [[1, R.mid, R.basal], [7, R.apical, R.mid]].forEach(([first, r0, r1]) => {
    for (let k = 0; k < 6; k++) out.push({ seg: first + k, r0, r1, a0: 60 + 60 * k, a1: 120 + 60 * k });
  });
  for (let k = 0; k < 4; k++) out.push({ seg: 13 + k, r0: R.apex, r1: R.apical, a0: 45 + 90 * k, a1: 135 + 90 * k });
  return out;
}

function sectorPath({ r0, r1, a0, a1 }: Sector, size: number) {
  const c = size / 2;
  const pt = (r: number, a: number) => {
    const rad = (a * Math.PI) / 180;
    return `${(c + r * c * Math.cos(rad)).toFixed(2)} ${(c - r * c * Math.sin(rad)).toFixed(2)}`;
  };
  return `M ${pt(r1, a0)} A ${r1 * c} ${r1 * c} 0 0 0 ${pt(r1, a1)} L ${pt(r0, a1)} A ${r0 * c} ${r0 * c} 0 0 1 ${pt(r0, a0)} Z`;
}

function labelPos({ r0, r1, a0, a1 }: Sector, size: number) {
  const c = size / 2;
  const r = ((r0 + r1) / 2) * c;
  const a = (((a0 + a1) / 2) * Math.PI) / 180;
  return { x: c + r * Math.cos(a), y: c - r * Math.sin(a) };
}

interface Props {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  onHover: (v: Vessel | null) => void;
  onSelect: (v: Vessel | null) => void;
}

export default function Bullseye({ probs, selected, onHover, onSelect }: Props) {
  const size = 150;
  const all = sectors();
  const opacity = (seg: number) => (selected && SEGMENT_VESSEL[seg] !== selected ? 0.35 : 1);
  return (
    <figure className="flex items-center gap-3">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label="17-segment map of the left ventricle shaded by each supplying artery's chance of narrowing"
        className="shrink-0"
        onMouseLeave={() => onHover(null)}
      >
        <defs>
          <pattern id="bullseye-undecided" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill={BAND_COLOR.Intermediate} />
            <rect width="2.5" height="6" fill="#7a4f0a" />
          </pattern>
        </defs>
        {all.map((s) => {
          const v = SEGMENT_VESSEL[s.seg];
          const p = labelPos(s, size);
          return (
            <g
              key={s.seg}
              className="cursor-pointer"
              onMouseEnter={() => onHover(v)}
              onClick={() => onSelect(selected === v ? null : v)}
            >
              <title>{`Segment ${s.seg}, ${SEGMENT_NAME[s.seg]}: usually supplied by the ${VESSEL[v].label}`}</title>
              <path d={sectorPath(s, size)} fill={segmentColor(s.seg, probs)} opacity={opacity(s.seg)} stroke="#0f172a" strokeWidth={1.2} />
              <text x={p.x} y={p.y} textAnchor="middle" dominantBaseline="central" fontSize={9} fill="#0f172a" fontWeight={600} pointerEvents="none">
                {s.seg}
              </text>
            </g>
          );
        })}
        <g className="cursor-pointer" onMouseEnter={() => onHover(SEGMENT_VESSEL[17])} onClick={() => onSelect(selected === SEGMENT_VESSEL[17] ? null : SEGMENT_VESSEL[17])}>
          <title>{`Segment 17, apex: usually supplied by the ${VESSEL[SEGMENT_VESSEL[17]].label}`}</title>
          <circle cx={size / 2} cy={size / 2} r={R.apex * size / 2} fill={segmentColor(17, probs)} opacity={opacity(17)} stroke="#0f172a" strokeWidth={1.2} />
          <text x={size / 2} y={size / 2} textAnchor="middle" dominantBaseline="central" fontSize={9} fill="#0f172a" fontWeight={600} pointerEvents="none">17</text>
        </g>
        {/* Orientation */}
        <text x={size / 2} y={6} textAnchor="middle" fontSize={7} fill="#94a3b8">anterior</text>
        <text x={size / 2} y={size - 1} textAnchor="middle" fontSize={7} fill="#94a3b8">inferior</text>
      </svg>
      <figcaption className="text-[11px] leading-snug text-gray-400">
        <div className="mb-1 text-gray-300">Muscle each artery usually supplies</div>
        Standard AHA 17-segment assignment ({VESSELS.map((v) => `${v.label} ${v.aha.join(", ")}`).join("; ")}; right
        dominance assumed), shaded by that artery&apos;s chance of narrowing. Septum on the left. Not a perfusion
        or wall-motion map. Hover to find the artery on the heart; click to inspect it.
      </figcaption>
    </figure>
  );
}
