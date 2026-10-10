import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ExplainResponse } from "../api/client";
import { formatPct } from "../risk";
import { VESSELS } from "../anatomy/registry";
import { FEATURE_LABELS } from "../featureLabels";
import ErrorBoundary from "../components/ErrorBoundary";
import CohortGalaxy, { type GalaxyHover } from "./CohortGalaxy";
import { CAD_COLOR, NO_CAD_COLOR } from "./palette";
import {
  buildSpace, closeLimit, cohortRates, comparedInputs, findSimilar, inputsFromExplanation, placeOnMap, summarise,
  type CohortData, type CohortPatient, type CohortSummary, type Compared, type RecordMatch, type Similar, type Space,
} from "./knn";

let loading: Promise<Space> | null = null;

/**
 * The cohort file (~80 KB) is fetched on first use, outside the main bundle: as an asset rather than
 * import()ed, because the browser caches a failed dynamic import for good and a retry would never refetch.
 */
function fetchCohort(): Promise<CohortData> {
  return fetch(new URL("./cohort.json", import.meta.url), { signal: AbortSignal.timeout(20000) }).then((res) => {
    if (!res.ok) throw new Error(`cohort.json: HTTP ${res.status}`);
    return res.json() as Promise<CohortData>;
  });
}

/** One quiet second attempt (a network blip or a dev-server reload); after that the card offers Retry. */
function loadSpace(): Promise<Space> {
  loading ??= fetchCohort()
    .catch(() => new Promise<void>((resolve) => setTimeout(resolve, 1500)).then(fetchCohort))
    .then(buildSpace);
  return loading;
}

/** Loads the cohort once there is a result. After a failure, Retry (or the next result) tries again. */
function useSpace(trigger: unknown): { space: Space | null; failed: boolean; retry: () => void } {
  const [state, setState] = useState<{ space: Space | null; failed: boolean }>({ space: null, failed: false });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!trigger || state.space) return;
    let alive = true;
    setState((s) => (s.failed ? { space: null, failed: false } : s)); // "Loading…" while trying again
    loadSpace().then(
      (space) => alive && setState({ space, failed: false }),
      () => {
        loading = null;
        if (alive) setState({ space: null, failed: true });
      },
    );
    return () => {
      alive = false;
    };
  }, [trigger, state.space, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}

interface Props {
  open: boolean;
  onToggle: () => void;
  /** The recorded patient's estimate (never a what-if scenario). */
  result: ExplainResponse | null;
  /** A what-if scenario is active elsewhere: neighbours stay those of the recorded values. */
  scenario: boolean;
  /** Deep link (?cohort=1): scroll the card into view once it has content. */
  scrollOnOpen?: boolean;
}

interface View {
  similar: Similar;
  /** The inputs compared: the entered ones (those the API filled in are left out). */
  compared: Compared;
  here: [number, number, number];
  summary: CohortSummary;
  /** The 10th most similar is about as near as for most of the dataset's own patients. */
  close: boolean;
}

/** "Similar patients": the 10 most similar real patients, their angiograms and held-out estimates. */
export default function CohortPanel({ open, onToggle, result, scenario, scrollOnOpen = false }: Props) {
  const { space, failed, retry } = useSpace(result);
  const view = useMemo<View | null>(() => {
    if (!space || !result) return null;
    const x = inputsFromExplanation(result.explanations?.cad, space.data.columns);
    if (!x) return null;
    const compared = comparedInputs(space, result.imputed);
    const similar = findSimilar(space, x, space.data.k, compared.use);
    const kth = similar.neighbours[similar.neighbours.length - 1]?.distance ?? Infinity;
    return {
      similar,
      compared,
      here: placeOnMap(space, similar),
      summary: summarise(space, similar.neighbours),
      close: kth <= closeLimit(space, compared.use),
    };
  }, [space, result]);

  const cardRef = useRef<HTMLDivElement>(null);
  const scrolled = useRef(false);
  useEffect(() => {
    if (!scrollOnOpen || scrolled.current || !open || !view) return;
    const scroll = () => cardRef.current && scrollToTop(cardRef.current);
    const raf = requestAnimationFrame(() => {
      scrolled.current = true;
      scroll();
    });
    const late = setTimeout(scroll, 800); // again once the cards above have settled
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(late);
    };
  }, [scrollOnOpen, open, view]);

  const headline = !view ? "" : view.close ? `${view.summary.counts.cad} of ${view.summary.k} had CAD` : "no close matches";

  return (
    <div ref={cardRef} className="rounded bg-slate-800 p-4">
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-center justify-between gap-2 text-left">
        <h2 className="text-sm font-semibold text-gray-200">
          Similar patients {headline && <span className="font-normal text-gray-400">· {headline}</span>}
        </h2>
        <span className="text-xs text-gray-400" aria-hidden>{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="mt-3 space-y-3">
          <div role="note" className="rounded border border-sky-800 bg-sky-950/40 px-2 py-1.5 text-[11px] text-sky-200">
            <b>Similar in recorded findings (weighted by what the CAD model uses), not identical patients;</b> outcomes
            from this single referral cohort.
          </div>
          {!result ? (
            <div className="text-xs text-gray-400">Run an estimate first.</div>
          ) : failed ? (
            <div className="flex flex-wrap items-center gap-2 text-xs text-red-300">
              Could not load the cohort data.
              <button onClick={retry} className="rounded bg-slate-700 px-2 py-0.5 font-semibold text-gray-100 hover:bg-slate-600">
                Retry
              </button>
            </div>
          ) : !space ? (
            <div className="text-xs text-gray-400">Loading the 303 patients…</div>
          ) : !view ? (
            <div className="text-xs text-gray-400">Similar patients are unavailable for this result.</div>
          ) : (
            <CohortBody space={space} view={view} scenario={scenario} />
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Bring the card to the top of the results column, or of the page when the layout is stacked
 * (scrollIntoView would also scroll the page and push the header and disclaimer out of view).
 */
function scrollToTop(el: HTMLElement) {
  for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
    const o = getComputedStyle(p).overflowY;
    if ((o === "auto" || o === "scroll") && p.scrollHeight > p.clientHeight + 1) {
      p.scrollTo({ top: p.scrollTop + el.getBoundingClientRect().top - p.getBoundingClientRect().top });
      return;
    }
  }
  window.scrollTo({ top: window.scrollY + el.getBoundingClientRect().top - 8 });
}

function CohortBody({ space, view, scenario }: { space: Space; view: View; scenario: boolean }) {
  const { data } = space;
  const { similar, compared, here, summary, close } = view;
  const cad = data.targets.indexOf("cad");
  const rates = useMemo(() => cohortRates(data), [data]);
  const neighbourIds = useMemo(() => similar.neighbours.map((n) => n.index), [similar]);

  const [hover, setHover] = useState<GalaxyHover | null>(null);
  const [rowHover, setRowHover] = useState<number | null>(null);
  const [resetKey, setResetKey] = useState(0);
  // Skip pointer moves that change neither the target nor the tooltip position (less re-rendering).
  const onHover = useCallback((h: GalaxyHover | null) => {
    setHover((prev) => (prev && h && prev.id === h.id && Math.abs(prev.x - h.x) < 2 && Math.abs(prev.y - h.y) < 2 ? prev : h));
  }, []);
  const highlight = rowHover ?? (hover && hover.id >= 0 ? hover.id : null);
  const mapRef = useRef<HTMLDivElement>(null);
  const keep = Math.round(data.embedding.neighbour_retention * data.k);
  const method = data.embedding.method.startsWith("t-SNE") ? "t-SNE" : "PCA";
  const anyArtery = VESSELS.some((v) => summary.counts[v.id] > 0);
  const same = compared.skipped ? "same values for everything entered" : "same recorded values";

  return (
    <>
      {scenario && <div className="text-[11px] text-sky-300">Shown for the recorded values, not the what-if scenario.</div>}
      <div>
        {close ? (
          <p className="text-sm text-gray-200">{summaryText(summary, true)}</p>
        ) : (
          <>
            <p role="note" className="rounded border border-amber-700/70 bg-amber-950/40 px-2 py-1.5 text-[11px] text-amber-200">
              <b>No close matches among the {data.n}:</b> these are the {summary.k} nearest, but less alike than the{" "}
              {summary.k} nearest of {pct(data.closeness.quantile)} of the dataset&apos;s own patients, so their outcomes say
              little about this patient.
            </p>
            <p className="mt-1.5 text-xs text-gray-400">{summaryText(summary, false)}</p>
          </>
        )}
        {anyArtery && (
          <p className="mt-0.5 text-[11px] text-gray-400">
            Similarity reflects CAD overall, so these counts say little about which artery.
          </p>
        )}
        <p className="mt-0.5 text-[11px] text-gray-500">
          All {data.n} patients: CAD {pct(rates.cad)}; {VESSELS.map((v) => `${v.label} ${pct(rates[v.id])}`).join(", ")}.
        </p>
        {similar.self !== null && (
          <p className="mt-1 text-[11px] text-gray-400">
            This is dataset patient <b className="text-gray-200">#{similar.self}</b> ({same}): it is left out of
            the {summary.k} and glows at its own place on the map.
          </p>
        )}
        {similar.edited && (
          <p className="mt-1 text-[11px] text-gray-400">
            This is dataset patient <b className="text-gray-200">#{similar.edited.index}</b> with{" "}
            {changedText(similar.edited)}: its record is left out of the {summary.k}, so its own outcome is not counted.
          </p>
        )}
        {compared.skipped > 0 && (
          <p className="mt-1 text-[11px] text-gray-500">
            Compared only on what was entered ({compared.n} of the model&apos;s {compared.n + compared.skipped} inputs;
            the {compared.skipped} not entered are left out). For its estimate the model fills those in: unrecorded
            findings count as absent, missing measurements take the cohort median.
          </p>
        )}
      </div>

      <div>
        <div
          ref={mapRef}
          role="img"
          aria-label={`3D map of the ${data.n} dataset patients coloured by angiography result; this patient among its ${summary.k} most similar. The list below has the same information.`}
          className="relative h-60 overflow-hidden rounded border border-slate-700 bg-slate-900"
        >
          <ErrorBoundary fallback={<div className="flex h-full items-center justify-center p-4 text-center text-xs text-gray-500">3D map unavailable on this device; the list below has the same patients.</div>}>
            <CohortGalaxy
              patients={data.patients}
              cadIndex={cad}
              neighbours={neighbourIds}
              self={similar.self}
              here={here}
              highlight={highlight}
              onHover={onHover}
              resetKey={resetKey}
            />
          </ErrorBoundary>
          {hover && (
            <Tooltip
              hover={hover}
              box={mapRef.current}
              space={space}
              self={similar.self}
              same={same}
              edited={similar.edited}
              rank={neighbourIds.indexOf(hover.id)}
            />
          )}
          <button
            onClick={() => setResetKey((k) => k + 1)}
            className="absolute right-1.5 top-1.5 rounded bg-slate-800/90 px-1.5 py-0.5 text-[10px] text-gray-300 hover:bg-slate-700"
          >
            Reset view
          </button>
          <span className="pointer-events-none absolute bottom-1 left-2 text-[10px] text-gray-500">Drag to rotate</span>
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-gray-400">
          <span className="flex items-center gap-1"><Swatch color={CAD_COLOR} /> CAD on angiography</span>
          <span className="flex items-center gap-1"><Swatch color={NO_CAD_COLOR} /> No CAD on angiography</span>
          <span className="flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full bg-white shadow-[0_0_6px_2px_rgba(255,255,255,0.55)]" /> This patient</span>
          <span>Ringed and linked: the {summary.k} most similar</span>
        </div>
        <p className="mt-1 text-[10px] leading-snug text-gray-500">
          Map of all {data.n} patients placed by similarity ({method}). It keeps about {keep} of each patient&apos;s {data.k}{" "}
          most similar nearby; the list uses the exact similarity. {groupsText(data)}
        </p>
      </div>

      <NeighbourTable space={space} summary={summary} neighbours={neighbourIds} highlight={highlight} onRowHover={setRowHover} />
    </>
  );
}

function NeighbourTable({ space, summary, neighbours, highlight, onRowHover }: {
  space: Space; summary: CohortSummary; neighbours: number[]; highlight: number | null; onRowHover: (i: number | null) => void;
}) {
  const { data } = space;
  const cols = [{ id: "cad", label: "CAD" }, ...VESSELS.map((v) => ({ id: v.id as string, label: v.label }))];
  const at = cols.map((c) => data.targets.indexOf(c.id));
  const odd = new Set(summary.inconsistent);
  return (
    <div>
      <table className="w-full text-[11px] tabular-nums">
        <caption className="mb-1 text-left text-[11px] text-gray-400">
          The {summary.k} most similar, closest first: held-out estimate and angiography
        </caption>
        <thead>
          <tr className="text-[10px] text-gray-500">
            <th scope="col" className="pl-1 text-left font-normal">Patient</th>
            {cols.map((c) => <th key={c.id} scope="col" className="font-normal">{c.label}</th>)}
          </tr>
          <tr className="text-[9px] text-gray-500">
            <td className="pb-0.5 pl-1">age, sex</td>
            <td colSpan={cols.length} className="pb-0.5 text-center">each: angiography · held-out estimate</td>
          </tr>
        </thead>
        <tbody>
          {neighbours.map((i) => {
            const p = data.patients[i];
            return (
              <tr
                key={i}
                tabIndex={0}
                onMouseEnter={() => onRowHover(i)}
                onMouseLeave={() => onRowHover(null)}
                onFocus={() => onRowHover(i)}
                onBlur={() => onRowHover(null)}
                className={`outline-none focus-visible:ring-1 focus-visible:ring-sky-400 ${highlight === i ? "bg-slate-700" : ""}`}
              >
                <th scope="row" className="py-0.5 pl-1 text-left font-normal text-gray-300">
                  #{i}
                  {odd.has(i) && <sup className="text-amber-300">*</sup>}{" "}
                  <span className="ml-1 text-gray-500">{p.age} {p.sex}</span>
                </th>
                {at.map((j, c) => (
                  <td key={cols[c].id} className="py-0.5 text-center">
                    <span className="inline-flex items-center gap-1">
                      <Outcome positive={p.angio[j] === 1} />
                      <span className="text-gray-300">{formatPct(p.p[j])}</span>
                    </span>
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
        <tfoot className="text-[10px] text-gray-400">
          <tr className="border-t border-slate-700">
            <th scope="row" className="pl-1 pt-1 text-left font-normal">≥50% on angiography</th>
            {cols.map((c) => <td key={c.id} className="pt-1 text-center">{summary.counts[c.id]} of {summary.k}</td>)}
          </tr>
          <tr>
            <th scope="row" className="pl-1 text-left font-normal">Mean held-out estimate</th>
            {cols.map((c) => <td key={c.id} className="text-center">{formatPct(summary.meanHeldOut[c.id])}</td>)}
          </tr>
        </tfoot>
      </table>
      <p className="mt-1.5 text-[10px] leading-snug text-gray-500">
        <span className="inline-flex items-center gap-1 align-middle"><Outcome positive /></span> ≥50% narrowing on angiography
        (CAD column: CAD) · <span className="inline-flex items-center gap-1 align-middle"><Outcome positive={false} /></span> below
        50% · % = held-out estimate from models that never saw that patient.
      </p>
      {summary.inconsistent.map((i) => (
        <p key={i} className="mt-1 text-[10px] leading-snug text-gray-500">
          <span className="text-amber-300">*</span> #{i} is labelled no CAD overall but has ≥50% narrowing of the{" "}
          {vesselsOf(space, data.patients[i]).join(", ")} in the dataset (a label inconsistency in the source data).
        </p>
      ))}
    </div>
  );
}

function Tooltip({ hover, box, space, self, same, edited, rank }: {
  hover: GalaxyHover; box: HTMLDivElement | null; space: Space; self: number | null; same: string; edited: RecordMatch | null;
  rank: number;
}) {
  const w = box?.clientWidth ?? 340;
  const h = box?.clientHeight ?? 240;
  const width = 176;
  const style: React.CSSProperties = {
    width,
    left: hover.x + 14 + width < w ? hover.x + 14 : Math.max(4, hover.x - 14 - width),
    top: hover.y + 74 < h ? hover.y + 12 : Math.max(4, hover.y - 70),
  };
  const { data } = space;
  const cad = data.targets.indexOf("cad");
  let body: React.ReactNode;
  if (hover.id < 0) {
    body = (
      <>
        <div className="font-semibold text-gray-100">This patient</div>
        <div className="text-gray-400">
          {self !== null ? `Dataset patient #${self} (${same})` : `Placed among its ${data.k} most similar`}
        </div>
      </>
    );
  } else {
    const p = data.patients[hover.id];
    const vessels = vesselsOf(space, p);
    const hadCad = p.angio[cad] === 1;
    const own = edited?.index === hover.id;
    body = (
      <>
        <div>
          <b className="text-gray-100">#{p.i}</b> <span className="text-gray-400">· {p.age} {p.sex}{rank >= 0 ? ` · similar no. ${rank + 1}` : ""}</span>
        </div>
        {own && <div className="text-gray-300">This patient&apos;s record before the changes (not counted)</div>}
        <div className="flex items-center gap-1.5 text-gray-200">
          <span className="h-0.5 w-3 shrink-0" style={{ background: hadCad ? CAD_COLOR : NO_CAD_COLOR }} />
          {hadCad ? `CAD; ≥50% in ${vessels.join(", ") || "no named artery"}` : vessels.length ? `No CAD label; ≥50% in ${vessels.join(", ")}` : "No ≥50% narrowing"}
        </div>
        <div className="text-gray-400"><b className="text-gray-200">{formatPct(p.p[cad])}</b> held-out CAD estimate</div>
      </>
    );
  }
  return (
    <div role="tooltip" className="pointer-events-none absolute z-10 rounded border border-slate-600 bg-slate-950/95 px-2 py-1.5 text-[11px] leading-snug shadow-lg" style={style}>
      {body}
    </div>
  );
}

/** "8 of 10 similar patients had CAD; LAD ≥50% narrowing in 6, ..." (close) or "Of these 10, ..." (not close). */
function summaryText(s: CohortSummary, close: boolean): string {
  const vessels = VESSELS.map((v) => ({ label: v.label, n: s.counts[v.id] ?? 0 }));
  if (s.counts.cad === 0 && vessels.every((v) => v.n === 0)) {
    return close
      ? `None of the ${s.k} similar patients had CAD or ≥50% narrowing of any artery.`
      : `None of these ${s.k} had CAD or ≥50% narrowing of any artery.`;
  }
  const [first, ...rest] = vessels;
  const odd = s.inconsistent.length;
  return `${close ? `${s.counts.cad} of ${s.k} similar patients had CAD` : `Of these ${s.k}, ${s.counts.cad} had CAD`}; `
    + `${first.label} ≥50% narrowing in ${first.n}, ${rest.map((v) => `${v.label} in ${v.n}`).join(", ")}`
    + (odd ? ` (counts include ${odd} patient${odd === 1 ? "" : "s"} labelled no CAD*)` : "") + ".";
}

/** "2 inputs changed (Age, Systolic blood pressure)". */
function changedText(m: RecordMatch): string {
  const n = m.changed.length;
  return `${n} input${n === 1 ? "" : "s"} changed (${m.changed.map((c) => FEATURE_LABELS[c]?.label ?? c).join(", ")})`;
}

/** Caption for the map's two groups, when one input splits it (written by make_cohort.py only then). */
function groupsText(data: CohortData): string {
  const g = data.groups;
  if (!g || g.levels.length !== 2) return "";
  const yes = g.levels.find((l) => l.value === 1);
  const no = g.levels.find((l) => l.value === 0);
  if (!yes || !no) return "";
  const name = g.column === "Typical Chest Pain" ? "typical angina" : (FEATURE_LABELS[g.column]?.label ?? g.column).toLowerCase();
  return `The two groups are patients with ${name} (${yes.n}; ${pct(yes.cad)} had CAD) and without (${no.n}; ${pct(no.cad)}), `
    + "the input the CAD model weighs most.";
}

function vesselsOf(space: Space, p: CohortPatient): string[] {
  return VESSELS.filter((v) => p.angio[space.data.targets.indexOf(v.id)] === 1).map((v) => v.label);
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

function Swatch({ color }: { color: string }) {
  return <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />;
}

/** Filled: >=50% narrowing (or CAD); ring: below 50%. Shape as well as colour. */
function Outcome({ positive }: { positive: boolean }) {
  return (
    <>
      <span
        aria-hidden
        className="inline-block h-2 w-2 shrink-0 rounded-full"
        style={positive ? { background: CAD_COLOR } : { boxShadow: `inset 0 0 0 1.5px ${NO_CAD_COLOR}` }}
      />
      <span className="sr-only">{positive ? "≥50% narrowing;" : "below 50%;"}</span>
    </>
  );
}
