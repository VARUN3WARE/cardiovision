import type { ExplainResponse } from "../api/client";
import { formatPct } from "../risk";
import { scenarioDeltas, type ScenarioStatus } from "./scenario";

interface Props {
  status: Exclude<ScenarioStatus, "off">;
  /** What the scenario on screen changes, in words (before its first reply: the one being calculated). */
  summary: string[];
  error: string | null;
  /** The patient's result and the scenario's (for the screen-reader summary once a reply has settled). */
  base: ExplainResponse | null;
  shown: ExplainResponse | null;
  onReset: () => void;
  onRetry: () => void;
}

/**
 * One line above the 3D view while a what-if scenario is on. App renders it in the 3D section's flow (not over the
 * scene), so it never covers the toolbar, inspector, captions or ECG; long summaries are truncated (full text on
 * hover).
 */
export default function ScenarioStrip({ status, summary, error, base, shown, onReset, onRetry }: Props) {
  const failed = status === "error";
  const changes = summary.join(", ") || "what-if inputs";
  const full = failed
    ? `What-if failed: ${error ?? ""} The 3D view shows the recorded patient.`
    : `What-if: ${changes}. Model sensitivity, not treatment effect.${status === "calculating" ? " Calculating…" : ""}`;
  const announce = failed ? full
    : status === "ready" && base && shown
      ? `What-if estimates: ${scenarioDeltas(base, shown).map((d) => `${d.label} ${formatPct(d.from)} to ${formatPct(d.to)}`).join(", ")}.`
      : "";
  const busy = status === "calculating" || status === "updating";
  return (
    <div
      role="group"
      aria-label="What-if scenario"
      title={full}
      className="flex h-7 shrink-0 items-center gap-2 rounded-t border-b border-sky-700 bg-sky-950 px-2 text-[11px] text-sky-100"
    >
      <span aria-hidden className={`h-2 w-2 shrink-0 rounded-full ${failed ? "bg-red-400" : "bg-sky-400"} ${busy ? "animate-pulse" : ""}`} />
      <span className="min-w-0 flex-1 truncate">
        {failed ? (
          <><b className="text-red-200">What-if failed:</b> {error}</>
        ) : (
          <><b>What-if:</b> {changes}<span className="text-sky-300"> · model sensitivity, not treatment effect</span></>
        )}
      </span>
      {status === "calculating" && <span className="shrink-0 text-sky-300">Calculating…</span>}
      {failed && (
        <button onClick={onRetry} className="shrink-0 rounded bg-red-700 px-1.5 py-0.5 font-semibold text-white hover:bg-red-600">
          Retry
        </button>
      )}
      <button
        onClick={onReset}
        title="Reset the what-if: back to the patient's recorded values"
        className="shrink-0 rounded bg-sky-600 px-1.5 py-0.5 font-semibold text-white hover:bg-sky-500"
      >
        Reset<span className="hidden max-lg:inline xl:inline"> to patient</span>
      </button>
      <span className="sr-only" role="status">{announce}</span>
    </div>
  );
}
