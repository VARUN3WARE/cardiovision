/**
 * Story mode's controls and words: the bar that takes the toolbar's place (step, previous / play-pause / next,
 * progress dots with the current step's progress, Exit), the caption that heads the notices column, and the
 * closing honesty card.
 */
import { useEffect, useRef, useState, type CSSProperties, type Ref } from "react";
import type { MetricsResponse } from "../api/client";
import { HONESTY, STORY_STEPS, honestyAucs, honestySpoken } from "./storySteps";

interface BarProps {
  step: number;
  /** The story plays (the user's choice; it may still be waiting for a heart to load). */
  playing: boolean;
  /** Playing, but waiting for a heart to be built: the clock has not started. */
  waiting: boolean;
  /** Restarts the progress line (a new run of a step). */
  run: number;
  /** The step was opened at its end (its progress line starts full). */
  atEnd: boolean;
  compact: boolean;
  /** The current step's progress line: the story clock scales it as the step plays. */
  progressRef: Ref<HTMLSpanElement>;
  onPlay: () => void;
  onPrev: () => void;
  onNext: () => void;
  onGoto: (step: number) => void;
  onExit: () => void;
}

const BTN = "min-w-[24px] rounded bg-slate-900/80 px-2 py-1 text-gray-200 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40";

/** Story controls, in the toolbar's place while the story runs (keyboard focus starts on play/pause). */
export function StoryBar({ step, playing, waiting, run, atEnd, compact, progressRef, onPlay, onPrev, onNext, onGoto, onExit }: BarProps) {
  const s = STORY_STEPS[step];
  const last = STORY_STEPS.length - 1;
  const play = useRef<HTMLButtonElement>(null);
  useEffect(() => play.current?.focus({ preventScroll: true }), []);
  const playLabel = !playing ? "Play" : waiting ? "Pause (waiting for the heart to load)" : "Pause";
  return (
    <div role="toolbar" aria-label="Story controls" className="flex flex-wrap items-center gap-1">
      {/* Narrow views: "Story 3/6 title" on one line, the controls on the next. */}
      {!compact && (
        <span className="flex items-center gap-1.5 rounded bg-fuchsia-800 px-2 py-1 font-semibold text-white">
          <span aria-hidden>▶</span> Story
        </span>
      )}
      <span className="flex min-w-0 max-w-full gap-1 rounded bg-slate-900/80 px-2 py-1 text-gray-100" aria-live="polite">
        <span className={`shrink-0 ${compact ? "font-semibold text-fuchsia-300" : "text-gray-400"}`}>
          {compact && "Story "}{step + 1}/{STORY_STEPS.length}
        </span>
        <span className="truncate">{s.title}</span>
      </span>
      <div className="flex items-center gap-1">
        <button onClick={onPrev} disabled={step === 0} aria-label="Previous step" title="Previous step (←)" className={BTN}>
          ‹
        </button>
        <button ref={play} onClick={onPlay} aria-label={playLabel} title={playLabel} className={`${BTN} w-8 text-center`}>
          <span className={waiting ? "animate-pulse" : undefined}>{playing ? "❚❚" : "▶"}</span>
        </button>
        <button onClick={onNext} disabled={step === last} aria-label="Next step" title="Next step (→)" className={BTN}>
          ›
        </button>
        <div role="group" aria-label="Story steps" className="flex items-center rounded bg-slate-900/80 px-1">
          {STORY_STEPS.map((x, i) => (
            <button
              key={x.id}
              onClick={() => onGoto(i)}
              aria-label={`Step ${i + 1}: ${x.title}`}
              aria-current={i === step ? "step" : undefined}
              title={`${i + 1}. ${x.title}`}
              className="group flex h-6 items-center px-[3px]"
            >
              <span
                className={`relative block h-2.5 overflow-hidden rounded-full transition-[width] ${
                  i === step ? "w-6 bg-slate-600" : i < step ? "w-2.5 bg-fuchsia-400" : "w-2.5 bg-slate-600 group-hover:bg-slate-400"}`}
              >
                {i === step && (
                  <span
                    key={run}
                    ref={progressRef}
                    aria-hidden
                    className="absolute inset-0 origin-left rounded-full bg-fuchsia-400"
                    style={{ transform: `scaleX(${atEnd ? 1 : 0})` }}
                  />
                )}
              </span>
            </button>
          ))}
        </div>
        <button
          onClick={onExit}
          aria-label="Exit story"
          title="Leave the story: back to your view (Esc)"
          className={`${BTN} whitespace-nowrap font-semibold`}
        >
          {compact ? "✕" : "Exit ✕"}
        </button>
      </div>
    </div>
  );
}

/**
 * The step's caption, first in the column of notices under the story bar. Narrow views show its first three lines
 * until it is tapped. Screen readers hear it once it has settled (the lab step's numbers change as it plays).
 */
export function StoryCaption({ text, compact }: { text: string; compact: boolean }) {
  const [open, setOpen] = useState(false);
  const [spoken, setSpoken] = useState("");
  useEffect(() => {
    const id = window.setTimeout(() => setSpoken(text), 1200);
    return () => window.clearTimeout(id);
  }, [text]);
  const box = "w-max max-w-full rounded border border-fuchsia-700/70 bg-slate-950/90 px-2.5 py-1 text-center leading-snug text-gray-100 shadow sm:max-w-[min(100%,460px)]";
  const live = <span role="status" aria-live="polite" className="sr-only">{spoken}</span>;
  if (!compact) {
    return (
      <div className={`${box} text-[11px]`}>
        <span aria-hidden>{text}</span>
        {live}
      </div>
    );
  }
  return (
    <button
      onClick={() => setOpen((o) => !o)}
      aria-expanded={open}
      aria-label={open ? "Show less of the caption" : "Show the whole caption"}
      title={open ? "Show less" : "Show the whole caption"}
      className={`${box} pointer-events-auto text-[10px] hover:border-fuchsia-500`}
    >
      <span aria-hidden className={open ? "block" : "line-clamp-3"}>{text}</span>
      {live}
    </button>
  );
}

/**
 * The closing card: what the 3D view, the models and their numbers are, and are not. Its buttons stay in view
 * (the text scrolls above them when space is short); screen readers hear the whole card once.
 */
export function StoryCard({ metrics, ended, onReplay, onExit, style }: {
  metrics: MetricsResponse | null;
  /** The story has played to its end (offer a replay). */
  ended: boolean;
  onReplay: () => void;
  onExit: () => void;
  style?: CSSProperties;
}) {
  const rows = honestyAucs(metrics);
  const [spoken, setSpoken] = useState("");
  useEffect(() => {
    const id = window.setTimeout(() => setSpoken(honestySpoken(metrics)), 600);
    return () => window.clearTimeout(id);
  }, [metrics]);
  return (
    <section
      aria-label="What this is, and is not"
      className="pointer-events-auto absolute left-1/2 z-20 flex w-[min(460px,calc(100%-1.5rem))] -translate-x-1/2 flex-col rounded border border-fuchsia-700/70 bg-slate-950 text-[11px] leading-snug text-gray-200 shadow-xl"
      style={style}
    >
      <div className="min-h-0 overflow-y-auto px-3 pt-2.5">
        <h3 className="text-sm font-semibold text-white">What this is, and is not</h3>
        <p className="mb-1 font-semibold text-amber-200">{HONESTY.use}</p>
        <ul className="list-disc space-y-0.5 pl-4">
          <li>{HONESTY.anatomy}</li>
          <li>{HONESTY.cohort}</li>
          <li>
            How well each model separates patients with and without the narrowing, in cross-validation (ROC AUC, with its
            bootstrap 95% CI):
            <table className="mt-0.5 w-full max-w-[300px] text-left">
              <tbody>
                {rows.map((r) => (
                  <tr key={r.target}>
                    <th scope="row" className="py-px pr-2 font-semibold text-gray-100">{r.label}</th>
                    <td className="py-px pr-2 tabular-nums text-white">{r.auc === null ? "see Model verification" : r.auc.toFixed(2)}</td>
                    <td className="py-px tabular-nums text-gray-400">{r.ci[0].toFixed(2)}–{r.ci[1].toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </li>
          <li>{HONESTY.caveats}</li>
        </ul>
      </div>
      <div className="flex shrink-0 justify-end gap-1.5 px-3 pb-2.5 pt-2 text-xs">
        {ended && (
          <button onClick={onReplay} className="rounded bg-slate-800 px-2 py-1 text-gray-200 hover:bg-slate-700">
            Replay
          </button>
        )}
        <button onClick={onExit} className="rounded bg-fuchsia-700 px-2 py-1 font-semibold text-white hover:bg-fuchsia-600">
          Exit story
        </button>
      </div>
      <span role="status" aria-live="polite" className="sr-only">{spoken}</span>
    </section>
  );
}
