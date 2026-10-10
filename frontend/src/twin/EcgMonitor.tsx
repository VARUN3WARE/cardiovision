import { useEffect, useRef } from "react";
import {
  RING_SIZE, TICK, ecgFindings, ecgSample, efCategory, heartClock, readTick,
  type BeatTiming, type Lead, type TwinParams,
} from "./twin";

const LEADS: Lead[] = ["II", "V1"];
const SWEEP_SECONDS = 4; // width of the trace window
const GAP_PX = 14; // erase bar ahead of the sweep, as on a bedside monitor

interface Props {
  params: TwinParams;
  timing: BeatTiming;
}

/**
 * Bedside-style monitor: an illustrative two-lead ECG synchronised with the 3D heartbeat, plus the
 * recorded values that drive the twin. Reads beat time from the shared heart clock (500 Hz ticks).
 */
export default function EcgMonitor({ params, timing }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const live = useRef({ params, timing });
  live.current = { params, timing };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let width = 0;
    let height = 0;
    let mins: Float32Array[] = [];
    let maxs: Float32Array[] = [];
    let cursor = 0;
    let acc = { n: 0, lo: [Infinity, Infinity], hi: [-Infinity, -Infinity] };
    let read = heartClock.count;
    let raf = 0;

    const resize = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      width = Math.max(50, Math.floor(canvas.clientWidth));
      height = Math.max(40, Math.floor(canvas.clientHeight));
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      mins = LEADS.map(() => new Float32Array(width).fill(NaN));
      maxs = LEADS.map(() => new Float32Array(width).fill(NaN));
      cursor = 0;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const { params: p, timing: tm } = live.current;
      const ticksPerPixel = SWEEP_SECONDS / width / TICK;
      if (heartClock.count - read > RING_SIZE) read = heartClock.count - RING_SIZE; // after a long pause
      for (; read < heartClock.count; read++) {
        const t = readTick(read);
        LEADS.forEach((lead, i) => {
          const v = ecgSample(t, tm, p, lead);
          acc.lo[i] = Math.min(acc.lo[i], v);
          acc.hi[i] = Math.max(acc.hi[i], v);
        });
        if (++acc.n >= ticksPerPixel) {
          LEADS.forEach((_, i) => {
            mins[i][cursor] = acc.lo[i];
            maxs[i][cursor] = acc.hi[i];
          });
          cursor = (cursor + 1) % width;
          acc = { n: 0, lo: [Infinity, Infinity], hi: [-Infinity, -Infinity] };
        }
      }
      draw();
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      const rowH = height / LEADS.length;
      const pxPerMv = rowH / 2.8;
      // Faint grid: 0.2 s major divisions, 0.5 mV.
      ctx.strokeStyle = "rgba(52, 211, 153, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let s = 0; s <= SWEEP_SECONDS; s += 0.2) {
        const x = Math.round((s / SWEEP_SECONDS) * width) + 0.5;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y <= height; y += pxPerMv * 0.5) {
        ctx.moveTo(0, Math.round(y) + 0.5);
        ctx.lineTo(width, Math.round(y) + 0.5);
      }
      ctx.stroke();

      LEADS.forEach((lead, i) => {
        const base = rowH * i + rowH * 0.56;
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, rowH * i, width, rowH);
        ctx.clip();
        ctx.strokeStyle = "#3ee69b";
        ctx.lineWidth = 1.6;
        ctx.lineJoin = "round";
        ctx.beginPath();
        let pen = false;
        for (let x = 0; x < width; x++) {
          const ahead = (x - cursor + width) % width;
          if (ahead < GAP_PX || Number.isNaN(mins[i][x])) {
            pen = false;
            continue;
          }
          const y0 = base - maxs[i][x] * pxPerMv;
          const y1 = base - mins[i][x] * pxPerMv;
          if (!pen) ctx.moveTo(x, (y0 + y1) / 2);
          ctx.lineTo(x, y0);
          ctx.lineTo(x, y1);
          pen = true;
        }
        ctx.stroke();
        ctx.restore();
        ctx.fillStyle = "rgba(167, 243, 208, 0.7)";
        ctx.font = "600 10px system-ui, sans-serif";
        ctx.fillText(lead, 6, rowH * i + 12);
      });
    };

    frame();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  const findings = ecgFindings(params);
  // While the ischaemia lab drives the twin, rate and pressure are the lab's exercise values, not recorded ones
  // (the EF stays the patient's, so its "not entered" note stays).
  const lab = params.lab !== undefined;
  const median = (k: keyof Omit<TwinParams, "notRecorded" | "lab">) => params.notRecorded.has(k) && !(lab && (k === "hr" || k === "bp"));
  return (
    <div className="flex h-full gap-3 border-t border-slate-700 bg-[#05090f] px-3 py-1.5 text-[10px] text-gray-400">
      <div className="relative min-w-0 flex-1">
        <canvas ref={canvasRef} className="h-full w-full" aria-label="Illustrative ECG, leads II and V1" />
        <div
          className="absolute bottom-0 right-1 max-w-full truncate text-[9px] text-gray-500"
          title={lab ? "Illustrative ECG: recorded findings plus the lab's sandbox (marked lab), not the patient's tracing" : "Illustrative ECG built from the recorded findings, not the patient's tracing"}
        >
          {lab
            ? "Illustrative ECG: recorded findings plus the lab's sandbox (marked lab), not the patient's tracing"
            : "Illustrative ECG built from the recorded findings, not the patient's tracing"}
        </div>
      </div>
      <div className="w-44 shrink-0 space-y-0.5 overflow-hidden leading-tight">
        <div className="text-[11px] font-semibold text-emerald-300">
          ♥ {Math.round(params.hr)} bpm{median("hr") && <NotEntered />}
          {lab && <LabChip title="Heart rate set by the ischaemia lab's exercise slider" />}
        </div>
        <div>
          <span className="text-gray-200">EF {Math.round(params.ef)}%</span> · {efCategory(params.ef)}
          {median("ef") && <NotEntered />}
        </div>
        <div>
          <span className="text-gray-200">BP {Math.round(params.bp)}</span> mmHg
          {median("bp") && <NotEntered />}
          {lab && <LabChip title="Systolic pressure set by the ischaemia lab's exercise slider" />}
        </div>
        <div className="flex flex-wrap gap-0.5 pt-0.5">
          {findings.length === 0 ? (
            <span className="text-gray-500">No recorded ECG abnormality</span>
          ) : (
            findings.map((f) =>
              f.lab ? (
                <span
                  key={`lab-${f.text}`}
                  className="rounded border border-amber-500/70 bg-amber-950/80 px-1 text-amber-100"
                  title="From the ischaemia lab (teaching sandbox), not this patient's ECG. ST depression does not show where the ischaemia is."
                >
                  {f.text} <span className="font-semibold text-amber-300">lab</span>
                </span>
              ) : (
                <span key={f.text} className="rounded bg-slate-800 px-1 text-gray-200" title={f.recorded ? "Recorded" : "Not entered: the model's default"}>
                  {f.text}
                </span>
              ),
            )
          )}
        </div>
      </div>
    </div>
  );
}

/** A measurement that was not entered: the model (and so the twin) used the cohort median. */
function NotEntered() {
  return (
    <span className="font-normal text-gray-500" title="Not entered: the model used the cohort median">
      {" "}(cohort median)
    </span>
  );
}

/** Marks a value on the monitor as coming from the ischaemia lab rather than the patient's record. */
function LabChip({ title }: { title: string }) {
  return (
    <span title={title} className="ml-1 rounded border border-amber-500/70 px-0.5 align-[1px] text-[9px] font-semibold text-amber-300">
      lab
    </span>
  );
}
