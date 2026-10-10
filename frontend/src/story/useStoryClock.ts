import { useEffect, useRef } from "react";

/** Where the story is: the step shown, whether it plays, and a run number that changes whenever a step (re)starts. */
export interface StoryState {
  step: number;
  playing: boolean;
  /** Increases on every jump to a step (also to the same one), restarting its clock. */
  run: number;
  /** The step opened at its end state (deep links): its clock starts full. */
  atEnd: boolean;
}

/**
 * The current step's clock: seconds into the step, advanced every animation frame while playing (paused time
 * does not count). Calls onTick with the time and onEnd once the step's duration is up. No re-renders of its own.
 */
export function useStoryClock(
  story: StoryState | null,
  duration: number,
  onTick: (t: number) => void,
  onEnd: () => void,
) {
  const elapsed = useRef(0);
  const ended = useRef(false);
  const live = useRef({ onTick, onEnd, duration });
  live.current = { onTick, onEnd, duration };
  const run = story?.run ?? -1;
  const atEnd = story?.atEnd ?? false;
  useEffect(() => {
    elapsed.current = atEnd ? duration : 0;
    ended.current = atEnd;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);
  const playing = !!story?.playing;
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      // A hidden tab or a long stall must not skip a step: count at most a tenth of a second per frame.
      elapsed.current += Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      const { onTick: tick, onEnd: end, duration: d } = live.current;
      tick(elapsed.current);
      if (!ended.current && elapsed.current >= d) {
        ended.current = true;
        end();
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, run]);
  return elapsed;
}
