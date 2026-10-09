import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import { heartClock, mechanics, type BeatTiming, type Mechanics, type TwinParams } from "./twin";

/**
 * Calls `apply` every frame with the patient's cardiac mechanics at the shared clock's current beat time.
 * The clock itself is advanced once per frame by <TwinClock> (HeartScene), so every heart model, the
 * ECG monitor and later simulations stay in step.
 */
export function useHeartMotion(twin: TwinParams, timing: BeatTiming, apply: (m: Mechanics) => void) {
  const live = useRef({ twin, timing, apply });
  live.current = { twin, timing, apply };
  useFrame(() => {
    const { twin: p, timing: tm, apply: f } = live.current;
    f(mechanics(heartClock.t, tm, p));
  });
}
