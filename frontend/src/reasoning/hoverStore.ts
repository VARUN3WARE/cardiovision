/**
 * Which data-type group is hovered, shared by the explanation panel's strip and the 3D streams so that
 * hovering either highlights the same group in both.
 */
import { useSyncExternalStore } from "react";
import type { DataGroupId } from "./dataGroups";

let hovered: DataGroupId | null = null;
const listeners = new Set<() => void>();

export const reasoningHover = {
  get: () => hovered,
  set(id: DataGroupId | null) {
    if (id === hovered) return;
    hovered = id;
    listeners.forEach((f) => f());
  },
  subscribe(f: () => void) {
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  },
};

export function useReasoningHover(): DataGroupId | null {
  return useSyncExternalStore(reasoningHover.subscribe, reasoningHover.get);
}
