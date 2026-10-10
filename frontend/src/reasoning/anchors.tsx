/**
 * Shared registry of world-space targets for the reasoning streams. Each heart model (Atlas, Realistic,
 * Schematic) mounts a <FlowAnchor> at every artery's label anchor and one at the heart's centre; the
 * streams read their live world position each frame (so they follow the heartbeat and model switches).
 * Anchors are where a model's estimate is shown, never where a finding is located.
 */
import { useLayoutEffect, useRef } from "react";
import * as THREE from "three";
import type { Vessel } from "../anatomy/registry";

export type AnchorId = Vessel | "heart";

const anchors = new Map<AnchorId, THREE.Object3D>();

/** An empty node registered under `id` while mounted (the most recently mounted one wins). */
export function FlowAnchor({ id, position }: { id: AnchorId; position?: THREE.Vector3 | [number, number, number] }) {
  const ref = useRef<THREE.Group>(null);
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    anchors.set(id, node);
    return () => {
      if (anchors.get(id) === node) anchors.delete(id);
    };
  }, [id]);
  return <group ref={ref} position={position} />;
}

/** Writes the anchor's current world position into `out`; false if no heart has registered it. */
export function anchorWorldPosition(id: AnchorId, out: THREE.Vector3): boolean {
  const node = anchors.get(id);
  if (!node) return false;
  node.getWorldPosition(out);
  return true;
}
