/**
 * Shared registry of world-space targets for the reasoning streams. Each heart model (Atlas, Realistic,
 * Schematic) mounts a <FlowAnchor> at every artery's label anchor and one at the centre of the heart's cardiac
 * mass (ventricles and atria, without the great vessels) carrying a small hull of that mass, so the streams can
 * draw a halo around the heart's outline and keep their towers off it. The streams read live world positions
 * each frame (so they follow the camera, the heartbeat and model switches). Anchors are where a model's
 * estimate is shown, never where a finding is located.
 */
import { useLayoutEffect, useRef } from "react";
import * as THREE from "three";
import type { Vessel } from "../anatomy/registry";

export type AnchorId = Vessel | "heart";

interface Anchor {
  node: THREE.Object3D;
  /** Hull points (xyz triplets) in the anchor's local frame. */
  hull: Float32Array | null;
}

const anchors = new Map<AnchorId, Anchor>();
// Development only: the registry for scripted checks (window.__anchors).
if (import.meta.env.DEV && typeof window !== "undefined") (window as unknown as { __anchors?: unknown }).__anchors = anchors;

/** An empty node registered under `id` while mounted (the most recently mounted one wins). */
export function FlowAnchor({ id, position, hull = null }: {
  id: AnchorId;
  position?: THREE.Vector3 | [number, number, number];
  hull?: Float32Array | null;
}) {
  const ref = useRef<THREE.Group>(null);
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const entry = { node, hull };
    anchors.set(id, entry);
    return () => {
      if (anchors.get(id) === entry) anchors.delete(id);
    };
  }, [id, hull]);
  return <group ref={ref} position={position} />;
}

/** Writes the anchor's current world position into `out`; false if no heart has registered it. */
export function anchorWorldPosition(id: AnchorId, out: THREE.Vector3): boolean {
  const a = anchors.get(id);
  if (!a) return false;
  a.node.getWorldPosition(out);
  return true;
}

/** The anchor's node (world matrix up to date) and hull, or null. */
export function anchorHull(id: AnchorId): { node: THREE.Object3D; hull: Float32Array } | null {
  const a = anchors.get(id);
  if (!a?.hull) return null;
  a.node.updateWorldMatrix(true, false);
  return { node: a.node, hull: a.hull };
}

// Fibonacci-sphere directions for the support-point hull.
const HULL_DIRECTIONS = (() => {
  const n = 48;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(1 - y * y);
    const phi = i * Math.PI * (3 - Math.sqrt(5));
    out.set([Math.cos(phi) * r, y, Math.sin(phi) * r], i * 3);
  }
  return out;
})();

export interface HullSource {
  /** Vertex positions (read through the attribute, so interleaved or normalised storage is fine). */
  positions: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
  /** Applied to every point first (e.g. a mesh's local transform). */
  matrix?: THREE.Matrix4;
  /** Keep only the points this accepts (after `matrix`). */
  keep?: (x: number, y: number, z: number, index: number) => boolean;
}

/**
 * A cheap convex-hull stand-in: the extreme point of the sources in each of 48 directions, relative to the
 * centre of their bounding box. Returns that centre and the points (xyz triplets, relative to it); computed
 * once per model, then projected every frame by the reasoning view.
 */
export function supportHull(sources: HullSource[]): { centre: THREE.Vector3; hull: Float32Array } {
  const nDir = HULL_DIRECTIONS.length / 3;
  const best = new Float64Array(nDir).fill(-Infinity);
  const pts = new Float32Array(nDir * 3);
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (const s of sources) {
    const a = s.positions;
    for (let i = 0; i < a.count; i++) {
      v.fromBufferAttribute(a, i);
      if (s.matrix) v.applyMatrix4(s.matrix);
      if (s.keep && !s.keep(v.x, v.y, v.z, i)) continue;
      box.expandByPoint(v);
      for (let d = 0; d < nDir; d++) {
        const dot = v.x * HULL_DIRECTIONS[3 * d] + v.y * HULL_DIRECTIONS[3 * d + 1] + v.z * HULL_DIRECTIONS[3 * d + 2];
        if (dot > best[d]) {
          best[d] = dot;
          pts[3 * d] = v.x;
          pts[3 * d + 1] = v.y;
          pts[3 * d + 2] = v.z;
        }
      }
    }
  }
  const centre = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
  for (let d = 0; d < nDir; d++) {
    pts[3 * d] -= centre.x;
    pts[3 * d + 1] -= centre.y;
    pts[3 * d + 2] -= centre.z;
  }
  return { centre, hull: pts };
}
