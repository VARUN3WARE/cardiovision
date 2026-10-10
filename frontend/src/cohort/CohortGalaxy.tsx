import { useEffect, useLayoutEffect, useMemo, useRef, type ElementRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { CohortPatient } from "./knn";
import { CAD_COLOR, MAP_BG as BG, NO_CAD_COLOR } from "./palette";

/**
 * "Cohort galaxy": the 303 dataset patients as points on the 3D similarity map, coloured by their
 * angiography result; the current patient glows among its 10 most similar (ringed and linked).
 * Kept light for a second WebGL canvas: instanced meshes (points, rings), unlit materials, no
 * post-processing, and frameloop "demand" (it renders only when something changes or the user rotates
 * it). Hover picks the nearest point on screen, so no raycasting.
 */

const POINT_RADIUS = 0.03;
const RING_CAPACITY = 32; // neighbour rings (k is 10)
const PICK_RADIUS_PX = 14; // hit area ~28 px, larger than the marks
const RING_COLOR = "#94a3b8"; // slate-400: neighbours' rings, so white means only "this patient"
const CORE_RADIUS = 0.065; // "this patient": larger than a neighbour's point (0.03 x 1.7 = 0.051)

/** Hover target: a dataset row index, or -1 for the current patient. */
export interface GalaxyHover {
  id: number;
  x: number;
  y: number;
}

interface Props {
  patients: CohortPatient[];
  /** Index of the CAD target in each patient's `angio`. */
  cadIndex: number;
  neighbours: number[];
  /** Dataset row that is the current patient (drawn as the glow instead of a point). */
  self: number | null;
  here: [number, number, number];
  /** Row highlighted from the list or by hovering. */
  highlight: number | null;
  onHover: (h: GalaxyHover | null) => void;
  resetKey: number;
}

export default function CohortGalaxy(props: Props) {
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 1.5]}
      camera={{ position: [0, 0.25, 3.4], fov: 33, near: 0.1, far: 20 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={[BG]} />
      <fog attach="fog" args={[BG, 3.2, 6.4]} />
      <Galaxy {...props} />
    </Canvas>
  );
}

function Galaxy({ patients, cadIndex, neighbours, self, here, highlight, onHover, resetKey }: Props) {
  const positions = useMemo(() => {
    const a = new Float32Array(patients.length * 3);
    patients.forEach((p, i) => a.set(p.xyz, i * 3));
    return a;
  }, [patients]);
  const controls = useRef<ElementRef<typeof OrbitControls>>(null);
  useEffect(() => {
    if (resetKey) controls.current?.reset();
  }, [resetKey]);

  return (
    <>
      <Points positions={positions} patients={patients} cadIndex={cadIndex} neighbours={neighbours} self={self} highlight={highlight} />
      <Links positions={positions} neighbours={neighbours} here={here} />
      <Glow position={here} />
      <Picker positions={positions} skip={self} here={here} onHover={onHover} />
      <OrbitControls ref={controls} enableZoom={false} enablePan={false} enableDamping dampingFactor={0.12} rotateSpeed={0.7} />
    </>
  );
}

function Points({ positions, patients, cadIndex, neighbours, self, highlight }: {
  positions: Float32Array; patients: CohortPatient[]; cadIndex: number; neighbours: number[]; self: number | null; highlight: number | null;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const rings = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((s) => s.invalidate);
  const geometry = useMemo(() => new THREE.IcosahedronGeometry(POINT_RADIUS, 1), []);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const n = patients.length;

  useLayoutEffect(() => {
    const mesh = ref.current;
    const ring = rings.current;
    if (!mesh || !ring) return;
    const near = new Set(neighbours);
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const p = new THREE.Vector3();
    const c = new THREE.Color();
    const cad = new THREE.Color(CAD_COLOR);
    const none = new THREE.Color(NO_CAD_COLOR);
    const bg = new THREE.Color(BG);
    const white = new THREE.Color("#ffffff");
    const size = (i: number) => (i === self ? 0 : i === highlight ? 2.4 : near.has(i) ? 1.7 : 1);
    for (let i = 0; i < n; i++) {
      p.fromArray(positions, i * 3);
      mesh.setMatrixAt(i, m.compose(p, q, s.setScalar(size(i))));
      c.copy(patients[i].angio[cadIndex] ? cad : none);
      if (i === highlight) c.lerp(white, 0.35);
      else if (!near.has(i) && near.size) c.lerp(bg, 0.62); // context points recede; the 10 stay vivid
      mesh.setColorAt(i, c);
    }
    // A thin mid-grey ring (larger back faces) around each of the most similar; white is only "this patient".
    const shown = neighbours.slice(0, RING_CAPACITY);
    shown.forEach((idx, k) => {
      p.fromArray(positions, idx * 3);
      ring.setMatrixAt(k, m.compose(p, q, s.setScalar(size(idx) * 1.4)));
    });
    ring.count = shown.length;
    mesh.instanceMatrix.needsUpdate = true;
    ring.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    invalidate();
  }, [positions, patients, cadIndex, neighbours, self, highlight, n, invalidate]);

  return (
    <>
      <instancedMesh ref={ref} args={[geometry, undefined, n]} frustumCulled={false}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <instancedMesh ref={rings} args={[geometry, undefined, RING_CAPACITY]} frustumCulled={false}>
        <meshBasicMaterial color={RING_COLOR} side={THREE.BackSide} toneMapped={false} />
      </instancedMesh>
    </>
  );
}

/** Thin lines from the current patient to each of its neighbours. */
function Links({ positions, neighbours, here }: { positions: Float32Array; neighbours: number[]; here: [number, number, number] }) {
  const geometry = useMemo(() => new THREE.BufferGeometry(), []);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useLayoutEffect(() => {
    const a = new Float32Array(neighbours.length * 6);
    neighbours.forEach((idx, k) => {
      a.set(here, k * 6);
      a.set(positions.subarray(idx * 3, idx * 3 + 3), k * 6 + 3);
    });
    geometry.setAttribute("position", new THREE.BufferAttribute(a, 3));
    geometry.computeBoundingSphere();
    invalidate();
  }, [geometry, positions, neighbours, here, invalidate]);
  return (
    <lineSegments geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial color="#e2e8f0" transparent opacity={0.4} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

/** The current patient: a white core drawn over every point, with an additive halo. */
function Glow({ position }: { position: [number, number, number] }) {
  const texture = useMemo(() => {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.15, "rgba(255,255,255,0.8)");
    g.addColorStop(0.4, "rgba(255,255,255,0.16)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    return new THREE.CanvasTexture(canvas);
  }, []);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <group position={position}>
      <mesh renderOrder={10}>
        <sphereGeometry args={[CORE_RADIUS, 16, 12]} />
        <meshBasicMaterial color="#ffffff" depthTest={false} fog={false} toneMapped={false} />
      </mesh>
      <sprite scale={[0.36, 0.36, 1]} renderOrder={11}>
        <spriteMaterial map={texture} transparent depthTest={false} depthWrite={false} blending={THREE.AdditiveBlending} fog={false} toneMapped={false} />
      </sprite>
    </group>
  );
}

/** Nearest point on screen within PICK_RADIUS_PX of the pointer (the glow wins ties); no raycasting. */
function Picker({ positions, skip, here, onHover }: {
  positions: Float32Array; skip: number | null; here: [number, number, number]; onHover: (h: GalaxyHover | null) => void;
}) {
  const camera = useThree((s) => s.camera);
  const el = useThree((s) => s.gl.domElement);
  useEffect(() => {
    const v = new THREE.Vector3();
    const move = (e: PointerEvent) => {
      if (e.buttons) return onHover(null); // rotating
      const r = el.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      /** Squared screen distance from the pointer to a point (Infinity behind the camera). */
      const dist2 = (x: number, y: number, z: number) => {
        v.set(x, y, z).project(camera);
        if (v.z > 1) return Infinity;
        const dx = ((v.x + 1) / 2) * r.width - px;
        const dy = ((1 - v.y) / 2) * r.height - py;
        return dx * dx + dy * dy;
      };
      const limit = PICK_RADIUS_PX * PICK_RADIUS_PX;
      let best = dist2(here[0], here[1], here[2]) < limit ? -1 : null;
      if (best === null) {
        let bestD = limit;
        for (let i = 0; i < positions.length / 3; i++) {
          if (i === skip) continue;
          const d = dist2(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
          if (d < bestD) {
            bestD = d;
            best = i;
          }
        }
      }
      onHover(best === null ? null : { id: best, x: px, y: py });
    };
    const leave = () => onHover(null);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerleave", leave);
    return () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerleave", leave);
    };
  }, [camera, el, positions, skip, here, onHover]);
  return null;
}
