/**
 * Reasoning flow: the selected model's exact attributions, summed by data type, drawn as towers around the
 * heart with animated streams flowing into the estimate they moved (the selected artery's label, or the
 * whole heart for the overall CAD model).
 *
 * Honesty rules: towers sit in a camera-facing frame around the heart, never on anatomy, so a stream is
 * data, not a location; stream width and particle count are proportional to |group log-odds| (the same
 * numbers as the panel's "Evidence by data type" strip); colours mean pushed up (red) or pulled down (blue).
 *
 * Performance: one instanced mesh for every particle, preallocated ribbon buffers updated in place, no
 * allocations per frame.
 */
import { Html } from "@react-three/drei";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { ShapExplanation, Target } from "../api/client";
import { VESSEL } from "../anatomy/registry";
import { formatFeatureValue } from "../featureLabels";
import { formatPct } from "../risk";
import { anchorWorldPosition } from "./anchors";
import {
  DOWN_COLOR, UP_COLOR, groupAttributions, groupWeight, oddsText,
  type DataGroupId, type GroupContribution,
} from "./dataGroups";
import { reasoningHover, useReasoningHover } from "./hoverStore";

/**
 * Tower positions in screen units: x as a fraction of the half-width, y of the half-height (the frame
 * scales with zoom and canvas shape, so towers stay around the heart and clear of the toolbar).
 */
const SLOTS: Record<DataGroupId, [number, number]> = {
  history: [-0.8, 0.46], symptoms: [-0.8, 0.08], pretest: [-0.8, -0.3],
  ecg: [0.8, 0.46], echo: [0.8, 0.08], labs: [0.8, -0.3], other: [0, -0.6],
};
const SAMPLES = 40; // points along each stream
const W_MAX = 0.16; // stream width (in screen half-heights) at GROUP_SCALE log-odds
const P_MAX = 70; // particles in a stream at GROUP_SCALE log-odds (density proportional below it)
const SPEED = 0.26; // stream lengths per second
const MIN_EFFECT = 0.05; // below this |log-odds| a group is drawn grey ("little effect")
const HEART_FRONT = 0.9; // world units in front of the heart centre where whole-heart streams land
const GREY = "#64748b";

interface Stream extends GroupContribution {
  weight: number;
  colour: string;
  slot: [number, number];
  particles: number;
}

interface Props {
  exp: ShapExplanation;
  target: Target;
  probability: number;
}

export default function ReasoningFlow({ exp, target, probability }: Props) {
  const hovered = useReasoningHover();
  const streams: Stream[] = useMemo(() => groupAttributions(exp).groups.map((g) => {
    const weight = groupWeight(g.sum);
    return {
      ...g,
      weight,
      colour: Math.abs(g.sum) < MIN_EFFECT ? GREY : g.sum > 0 ? UP_COLOR : DOWN_COLOR,
      slot: SLOTS[g.id],
      particles: Math.round(P_MAX * weight),
    };
  }), [exp]);
  const n = streams.length;

  const rig = useRef<THREE.Group>(null);
  const towers = useRef<(THREE.Group | null)[]>([]);
  const marker = useRef<THREE.Group>(null);
  const particleMesh = useRef<THREE.InstancedMesh>(null);

  // Bezier control points per stream (P0..P3, xyz each) in the rig's local frame.
  const ctrl = useMemo(() => new Float32Array(n * 12), [n]);
  const ribbons = useMemo(() => streams.map(() => ({ visible: ribbonGeometry(), hit: ribbonGeometry() })), [streams]);
  useEffect(() => () => ribbons.forEach((r) => { r.visible.dispose(); r.hit.dispose(); }), [ribbons]);

  // Per-particle constants: stream, phase along it, lane across it, size jitter.
  const total = streams.reduce((s, x) => s + x.particles, 0);
  const particles = useMemo(() => {
    const stream = new Uint8Array(total);
    const phase = new Float32Array(total);
    const lane = new Float32Array(total);
    const size = new Float32Array(total);
    let i = 0;
    streams.forEach((s, k) => {
      for (let j = 0; j < s.particles; j++, i++) {
        stream[i] = k;
        phase[i] = (j + hash(i * 3.1)) / s.particles;
        lane[i] = hash(i * 7.7) * 2 - 1;
        size[i] = 0.75 + 0.5 * hash(i * 1.3);
      }
    });
    return { stream, phase, lane, size };
  }, [streams, total]);

  const particleGeometry = useMemo(() => new THREE.IcosahedronGeometry(1, 1), []);
  const particleMaterial = useMemo(() => new THREE.MeshBasicMaterial({
    transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
  }), []);
  const ribbonMaterials = useMemo(() => streams.map(() => new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.3, side: THREE.DoubleSide,
    depthTest: false, depthWrite: false, toneMapped: false,
  })), [streams]);
  const hitMaterial = useMemo(() => new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, depthTest: false, side: THREE.DoubleSide }), []);
  useEffect(() => () => {
    particleGeometry.dispose();
    particleMaterial.dispose();
    hitMaterial.dispose();
  }, [particleGeometry, particleMaterial, hitMaterial]);
  useEffect(() => () => ribbonMaterials.forEach((m) => m.dispose()), [ribbonMaterials]);

  // Colours: ribbons fade in and out along their length; particles glow (above the bloom threshold).
  // A hovered group stays bright while the others dim.
  useLayoutEffect(() => {
    const c = new THREE.Color();
    streams.forEach((s, k) => {
      const col = ribbons[k].visible.getAttribute("color") as THREE.BufferAttribute;
      for (let i = 0; i < SAMPLES; i++) {
        const t = i / (SAMPLES - 1);
        c.set(s.colour).multiplyScalar(endsFade(t));
        col.setXYZ(2 * i, c.r, c.g, c.b);
        col.setXYZ(2 * i + 1, c.r, c.g, c.b);
      }
      col.needsUpdate = true;
      ribbonMaterials[k].opacity = hovered === null ? 0.34 : hovered === s.id ? 0.6 : 0.1;
    });
    const mesh = particleMesh.current;
    if (!mesh || total === 0) return;
    for (let i = 0; i < total; i++) {
      const s = streams[particles.stream[i]];
      const dim = hovered !== null && hovered !== s.id;
      c.set(s.colour).lerp(WHITE, 0.25).multiplyScalar(dim ? 0.35 : 1.9);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [streams, ribbons, ribbonMaterials, particles, total, hovered]);

  useFrame((state) => {
    const g = rig.current;
    if (!g) return;
    const cam = state.camera as THREE.PerspectiveCamera;
    // Camera-facing frame through the heart's depth: local x/y are screen units, z points at the camera.
    if (!anchorWorldPosition("heart", _centre)) _centre.set(0, 0, 0);
    _dir.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const depth = Math.max(0.5, _tmp.copy(_centre).sub(cam.position).dot(_dir));
    const halfH = depth * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const aspect = state.size.width / Math.max(1, state.size.height);
    g.position.copy(cam.position).addScaledVector(_dir, depth);
    g.quaternion.copy(cam.quaternion);
    g.scale.setScalar(halfH);
    for (let k = 0; k < n; k++) towers.current[k]?.position.set(streams[k].slot[0] * aspect, streams[k].slot[1], 0);
    g.updateMatrixWorld();

    // Target: the selected artery's label anchor, or the front of the whole heart.
    if (target === "cad" || !anchorWorldPosition(target, _target)) {
      _target.copy(cam.position).sub(_centre).normalize().multiplyScalar(HEART_FRONT).add(_centre);
    }
    g.worldToLocal(_target);
    const tx = _target.x, ty = _target.y, tz = _target.z;
    const pulse = 1 + 0.18 * Math.sin(state.clock.elapsedTime * 3);
    marker.current?.position.set(tx, ty, tz);
    marker.current?.scale.setScalar(pulse);
    marker.current?.updateMatrixWorld();

    for (let k = 0; k < n; k++) {
      const x0 = streams[k].slot[0] * aspect, y0 = streams[k].slot[1];
      const o = k * 12;
      ctrl[o] = x0; ctrl[o + 1] = y0; ctrl[o + 2] = 0;
      ctrl[o + 3] = x0 + (tx - x0) * 0.45; ctrl[o + 4] = y0 + 0.14; ctrl[o + 5] = 0.3;
      ctrl[o + 6] = tx + (x0 - tx) * 0.2; ctrl[o + 7] = ty + (y0 - ty) * 0.2 + 0.06; ctrl[o + 8] = tz + 0.35;
      ctrl[o + 9] = tx; ctrl[o + 10] = ty; ctrl[o + 11] = tz;

      const half = 0.5 * W_MAX * streams[k].weight;
      const vis = ribbons[k].visible.getAttribute("position") as THREE.BufferAttribute;
      const hit = ribbons[k].hit.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < SAMPLES; i++) {
        const t = i / (SAMPLES - 1);
        bezier(ctrl, o, t, _b);
        const taper = 1 - 0.4 * t;
        writeRibbon(vis.array as Float32Array, i, half * taper);
        writeRibbon(hit.array as Float32Array, i, half * taper + 0.03);
      }
      vis.needsUpdate = true;
      hit.needsUpdate = true;
      ribbons[k].hit.computeBoundingSphere();
    }

    const mesh = particleMesh.current;
    if (!mesh) return;
    const m = mesh.instanceMatrix.array as Float32Array;
    const time = state.clock.elapsedTime;
    for (let i = 0; i < total; i++) {
      const k = particles.stream[i];
      const t = (particles.phase[i] + time * SPEED) % 1;
      bezier(ctrl, k * 12, t, _b);
      const half = 0.5 * W_MAX * streams[k].weight * (1 - 0.4 * t);
      const r = (0.006 + 0.006 * streams[k].weight) * particles.size[i] * endsFade(t);
      const j = i * 16;
      m[j] = r; m[j + 5] = r; m[j + 10] = r; m[j + 15] = 1;
      m[j + 12] = _b[0] + _b[6] * half * particles.lane[i];
      m[j + 13] = _b[1] + _b[7] * half * particles.lane[i];
      m[j + 14] = _b[2] + 0.01;
    }
    mesh.instanceMatrix.needsUpdate = true;
    // After the orbit controls (-1) have moved the camera, before the HTML labels (0) project themselves,
    // so the tower labels never lag a frame behind the towers.
  }, -0.5);

  const modelName = target === "cad" ? "overall CAD" : VESSEL[target].label;
  return (
    <group ref={rig}>
      {streams.map((s, k) => (
        <group key={s.id}>
          <mesh geometry={ribbons[k].visible} material={ribbonMaterials[k]} renderOrder={30} raycast={() => null} frustumCulled={false} />
          <mesh
            geometry={ribbons[k].hit}
            material={hitMaterial}
            frustumCulled={false}
            onPointerOver={(e: ThreeEvent<PointerEvent>) => {
              e.stopPropagation();
              reasoningHover.set(s.id);
            }}
            onPointerMove={(e: ThreeEvent<PointerEvent>) => e.stopPropagation()}
            onPointerOut={() => reasoningHover.set(null)}
          />
          <group ref={(el) => (towers.current[k] = el)}>
            <Tower stream={s} active={hovered === s.id} dimmed={hovered !== null && hovered !== s.id} />
            {hovered === s.id && <Tooltip stream={s} modelName={modelName} />}
          </group>
        </group>
      ))}
      {total > 0 && (
        <instancedMesh
          // Remount when the particle count changes (the instance buffer has a fixed size).
          key={total}
          ref={particleMesh}
          args={[particleGeometry, particleMaterial, total]}
          renderOrder={31}
          frustumCulled={false}
          raycast={() => null}
        />
      )}
      <group ref={marker}>
        <mesh renderOrder={32} raycast={() => null}>
          <ringGeometry args={[0.03, 0.038, 40]} />
          <meshBasicMaterial color="#f8fafc" transparent opacity={0.85} depthTest={false} depthWrite={false} toneMapped={false} />
        </mesh>
        {target === "cad" && (
          <Html center zIndexRange={[12, 11]} position={[0, -0.1, 0]} style={{ pointerEvents: "none" }}>
            <div className="whitespace-nowrap rounded-full border border-slate-300/70 bg-slate-950/80 px-1.5 text-[9px] text-gray-200 shadow">
              Overall CAD {formatPct(probability)}
            </div>
          </Html>
        )}
      </group>
    </group>
  );
}

/** A data-type tower: glowing bar (height ∝ |log-odds|) on a node, with its name and odds factor. */
function Tower({ stream: s, active, dimmed }: { stream: Stream; active: boolean; dimmed: boolean }) {
  const h = 0.015 + 0.2 * s.weight;
  const arrow = Math.abs(s.sum) < MIN_EFFECT ? "≈" : s.sum > 0 ? "↑" : "↓";
  const hover = {
    onPointerOver: (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      reasoningHover.set(s.id);
    },
    onPointerOut: () => reasoningHover.set(null),
  };
  return (
    <>
      <mesh renderOrder={33} {...hover}>
        <sphereGeometry args={[0.026, 16, 12]} />
        <meshBasicMaterial color={s.colour} toneMapped={false} depthTest={false} transparent opacity={dimmed ? 0.4 : 1} />
      </mesh>
      <mesh position={[0, 0.03 + h / 2, 0]} renderOrder={33} {...hover}>
        <boxGeometry args={[0.03, h, 0.03]} />
        <meshBasicMaterial color={s.colour} toneMapped={false} depthTest={false} transparent opacity={dimmed ? 0.35 : 0.9} />
      </mesh>
      <Html center position={[0, -0.075, 0]} zIndexRange={[14, 12]}>
        <div
          onMouseEnter={() => reasoningHover.set(s.id)}
          onMouseLeave={() => reasoningHover.set(null)}
          className={`cursor-default whitespace-nowrap rounded border bg-slate-950/85 px-1.5 py-0.5 text-center text-[10px] leading-tight shadow transition-opacity ${dimmed ? "opacity-50" : ""}`}
          style={{ borderColor: active ? "#f8fafc" : s.colour }}
        >
          <div className="font-semibold text-white">{s.short}</div>
          <div style={{ color: s.colour === GREY ? "#94a3b8" : s.colour }}>
            {oddsText(s.sum)} odds {arrow}
          </div>
        </div>
      </Html>
    </>
  );
}

/** Hover card: the group's exact sum and its top findings, each with its own odds factor. */
function Tooltip({ stream: s, modelName }: { stream: Stream; modelName: string }) {
  const top = s.findings.slice(0, 4);
  const rest = s.findings.slice(4);
  const restSum = rest.reduce((a, f) => a + f.shap, 0);
  const left = s.slot[0] > 0; // right-hand towers open their card to the left
  const verb = Math.abs(s.sum) < MIN_EFFECT ? "barely moved" : s.sum > 0 ? "pushed up" : "pulled down";
  return (
    <Html zIndexRange={[40, 35]} style={{ pointerEvents: "none" }}>
      <div
        className="w-[230px] rounded bg-slate-950/95 p-2 text-[11px] leading-tight text-gray-200 shadow-lg ring-1 ring-slate-600"
        style={{ transform: left ? "translate(calc(-100% - 22px), -50%)" : "translate(22px, -50%)" }}
      >
        <div className="font-semibold text-white">
          {s.label} <span style={{ color: s.colour === GREY ? "#94a3b8" : s.colour }}>{oddsText(s.sum)} odds</span>
        </div>
        <div className="mb-1 text-[10px] text-gray-400">
          {verb} the {modelName} estimate ({s.sum >= 0 ? "+" : "−"}{Math.abs(s.sum).toFixed(2)} log-odds over {s.findings.length} inputs)
        </div>
        <ul className="space-y-0.5">
          {top.map((f) => (
            <li key={f.name} className="flex justify-between gap-2">
              <span className="truncate">
                {f.label} <span className="text-gray-500">{formatFeatureValue(f.name, f.value)}</span>
              </span>
              <span className={`shrink-0 ${Math.abs(f.shap) < 0.01 ? "text-gray-500" : f.shap > 0 ? "text-red-300" : "text-sky-300"}`}>
                {oddsText(f.shap)}
              </span>
            </li>
          ))}
          {rest.length > 0 && (
            <li className="flex justify-between gap-2 text-gray-500">
              <span>{rest.length} more</span>
              <span>{oddsText(restSum)}</span>
            </li>
          )}
        </ul>
        <div className="mt-1 text-[10px] text-gray-500">Associations in this referral cohort, not causes.</div>
      </div>
    </Html>
  );
}

function ribbonGeometry() {
  const g = new THREE.BufferGeometry();
  const pos = new THREE.BufferAttribute(new Float32Array(SAMPLES * 6), 3);
  pos.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute("position", pos);
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(SAMPLES * 6), 3));
  const index: number[] = [];
  for (let i = 0; i < SAMPLES - 1; i++) {
    const a = 2 * i;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  g.setIndex(index);
  return g;
}

/** Cubic Bezier point (out[0..2]), unit tangent (out[3..5]) and in-screen normal (out[6..7]). */
function bezier(c: Float32Array, o: number, t: number, out: Float32Array) {
  const u = 1 - t;
  const b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
  const d0 = -3 * u * u, d1 = 3 * u * u - 6 * u * t, d2 = 6 * u * t - 3 * t * t, d3 = 3 * t * t;
  for (let a = 0; a < 3; a++) {
    out[a] = b0 * c[o + a] + b1 * c[o + 3 + a] + b2 * c[o + 6 + a] + b3 * c[o + 9 + a];
    out[3 + a] = d0 * c[o + a] + d1 * c[o + 3 + a] + d2 * c[o + 6 + a] + d3 * c[o + 9 + a];
  }
  const len = Math.hypot(out[3], out[4]) || 1;
  out[6] = -out[4] / len;
  out[7] = out[3] / len;
}

/** Writes the two edge vertices of ribbon sample i from the current Bezier sample in _b. */
function writeRibbon(arr: Float32Array, i: number, half: number) {
  const j = i * 6;
  arr[j] = _b[0] + _b[6] * half; arr[j + 1] = _b[1] + _b[7] * half; arr[j + 2] = _b[2];
  arr[j + 3] = _b[0] - _b[6] * half; arr[j + 4] = _b[1] - _b[7] * half; arr[j + 5] = _b[2];
}

function endsFade(t: number) {
  const s = (e0: number, e1: number, x: number) => {
    const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return k * k * (3 - 2 * k);
  };
  return s(0, 0.06, t) * (1 - s(0.92, 1, t));
}

/** Deterministic hash in [0, 1) so particle layouts do not jump between renders. */
function hash(x: number) {
  const s = Math.sin(x * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

const WHITE = new THREE.Color("#ffffff");
const _centre = new THREE.Vector3();
const _target = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _b = new Float32Array(8);
