/**
 * Reasoning flow: the selected model's exact attributions, summed by data type, drawn as towers around the
 * heart with animated streams flowing into the estimate they moved: the selected artery's label, or, for the
 * overall CAD model, a halo around the heart's outline (never a point on the heart).
 *
 * Honesty rules: towers sit in a camera-facing frame around the heart, kept off its projected outline and
 * clear of the toolbar, captions and legend, so a stream is data, not a location; stream width and particle
 * count are proportional to |group log-odds| (the same numbers as the panel's "Evidence by data type" strip);
 * colours mean pushed up (red), pulled down (blue) or little effect (grey); typical fill-in values say so.
 *
 * Layout: towers at the heart's sides when they fit, else one-line chips in rows above and below the heart
 * (narrow views, or zoomed in). Performance: one instanced mesh for every particle, preallocated ribbon and
 * halo buffers updated in place, no allocations per frame.
 */
import { Html } from "@react-three/drei";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import * as THREE from "three";
import type { ShapExplanation, Target } from "../api/client";
import { VESSEL } from "../anatomy/registry";
import { formatFeatureValue } from "../featureLabels";
import { formatPct } from "../risk";
import { anchorHull, anchorWorldPosition } from "./anchors";
import {
  DOWN_COLOR, FILL_TEXT, MIN_EFFECT, NEUTRAL_COLOR, UP_COLOR, filledNote, groupAttributions, groupWeight, oddsText, sigmoid,
  type DataGroupId, type FillKind, type GroupContribution,
} from "./dataGroups";
import { reasoningHover, useReasoningHover } from "./hoverStore";

/** A DOM overlay's box in canvas pixels (from the canvas's top-left corner). */
export interface OverlayRect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Where the towers may go, in canvas pixels, measured from the 3D view's DOM overlays. */
export interface ReasoningFrame {
  /** Bottom of the toolbar. */
  top: number;
  /** The centred column of notices and captions under the toolbar (towers overlapping it start below it). */
  notices: OverlayRect | null;
  /** Overlays along the bottom edge (legend, view buttons, credit line). */
  bottoms: OverlayRect[];
}

interface Props {
  exp: ShapExplanation;
  target: Target;
  /** Columns the API filled in because the form left them empty, and how (dataGroups.typicalColumns). */
  typical: ReadonlyMap<string, FillKind> | null;
  /** The estimates on screen are newer than this explanation (inputs changed, or a what-if drag): dim. */
  stale: boolean;
  /** These are a what-if scenario's attributions. */
  whatIf: boolean;
  frame: ReasoningFrame;
  /** Narrow 3D view: chips above and below the heart instead of towers at its sides. */
  narrow: boolean;
  /** A click on a stream or tower with nothing else under it counts as a click on the background. */
  onBackgroundClick: () => void;
  /** Called when some streams end off screen (zoomed in) and are hidden, or all are back. */
  onOffscreen?: (some: boolean) => void;
}

/** Wide view: side (-1 left, +1 right) and row (top first). Narrow view: chip row (0 above, 1 below the heart). */
const SLOTS: Record<DataGroupId, { side: -1 | 1; row: number; chipRow: 0 | 1 }> = {
  history: { side: -1, row: 0, chipRow: 0 }, symptoms: { side: -1, row: 1, chipRow: 0 }, pretest: { side: -1, row: 2, chipRow: 1 },
  ecg: { side: 1, row: 0, chipRow: 0 }, echo: { side: 1, row: 1, chipRow: 1 }, labs: { side: 1, row: 2, chipRow: 1 },
  other: { side: 1, row: 3, chipRow: 1 },
};
const SAMPLES = 40; // points along each stream
const W_MAX = 0.14; // stream width (in screen half-heights) at GROUP_SCALE log-odds
const P_MAX = 40; // particles in a stream at GROUP_SCALE log-odds (density proportional below it)
const SPEED = 0.22; // stream lengths per second
const STREAM_REF = 0.9; // stream length (screen half-heights) that shows all of a stream's particles
const GLOW = 1.2; // particle colour multiplier
const RIBBON_OPACITY = 0.22;
const NODE_R = 0.026; // tower node radius (screen half-heights)
const BAR_W = 0.03;
const LABEL_GAP_PX = 5; // node to label
const EDGE_PX = 8; // label to canvas edge
const CLEAR_PX = 10; // halo to tower
const HALO_PX = 12; // heart outline to halo
const HALO_W_PX = 1.3; // half-width of the halo line
const RING_SEG = 96;
const MAX_ANGLE = (55 * Math.PI) / 180; // streams land on the halo's flanks
const DEFAULT_LABEL = [96, 30]; // px, until measured
const REASONING = { reasoning: true };
const CHIP_GAP_PX = 2; // gap-0.5
const BADGE_HALF_W = 70; // px, generous half-width of the "Overall CAD" badge

interface Stream extends GroupContribution {
  weight: number;
  colour: string;
  particles: number;
  slot: (typeof SLOTS)[DataGroupId];
}

export default function ReasoningFlow({
  exp, target, typical, stale, whatIf, frame, narrow, onBackgroundClick, onOffscreen,
}: Props) {
  const hovered = useReasoningHover();
  const still = useReducedMotion();
  const grouped = useMemo(() => groupAttributions(exp, typical), [exp, typical]);
  const streams: Stream[] = useMemo(() => grouped.groups.map((g) => {
    const weight = groupWeight(g.sum);
    return {
      ...g,
      weight,
      colour: Math.abs(g.sum) < MIN_EFFECT ? NEUTRAL_COLOR : g.sum > 0 ? UP_COLOR : DOWN_COLOR,
      particles: Math.round(P_MAX * weight),
      slot: SLOTS[g.id],
    };
  }), [grouped]);
  const n = streams.length;
  // The badge shows the probability these attributions add up to (never a newer, unexplained estimate).
  const probability = sigmoid(grouped.logOdds);
  const modelName = target === "cad" ? "overall CAD" : VESSEL[target].label;

  // Towers at the sides while they fit; chips otherwise (switched from the frame loop, with hysteresis).
  const [fits, setFits] = useState(true);
  const fitsRef = useRef(true);
  const chips = narrow || !fits;

  const layout = useMemo(() => {
    const sides: Record<-1 | 1, number[]> = { [-1]: [], [1]: [] };
    streams.forEach((s, k) => sides[s.slot.side].push(k));
    for (const side of [-1, 1] as const) sides[side].sort((a, b) => streams[a].slot.row - streams[b].slot.row);
    const rows: [number[], number[]] = [[], []]; // in display order
    streams.forEach((s, k) => rows[s.slot.chipRow].push(k));
    return { left: sides[-1], right: sides[1], rows };
  }, [streams]);

  const canvasH = useThree((st) => st.size.height);
  const canvasW = useThree((st) => st.size.width);
  const rig = useRef<THREE.Group>(null);
  const towers = useRef<(THREE.Group | null)[]>([]);
  const labelGroups = useRef<(THREE.Group | null)[]>([]);
  const labelEls = useRef<(HTMLElement | null)[]>([]);
  const chipEls = useRef<(HTMLElement | null)[]>([]);
  const badge = useRef<THREE.Group>(null);
  const badgeEl = useRef<HTMLDivElement>(null);
  const marker = useRef<THREE.Group>(null);
  const particleMesh = useRef<THREE.InstancedMesh>(null);
  const frameCount = useRef(0);
  const offscreen = useRef(false);
  const chipsDirty = useRef(true);

  // Per stream: measured label size (px), tower position (local), chip anchor (canvas px), tower centre (px).
  const geo = useMemo(() => ({
    label: new Float32Array(n * 2),
    towerX: new Float32Array(n),
    towerY: new Float32Array(n),
    chip: new Float32Array(n * 2).fill(NaN),
    towerPx: new Float32Array(n * 2),
    bottomRowTop: { px: NaN },
  }), [n]);

  // Chips go in two rows, each in display order; when the default split would wrap, the best-fitting one.
  const [split, setSplit] = useState<[number[], number[]] | null>(null);
  useEffect(() => setSplit(null), [layout]);
  const rows = split ?? layout.rows;
  const rowsRef = useRef(rows);
  rowsRef.current = rows;

  // Label and chip sizes (fonts differ between systems, so measure rather than guess).
  const observer = useMemo(() => (typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => {
    labelEls.current.forEach((el, k) => {
      if (el) geo.label.set([el.offsetWidth, el.offsetHeight], k * 2);
    });
    chipsDirty.current = true;
    const els = chipEls.current;
    const rowWidth = (els.find((el) => el?.offsetParent) ?.offsetParent as HTMLElement | null)?.clientWidth ?? 0;
    if (rowWidth > 0 && els.every((el, k) => el || k >= n)) {
      const best = bestSplit(layout.rows, els.map((el) => el?.offsetWidth ?? 0), rowWidth, CHIP_GAP_PX);
      const cur = rowsRef.current;
      if (best[0].join() !== cur[0].join()) setSplit(best);
    }
  })), [geo, layout, n]);
  useEffect(() => () => observer?.disconnect(), [observer]);
  const observe = (list: typeof labelEls, k: number) => (el: HTMLElement | null) => {
    const prev = list.current[k];
    if (prev === el) return;
    if (prev) observer?.unobserve(prev);
    list.current[k] = el;
    if (el) observer?.observe(el);
    chipsDirty.current = true;
  };

  // Bezier control points per stream (P0..P3, xyz each) in the rig's local frame.
  const ctrl = useMemo(() => new Float32Array(n * 12), [n]);
  const ribbons = useMemo(() => streams.map(() => ({ visible: ribbonGeometry(), hit: ribbonGeometry(true) })), [streams]);
  useEffect(() => () => ribbons.forEach((r) => { r.visible.dispose(); r.hit.dispose(); }), [ribbons]);
  const halo = useMemo(() => haloGeometry(), []);

  // Per-particle constants: stream, phase along it, lane across it, size jitter.
  const total = streams.reduce((s, x) => s + x.particles, 0);
  const particles = useMemo(() => {
    const stream = new Uint8Array(total);
    const rank = new Uint16Array(total);
    const phase = new Float32Array(total);
    const lane = new Float32Array(total);
    const size = new Float32Array(total);
    let i = 0;
    streams.forEach((s, k) => {
      for (let j = 0; j < s.particles; j++, i++) {
        stream[i] = k;
        rank[i] = j;
        phase[i] = (j + hash(i * 3.1)) / s.particles;
        lane[i] = hash(i * 7.7) * 2 - 1;
        size[i] = 0.75 + 0.5 * hash(i * 1.3);
      }
    });
    return { stream, rank, phase, lane, size, stride: new Uint8Array(streams.length).fill(1), hidden: new Uint8Array(streams.length) };
  }, [streams, total]);

  const materials = useMemo(() => ({
    particle: new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false, toneMapped: false }),
    particleGeometry: new THREE.IcosahedronGeometry(1, 1),
    halo: new THREE.MeshBasicMaterial({
      color: "#cbd5e1", transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthTest: false, depthWrite: false, toneMapped: false,
    }),
  }), []);
  const ribbonMaterials = useMemo(() => streams.map(() => new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: RIBBON_OPACITY, side: THREE.DoubleSide,
    depthTest: false, depthWrite: false, toneMapped: false,
  })), [streams]);
  useEffect(() => () => {
    materials.particle.dispose();
    materials.particleGeometry.dispose();
    materials.halo.dispose();
    halo.dispose();
  }, [materials, halo]);
  useEffect(() => () => ribbonMaterials.forEach((m) => m.dispose()), [ribbonMaterials]);
  useEffect(() => () => onOffscreen?.(false), [onOffscreen]);

  // Colours: ribbons fade in, then out over their last quarter; particles glow softly. A hovered group stays
  // bright while the others dim; everything dims while the estimates on screen are newer than these numbers.
  useLayoutEffect(() => {
    const c = new THREE.Color();
    const dimAll = stale ? 0.4 : 1;
    streams.forEach((s, k) => {
      const col = ribbons[k].visible.getAttribute("color") as THREE.BufferAttribute;
      for (let i = 0; i < SAMPLES; i++) {
        const t = i / (SAMPLES - 1);
        c.set(s.colour).multiplyScalar(smooth(0, 0.06, t) * (1 - smooth(0.75, 1, t)));
        col.setXYZ(2 * i, c.r, c.g, c.b);
        col.setXYZ(2 * i + 1, c.r, c.g, c.b);
      }
      col.needsUpdate = true;
      ribbonMaterials[k].opacity = dimAll * (hovered === null ? RIBBON_OPACITY : hovered === s.id ? 0.45 : 0.08);
    });
    materials.halo.opacity = stale ? 0.2 : 0.45;
    const mesh = particleMesh.current;
    if (!mesh || total === 0) return;
    for (let i = 0; i < total; i++) {
      const s = streams[particles.stream[i]];
      const dim = hovered !== null && hovered !== s.id;
      c.set(s.colour).lerp(WHITE, 0.2).multiplyScalar((dim ? 0.35 : GLOW) * (stale ? 0.45 : 1));
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [streams, ribbons, ribbonMaterials, materials, particles, total, hovered, stale]);

  useFrame((state) => {
    const g = rig.current;
    if (!g) return;
    const W = state.size.width, H = state.size.height;
    if (!W || !H) return;
    const cam = state.camera as THREE.PerspectiveCamera;
    cam.updateMatrixWorld(); // the camera rig may have moved it this frame
    const px = 2 / H; // local units per pixel: the rig's unit is half the view height
    const aspect = W / H;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));

    // Camera-facing frame through the heart's depth: local x/y are screen units, z points at the camera.
    if (!anchorWorldPosition("heart", _centre)) _centre.set(0, 0, 0);
    _dir.set(0, 0, -1).applyQuaternion(cam.quaternion);
    const depth = Math.max(0.5, _tmp.copy(_centre).sub(cam.position).dot(_dir));
    g.position.copy(cam.position).addScaledVector(_dir, depth);
    g.quaternion.copy(cam.quaternion);
    g.scale.setScalar(depth * tanHalf);
    g.updateMatrixWorld();
    _inv.copy(g.matrixWorld).invert();
    fitOutline(_inv, 1 / tanHalf, HALO_PX * px);
    const cx = _ell[0], cy = _ell[1], A = _ell[2], B = _ell[3];

    // Visible artery labels are obstacles too (they can stand off the heart's outline).
    if (frameCount.current % 10 === 0) readLabelBoxes(state.gl.domElement, W, H, px);
    // Wide layout (always computed: it also decides whether the towers fit). Each side is laid out below the
    // toolbar; if its top tower then reaches under the notices column, again below that column.
    const nf = frame.notices;
    let overflow = -Infinity;
    for (const side of [-1, 1] as const) {
      const ks = side < 0 ? layout.left : layout.right;
      if (!ks.length) continue;
      const bottomPx = H - floorAt(frame.bottoms, side < 0 ? 0 : W / 2, side < 0 ? W / 2 : W, H);
      let o = layoutSide(ks, side, streams, geo, frame.top + 6, bottomPx + 6, px, aspect);
      if (nf) {
        const k = ks[0];
        const half = labelW(geo, k) / 2;
        const xPx = toPxX(geo.towerX[k], W, px);
        const barTopPx = H / 2 - (geo.towerY[k] + 0.03 + barH(streams[k])) / px;
        if (xPx - half < nf.right && xPx + half > nf.left && barTopPx < nf.bottom + 6) {
          o = layoutSide(ks, side, streams, geo, nf.bottom + 6, bottomPx + 6, px, aspect);
        }
      }
      overflow = Math.max(overflow, o);
    }
    const fitsNow = fitsRef.current ? overflow <= 0 : overflow <= -12;
    if (fitsNow !== fitsRef.current) {
      fitsRef.current = fitsNow;
      setFits(fitsNow);
    }
    // Development only: layout numbers for scripted checks (window.__reasoning), a few times a second.
    if (import.meta.env.DEV && frameCount.current % 15 === 0) {
      (window as unknown as { __reasoning?: unknown }).__reasoning = {
        overflow, chips, ellipsePx: [toPxX(cx, W, px), H / 2 - cy / px, A / px, B / px],
        towers: streams.map((s, k) => ({ id: s.id, x: toPxX(geo.towerX[k], W, px), y: H / 2 - geo.towerY[k] / px, w: labelW(geo, k), h: labelH(geo, k) })),
        frame,
      };
    }

    // Stream starts: tower nodes, or the chips' edges facing the heart.
    if (chips && (chipsDirty.current || frameCount.current % 10 === 0)) {
      chipsDirty.current = false;
      readChips(chipEls.current, geo, rowsRef.current);
    }
    frameCount.current++;
    // The selected artery's label anchor, or (overall CAD model) the halo.
    const vessel = target !== "cad" && anchorWorldPosition(target, _target);
    if (vessel) {
      _target.applyMatrix4(_inv);
      marker.current?.position.set(_target.x, _target.y, _target.z);
    }
    let hiddenCount = 0;
    for (let k = 0; k < n; k++) {
      let x0 = geo.towerX[k], y0 = geo.towerY[k];
      if (chips) {
        if (Number.isNaN(geo.chip[2 * k])) continue;
        x0 = (geo.chip[2 * k] - W / 2) * px;
        y0 = (H / 2 - geo.chip[2 * k + 1]) * px;
      } else {
        towers.current[k]?.position.set(x0, y0, 0);
        labelGroups.current[k]?.position.set(0, -NODE_R - (LABEL_GAP_PX + labelH(geo, k) / 2) * px, 0);
        geo.towerPx[2 * k] = toPxX(x0, W, px);
        geo.towerPx[2 * k + 1] = H / 2 - y0 / px;
      }
      // End: the selected artery's label anchor, or the halo point facing the start (on the halo's flanks).
      let tx: number, ty: number, tz = 0, nx = 0, ny = 0;
      if (vessel) {
        tx = _target.x; ty = _target.y; tz = _target.z;
      } else {
        const th0 = Math.atan2((y0 - cy) / B, (x0 - cx) / A);
        const base = Math.cos(th0) >= 0 ? 0 : Math.PI;
        const rel = Math.atan2(Math.sin(th0 - base), Math.cos(th0 - base));
        const th = base + Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, rel));
        tx = cx + A * Math.cos(th);
        ty = cy + B * Math.sin(th);
        nx = Math.cos(th) / A;
        ny = Math.sin(th) / B;
        const nl = Math.hypot(nx, ny) || 1;
        nx /= nl; ny /= nl;
      }
      // An end off the visible canvas (zoomed in): hide the stream rather than run it under the toolbar or away.
      const off = Math.abs(tx) > aspect - 8 * px || ty > 1 - (frame.top + 4) * px || ty < -1 + 8 * px;
      particles.hidden[k] = off ? 1 : 0;
      if (off) hiddenCount++;
      const dx = tx - x0, dy = ty - y0, L = Math.hypot(dx, dy) || 1;
      // Particles per unit length stay proportional to |log-odds|: short streams show every n-th particle.
      particles.stride[k] = Math.max(1, Math.round(STREAM_REF / L));
      // Bend away from the heart's centre.
      let px1 = -dy / L, py1 = dx / L;
      if (px1 * ((x0 + tx) / 2 - cx) + py1 * ((y0 + ty) / 2 - cy) < 0) { px1 = -px1; py1 = -py1; }
      const o = k * 12;
      ctrl[o] = x0; ctrl[o + 1] = y0; ctrl[o + 2] = 0;
      if (vessel) {
        ctrl[o + 3] = x0 + dx * 0.4 + px1 * 0.1 * L; ctrl[o + 4] = y0 + dy * 0.4 + py1 * 0.1 * L; ctrl[o + 5] = 0.2;
        ctrl[o + 6] = tx - dx * 0.2 + px1 * 0.05 * L; ctrl[o + 7] = ty - dy * 0.2 + py1 * 0.05 * L; ctrl[o + 8] = tz + 0.35;
      } else {
        ctrl[o + 3] = x0 + dx * 0.35 + px1 * 0.06 * L; ctrl[o + 4] = y0 + dy * 0.35 + py1 * 0.06 * L; ctrl[o + 5] = 0;
        ctrl[o + 6] = tx + nx * 0.3 * L; ctrl[o + 7] = ty + ny * 0.3 * L; ctrl[o + 8] = 0;
      }
      ctrl[o + 9] = tx; ctrl[o + 10] = ty; ctrl[o + 11] = tz;

      const half = off ? 0 : 0.5 * W_MAX * streams[k].weight;
      const vis = ribbons[k].visible.getAttribute("position") as THREE.BufferAttribute;
      const hit = ribbons[k].hit.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < SAMPLES; i++) {
        bezier(ctrl, o, i / (SAMPLES - 1), _b);
        const taper = 1 - 0.4 * (i / (SAMPLES - 1));
        writeRibbon(vis.array as Float32Array, i, half * taper);
        writeRibbon(hit.array as Float32Array, i, off ? 0 : half * taper + 0.03);
      }
      vis.needsUpdate = true;
      hit.needsUpdate = true;
    }
    if ((hiddenCount > 0) !== offscreen.current) {
      offscreen.current = hiddenCount > 0;
      onOffscreen?.(offscreen.current);
    }

    // Halo and badge (overall CAD model): around the heart, the badge just outside it (below, else above).
    if (target === "cad") {
      const pos = halo.getAttribute("position") as THREE.BufferAttribute;
      const arr = pos.array as Float32Array;
      const w = HALO_W_PX * px;
      for (let i = 0; i <= RING_SEG; i++) {
        const c = RING_COS[i], s = RING_SIN[i];
        arr[6 * i] = cx + (A - w) * c; arr[6 * i + 1] = cy + (B - w) * s; arr[6 * i + 2] = 0;
        arr[6 * i + 3] = cx + (A + w) * c; arr[6 * i + 4] = cy + (B + w) * s; arr[6 * i + 5] = 0;
      }
      pos.needsUpdate = true;
      const halfBadge = 11 * px, gap = 6 * px;
      const bx = toPxX(cx, W, px);
      const floorPx = Math.min(
        floorAt(frame.bottoms, bx - BADGE_HALF_W, bx + BADGE_HALF_W, H),
        chips && !Number.isNaN(geo.bottomRowTop.px) ? geo.bottomRowTop.px : H,
      ) - 4;
      const floor = (H / 2 - floorPx) * px;
      const ceiling = 1 - ((nf ? nf.bottom : frame.top) + 4) * px;
      // Below the halo, else above it; with no room outside it, hidden (the caption names the model).
      let by = cy - B - gap - halfBadge;
      if (by - halfBadge < floor) by = cy + B + gap + halfBadge;
      const show = by + halfBadge <= ceiling;
      badge.current?.position.set(cx, by, 0);
      if (badgeEl.current && badgeEl.current.style.display !== (show ? "" : "none")) badgeEl.current.style.display = show ? "" : "none";
    } else {
      marker.current?.updateMatrixWorld();
    }

    const mesh = particleMesh.current;
    if (!mesh) return;
    const m = mesh.instanceMatrix.array as Float32Array;
    const time = still ? 0 : state.clock.elapsedTime;
    for (let i = 0; i < total; i++) {
      const k = particles.stream[i];
      const j = i * 16;
      if ((chips && Number.isNaN(geo.chip[2 * k])) || particles.hidden[k] || particles.rank[i] % particles.stride[k] !== 0) {
        m[j] = 0; m[j + 5] = 0; m[j + 10] = 0; m[j + 15] = 1;
        continue;
      }
      const t = (particles.phase[i] + time * SPEED) % 1;
      bezier(ctrl, k * 12, t, _b);
      const half = 0.5 * W_MAX * streams[k].weight * (1 - 0.4 * t);
      const r = (0.006 + 0.006 * streams[k].weight) * particles.size[i] * smooth(0, 0.06, t) * (1 - smooth(0.9, 1, t));
      m[j] = r; m[j + 5] = r; m[j + 10] = r; m[j + 15] = 1;
      m[j + 12] = _b[0] + _b[6] * half * particles.lane[i];
      m[j + 13] = _b[1] + _b[7] * half * particles.lane[i];
      m[j + 14] = _b[2] + 0.01;
    }
    mesh.instanceMatrix.needsUpdate = true;
    // After the orbit controls (-1) and the camera rig (-0.75) have moved the camera, before the HTML labels
    // (0) project themselves, so the towers and labels never lag a frame behind the camera.
  }, -0.5);

  const backgroundClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 2) return;
    // Only when nothing else (an artery, the heart wall) is under the pointer: then that object gets the click.
    if (e.intersections.every((i) => i.eventObject.userData.reasoning)) {
      e.stopPropagation();
      onBackgroundClick();
    }
  };
  const hoverMesh = (id: DataGroupId): MeshEvents => ({
    userData: REASONING,
    onPointerOver: () => reasoningHover.set(id),
    onPointerOut: () => reasoningHover.get() === id && reasoningHover.set(null),
    onClick: backgroundClick,
  });

  return (
    <group ref={rig}>
      {streams.map((s, k) => (
        <group key={s.id}>
          <mesh geometry={ribbons[k].visible} material={ribbonMaterials[k]} renderOrder={30} raycast={() => null} frustumCulled={false} />
          {/* Invisible, slightly wider hit area (raycasts ignore visibility). */}
          <mesh geometry={ribbons[k].hit} visible={false} {...hoverMesh(s.id)} />
          {!chips && (
            <group ref={(el) => (towers.current[k] = el)}>
              <TowerBar stream={s} dimmed={(hovered !== null && hovered !== s.id) || stale} events={hoverMesh(s.id)} />
              <group ref={(el) => (labelGroups.current[k] = el)}>
                <Html center zIndexRange={[14, 12]}>
                  <GroupButton
                    stream={s}
                    modelName={modelName}
                    active={hovered === s.id}
                    dimmed={(hovered !== null && hovered !== s.id) || stale}
                    elRef={observe(labelEls, k)}
                    variant="tower"
                  />
                </Html>
              </group>
              {hovered === s.id && (
                <Html zIndexRange={[40, 35]} style={{ pointerEvents: "none" }}>
                  <TowerTooltip stream={s} modelName={modelName} towerY={geo.towerPx[2 * k + 1]} canvasH={canvasH} />
                </Html>
              )}
            </group>
          )}
        </group>
      ))}
      {total > 0 && (
        <instancedMesh
          // Remount when the particle count changes (the instance buffer has a fixed size).
          key={total}
          ref={particleMesh}
          args={[materials.particleGeometry, materials.particle, total]}
          renderOrder={31}
          frustumCulled={false}
          raycast={() => null}
        />
      )}
      {target === "cad" ? (
        <>
          <mesh geometry={halo} material={materials.halo} renderOrder={29} raycast={() => null} frustumCulled={false} />
          <group ref={badge}>
            <Html center zIndexRange={[13, 12]} style={{ pointerEvents: "none" }}>
              <div ref={badgeEl} className={`whitespace-nowrap rounded-full border border-slate-300/70 bg-slate-950/85 px-2 py-0.5 text-[11px] font-semibold text-gray-100 shadow ${stale ? "opacity-60" : ""}`}>
                Overall CAD {formatPct(probability)}
                {whatIf && <span className="font-normal text-sky-300"> · what-if</span>}
              </div>
            </Html>
          </group>
        </>
      ) : (
        <group ref={marker}>
          <mesh renderOrder={32} raycast={() => null}>
            <ringGeometry args={[0.022, 0.029, 40]} />
            <meshBasicMaterial color="#f8fafc" transparent opacity={stale ? 0.35 : 0.7} depthTest={false} depthWrite={false} toneMapped={false} />
          </mesh>
        </group>
      )}
      {chips && (
        <Html fullscreen zIndexRange={[14, 12]} style={{ pointerEvents: "none" }}>
          {rows.map((row, r) => (
            <div
              key={r}
              className="absolute inset-x-1.5 flex flex-wrap justify-center gap-0.5"
              style={r === 0
                ? { top: Math.max(frame.top, frame.notices?.bottom ?? 0) + 6 }
                : { bottom: canvasH - floorAt(frame.bottoms, 0, canvasW, canvasH) + 6 }}
            >
              {row.map((k, i) => (
                <div key={streams[k].id} ref={observe(chipEls, k)} className="relative">
                  <GroupButton
                    stream={streams[k]}
                    modelName={modelName}
                    active={hovered === streams[k].id}
                    dimmed={(hovered !== null && hovered !== streams[k].id) || stale}
                    variant="chip"
                  />
                  {hovered === streams[k].id && (
                    <div
                      className={`absolute z-10 ${r === 0 ? "top-full mt-1.5" : "bottom-full mb-1.5"} ${
                        i === 0 && row.length > 1 ? "left-0" : i === row.length - 1 && row.length > 1 ? "right-0" : "left-1/2 -translate-x-1/2"
                      }`}
                    >
                      <TooltipCard stream={streams[k]} modelName={modelName} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </Html>
      )}
    </group>
  );
}

/** A data-type tower in 3D: glowing bar (height ∝ |log-odds|) on a node. */
type MeshEvents = {
  userData: typeof REASONING;
  onPointerOver: () => void;
  onPointerOut: () => void;
  onClick: (e: ThreeEvent<MouseEvent>) => void;
};

function TowerBar({ stream: s, dimmed, events }: { stream: Stream; dimmed: boolean; events: MeshEvents }) {
  const h = barH(s);
  return (
    <>
      <mesh renderOrder={33} {...events}>
        <sphereGeometry args={[NODE_R, 16, 12]} />
        <meshBasicMaterial color={s.colour} toneMapped={false} depthTest={false} transparent opacity={dimmed ? 0.4 : 1} />
      </mesh>
      <mesh position={[0, 0.03 + h / 2, 0]} renderOrder={33} {...events}>
        <boxGeometry args={[BAR_W, h, BAR_W]} />
        <meshBasicMaterial color={s.colour} toneMapped={false} depthTest={false} transparent opacity={dimmed ? 0.35 : 0.9} />
      </mesh>
    </>
  );
}

/**
 * A tower's name tag (two lines) or a one-line chip: hover (mouse), focus (keyboard) or tap (touch) shows the
 * group's findings.
 */
function GroupButton({ stream: s, modelName, active, dimmed, elRef, variant }: {
  stream: Stream;
  modelName: string;
  active: boolean;
  dimmed: boolean;
  elRef?: (el: HTMLElement | null) => void;
  variant: "tower" | "chip";
}) {
  // The pointer that pressed the button, and whether its findings were open before the press (a tap also focuses
  // the button, which may open them; the tap then decides from the state before it).
  const press = useRef({ type: "", wasOpen: false });
  const odds = `${oddsText(s.sum)}${variant === "tower" ? " odds" : ""} ${arrowOf(s.sum)}`;
  return (
    <button
      ref={elRef}
      type="button"
      aria-expanded={active}
      aria-label={`${s.label}: ${oddsText(s.sum)} odds, ${verbOf(s.sum)} the ${modelName} estimate. Show findings`}
      onPointerDown={(e) => {
        // Keep the press from the canvas behind, which would take it for a click on empty space.
        e.stopPropagation();
        press.current = { type: e.pointerType, wasOpen: reasoningHover.get() === s.id };
      }}
      onPointerEnter={(e) => e.pointerType === "mouse" && reasoningHover.set(s.id)}
      onPointerLeave={(e) => e.pointerType === "mouse" && reasoningHover.get() === s.id && reasoningHover.set(null)}
      onFocus={(e) => e.currentTarget.matches(":focus-visible") && reasoningHover.set(s.id)}
      onBlur={() => reasoningHover.get() === s.id && reasoningHover.set(null)}
      onClick={(e) => {
        e.stopPropagation();
        // Touch and keyboard toggle the findings; a mouse already shows them on hover.
        const { type, wasOpen } = press.current;
        if (type === "") reasoningHover.set(reasoningHover.get() === s.id ? null : s.id); // keyboard
        else if (type !== "mouse") reasoningHover.set(wasOpen ? null : s.id);
        press.current = { type: "", wasOpen: false };
      }}
      className={`pointer-events-auto cursor-default whitespace-nowrap rounded border bg-slate-950/85 text-center text-[10px] leading-tight shadow outline-none transition-opacity focus-visible:ring-1 focus-visible:ring-white ${
        variant === "tower" ? "block px-1.5 py-0.5" : "px-[3px] py-[3px]"} ${dimmed ? "opacity-50" : ""}`}
      style={{ borderColor: active ? "#f8fafc" : s.colour }}
    >
      {variant === "tower" ? (
        <>
          <span className="block font-semibold text-white">{s.short}</span>
          <span className="block" style={{ color: textColour(s) }}>{odds}</span>
        </>
      ) : (
        <>
          <span className="font-semibold text-white">{s.chip}</span> <span style={{ color: textColour(s) }}>{oddsText(s.sum)}{arrowOf(s.sum)}</span>
        </>
      )}
    </button>
  );
}

/** Wide view: the findings card beside its tower (right-hand towers open it to the left), kept on the canvas. */
function TowerTooltip({ stream: s, modelName, towerY, canvasH }: {
  stream: Stream;
  modelName: string;
  towerY: number;
  canvasH: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [dy, setDy] = useState(0);
  useLayoutEffect(() => {
    const h = ref.current?.offsetHeight ?? 0;
    if (!h || !canvasH) return;
    const top = towerY - h / 2;
    setDy(Math.min(Math.max(top, 6), canvasH - h - 6) - top);
  }, [towerY, canvasH]);
  const left = s.slot.side > 0;
  return (
    <div ref={ref} style={{ transform: `${left ? "translate(calc(-100% - 22px)," : "translate(22px,"} calc(-50% + ${dy}px))` }}>
      <TooltipCard stream={s} modelName={modelName} />
    </div>
  );
}

/** The group's exact sum and its top findings, each with its own odds factor; filled-in values said so. */
function TooltipCard({ stream: s, modelName }: { stream: Stream; modelName: string }) {
  const top = s.findings.slice(0, 4);
  const rest = s.findings.slice(4);
  const restSum = rest.reduce((a, f) => a + f.shap, 0);
  const note = filledNote(s.filled, s.inputs);
  return (
    <div className="w-[min(230px,calc(100vw-24px))] rounded bg-slate-950/95 p-2 text-left text-[11px] leading-tight text-gray-200 shadow-lg ring-1 ring-slate-600">
      <div className="font-semibold text-white">
        {s.label} <span style={{ color: textColour(s) }}>{oddsText(s.sum)} odds</span>
      </div>
      <div className="mb-1 text-[10px] text-gray-400">
        {verbOf(s.sum)} the {modelName} estimate ({s.sum >= 0 ? "+" : "−"}{Math.abs(s.sum).toFixed(2)} log-odds over {s.findings.length} inputs)
      </div>
      <ul className="space-y-0.5">
        {top.map((f) => (
          <li key={f.name} className="flex justify-between gap-2">
            <span className="truncate">
              {f.label}{" "}
              {f.fill
                ? <i className="text-gray-500">{FILL_TEXT[f.fill]}</i>
                : <span className="text-gray-500">{formatFeatureValue(f.name, f.value)}</span>}
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
      {note && <div className="mt-1 text-[10px] text-gray-400">{note}: defaults used.</div>}
      <div className="mt-1 text-[10px] text-gray-500">Associations in this referral cohort, not causes.</div>
    </div>
  );
}

const arrowOf = (sum: number) => (Math.abs(sum) < MIN_EFFECT ? "≈" : sum > 0 ? "↑" : "↓");
const verbOf = (sum: number) => (Math.abs(sum) < MIN_EFFECT ? "barely moved" : sum > 0 ? "pushed up" : "pulled down");
const textColour = (s: Stream) => (s.colour === NEUTRAL_COLOR ? "#94a3b8" : s.colour);
const barH = (s: Stream) => 0.015 + 0.2 * s.weight;
const labelW = (geo: { label: Float32Array }, k: number) => geo.label[2 * k] || DEFAULT_LABEL[0];
const labelH = (geo: { label: Float32Array }, k: number) => geo.label[2 * k + 1] || DEFAULT_LABEL[1];
const toPxX = (x: number, W: number, px: number) => W / 2 + x / px;

/**
 * Towers of one side (top first) between topPx and bottomPx (canvas px from the top and bottom edges): evenly
 * spaced rows, each as close to the heart's outline (_ell) as it can be without touching it, inside the canvas.
 * Writes geo.towerX/towerY (local units); returns how many px the layout lacks (<= 0 when it fits).
 */
function layoutSide(
  ks: number[], side: -1 | 1, streams: Stream[], geo: { label: Float32Array; towerX: Float32Array; towerY: Float32Array },
  topPx: number, bottomPx: number, px: number, aspect: number,
) {
  const first = ks[0], last = ks[ks.length - 1];
  const yFirst = 1 - topPx * px - (0.03 + barH(streams[first])) - 2 * px;
  const yLast = -1 + bottomPx * px + NODE_R + (LABEL_GAP_PX + labelH(geo, last) + 2) * px;
  let overflow = -Infinity;
  ks.forEach((k, r) => {
    geo.towerY[k] = ks.length === 1 ? (yFirst + yLast) / 2 : yFirst + ((yLast - yFirst) * r) / (ks.length - 1);
  });
  for (let r = 0; r + 1 < ks.length; r++) {
    const a = ks[r], b = ks[r + 1];
    const need = NODE_R + (LABEL_GAP_PX + labelH(geo, a) + 4) * px + 0.03 + barH(streams[b]);
    overflow = Math.max(overflow, (need - (geo.towerY[a] - geo.towerY[b])) / px);
  }
  const cx = _ell[0];
  for (const k of ks) {
    const y = geo.towerY[k];
    const lw = labelW(geo, k) * px, lh = labelH(geo, k) * px;
    const labelTop = y - NODE_R - LABEL_GAP_PX * px;
    const need = Math.max(
      clearance(y - NODE_R, y + 0.03 + barH(streams[k]), Math.max(NODE_R, BAR_W / 2), CLEAR_PX * px),
      clearance(labelTop - lh, labelTop, lw / 2, CLEAR_PX * px),
    );
    // Step outside any artery label on this side that the tower would overlap vertically.
    const yHi = y + 0.03 + barH(streams[k]), yLo = labelTop - lh, gap = 6 * px;
    let need2 = need;
    for (let j = 0; j < _labelCount; j++) {
      const l = _labels[4 * j], r = _labels[4 * j + 1], b = _labels[4 * j + 2], t = _labels[4 * j + 3];
      if (b > yHi + gap || t < yLo - gap || (side < 0 ? (l + r) / 2 > cx : (l + r) / 2 < cx)) continue;
      need2 = Math.max(need2, (side < 0 ? cx - l : r - cx) + lw / 2 + gap);
    }
    const room = (side < 0 ? cx + aspect : aspect - cx) - (lw / 2 + EDGE_PX * px);
    overflow = Math.max(overflow, (need2 - room) / px);
    geo.towerX[k] = cx + side * (need2 + Math.min(Math.max(0, room - need2) * 0.5, 0.3));
  }
  return overflow;
}

/** Screen boxes (local units: left, right, bottom, top) of the visible artery labels. */
function readLabelBoxes(canvas: HTMLCanvasElement, W: number, H: number, px: number) {
  const c = canvas.getBoundingClientRect();
  _labelCount = 0;
  for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-vessel-label]"))) {
    if (_labelCount >= MAX_LABELS) break;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue; // occluded (hidden) or not laid out
    const o = 4 * _labelCount++;
    _labels[o] = (r.left - c.left - W / 2) * px;
    _labels[o + 1] = (r.right - c.left - W / 2) * px;
    _labels[o + 2] = (H / 2 - (r.bottom - c.top)) * px;
    _labels[o + 3] = (H / 2 - (r.top - c.top)) * px;
  }
}

/**
 * The two chip rows (each kept in display order) that fit the row width with the fewest chips moved from the
 * default split; when nothing fits, the split whose wider row is narrowest.
 */
function bestSplit(rows: [number[], number[]], widths: number[], avail: number, gap: number): [number[], number[]] {
  const all = [...rows[0], ...rows[1]].sort((a, b) => a - b);
  const width = (ks: number[]) => ks.reduce((s, k) => s + widths[k], 0) + gap * Math.max(0, ks.length - 1);
  const home = new Set(rows[0]);
  let best: [number[], number[]] = rows, bestCost = Infinity;
  const size = rows[0].length;
  const pick = (start: number, top: number[]) => {
    if (top.length === size) {
      const bottom = all.filter((k) => !top.includes(k));
      const wide = Math.max(width(top), width(bottom));
      const moved = top.filter((k) => !home.has(k)).length;
      const cost = wide <= avail ? moved : 1000 + wide;
      if (cost < bestCost) {
        bestCost = cost;
        best = [[...top], bottom];
      }
      return;
    }
    for (let i = start; i < all.length; i++) pick(i + 1, [...top, all[i]]);
  };
  pick(0, []);
  return best;
}

/** The highest top edge (canvas px) among the bottom overlays spanning any of x0..x1, else the canvas bottom. */
function floorAt(rects: OverlayRect[], x0: number, x1: number, H: number) {
  let y = H;
  for (const r of rects) if (r.left < x1 && r.right > x0 && r.top < y) y = r.top;
  return y;
}

/** Chip anchors in canvas px: the middle of the edge facing the heart (bottom of the top row, top of the bottom row). */
function readChips(
  els: (HTMLElement | null)[],
  geo: { chip: Float32Array; bottomRowTop: { px: number } },
  rows: [number[], number[]],
) {
  geo.bottomRowTop.px = NaN;
  rows.forEach((row, r) => {
    for (const k of row) {
      const el = els[k];
      const rowEl = el?.offsetParent as HTMLElement | null;
      if (!el || !rowEl) {
        geo.chip[2 * k] = NaN;
        continue;
      }
      const top = rowEl.offsetTop + el.offsetTop;
      geo.chip[2 * k] = rowEl.offsetLeft + el.offsetLeft + el.offsetWidth / 2;
      geo.chip[2 * k + 1] = r === 0 ? top + el.offsetHeight : top;
      if (r === 1) geo.bottomRowTop.px = Number.isNaN(geo.bottomRowTop.px) ? top : Math.min(geo.bottomRowTop.px, top);
    }
  });
}

/**
 * The heart's outline on the rig plane: the cardiac-mass hull projected through the camera (perspective), then
 * the smallest ellipse of the bounding box's proportions that holds every point, plus a margin. Writes
 * [cx, cy, A, B] (local units) into _ell.
 */
function fitOutline(inv: THREE.Matrix4, camDist: number, margin: number) {
  const h = anchorHull("heart");
  if (!h) {
    _tmp.copy(_centre).applyMatrix4(inv);
    _ell[0] = _tmp.x; _ell[1] = _tmp.y; _ell[2] = 0.45; _ell[3] = 0.45;
    return;
  }
  _m.multiplyMatrices(inv, h.node.matrixWorld);
  const pts = h.hull, count = pts.length / 3;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < count; i++) {
    _tmp.set(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]).applyMatrix4(_m);
    const k = camDist / Math.max(1e-3, camDist - _tmp.z);
    const x = _tmp.x * k, y = _tmp.y * k;
    _proj[2 * i] = x; _proj[2 * i + 1] = y;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  // Smallest such ellipse over a few centre shifts and aspect ratios (a heart is wider at the base than at the
  // apex, so an ellipse centred on its bounding box would be needlessly large).
  const bx = (x0 + x1) / 2, by = (y0 + y1) / 2;
  const a0 = Math.max((x1 - x0) / 2, 1e-3), b0 = Math.max((y1 - y0) / 2, 1e-3);
  let best = Infinity;
  for (const ox of FIT_SHIFT_X) {
    for (const oy of FIT_SHIFT_Y) {
      for (const ratio of FIT_RATIO) {
        const cx = bx + ox * a0, cy = by + oy * b0, b = b0 * ratio;
        let s2 = 0;
        for (let i = 0; i < count; i++) {
          const dx = (_proj[2 * i] - cx) / a0, dy = (_proj[2 * i + 1] - cy) / b;
          s2 = Math.max(s2, dx * dx + dy * dy);
        }
        const area = s2 * a0 * b;
        if (area < best) {
          best = area;
          const sc = Math.sqrt(s2);
          _ell[0] = cx; _ell[1] = cy; _ell[2] = sc * a0 + margin; _ell[3] = sc * b + margin;
        }
      }
    }
  }
}
const FIT_SHIFT_X = [-0.1, 0, 0.1];
const FIT_SHIFT_Y = [-0.2, -0.1, 0, 0.1, 0.2];
const FIT_RATIO = [0.85, 1, 1.15];

/** Horizontal distance from the outline's centre that a box (rows y1..y2, half-width hw) needs to clear it. */
function clearance(y1: number, y2: number, hw: number, gap: number) {
  const a = _ell[2] + gap, b = _ell[3] + gap;
  const dy = Math.min(Math.max(_ell[1], y1), y2) - _ell[1];
  return (Math.abs(dy) >= b ? 0 : a * Math.sqrt(1 - (dy / b) ** 2)) + hw;
}

function ribbonGeometry(hit = false) {
  const g = new THREE.BufferGeometry();
  const pos = new THREE.BufferAttribute(new Float32Array(SAMPLES * 6), 3);
  pos.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute("position", pos);
  if (!hit) g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(SAMPLES * 6), 3));
  const index: number[] = [];
  for (let i = 0; i < SAMPLES - 1; i++) {
    const a = 2 * i;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  g.setIndex(index);
  // Fixed bounds covering the whole frame, so raycasts never need a per-frame recomputation.
  if (hit) g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4);
  return g;
}

function haloGeometry() {
  const g = new THREE.BufferGeometry();
  const pos = new THREE.BufferAttribute(new Float32Array((RING_SEG + 1) * 6), 3);
  pos.setUsage(THREE.DynamicDrawUsage);
  g.setAttribute("position", pos);
  const index: number[] = [];
  for (let i = 0; i < RING_SEG; i++) {
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

function smooth(e0: number, e1: number, x: number) {
  const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return k * k * (3 - 2 * k);
}

/** Deterministic hash in [0, 1) so particle layouts do not jump between renders. */
function hash(x: number) {
  const s = Math.sin(x * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";
function subscribeReducedMotion(onChange: () => void) {
  const m = typeof window !== "undefined" ? window.matchMedia?.(REDUCED_MOTION) : undefined;
  m?.addEventListener?.("change", onChange);
  return () => m?.removeEventListener?.("change", onChange);
}
/** prefers-reduced-motion: the particles stand still. */
function useReducedMotion() {
  return useSyncExternalStore(subscribeReducedMotion, () => !!window.matchMedia?.(REDUCED_MOTION).matches, () => false);
}

const RING_COS = Float32Array.from({ length: RING_SEG + 1 }, (_, i) => Math.cos((2 * Math.PI * i) / RING_SEG));
const RING_SIN = Float32Array.from({ length: RING_SEG + 1 }, (_, i) => Math.sin((2 * Math.PI * i) / RING_SEG));
const WHITE = new THREE.Color("#ffffff");
const _centre = new THREE.Vector3();
const _target = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _inv = new THREE.Matrix4();
const _m = new THREE.Matrix4();
const _b = new Float32Array(8);
const _ell = new Float64Array(4);
const _proj = new Float32Array(48 * 2);
const MAX_LABELS = 8;
const _labels = new Float32Array(4 * MAX_LABELS);
let _labelCount = 0;
