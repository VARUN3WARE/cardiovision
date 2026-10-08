import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { useEffect, useMemo, type RefObject } from "react";
import * as THREE from "three";
import { formatPct } from "../risk";
import { patchMaterial } from "./atlasMaterials";
import { NEUTRAL_ARTERY, riskColor, taperedTube, undecided, type Vessel } from "./heartAnatomy";

import type { ArteryProbs } from "../anatomy/registry";
export type { ArteryProbs };

interface Props {
  /** null for vessels without a model output (the left main is not labelled in the dataset). */
  vessel: Vessel | null;
  name: string;
  curve: THREE.Curve<THREE.Vector3>;
  radiusAt: (t: number) => number;
  /** Where to show the vessel's risk label (null = no label). */
  labelPos: THREE.Vector3 | null;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  occluders: RefObject<THREE.Object3D>[];
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
}

/**
 * Risk-coloured coronary material shared by every heart model: the colour glides to a new estimate
 * (never flashing grey while inputs change), stripes mark the undecided middle band, the selected
 * vessel pulses and the others dim.
 */
export function useVesselMaterial(vessel: Vessel | null, probs: ArteryProbs | null, selected: Vessel | null, hovered: Vessel | null) {
  const material = useMemo(() => patchMaterial(new THREE.MeshPhysicalMaterial({
    color: NEUTRAL_ARTERY.clone(),
    emissive: NEUTRAL_ARTERY.clone(),
    roughness: 0.28,
    clearcoat: 1,
    clearcoatRoughness: 0.15,
    side: THREE.DoubleSide,
  }), { stripes: true }), []);
  useEffect(() => () => material.dispose(), [material]);
  const p = probs && vessel ? probs[vessel] : null;
  const target = useMemo(() => (p === null ? NEUTRAL_ARTERY.clone() : riskColor(p)), [p]);
  const isSelected = vessel !== null && vessel === selected;
  const dimmed = selected !== null && !isSelected;

  useFrame(({ clock }, dt) => {
    const k = 1 - Math.exp(-8 * Math.min(dt, 0.1));
    material.color.lerp(target, k);
    material.emissive.copy(material.color);
    const stripe = material.userData.uStripe as { value: number };
    stripe.value += ((p === null ? 0 : undecided(p)) - stripe.value) * k;
    // The selected vessel pulses bright enough to catch the bloom pass.
    material.emissiveIntensity = dimmed
      ? 0.04
      : isSelected
        ? 2.2 + 1.6 * Math.sin(clock.elapsedTime * 4)
        : vessel && vessel === hovered
          ? 1.0
          : p !== null && p >= 0.66
            ? 0.5
            : vessel ? 0.22 : 0.05;
    const opacity = dimmed ? 0.35 : 1;
    if (material.opacity !== opacity) {
      material.opacity = opacity;
      material.transparent = opacity < 1;
      material.needsUpdate = true;
    }
  });
  return { material, p, color: target };
}

/** One coronary vessel: risk-coloured tapered tube, clickable, with an optional risk label. */
export default function ArteryTube(props: Props) {
  const { vessel, name, curve, radiusAt, labelPos, probs, selected, hovered, occluders, onSelect, onHover } = props;
  const { geometry, tip, tipRadius } = useMemo(() => ({
    geometry: taperedTube(curve, radiusAt, 180, 14),
    tip: curve.getPoint(1),
    tipRadius: radiusAt(1),
  }), [curve, radiusAt]);
  const { material, p, color } = useVesselMaterial(vessel, probs, selected, hovered);

  const handlers = vessel
    ? {
        onClick: (e: { stopPropagation: () => void }) => {
          e.stopPropagation();
          onSelect(vessel === selected ? null : vessel);
        },
        onPointerOver: (e: { stopPropagation: () => void }) => {
          e.stopPropagation();
          onHover(vessel);
        },
        onPointerOut: () => onHover(null),
      }
    : {};

  return (
    <group>
      <mesh geometry={geometry} material={material} {...handlers} />
      {/* Rounded distal end */}
      <mesh position={tip} material={material} {...handlers}>
        <sphereGeometry args={[tipRadius, 10, 8]} />
      </mesh>
      {labelPos && vessel && (
        <VesselLabel
          vessel={vessel}
          name={name}
          position={labelPos}
          p={p}
          color={color}
          selected={selected}
          occluders={occluders}
          onSelect={onSelect}
        />
      )}
    </group>
  );
}

interface LabelProps {
  vessel: Vessel;
  name: string;
  position: THREE.Vector3;
  p: number | null;
  color: THREE.Color;
  selected: Vessel | null;
  occluders: RefObject<THREE.Object3D>[] | undefined;
  onSelect: (v: Vessel | null) => void;
}

/**
 * Clickable risk label pinned to a vessel (shared by every heart model). It reads "LAD · 70% chance"
 * (a chance of ≥50% narrowing), never "LAD 70%", which a cardiologist would read as 70% narrowing.
 */
export function VesselLabel({ vessel, name, position, p, color, selected, occluders, onSelect }: LabelProps) {
  const isSelected = vessel === selected;
  const mid = p !== null && undecided(p) > 0.5;
  return (
    <Html position={position} center distanceFactor={5} zIndexRange={[10, 0]} occlude={occluders}>
      <button
        // Stop the DOM event reaching the canvas, which would treat it as a missed click.
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(isSelected ? null : vessel);
        }}
        title={p !== null ? `Chance of ≥50% narrowing in the ${name}${mid ? " (undecided band)" : ""}` : undefined}
        className={`whitespace-nowrap rounded-full border-2 px-2 py-0.5 text-[11px] font-semibold shadow ${
          isSelected ? "bg-white text-slate-900" : "bg-slate-900/85 text-white"
        } ${mid ? "border-dashed" : ""}`}
        style={{ borderColor: `#${color.getHexString()}` }}
      >
        {name}
        {p !== null && <span className="ml-1 font-normal opacity-80">· {formatPct(p)} chance</span>}
      </button>
    </Html>
  );
}
