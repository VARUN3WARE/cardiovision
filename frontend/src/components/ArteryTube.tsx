import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import { useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";
import { NEUTRAL_ARTERY, riskColor, taperedTube, type Vessel } from "./heartAnatomy";

export interface ArteryProbs {
  lad: number;
  lcx: number;
  rca: number;
}

interface Props {
  /** null for the left main, which is coloured by the higher of LAD / LCX. */
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

/** One coronary vessel: risk-coloured tapered tube, clickable, with an optional risk label. */
export default function ArteryTube(props: Props) {
  const { vessel, name, curve, radiusAt, labelPos, probs, selected, hovered, occluders, onSelect, onHover } = props;
  const { geometry, tip, tipRadius } = useMemo(() => ({
    geometry: taperedTube(curve, radiusAt, 180, 14),
    tip: curve.getPoint(1),
    tipRadius: radiusAt(1),
  }), [curve, radiusAt]);
  const material = useRef<THREE.MeshPhysicalMaterial>(null);
  const capMaterial = useRef<THREE.MeshPhysicalMaterial>(null);

  const p = probs
    ? vessel ? probs[vessel] : Math.max(probs.lad, probs.lcx) // left main feeds LAD + LCX
    : null;
  const color = useMemo(() => (p === null ? NEUTRAL_ARTERY.clone() : riskColor(p)), [p]);
  const isSelected = vessel !== null && vessel === selected;
  const dimmed = selected !== null && !isSelected;

  useFrame(({ clock }) => {
    // Selected vessel pulses bright enough to catch the bloom pass.
    const glow = dimmed
      ? 0.04
      : isSelected
        ? 2.2 + 1.6 * Math.sin(clock.elapsedTime * 4)
        : vessel && vessel === hovered
          ? 1.0
          : p !== null && p >= 0.66
            ? 0.5
            : 0.2;
    if (material.current) material.current.emissiveIntensity = glow;
    if (capMaterial.current) capMaterial.current.emissiveIntensity = glow;
  });

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

  const materialProps = {
    color,
    emissive: color,
    roughness: 0.28,
    clearcoat: 1,
    clearcoatRoughness: 0.15,
    transparent: dimmed,
    opacity: dimmed ? 0.4 : 1,
  };

  return (
    <group>
      <mesh geometry={geometry} {...handlers}>
        <meshPhysicalMaterial ref={material} {...materialProps} />
      </mesh>
      {/* Rounded distal end */}
      <mesh position={tip} {...handlers}>
        <sphereGeometry args={[tipRadius, 10, 8]} />
        <meshPhysicalMaterial ref={capMaterial} {...materialProps} />
      </mesh>
      {labelPos && vessel && (
        <Html position={labelPos} center distanceFactor={5} zIndexRange={[10, 0]} occlude={occluders}>
          <button
            // Stop the DOM event reaching the canvas, which would treat it as a missed click.
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(vessel === selected ? null : vessel);
            }}
            className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-[11px] font-semibold shadow ${
              isSelected ? "bg-white text-slate-900" : "bg-slate-900/85 text-white"
            }`}
            style={{ borderColor: `#${color.getHexString()}` }}
          >
            {name}
            {p !== null && <span className="ml-1 font-normal opacity-80">{(p * 100).toFixed(0)}%</span>}
          </button>
        </Html>
      )}
    </group>
  );
}
