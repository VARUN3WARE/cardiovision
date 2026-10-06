import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

export interface ArteryProbs {
  lad: number;
  lcx: number;
  rca: number;
}

/** Map a probability [0,1] to a green→yellow→red color. */
function probToColor(p: number): THREE.Color {
  // 0 = green (healthy), 0.5 = yellow, 1 = red
  const c = new THREE.Color();
  c.setHSL((1 - p) * 0.33, 0.9, 0.5); // hue 0.33 green → 0 red
  return c;
}

interface ArteryTubeProps {
  points: THREE.Vector3[];
  color: THREE.Color;
  name: string;
  onClick: () => void;
}

function ArteryTube({ points, color, name, onClick }: ArteryTubeProps) {
  const geometry = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3(points);
    return new THREE.TubeGeometry(curve, 64, 0.08, 12, false);
  }, [points]);
  return (
    <mesh
      geometry={geometry}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      name={name}
    >
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
    </mesh>
  );
}

interface HeartSceneProps {
  probs: ArteryProbs;
  onSelect: (vessel: keyof ArteryProbs) => void;
  selected: keyof ArteryProbs | null;
}

export default function HeartScene({ probs, onSelect, selected }: HeartSceneProps) {
  // TODO (Day 1/2): replace these placeholder tube paths with the real
  // LAD/LCX/RCA geometry from the loaded .glb heart model, or keep tubes as
  // a stylized overlay if the mesh has no named artery nodes.
  const ladPoints = [
    new THREE.Vector3(0, 0.2, 0.3),
    new THREE.Vector3(0.15, 0.5, 0.25),
    new THREE.Vector3(0.3, 0.8, 0.2),
  new THREE.Vector3(0.4, 1.1, 0.15),
  ];
  const lcxPoints = [
    new THREE.Vector3(-0.05, 0.3, 0.25),
    new THREE.Vector3(-0.25, 0.5, 0.1),
    new THREE.Vector3(-0.4, 0.4, -0.05),
  ];
  const rcaPoints = [
    new THREE.Vector3(0.1, 0.1, -0.2),
    new THREE.Vector3(0.2, -0.1, -0.3),
    new THREE.Vector3(0.15, -0.4, -0.25),
  ];

  return (
    <Canvas camera={{ position: [0, 0.5, 3], fov: 50 }} style={{ height: "100%" }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[3, 3, 3]} intensity={1.2} />
      <directionalLight position={[-3, -2, -2]} intensity={0.4} />

      {/* Stylized heart proxy until a real .glb is sourced (Day 1). */}
      <mesh>
        <sphereGeometry args={[0.9, 48, 48]} />
        <meshStandardMaterial color="#8b1a1a" roughness={0.6} transparent opacity={0.85} />
      </mesh>

      <ArteryTube
        points={ladPoints}
        color={probToColor(probs.lad)}
        name="LAD"
        onClick={() => onSelect("lad")}
      />
      <ArteryTube
        points={lcxPoints}
        color={probToColor(probs.lcx)}
        name="LCX"
        onClick={() => onSelect("lcx")}
      />
      <ArteryTube
        points={rcaPoints}
        color={probToColor(probs.rca)}
        name="RCA"
        onClick={() => onSelect("rca")}
      />

      {selected && (
        <mesh position={[0, 1.5, 0]}>
          <sphereGeometry args={[0.03, 8, 8]} />
          <meshBasicMaterial color="white" />
        </mesh>
      )}

      <OrbitControls enablePan={false} minDistance={2} maxDistance={6} />
    </Canvas>
  );
}
