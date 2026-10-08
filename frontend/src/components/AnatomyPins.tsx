import { Html } from "@react-three/drei";
import { useState, type RefObject } from "react";
import * as THREE from "three";
import { ANATOMY_LABELS } from "./anatomyLabels";

interface Props {
  occluders: RefObject<THREE.Object3D>[];
  /** Pin positions for a different heart model, keyed by label id (defaults to the schematic heart). */
  positions?: Record<string, [number, number, number]>;
}

/** Numbered annotation pins; click one to read a short clinical note. */
export default function AnatomyPins({ occluders, positions }: Props) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <>
      {ANATOMY_LABELS.map((label, i) => (
        <Html
          key={label.id}
          position={positions?.[label.id] ? new THREE.Vector3(...positions[label.id]) : label.position}
          center
          distanceFactor={5}
          zIndexRange={[20, 10]}
          occlude={occluders}
        >
          <div className="relative">
            <button
              // Keep the click away from the canvas (which would clear the selection).
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                setOpen(open === label.id ? null : label.id);
              }}
              title={label.name}
              className={`flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-bold shadow-md transition ${
                open === label.id
                  ? "border-white bg-sky-400 text-slate-900"
                  : "border-white/80 bg-sky-600/90 text-white hover:bg-sky-400"
              }`}
            >
              {i + 1}
            </button>
            {open === label.id && (
              <div className="absolute left-6 top-1/2 w-48 -translate-y-1/2 rounded bg-slate-900/95 p-2 text-[11px] leading-snug text-gray-300 shadow-lg">
                <div className="mb-0.5 font-semibold text-white">{label.name}</div>
                {label.note}
              </div>
            )}
          </div>
        </Html>
      ))}
    </>
  );
}
