/**
 * Anatomical annotation pins (in the heart's local frame, see heartAnatomy.ts).
 * Ventricular landmarks are placed on the analytic surface; the rest sit on the
 * outer wall of the great vessels / atria they name.
 */
import * as THREE from "three";
import { surfaceNormal, surfacePoint } from "./heartAnatomy";

export interface AnatomyLabel {
  id: string;
  name: string;
  note: string;
  position: THREE.Vector3;
}

function onVentricle(u: number, v: number, lift = 0.06) {
  return surfacePoint(u, v).add(surfaceNormal(u, v).multiplyScalar(lift));
}

export const ANATOMY_LABELS: AnatomyLabel[] = [
  {
    id: "aorta", name: "Ascending aorta",
    note: "Leaves the left ventricle; the left and right coronary arteries arise from its root (aortic sinuses).",
    position: new THREE.Vector3(-0.3, 1.35, 0.33),
  },
  {
    id: "arch", name: "Aortic arch",
    note: "Gives off the brachiocephalic trunk, left common carotid and left subclavian arteries.",
    position: new THREE.Vector3(0.0, 2.25, -0.02),
  },
  {
    id: "pt", name: "Pulmonary trunk",
    note: "Carries deoxygenated blood from the right ventricle to the lungs; the left main coronary passes behind it.",
    position: new THREE.Vector3(-0.02, 1.02, 0.74),
  },
  {
    id: "svc", name: "Superior vena cava",
    note: "Returns venous blood from the upper body to the right atrium.",
    position: new THREE.Vector3(-0.92, 1.55, 0.02),
  },
  {
    id: "raa", name: "Right atrial appendage",
    note: "Overlaps the aortic root; the RCA runs in the groove just beneath it.",
    position: new THREE.Vector3(-0.55, 0.9, 0.66),
  },
  {
    id: "laa", name: "Left atrial appendage",
    note: "Sits over the origin of the left circumflex artery; a common site of thrombus in atrial fibrillation.",
    position: new THREE.Vector3(0.98, 0.66, 0.22),
  },
  {
    id: "rv", name: "Right ventricle",
    note: "Forms most of the anterior surface; supplied mainly by RCA branches.",
    position: onVentricle(0.55, 2.1),
  },
  {
    id: "lv", name: "Left ventricle",
    note: "Thick-walled systemic pump forming the left border; supplied by the LAD and LCX.",
    position: onVentricle(0.55, 0.35),
  },
  {
    id: "apex", name: "Apex",
    note: "Formed by the left ventricle; classically supplied by the distal LAD, which often wraps around it.",
    position: onVentricle(0.03, 1.2, 0.05),
  },
  {
    id: "ivg", name: "Anterior interventricular groove",
    note: "Marks the septum on the surface; carries the LAD and the great cardiac vein in epicardial fat.",
    position: onVentricle(0.42, 1.12, 0.09),
  },
];
