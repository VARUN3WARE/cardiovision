/**
 * Shader patches for the atlas meshes (which have no UVs):
 *  - bump: procedural tissue relief from 3D noise in object space (derivative bump mapping);
 *  - cutaway: discards whatever lies between the camera and the heart within a radius, so the
 *    chest wall opens a window onto the heart from any viewing angle;
 *  - fresnel: alpha rises towards silhouettes (ghosted skin);
 *  - lab: the ischaemia lab (a teaching sandbox). Every ventricular structure ("wall", "vessel", "flow")
 *    can cancel part of the twin's squeeze where the chosen artery's territory is hypokinetic; the wall
 *    also takes the ischaemic tint, and the coronary trees ("flow") carry glowing pulses moving outward
 *    from the ostia (density beyond the narrowing scaled by the flow that gets through) and the waist
 *    drawn at the narrowing.
 */
import * as THREE from "three";
import { ATLAS_SCALE } from "./atlasAnatomy";

/** Shared by every patched material, so one update moves every cutaway. Radius in world units. */
export const CUTAWAY = {
  uCutRadius: { value: 0.085 * ATLAS_SCALE }, // the heart's silhouette radius plus a margin
  uCutOn: { value: 1 },
};

/** Ischaemia lab uniforms, shared by every lab-patched material (one atlas heart at a time). */
export const LAB = {
  uLabPivot: { value: new THREE.Vector3() }, // ventricular pivot (centred atlas frame, metres)
  uLabScale: { value: new THREE.Vector3(1, 1, 1) }, // the ventricles' scale this frame (the twin's squeeze)
  uLabHypo: { value: 0 }, // share of the squeeze lost in the territory core: 1 − wall thickening
  uLabTint: { value: 0 }, // ischaemic tint strength
  uLabOutline: { value: 0 }, // territory outline strength
  uFlowRel: { value: 1 }, // share of pulses that get through the narrowing (flow relative to demand)
  uFlowGain: { value: 0 }, // pulse brightness
  uFlowColor: { value: new THREE.Color("#ffe9b8") },
  uPinch: { value: 0 }, // drawn narrowing (diameter fraction)
  uMarkerPos: { value: new THREE.Vector3() }, // narrowing centre on the trunk axis (metres)
  uMarkerDir: { value: new THREE.Vector3(0, 1, 0) }, // trunk direction there
};
/** Distance between successive flow pulses along an artery, mm. */
export const FLOW_SPACING_MM = 12;

interface Patch {
  bump?: { freq: number; strength: number };
  cutaway?: boolean;
  fresnel?: { min: number; power: number };
  /** Diagonal stripes whose strength is material.userData.uStripe.value (0..1): the "undecided" band. */
  stripes?: boolean;
  /** Ischaemia lab: "wall" (hypokinesis + tint), "vessel" (hypokinesis) or "flow" (hypokinesis + pulses + waist). */
  lab?: "wall" | "vessel" | "flow";
}

// Vertex stage: pinch at the narrowing (flow only), then hypokinesis. The group around the ventricles
// scales them by uLabScale about uLabPivot; pre-scaling by local/uLabScale leaves the share aLabW·uLabHypo
// of that squeeze undone.
const LAB_VERTEX = (flow: boolean) => `
  {
    ${flow ? `vec3 cvRel = transformed - uMarkerPos;
    vec3 cvAxis = uMarkerPos + uMarkerDir * dot(cvRel, uMarkerDir);
    transformed = cvAxis + (transformed - cvAxis) * (1.0 - uPinch * aPinch);` : ""}
    vec3 cvS = max(uLabScale, vec3(1e-3));
    vec3 cvLocal = 1.0 - (1.0 - cvS) * (1.0 - clamp(aLabW * uLabHypo, 0.0, 1.0));
    transformed = uLabPivot + (transformed - uLabPivot) * cvLocal / cvS;
  }`;

// Simplex noise 3D: Ian McEwan / Ashima Arts, MIT licence (webgl-noise).
const NOISE = /* glsl */ `
vec3 cvMod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 cvMod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 cvPermute(vec4 x){return cvMod289(((x*34.0)+10.0)*x);}
vec4 cvTaylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float cvSnoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.0-g;vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;
  i=cvMod289(i);
  vec4 p=cvPermute(cvPermute(cvPermute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857;vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0;vec4 s1=floor(b1)*2.0+1.0;vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=cvTaylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.5-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);m=m*m;
  return 105.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
vec3 cvPerturb(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDir){
  vec3 sx=normalize(dFdx(surf_pos)); vec3 sy=normalize(dFdy(surf_pos));
  vec3 r1=cross(sy,surf_norm); vec3 r2=cross(surf_norm,sx);
  float det=dot(sx,r1)*faceDir;
  vec3 grad=sign(det)*(dHdxy.x*r1+dHdxy.y*r2);
  return normalize(abs(det)*surf_norm-grad);
}
`;

export function patchMaterial<T extends THREE.Material>(material: T, patch: Patch): T {
  const key = JSON.stringify(patch);
  material.customProgramCacheKey = () => key;
  if (patch.stripes) material.userData.uStripe = { value: 0 };
  if (patch.lab === "flow") material.userData.uPhase = { value: 0 };
  material.onBeforeCompile = (shader) => {
    if (patch.stripes) shader.uniforms.uStripe = material.userData.uStripe;
    if (patch.lab) {
      Object.assign(shader.uniforms, LAB);
      if (patch.lab === "flow") shader.uniforms.uPhase = material.userData.uPhase;
    }
    shader.uniforms.uCutRadius = CUTAWAY.uCutRadius;
    shader.uniforms.uCutOn = CUTAWAY.uCutOn;
    shader.uniforms.uBumpFreq = { value: patch.bump?.freq ?? 0 };
    shader.uniforms.uBumpStrength = { value: patch.bump?.strength ?? 0 };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vCvObj;\nvarying vec3 vCvWorld;")
      .replace(
        "#include <project_vertex>",
        "#include <project_vertex>\nvCvObj = transformed;\nvCvWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;",
      );
    if (patch.lab) {
      const flow = patch.lab === "flow";
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          `#include <common>
attribute float aLabW;
uniform vec3 uLabPivot;
uniform vec3 uLabScale;
uniform float uLabHypo;
varying float vCvLabW;
${flow ? `attribute float aOstium;
attribute float aRestrict;
attribute float aPinch;
uniform float uPinch;
uniform vec3 uMarkerPos;
uniform vec3 uMarkerDir;
varying float vCvOstium;
varying float vCvRestrict;` : ""}`,
        )
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
${LAB_VERTEX(flow)}
  vCvLabW = aLabW;
  ${flow ? "vCvOstium = aOstium;\n  vCvRestrict = aRestrict;" : ""}`,
        );
    }
    let frag = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
varying vec3 vCvObj;
varying vec3 vCvWorld;
uniform float uCutRadius;
uniform float uCutOn;
uniform float uBumpFreq;
uniform float uBumpStrength;
${patch.stripes ? "uniform float uStripe;" : ""}
${patch.lab ? "varying float vCvLabW;\nuniform float uLabTint;\nuniform float uLabOutline;" : ""}
${patch.lab === "flow" ? "varying float vCvOstium;\nvarying float vCvRestrict;\nuniform float uPhase;\nuniform float uFlowRel;\nuniform float uFlowGain;\nuniform vec3 uFlowColor;" : ""}
${patch.bump ? NOISE : ""}`,
    );
    if (patch.lab === "wall") {
      // Ischaemic territory: dark blue tint, plus a faint outline where the territory weight crosses 0.5.
      frag = frag.replace(
        "#include <color_fragment>",
        `#include <color_fragment>
  float cvIsch = uLabTint * smoothstep(0.3, 0.85, vCvLabW);
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.03, 0.07, 0.27), cvIsch);
  {
    float cvFw = max(fwidth(vCvLabW), 1e-4);
    float cvLine = 1.0 - smoothstep(0.5 * cvFw, 1.5 * cvFw, abs(vCvLabW - 0.5));
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.6, 0.76, 1.0), cvLine * uLabOutline);
  }`,
      );
      // The red sheen would turn the ischaemic blue purple at grazing angles.
      frag = frag.replace(
        "#include <lights_physical_fragment>",
        `#include <lights_physical_fragment>
  #ifdef USE_SHEEN
    material.sheenColor *= 1.0 - cvIsch;
  #endif`,
      );
    }
    if (patch.lab === "flow") {
      // Comet-shaped pulses moving outward (bright head downstream). Beyond the narrowing each pulse gets
      // through with probability uFlowRel (a per-pulse hash, so a pulse either travels on or is missing).
      frag = frag.replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
  {
    float cvPh = (vCvOstium - uPhase) / ${FLOW_SPACING_MM.toFixed(1)};
    float cvN = floor(cvPh);
    float cvX = fract(cvPh);
    float cvPulse = smoothstep(0.5, 0.86, cvX) * (1.0 - smoothstep(0.86, 0.96, cvX));
    float cvKeep = step(fract(sin(cvN * 12.9898 + 4.1) * 43758.5453), uFlowRel);
    cvPulse *= mix(1.0, cvKeep, step(0.5, vCvRestrict));
    totalEmissiveRadiance += uFlowColor * (cvPulse * uFlowGain);
  }`,
      );
    }
    if (patch.stripes) {
      // World-space diagonal stripes (same width on every heart model); dim both the base and emissive colour.
      frag = frag.replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
  {
    float cvStripe = step(0.5, fract(dot(vCvWorld, vec3(1.0, 1.0, 0.6)) * 6.0)) * uStripe;
    diffuseColor.rgb *= 1.0 - 0.55 * cvStripe;
    totalEmissiveRadiance *= 1.0 - 0.7 * cvStripe;
  }`,
      );
    }
    if (patch.cutaway) {
      // The heart sits at the world origin; cut a cylinder from the camera up to it.
      frag = frag.replace(
        "void main() {",
        `void main() {
  if (uCutOn > 0.5) {
    vec3 axis = -cameraPosition; float len = length(axis); axis /= len;
    vec3 rel = vCvWorld - cameraPosition; float t = dot(rel, axis);
    if (t < len && length(rel - axis * t) < uCutRadius) discard;
  }`,
      );
    }
    if (patch.bump) {
      frag = frag.replace(
        "#include <normal_fragment_maps>",
        `#include <normal_fragment_maps>
  {
    vec3 q = vCvObj * uBumpFreq;
    float h = cvSnoise(q) * 0.6 + cvSnoise(q * 2.7 + 11.0) * 0.3 + cvSnoise(vec3(q.x * 0.6, q.y * 5.0, q.z * 0.6) + 3.0) * 0.25;
    normal = cvPerturb(-vViewPosition, normal, vec2(dFdx(h), dFdy(h)) * uBumpStrength, faceDirection);
  }`,
      );
    }
    if (patch.fresnel) {
      frag = frag.replace(
        "#include <opaque_fragment>",
        `{
    // Clamp: float error can push |n.v| just above 1, and pow() of a negative base is NaN (which the
    // bloom pass would then smear across the whole frame).
    float f = pow(clamp(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), 0.0, 1.0), ${patch.fresnel.power.toFixed(2)});
    diffuseColor.a *= mix(${patch.fresnel.min.toFixed(3)}, 1.0, f);
  }
  #include <opaque_fragment>`,
      );
    }
    shader.fragmentShader = frag;
  };
  return material;
}
