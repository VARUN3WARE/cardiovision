/**
 * Shader patches for the atlas meshes (which have no UVs):
 *  - bump: procedural tissue relief from 3D noise in object space (derivative bump mapping);
 *  - cutaway: discards whatever lies between the camera and the heart within a radius, so the
 *    chest wall opens a window onto the heart from any viewing angle;
 *  - fresnel: alpha rises towards silhouettes (ghosted skin).
 */
import * as THREE from "three";
import { ATLAS_SCALE } from "./atlasAnatomy";

/** Shared by every patched material, so one update moves every cutaway. Radius in world units. */
export const CUTAWAY = {
  uCutRadius: { value: 0.085 * ATLAS_SCALE }, // the heart's silhouette radius plus a margin
  uCutOn: { value: 1 },
};

interface Patch {
  bump?: { freq: number; strength: number };
  cutaway?: boolean;
  fresnel?: { min: number; power: number };
}

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
  material.onBeforeCompile = (shader) => {
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
    let frag = shader.fragmentShader.replace(
      "#include <common>",
      `#include <common>
varying vec3 vCvObj;
varying vec3 vCvWorld;
uniform float uCutRadius;
uniform float uCutOn;
uniform float uBumpFreq;
uniform float uBumpStrength;
${patch.bump ? NOISE : ""}`,
    );
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
    float f = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), ${patch.fresnel.power.toFixed(2)});
    diffuseColor.a *= mix(${patch.fresnel.min.toFixed(3)}, 1.0, f);
  }
  #include <opaque_fragment>`,
      );
    }
    shader.fragmentShader = frag;
  };
  return material;
}
