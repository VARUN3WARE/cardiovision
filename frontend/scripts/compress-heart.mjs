// Shrink the Realistic heart (public/models/heart.glb) for a faster first load: re-encode its PNG
// textures as WebP (EXT_texture_webp; supported by three.js and every current browser), base colour at
// 2048 px and the normal / roughness maps at 1024 px. Geometry is untouched.
//
//   node scripts/compress-heart.mjs [input.glb] [output.glb]
//
// Image work is done by Python Pillow (python3 -m pip install pillow).
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS, EXTTextureWebP } from "@gltf-transform/extensions";

const [input = "public/models/heart.glb", output = input] = process.argv.slice(2);
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(input);
const dir = mkdtempSync(join(tmpdir(), "heart-tex-"));

const PIL = `
import sys
from PIL import Image
src, dst, size, quality = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
im = Image.open(src)
im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
if max(im.size) > size:
    im = im.resize((size, size), Image.LANCZOS)
im.save(dst, "WEBP", quality=quality, method=6)
`;

for (const [i, texture] of doc.getRoot().listTextures().entries()) {
  const slots = doc.getGraph().listParentEdges(texture).map((e) => e.getName());
  const isBase = slots.includes("baseColorTexture");
  const isNormal = slots.includes("normalTexture");
  const size = isBase ? 2048 : 1024;
  const quality = isNormal ? 95 : 90; // normals band visibly at lower quality
  const src = join(dir, `${i}.png`);
  const dst = join(dir, `${i}.webp`);
  writeFileSync(src, texture.getImage());
  execFileSync("python3", ["-c", PIL, src, dst, String(size), String(quality)]);
  const before = texture.getImage().byteLength;
  texture.setImage(readFileSync(dst)).setMimeType("image/webp").setURI("");
  console.log(`${slots.filter((s) => s !== "textures").join(",")}: ${(before / 1e6).toFixed(2)} MB -> ${(texture.getImage().byteLength / 1e6).toFixed(2)} MB (${size} px)`);
}
doc.createExtension(EXTTextureWebP).setRequired(true);
await io.write(output, doc);
console.log(`${output}: ${(statSync(output).size / 1e6).toFixed(2)} MB`);
