# 3D model attribution

## heart.glb (Realistic heart)

"Realistic Human Heart" by neshallads
<https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089>,
licensed under Creative Commons Attribution 4.0 (CC BY 4.0): <https://creativecommons.org/licenses/by/4.0/>

Downloaded from Sketchfab as GLB (2k textures). Used unmodified; CardioVision bakes the node transforms, fits its
own risk-coloured coronary tree and perfusion territories onto the surface at runtime
(`frontend/src/components/specimenAnatomy.ts`), and overrides metalness to 0. If this file is missing, the app
falls back to its procedural (schematic) heart.

## bp3d_heart.glb and bp3d_thorax.glb (Atlas heart and chest)

**BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.**
Licence page: <https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html> (CC BY 4.0, updated 2025-02-27).
Source: BodyParts3D v4.0, IS-A tree, 99% polygon reduction
(<https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip>).
Reference: Mitsuhashi N, et al. BodyParts3D: 3D structure database for anatomical concepts.
*Nucleic Acids Res.* 2009;37:D782–5 (PMID 18835852).

The original 2013 OBJ headers name CC BY-SA 2.1 Japan. So that the files satisfy either licence, these derived
GLB files are distributed under **CC BY-SA 4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>). The app's
source code is not affected.

**Changes made** (`scripts/bp3d/fetch_bp3d.py`, `scripts/bp3d/parts.json`, `frontend/scripts/build-bp3d.mjs`):
181 element files were extracted from the archive (heart: 126, chest: 55); coordinates converted from millimetres
(x, y, z) to glTF metres (x, z, −y); vertices welded; loose fragments under 2% of an element removed; meshes
simplified with meshoptimizer; elements merged into one mesh per part with a per-vertex branch index; skin cropped
to the chest (z 1000–1450 mm); ventricular wall annotated with its distance to each coronary tree and to the groove
vessels; meshopt-compressed. Every element keeps its BodyParts3D file id (FJ…) and FMA concept id in the GLB
extras. The full list is in `scripts/bp3d/parts.json`.
