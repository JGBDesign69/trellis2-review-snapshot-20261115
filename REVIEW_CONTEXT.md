# Review context and questions

This repository is a **temporary public review snapshot**. The worker code under `serverless/` is copied from the canonical private commit `c666b092`.

## What is already proven

The RunPod A40 worker now boots and executes the native CUDA path on compute capability 8.6. A full four-view Audi generation completed and produced the GLB described in `RESULT_METADATA.json`.

The old blocker `CUDA error: no kernel image is available for execution on the device` is considered fixed for this build.

## Current quality status

The generated vehicle is recognisable and global proportions are reasonable, but it is **not production ready**.

Observed defects:
- front grille/headlights/bumper/lower lip are deformed;
- hood/fender transitions are wavy;
- windshield/A-pillar/roof edges are messy;
- rear roof/spoiler/rear-window region is distorted;
- rear bumper/tailgate transitions are imprecise;
- wheels/tyres are noisy;
- window/material detail is smeared;
- fine panel/detail fidelity is weak.

Android `model-viewer` displayed the same GLB almost white, while a desktop viewer showed much more material/color contrast. Treat mobile rendering as a separate issue.

## Important preprocessing fact

Please read `serverless/handler.py` before interpreting the recorded bbox ratios.

Each alpha cutout is framed with:
- `CUTOUT_SIZE=1024`
- `CUTOUT_INNER=896`
- `scale=min(896/crop.width, 896/crop.height)`

The aspect ratio is preserved and the framed cutout is centered. The result metrics in `RESULT_METADATA.json` are measured **after** this framing. A smaller side-view bbox area therefore does not by itself prove lower linear side-view resolution.

## Current workflow

Four views (front/rear/left/right) feed:
1. `Trellis2ImageCondMultiViewGenerator`
2. `Trellis2SparseMultiViewGenerator`
3. `Trellis2ShapeMultiViewGenerator`
4. `Trellis2ShapeCascadeMultiViewGenerator`
5. `Trellis2TexSlatMultiViewGenerator`
6. decode / mesh reconstruction
7. dual contouring / triangulation
8. simplify
9. xatlas unwrap/rasterizer
10. GLB export

Current important values:
- sparse structure resolution: 64
- cascade resolution: 1536
- target face count: 500,000
- current dino_lock: 0
- xFormers attention backend

## Proposed improvement under debate

The current candidate is:

```text
successful baseline mesh
→ Trellis2MeshRefiner with the same four source images
→ stronger image conditioning / DINO lock if justified by pinned source
→ Trellis2MultiViewTexturing with the original four images
→ GLB
→ fixed-view renders + quantitative comparison
```

Possible fallback/parallel candidate:
- Pixal3D multi-view route with explicit camera geometry and `framing=auto`.

This proposal is **not accepted yet**. Please challenge it.

## Questions for independent reviewers

1. Rank the likely causes of the remaining local geometry defects.
2. Rank the likely causes of the weak/near-monochrome base-color detail.
3. Is MeshRefiner → MultiViewTexturing the best next paid A40 experiment? If not, propose a better one.
4. Read the pinned source and explain exactly what `dino_lock` does in this build. Recommend whether it should remain 0 or be changed, and why.
5. Does four-view conditioning help or hurt compared with front+rear in this implementation? Base the answer on the actual multi-view sampler.
6. Is sparse resolution 64 a likely quality bottleneck for vehicle local detail? Would 96/128 be meaningful or merely interpolation/off-distribution?
7. Could simplify/reconstruct/hole-filling be destroying useful geometry? Specify the cheapest way to isolate this.
8. Should Pixal3D be tested earlier because it uses explicit projection/camera geometry, despite handheld photos not being exact orthogonal views?
9. How should generative Texture-SLat and direct MultiViewTexturing be A/B tested on the **same mesh** with minimal GPU cost?
10. Assess the one-material PBR setup (`metallicFactor=1`, `roughnessFactor=1`, metallic-roughness texture) and whether it plausibly contributes to the Android white rendering.
11. Design exactly one next A40 experiment that maximizes information per GPU-euro. State what must **not** be changed in the same experiment.
12. Define objective production-readiness evidence for vehicle reconstruction (silhouette/contours, detail, texture, repeatability, GLB validity, mobile performance, failure rate, runtime/cost).

## Review rule

Do not infer behaviour from generic TRELLIS knowledge when the relevant implementation is present in this snapshot. Cite file/function/node names when possible.
