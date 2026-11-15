# TRELLIS.2 A40 review snapshot — GoAido

Temporary **public, sanitized review snapshot** of the GoAido TRELLIS.2 four-view vehicle worker.

## Canonical source

- Private source repository: `JGBDesign69/trellis2-car-multiview`
- Canonical source commit: `c666b092`
- Snapshot date: 2026-11-15
- This repository is for external technical review only.
- No credentials, source vehicle photos, signed URLs, private storage objects, or user data are included.

The files under `serverless/` are copied from commit `c666b092` without functional changes.

## Runtime/build proven on RunPod

- GPU: NVIDIA A40, compute capability 8.6
- CUDA base: 12.8.1-devel
- Python 3.12
- PyTorch 2.9.1 + cu128
- xFormers 0.0.33.post2
- ComfyUI pin: `63e5bf594339f676b4b15e072529873e3448409e`
- ComfyUI-Trellis2 pin: `b0e84c0f29857a444e8bcb5b9be2ea218e96efd0`

The previous A40 failure was:

`CUDA error: no kernel image is available for execution on the device`

The current image uses the upstream Torch270 Linux wheels for Ampere-compatible native extensions and builds nvdiffrast from pinned source. The startup gate executes real xFormers dense/sparse attention and `flex_gemm.ops.spconv.sparse_submanifold_conv3d` on the A40 before accepting jobs.

## Successful Audi run

RunPod task:

`ea400bc8-fbfb-4dc4-8e93-2a9eeab8f705-e1`

The end-to-end run completed successfully through sparse structure, shape SLat, HR/cascade, texture SLat, mesh reconstruction, dual contouring, triangulation, simplification, xatlas, GLB export and private upload.

Final artifact statistics:

- GLB: 18,435,780 bytes
- vertices: 284,534
- indices: 995,937
- triangles: 331,979
- PBR materials: 1
- embedded textures: 2
- embedded PNG dimensions: 2048 × 2048
- attributes: `POSITION`, `TEXCOORD_0`
- baseColorTexture: texture 0
- metallicRoughnessTexture: texture 1
- baseColorFactor: `[1,1,1,1]`
- metallicFactor: `1`
- roughnessFactor: `1`
- alphaMode: `OPAQUE`

Base-color statistics on active RGB texels:

- mean RGB ≈ 116.08 / 115.42 / 115.27
- mean saturation ≈ 0.0572
- mean chroma ≈ 5.15

The generated GLB itself is **not included** because it was derived from private customer/user-provided vehicle photos.

## Visual result

The desktop result is recognisable as an Audi A3 Sportback and has reasonable global proportions, but is not production-ready.

Observed remaining defects include:

- deformed grille, headlights, front bumper and lower lip
- wavy hood/fender transitions
- messy windshield/A-pillar/roof edges
- distorted rear roof/spoiler/rear-window region
- imprecise rear bumper/tailgate
- noisy wheels/tyres
- smeared window/material detail
- weak fine panel/detail fidelity

The same GLB looked almost white in Android `model-viewer`, while desktop rendering showed substantially more material/color contrast. Mobile material rendering therefore needs separate investigation.

## Preprocessing detail

The worker removes the background with rembg/u2net and then frames the alpha cutout.

`CUTOUT_SIZE=1024`
`CUTOUT_INNER=896`

The actual scale is:

```python
scale = min(CUTOUT_INNER / crop.width, CUTOUT_INNER / crop.height)
```

The crop preserves aspect ratio and is centered on a transparent 1024×1024 canvas.

Important: the recorded `foreground_ratio` and `bbox_ratio` below are measured **after** this framing:

- front: foreground 0.5346, bbox 0.6425
- rear: foreground 0.4684, bbox 0.5530
- left: foreground 0.1530, bbox 0.2082
- right: foreground 0.1518, bbox 0.2082

Do not infer lower linear side-view resolution from bbox area alone; the side view is naturally more elongated.

## Current questions under review

See [REVIEW_CONTEXT.md](./REVIEW_CONTEXT.md) for the exact review questions and proposed next experiments.

## Sanitization

Before publication, the selected snapshot files were scanned for common credential patterns. No real credentials were found. The only token-looking string is a deliberately fake `signed.example` token inside a unit test.
