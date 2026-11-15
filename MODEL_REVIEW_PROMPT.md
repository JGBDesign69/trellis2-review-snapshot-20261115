# Independent review prompt

Review this public repository as a senior specialist in TRELLIS.2, ComfyUI, multi-view 3D reconstruction, PyTorch/CUDA and vehicle reconstruction:

https://github.com/JGBDesign69/trellis2-review-snapshot-20261115

Please **open and inspect the actual files**, especially:
- `serverless/handler.py`
- `serverless/workflow_ui.json`
- `serverless/verify-gpu-runtime.py`
- `serverless/Dockerfile`
- `REVIEW_CONTEXT.md`
- `RESULT_METADATA.json`
- `SOURCE_MANIFEST.json`

Do not rely only on generic TRELLIS knowledge when the relevant implementation is present in the repository.

The A40/CUDA runtime issue is already solved and one full Audi GLB generation completed. The current problem is visual quality: recognisable/global shape is substantially improved, but local vehicle geometry, fine detail, texture fidelity and mobile material rendering are not production-ready.

Critically evaluate the proposal in `REVIEW_CONTEXT.md`. In particular:
1. identify the actual highest-probability geometry bottleneck(s);
2. identify the actual texture/material bottleneck(s);
3. verify the exact semantics of multi-view blending and `dino_lock` from the pinned code;
4. decide whether MeshRefiner → MultiViewTexturing is the best next experiment;
5. decide whether front+rear or all four views should be used next;
6. decide whether sparse resolution 64, post-processing, mask quality, camera pose or the model prior is the dominant limitation;
7. evaluate whether Pixal3D should move earlier;
8. propose exactly one next paid A40 experiment with maximum information gain and minimal simultaneous variable changes;
9. specify zero-GPU checks that should be completed first;
10. define objective evidence required before calling this vehicle generator production-ready.

Return:
- what is proven;
- what is disproven;
- what is still hypothesis;
- GO / NO-GO on the current proposed improvement;
- exact next experiment and parameters;
- what must not be changed in that experiment.

If you cannot access a file, say exactly which file rather than filling the gap with assumptions.
