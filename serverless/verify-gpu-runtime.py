"""Fail fast when the RunPod GPU/runtime cannot execute the pinned TRELLIS runtime."""
import importlib
import sys
from packaging.version import Version
import torch

EXPECTED_TORCH = Version("2.9.1")
EXPECTED_XFORMERS = Version("0.0.33.post2")
MIN_CUDA = Version("12.8")

native_imports = (
    "cumesh",
    "o_voxel",
    "nvdiffrast.torch",
    "flex_gemm.ops.grid_sample",
)
for module_name in native_imports:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        raise SystemExit(
            f"TRELLIS_NATIVE_IMPORT_FAILED: {module_name}: {type(exc).__name__}: {exc}"
        ) from exc

try:
    import xformers
    import xformers.ops as xops
except Exception as exc:
    raise SystemExit(f"XFORMERS_IMPORT_FAILED: {type(exc).__name__}: {exc}") from exc

torch_version = Version(torch.__version__.split("+", 1)[0])
if torch_version != EXPECTED_TORCH:
    sys.exit(f"GPU_RUNTIME_VERSION_MISMATCH: expected torch {EXPECTED_TORCH}, got {torch.__version__}")

xformers_version = Version(xformers.__version__.split("+", 1)[0])
if xformers_version != EXPECTED_XFORMERS:
    sys.exit(f"GPU_RUNTIME_VERSION_MISMATCH: expected xformers {EXPECTED_XFORMERS}, got {xformers.__version__}")

if not torch.version.cuda:
    sys.exit("GPU_RUNTIME_VERSION_MISMATCH: PyTorch has no CUDA runtime")
cuda_version = Version(torch.version.cuda)
if cuda_version < MIN_CUDA or cuda_version >= Version("13.0"):
    sys.exit(f"GPU_RUNTIME_VERSION_MISMATCH: expected CUDA 12.8.x runtime, got {cuda_version}")

if not torch.cuda.is_available():
    sys.exit("XFORMERS_GPU_PROBE_FAILED: CUDA is not available")

device = torch.device("cuda:0")
name = torch.cuda.get_device_name(device)
major, minor = torch.cuda.get_device_capability(device)
if (major, minor) != (8, 6):
    sys.exit(f"GPU_RUNTIME_TARGET_MISMATCH: expected A40 compute capability 8.6, got {major}.{minor} on {name}")

try:
    q = torch.randn((1, 32, 2, 64), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dense_out = xops.memory_efficient_attention(q, k, v)

    mask = xops.fmha.BlockDiagonalMask.from_seqlens([16, 16], [16, 16])
    sparse_out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
    torch.cuda.synchronize(device)
except Exception as exc:
    raise SystemExit(f"XFORMERS_GPU_PROBE_FAILED: {type(exc).__name__}: {exc}") from exc

expected = (1, 32, 2, 64)
for label, out in (("dense", dense_out), ("sparse", sparse_out)):
    if tuple(out.shape) != expected or not torch.isfinite(out).all().item():
        sys.exit(f"XFORMERS_GPU_PROBE_FAILED: invalid {label} output shape/values: {tuple(out.shape)}")

# Execute the exact native sparse-convolution kernel family used by the
# Trellis2ShapeCascadeMultiViewGenerator. Imports alone cannot detect a wheel
# whose cubins omit the A40's sm_86 target.
try:
    import flex_gemm
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d

    flex_gemm.ops.spconv.set_algorithm(
        flex_gemm.ops.spconv.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK
    )
    flex_gemm.ops.spconv.set_hashmap_ratio(2.0)

    ci = co = 8
    coords = torch.tensor(
        [[0, x, y, z] for x in range(2) for y in range(2) for z in range(2)],
        dtype=torch.int32,
        device=device,
    )
    feats = torch.randn((coords.shape[0], ci), device=device, dtype=torch.float16)
    weight = torch.randn((co, 3, 3, 3, ci), device=device, dtype=torch.float16)
    bias = torch.randn((co,), device=device, dtype=torch.float16)
    conv_out, _ = sparse_submanifold_conv3d(
        feats,
        coords,
        torch.Size([1, ci, 4, 4, 4]),
        weight,
        bias,
    )
    torch.cuda.synchronize(device)
except Exception as exc:
    raise SystemExit(f"FLEX_GEMM_A40_PROBE_FAILED: {type(exc).__name__}: {exc}") from exc

if tuple(conv_out.shape) != (coords.shape[0], co) or not torch.isfinite(conv_out).all().item():
    sys.exit(
        f"FLEX_GEMM_A40_PROBE_FAILED: invalid output shape/values: {tuple(conv_out.shape)}"
    )

print(
    f"TRELLIS native imports + xFormers GPU probe passed on {name} (CC {major}.{minor}); "
    f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, xformers={xformers.__version__}"
)
