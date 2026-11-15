"""Fail fast when CPU-loadable TRELLIS native modules cannot actually import during image build."""
import importlib
import sys
from packaging.version import Version
import torch

EXPECTED_TORCH = Version("2.9.1")
EXPECTED_CUDA_PREFIX = "12.8"

torch_version = Version(torch.__version__.split("+", 1)[0])
if torch_version != EXPECTED_TORCH:
    sys.exit(f"NATIVE_RUNTIME_VERSION_MISMATCH: expected torch {EXPECTED_TORCH}, got {torch.__version__}")
if not torch.version.cuda or not str(torch.version.cuda).startswith(EXPECTED_CUDA_PREFIX):
    sys.exit(
        f"NATIVE_RUNTIME_VERSION_MISMATCH: expected torch CUDA {EXPECTED_CUDA_PREFIX}.x, "
        f"got {torch.version.cuda!r}"
    )

# These imports do not require an NVIDIA driver, but they do load the native
# libraries. In particular cumesh catches a CUDA-major mismatch such as a wheel
# linked to libcudart.so.12 inside a CUDA 13-only image.
imports = (
    "cumesh",
    "nvdiffrast.torch",
    "xformers",
)
failures = []
for name in imports:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")

if failures:
    sys.exit("TRELLIS_NATIVE_IMPORT_FAILED: " + " | ".join(failures))

print(
    "CPU-loadable TRELLIS native imports passed; "
    f"torch={torch.__version__}, torch_cuda={torch.version.cuda}. "
    "Driver-dependent o_voxel/FlexGEMM/nvdiffrec imports are checked by the A40 startup probe."
)
