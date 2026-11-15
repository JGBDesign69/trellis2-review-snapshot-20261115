# RunPod build candidate: ComfyUI + visualbruno TRELLIS.2
# Build from repository root: docker build -f serverless/Dockerfile .
# This image is not yet GPU-validated. Native wheel compatibility is checked at build.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6" \
    COMFY_URL=http://127.0.0.1:8188 HF_HOME=/opt/huggingface \
    ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates build-essential ninja-build cmake \
    libgl1 libglib2.0-0 libgomp1 libegl1 libopengl0 libeigen3-dev \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
# Ubuntu 24.04 provides Python 3.12; keep the interpreter explicit for wheel compatibility.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CuMesh cubvh expects <Eigen/Dense> on its include path.
ENV CPLUS_INCLUDE_PATH=/usr/include/eigen3
ENV CPATH=/usr/include/eigen3
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
# TRELLIS.2 supports xFormers for dense, sparse and windowed sparse attention.
# Use the official PyTorch 2.9.1/cu128 binary wheel and fail closed if a wheel is
# unavailable; hosted builds must never fall back to a native attention compile.
ARG XFORMERS_VERSION=0.0.33.post2
RUN python -m pip install --only-binary=:all: --no-deps \
    "xformers==${XFORMERS_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128
WORKDIR /opt
# Immutable upstream pins: these are the revisions audited against the GoAido workflow.
ARG COMFYUI_REF=63e5bf594339f676b4b15e072529873e3448409e
ARG TRELLIS_NODES_REF=b0e84c0f29857a444e8bcb5b9be2ea218e96efd0
ARG WORKFLOW_CONVERTER_REF=bc8538278f82053b3ca10a44d62d02596f8e1a37
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout "$COMFYUI_REF" \
    && git clone https://github.com/visualbruno/ComfyUI-Trellis2.git /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2 \
    && git -C /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2 checkout "$TRELLIS_NODES_REF" \
    && git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git /opt/ComfyUI/custom_nodes/workflow-to-api-converter \
    && git -C /opt/ComfyUI/custom_nodes/workflow-to-api-converter checkout "$WORKFLOW_CONVERTER_REF"
COPY serverless/torch-constraints.txt /opt/torch-constraints.txt
ENV PIP_CONSTRAINT=/opt/torch-constraints.txt
RUN python -m pip install -r /opt/ComfyUI/requirements.txt \
    && python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/requirements.txt
# Keep current upstream runtime semantics while removing two known-noise warnings:
# - huggingface_hub removed local_dir_use_symlinks
# - this serverless worker intentionally uses ComfyUI's single default user
RUN python - <<'PY'
from pathlib import Path
import re
trellis = Path("/opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/trellis2/pipelines/trellis2_image_to_3d.py")
text = trellis.read_text()
text, count = re.subn(r"^\s*local_dir_use_symlinks=False,\s*$\n?", "", text, flags=re.MULTILINE)
if count < 1:
    raise SystemExit("Expected deprecated local_dir_use_symlinks arguments were not found")
# mmgp is optional in this serverless path; upstream explicitly states that no patch
# is needed when it is absent. Remove only the noisy informational print.
text = text.replace("            print('mmgp not installed')\n", "")
trellis.write_text(text)

user_manager = Path("/opt/ComfyUI/app/user_manager.py")
text = user_manager.read_text()
for message in (
    "****** User settings have been changed to be stored on the server instead of browser storage. ******",
    "****** For multi-user setups add the --multi-user CLI argument to enable multiple user profiles. ******",
):
    old = f'logging.warning("{message}")'
    new = f'logging.info("{message}")'
    if old not in text:
        raise SystemExit(f"Expected ComfyUI startup warning not found: {message}")
    text = text.replace(old, new)
user_manager.write_text(text)
PY
# A40 is Ampere compute capability 8.6. The upstream Torch291 Linux wheels ship
# sm_120-only CUDA cubins and fail on Ampere at runtime. The upstream Torch270
# cp312 wheels are the tested Linux/Ampere path (sm_80/86/89/90) and are ABI-
# compatible with torch 2.9.x for these extensions.
#
# nvdiffrast is the exception: its Torch270 wheel carries a torch-2.7 ABI, so
# build nvdiffrast from pinned NVlabs source against the installed torch 2.9.1.
# The Audi workflow does not use nvdiffrec_render; do not add that unrelated
# native extension to the image or startup gate.
ENV TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4
RUN python -m pip install --force-reinstall --no-deps \
    /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch270/cumesh-1.0-cp312-cp312-linux_x86_64.whl \
    /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch270/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl \
    /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch270/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl \
    /opt/ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch270/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl
ARG NVDIFFRAST_REF=253ac4fcea7de5f396371124af597e6cc957bfae
RUN python -m pip install --force-reinstall --no-build-isolation --no-deps \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@${NVDIFFRAST_REF}"
# Install lightweight runtime helpers needed by native module imports before
# the build-time ABI probe. Driver-dependent modules are checked on the A40.
RUN python -m pip install easydict plyfile trimesh zstandard
COPY serverless/verify-native-deps.py /opt/verify-native-deps.py
RUN python /opt/verify-native-deps.py
RUN python -m pip check
WORKDIR /opt/trellis-worker
COPY serverless/requirements.txt ./requirements.txt
ENV REMBG_HOME=/opt/rembg REMBG_MODEL=u2net
RUN python -m pip install -r requirements.txt \
    && python -c "from rembg import new_session; new_session('u2net')"
COPY serverless/handler.py ./handler.py
COPY serverless/workflow_ui.json ./workflow_ui.json
COPY serverless/verify-gpu-runtime.py ./verify-gpu-runtime.py
COPY serverless/verify-comfy-workflow.py ./verify-comfy-workflow.py
COPY serverless/start.sh ./start.sh
RUN chmod +x ./start.sh
ENV COMFY_WORKFLOW_PATH=/opt/trellis-worker/workflow_ui.json \
    ALLOWED_IMAGE_HOSTS=zfunmsfdvgooblwfspxy.supabase.co \
    ALLOWED_OUTPUT_HOSTS=zfunmsfdvgooblwfspxy.supabase.co \
    STALL_TIMEOUT_SECONDS=300
CMD ["/opt/trellis-worker/start.sh"]
