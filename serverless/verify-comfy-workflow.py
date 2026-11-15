"""Fail closed unless ComfyUI loaded the pinned TRELLIS nodes and converts the workflow correctly."""
import json
import os
import pathlib
import sys
import urllib.request

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WORKFLOW_PATH = pathlib.Path(
    os.environ.get("COMFY_WORKFLOW_PATH", "/opt/trellis-worker/workflow_ui.json")
)

def request_json(path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"content-type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

required_classes = (
    "Trellis2LoadModel",
    "Trellis2LoadImageWithTransparency",
    "Trellis2PreProcessImage",
    "Trellis2ImageCondMultiViewGenerator",
    "Trellis2SparseMultiViewGenerator",
    "Trellis2ShapeMultiViewGenerator",
    "Trellis2ShapeCascadeMultiViewGenerator",
    "Trellis2TexSlatMultiViewGenerator",
    "Trellis2ExportMesh",
)

object_info = request_json("/object_info")
missing = [name for name in required_classes if name not in object_info]
if missing:
    sys.exit("COMFY_TRELLIS_NODE_LOAD_FAILED: missing " + ", ".join(missing))

ui = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
workflow = request_json("/workflow/convert", ui)

pre = [n for n in workflow.values() if n.get("class_type") == "Trellis2PreProcessImage"]
if len(pre) != 4:
    sys.exit(f"COMFY_WORKFLOW_CONVERSION_FAILED: expected 4 preprocessors, got {len(pre)}")

mv = [n for n in workflow.values() if n.get("class_type") == "Trellis2ImageCondMultiViewGenerator"]
if len(mv) != 1:
    sys.exit(f"COMFY_WORKFLOW_CONVERSION_FAILED: expected 1 multiview conditioning node, got {len(mv)}")
required_views = ("front_image", "back_image", "left_image", "right_image")
if not all(name in mv[0].get("inputs", {}) for name in required_views):
    sys.exit("COMFY_WORKFLOW_CONVERSION_FAILED: multiview conditioning is missing one or more views")

exporters = [n for n in workflow.values() if n.get("class_type") == "Trellis2ExportMesh"]
if len(exporters) != 1:
    sys.exit(f"COMFY_WORKFLOW_CONVERSION_FAILED: expected 1 exporter, got {len(exporters)}")
if exporters[0].get("inputs", {}).get("file_format") != "glb":
    sys.exit(
        "COMFY_WORKFLOW_CONVERSION_FAILED: exporter file_format is "
        + repr(exporters[0].get("inputs", {}).get("file_format"))
    )

print("ComfyUI TRELLIS node load and workflow conversion probe passed.")
