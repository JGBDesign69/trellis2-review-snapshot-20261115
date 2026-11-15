import importlib.util,io,json,pathlib,sys,types,unittest
from unittest.mock import patch
from PIL import Image
fake_runpod=types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda *_:None))
fake_websocket=types.SimpleNamespace(create_connection=lambda *_a,**_k:None,WebSocketTimeoutException=TimeoutError)
sys.modules.setdefault("runpod",fake_runpod)
sys.modules.setdefault("websocket",fake_websocket)
spec=importlib.util.spec_from_file_location("trellis2_handler",pathlib.Path(__file__).with_name("handler.py"))
handler=importlib.util.module_from_spec(spec); spec.loader.exec_module(handler)
VALID={"front":"https://storage.example/front.webp","rear":"https://storage.example/rear.webp","left":"https://storage.example/left.webp","right":"https://storage.example/right.webp"}
FULL_CROP={"x":0.0,"y":0.0,"width":1.0,"height":1.0}

class ContractTests(unittest.TestCase):
 def test_requires_exactly_four_named_views(self):
  with self.assertRaisesRegex(ValueError,"Exactly four named views"): handler.handler({"input":{"views":{"front":VALID["front"]}}})
 def test_rejects_extra_view(self):
  views=dict(VALID); views["front_three_quarter"]="https://storage.example/front-quarter.webp"
  with self.assertRaisesRegex(ValueError,"Exactly four named views"): handler.handler({"input":{"views":views}})
 def test_requires_signed_artifact_upload_target(self):
  with self.assertRaisesRegex(ValueError,"Signed artifact upload target"):
   handler.handler({"input":{"views":VALID}})
 def test_download_rejects_http_before_network(self):
  with self.assertRaisesRegex(ValueError,"HTTPS"): handler._download_image("http://storage.example/front.webp")
 def test_download_rejects_unapproved_host_before_network(self):
  with patch.dict(handler.os.environ,{"ALLOWED_IMAGE_HOSTS":"storage.example"},clear=False):
   with self.assertRaisesRegex(ValueError,"ALLOWED_IMAGE_HOSTS"): handler._download_image("https://evil.example/front.webp")
 def test_ui_workflow_wires_all_four_multiview_inputs(self):
  self.assertEqual(handler.VIEWS,("front","rear","left","right"))
  w=json.loads(pathlib.Path(__file__).with_name("workflow_ui.json").read_text())
  load=next(n for n in w["nodes"] if n["id"]==1)
  self.assertEqual(load["widgets_values"][1],"xformers")
  self.assertEqual(load["widgets_values"][6],"xformers")
  self.assertNotIn("flash_attn",json.dumps(w))
  for node in w["nodes"]:
   props=node.get("properties",{})
   if props.get("aux_id")=="visualbruno/ComfyUI-Trellis2":
    self.assertEqual(props.get("ver"),"b0e84c0f29857a444e8bcb5b9be2ea218e96efd0")
  mv=next(n for n in w["nodes"] if n["id"]==6)
  links={x["name"]:x.get("link") for x in mv["inputs"]}
  for name in ("front_image","back_image","left_image","right_image"): self.assertIsNotNone(links[name])
  sparse=next(n for n in w["nodes"] if n["id"]==21)
  cascade=next(n for n in w["nodes"] if n["id"]==20)
  self.assertEqual(sparse["widgets_values"][7],64)
  self.assertEqual(cascade["widgets_values"][1],1536)
  self.assertEqual(cascade["widgets_values"][2],64)
  self.assertTrue(any(n.get("type")=="Trellis2ExportMesh" and n.get("widgets_values",[])[-1]=="glb" for n in w["nodes"]))
 def test_existing_vehicle_alpha_is_preserved_and_framed(self):
  source=io.BytesIO(); image=Image.new("RGBA",(320,240),(0,0,0,0))
  image.paste((12,34,56,255),(60,80,260,190)); image.save(source,format="PNG")
  normalized,mime,metrics=handler._prepare_vehicle_rgba(source.getvalue(),FULL_CROP)
  self.assertEqual(mime,"image/png"); self.assertEqual(metrics["source"],"provided_alpha"); self.assertTrue(metrics["crop_applied"])
  with Image.open(io.BytesIO(normalized)) as output:
   self.assertEqual(output.mode,"RGBA"); self.assertEqual(output.size,(1024,1024))
   self.assertLess(output.getchannel("A").getextrema()[0],255)

 def test_opaque_phone_photo_requires_background_removal(self):
  source=io.BytesIO(); Image.new("RGB",(320,240),(12,34,56)).save(source,format="WEBP")
  cutout=Image.new("RGBA",(320,240),(0,0,0,0)); cutout.paste((12,34,56,255),(55,75,265,195))
  with patch.object(handler,"_remove_background",return_value=cutout):
   normalized,mime,metrics=handler._prepare_vehicle_rgba(source.getvalue(),FULL_CROP)
  self.assertEqual(mime,"image/png"); self.assertEqual(metrics["source"],"rembg")
  with Image.open(io.BytesIO(normalized)) as output:
   self.assertEqual(output.size,(1024,1024))
   self.assertLess(output.getchannel("A").getextrema()[0],255)

 def test_fully_opaque_segmentation_is_rejected_before_trellis(self):
  source=io.BytesIO(); opaque=Image.new("RGB",(320,240),(12,34,56)); opaque.save(source,format="PNG")
  with patch.object(handler,"_remove_background",return_value=opaque.convert("RGBA")):
   with self.assertRaisesRegex(ValueError,"segmentation failed quality checks"):
    handler._prepare_vehicle_rgba(source.getvalue(),FULL_CROP)

 def test_scene_leak_into_corners_is_rejected_before_trellis(self):
  source=io.BytesIO(); Image.new("RGB",(320,240),(12,34,56)).save(source,format="PNG")
  leaked=Image.new("RGBA",(320,240),(0,0,0,0))
  leaked.paste((12,34,56,255),(0,0,320,40))
  leaked.paste((12,34,56,255),(55,70,265,195))
  with patch.object(handler,"_remove_background",return_value=leaked):
   with self.assertRaisesRegex(ValueError,"segmentation failed quality checks"):
    handler._prepare_vehicle_rgba(source.getvalue(),FULL_CROP)

 def test_crop_is_applied_before_resize_and_segmentation(self):
  source=io.BytesIO(); image=Image.new("RGB",(1600,1200),(20,30,40))
  image.paste((220,20,20),(400,300,1200,900)); image.save(source,format="JPEG",quality=95)
  cropped=handler._decode_rgba(source.getvalue(),{"x":0.25,"y":0.25,"width":0.5,"height":0.5})
  self.assertEqual(cropped.size,(800,600))
  self.assertGreater(cropped.getpixel((400,300))[0],100)
  with self.assertRaisesRegex(ValueError,"outside accepted bounds"):
   handler._decode_rgba(source.getvalue(),{"x":0.9,"y":0.1,"width":0.4,"height":0.4})
  with self.assertRaisesRegex(ValueError,"resolution is too low"):
   handler._decode_rgba(source.getvalue(),{"x":0.1,"y":0.1,"width":0.15,"height":0.15})

 def test_unapproved_background_model_fails_closed(self):
  previous=handler.REMBG_MODEL
  handler.REMBG_MODEL="birefnet-general-lite"
  handler._REMBG_SESSION=None
  try:
   with self.assertRaisesRegex(ValueError,"not approved"):
    handler._rembg_session()
  finally:
   handler.REMBG_MODEL=previous
   handler._REMBG_SESSION=None
 def test_comfy_failure_keeps_node_and_message_but_redacts_urls_and_long_tokens(self):
  record={"status":{"messages":[["execution_error",{"node_id":"21","node_type":"Trellis2Sampler","exception_type":"RuntimeError","exception_message":"CUDA out of memory while reading https://signed.example/private.png?token=abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"}]]}}
  summary=handler._comfy_failure(record)
  self.assertIn("Trellis2Sampler#21",summary); self.assertIn("CUDA out of memory",summary)
  self.assertNotIn("signed.example",summary); self.assertNotIn("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",summary)
 def test_extract_glb_returns_binary_artifact_reference(self):
  payload=json.dumps({"asset":{"version":"2.0"}}).encode()
  payload+=b" "*((-len(payload))%4)
  blob=b"glTF"+(2).to_bytes(4,"little")+(20+len(payload)).to_bytes(4,"little")+len(payload).to_bytes(4,"little")+b"JSON"+payload
  outputs={"17":{"3d":[{"filename":"MV_00001_.glb","subfolder":"","type":"output"}]}}
  with patch.object(handler,"_request_bytes",return_value=blob):
   content,result=handler._extract_glb(outputs)
  self.assertEqual(content,blob); self.assertEqual(result["filename"],"MV_00001_.glb")
  self.assertNotIn("base64",result)

 def test_glb_embeds_preprocessing_attestation_without_changing_payload_chunks(self):
  gltf={"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}
  payload=json.dumps(gltf,separators=(",",":")).encode(); payload+=b" "*((-len(payload))%4)
  bin_chunk=b"\x04\x00\x00\x00BIN\x00test"
  blob=b"glTF"+(2).to_bytes(4,"little")+(20+len(payload)+len(bin_chunk)).to_bytes(4,"little")+len(payload).to_bytes(4,"little")+b"JSON"+payload+bin_chunk
  views={name:{"source":"rembg","transparent_ratio":0.6,"foreground_ratio":0.4,"bbox_ratio":0.5,"corner_transparent_ratio":1.0} for name in handler.VIEWS}
  rebuilt=handler._embed_preprocessing_attestation(blob,views)
  self.assertEqual(rebuilt[:4],b"glTF")
  self.assertEqual(int.from_bytes(rebuilt[8:12],"little"),len(rebuilt))
  size=int.from_bytes(rebuilt[12:16],"little")
  parsed=json.loads(rebuilt[20:20+size].decode().rstrip())
  self.assertEqual(parsed["asset"]["extras"]["goaido"]["preprocessing_version"],1)
  self.assertEqual(parsed["asset"]["extras"]["goaido"]["views"],views)
  self.assertEqual(rebuilt[20+size:],bin_chunk)
 def test_signed_glb_upload_rejects_unapproved_host_before_network(self):
  blob=b"glTF"+b"\x02\x00\x00\x00"+b"\x0c\x00\x00\x00"
  with patch.dict(handler.os.environ,{"ALLOWED_OUTPUT_HOSTS":"storage.example"},clear=False):
   with self.assertRaisesRegex(ValueError,"not trusted"):
    handler._upload_glb("https://evil.example/storage/v1/object/upload/sign/vehicle-3d-private/pending/a.glb?token=x","pending/a.glb",blob)
 def test_worker_image_pins_binary_xformers_runtime(self):
  root=pathlib.Path(__file__).parent
  docker=(root/"Dockerfile").read_text()
  constraints=(root/"torch-constraints.txt").read_text()
  native=(root/"verify-native-deps.py").read_text()
  startup=(root/"start.sh").read_text()
  probe=(root/"verify-gpu-runtime.py").read_text()
  workflow_probe=(root/"verify-comfy-workflow.py").read_text()
  self.assertIn("nvidia/cuda:12.8.1-devel-ubuntu24.04",docker)
  self.assertIn("torch==2.9.1",docker)
  self.assertIn("https://download.pytorch.org/whl/cu128",docker)
  self.assertNotIn("https://download.pytorch.org/whl/cu130",docker)
  self.assertIn("XFORMERS_VERSION=0.0.33.post2",docker)
  self.assertIn("--only-binary=:all: --no-deps",docker)
  self.assertIn("ATTN_BACKEND=xformers",docker)
  self.assertIn("SPARSE_ATTN_BACKEND=xformers",docker)
  self.assertIn("COMFYUI_REF=63e5bf594339f676b4b15e072529873e3448409e",docker)
  self.assertIn("TRELLIS_NODES_REF=b0e84c0f29857a444e8bcb5b9be2ea218e96efd0",docker)
  self.assertIn("wheels/Linux/Torch270/cumesh",docker)
  self.assertIn("wheels/Linux/Torch270/flex_gemm",docker)
  self.assertIn("wheels/Linux/Torch270/o_voxel",docker)
  self.assertNotIn("wheels/Linux/Torch291/flex_gemm",docker)
  self.assertIn("NVDIFFRAST_REF=253ac4fcea7de5f396371124af597e6cc957bfae",docker)
  self.assertIn("nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@",docker)
  self.assertNotIn("NVDIFFREC_REF=",docker)
  self.assertNotIn("/tmp/nvdiffrec",docker)
  self.assertNotIn("flash-attn",docker)
  self.assertEqual(docker.count("text = trellis.read_text()"),1)
  self.assertIn("torch==2.9.1",constraints)
  self.assertIn("xformers==0.0.33.post2",constraints)
  self.assertIn('importlib.import_module(name)',native)
  self.assertIn('EXPECTED_CUDA_PREFIX = "12.8"',native)
  self.assertIn("verify-gpu-runtime.py",startup)
  self.assertIn("verify-comfy-workflow.py",startup)
  self.assertIn('MIN_CUDA = Version("12.8")',probe)
  self.assertIn('Version("13.0")',probe)
  self.assertIn('EXPECTED_XFORMERS = Version("0.0.33.post2")',probe)
  self.assertIn("memory_efficient_attention",probe)
  self.assertIn("BlockDiagonalMask",probe)
  self.assertIn("sparse_submanifold_conv3d",probe)
  self.assertIn("MASKED_IMPLICIT_GEMM_SPLITK",probe)
  self.assertIn("FLEX_GEMM_A40_PROBE_FAILED",probe)
  self.assertNotIn("nvdiffrec_render.light",probe)
  self.assertIn('"Trellis2ExportMesh"',workflow_probe)
  self.assertIn('get("file_format") != "glb"',workflow_probe)
  self.assertIn('"Trellis2ImageCondMultiViewGenerator"',workflow_probe)

 def test_worker_pins_and_precaches_background_removal_model(self):
  root=pathlib.Path(__file__).parent
  docker=(root/"Dockerfile").read_text()
  requirements=(root/"requirements.txt").read_text()
  self.assertIn("rembg[cpu]==2.0.86",requirements)
  self.assertIn("REMBG_MODEL=u2net",docker)
  self.assertIn("new_session('u2net')",docker)

 def test_progress_watchdog_defaults_to_five_minutes_for_inactive_prompts(self):
  root=pathlib.Path(__file__).parent
  docker=(root/"Dockerfile").read_text()
  env_example=(root/".env.example").read_text()
  requirements=(root/"requirements.txt").read_text()
  source=(root/"handler.py").read_text()
  self.assertIn("websocket-client>=1.8,<2",requirements)
  self.assertIn("STALL_TIMEOUT_SECONDS=300",docker)
  self.assertIn("STALL_TIMEOUT_SECONDS=300",env_example)
  self.assertIn('os.environ.get("STALL_TIMEOUT_SECONDS","300")',source)
  self.assertIn("ComfyUI job stalled for ",source)
  self.assertIn("_prompt_is_active(prompt_id)",source)

 def test_progress_event_must_belong_to_current_prompt(self):
  self.assertTrue(handler._is_progress_event({"type":"progress","data":{"prompt_id":"abc","value":2,"max":12}},"abc"))
  self.assertTrue(handler._is_progress_event({"type":"executing","data":{"prompt_id":"abc","node":"21"}},"abc"))
  self.assertFalse(handler._is_progress_event({"type":"progress","data":{"prompt_id":"other","value":2,"max":12}},"abc"))
  self.assertFalse(handler._is_progress_event({"type":"status","data":{"prompt_id":"abc"}},"abc"))

 def test_queue_liveness_accepts_running_and_pending_prompt(self):
  self.assertTrue(handler._queue_contains_prompt({"queue_running":[[1,"abc",{},{}]],"queue_pending":[]},"abc"))
  self.assertTrue(handler._queue_contains_prompt({"queue_running":[],"queue_pending":[{"prompt_id":"abc"}]},"abc"))
  self.assertFalse(handler._queue_contains_prompt({"queue_running":[[1,"other",{},{}]],"queue_pending":[]},"abc"))

 def test_prompt_liveness_returns_none_when_queue_probe_fails(self):
  with patch.object(handler,"_request",side_effect=OSError("local ComfyUI unavailable")):
   self.assertIsNone(handler._prompt_is_active("abc"))

 def test_stall_interrupts_running_and_queued_prompt(self):
  calls=[]
  def fake_request(path,payload=None,**kwargs):
   calls.append((path,payload,kwargs))
   return {}
  with patch.object(handler,"_request",side_effect=fake_request):
   handler._interrupt_prompt("prompt-123")
  self.assertEqual(calls[0][0],"/interrupt")
  self.assertEqual(calls[1][0],"/queue")
  self.assertEqual(calls[1][1],{"delete":["prompt-123"]})
  self.assertEqual(calls[0][2].get("timeout"),10)
  self.assertEqual(calls[1][2].get("timeout"),10)

if __name__=="__main__": unittest.main()
