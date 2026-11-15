"""RunPod Serverless adapter for TRELLIS.2 four-view generation."""
import hashlib, io, json, os, pathlib, re, time, urllib.parse, urllib.request, uuid
from urllib.parse import urlparse
from PIL import Image, ImageOps
import runpod
import websocket

BASE=os.environ.get("COMFY_URL","http://127.0.0.1:8188")
VIEWS=("front","rear","left","right")
DEFAULT_MAPPING={"front":"2","rear":"3","left":"27","right":"29"}
MAX_BYTES=20*1024*1024
MAX_GLB_BYTES=int(os.environ.get("MAX_GLB_RESPONSE_BYTES",str(50*1024*1024)))
REMBG_MODEL=os.environ.get("REMBG_MODEL","u2net")
ALLOWED_REMBG_MODELS={"u2net"}
CUTOUT_SIZE=int(os.environ.get("CUTOUT_SIZE","1024"))
CUTOUT_INNER=int(os.environ.get("CUTOUT_INNER","896"))
_REMBG_SESSION=None

def _request(path,payload=None,data=None,headers=None,timeout=120):
    body=data if data is not None else (json.dumps(payload).encode() if payload is not None else None)
    req=urllib.request.Request(BASE+path,data=body,headers=headers or ({"Content-Type":"application/json"} if payload is not None else {}))
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)

def _request_bytes(path):
    with urllib.request.urlopen(BASE+path,timeout=120) as response: return response.read(MAX_GLB_BYTES+1)

def _download_image(url):
    parsed=urlparse(url)
    if parsed.scheme!="https" or not parsed.hostname: raise ValueError("Image inputs must be HTTPS URLs")
    allowed=set(filter(None,os.environ.get("ALLOWED_IMAGE_HOSTS","").split(",")))
    if parsed.hostname not in allowed: raise ValueError("Image hostname is not in ALLOWED_IMAGE_HOSTS")
    req=urllib.request.Request(url,headers={"User-Agent":"trellis2-four-view/1"})
    with urllib.request.urlopen(req,timeout=60) as resp:
        if resp.url!=url: raise ValueError("Redirects are not accepted for image inputs")
        mime=resp.headers.get_content_type()
        if mime not in ("image/jpeg","image/png","image/webp"): raise ValueError("Expected JPEG, PNG or WEBP")
        content=resp.read(MAX_BYTES+1)
        if len(content)>MAX_BYTES: raise ValueError("Image too large")
        return content,mime

def _upload(content,mime,filename):
    boundary="----trellis"+uuid.uuid4().hex
    def part(name,value,filename=None,content_type=None):
        head=f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename: head+=f'; filename="{filename}"'
        head+="\r\n"
        if content_type: head+=f"Content-Type: {content_type}\r\n"
        return head.encode()+b"\r\n"+value+b"\r\n"
    data=part("image",content,filename,mime)+part("type",b"input")+f"--{boundary}--\r\n".encode()
    return _request("/upload/image",data=data,headers={"Content-Type":f"multipart/form-data; boundary={boundary}"})

def _validated_crop(value):
    if not isinstance(value,dict) or set(value)!={"x","y","width","height"}:
        raise ValueError("Every vehicle view requires an exact normalized crop")
    try:
        x=float(value["x"]); y=float(value["y"]); width=float(value["width"]); height=float(value["height"])
    except (TypeError,ValueError) as exc:
        raise ValueError("Vehicle crop coordinates are invalid") from exc
    if not all(map(lambda n:n==n and abs(n)!=float("inf"),(x,y,width,height))):
        raise ValueError("Vehicle crop coordinates are invalid")
    if x<0 or y<0 or width<=0.1 or height<=0.1 or x+width>1.000001 or y+height>1.000001:
        raise ValueError("Vehicle crop is outside accepted bounds")
    return {"x":x,"y":y,"width":width,"height":height}

def _decode_rgba(content,crop):
    try:
        with Image.open(io.BytesIO(content)) as source:
            source=ImageOps.exif_transpose(source)
            width,height=source.size
            if width<1 or height<1 or width*height>36_000_000:
                raise ValueError("Image dimensions are not accepted")
            crop=_validated_crop(crop)
            left=max(0,min(width-1,round(crop["x"]*width)))
            top=max(0,min(height-1,round(crop["y"]*height)))
            right=max(left+1,min(width,round((crop["x"]+crop["width"])*width)))
            bottom=max(top+1,min(height,round((crop["y"]+crop["height"])*height)))
            if right-left<320 or bottom-top<240:
                raise ValueError("Vehicle crop resolution is too low")
            image=source.crop((left,top,right,bottom)).convert("RGBA")
            if max(image.size)>2048:
                image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
            return image.copy()
    except (OSError,ValueError) as exc:
        if isinstance(exc,ValueError): raise
        raise ValueError("Image could not be normalized for TRELLIS") from exc

def _alpha_metrics(image):
    alpha=image.getchannel("A")
    hist=alpha.histogram()
    total=max(1,image.width*image.height)
    transparent=sum(hist[:250])/total
    foreground=sum(hist[32:])/total
    mask=alpha.point(lambda p:255 if p>=32 else 0)
    bbox=mask.getbbox()
    bbox_ratio=0.0 if not bbox else ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))/total
    corner=max(1,min(image.width,image.height)//12)
    corners=(
        alpha.crop((0,0,corner,corner)),
        alpha.crop((image.width-corner,0,image.width,corner)),
        alpha.crop((0,image.height-corner,corner,image.height)),
        alpha.crop((image.width-corner,image.height-corner,image.width,image.height)),
    )
    corner_pixels=max(1,4*corner*corner)
    corner_transparent=sum(sum(c.histogram()[:250]) for c in corners)/corner_pixels
    return {
        "transparent_ratio":transparent,
        "foreground_ratio":foreground,
        "bbox_ratio":bbox_ratio,
        "corner_transparent_ratio":corner_transparent,
        "bbox":bbox,
    }

def _valid_cutout(metrics):
    return (
        metrics["transparent_ratio"]>=0.03
        and 0.04<=metrics["foreground_ratio"]<=0.88
        and 0.06<=metrics["bbox_ratio"]<=0.94
        and metrics["corner_transparent_ratio"]>=0.85
        and metrics["bbox"] is not None
    )

def _rembg_session():
    global _REMBG_SESSION
    if REMBG_MODEL not in ALLOWED_REMBG_MODELS:
        raise ValueError("Background-removal model is not approved for this deployment")
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION=new_session(REMBG_MODEL)
    return _REMBG_SESSION

def _remove_background(image):
    from rembg import remove
    output=remove(image.convert("RGB"),session=_rembg_session())
    if isinstance(output,(bytes,bytearray)):
        with Image.open(io.BytesIO(output)) as decoded:
            return decoded.convert("RGBA")
    if isinstance(output,Image.Image):
        return output.convert("RGBA")
    raise ValueError("Background removal returned an unsupported image")

def _frame_cutout(image,metrics):
    bbox=metrics["bbox"]
    if not bbox: raise ValueError("Vehicle foreground mask is empty")
    left,top,right,bottom=bbox
    width,height=right-left,bottom-top
    margin=max(8,int(max(width,height)*0.08))
    crop=image.crop((max(0,left-margin),max(0,top-margin),min(image.width,right+margin),min(image.height,bottom+margin)))
    scale=min(CUTOUT_INNER/max(1,crop.width),CUTOUT_INNER/max(1,crop.height))
    size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))
    resized=crop.convert("RGBa").resize(size,Image.Resampling.LANCZOS).convert("RGBA")
    canvas=Image.new("RGBA",(CUTOUT_SIZE,CUTOUT_SIZE),(0,0,0,0))
    canvas.alpha_composite(resized,((CUTOUT_SIZE-size[0])//2,(CUTOUT_SIZE-size[1])//2))
    return canvas

def _prepare_vehicle_rgba(content,crop):
    image=_decode_rgba(content,crop)
    metrics=_alpha_metrics(image)
    source="provided_alpha" if _valid_cutout(metrics) else "rembg"
    if source=="rembg":
        image=_remove_background(image)
        metrics=_alpha_metrics(image)
    if not _valid_cutout(metrics):
        raise ValueError("Vehicle foreground segmentation failed quality checks")
    framed=_frame_cutout(image,metrics)
    framed_metrics=_alpha_metrics(framed)
    if not _valid_cutout(framed_metrics):
        raise ValueError("Normalized vehicle cutout failed alpha validation")
    output=io.BytesIO()
    framed.save(output,format="PNG",optimize=False)
    normalized=output.getvalue()
    if not normalized or len(normalized)>MAX_BYTES:
        raise ValueError("Normalized image is too large")
    safe_metrics={key:round(float(value),4) for key,value in framed_metrics.items() if key!="bbox"}
    return normalized,"image/png",{"source":source,"crop_applied":True,**safe_metrics}

def _load_api_workflow():
    ui=json.loads(pathlib.Path(os.environ.get("COMFY_WORKFLOW_PATH","/opt/trellis-worker/workflow_ui.json")).read_text())
    workflow=_request("/workflow/convert",payload=ui)
    mapping=json.loads(os.environ.get("COMFY_VIEW_NODE_IDS",json.dumps(DEFAULT_MAPPING)))
    if set(mapping)!=set(VIEWS): raise ValueError("COMFY_VIEW_NODE_IDS must map all four views")
    for node_id in mapping.values():
        node=workflow.get(str(node_id))
        if not node: raise ValueError(f"Missing image node {node_id} in converted workflow")
        if node.get("class_type")!="Trellis2LoadImageWithTransparency":
            raise ValueError(f"Image node {node_id} must preserve alpha transparency")
    preprocess=[n for n in workflow.values() if n.get("class_type")=="Trellis2PreProcessImage"]
    if len(preprocess)!=4:
        raise ValueError("Converted workflow must contain four TRELLIS preprocessors")
    for node in preprocess:
        node.setdefault("inputs",{})["remove_background"]=False
    mv=[n for n in workflow.values() if n.get("class_type")=="Trellis2ImageCondMultiViewGenerator"]
    if len(mv)!=1 or not all(k in mv[0].get("inputs",{}) for k in ("front_image","back_image","left_image","right_image")):
        raise ValueError("Converted workflow is not wired for four-view conditioning")
    exporters=[n for n in workflow.values() if n.get("class_type")=="Trellis2ExportMesh"]
    if len(exporters)!=1 or exporters[0].get("inputs",{}).get("file_format")!="glb":
        raise ValueError("Converted workflow must contain one GLB exporter")
    return workflow,{k:str(v) for k,v in mapping.items()}

def _extract_glb(outputs):
    for output in outputs.values():
        for item in output.get("3d",[]):
            if str(item.get("filename","")).lower().endswith(".glb"):
                q=urllib.parse.urlencode({"filename":item["filename"],"subfolder":item.get("subfolder",""),"type":item.get("type","output")})
                content=_request_bytes("/view?"+q)
                if len(content)>MAX_GLB_BYTES: raise ValueError("Generated GLB exceeds response limit")
                if len(content)<20 or content[:4]!=b"glTF": raise ValueError("Generated artifact is not a binary GLB")
                return content,{"filename":item["filename"]}
    raise RuntimeError("ComfyUI completed without a GLB artifact")

def _embed_preprocessing_attestation(content,preprocessing):
    if len(content)<20 or content[:4]!=b"glTF":
        raise ValueError("Generated artifact is not a binary GLB")
    version=int.from_bytes(content[4:8],"little")
    total=int.from_bytes(content[8:12],"little")
    json_length=int.from_bytes(content[12:16],"little")
    json_type=content[16:20]
    if version!=2 or total!=len(content) or json_type!=b"JSON" or json_length<2 or 20+json_length>len(content):
        raise ValueError("Generated GLB header is invalid")
    try:
        gltf=json.loads(content[20:20+json_length].rstrip(b" \x00").decode("utf-8"))
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise ValueError("Generated GLB JSON chunk is invalid") from exc
    if not isinstance(gltf,dict) or not isinstance(gltf.get("asset"),dict):
        raise ValueError("Generated GLB has no asset metadata")
    extras=gltf["asset"].get("extras")
    if not isinstance(extras,dict): extras={}
    extras["goaido"]={"preprocessing_version":1,"views":preprocessing}
    gltf["asset"]["extras"]=extras
    encoded=json.dumps(gltf,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    encoded+=b" "*((-len(encoded))%4)
    rest=content[20+json_length:]
    rebuilt=bytearray()
    rebuilt.extend(b"glTF")
    rebuilt.extend((2).to_bytes(4,"little"))
    rebuilt.extend((20+len(encoded)+len(rest)).to_bytes(4,"little"))
    rebuilt.extend(len(encoded).to_bytes(4,"little"))
    rebuilt.extend(b"JSON")
    rebuilt.extend(encoded)
    rebuilt.extend(rest)
    if len(rebuilt)>MAX_GLB_BYTES:
        raise ValueError("Attested GLB exceeds response limit")
    return bytes(rebuilt)

def _upload_glb(url,path,content):
    if not isinstance(url,str) or not isinstance(path,str) or not path or len(path)>700 or ".." in path:
        raise ValueError("Invalid artifact upload target")
    parsed=urlparse(url)
    allowed=set(filter(None,os.environ.get("ALLOWED_OUTPUT_HOSTS","").split(",")))
    expected="/storage/v1/object/upload/sign/vehicle-3d-private/"+urllib.parse.quote(path,safe="/")
    token=urllib.parse.parse_qs(parsed.query).get("token",[])
    if parsed.scheme!="https" or not parsed.hostname or parsed.hostname not in allowed or parsed.path!=expected \
       or len(token)!=1 or not token[0] or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Artifact upload target is not trusted")
    req=urllib.request.Request(url,data=content,method="PUT",headers={
        "Content-Type":"model/gltf-binary","Cache-Control":"max-age=3600","x-upsert":"false"
    })
    with urllib.request.urlopen(req,timeout=120) as response:
        if response.url!=url: raise ValueError("Redirects are not accepted for artifact uploads")
        if response.status not in (200,201): raise RuntimeError("Artifact upload failed")

def _safe_diag(value,limit=240):
    text=str(value or "")
    text=re.sub(r"https?://[^\s]+","[url]",text,flags=re.I)
    text=re.sub(r"[A-Za-z0-9_-]{40,}","[redacted]",text)
    return re.sub(r"\s+"," ",text).strip()[:limit]

def _comfy_failure(record):
    status=record.get("status",{}) if isinstance(record,dict) else {}
    messages=status.get("messages",[]) if isinstance(status,dict) else []
    if isinstance(messages,list):
        for entry in reversed(messages):
            if not isinstance(entry,(list,tuple)) or len(entry)<2 or entry[0]!="execution_error" or not isinstance(entry[1],dict):
                continue
            detail=entry[1]
            node_type=_safe_diag(detail.get("node_type"),80)
            node_id=_safe_diag(detail.get("node_id"),32)
            exc_type=_safe_diag(detail.get("exception_type"),80)
            message=_safe_diag(detail.get("exception_message"),240)
            label=node_type+(("#"+node_id) if node_id else "")
            suffix=": "+message if message else (": "+exc_type if exc_type else "")
            return "ComfyUI workflow failed"+((" ["+label+"]") if label else "")+suffix
    return "ComfyUI workflow failed; no safe node diagnostic was returned"

_PROGRESS_EVENT_TYPES={"execution_start","execution_cached","executing","progress","executed","execution_success"}

def _comfy_ws_url(client_id):
    parsed=urllib.parse.urlsplit(BASE)
    scheme="wss" if parsed.scheme=="https" else "ws"
    base_path=parsed.path.rstrip("/")
    path=(base_path+"/ws") if base_path else "/ws"
    return urllib.parse.urlunsplit((scheme,parsed.netloc,path,"clientId="+urllib.parse.quote(client_id),""))

def _is_progress_event(message,prompt_id):
    if not isinstance(message,dict) or message.get("type") not in _PROGRESS_EVENT_TYPES:
        return False
    data=message.get("data")
    if not isinstance(data,dict):
        return False
    event_prompt=data.get("prompt_id")
    return event_prompt is None or str(event_prompt)==str(prompt_id)

def _interrupt_prompt(prompt_id):
    for path,payload in (("/interrupt",{}),("/queue",{"delete":[prompt_id]})):
        try:
            _request(path,payload=payload,timeout=10)
        except Exception:
            pass

def _queue_contains_prompt(queue,prompt_id):
    if not isinstance(queue,dict):
        return False
    target=str(prompt_id)
    for key in ("queue_running","queue_pending"):
        entries=queue.get(key,[])
        if not isinstance(entries,list):
            continue
        for entry in entries:
            if isinstance(entry,(list,tuple)) and len(entry)>1 and str(entry[1])==target:
                return True
            if isinstance(entry,dict) and str(entry.get("prompt_id",""))==target:
                return True
    return False

def _prompt_is_active(prompt_id):
    try:
        return _queue_contains_prompt(_request("/queue",timeout=10),prompt_id)
    except Exception:
        return None

def handler(job):
    inp=job.get("input") or {}; views=inp.get("views") or {}; artifact_upload=inp.get("artifact_upload") or {}
    if set(views)!=set(VIEWS): raise ValueError("Exactly four named views are required")
    if inp.get("output_format","glb")!="glb": raise ValueError("Only GLB output is supported")
    if not isinstance(artifact_upload,dict) or not artifact_upload.get("url") or not artifact_upload.get("path"):
        raise ValueError("Signed artifact upload target is required")
    workflow,mapping=_load_api_workflow()
    preprocessing={}
    for view in VIEWS:
        spec=views[view]
        if not isinstance(spec,dict) or set(spec)!={"url","crop"}:
            raise ValueError("Each vehicle view must contain url and crop")
        content,_mime=_download_image(spec["url"])
        content,mime,metrics=_prepare_vehicle_rgba(content,spec["crop"])
        preprocessing[view]=metrics
        name=f"{uuid.uuid4().hex}.png"
        uploaded=_upload(content,mime,name); workflow[mapping[view]]["inputs"]["image"]=uploaded["name"]
    started=time.monotonic()
    client_id=str(uuid.uuid4())
    total_timeout=int(os.environ.get("JOB_TIMEOUT_SECONDS","1800"))
    stall_timeout=int(os.environ.get("STALL_TIMEOUT_SECONDS","300"))
    if stall_timeout<30: raise ValueError("STALL_TIMEOUT_SECONDS must be at least 30")
    ws=websocket.create_connection(_comfy_ws_url(client_id),timeout=10,http_proxy_host=None,http_proxy_port=None)
    prompt_id=None
    try:
        queued=_request("/prompt",payload={"prompt":workflow,"client_id":client_id},timeout=30)
        prompt_id=str(queued["prompt_id"])
        deadline=time.monotonic()+total_timeout
        last_progress=time.monotonic()
        while True:
            now=time.monotonic()
            if now>=deadline:
                _interrupt_prompt(prompt_id)
                raise TimeoutError("ComfyUI job exceeded total deadline")
            if now-last_progress>=stall_timeout:
                active=_prompt_is_active(prompt_id)
                if active is True:
                    # Some TRELLIS nodes spend minutes inside one GPU operation without
                    # emitting websocket progress. Being present in ComfyUI's running
                    # queue is a liveness signal, so do not kill a healthy long node.
                    last_progress=now
                elif active is None:
                    # A single local queue probe failure is not enough evidence to kill
                    # an expensive GPU job. Retry shortly; the hard job deadline still
                    # bounds total runtime.
                    last_progress=now-max(0,stall_timeout-30)
                else:
                    _interrupt_prompt(prompt_id)
                    raise TimeoutError("ComfyUI job stalled for "+str(stall_timeout)+" seconds and is no longer active")
            history=_request("/history/"+prompt_id,timeout=10)
            if prompt_id in history:
                record=history[prompt_id]
                if record.get("status",{}).get("status_str")=="error": raise RuntimeError(_comfy_failure(record))
                content,glb=_extract_glb(record.get("outputs",{}))
                content=_embed_preprocessing_attestation(content,preprocessing)
                artifact={**glb,"format":"glb","storage_path":artifact_upload["path"],
                          "bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
                _upload_glb(artifact_upload["url"],artifact_upload["path"],content)
                return {"status":"completed","prompt_id":prompt_id,"elapsed_seconds":round(time.monotonic()-started,2),
                        "preprocessing":preprocessing,
                        "artifact":artifact}
            remaining=min(3.0,deadline-time.monotonic(),stall_timeout-(time.monotonic()-last_progress))
            ws.settimeout(max(0.1,remaining))
            try:
                raw=ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                _interrupt_prompt(prompt_id)
                raise RuntimeError("ComfyUI progress channel failed; generation interrupted") from exc
            if isinstance(raw,str):
                try:
                    event=json.loads(raw)
                except json.JSONDecodeError:
                    event=None
                if _is_progress_event(event,prompt_id):
                    last_progress=time.monotonic()
    finally:
        try: ws.close()
        except Exception: pass

if __name__=="__main__": runpod.serverless.start({"handler":handler})
