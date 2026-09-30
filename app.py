import io
import os
import time
import threading
import urllib.request

import numpy as np
from flask import Flask, request, send_file, Response, jsonify
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # reject uploads over 8MB

MAX_SIDE = 1600  # keep images small enough for 512MB RAM
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = "/tmp/u2netp.onnx"

_session = None
_process_lock = threading.Lock()  # only run one image at a time (limited RAM)
_last_seen = {}  # simple per-IP rate limit, resets if the server restarts
RATE_LIMIT_SECONDS = 8


def get_session():
    """Load the small u2netp model with plain onnxruntime (no rembg, no scipy,
    no numba), which keeps memory use low. Loaded lazily on first request so the
    web server binds its port immediately."""
    global _session
    if _session is None:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        _session = ort.InferenceSession(MODEL_PATH, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    return _session


def make_mask(session, img):
    small = img.convert("RGB").resize((320, 320), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
    name = session.get_inputs()[0].name
    pred = session.run(None, {name: arr})[0][:, 0, :, :]
    ma, mi = pred.max(), pred.min()
    pred = (pred - mi) / max(float(ma - mi), 1e-6)
    pred = np.squeeze(pred)
    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    return mask.resize(img.size, Image.LANCZOS)


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Background Remover</title>
<style>
body{margin:0;background:#121316;color:#eeece5;font-family:-apple-system,Arial,sans-serif;padding:24px;max-width:560px;margin:auto}
h1{font-size:1.3rem}
.drop{border:1px solid #333;border-radius:10px;padding:34px 16px;text-align:center;background:#1a1c20;cursor:pointer}
button{margin-top:10px;padding:12px 18px;border-radius:7px;border:1px solid #333;background:transparent;color:#eeece5;font-weight:600;width:100%}
button.primary{background:#c1832f;color:#171310;border-color:#c1832f}
button.active{background:#3a5a45;border-color:#5fae5a;color:#dfffe0}
.row{display:flex;gap:8px}
.row button{flex:1}
#stage{display:none;position:relative;margin-top:14px;border-radius:10px;overflow:hidden;
  background-image:linear-gradient(45deg,#55575c 25%,transparent 25%),linear-gradient(-45deg,#55575c 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#55575c 75%),linear-gradient(-45deg,transparent 75%,#55575c 75%);
  background-color:#3a3c40;background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0px;}
canvas{display:block;width:100%;touch-action:none}
#st{color:#93959c;margin-top:10px;font-size:.9rem}
.sizerow{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:.85rem;color:#93959c}
.sizerow input{-webkit-appearance:auto;appearance:auto;display:block;flex:1}
input[type=file]{display:none}
</style></head><body>
<h1>AI Background Remover</h1>
<div class="drop" onclick="f.click()">Tap to choose a photo</div>
<input type="file" id="f" accept="image/*">
<div id="st"></div>

<div id="stage"><canvas id="cv"></canvas></div>

<div id="tools" style="display:none">
  <div class="sizerow">Brush size
    <input type="range" id="brush" min="6" max="80" value="24">
  </div>
  <div class="row">
    <button id="eraseBtn" class="active">Erase more</button>
    <button id="restoreBtn">Restore</button>
  </div>
  <button id="undoBtn">↶ Undo last stroke</button>
  <button class="primary" id="dlBtn">Download PNG</button>
  <button id="resetBtn">Start Over</button>
</div>

<script>
const f=document.getElementById('f'),st=document.getElementById('st');
const stage=document.getElementById('stage'),cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const tools=document.getElementById('tools'),brush=document.getElementById('brush');
const eraseBtn=document.getElementById('eraseBtn'),restoreBtn=document.getElementById('restoreBtn');
const dlBtn=document.getElementById('dlBtn'),resetBtn=document.getElementById('resetBtn');
const undoBtn=document.getElementById('undoBtn');

let mode='erase', origImg=null, drawing=false;
let history=[];

f.onchange=async()=>{
  if(!f.files[0])return;
  st.textContent='Processing... (first photo can take 1-2 minutes while the AI model downloads)';
  stage.style.display='none';tools.style.display='none';
  const fd=new FormData();fd.append('image',f.files[0]);
  try{
    const r=await fetch('/remove',{method:'POST',body:fd});
    if(!r.ok){const t=await r.text();throw new Error(r.status+' '+t.slice(0,200));}
    const blob=await r.blob();
    const resultImg=await loadImg(URL.createObjectURL(blob));
    origImg=await loadImg(URL.createObjectURL(f.files[0]));
    cv.width=resultImg.naturalWidth; cv.height=resultImg.naturalHeight;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(resultImg,0,0,cv.width,cv.height);
    history=[];
    stage.style.display='block';tools.style.display='block';
    st.textContent='Done. Use Erase to remove leftover bits, or Restore to bring parts back.';
  }catch(e){st.textContent='Error: '+e.message;}
};

function loadImg(src){return new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src;});}

function setMode(m){mode=m;eraseBtn.classList.toggle('active',m==='erase');restoreBtn.classList.toggle('active',m==='restore');}
eraseBtn.onclick=()=>setMode('erase');
restoreBtn.onclick=()=>setMode('restore');

function canvasPoint(e){
  const rect=cv.getBoundingClientRect();
  const scaleX=cv.width/rect.width, scaleY=cv.height/rect.height;
  const p=e.touches?e.touches[0]:e;
  return {x:(p.clientX-rect.left)*scaleX, y:(p.clientY-rect.top)*scaleY};
}

function paintAt(x,y){
  const r=parseFloat(brush.value);
  if(mode==='erase'){
    ctx.globalCompositeOperation='destination-out';
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();
  } else {
    ctx.save();
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.clip();
    ctx.globalCompositeOperation='destination-over';
    ctx.drawImage(origImg,0,0,cv.width,cv.height);
    ctx.restore();
  }
  ctx.globalCompositeOperation='source-over';
}

function saveHistory(){
  history.push(ctx.getImageData(0,0,cv.width,cv.height));
  if(history.length>15) history.shift();
}
function start(e){e.preventDefault();drawing=true;saveHistory();const p=canvasPoint(e);paintAt(p.x,p.y);}
function move(e){if(!drawing)return;e.preventDefault();const p=canvasPoint(e);paintAt(p.x,p.y);}
function end(){drawing=false;}
cv.addEventListener('pointerdown',start);cv.addEventListener('pointermove',move);
window.addEventListener('pointerup',end);

undoBtn.onclick=()=>{
  if(history.length===0)return;
  const prev=history.pop();
  ctx.putImageData(prev,0,0);
};

dlBtn.onclick=()=>{
  const a=document.createElement('a');
  a.download='background-removed.png';
  a.href=cv.toDataURL('image/png');
  a.click();
};
resetBtn.onclick=()=>{stage.style.display='none';tools.style.display='none';st.textContent='';f.value='';};
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/remove", methods=["POST"])
def remove_bg():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    now = time.time()
    last = _last_seen.get(ip, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return jsonify(error="Too many requests, please wait a few seconds and try again."), 429
    _last_seen[ip] = now

    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400

    if not _process_lock.acquire(blocking=False):
        return jsonify(error="Server is busy processing another photo, please try again shortly."), 503

    try:
        img = Image.open(file.stream).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        session = get_session()
        mask = make_mask(session, img)
        result = img.convert("RGBA")
        result.putalpha(mask)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify(error=f"Could not process this photo: {e}"), 500
    finally:
        _process_lock.release()


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="Photo is too large (max 8MB). Try a smaller photo."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
