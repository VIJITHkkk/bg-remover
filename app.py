import io
import os
import urllib.request

import numpy as np
from flask import Flask, request, send_file, Response
from PIL import Image

app = Flask(__name__)

MAX_SIDE = 1600  # keep images small enough for 512MB RAM
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_PATH = "/tmp/u2netp.onnx"

_session = None


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
button{margin-top:12px;padding:12px 18px;border-radius:7px;border:0;background:#c1832f;color:#171310;font-weight:600;width:100%}
img{width:100%;margin-top:14px;border-radius:10px;background:repeating-conic-gradient(#55575c 0% 25%,#3a3c40 0% 50%) 50%/20px 20px}
#st{color:#93959c;margin-top:10px;font-size:.9rem}
input{display:none}
</style></head><body>
<h1>AI Background Remover</h1>
<div class="drop" onclick="f.click()">Tap to choose a photo</div>
<input type="file" id="f" accept="image/*">
<div id="st"></div>
<img id="out" style="display:none">
<a id="dl" download="background-removed.png"><button id="b" style="display:none">Download PNG</button></a>
<script>
const f=document.getElementById('f'),st=document.getElementById('st'),out=document.getElementById('out'),b=document.getElementById('b'),dl=document.getElementById('dl');
f.onchange=async()=>{
  if(!f.files[0])return;
  st.textContent='Processing... (first photo can take 1-2 minutes while the AI model downloads)';
  out.style.display='none';b.style.display='none';
  const fd=new FormData();fd.append('image',f.files[0]);
  try{
    const r=await fetch('/remove',{method:'POST',body:fd});
    if(!r.ok){const t=await r.text();throw new Error(r.status+' '+t.slice(0,200));}
    const blob=await r.blob();const url=URL.createObjectURL(blob);
    out.src=url;out.style.display='block';dl.href=url;b.style.display='block';st.textContent='Done.';
  }catch(e){st.textContent='Error: '+e.message;}
};
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/remove", methods=["POST"])
def remove_bg():
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
