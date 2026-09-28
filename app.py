import io
import os
from flask import Flask, request, send_file, Response
from PIL import Image
from rembg import remove, new_session

app = Flask(__name__)

# Small model (~5MB) instead of the default 176MB one, to stay under 512MB RAM.
session = new_session("u2netp")

MAX_SIDE = 2048  # cap size so big photos don't exhaust memory

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
  st.textContent='Processing... (first request may take up to a minute while the server wakes up)';
  out.style.display='none';b.style.display='none';
  const fd=new FormData();fd.append('image',f.files[0]);
  try{
    const r=await fetch('/remove',{method:'POST',body:fd});
    if(!r.ok)throw new Error(await r.text());
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
    result = remove(img, session=session)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
