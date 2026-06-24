# SEABED — Sonar Object Detection — Handoff

> **For Cursor (read this first):** The model weights live at `runs/detect/sonar_yolo/weights/best.pt` (8 seabed classes). Your job is to **run the web app** — upload sonar images and verify detections. Do not change the class color map unless asked. To retrain on synthetic + UATD data, use `python retrain_sonar.py` (see Training below).

## What this project is
NautiCAI seabed mapping PoC: upload side-scan sonar frames → YOLO detects seabed objects → review in Data Review, 3D twin, mosaic view, measurement tool, and multi-frame PDF reports. FastAPI (`server.py`) serves all pages and runs inference on `/detect`.

## Current state
- Trained model: `runs/detect/sonar_yolo/weights/best.pt`
- **8 classes:** rock, boulder, wreck, pipeline, debris, cable, scour, obstruction
- Trained on **synthetic side-scan** frames; can be **fine-tuned** on real UATD via `retrain_sonar.py` (Test_1 + Test_2 mapped to seabed classes, merged with synthetic data in `data/combined/`)
- `server.py` auto-loads `best.pt`; falls back to `yolov8n` if missing (not sonar-trained)

## Platform features (NautiCAI PoC)
- **Data Review** (`/review`, `console.html`) — multi-frame upload, detection boxes, session restore
- **Confidence threshold** — Settings page + Review sidebar; synced via `localStorage` key `set_conf`; used on `/detect` and PDF reports; **Re-detect with threshold** button
- **Survey Mosaic** — View Mode → *Survey Mosaic* stitches consecutive frames into one strip with detection overlays
- **Measurement tool** — ↔ button on sonar view: two clicks measure distance in metres using range scale
- **3D Twin** (`/twin`) — per-frame seabed heatmap with markers and labels
- **Multi-frame PDF** — `POST /report-multi` from Review; Playwright embeds live twin screenshots
- **Range scale** — adjustable 50–300 m; drives ruler, detection ranges, and measurements

## Project layout
```
seabed/
├── server.py        # FastAPI backend (pages + /detect + /report + /report-multi)
├── console.html     # Data Review (main workflow)
├── twin.html        # 3D seabed digital twin
├── report.py        # PDF report builder
├── brand.py         # NautiCAI branding constants
├── index.html       # simple single-image upload UI
├── detect.py        # optional CLI inference
├── retrain_sonar.py # prepare synthetic+UATD dataset and fine-tune
├── convert_uatd.py  # UATD VOC XML → YOLO format
├── requirements.txt
├── sonar.yaml       # 3-class legacy synthetic config
└── runs/detect/sonar_yolo/weights/best.pt
```

## Retraining (synthetic + UATD)
Requires `~/Downloads/21331143.zip` (UATD figshare bundle).

```powershell
python retrain_sonar.py --epochs 20 --batch 8 --device cpu
```

This will:
1. Generate 800 synthetic frames (`make_sidescan.py`)
2. Extract & convert UATD Test_1 (train) + Test_2 (val)
3. Map UATD classes → 8 seabed classes and merge into `data/combined/`
4. Fine-tune `best.pt` (backs up previous weights as `best.pt.bak`)

Dataset yaml must use an absolute `path:` — see `data/combined/seabed.yaml`.

## Steps to run (Windows)
```powershell
# 1. from the seabed project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should show (.venv)

# 2. install dependencies (first time only, a few minutes)
pip install -r requirements.txt
playwright install chromium   # optional: live 3D twin screenshots in PDF reports

# 3. confirm the model is in place:
#    runs\detect\sonar_yolo\weights\best.pt   (the UATD best.pt)

# 4. start the app
python server.py
#    -> it prints: Open http://127.0.0.1:8000/review
#    -> the terminal should NOT print the "trained weights not found" warning.
```
Then open **http://127.0.0.1:8000/review** (Data Review console). Drag one or more
sonar BMPs onto the viewer — detections run automatically. Use **Settings** or the
sidebar confidence slider (synced via `set_conf` in localStorage). For multi-frame
surveys, switch **View Mode → Survey Mosaic** or click **Generate report (PDF)**.

> If the terminal prints `[warn] trained weights not found`, then `best.pt` is
> not at `runs/detect/sonar_yolo/weights/best.pt` — fix that path and restart.

## Test images
You need at least one real sonar image to upload. UATD images are `.bmp`. If none
are on the machine, reconnect the Colab notebook and run this in a cell to grab a few:
```python
from google.colab import files
from pathlib import Path
import random
imgs = [p for p in Path("/content/uatd_raw").rglob("*") if p.suffix.lower()==".bmp"]
for p in random.sample(imgs, min(3, len(imgs))):
    files.download(str(p))
```
(That requires re-downloading the dataset since Colab wipes files on disconnect.)

---

## Files (canonical contents)

### `requirements.txt`
```text
ultralytics>=8.0
opencv-python>=4.8
numpy>=1.24
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
gradio>=4.0

```

### `server.py`
```python
"""
server.py -- FastAPI backend for the sonar detection web UI (index.html).

    pip install fastapi "uvicorn[standard]" python-multipart ultralytics
    python server.py            # then open http://127.0.0.1:8000

Serves index.html at "/" and exposes POST /detect (image upload -> annotated
image + JSON detections). Loads runs/detect/sonar_yolo/weights/best.pt if it
exists, otherwise falls back to yolov8n (NOT trained on sonar -- train first
for meaningful detections).
"""
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
TRAINED = HERE / "runs" / "detect" / "sonar_yolo" / "weights" / "best.pt"

weights = str(TRAINED) if TRAINED.exists() else "yolov8n.pt"
if weights == "yolov8n.pt":
    print("[warn] trained weights not found -> using yolov8n (not sonar-trained).")
model = YOLO(weights)

app = FastAPI(title="Seabed Sonar Detector")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)

    r = model.predict(source=img, conf=conf, verbose=False)[0]
    annotated = r.plot()                      # BGR numpy with boxes drawn
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    detections = [
        {
            "cls": model.names[int(b.cls)],
            "conf": round(float(b.conf), 3),
            "box": [int(v) for v in b.xyxy[0]],
        }
        for b in r.boxes
    ]
    return JSONResponse({
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": detections,
    })


if __name__ == "__main__":
    import uvicorn
    print("Open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

```

### `index.html`
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonar Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;}
  body{
    margin:0; font-family:'Sora','Segoe UI',system-ui,sans-serif; color:#16282e;
    background:#e7eef0; background-image:linear-gradient(180deg,#f2f7f8,#e2eced);
    min-height:100vh; padding:28px 16px;
  }
  .container{max-width:720px; margin:0 auto;}

  /* header */
  .header{display:flex; align-items:center; margin-bottom:22px;}
  .logo{width:42px; height:42px; border-radius:12px; background:#0d8a7e;
        display:flex; align-items:center; justify-content:center; margin-right:13px; flex:none;}
  .logo svg{width:23px; height:23px; stroke:#fff;}
  .header h1{font-size:20px; font-weight:700; margin:0;}
  .header p{margin:2px 0 0; font-size:12px; color:#6a7e84;}
  .pill{margin-left:auto; font-size:12px; font-weight:600; padding:7px 13px; border-radius:999px;
        background:#fdf3e2; color:#c2841a; display:flex; align-items:center;}
  .pill .d{width:8px; height:8px; border-radius:50%; background:#d99a2b; margin-right:7px;}
  .pill.on{background:#e2f5f1; color:#0a6f66;}
  .pill.on .d{background:#0d8a7e;}

  .card{background:#fff; border:1px solid #d9e3e4; border-radius:16px;
        box-shadow:0 6px 20px rgba(16,40,46,.06); padding:18px; margin-bottom:18px;}

  /* upload / scope */
  .scope{border:2px dashed #bcced0; border-radius:13px; background:#f6fafa; min-height:300px;
         display:flex; align-items:center; justify-content:center; text-align:center; cursor:pointer; padding:20px; transition:.15s;}
  .scope.drag{border-color:#0d8a7e; background:#edf8f5;}
  .up{width:56px; height:56px; border-radius:50%; background:#e3f3f0;
      display:flex; align-items:center; justify-content:center; margin:0 auto 14px;}
  .up svg{width:27px; height:27px; stroke:#0d8a7e;}
  .scope b{display:block; font-size:15px; margin-bottom:3px;}
  .scope .hint{font-size:12px; color:#6a7e84;}
  .scope img{max-width:100%; max-height:460px; border-radius:9px; display:block;}

  /* controls */
  .ctl-top{display:flex; align-items:center; margin-bottom:11px;}
  .ctl-top svg{width:18px; height:18px; stroke:#0d8a7e; margin-right:8px;}
  .ctl-top .lab{font-size:13px; font-weight:600;}
  .ctl-top .val{margin-left:auto; font-size:14px; font-weight:700; color:#0d8a7e;}
  input[type=range]{width:100%; height:5px; accent-color:#0d8a7e; cursor:pointer;}
  .btn{margin-top:16px; width:100%; padding:13px; border:0; border-radius:11px; background:#0d8a7e;
       color:#fff; font-family:inherit; font-size:14px; font-weight:600;
       display:flex; align-items:center; justify-content:center; cursor:pointer; transition:.15s;}
  .btn svg{width:17px; height:17px; stroke:#fff; margin-right:9px;}
  .btn:hover:not(:disabled){background:#0a6f66;}
  .btn:disabled{background:#aec5c2; cursor:not-allowed;}

  /* contacts */
  .res-head{display:flex; align-items:center; margin-bottom:4px;}
  .res-head svg{width:18px; height:18px; stroke:#0d8a7e; margin-right:8px;}
  .res-head h2{font-size:14px; font-weight:600; margin:0;}
  .res-head .count{margin-left:auto; font-size:13px; font-weight:700; color:#0a6f66;
                   background:#e3f3f0; border-radius:8px; padding:2px 10px;}
  .empty{padding:26px 10px; text-align:center; color:#8a9ca0; font-size:13px;}
  .row{display:flex; align-items:center; padding:12px 4px; border-top:1px solid #eef3f3;}
  .row .dot{width:11px; height:11px; border-radius:3px; margin-right:12px; flex:none;}
  .row .name{font-size:14px; text-transform:capitalize;}
  .row .meta{font-size:11px; color:#8a9ca0; margin-top:2px;}
  .row .bar{width:84px; height:5px; border-radius:3px; background:#e7eeed; margin-left:auto; overflow:hidden; flex:none;}
  .row .bar i{display:block; height:100%; background:#0d8a7e;}
  .row .conf{font-size:14px; font-weight:700; color:#16282e; margin-left:14px; width:42px; text-align:right;}

  .foot{text-align:center; font-size:11px; color:#8a9ca0; margin-top:4px;}
  .foot code{background:#e3edee; border-radius:5px; padding:1px 6px; color:#0a6f66;}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div class="logo">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12a9 9 0 1 1-9-9"/><path d="M16.5 12a4.5 4.5 0 1 1-4.5-4.5"/>
        <circle cx="12" cy="12" r="1.4" fill="#fff" stroke="none"/><path d="M12 12l8-6"/>
      </svg>
    </div>
    <div>
      <h1>Sonar Detector</h1>
      <p>Upload a sonar image to detect targets</p>
    </div>
    <div class="pill" id="pill"><span class="d"></span>Standby</div>
  </div>

  <div class="card">
    <div class="scope" id="scope">
      <div id="prompt">
        <div class="up">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 16V5"/><path d="M7 10l5-5 5 5"/><path d="M5 19h14"/>
          </svg>
        </div>
        <b>Drop a sonar image</b>
        <div class="hint">or click to browse · PNG / JPG</div>
      </div>
      <img id="img" alt="" hidden>
    </div>
    <input type="file" id="file" accept="image/*" hidden>
  </div>

  <div class="card">
    <div class="ctl-top">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 6h8M16 6h4M4 12h4M12 12h8M4 18h6M14 18h6"/>
        <circle cx="14" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="12" cy="18" r="2"/>
      </svg>
      <span class="lab">Confidence</span>
      <span class="val" id="cv">0.25</span>
    </div>
    <input type="range" id="conf" min="0.05" max="0.9" step="0.05" value="0.25">
    <button class="btn" id="run" disabled>
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 5v14l11-7z"/></svg>
      Run detection
    </button>
  </div>

  <div class="card">
    <div class="res-head">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="8"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>
        <circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/>
      </svg>
      <h2>Contacts</h2>
      <span class="count" id="count">0</span>
    </div>
    <div id="list"><div class="empty">Load an image and run detection to see targets.</div></div>
  </div>

  <div class="foot">Run <code>python server.py</code> and open <code>127.0.0.1:8000</code> to use it live.</div>
</div>

<script>
  const $=s=>document.querySelector(s);
  const scope=$('#scope'),file=$('#file'),prompt=$('#prompt'),img=$('#img');
  const run=$('#run'),conf=$('#conf'),cv=$('#cv'),list=$('#list'),count=$('#count'),pill=$('#pill');
  const COLORS={mine:'#e0612f',wreck:'#0d8a7e',rock:'#3b82c4'};
  const PLAY='<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 5v14l11-7z"/></svg> Run detection';
  let cur=null;

  scope.addEventListener('click',()=>file.click());
  ['dragover','dragenter'].forEach(e=>scope.addEventListener(e,ev=>{ev.preventDefault();scope.classList.add('drag');}));
  ['dragleave','dragend','drop'].forEach(e=>scope.addEventListener(e,ev=>{ev.preventDefault();scope.classList.remove('drag');}));
  scope.addEventListener('drop',ev=>{if(ev.dataTransfer.files[0])load(ev.dataTransfer.files[0]);});
  file.addEventListener('change',()=>{if(file.files[0])load(file.files[0]);});
  conf.addEventListener('input',()=>cv.textContent=Number(conf.value).toFixed(2));

  function load(f){
    if(!f.type.startsWith('image/'))return;
    cur=f; const r=new FileReader();
    r.onload=e=>{img.src=e.target.result; img.hidden=false; prompt.hidden=true;};
    r.readAsDataURL(f);
    run.disabled=false;
    list.innerHTML='<div class="empty">Image loaded — press Run detection.</div>'; count.textContent='0';
  }

  run.addEventListener('click',async()=>{
    if(!cur)return;
    run.disabled=true; run.textContent='Scanning…';
    const fd=new FormData(); fd.append('file',cur); fd.append('conf',conf.value);
    try{
      const c=new AbortController(); const t=setTimeout(()=>c.abort(),8000);
      const res=await fetch('/detect',{method:'POST',body:fd,signal:c.signal}); clearTimeout(t);
      if(!res.ok)throw 0;
      const d=await res.json();
      if(d.image){img.src=d.image; img.hidden=false; prompt.hidden=true;}
      contacts(d.detections||[]); setPill(true);
    }catch(e){
      list.innerHTML='<div class="empty">Backend offline — start <b>python server.py</b> and reload to run live detection.</div>';
      count.textContent='0'; setPill(false);
    }finally{
      run.disabled=false; run.innerHTML=PLAY;
    }
  });

  function contacts(dets){
    count.textContent=dets.length;
    if(!dets.length){list.innerHTML='<div class="empty">No targets above the threshold — try lowering confidence.</div>'; return;}
    list.innerHTML='';
    dets.sort((a,b)=>b.conf-a.conf).forEach(d=>{
      const pct=Math.round(d.conf*100), col=COLORS[d.cls]||'#0d8a7e';
      const box=d.box?`${d.box[2]-d.box[0]}×${d.box[3]-d.box[1]} px at ${d.box[0]},${d.box[1]}`:'';
      const row=document.createElement('div'); row.className='row';
      row.innerHTML='<span class="dot" style="background:'+col+'"></span>'
        +'<div><div class="name">'+d.cls+'</div><div class="meta">'+box+'</div></div>'
        +'<span class="bar"><i style="width:'+pct+'%"></i></span>'
        +'<span class="conf">'+pct+'%</span>';
      list.appendChild(row);
    });
  }

  function setPill(on){pill.className='pill'+(on?' on':''); pill.innerHTML='<span class="d"></span>'+(on?'Online':'Standby');}
</script>
</body>
</html>

```

### `detect.py`
```python
"""
detect.py
---------
Run a trained model on images / a folder and save annotated results.

    pip install ultralytics
    python detect.py --source data/images/val
    python detect.py --weights runs/detect/sonar_yolo/weights/best.pt \
                     --source path/to/some_sonar_image.png --conf 0.3
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/sonar_yolo/weights/best.pt")
    ap.add_argument("--source", required=True, help="image, folder, or glob")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, conf=args.conf, device=args.device, save=True)
    print("Annotated results saved under runs/detect/predict*/")


if __name__ == "__main__":
    main()

```

### `sonar.yaml`
```yaml
# Only used for (re)training on the synthetic set. The running app reads the
# real class names from best.pt, so this file is NOT needed for inference.
path: ./data
train: images/train
val: images/val

names:
  0: mine
  1: wreck
  2: rock

```
