"""
server.py -- FastAPI backend for the NautiCAI sonar detection app.

    pip install fastapi "uvicorn[standard]" python-multipart ultralytics reportlab matplotlib
    python server.py            # then open http://127.0.0.1:8000/review

Serves HTML pages and exposes POST /detect and POST /report.
Loads runs/detect/sonar_yolo/weights/best.pt if present, else yolov8n fallback.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from report import build_report

HERE = Path(__file__).resolve().parent
TRAINED = HERE / "runs" / "detect" / "sonar_yolo" / "weights" / "best.pt"
_LOGO_FILE = HERE / "assets" / "nauticai-logo-nav.png"
if _LOGO_FILE.exists():
    _LOGO_SRC = "data:image/png;base64," + base64.b64encode(_LOGO_FILE.read_bytes()).decode()
else:
    _LOGO_SRC = "/assets/nauticai-logo-nav.png"

weights = str(TRAINED) if TRAINED.exists() else "yolov8n.pt"
if weights == "yolov8n.pt":
    print("[warn] trained weights not found -> using yolov8n (not sonar-trained).")
model = YOLO(weights)

app = FastAPI(title="NautiCAI Seabed Detector")
app.mount("/assets", StaticFiles(directory=HERE / "assets"), name="assets")


@app.get("/logo")
def logo():
    path = HERE / "assets" / "nauticai-logo-nav.png"
    if not path.exists():
        path = HERE / "assets" / "nauticai-logo.png"
    if not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/png")


def _detections_from_result(r):
    return [
        {
            "cls": model.names[int(b.cls)],
            "conf": round(float(b.conf), 3),
            "box": [int(v) for v in b.xyxy[0]],
        }
        for b in r.boxes
    ]


def _page(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8").replace(
        "/assets/nauticai-logo-nav.png", _LOGO_SRC
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return _page("index.html")


@app.get("/review", response_class=HTMLResponse)
def review():
    return _page("console.html")


@app.get("/twin", response_class=HTMLResponse)
def twin():
    return _page("twin.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return _page("dashboard.html")


@app.get("/map", response_class=HTMLResponse)
def survey_map():
    return _page("map.html")


@app.get("/detections", response_class=HTMLResponse)
def detections_page():
    return _page("detections.html")


@app.get("/reports", response_class=HTMLResponse)
def reports_page():
    return _page("reports.html")


@app.get("/settings", response_class=HTMLResponse)
def settings_page():
    return _page("settings.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)

    r = model.predict(source=img, conf=conf, verbose=False)[0]
    annotated = r.plot()
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": _detections_from_result(r),
    })


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    project: str = Form("Offshore Wind Site Assessment"),
    location: str = Form("North Sea Sector A"),
    survey_id: str = Form("SURV-2026-001"),
    sensor: str = Form("Side-scan sonar (SSS)"),
):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)

    r = model.predict(source=img, conf=conf, verbose=False)[0]
    detections = _detections_from_result(r)
    meta = {
        "project": project,
        "location": location,
        "survey_id": survey_id,
        "sensor": sensor,
        "filename": file.filename or "sonar_frame",
        "threshold": conf,
        "model": "YOLOv8 (trained)" if TRAINED.exists() else "YOLOv8n (untrained fallback)",
    }
    try:
        pdf = build_report(raw, detections, meta)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="Seabed_Object_Detection_Report.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    print("Open http://127.0.0.1:8000/review  (dashboard: /dashboard)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
