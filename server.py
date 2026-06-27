"""
server.py -- FastAPI backend for the NautiCAI sonar detection app.

    pip install -r requirements.txt
    copy .env.example .env   # then set ANTHROPIC_API_KEY
    python server.py         # http://127.0.0.1:8000/review
"""
import asyncio
import base64
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load .env from project root (does not override existing env vars)."""
    env_file = HERE / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_file, override=False)


_load_dotenv()

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from brand import BRAND, TAGLINE, WEBSITE
from report import (
    build_report,
    build_survey_report,
    build_clearance_certificate,
    build_defects_executive_report,
    annotate_image,
)
from agent_service import router as agent_router
from agent_actions import router as agent_actions_router
from agent_triage import router as triage_router
from agent_rov import router as rov_router
from shadow_physics import annotate_detections

_pw_browsers = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
if _pw_browsers.is_dir() and "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_pw_browsers)

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
app.include_router(agent_router)
app.include_router(agent_actions_router)
app.include_router(triage_router)
app.include_router(rov_router)
app.mount("/assets", StaticFiles(directory=HERE / "assets"), name="assets")


@app.get("/api/brand")
def brand_info():
    return JSONResponse({"brand": BRAND, "tagline": TAGLINE, "website": WEBSITE})


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
    html = (HERE / name).read_text(encoding="utf-8").replace(
        "/assets/nauticai-logo-nav.png", _LOGO_SRC
    )
    if '<meta name="application-name"' not in html:
        html = html.replace(
            "<head>",
            f'<head>\n<meta name="application-name" content="{BRAND}">',
            1,
        )
    return html


@app.get("/")
def index():
    return RedirectResponse(url="/review", status_code=302)


@app.get("/demo", response_class=HTMLResponse)
def demo_upload():
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


@app.get("/defects-report", response_class=HTMLResponse)
def defects_report_page():
    return _page("defects_report.html")


@app.get("/briefing", response_class=HTMLResponse)
def briefing_page():
    return _page("briefing.html")


@app.get("/workflow", response_class=HTMLResponse)
def workflow_page():
    return _page("workflow.html")


@app.get("/rov", response_class=HTMLResponse)
def rov_page():
    return _page("rov.html")


@app.get("/settings", response_class=HTMLResponse)
def settings_page():
    return _page("settings.html")


@app.get("/intelligence", response_class=HTMLResponse)
def intelligence_page():
    return _page("intelligence.html")


@app.get("/copilot", response_class=HTMLResponse)
@app.get("/chatbot", response_class=HTMLResponse)
def copilot_page():
    return _page("chat.html")


@app.get("/triage", response_class=HTMLResponse)
def triage_page():
    return _page("triage.html")


@app.get("/dive-plan", response_class=HTMLResponse)
def dive_plan_page():
    return _page("dive_plan.html")


@app.post("/api/intelligence/clearance")
async def intelligence_clearance(request: Request):
    try:
        meta = await request.json()
    except Exception:
        meta = {}
    meta.setdefault("brand", BRAND)
    pdf = build_clearance_certificate(meta)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=NautiCAI_Clearance_Certificate.pdf"},
    )


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    range_scale: float = Form(150.0),
):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)

    r = model.predict(source=img, conf=conf, verbose=False)[0]
    detections = _detections_from_result(r)
    # Measure the real acoustic shadow behind each contact → pixel-based height.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annotate_detections(gray, detections, range_scale=range_scale)
    annotated = annotate_image(img, detections)
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": detections,
    })


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    project: str = Form("Offshore Wind Site Assessment"),
    location: str = Form("North Sea Sector A"),
    survey_id: str = Form("SURV-2026-001"),
    sensor: str = Form("Side-scan sonar (SSS)"),
    vessel: str = Form(""),
):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)

    r = model.predict(source=img, conf=conf, verbose=False)[0]
    detections = _detections_from_result(r)
    meta = {
        "brand": BRAND,
        "project": project,
        "location": location,
        "vessel": vessel,
        "survey_id": survey_id,
        "sensor": sensor,
        "filename": file.filename or "sonar_frame",
        "threshold": conf,
        "model": "YOLOv8 (trained)" if TRAINED.exists() else "YOLOv8n (untrained fallback)",
        "base_url": "http://127.0.0.1:8000",
    }
    try:
        pdf = await asyncio.to_thread(build_report, raw, detections, meta)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{BRAND}_Seabed_Object_Detection_Report.pdf"'},
    )


def _survey_pdf_from_body(body):
    frames_in = body.get("frames") or []
    if not frames_in:
        return None, JSONResponse({"error": "no frames supplied"}, status_code=400)

    frames_data = []
    for i, fr in enumerate(frames_in):
        b64 = fr.get("image_b64") or ""
        if isinstance(b64, str) and "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None, JSONResponse({"error": "could not decode frame %d image" % (i + 1)}, status_code=400)
        if cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is None:
            return None, JSONResponse({"error": "could not read frame %d image" % (i + 1)}, status_code=400)
        dets = []
        for d in fr.get("detections") or []:
            box = d.get("box")
            if (not box or len(box) < 4) and d.get("cxn") is None:
                continue
            det = {
                "cls": d.get("cls", "object"),
                "conf": float(d.get("conf", 0)),
                "cxn": d.get("cxn"),
                "cyn": d.get("cyn"),
                "wnorm": d.get("wnorm"),
            }
            if box and len(box) >= 4:
                det["box"] = [int(v) for v in box[:4]]
            else:
                det["box"] = [0, 0, 0, 0]
            dets.append(det)
        frames_data.append(
            {
                "image_bytes": raw,
                "detections": dets,
                "filename": fr.get("filename") or ("frame_%d.png" % (i + 1)),
                "nat_w": fr.get("natW") or fr.get("nat_w"),
                "nat_h": fr.get("natH") or fr.get("nat_h"),
            }
        )

    meta = {
        "brand": BRAND,
        "project": body.get("project", "Offshore Wind Site Assessment"),
        "location": body.get("location", "North Sea Sector A"),
        "vessel": body.get("vessel", ""),
        "survey_id": body.get("survey_id", "SURV-2026-001"),
        "sensor": body.get("sensor", "Side-scan sonar (SSS)"),
        "threshold": body.get("threshold", 0.25),
        "model": "YOLOv8 (trained)" if TRAINED.exists() else "YOLOv8n (untrained fallback)",
        "base_url": body.get("base_url", "http://127.0.0.1:8000"),
        "rov_queue": body.get("rov_queue"),
        "lifecycle": body.get("lifecycle"),
        "intelligence_roi": body.get("intelligence_roi"),
        "fused_contacts": body.get("fused_contacts"),
    }
    try:
        return build_survey_report(frames_data, meta), None
    except Exception as e:
        return None, JSONResponse({"error": str(e)}, status_code=500)


async def _survey_pdf_response(request: Request, filename: str):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    pdf, err = await asyncio.to_thread(_survey_pdf_from_body, body)
    if err is not None:
        return err
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/report/survey")
async def report_survey(request: Request):
    return await _survey_pdf_response(request, f"{BRAND}_Seabed_Object_Detection_Report.pdf")


@app.post("/report-multi")
async def report_multi(request: Request):
    return await _survey_pdf_response(request, "Seabed_Survey_Report.pdf")


@app.post("/report/defects-executive")
async def report_defects_executive(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    try:
        pdf = await asyncio.to_thread(build_defects_executive_report, body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="NautiCAI_Seabed_Defects_Report.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn
    print("Open http://127.0.0.1:8000/review  (copilot: /copilot, intelligence: /intelligence)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
