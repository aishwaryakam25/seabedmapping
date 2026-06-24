"""Capture Survey Mosaic + Measurement tool screenshots from /review."""
import base64
import json
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "ui_demos"
IMAGES = [
    ROOT / "data/combined/images/val/syn_ss_00000.png",
    ROOT / "data/combined/images/val/syn_ss_00001.png",
    ROOT / "data/combined/images/val/syn_ss_00002.png",
    ROOT / "data/combined/images/val/syn_ss_00003.png",
]
BASE = "http://127.0.0.1:8000"
CONF = 0.25


def process_dets(raw, nat_w, nat_h):
    out = []
    for d in raw:
        box = d["box"]
        cxn = (box[0] + box[2]) / 2 / nat_w
        cyn = (box[1] + box[3]) / 2 / nat_h
        out.append(
            {
                "cls": d["cls"],
                "conf": d["conf"],
                "box": box,
                "cxn": cxn,
                "cyn": cyn,
                "wnorm": (box[2] - box[0]) / nat_w,
                "east": 512345.67 + (cxn - 0.5) * 200,
                "north": 6123456.78 - (cyn - 0.5) * 200,
                "status": "Review",
                "note": "",
            }
        )
    return out


def detect_image(path: Path):
    import cv2

    raw = path.read_bytes()
    img = cv2.imdecode(__import__("numpy").frombuffer(raw, __import__("numpy").uint8), cv2.IMREAD_COLOR)
    nat_h, nat_w = img.shape[:2]
    files = {"file": (path.name, raw, "image/png")}
    data = {"conf": str(CONF)}
    r = requests.post(f"{BASE}/detect", files=files, data=data, timeout=180)
    r.raise_for_status()
    dets = process_dets(r.json().get("detections") or [], nat_w, nat_h)
    b64 = base64.b64encode(raw).decode()
    data_url = f"data:image/png;base64,{b64}"
    return {
        "name": path.name,
        "thumb": data_url,
        "pdfSrc": data_url,
        "natW": nat_w,
        "natH": nat_h,
        "inferMs": 120,
        "dets": dets,
        "dataUrl": data_url,
    }


def build_session(frames):
    detections = []
    fw, fh, sp = 60, 60, 72
    for fi, fr in enumerate(frames):
        ox = fi * sp
        for d in fr["dets"]:
            detections.append(
                {
                    "cls": d["cls"],
                    "conf": d["conf"],
                    "frame": fi + 1,
                    "x": ox + d["cxn"] * fw,
                    "y": d["cyn"] * fh,
                    "range": abs(d["cxn"] - 0.5) * 2 * 150,
                    "status": d.get("status", "Review"),
                    "crop": "",
                }
            )
    sess = {
        "frames": len(frames),
        "detections": detections,
        "generated": 1,
        "hasData": True,
        "project": "Demo Survey",
        "survey_line": "Line 01",
        "date": "2026-06-16",
        "vessel": "Demo Vessel",
        "sensor": "Side-scan sonar (SSS)",
        "range": "150 m",
    }
    stored = []
    for fr in frames:
        stored.append(
            {
                "name": fr["name"],
                "thumb": fr["thumb"],
                "pdfSrc": fr["pdfSrc"],
                "natW": fr["natW"],
                "natH": fr["natH"],
                "inferMs": fr["inferMs"],
                "dets": fr["dets"],
            }
        )
    return sess, stored


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [p for p in IMAGES if not p.exists()]
    if missing:
        raise SystemExit("Missing images: " + ", ".join(str(p) for p in missing))

    print("Detecting frames via API...")
    frames = [detect_image(p) for p in IMAGES]
    sess, stored = build_session(frames)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"{BASE}/review", wait_until="domcontentloaded", timeout=60000)
        page.evaluate(
            """([sess, stored]) => {
              localStorage.setItem('nauticai_session', JSON.stringify(sess));
              localStorage.setItem('nauticai_review_frames', JSON.stringify(stored));
            }""",
            [sess, stored],
        )
        page.reload(wait_until="domcontentloaded")
        page.wait_for_function(
            "() => parseInt(document.getElementById('kFrames').textContent || '0', 10) >= 4",
            timeout=60000,
        )
        page.wait_for_timeout(1200)

        page.select_option("#viewMode", "mosaic")
        page.wait_for_timeout(900)
        page.locator("#mosaicwrap.on").wait_for(state="visible", timeout=10000)
        page.screenshot(path=str(OUT / "survey_mosaic.png"))

        page.select_option("#viewMode", "sonar")
        page.wait_for_timeout(600)
        page.locator("#measureBtn").click()
        page.wait_for_selector("#measureOverlay.on", timeout=5000)
        stage = page.locator("#stage")
        box = stage.bounding_box()
        if box:
            page.mouse.click(box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.42)
            page.wait_for_timeout(250)
            page.mouse.click(box["x"] + box["width"] * 0.62, box["y"] + box["height"] * 0.55)
            page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "measurement_tool.png"))

        browser.close()

    print("Wrote", OUT / "survey_mosaic.png")
    print("Wrote", OUT / "measurement_tool.png")


if __name__ == "__main__":
    main()
