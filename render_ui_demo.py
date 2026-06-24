"""Render Survey Mosaic + Measurement tool demo images (no browser)."""
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "ui_demos"
IMAGES = [
    ROOT / "data/combined/images/val/syn_ss_00000.png",
    ROOT / "data/combined/images/val/syn_ss_00001.png",
    ROOT / "data/combined/images/val/syn_ss_00002.png",
    ROOT / "data/combined/images/val/syn_ss_00003.png",
]
BASE = "http://127.0.0.1:8000"
RANGE = 150

COLORS = {
    "rock": (27, 177, 217),
    "boulder": (30, 105, 210),
    "wreck": (93, 24, 190),
    "pipeline": (196, 179, 47),
    "debris": (214, 109, 155),
    "cable": (106, 174, 63),
    "scour": (90, 160, 122),
    "obstruction": (178, 167, 154),
    "mine": (93, 24, 190),
    "uxo": (93, 24, 190),
}


def color_for(cls):
    return COLORS.get(cls.lower().replace(" ", "_"), (44, 182, 196))


def detect(path: Path):
    raw = path.read_bytes()
    r = requests.post(
        f"{BASE}/detect",
        files={"file": (path.name, raw, "image/png")},
        data={"conf": "0.25"},
        timeout=180,
    )
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img, r.json().get("detections") or []


def draw_boxes(img, dets, conf=0.25):
    out = img.copy()
    h, w = out.shape[:2]
    for d in dets:
        if d["conf"] < conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        col = color_for(d["cls"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{d['cls'].title()} {int(d['conf']*100)}%"
        cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + 8 + 7 * len(label), y1), col, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 22, 26), 1, cv2.LINE_AA)
    return out


def render_mosaic(frames, tile_h=200):
    gap = 4
    tiles = []
    for i, (img, dets, name) in enumerate(frames):
        ann = draw_boxes(img, dets)
        th, tw = ann.shape[:2]
        tw2 = max(50, int(tw / th * tile_h))
        tile = cv2.resize(ann, (tw2, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tw2 - 1, tile_h - 1), (71, 48, 27), 1)
        cv2.putText(tile, f"F{i+1}", (6, tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (147, 133, 111), 1, cv2.LINE_AA)
        tiles.append(tile)
    total_w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) + 1)
    header_h = 56
    canvas = np.full((tile_h + header_h + 24, max(total_w, 900), 3), (23, 15, 7), dtype=np.uint8)
    cv2.putText(canvas, "Survey Mosaic view", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 243, 234), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "View Mode: Survey Mosaic  |  consecutive frames with detections overlaid",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (147, 133, 111),
        1,
        cv2.LINE_AA,
    )
    x = gap
    y = header_h
    for tile in tiles:
        canvas[y : y + tile_h, x : x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    return canvas


def render_measure(img, dets):
    ann = draw_boxes(img, dets)
    h, w = ann.shape[:2]
    pad = 50
    canvas = np.full((h + pad * 2 + 50, w + pad * 2, 3), (23, 15, 7), dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = ann
    cv2.putText(canvas, "Measurement tool", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 243, 234), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Click <-> then two points on the sonar image", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (147, 133, 111), 1, cv2.LINE_AA)

    # Ruler bar
    rx0, ry0 = pad, pad - 22
    cv2.line(canvas, (rx0, ry0), (rx0 + w, ry0), (71, 48, 27), 1)
    for i, lbl in enumerate([f"{RANGE} m", f"{RANGE*2//3} m", f"{RANGE//3} m", "0 m", f"{RANGE//3} m", f"{RANGE*2//3} m", f"{RANGE} m"]):
        tx = rx0 + int(w * i / 6)
        cv2.putText(canvas, lbl, (tx, ry0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (147, 133, 111), 1, cv2.LINE_AA)

    x1, y1 = int(w * 0.35), int(h * 0.42)
    x2, y2 = int(w * 0.62), int(h * 0.55)
    ax1, ay1 = pad + x1, pad + y1
    ax2, ay2 = pad + x2, pad + y2
    cv2.line(canvas, (ax1, ay1), (ax2, ay2), (196, 182, 43), 2, cv2.LINE_AA)
    cv2.circle(canvas, (ax1, ay1), 5, (196, 182, 43), -1, cv2.LINE_AA)
    cv2.circle(canvas, (ax2, ay2), 5, (196, 182, 43), -1, cv2.LINE_AA)
    mx = (x2 - x1) / w * (2 * RANGE)
    my = (y2 - y1) / h * RANGE
    dist = float(np.hypot(mx, my))
    mid = ((ax1 + ax2) // 2, (ay1 + ay2) // 2 - 10)
    cv2.rectangle(canvas, (mid[0] - 42, mid[1] - 14), (mid[0] + 42, mid[1] + 4), (40, 26, 11), -1)
    cv2.putText(canvas, f"{dist:.1f} m", (mid[0] - 32, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 243, 234), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Distance: {dist:.1f} m  (range scale {RANGE} m)", (pad, h + pad + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (196, 182, 43), 1, cv2.LINE_AA)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in IMAGES:
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        img, dets = detect(p)
        frames.append((img, dets, p.name))
    mosaic = render_mosaic(frames)
    measure = render_measure(frames[1][0], frames[1][1])
    cv2.imwrite(str(OUT / "survey_mosaic.png"), mosaic)
    cv2.imwrite(str(OUT / "measurement_tool.png"), measure)
    print(OUT / "survey_mosaic.png")
    print(OUT / "measurement_tool.png")


if __name__ == "__main__":
    main()
