"""
make_sidescan.py
----------------
Generate synthetic SIDE-SCAN-SONAR-style images + YOLO labels for a
seabed-object detection proof of concept.

Classes (8):
    0 rock        1 boulder     2 wreck       3 pipeline
    4 debris      5 cable       6 scour       7 obstruction

Why synthetic: public side-scan datasets cover wrecks / mines / aircraft well,
but not rock / boulder / pipeline. Synthetic data lets a PoC demonstrate the
exact target classes. Each object is rendered with the characteristic side-scan
look: a bright acoustic HIGHLIGHT (sonar-facing side) plus a dark acoustic
SHADOW cast away from the sonar track (the nadir line down the image centre).

Output layout (Ultralytics YOLO format):
    <out>/images/train/*.png   <out>/labels/train/*.txt
    <out>/images/val/*.png     <out>/labels/val/*.txt

Usage:
    python make_sidescan.py --out dataset_sidescan --n 800 --val 0.15
"""
import os, argparse, random
import numpy as np
import cv2

CLASSES = ["rock", "boulder", "wreck", "pipeline", "debris", "cable", "scour", "obstruction"]
W = H = 640
NADIR_X = W // 2          # sonar track runs down the image centre
NADIR_HALF = 16           # half-width of the water-column band


def seabed_background(rng):
    """Granular seabed backscatter with low-frequency tonal variation + along-track striping."""
    base = rng.integers(78, 120)
    img = np.full((H, W), base, np.float32)
    # low-frequency tone (blurred noise)
    lf = rng.normal(0, 1, (H // 16, W // 16)).astype(np.float32)
    lf = cv2.resize(lf, (W, H), interpolation=cv2.INTER_CUBIC)
    img += lf * 22
    # fine speckle (sonar backscatter)
    img += rng.normal(0, 16, (H, W)).astype(np.float32)
    # subtle along-track (horizontal) striping
    stripe = np.sin(np.linspace(0, rng.uniform(20, 45), H))[:, None] * rng.uniform(3, 7)
    img += stripe
    # water column / nadir band darker, with bright first-return edges
    img[:, NADIR_X - NADIR_HALF:NADIR_X + NADIR_HALF] *= 0.45
    img[:, NADIR_X - NADIR_HALF - 1:NADIR_X - NADIR_HALF + 1] = 210
    img[:, NADIR_X + NADIR_HALF - 1:NADIR_X + NADIR_HALF + 1] = 210
    return img


def shadow_dir(cx):
    """Acoustic shadow falls AWAY from the nadir (track) line."""
    return -1 if cx < NADIR_X else 1


def add_highlight(img, mask_pts, bright):
    cv2.fillPoly(img, [mask_pts], float(bright))


def clamp_box(x1, y1, x2, y2):
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def draw_rock(img, rng):
    cx = rng.integers(NADIR_X + NADIR_HALF + 30, W - 30) if rng.random() < 0.5 \
        else rng.integers(30, NADIR_X - NADIR_HALF - 30)
    cy = rng.integers(40, H - 40)
    r = int(rng.integers(6, 14))
    sd = shadow_dir(cx)
    # shadow
    slen = int(r * rng.uniform(1.6, 2.8))
    cv2.ellipse(img, (int(cx + sd * (r + slen // 2)), cy), (slen, max(3, r // 2)), 0, 0, 360, 28, -1)
    # highlight
    cv2.circle(img, (cx, cy), r, int(rng.integers(200, 255)), -1)
    return clamp_box(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2)


def draw_boulder(img, rng):
    cx = rng.integers(NADIR_X + NADIR_HALF + 50, W - 50) if rng.random() < 0.5 \
        else rng.integers(50, NADIR_X - NADIR_HALF - 50)
    cy = rng.integers(60, H - 60)
    r = int(rng.integers(18, 34))
    sd = shadow_dir(cx)
    slen = int(r * rng.uniform(1.8, 3.0))
    cv2.ellipse(img, (int(cx + sd * (r + slen // 2)), cy), (slen, int(r * 0.7)), 0, 0, 360, 24, -1)
    # irregular bright blob
    pts = []
    n = 10
    for i in range(n):
        a = 2 * np.pi * i / n
        rr = r * rng.uniform(0.8, 1.15)
        pts.append([int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a))])
    cv2.fillPoly(img, [np.array(pts, np.int32)], int(rng.integers(205, 255)))
    return clamp_box(cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3)


def draw_wreck(img, rng):
    cx = rng.integers(NADIR_X + NADIR_HALF + 90, W - 90) if rng.random() < 0.5 \
        else rng.integers(90, NADIR_X - NADIR_HALF - 90)
    cy = rng.integers(110, H - 110)
    L = int(rng.integers(70, 150)); Wd = int(rng.integers(16, 30))
    ang = rng.uniform(-0.5, 0.5)
    sd = shadow_dir(cx)
    box = cv2.boxPoints(((float(cx), float(cy)), (float(L), float(Wd)), float(np.degrees(ang)))).astype(np.int32)
    # long acoustic shadow
    shp = box.copy().astype(np.float32)
    shadow = shp + np.array([sd * rng.uniform(28, 60), 0], np.float32)
    cv2.fillPoly(img, [shadow.astype(np.int32)], 22)
    # bright hull with internal structure
    cv2.fillPoly(img, [box], int(rng.integers(200, 250)))
    for _ in range(rng.integers(3, 6)):
        px = int(cx + rng.uniform(-L / 2.4, L / 2.4) * np.cos(ang))
        py = int(cy + rng.uniform(-L / 2.4, L / 2.4) * np.sin(ang))
        cv2.circle(img, (px, py), int(rng.integers(2, 5)), int(rng.integers(120, 180)), -1)
    xs = box[:, 0]; ys = box[:, 1]
    return clamp_box(int(xs.min()) - 3, int(ys.min()) - 3, int(xs.max()) + 3, int(ys.max()) + 3)


def draw_pipeline(img, rng):
    # long bright near-linear feature crossing the seabed (one side of nadir)
    side = 1 if rng.random() < 0.5 else -1
    if side > 0:
        x1 = rng.integers(NADIR_X + NADIR_HALF + 10, W - 60)
    else:
        x1 = rng.integers(20, NADIR_X - NADIR_HALF - 80)
    y1 = rng.integers(20, H - 20)
    ang = rng.uniform(-0.6, 0.6) + (np.pi / 2 if rng.random() < 0.5 else 0)
    length = int(rng.integers(220, 420))
    x2 = int(x1 + length * np.cos(ang)); y2 = int(y1 + length * np.sin(ang))
    th = int(rng.integers(4, 8))
    sd = shadow_dir((x1 + x2) // 2)
    cv2.line(img, (x1 + sd * 6, y1), (x2 + sd * 6, y2), 26, th + 4)   # shadow
    cv2.line(img, (x1, y1), (x2, y2), int(rng.integers(205, 255)), th)  # bright pipe
    return clamp_box(min(x1, x2) - th, min(y1, y2) - th, max(x1, x2) + th, max(y1, y2) + th)


def draw_debris(img, rng):
    cx = rng.integers(NADIR_X + NADIR_HALF + 40, W - 50) if rng.random() < 0.5 \
        else rng.integers(50, NADIR_X - NADIR_HALF - 40)
    cy = rng.integers(50, H - 50)
    sd = shadow_dir(cx)
    spread = int(rng.integers(18, 38))
    xs, ys = [], []
    for _ in range(rng.integers(3, 6)):
        px = int(cx + rng.uniform(-spread, spread)); py = int(cy + rng.uniform(-spread, spread))
        rr = int(rng.integers(2, 6))
        cv2.ellipse(img, (px + sd * (rr + 4), py), (rr * 2, rr), 0, 0, 360, 26, -1)  # shadow
        cv2.circle(img, (px, py), rr, int(rng.integers(195, 255)), -1)
        xs += [px - rr, px + rr]; ys += [py - rr, py + rr]
    return clamp_box(min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3)


def draw_cable(img, rng):
    """Thin, sinuous, fainter than a pipeline; minimal shadow."""
    side = 1 if rng.random() < 0.5 else -1
    if side > 0:
        x1 = rng.integers(NADIR_X + NADIR_HALF + 10, W - 40)
    else:
        x1 = rng.integers(20, NADIR_X - NADIR_HALF - 60)
    y1 = rng.integers(20, H - 20)
    ang = rng.uniform(-0.5, 0.5) + (np.pi / 2 if rng.random() < 0.5 else 0)
    seg = int(rng.integers(40, 70)); pts = [[int(x1), int(y1)]]; cur = ang
    for _ in range(int(rng.integers(5, 8))):
        cur += rng.uniform(-0.4, 0.4)
        pts.append([int(pts[-1][0] + seg * np.cos(cur)), int(pts[-1][1] + seg * np.sin(cur))])
    pts = np.array(pts, np.int32); th = int(rng.integers(2, 4))
    sd = shadow_dir(int(pts[:, 0].mean()))
    cv2.polylines(img, [pts + np.array([sd * 3, 0])], False, 28, th + 2)   # faint shadow
    cv2.polylines(img, [pts], False, int(rng.integers(185, 230)), th)      # thin bright cable
    xs = pts[:, 0]; ys = pts[:, 1]
    return clamp_box(int(xs.min()) - th, int(ys.min()) - th, int(xs.max()) + th, int(ys.max()) + th)


def draw_scour(img, rng):
    """Scour / erosion: a DARK depression (low backscatter) with a bright rim on the sonar-facing edge."""
    cx = rng.integers(NADIR_X + NADIR_HALF + 55, W - 55) if rng.random() < 0.5 \
        else rng.integers(55, NADIR_X - NADIR_HALF - 55)
    cy = rng.integers(60, H - 60)
    a = int(rng.integers(26, 50)); b = int(rng.integers(14, 28)); ang = float(rng.uniform(0, 180))
    sd = shadow_dir(cx)
    cv2.ellipse(img, (int(cx), int(cy)), (a, b), ang, 0, 360, int(rng.integers(38, 60)), -1)   # dark hollow
    cv2.ellipse(img, (int(cx - sd * int(a * 0.5)), int(cy)), (int(a * 0.6), int(b * 0.7)), ang, 0, 360, int(rng.integers(150, 190)), 3)  # bright rim
    return clamp_box(cx - a - 3, cy - b - 3, cx + a + 3, cy + b + 3)


def draw_obstruction(img, rng):
    """Generic hard raised object: angular bright return + a strong, long acoustic shadow."""
    cx = rng.integers(NADIR_X + NADIR_HALF + 55, W - 60) if rng.random() < 0.5 \
        else rng.integers(60, NADIR_X - NADIR_HALF - 55)
    cy = rng.integers(70, H - 70)
    r = int(rng.integers(20, 40)); sd = shadow_dir(cx)
    slen = int(r * rng.uniform(2.2, 3.6))
    cv2.ellipse(img, (int(cx + sd * (r + slen // 2)), cy), (slen, int(r * 0.8)), 0, 0, 360, 18, -1)  # strong shadow
    pts = []; n = int(rng.integers(5, 8))
    for i in range(n):
        aa = 2 * np.pi * i / n; rr = r * rng.uniform(0.6, 1.2)
        pts.append([int(cx + rr * np.cos(aa)), int(cy + rr * np.sin(aa))])
    cv2.fillPoly(img, [np.array(pts, np.int32)], int(rng.integers(210, 255)))
    return clamp_box(cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3)


DRAW = [draw_rock, draw_boulder, draw_wreck, draw_pipeline, draw_debris, draw_cable, draw_scour, draw_obstruction]


def make_image(rng):
    img = seabed_background(rng)
    labels = []
    n_obj = rng.integers(1, 5)
    for _ in range(n_obj):
        cls = int(rng.integers(0, len(CLASSES)))
        x1, y1, x2, y2 = DRAW[cls](img, rng)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cxn = ((x1 + x2) / 2) / W; cyn = ((y1 + y2) / 2) / H
        wn = (x2 - x1) / W; hn = (y2 - y1) / H
        labels.append((cls, cxn, cyn, wn, hn))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # soften
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_sidescan")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    n_val = int(a.n * a.val)
    for i in range(a.n):
        split = "val" if i < n_val else "train"
        img, labels = make_image(rng)
        name = f"ss_{i:05d}"
        cv2.imwrite(os.path.join(a.out, "images", split, name + ".png"), img)
        with open(os.path.join(a.out, "labels", split, name + ".txt"), "w") as f:
            for cls, cx, cy, w, h in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    # dataset yaml
    with open(os.path.join(a.out, "seabed.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\n")
        f.write("train: images/train\nval: images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write("names: [" + ", ".join(CLASSES) + "]\n")
    print(f"Wrote {a.n} images to {a.out} ({n_val} val). Classes: {CLASSES}")


if __name__ == "__main__":
    main()
