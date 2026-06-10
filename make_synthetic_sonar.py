"""
make_synthetic_sonar.py
------------------------
Generate a small synthetic side-scan-sonar-style OBJECT DETECTION dataset in
YOLO format, using only numpy + opencv (no downloads, no torch).

WHY THIS EXISTS:
It lets you run the whole train -> detect loop end to end before any real data is
in place. The images are crude approximations of side-scan sonar (speckled
seabed + bright object highlights + dark acoustic shadows). Do NOT read anything
into the accuracy numbers from this data -- swap in a real dataset
(UATD / SWDD / KLSG-II / AI4Shipwrecks) for meaningful results. See README.md.
"""
import random
from pathlib import Path

import cv2
import numpy as np

# ---- config ------------------------------------------------------------
SEED = 0
IMG_SIZE = 640
N_TRAIN = 80
N_VAL = 20
MAX_TARGETS = 4
CLASSES = ["mine", "wreck", "rock"]          # class ids 0, 1, 2
# ------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def seabed_background(size):
    """Range-faded, speckled, lightly-rippled seabed (sonar-ish texture)."""
    h = w = size
    # brighter near the sensor track (top), fading with range (down)
    grad = np.linspace(1.15, 0.70, h)[:, None] * np.ones((1, w))
    yy, xx = np.mgrid[0:h, 0:w]
    ripple = 0.06 * np.sin(2 * np.pi * (xx / 40.0) + np.random.uniform(0, 2 * np.pi))
    ripple += 0.04 * np.sin(2 * np.pi * (yy / 70.0) + np.random.uniform(0, 2 * np.pi))
    base = 90.0 * grad * (1.0 + ripple)
    # multiplicative Rayleigh speckle is characteristic of sonar imagery
    speckle = np.random.rayleigh(scale=1.0, size=(h, w))
    speckle = speckle / speckle.mean()
    img = base * (0.6 + 0.4 * speckle)
    return np.clip(img, 0, 255)


def add_target(img, cls):
    """Render one object (highlight + down-range shadow) and return its YOLO box."""
    h, w = img.shape
    if cls == 0:        # mine: small, round
        a, b = random.randint(6, 12), random.randint(6, 12)
    elif cls == 1:      # wreck: large, elongated
        a, b = random.randint(28, 55), random.randint(8, 16)
    else:               # rock: medium
        a, b = random.randint(12, 22), random.randint(10, 18)
    angle = random.uniform(0, 180)
    cx = random.randint(int(0.08 * w), int(0.92 * w))
    cy = random.randint(int(0.10 * h), int(0.85 * h))

    # acoustic shadow: a dark region down-range (increasing y) from the object
    shadow = np.zeros_like(img)
    shadow_len = int(max(a, b) * random.uniform(1.5, 3.0))
    cv2.ellipse(shadow, (cx, cy + shadow_len // 2), (int(a * 0.9), shadow_len),
                angle, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 4)
    img = img * (1.0 - 0.7 * np.clip(shadow, 0, 1))

    # bright highlight (the object echo)
    hl = np.zeros_like(img)
    cv2.ellipse(hl, (cx, cy), (a, b), angle, 0, 360, 1.0, -1)
    hl = cv2.GaussianBlur(hl, (0, 0), 1.5)
    img = np.clip(img + 150.0 * np.clip(hl, 0, 1), 0, 255)

    # exact bbox from the highlight footprint
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (cx, cy), (a + 2, b + 2), angle, 0, 360, 255, -1)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return img, (cls, (bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h)


def make_split(name, n):
    img_dir = DATA / "images" / name
    lbl_dir = DATA / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = seabed_background(IMG_SIZE)
        labels = []
        for _ in range(random.randint(1, MAX_TARGETS)):
            cls = random.randint(0, len(CLASSES) - 1)
            img, lab = add_target(img, cls)
            labels.append(lab)
        out3 = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        stem = f"{name}_{i:04d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), out3)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for cls, cx, cy, bw, bh in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"  {name}: {n} images -> {img_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    print("Generating synthetic sonar dataset (validation only)...")
    make_split("train", N_TRAIN)
    make_split("val", N_VAL)
    print("Done. Classes:", CLASSES)
