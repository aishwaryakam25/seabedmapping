"""
visualize_labels.py
-------------------
Draw the YOLO boxes onto a few training images and save a preview grid, so you
can eyeball that the labels line up with the objects. Uses only opencv + numpy.
"""
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CLASSES = ["mine", "wreck", "rock"]
COLORS = [(0, 0, 255), (0, 255, 0), (255, 128, 0)]   # BGR per class


def draw(img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = DATA / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            c = int(c)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], 2)
            cv2.putText(img, CLASSES[c], (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    imgs = sorted((DATA / "images" / "train").glob("*.jpg"))
    random.seed(1)
    pick = random.sample(imgs, min(6, len(imgs)))
    tiles = [cv2.resize(draw(p), (320, 320)) for p in pick]
    while len(tiles) < 6:                       # pad if very few images
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    out = ROOT / "preview_labels.jpg"
    cv2.imwrite(str(out), grid)
    print("Saved", out.name)
