"""Quick batch detection on synthetic test frames."""
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

m = YOLO("runs/detect/sonar_yolo/weights/best.pt")
imgs = sorted(Path("test_frames/images/train").glob("*.png"))[:10]
all_cls = Counter()
for p in imgs:
    r = m.predict(str(p), conf=0.25, verbose=False)[0]
    cls = [m.names[int(b.cls)] for b in r.boxes]
    all_cls.update(cls)
    label = ", ".join(cls) if cls else "(no detections)"
    print(f"{p.name}: {label}")
print("\nTotals:", dict(sorted(all_cls.items())))
print("Classes seen:", sorted(all_cls.keys()))
