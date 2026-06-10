# NautiCAI Seabed Mapping

Side-scan sonar object detection PoC — YOLO backend, multi-page web UI, PDF reports, and 3D seabed digital twin.

## Classes

`rock`, `boulder`, `wreck`, `pipeline`, `debris`, `cable`, `scour`, `obstruction`

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place your trained weights at:

```
runs/detect/sonar_yolo/weights/best.pt
```

Then run:

```bash
python server.py
```

Open **http://127.0.0.1:8000/review**

## Training

- Generate synthetic frames: `python make_sidescan.py --out test_frames --n 30`
- Train locally: `python train.py`
- Or use `sidescan_train_colab.ipynb` on Google Colab

Dataset config: `seabed.yaml`

## Pages

| Route | Description |
|-------|-------------|
| `/review` | Data Review — load frames, run detection |
| `/dashboard` | Session summary |
| `/map` | Survey map |
| `/detections` | Detection list |
| `/twin` | 3D digital twin |
| `/reports` | PDF reports |
| `/settings` | Preferences |
