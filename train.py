"""
train.py
--------
Train YOLO on the sonar dataset.

    pip install ultralytics          # one-time
    python train.py                  # uses sonar.yaml, GPU auto-detected
    python make_sidescan.py --out dataset_sidescan --n 1600 --val 0.15
    python train.py --data dataset_sidescan/seabed.yaml --epochs 60
    python train.py --device cpu --epochs 5    # quick smoke test on CPU

The first run downloads the pretrained weights (yolov8n.pt) once -- needs
internet. Pass --model yolov8n.yaml to build from scratch with no download
(only sensible if you have a lot of data).

On the synthetic dataset this just proves the loop runs. Use a real dataset for
results -- see README.md.
"""
import argparse
import sys

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt",
                    help="pretrained .pt, or a .yaml to train from scratch")
    ap.add_argument("--data", default="sonar.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="'0' for first GPU, 'cpu', or leave blank to auto-detect")
    ap.add_argument("--name", default="sonar_yolo",
                    help="run folder under runs/detect/ (weights -> runs/detect/<name>/weights/best.pt)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0 if sys.platform == "win32" else 8,
        name=args.name,
    )
    # quick validation summary on the val split
    metrics = model.val()
    print("mAP50-95:", metrics.box.map, " mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()
