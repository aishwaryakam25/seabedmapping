"""
detect.py
---------
Run a trained model on images / a folder and save annotated results.

    pip install ultralytics
    python detect.py --source data/images/val
    python detect.py --weights runs/detect/sonar_yolo/weights/best.pt \
                     --source path/to/some_sonar_image.png --conf 0.3
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/sonar_yolo/weights/best.pt")
    ap.add_argument("--source", required=True, help="image, folder, or glob")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, conf=args.conf, device=args.device, save=True)
    print("Annotated results saved under runs/detect/predict*/")


if __name__ == "__main__":
    main()
