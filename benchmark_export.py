"""
benchmark_export.py — export, quantize, and benchmark the sonar YOLO model.

Exports the trained weights to ONNX (FP32 / FP16 / INT8), measures inference
latency on real val frames, and compares mAP against the PyTorch checkpoint.

    pip install onnx onnxruntime          # INT8 export + ORT inference
    pip install onnxruntime-gpu           # optional GPU ORT

    python benchmark_export.py
    python benchmark_export.py --quick      # 10 images, skip INT8
    python benchmark_export.py --compare runs/detect/other/weights/best.pt

On NVIDIA Jetson / Linux with TensorRT:
    python benchmark_export.py --tensorrt   # also builds .engine (FP16/INT8)

Outputs under runs/export/sonar_yolo/ and benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "sonar_yolo" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "data" / "combined" / "seabed.yaml"
EXPORT_DIR = ROOT / "runs" / "export" / "sonar_yolo"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _val_images(data_yaml: Path, limit: int | None) -> list[Path]:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_rel = cfg.get("val", "images/val")
    val_dir = (root / val_rel).resolve()
    imgs = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No val images in {val_dir}")
    return imgs[:limit] if limit else imgs


def _export_variant(model, name: str, out_dir: Path, data_yaml: Path, **kwargs) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = model.export(project=str(out_dir), name=name, exist_ok=True, **kwargs)
        return Path(path)
    except Exception as exc:
        print(f"[skip] {name} export failed: {exc}")
        return None


def export_all(weights: Path, data_yaml: Path, imgsz: int, device: str, int8: bool, tensorrt: bool) -> dict[str, Path]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    stem = weights.stem
    out: dict[str, Path] = {"pytorch": weights}

    onnx_fp32 = _export_variant(
        model, f"{stem}_fp32", EXPORT_DIR,
        data_yaml, format="onnx", imgsz=imgsz, device=device, simplify=True, dynamic=False,
    )
    if onnx_fp32:
        out["onnx_fp32"] = onnx_fp32

    onnx_fp16 = _export_variant(
        model, f"{stem}_fp16", EXPORT_DIR,
        data_yaml, format="onnx", imgsz=imgsz, device=device, half=True, simplify=True, dynamic=False,
    )
    if onnx_fp16:
        out["onnx_fp16"] = onnx_fp16

    if int8:
        onnx_int8 = _export_variant(
            model, f"{stem}_int8", EXPORT_DIR,
            data_yaml, format="onnx", imgsz=imgsz, device=device, int8=True,
            data=str(data_yaml), simplify=True, dynamic=False,
        )
        if onnx_int8:
            out["onnx_int8"] = onnx_int8

    if tensorrt and sys.platform != "win32":
        for label, half, quant_int8 in [("trt_fp16", True, False), ("trt_int8", False, True)]:
            trt = _export_variant(
                model, f"{stem}_{label}", EXPORT_DIR,
                data_yaml, format="engine", imgsz=imgsz, device=device,
                half=half, int8=quant_int8, data=str(data_yaml), simplify=True,
            )
            if trt:
                out[label] = trt
    elif tensorrt:
        print("[skip] TensorRT export is Linux/Jetson only (skipped on Windows)")

    return out


def benchmark_speed(model_path: Path, images: list[Path], warmup: int = 5, device: str | None = None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kw = {"verbose": False}
    if device:
        kw["device"] = device

    for im in images[:warmup]:
        model.predict(str(im), **kw)

    times_ms = []
    for im in images:
        t0 = time.perf_counter()
        model.predict(str(im), **kw)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    return {
        "n_images": len(times_ms),
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(times_ms[int(len(times_ms) * 0.95) - 1], 2),
        "fps": round(1000.0 / statistics.mean(times_ms), 2),
    }


def evaluate_map(model_path: Path, data_yaml: Path, imgsz: int, device: str | None) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    kw = {"data": str(data_yaml), "imgsz": imgsz, "verbose": False, "plots": False}
    if device:
        kw["device"] = device
    metrics = model.val(**kw)
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def run_benchmark(
    artifacts: dict[str, Path],
    data_yaml: Path,
    imgsz: int,
    bench_n: int,
    val_map: bool,
    device: str | None,
) -> list[dict]:
    images = _val_images(data_yaml, bench_n)
    rows = []
    for label, path in artifacts.items():
        print(f"\n=== {label} ({path.name}) ===")
        row = {"variant": label, "path": str(path)}
        try:
            row["speed"] = benchmark_speed(path, images, device=device)
            print(
                f"  speed: {row['speed']['mean_ms']} ms/img  "
                f"({row['speed']['fps']} FPS)  p95={row['speed']['p95_ms']} ms"
            )
        except Exception as exc:
            row["speed_error"] = str(exc)
            print(f"  speed failed: {exc}")
        if val_map:
            try:
                row["metrics"] = evaluate_map(path, data_yaml, imgsz, device)
                m = row["metrics"]
                print(f"  mAP50={m['map50']}  mAP50-95={m['map50_95']}  P={m['precision']}  R={m['recall']}")
            except Exception as exc:
                row["metrics_error"] = str(exc)
                print(f"  val failed: {exc}")
        rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'Variant':<14} {'ms/img':>8} {'FPS':>7} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 72)
    for r in rows:
        sp = r.get("speed") or {}
        mt = r.get("metrics") or {}
        print(
            f"{r['variant']:<14} "
            f"{sp.get('mean_ms', '—'):>8} "
            f"{sp.get('fps', '—'):>7} "
            f"{mt.get('map50', '—'):>8} "
            f"{mt.get('map50_95', '—'):>10}"
        )
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Export, quantize, and benchmark sonar YOLO")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'cpu', '0' for GPU, or auto")
    ap.add_argument("--bench-images", type=int, default=50, help="Val frames for latency test")
    ap.add_argument("--quick", action="store_true", help="10 images, skip INT8 export")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--skip-export", action="store_true", help="Benchmark existing exports only")
    ap.add_argument("--no-map", action="store_true", help="Skip mAP validation (faster)")
    ap.add_argument("--tensorrt", action="store_true", help="Also export TensorRT engine (Linux/Jetson)")
    ap.add_argument("--compare", type=Path, default=None, help="Extra .pt weights to benchmark (e.g. YOLO11n run)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise SystemExit(f"Dataset yaml not found: {args.data}")

    bench_n = 10 if args.quick else args.bench_images
    do_int8 = not (args.skip_int8 or args.quick)

    if args.skip_export:
        artifacts = {"pytorch": args.weights}
        for p in sorted(EXPORT_DIR.glob("*.onnx")) + sorted(EXPORT_DIR.glob("*.engine")):
            if "_fp32" in p.stem:
                artifacts["onnx_fp32"] = p
            elif "_fp16" in p.stem:
                artifacts["onnx_fp16"] = p
            elif "_int8" in p.stem:
                artifacts["onnx_int8"] = p
            elif "_trt_" in p.stem:
                artifacts[p.stem.split("_")[-2] + "_" + p.stem.split("_")[-1]] = p
    else:
        print(f"Exporting from {args.weights} -> {EXPORT_DIR}")
        artifacts = export_all(args.weights, args.data, args.imgsz, args.device or "cpu", do_int8, args.tensorrt)

    if args.compare and args.compare.exists():
        artifacts["compare_pt"] = args.compare

    rows = run_benchmark(artifacts, args.data, args.imgsz, bench_n, not args.no_map, args.device)
    print_summary(rows)

    out_json = EXPORT_DIR / "benchmark_results.json"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "bench_images": bench_n,
        "results": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
