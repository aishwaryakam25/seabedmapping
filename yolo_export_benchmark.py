#!/usr/bin/env python3
"""
YOLO export + quantization + speed/accuracy benchmark
=====================================================
For the NautiCAI seabed-detector. Takes your trained YOLO .pt model and
compares it across deployment formats so you can see the speed/accuracy trade-off:

    - PyTorch (.pt)             baseline
    - ONNX (FP32)
    - ONNX (FP16)               GPU only
    - TensorRT engine (FP16)    GPU + TensorRT
    - TensorRT engine (INT8)    GPU + TensorRT, calibrated on your val set

It then runs a pure-onnxruntime latency demo, including a dynamic INT8
quantization (the easy, CPU/edge-friendly option).

mAP (mAP50 and mAP50-95) is measured on your real validation split via
Ultralytics .val(); inference time comes from the same call. The onnxruntime
block measures raw latency on a dummy input (timing depends on tensor shape,
not image content).

------------------------------------------------------------------
RUN  (Colab / any GPU box — recommended, this is where the speedups show):
    pip install -U ultralytics onnx onnxslim onnxruntime-gpu
    python yolo_export_benchmark.py \\
        --model runs/detect/sonar_yolo/weights/best.pt \\
        --data data/combined/seabed.yaml --imgsz 640

RUN  (CPU only — still shows ONNX + dynamic-INT8 CPU latency):
    pip install -U ultralytics onnx onnxslim onnxruntime
    python yolo_export_benchmark.py \\
        --model runs/detect/sonar_yolo/weights/best.pt \\
        --data data/combined/seabed.yaml --device cpu
------------------------------------------------------------------
Notes:
  * --data must point at the same data.yaml you trained with (needs a 'val' split).
  * INT8 trades a little accuracy for speed — always read its mAP before relying on it.
  * TensorRT/FP16 need a GPU; on CPU those rows are skipped automatically.
"""

import argparse
import csv
import os
import time


def human_mb(path):
    try:
        return os.path.getsize(str(path)) / 1e6
    except OSError:
        return float("nan")


def val_format(YOLO, weights, data, imgsz, device):
    """Validate an exported/native model and return (mAP50, mAP50-95, inference_ms)."""
    m = YOLO(weights)
    metrics = m.val(data=data, imgsz=imgsz, device=device, verbose=False)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    infer_ms = float(metrics.speed.get("inference", float("nan")))
    return map50, map5095, infer_ms


def ort_latency_block(onnx_path, imgsz, runs, on_gpu):
    """Pure onnxruntime latency benchmark + dynamic INT8 quantization (CPU-friendly)."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        print(f"[skip] onnxruntime/numpy not installed: {e}")
        return

    print("\n---- onnxruntime latency (dummy input; shape-only timing) ----")
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    def bench(path, providers, label):
        try:
            sess = ort.InferenceSession(str(path), providers=providers)
            inp = sess.get_inputs()[0].name
            for _ in range(20):  # warmup
                sess.run(None, {inp: dummy})
            t0 = time.perf_counter()
            for _ in range(runs):
                sess.run(None, {inp: dummy})
            ms = (time.perf_counter() - t0) / runs * 1000.0
            print(f"[ort] {label:30s} {ms:7.2f} ms  ({1000.0/ms:6.1f} FPS)  size={human_mb(path):.2f} MB")
        except Exception as e:
            print(f"[ort-skip] {label}: {e}")

    providers = ort.get_available_providers()
    bench(onnx_path, ["CPUExecutionProvider"], "ONNX FP32 (CPU)")
    if on_gpu and "CUDAExecutionProvider" in providers:
        bench(onnx_path, ["CUDAExecutionProvider", "CPUExecutionProvider"], "ONNX FP32 (CUDA)")

    # Dynamic INT8: quantizes weights to int8 with no calibration set needed.
    # Best speedups on CPU / small edge devices.
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = str(onnx_path).replace(".onnx", "_int8dyn.onnx")
        quantize_dynamic(str(onnx_path), int8_path, weight_type=QuantType.QInt8)
        bench(int8_path, ["CPUExecutionProvider"], "ONNX INT8-dynamic (CPU)")
        print(
            "[note] dynamic INT8 shrinks the model and speeds up CPU inference; check its mAP "
            "before relying on it. For maximum GPU/edge speed use the TensorRT INT8 engine above."
        )
    except Exception as e:
        print(f"[ort-skip] dynamic INT8: {e}")


def main():
    ap = argparse.ArgumentParser(description="YOLO export + quantization + speed/accuracy benchmark")
    ap.add_argument("--model", required=True, help="Path to trained YOLO .pt")
    ap.add_argument("--data", required=True, help="Path to dataset data.yaml (must have a val split)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="'cpu', '0', etc.  Default: auto-detect")
    ap.add_argument("--runs", type=int, default=200, help="onnxruntime timing iterations")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    device = args.device if args.device is not None else ("0" if cuda else "cpu")
    on_gpu = str(device) != "cpu"
    print(f"[info] device={device}  cuda_available={cuda}  imgsz={args.imgsz}")

    rows = []  # [format, precision, size_MB, mAP50, mAP50-95, infer_ms, FPS, note]

    def record(fmt, prec, weights, note=""):
        try:
            map50, map5095, ms = val_format(YOLO, weights, args.data, args.imgsz, device)
            fps = 1000.0 / ms if (ms == ms and ms > 0) else float("nan")
            rows.append(
                [
                    fmt,
                    prec,
                    round(human_mb(weights), 2),
                    round(map50, 4),
                    round(map5095, 4),
                    round(ms, 2),
                    round(fps, 1),
                    note,
                ]
            )
            print(
                f"[ok]   {fmt:16s} {prec:5s} mAP50={map50:.3f}  mAP50-95={map5095:.3f}  "
                f"{ms:7.2f} ms ({fps:6.1f} FPS)"
            )
        except Exception as e:
            size = round(human_mb(weights), 2) if os.path.exists(str(weights)) else ""
            rows.append([fmt, prec, size, "", "", "", "", f"FAILED: {e}"])
            print(f"[skip] {fmt} {prec}: {e}")

    # 1) PyTorch baseline
    record("PyTorch", "fp32", args.model, "baseline")

    # 2) ONNX FP32
    onnx_fp32 = None
    try:
        onnx_fp32 = YOLO(args.model).export(
            format="onnx", imgsz=args.imgsz, simplify=True, opset=12, dynamic=False
        )
        record("ONNX", "fp32", onnx_fp32)
    except Exception as e:
        print(f"[skip] ONNX fp32 export: {e}")

    # 3) ONNX FP16 (GPU benefit)
    if on_gpu:
        try:
            onnx_fp16 = YOLO(args.model).export(
                format="onnx", imgsz=args.imgsz, half=True, simplify=True, opset=12
            )
            record("ONNX", "fp16", onnx_fp16, "GPU")
        except Exception as e:
            print(f"[skip] ONNX fp16 export: {e}")
    else:
        print("[info] skipping ONNX-FP16 (needs GPU)")

    # 4 + 5) TensorRT FP16 and INT8
    if on_gpu:
        try:
            trt_fp16 = YOLO(args.model).export(format="engine", imgsz=args.imgsz, half=True)
            record("TensorRT", "fp16", trt_fp16, "GPU+TRT")
        except Exception as e:
            print(f"[skip] TensorRT fp16: {e}")
        try:
            trt_int8 = YOLO(args.model).export(
                format="engine", imgsz=args.imgsz, int8=True, data=args.data
            )
            record("TensorRT", "int8", trt_int8, "GPU+TRT, calibrated")
        except Exception as e:
            print(f"[skip] TensorRT int8: {e}")
    else:
        print("[info] skipping TensorRT FP16/INT8 (needs GPU + TensorRT)")

    # onnxruntime latency demo (incl. dynamic INT8)
    if onnx_fp32 and os.path.exists(str(onnx_fp32)):
        ort_latency_block(onnx_fp32, args.imgsz, args.runs, on_gpu)

    # comparison table + CSV
    print("\n==================== COMPARISON ====================")
    hdr = ["format", "precision", "size_MB", "mAP50", "mAP50-95", "infer_ms", "FPS", "note"]
    widths = [10, 9, 8, 7, 9, 9, 7, 26]
    print("  ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)
    print(f"\n[saved] {args.out}")
    print(
        "[tip] On-vessel/edge: deploy the TensorRT INT8 engine (Jetson) or the dynamic-INT8 "
        "ONNX (CPU). Keep the format whose mAP50 stays closest to the PyTorch baseline."
    )


if __name__ == "__main__":
    main()
