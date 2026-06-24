"""
retrain_sonar.py
----------------
Prepare combined synthetic + real UATD sonar data and fine-tune YOLO.

Uses UATD_Test_1 + UATD_OpenSLT from the figshare bundle (21331143.zip)
as additional real side-scan imagery, mapped onto the 8 seabed PoC classes.

    python retrain_sonar.py --prepare-only     # build dataset only
    python retrain_sonar.py --epochs 30        # prepare + train (CPU-friendly)
    python retrain_sonar.py --uatd-zip path/to/21331143.zip

Outputs:
    data/combined/          merged YOLO dataset
    data/combined/seabed.yaml
    runs/detect/sonar_yolo/weights/best.pt   (overwritten after training)
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SEABED_CLASSES = [
    "rock", "boulder", "wreck", "pipeline", "debris", "cable", "scour", "obstruction"
]
CLASS_TO_ID = {c: i for i, c in enumerate(SEABED_CLASSES)}

# Map UATD object names -> seabed PoC vocabulary
UATD_TO_SEABED = {
    "ball": "debris",
    "circle cage": "obstruction",
    "cube": "rock",
    "cylinder": "pipeline",
    "human body": "obstruction",
    "metal bucket": "debris",
    "plane": "wreck",
    "rov": "obstruction",
    "square cage": "obstruction",
    "tyre": "debris",
}

DEFAULT_UATD_ZIP = Path.home() / "Downloads" / "21331143.zip"
COMBINED = ROOT / "data" / "combined"
SYNTH_OUT = ROOT / "dataset_sidescan"
UATD_RAW = ROOT / "data" / "uatd_raw"
UATD_YOLO = ROOT / "data" / "uatd_yolo"
WEIGHTS = ROOT / "runs" / "detect" / "sonar_yolo" / "weights" / "best.pt"


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def extract_nested_zip(outer: Path, member: str, dest: Path) -> Path:
    """Extract one inner zip from the figshare bundle, then unzip it."""
    dest.mkdir(parents=True, exist_ok=True)
    inner_zip = dest / member
    if not inner_zip.exists():
        print(f"Extracting {member} from {outer.name} ...")
        with zipfile.ZipFile(outer, "r") as zf:
            zf.extract(member, dest)
    folder_name = member.replace(".zip", "")
    out_folder = dest / folder_name
    if not out_folder.exists():
        print(f"Unzipping {member} ...")
        with zipfile.ZipFile(inner_zip, "r") as zf:
            zf.extractall(dest)
    if not out_folder.exists():
        # some archives nest one level deeper
        subs = [p for p in dest.iterdir() if p.is_dir() and p.name != folder_name]
        if len(subs) == 1:
            return subs[0]
        raise FileNotFoundError(f"Expected folder {out_folder} after extracting {member}")
    return out_folder


def prepare_uatd(uatd_zip: Path) -> Path | None:
    if not uatd_zip.exists():
        print(f"[skip] UATD bundle not found: {uatd_zip}")
        return None

    train_root = extract_nested_zip(uatd_zip, "UATD_OpenSLT.zip", UATD_RAW)
    val_root = extract_nested_zip(uatd_zip, "UATD_Test_1.zip", UATD_RAW)

    _run([
        sys.executable, "convert_uatd.py",
        "--train-root", str(train_root),
        "--val-root", str(val_root),
        "--out", str(UATD_YOLO),
    ])
    yaml_path = ROOT / "uatd.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError("convert_uatd.py did not write uatd.yaml")
    return yaml_path


def remap_uatd_labels(src_root: Path, dst_root: Path, split: str) -> tuple[int, int]:
    """Copy UATD YOLO images/labels into dst, remapping class ids."""
    from convert_uatd import discover_classes, split_dirs  # noqa: F401

    # rebuild id map from original uatd class order in converted labels
    names_path = UATD_YOLO
    uatd_yaml = ROOT / "uatd.yaml"
    uatd_names: list[str] = []
    if uatd_yaml.exists():
        cfg = yaml.safe_load(uatd_yaml.read_text(encoding="utf-8"))
        names = cfg.get("names") or {}
        if isinstance(names, dict):
            uatd_names = [names[i] for i in sorted(names, key=lambda x: int(x))]
        elif isinstance(names, list):
            uatd_names = names

    if not uatd_names:
        raise RuntimeError("Could not read UATD class names from uatd.yaml")

    uatd_id_to_seabed: dict[int, int] = {}
    for i, name in enumerate(uatd_names):
        mapped = UATD_TO_SEABED.get(name.lower())
        if mapped:
            uatd_id_to_seabed[i] = CLASS_TO_ID[mapped]

    img_in = src_root / "images" / split
    lbl_in = src_root / "labels" / split
    img_out = dst_root / "images" / split
    lbl_out = dst_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for lbl_path in sorted(lbl_in.glob("*.txt")):
        stem = lbl_path.stem
        img_path = img_in / f"{stem}.jpg"
        if not img_path.exists():
            for ext in (".png", ".bmp", ".jpeg"):
                alt = img_in / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            continue

        new_lines = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            old_id = int(parts[0])
            if old_id not in uatd_id_to_seabed:
                continue
            parts[0] = str(uatd_id_to_seabed[old_id])
            new_lines.append(" ".join(parts))
        if not new_lines:
            continue

        out_stem = f"uatd_{stem}"
        shutil.copy2(img_path, img_out / f"{out_stem}{img_path.suffix}")
        (lbl_out / f"{out_stem}.txt").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        n_img += 1
        n_box += len(new_lines)
    return n_img, n_box


def copy_split(src: Path, dst: Path, split: str, prefix: str = "") -> tuple[int, int]:
    img_in = src / "images" / split
    lbl_in = src / "labels" / split
    img_out = dst / "images" / split
    lbl_out = dst / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for lbl in sorted(lbl_in.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            p = img_in / f"{stem}{ext}"
            if p.exists():
                img = p
                break
        if img is None:
            continue
        out_stem = f"{prefix}{stem}" if prefix else stem
        shutil.copy2(img, img_out / f"{out_stem}{img.suffix}")
        shutil.copy2(lbl, lbl_out / f"{out_stem}.txt")
        n_img += 1
        n_box += sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())
    return n_img, n_box


def write_combined_yaml(path: Path) -> None:
  try:
    rel = COMBINED.resolve().relative_to(path.resolve().parent)
    path_str = "./" + rel.as_posix()
  except ValueError:
    path_str = COMBINED.as_posix()
  lines = [
    "# Combined synthetic + UATD sonar — auto-generated by retrain_sonar.py",
    f"path: {path_str}",
    "train: images/train",
    "val: images/val",
    "",
    "names:",
  ]
  for i, c in enumerate(SEABED_CLASSES):
    lines.append(f"  {i}: {c}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(uatd_zip: Path, synth_n: int = 1200) -> Path:
    print("=== 1/3 Synthetic side-scan frames ===")
    if SYNTH_OUT.exists():
        shutil.rmtree(SYNTH_OUT)
    _run([sys.executable, "make_sidescan.py", "--out", str(SYNTH_OUT), "--n", str(synth_n), "--val", "0.15"])

    print("=== 2/3 Real UATD (OpenSLT train + Test_1 val) ===")
    uatd_yaml = prepare_uatd(uatd_zip)

    print("=== 3/3 Merge datasets ===")
    if COMBINED.exists():
        shutil.rmtree(COMBINED)
    COMBINED.mkdir(parents=True)

    tr_s, tb_s = copy_split(SYNTH_OUT, COMBINED, "train", prefix="syn_")
    va_s, vb_s = copy_split(SYNTH_OUT, COMBINED, "val", prefix="syn_")
    tr_u = tb_u = va_u = vb_u = 0
    if uatd_yaml and UATD_YOLO.exists():
        tr_u, tb_u = remap_uatd_labels(UATD_YOLO, COMBINED, "train")
        va_u, vb_u = remap_uatd_labels(UATD_YOLO, COMBINED, "val")

    yaml_path = COMBINED / "seabed.yaml"
    write_combined_yaml(yaml_path)

    print(f"  train: {tr_s + tr_u} images ({tr_s} synthetic + {tr_u} UATD), {tb_s + tb_u} boxes")
    print(f"  val:   {va_s + va_u} images ({va_s} synthetic + {va_u} UATD), {vb_s + vb_u} boxes")
    print(f"  wrote {yaml_path}")
    return yaml_path


def train(yaml_path: Path, epochs: int, batch: int, device: str | None) -> None:
    if WEIGHTS.exists():
        backup = WEIGHTS.with_name("best.pt.bak")
        shutil.copy2(WEIGHTS, backup)
        print(f"Backed up weights -> {backup}")
        model_arg = str(WEIGHTS)
    else:
        model_arg = "yolov8n.pt"

    cmd = [
        sys.executable, "train.py",
        "--model", model_arg,
        "--data", str(yaml_path),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--name", "sonar_yolo",
    ]
    if device:
        cmd.extend(["--device", device])
    _run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Retrain seabed sonar detector on synthetic + UATD data")
    ap.add_argument("--uatd-zip", type=Path, default=DEFAULT_UATD_ZIP,
                    help="Figshare bundle 21331143.zip (default: ~/Downloads/21331143.zip)")
    ap.add_argument("--synth-n", type=int, default=1200, help="Synthetic training frames to generate")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu, 0, or blank for auto")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    yaml_path = prepare_dataset(args.uatd_zip, synth_n=args.synth_n)
    if args.prepare_only:
        print("Dataset ready. Train with:")
        print(f"  python train.py --model {WEIGHTS} --data {yaml_path} --epochs {args.epochs}")
        return

    print("=== Training ===")
    train(yaml_path, epochs=args.epochs, batch=args.batch, device=args.device)
    print("Done. Restart server.py to use the new best.pt")


if __name__ == "__main__":
    main()
