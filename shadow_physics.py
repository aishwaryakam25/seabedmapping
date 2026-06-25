"""
Acoustic-shadow physics — measure the real shadow behind a side-scan contact
and estimate its height above the seabed.

Unlike the box-aspect heuristic in seabed_intelligence.js (`"illustrative physics"`),
this reads the ACTUAL image pixels: the acoustic shadow is the dark run that falls
*outward from nadir* immediately behind a proud object. We measure that run's length
in pixels, convert to metres via the across-track range scale, and derive height from
flat-bottom side-scan geometry:

    h = H * L / (R + L)

where H = sonar (fish) altitude above seabed, L = shadow length (ground range),
R = ground range from nadir to the object. h is bounded by H, as physics requires.

H is the one assumption we cannot read from a single frame, so it is an explicit
parameter (default scaled from the range setting) and is surfaced in the UI. The
measured shadow length is the genuine, defensible quantity; height follows from it.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def default_fish_altitude(range_scale: float) -> float:
    """Typical side-scan survey altitude ≈ 10–12% of slant range."""
    return max(4.0, round(0.12 * float(range_scale), 1))


def _profile(gray: np.ndarray, y1: int, y2: int) -> np.ndarray:
    """Mean intensity across the object's vertical extent → 1D across-track profile."""
    h = gray.shape[0]
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(y1 + 1, min(h, int(y2)))
    band = gray[y1:y2, :].astype(np.float32)
    prof = band.mean(axis=0)
    # light smoothing to suppress speckle without erasing the shadow edge
    k = np.ones(3, dtype=np.float32) / 3.0
    return np.convolve(prof, k, mode="same")


def measure_shadow(
    gray: np.ndarray,
    box: list[int],
    range_scale: float = 150.0,
    nadir_x_frac: float = 0.5,
    fish_altitude_m: float | None = None,
    gap_tolerance: int = 3,
) -> dict[str, Any]:
    """
    Measure the acoustic shadow behind one contact.

    gray: single-channel image (H x W), 0–255.
    box:  [x1, y1, x2, y2] in image pixels.
    Returns measured shadow geometry + a height estimate, all honestly labelled.
    """
    H_img, W_img = gray.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    nadir_x = nadir_x_frac * W_img
    direction = 1 if cx >= nadir_x else -1          # shadow falls away from nadir
    far_x = int(round(x2 if direction > 0 else x1)) # object edge the shadow starts behind

    prof = _profile(gray, y1, y2)
    obj_slice = prof[int(max(0, x1)):int(min(W_img, x2)) + 1]
    obj_level = float(np.median(obj_slice)) if obj_slice.size else float(prof.mean())
    bg_level = float(np.median(prof))
    # A shadow is meaningfully darker than the surrounding seabed return.
    shadow_thresh = bg_level * 0.55

    # Walk outward from the object's far edge, counting the dark run.
    max_walk = int(min(W_img, abs(W_img * 0.6)))
    px = far_x + direction
    length = 0
    gaps = 0
    last_dark = 0
    while 0 <= px < W_img and length < max_walk:
        if prof[px] < shadow_thresh:
            length += 1
            last_dark = length
            gaps = 0
        else:
            gaps += 1
            if gaps > gap_tolerance:
                break
            length += 1
        px += direction
    shadow_len_px = last_dark  # trim trailing recovered pixels

    # Across-track metres per pixel: full image width spans port+starboard (2 × range).
    m_per_px = (2.0 * float(range_scale)) / max(1.0, W_img)
    shadow_len_m = round(shadow_len_px * m_per_px, 2)
    range_m = round(abs(cx - nadir_x) * m_per_px, 1)

    H = default_fish_altitude(range_scale) if fish_altitude_m is None else float(fish_altitude_m)
    if shadow_len_m > 0 and (range_m + shadow_len_m) > 0:
        height_m = round(H * shadow_len_m / (range_m + shadow_len_m), 2)
    else:
        height_m = 0.0

    if shadow_len_px < 2 or height_m < 0.3:
        burial, burial_label = "buried", "Likely buried / flush"
    elif height_m < 0.9:
        burial, burial_label = "partially_buried", "Partially buried"
    else:
        burial, burial_label = "proud", "Proud on seabed"

    # Shadow rectangle in image coords for the overlay (same vertical band as object).
    sx1 = far_x
    sx2 = far_x + direction * shadow_len_px
    shadow_box = [int(min(sx1, sx2)), int(y1), int(max(sx1, sx2)), int(y2)]

    contrast = round((bg_level - obj_level) / max(1.0, bg_level), 2)
    measured = shadow_len_px >= 2
    return {
        "measured": measured,
        "shadow_len_px": int(shadow_len_px),
        "shadow_len_m": shadow_len_m,
        "shadow_box": shadow_box,
        "shadow_dir": int(direction),
        "range_m": range_m,
        "fish_altitude_m": round(H, 1),
        "height_m": height_m,
        "burial": burial,
        "burial_label": burial_label,
        "object_contrast": contrast,
        "method": "pixel-measured shadow (flat-bottom geometry)" if measured
                  else "no clear shadow — likely buried/flush",
        "assumption": f"fish altitude H = {round(H, 1)} m (h = H·L/(R+L))",
    }


def annotate_detections(
    gray: np.ndarray,
    detections: list[dict],
    range_scale: float = 150.0,
    nadir_x_frac: float = 0.5,
    fish_altitude_m: float | None = None,
) -> list[dict]:
    """Attach a `shadow` block to each detection in place; returns the list."""
    for det in detections:
        box = det.get("box")
        if not box or len(box) < 4:
            continue
        det["shadow"] = measure_shadow(
            gray, box, range_scale=range_scale,
            nadir_x_frac=nadir_x_frac, fish_altitude_m=fish_altitude_m,
        )
    return detections
