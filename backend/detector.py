"""
Crop / weed detection engine.

Pipeline
--------
1. Vegetation segmentation  - Excess-Green index (ExG = 2g - r - b) with an adaptive
                              (Otsu) threshold and an HSV guard to reject soil/shadows.
2. Blob analysis            - connected components -> area, bbox, centroid, solidity.
3. Crop-row estimation      - project vegetation pixels onto a rotating axis; the angle
                              with the sharpest periodic profile is the row direction.
                              Row spacing comes from the profile autocorrelation.
4. Classification           - plants that sit on a row line AND are close to the typical
                              crop size are CROP; everything else is WEED. If no clear row
                              structure exists we fall back to a size/shape heuristic.
5. Analytics                - coverage %, weed pressure level, weed-density heat map and
                              a spot-spraying plan (how much of the field needs treatment).

An optional deep-learning backend (YOLO) can be dropped in via models/best.pt, see
`YoloDetector` at the bottom of this file.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

MAX_WIDTH = 1280
GRID_COLS = 16


# --------------------------------------------------------------------------- helpers
def resize_max(img: np.ndarray, max_w: int = MAX_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def vegetation_mask(bgr: np.ndarray, sensitivity: int = 50) -> tuple[np.ndarray, float]:
    """Return (binary uint8 mask 0/255, exg threshold used)."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    total = r + g + b + 1e-6
    rn, gn, bn = r / total, g / total, b / total
    exg = 2 * gn - rn - bn  # roughly [-1, 2]

    scaled = np.clip((exg + 1.0) / 3.0 * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(otsu) / 255.0 * 3.0 - 1.0
    thr = float(np.clip(thr, 0.03, 0.22))
    thr -= (sensitivity - 50) / 50.0 * 0.04  # higher sensitivity -> keep more vegetation

    mask = exg > thr

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    guard = (h >= 22) & (h <= 100) & (s >= 30) & (v >= 28)
    mask = (mask & guard).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask, thr


@dataclass
class RowModel:
    detected: bool
    angle_deg: float          # row direction measured from the horizontal axis, (-90, 90]
    confidence: float         # 0..1 (autocorrelation strength)
    spacing_px: float
    centers: np.ndarray       # positions along the row normal (px, relative to image centre)
    normal: tuple[float, float]
    centre_xy: tuple[float, float]

    def project(self, x: float, y: float) -> float:
        return (x - self.centre_xy[0]) * self.normal[0] + (y - self.centre_xy[1]) * self.normal[1]


def estimate_rows(mask: np.ndarray) -> RowModel:
    h, w = mask.shape
    centre = (w / 2.0, h / 2.0)
    empty = RowModel(False, 0.0, 0.0, 0.0, np.array([]), (1.0, 0.0), centre)

    # Sub-sample vegetation pixels for the angle search.
    ys, xs = np.nonzero(mask)
    if len(xs) < 500:
        return empty
    rng = np.random.default_rng(0)
    if len(xs) > 40000:
        idx = rng.choice(len(xs), 40000, replace=False)
        xs, ys = xs[idx], ys[idx]
    px = xs.astype(np.float32) - centre[0]
    py = ys.astype(np.float32) - centre[1]
    diag = math.hypot(w, h)
    nbins = int(diag // 3) + 1

    best_score, best_theta = -1.0, 0.0
    scores = []
    for deg in np.arange(-90, 90, 1.0):
        th = math.radians(deg)
        proj = px * math.cos(th) + py * math.sin(th)
        hist, _ = np.histogram(proj, bins=nbins, range=(-diag / 2, diag / 2))
        hist = hist.astype(np.float32)
        score = float(np.sum(hist * hist)) / (float(hist.sum()) ** 2 + 1e-6)
        scores.append(score)
        if score > best_score:
            best_score, best_theta = score, th

    # Refine the angle in 0.25 deg steps around the optimum.
    base = math.degrees(best_theta)
    for deg in np.arange(base - 1.0, base + 1.01, 0.25):
        th = math.radians(deg)
        proj = px * math.cos(th) + py * math.sin(th)
        hist, _ = np.histogram(proj, bins=nbins, range=(-diag / 2, diag / 2))
        hist = hist.astype(np.float32)
        score = float(np.sum(hist * hist)) / (float(hist.sum()) ** 2 + 1e-6)
        if score > best_score:
            best_score, best_theta = score, th

    normal = (math.cos(best_theta), math.sin(best_theta))

    # Full-resolution profile along the normal (1 px bins).
    ys_f, xs_f = np.nonzero(mask)
    proj_f = (xs_f - centre[0]) * normal[0] + (ys_f - centre[1]) * normal[1]
    lo = -diag / 2
    nb = int(diag) + 1
    prof, _ = np.histogram(proj_f, bins=nb, range=(lo, lo + nb))
    prof = prof.astype(np.float32)

    # Only consider the span that actually contains vegetation, to keep the autocorrelation clean.
    nz = np.nonzero(prof)[0]
    if len(nz) < 20:
        return empty
    a, b = nz[0], nz[-1]
    sm = cv2.GaussianBlur(prof[a:b + 1].reshape(1, -1), (0, 0), sigmaX=3.0).ravel()
    centred = sm - sm.mean()
    if np.allclose(centred, 0):
        return empty

    ac = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
    ac = ac / (ac[0] + 1e-9)
    min_lag = 14
    max_lag = min(len(ac) - 2, 420)
    if max_lag <= min_lag + 3:
        return empty
    window = ac[min_lag:max_lag]
    lag = int(np.argmax(window)) + min_lag
    # prefer the first strong peak (fundamental), not a harmonic
    for k in range(min_lag + 1, max_lag - 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] >= 0.8 * ac[lag]:
            lag = k
            break
    confidence = float(np.clip(ac[lag], 0, 1))

    # Row centres = local maxima of the smoothed profile, at least 0.6*spacing apart.
    peaks = []
    thr = 0.35 * float(sm.max())
    for i in range(1, len(sm) - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= thr:
            peaks.append(i)
    pruned: list[int] = []
    for p in sorted(peaks, key=lambda i: -sm[i]):
        if all(abs(p - q) >= 0.6 * lag for q in pruned):
            pruned.append(p)
    pruned.sort()
    centers = np.array([lo + a + p + 0.5 for p in pruned], dtype=np.float32)

    # Row direction d = (-sin, cos); express as angle from horizontal in (-90, 90].
    ang = math.degrees(math.atan2(normal[0], -normal[1]))
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180

    detected = confidence >= 0.22 and len(centers) >= 2
    return RowModel(detected, ang, confidence, float(lag), centers, normal, centre)


# --------------------------------------------------------------------------- analysis
def _blobs(mask: np.ndarray, min_area: int):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < min_area:
            continue
        sub = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 1.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = float(cv2.contourArea(c) / hull_area) if hull_area > 0 else 1.0
        blobs.append({
            "label": i, "x": x, "y": y, "w": bw, "h": bh, "area": area,
            "cx": float(cents[i][0]), "cy": float(cents[i][1]), "solidity": solidity,
        })
    blobs.sort(key=lambda b: -b["area"])
    return blobs[:700], labels


def _classify(blobs, rows: RowModel, sensitivity: int, mode: str):
    """Annotate each blob dict with 'label_name' and 'confidence'. Returns mode actually used."""
    if not blobs:
        return "none"
    use_rows = rows.detected and mode in ("auto", "rows")
    if mode == "size":
        use_rows = False

    if use_rows:
        tol = max(0.28 * rows.spacing_px, 8.0)
        for b in blobs:
            xp = rows.project(b["cx"], b["cy"])
            b["row_dist"] = float(np.min(np.abs(rows.centers - xp)))
            b["in_row"] = b["row_dist"] <= tol
        cand = [b["area"] for b in blobs if b["in_row"]]
        if len(cand) >= 3:
            med = np.median(cand)
            top = [a for a in cand if a >= med]
            crop_ref = float(np.median(top))
        else:
            crop_ref = float(np.median([b["area"] for b in blobs[: max(3, len(blobs) // 4)]]))
        ratio_thr = 0.2 + sensitivity / 100.0 * 0.3  # 0.2 .. 0.5
        for b in blobs:
            size_ratio = b["area"] / max(crop_ref, 1.0)
            if b["in_row"] and size_ratio >= ratio_thr:
                b["label_name"] = "crop"
                conf = 0.62 + 0.22 * (1 - b["row_dist"] / tol) + 0.15 * min(1.0, size_ratio)
            else:
                b["label_name"] = "weed"
                off = 0.0 if b["in_row"] else min(1.0, (b["row_dist"] - tol) / max(rows.spacing_px * 0.25, 1))
                small = 1.0 - min(1.0, size_ratio / ratio_thr) if b["in_row"] else 0.4
                conf = 0.6 + 0.22 * off + 0.16 * small
            b["confidence"] = float(np.clip(conf, 0.55, 0.98))
        return "rows"

    # Size / shape fallback
    areas = [b["area"] for b in blobs]
    k = max(3, int(len(areas) * 0.35))
    crop_ref = float(np.median(areas[:k]))
    ratio_thr = 0.28 + sensitivity / 100.0 * 0.3
    for b in blobs:
        size_ratio = b["area"] / max(crop_ref, 1.0)
        if size_ratio >= ratio_thr and b["solidity"] >= 0.55:
            b["label_name"] = "crop"
            conf = 0.58 + 0.2 * min(1.0, size_ratio) + 0.15 * b["solidity"]
        else:
            b["label_name"] = "weed"
            conf = 0.58 + 0.25 * (1 - min(1.0, size_ratio / ratio_thr)) + 0.1 * (1 - b["solidity"])
        b["confidence"] = float(np.clip(conf, 0.5, 0.92))
    return "size"


def _level(weed_ratio: float) -> tuple[str, str]:
    if weed_ratio < 0.05:
        return "Low", "Weed pressure is low. Keep monitoring and treat the flagged zones only."
    if weed_ratio < 0.15:
        return "Moderate", "Moderate infestation. Schedule targeted control within 7-10 days before weeds compete for light and nutrients."
    if weed_ratio < 0.30:
        return "High", "High weed pressure. Plan control soon; prioritise the red zones on the spray map to protect yield."
    return "Severe", "Severe infestation. Weeds are likely suppressing the crop; act immediately and consider an integrated control plan."


def _rgba(shape, color_bgr, alpha, mask):
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    layer[mask > 0] = (*color_bgr, alpha)
    return layer


def analyze(bgr: np.ndarray, sensitivity: int = 50, min_area: int = 60, mode: str = "auto") -> dict:
    """Run the full pipeline. Returns dict with 'image', 'layers' (BGRA arrays) and 'result'."""
    t0 = time.time()
    sensitivity = int(np.clip(sensitivity, 0, 100))
    min_area = int(np.clip(min_area, 10, 5000))
    img = resize_max(bgr)
    h, w = img.shape[:2]
    total_px = float(h * w)

    veg, exg_thr = vegetation_mask(img, sensitivity)
    blobs, labels = _blobs(veg, min_area)

    # keep only retained blobs in the mask used downstream
    keep = np.zeros(int(labels.max()) + 1, np.uint8)
    for b in blobs:
        keep[b["label"]] = 1
    veg = (keep[labels] * 255).astype(np.uint8)

    rows = estimate_rows(veg) if blobs else RowModel(False, 0, 0, 0, np.array([]), (1, 0), (w / 2, h / 2))
    used_mode = _classify(blobs, rows, sensitivity, mode)

    crop_lut = np.zeros(int(labels.max()) + 1, np.uint8)
    weed_lut = np.zeros_like(crop_lut)
    dets = []
    ci = wi = 0
    for b in sorted(blobs, key=lambda b: (b["y"], b["x"])):
        if b["label_name"] == "crop":
            ci += 1
            did = f"C{ci}"
            crop_lut[b["label"]] = 1
        else:
            wi += 1
            did = f"W{wi}"
            weed_lut[b["label"]] = 1
        dets.append({
            "id": did, "label": b["label_name"], "confidence": round(b["confidence"], 3),
            "bbox": [b["x"], b["y"], b["w"], b["h"]],
            "centroid": [round(b["cx"], 1), round(b["cy"], 1)],
            "area_px": b["area"], "area_pct": round(b["area"] / total_px * 100, 3),
        })

    crop_mask = crop_lut[labels] * 255
    weed_mask = weed_lut[labels] * 255
    crop_mask = crop_mask.astype(np.uint8)
    weed_mask = weed_mask.astype(np.uint8)

    crop_area = int(np.count_nonzero(crop_mask))
    weed_area = int(np.count_nonzero(weed_mask))
    veg_area = crop_area + weed_area
    weed_ratio = weed_area / veg_area if veg_area else 0.0
    level, advice = _level(weed_ratio)

    # --- weed density grid / spray map
    cols = GRID_COLS
    rws = max(4, int(round(cols * h / w)))
    frac = cv2.resize((weed_mask > 0).astype(np.float32), (cols, rws), interpolation=cv2.INTER_AREA)
    flag_thr = 0.01
    flagged = frac > flag_thr
    spray_pct = float(flagged.sum()) / flagged.size * 100.0
    saving_pct = 100.0 - spray_pct

    heat = cv2.resize(frac, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat / max(0.12, float(frac.max())), 0, 1)
    heat8 = (heat * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat8, cv2.COLORMAP_INFERNO)
    heat_a = (np.clip(heat * 1.2, 0, 1) * 170).astype(np.uint8)
    heat_layer = np.dstack([heat_bgr, heat_a])

    # --- overlay layers
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    layers = {}
    for name, m, col, alpha in (("crop", crop_mask, (102, 168, 47), 120), ("weed", weed_mask, (46, 87, 228), 150)):
        layer = _rgba((h, w), col, alpha, m)
        edge = cv2.subtract(cv2.dilate(m, k3), cv2.erode(m, k3))
        layer[edge > 0] = (*col, 255)
        layers[name] = layer
    layers["mask"] = _rgba((h, w), (255, 214, 64), 170, veg)
    layers["heat"] = heat_layer

    recs = [advice]
    if veg_area:
        recs.append(
            f"Spot-spraying only the {int(flagged.sum())} flagged grid cells treats ~{spray_pct:.0f}% of the field "
            f"instead of 100% - an estimated {saving_pct:.0f}% reduction in herbicide use versus blanket spraying."
        )
    if rows.detected:
        recs.append(
            f"Crop rows detected ({len(rows.centers)} rows, ~{rows.spacing_px:.0f}px spacing). "
            "Inter-row weeds can be handled mechanically (cultivator / hoe) while intra-row weeds need spot treatment."
        )
    else:
        recs.append("No clear crop-row pattern found, so classification used plant size and shape. For best accuracy use a top-down image with visible rows.")
    recs.append("Indicative analysis from a single image - confirm with a field scouting visit or an agronomist before applying treatment.")

    result = {
        "width": w, "height": h,
        "detections": dets,
        "summary": {
            "crop_count": ci, "weed_count": wi,
            "vegetation_cover_pct": round(veg_area / total_px * 100, 2),
            "crop_cover_pct": round(crop_area / total_px * 100, 2),
            "weed_cover_pct": round(weed_area / total_px * 100, 2),
            "weed_ratio_pct": round(weed_ratio * 100, 2),
            "pressure_level": level,
            "spray_area_pct": round(spray_pct, 1),
            "herbicide_saving_pct": round(saving_pct, 1),
        },
        "rows": {
            "detected": bool(rows.detected), "angle_deg": round(rows.angle_deg, 1),
            "confidence": round(rows.confidence, 3), "spacing_px": round(rows.spacing_px, 1),
            "count": int(len(rows.centers)),
        },
        "grid": {"cols": cols, "rows": rws, "weed_fraction": np.round(frac, 4).tolist(), "flag_threshold": flag_thr},
        "method": {"engine": "classical-cv", "classifier": used_mode, "exg_threshold": round(exg_thr, 3),
                   "sensitivity": sensitivity, "min_area": min_area, "mode_requested": mode},
        "recommendations": recs,
        "processing_ms": int((time.time() - t0) * 1000),
    }
    return {"image": img, "layers": layers, "result": result}


# --------------------------------------------------------------------------- optional DL backend
class YoloDetector:
    """Optional: drop a trained Ultralytics YOLO model at backend/models/best.pt with classes
    named 'crop' and 'weed' and use `YoloDetector.load()` to get detections in the same
    schema as the classical pipeline (boxes only). Returns None if unavailable."""

    @staticmethod
    def load(path: str | None = None):
        path = path or os.path.join(os.path.dirname(__file__), "models", "best.pt")
        if not os.path.exists(path):
            return None
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            return None
        return YOLO(path)

    @staticmethod
    def detect(model, bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
        out = []
        res = model.predict(bgr, conf=conf, verbose=False)[0]
        names = res.names
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            label = str(names[int(box.cls[0])]).lower()
            out.append({
                "id": f"{'C' if label == 'crop' else 'W'}{i + 1}", "label": "crop" if label == "crop" else "weed",
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            })
        return out
