"""Synthetic top-down field generator (crop rows + scattered weeds) for demos and tests."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _leaf(img, cx, cy, length, width, angle, color):
    cv2.ellipse(img, (int(cx), int(cy)), (max(2, int(length)), max(1, int(width))), angle, 0, 360, color, -1, cv2.LINE_AA)


def _plant(img, cx, cy, radius, rng, base, leaves):
    # soft shadow
    shadow = (int(base[0] * 0.35), int(base[1] * 0.35), int(base[2] * 0.35))
    for _ in range(leaves):
        a = rng.uniform(0, 360)
        _leaf(img, cx + 3, cy + 4, radius * rng.uniform(0.7, 1.0), radius * rng.uniform(0.22, 0.34), a, shadow)
    for _ in range(leaves):
        a = rng.uniform(0, 360)
        d = radius * rng.uniform(0.35, 0.6)
        ox, oy = cx + math.cos(math.radians(a)) * d * 0.6, cy + math.sin(math.radians(a)) * d * 0.6
        jitter = rng.integers(-14, 15, 3)
        col = tuple(int(np.clip(base[i] + jitter[i], 0, 255)) for i in range(3))
        _leaf(img, ox, oy, radius * rng.uniform(0.6, 0.95), radius * rng.uniform(0.2, 0.32), a, col)
    cv2.circle(img, (int(cx), int(cy)), max(2, int(radius * 0.18)), (40, 110, 60), -1, cv2.LINE_AA)


def generate_field(seed: int = 7, width: int = 1100, height: int = 740, angle: float | None = None,
                   weed_level: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-14, 14)) if angle is None else float(angle)

    # soil: low-frequency tint + grain + pebbles
    soil = np.full((height, width, 3), (62, 92, 128), np.float32)  # BGR brown
    low = cv2.resize(rng.normal(0, 1, (height // 60 + 2, width // 60 + 2)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    soil += low[..., None] * np.array([6, 8, 10], np.float32)
    soil += rng.normal(0, 7, (height, width, 1)).astype(np.float32)
    img = np.clip(soil, 0, 255).astype(np.uint8)
    for _ in range(260):
        cv2.circle(img, (int(rng.integers(0, width)), int(rng.integers(0, height))), int(rng.integers(1, 4)),
                   tuple(int(v) for v in rng.integers(95, 165, 3)), -1, cv2.LINE_AA)

    # furrows along the row direction
    th = math.radians(angle)
    d = np.array([math.cos(th), math.sin(th)])
    n = np.array([-d[1], d[0]])
    centre = np.array([width / 2, height / 2])
    diag = math.hypot(width, height)
    spacing = rng.uniform(112, 128)

    row_offsets = np.arange(-diag / 2, diag / 2, spacing)
    for off in row_offsets:
        p0 = centre + n * off - d * diag
        p1 = centre + n * off + d * diag
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (48, 74, 104), 14, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)

    crop_base = (58, 150, 62)
    weed_bases = [(50, 175, 120), (60, 160, 95), (45, 140, 110)]

    # crop plants on the rows
    plant_step = rng.uniform(62, 74)
    for off in row_offsets:
        t = -diag / 2
        while t < diag / 2:
            pos = centre + n * (off + rng.normal(0, 2.2)) + d * (t + rng.normal(0, 4))
            if 10 < pos[0] < width - 10 and 10 < pos[1] < height - 10 and rng.random() > 0.05:
                _plant(img, pos[0], pos[1], rng.uniform(21, 30), rng, crop_base, int(rng.integers(6, 9)))
            t += plant_step

    # weeds: mostly between rows, some in-row, mixed sizes
    n_weeds = int(95 * weed_level)
    for _ in range(n_weeds):
        x, y = rng.uniform(12, width - 12), rng.uniform(12, height - 12)
        base = weed_bases[int(rng.integers(0, len(weed_bases)))]
        kind = rng.random()
        if kind < 0.65:
            _plant(img, x, y, rng.uniform(7, 14), rng, base, int(rng.integers(3, 6)))
        elif kind < 0.9:
            for _ in range(int(rng.integers(5, 10))):  # grass blades
                a = math.radians(rng.uniform(0, 360))
                ln = rng.uniform(10, 20)
                cv2.line(img, (int(x), int(y)), (int(x + math.cos(a) * ln), int(y + math.sin(a) * ln)), base, 2, cv2.LINE_AA)
        else:
            for _ in range(int(rng.integers(3, 6))):  # patch
                _plant(img, x + rng.normal(0, 12), y + rng.normal(0, 12), rng.uniform(7, 12), rng, base, 4)

    # camera feel: slight vignette, sensor noise, slight blur
    yy, xx = np.mgrid[0:height, 0:width]
    vign = 1 - 0.18 * (((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
    img = np.clip(img.astype(np.float32) * vign[..., None] + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 0.8)
