/* FieldSense in-browser detection engine — JavaScript port of backend/detector.py
 * Pure functions on RGBA pixel data; no dependencies. Exposes window.FieldSenseDetector.analyze().
 */
(function (global) {
  "use strict";

  const GRID_COLS = 16;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function median(arr) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /* ---------------------------------------------------------------- segmentation */
  function otsu(hist, total) {
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, max = -1, thr = 0;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > max) { max = between; thr = t; }
    }
    return thr;
  }

  function morph(mask, w, h, r, erode) {
    const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h);
    const init = erode ? 1 : 0;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let v = init;
        for (let k = -r; k <= r; k++) {
          const xx = x + k;
          if (xx < 0 || xx >= w) continue;
          const m = mask[row + xx];
          if (erode) { if (!m) { v = 0; break; } } else if (m) { v = 1; break; }
        }
        tmp[row + x] = v;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v = init;
        for (let k = -r; k <= r; k++) {
          const yy = y + k;
          if (yy < 0 || yy >= h) continue;
          const m = tmp[yy * w + x];
          if (erode) { if (!m) { v = 0; break; } } else if (m) { v = 1; break; }
        }
        out[y * w + x] = v;
      }
    }
    return out;
  }

  function vegetationMask(data, w, h, sensitivity) {
    const n = w * h;
    const exg = new Float32Array(n);
    const ok = new Uint8Array(n);
    const hist = new Uint32Array(256);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const r = data[p], g = data[p + 1], b = data[p + 2];
      const tot = r + g + b + 1e-6;
      const e = (2 * g - r - b) / tot;
      exg[i] = e;
      hist[clamp(Math.floor(((e + 1) / 3) * 255), 0, 255)]++;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
      const s = mx ? (d / mx) * 255 : 0;
      let hue = 0;
      if (d) {
        if (mx === r) hue = 60 * (((g - b) / d) % 6);
        else if (mx === g) hue = 60 * (2 + (b - r) / d);
        else hue = 60 * (4 + (r - g) / d);
        if (hue < 0) hue += 360;
      }
      hue /= 2; // OpenCV scale 0..180
      ok[i] = hue >= 22 && hue <= 100 && s >= 30 && mx >= 28 ? 1 : 0;
    }
    const t = otsu(hist, n);
    let thr = (t / 255) * 3 - 1;
    thr = clamp(thr, 0.03, 0.22);
    thr -= ((sensitivity - 50) / 50) * 0.04;

    let mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = exg[i] > thr && ok[i] ? 1 : 0;
    mask = morph(morph(mask, w, h, 1, true), w, h, 1, false);   // open 3x3
    mask = morph(morph(mask, w, h, 2, false), w, h, 2, true);   // close 5x5
    return { mask, thr };
  }

  /* ---------------------------------------------------------------- components */
  function components(mask, w, h) {
    const labels = new Int32Array(w * h);
    const parent = [0];
    const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) { if (ra < rb) parent[rb] = ra; else parent[ra] = rb; } };
    let next = 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        const a = x > 0 ? labels[i - 1] : 0;
        const b = y > 0 && x > 0 ? labels[i - w - 1] : 0;
        const c = y > 0 ? labels[i - w] : 0;
        const d = y > 0 && x < w - 1 ? labels[i - w + 1] : 0;
        let m = 0;
        if (a) m = a;
        if (b && (!m || b < m)) m = b;
        if (c && (!m || c < m)) m = c;
        if (d && (!m || d < m)) m = d;
        if (!m) { labels[i] = next; parent.push(next); next++; }
        else {
          labels[i] = m;
          if (a) union(a, m); if (b) union(b, m); if (c) union(c, m); if (d) union(d, m);
        }
      }
    }
    const ids = new Int32Array(next);
    let count = 0;
    for (let l = 1; l < next; l++) { const r = find(l); if (!ids[r]) ids[r] = ++count; }
    const final = new Int32Array(next);
    for (let l = 1; l < next; l++) final[l] = ids[find(l)];

    const area = new Float64Array(count + 1), minx = new Int32Array(count + 1).fill(1e9), maxx = new Int32Array(count + 1).fill(-1),
      miny = new Int32Array(count + 1).fill(1e9), maxy = new Int32Array(count + 1).fill(-1),
      sx = new Float64Array(count + 1), sy = new Float64Array(count + 1);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const l = labels[i];
        if (!l) continue;
        const id = final[l];
        labels[i] = id;
        area[id]++; sx[id] += x; sy[id] += y;
        if (x < minx[id]) minx[id] = x; if (x > maxx[id]) maxx[id] = x;
        if (y < miny[id]) miny[id] = y; if (y > maxy[id]) maxy[id] = y;
      }
    }
    return { labels, count, area, minx, maxx, miny, maxy, sx, sy };
  }

  function hullStats(pts) {
    if (pts.length < 3) return { area: 0, perim: 0 };
    pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
    upper.pop(); lower.pop();
    const hull = lower.concat(upper);
    let a = 0, per = 0;
    for (let i = 0; i < hull.length; i++) {
      const p = hull[i], q = hull[(i + 1) % hull.length];
      a += p[0] * q[1] - q[0] * p[1];
      per += Math.hypot(q[0] - p[0], q[1] - p[1]);
    }
    return { area: Math.abs(a) / 2, perim: per };
  }

  function buildBlobs(cc, w, minArea) {
    const blobs = [];
    for (let id = 1; id <= cc.count; id++) {
      const area = cc.area[id];
      if (area < minArea) continue;
      const x = cc.minx[id], y = cc.miny[id], bw = cc.maxx[id] - x + 1, bh = cc.maxy[id] - y + 1;
      // hull from the leftmost / rightmost pixel of every scan-line
      const pts = [];
      for (let yy = y; yy < y + bh; yy++) {
        let l = -1, r = -1;
        for (let xx = x; xx < x + bw; xx++) {
          if (cc.labels[yy * w + xx] === id) { if (l < 0) l = xx; r = xx; }
        }
        if (l >= 0) { pts.push([l, yy]); if (r !== l) pts.push([r, yy]); }
      }
      const hs = hullStats(pts);
      const denom = hs.area + hs.perim / 2 + 1;
      blobs.push({
        label: id, x, y, w: bw, h: bh, area,
        cx: cc.sx[id] / area, cy: cc.sy[id] / area,
        solidity: denom > 0 ? Math.min(1, area / denom) : 1,
      });
    }
    blobs.sort((a, b) => b.area - a.area);
    return blobs.slice(0, 700);
  }

  /* ---------------------------------------------------------------- crop rows */
  function gaussian1d(arr, sigma) {
    const rad = Math.ceil(sigma * 3), k = [];
    let s = 0;
    for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); s += v; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    const n = arr.length, out = new Float64Array(n);
    const refl = (i) => { if (n === 1) return 0; while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * (n - 1) - i; } return i; };
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let j = -rad; j <= rad; j++) v += arr[refl(i + j)] * k[j + rad];
      out[i] = v;
    }
    return out;
  }

  function estimateRows(mask, w, h) {
    const centre = [w / 2, h / 2];
    const empty = { detected: false, angle: 0, confidence: 0, spacing: 0, centers: [], normal: [1, 0], centre };
    let idxs = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) idxs.push(i);
    if (idxs.length < 500) return empty;
    const full = idxs;
    if (idxs.length > 40000) {
      const rnd = mulberry32(0), pick = new Array(40000);
      for (let k = 0; k < 40000; k++) pick[k] = idxs[Math.floor(rnd() * idxs.length)];
      idxs = pick;
    }
    const m = idxs.length, px = new Float32Array(m), py = new Float32Array(m);
    for (let k = 0; k < m; k++) { px[k] = (idxs[k] % w) - centre[0]; py[k] = Math.floor(idxs[k] / w) - centre[1]; }
    const diag = Math.hypot(w, h), nbins = Math.floor(diag / 3) + 1;
    const hist = new Float32Array(nbins);

    function scoreAt(deg) {
      const th = (deg * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th);
      hist.fill(0);
      for (let k = 0; k < m; k++) {
        const proj = px[k] * c + py[k] * s;
        let b = Math.floor(((proj + diag / 2) / diag) * nbins);
        if (b < 0 || b > nbins) continue;
        if (b === nbins) b = nbins - 1;
        hist[b]++;
      }
      let sq = 0, tot = 0;
      for (let i = 0; i < nbins; i++) { sq += hist[i] * hist[i]; tot += hist[i]; }
      return sq / (tot * tot + 1e-6);
    }

    let bestScore = -1, bestDeg = 0;
    for (let deg = -90; deg < 90; deg += 1) { const sc = scoreAt(deg); if (sc > bestScore) { bestScore = sc; bestDeg = deg; } }
    const base = bestDeg;
    for (let deg = base - 1; deg <= base + 1.0001; deg += 0.25) { const sc = scoreAt(deg); if (sc > bestScore) { bestScore = sc; bestDeg = deg; } }

    const th = (bestDeg * Math.PI) / 180;
    const normal = [Math.cos(th), Math.sin(th)];

    const lo = -diag / 2, nb = Math.floor(diag) + 1;
    const prof = new Float64Array(nb);
    for (let k = 0; k < full.length; k++) {
      const x = (full[k] % w) - centre[0], y = Math.floor(full[k] / w) - centre[1];
      let b = Math.floor(x * normal[0] + y * normal[1] - lo);
      if (b < 0 || b > nb) continue;
      if (b === nb) b = nb - 1;
      prof[b]++;
    }
    let first = -1, last = -1, nzCount = 0;
    for (let i = 0; i < nb; i++) if (prof[i]) { nzCount++; if (first < 0) first = i; last = i; }
    if (nzCount < 20) return empty;

    const sm = gaussian1d(prof.subarray(first, last + 1), 3.0);
    const L = sm.length;
    let mean = 0, smax = 0;
    for (let i = 0; i < L; i++) { mean += sm[i]; if (sm[i] > smax) smax = sm[i]; }
    mean /= L;
    const c = new Float64Array(L);
    let allZero = true;
    for (let i = 0; i < L; i++) { c[i] = sm[i] - mean; if (Math.abs(c[i]) > 1e-9) allZero = false; }
    if (allZero) return empty;

    const minLag = 14, maxLag = Math.min(L - 2, 420);
    if (maxLag <= minLag + 3) return empty;
    const ac = new Float64Array(maxLag + 2);
    for (let k = 0; k <= maxLag + 1 && k < L; k++) {
      let s = 0;
      for (let i = 0; i + k < L; i++) s += c[i] * c[i + k];
      ac[k] = s;
    }
    const ac0 = ac[0] + 1e-9;
    for (let k = 0; k < ac.length; k++) ac[k] /= ac0;

    let lag = minLag, bestV = -Infinity;
    for (let k = minLag; k < maxLag; k++) if (ac[k] > bestV) { bestV = ac[k]; lag = k; }
    const lag0 = lag;
    for (let k = minLag + 1; k < maxLag - 1; k++) {
      if (ac[k] > ac[k - 1] && ac[k] >= ac[k + 1] && ac[k] >= 0.8 * ac[lag0]) { lag = k; break; }
    }
    const confidence = clamp(ac[lag], 0, 1);

    const peaks = [], pthr = 0.35 * smax;
    for (let i = 1; i < L - 1; i++) if (sm[i] >= sm[i - 1] && sm[i] > sm[i + 1] && sm[i] >= pthr) peaks.push(i);
    const pruned = [];
    for (const p of [...peaks].sort((a, b) => sm[b] - sm[a])) {
      if (pruned.every((q) => Math.abs(p - q) >= 0.6 * lag)) pruned.push(p);
    }
    pruned.sort((a, b) => a - b);
    const centers = pruned.map((p) => lo + first + p + 0.5);

    let ang = (Math.atan2(normal[0], -normal[1]) * 180) / Math.PI;
    while (ang <= -90) ang += 180;
    while (ang > 90) ang -= 180;

    return { detected: confidence >= 0.22 && centers.length >= 2, angle: ang, confidence, spacing: lag, centers, normal, centre };
  }

  /* ---------------------------------------------------------------- classification */
  function classify(blobs, rows, sensitivity, mode) {
    if (!blobs.length) return "none";
    const useRows = rows.detected && (mode === "auto" || mode === "rows") && mode !== "size";
    if (useRows) {
      const tol = Math.max(0.28 * rows.spacing, 8);
      for (const b of blobs) {
        const xp = (b.cx - rows.centre[0]) * rows.normal[0] + (b.cy - rows.centre[1]) * rows.normal[1];
        let d = Infinity;
        for (const c of rows.centers) d = Math.min(d, Math.abs(c - xp));
        b.rowDist = d; b.inRow = d <= tol;
      }
      const cand = blobs.filter((b) => b.inRow).map((b) => b.area);
      let cropRef;
      if (cand.length >= 3) { const med = median(cand); cropRef = median(cand.filter((a) => a >= med)); }
      else cropRef = median(blobs.slice(0, Math.max(3, blobs.length >> 2)).map((b) => b.area));
      const ratioThr = 0.2 + (sensitivity / 100) * 0.3;
      for (const b of blobs) {
        const sr = b.area / Math.max(cropRef, 1);
        let conf;
        if (b.inRow && sr >= ratioThr) {
          b.name = "crop";
          conf = 0.62 + 0.22 * (1 - b.rowDist / tol) + 0.15 * Math.min(1, sr);
        } else {
          b.name = "weed";
          const off = b.inRow ? 0 : Math.min(1, (b.rowDist - tol) / Math.max(rows.spacing * 0.25, 1));
          const small = b.inRow ? 1 - Math.min(1, sr / ratioThr) : 0.4;
          conf = 0.6 + 0.22 * off + 0.16 * small;
        }
        b.conf = clamp(conf, 0.55, 0.98);
      }
      return "rows";
    }
    const k = Math.max(3, Math.floor(blobs.length * 0.35));
    const cropRef = median(blobs.slice(0, k).map((b) => b.area));
    const ratioThr = 0.28 + (sensitivity / 100) * 0.3;
    for (const b of blobs) {
      const sr = b.area / Math.max(cropRef, 1);
      let conf;
      if (sr >= ratioThr && b.solidity >= 0.55) { b.name = "crop"; conf = 0.58 + 0.2 * Math.min(1, sr) + 0.15 * b.solidity; }
      else { b.name = "weed"; conf = 0.58 + 0.25 * (1 - Math.min(1, sr / ratioThr)) + 0.1 * (1 - b.solidity); }
      b.conf = clamp(conf, 0.5, 0.92);
    }
    return "size";
  }

  function level(r) {
    if (r < 0.05) return ["Low", "Weed pressure is low. Keep monitoring and treat the flagged zones only."];
    if (r < 0.15) return ["Moderate", "Moderate infestation. Schedule targeted control within 7-10 days before weeds compete for light and nutrients."];
    if (r < 0.3) return ["High", "High weed pressure. Plan control soon; prioritise the red zones on the spray map to protect yield."];
    return ["Severe", "Severe infestation. Weeds are likely suppressing the crop; act immediately and consider an integrated control plan."];
  }

  /* ---------------------------------------------------------------- layers */
  const INFERNO = [[0, [0, 0, 4]], [0.25, [87, 16, 110]], [0.5, [188, 55, 84]], [0.75, [249, 142, 9]], [1, [252, 255, 164]]];
  function inferno(v) {
    for (let i = 1; i < INFERNO.length; i++) {
      if (v <= INFERNO[i][0]) {
        const [t0, c0] = INFERNO[i - 1], [t1, c1] = INFERNO[i], f = (v - t0) / (t1 - t0);
        return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
      }
    }
    return INFERNO[INFERNO.length - 1][1];
  }

  function colorLayer(selected, w, h, rgb, alpha) {
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!selected[i]) continue;
        const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || !selected[i - 1] || !selected[i + 1] || !selected[i - w] || !selected[i + w];
        const p = i * 4;
        out[p] = rgb[0]; out[p + 1] = rgb[1]; out[p + 2] = rgb[2]; out[p + 3] = edge ? 255 : alpha;
      }
    }
    return out;
  }

  /* ---------------------------------------------------------------- main */
  function analyze(imageData, opts = {}) {
    const t0 = performance.now();
    const sensitivity = clamp(Math.round(opts.sensitivity ?? 50), 0, 100);
    const minArea = clamp(Math.round(opts.minArea ?? 60), 10, 5000);
    const mode = opts.mode || "auto";
    const { data, width: w, height: h } = imageData;
    const total = w * h;

    const { mask: rawMask, thr } = vegetationMask(data, w, h, sensitivity);
    const cc = components(rawMask, w, h);
    const blobs = buildBlobs(cc, w, minArea);

    const veg = new Uint8Array(total);
    const keep = new Uint8Array(cc.count + 1);
    for (const b of blobs) keep[b.label] = 1;
    for (let i = 0; i < total; i++) { const l = cc.labels[i]; if (l && keep[l]) veg[i] = 1; }

    const rows = blobs.length ? estimateRows(veg, w, h) : { detected: false, angle: 0, confidence: 0, spacing: 0, centers: [], normal: [1, 0], centre: [w / 2, h / 2] };
    const used = classify(blobs, rows, sensitivity, mode);

    const cropLut = new Uint8Array(cc.count + 1), weedLut = new Uint8Array(cc.count + 1);
    const dets = [];
    let ci = 0, wi = 0;
    for (const b of [...blobs].sort((p, q) => p.y - q.y || p.x - q.x)) {
      let id;
      if (b.name === "crop") { id = "C" + ++ci; cropLut[b.label] = 1; } else { id = "W" + ++wi; weedLut[b.label] = 1; }
      dets.push({
        id, label: b.name, confidence: Math.round(b.conf * 1000) / 1000,
        bbox: [b.x, b.y, b.w, b.h], centroid: [Math.round(b.cx * 10) / 10, Math.round(b.cy * 10) / 10],
        area_px: b.area, area_pct: Math.round((b.area / total) * 100000) / 1000,
      });
    }

    const cropMask = new Uint8Array(total), weedMask = new Uint8Array(total);
    let cropArea = 0, weedArea = 0;
    for (let i = 0; i < total; i++) {
      const l = cc.labels[i];
      if (!l) continue;
      if (cropLut[l]) { cropMask[i] = 1; cropArea++; } else if (weedLut[l]) { weedMask[i] = 1; weedArea++; }
    }
    const vegArea = cropArea + weedArea;
    const weedRatio = vegArea ? weedArea / vegArea : 0;
    const [lvl, advice] = level(weedRatio);

    // weed grid / spray plan
    const cols = GRID_COLS, rws = Math.max(4, Math.round((cols * h) / w));
    const cell = new Float64Array(cols * rws), cellN = new Float64Array(cols * rws);
    for (let y = 0; y < h; y++) {
      const gy = Math.min(rws - 1, Math.floor((y * rws) / h));
      for (let x = 0; x < w; x++) {
        const gi = gy * cols + Math.min(cols - 1, Math.floor((x * cols) / w));
        cellN[gi]++;
        if (weedMask[y * w + x]) cell[gi]++;
      }
    }
    const frac = [];
    let fmax = 0, flagged = 0;
    const flagThr = 0.01;
    for (let gy = 0; gy < rws; gy++) {
      const row = [];
      for (let gx = 0; gx < cols; gx++) {
        const f = cell[gy * cols + gx] / Math.max(1, cellN[gy * cols + gx]);
        row.push(Math.round(f * 10000) / 10000);
        if (f > fmax) fmax = f;
        if (f > flagThr) flagged++;
      }
      frac.push(row);
    }
    const sprayPct = (flagged / (cols * rws)) * 100, savingPct = 100 - sprayPct;

    const heatTiny = new Uint8ClampedArray(cols * rws * 4);
    const norm = Math.max(0.12, fmax);
    for (let gy = 0; gy < rws; gy++) for (let gx = 0; gx < cols; gx++) {
      const v = clamp(frac[gy][gx] / norm, 0, 1), p = (gy * cols + gx) * 4, c = inferno(v);
      heatTiny[p] = c[0]; heatTiny[p + 1] = c[1]; heatTiny[p + 2] = c[2]; heatTiny[p + 3] = clamp(v * 1.2, 0, 1) * 170;
    }

    const recs = [advice];
    if (vegArea) recs.push(`Spot-spraying only the ${flagged} flagged grid cells treats ~${sprayPct.toFixed(0)}% of the field instead of 100% - an estimated ${savingPct.toFixed(0)}% reduction in herbicide use versus blanket spraying.`);
    if (rows.detected) recs.push(`Crop rows detected (${rows.centers.length} rows, ~${rows.spacing.toFixed(0)}px spacing). Inter-row weeds can be handled mechanically (cultivator / hoe) while intra-row weeds need spot treatment.`);
    else recs.push("No clear crop-row pattern found, so classification used plant size and shape. For best accuracy use a top-down image with visible rows.");
    recs.push("Indicative analysis from a single image - confirm with a field scouting visit or an agronomist before applying treatment.");

    const r1 = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
    const result = {
      width: w, height: h, detections: dets,
      summary: {
        crop_count: ci, weed_count: wi,
        vegetation_cover_pct: r1((vegArea / total) * 100), crop_cover_pct: r1((cropArea / total) * 100),
        weed_cover_pct: r1((weedArea / total) * 100), weed_ratio_pct: r1(weedRatio * 100),
        pressure_level: lvl, spray_area_pct: r1(sprayPct, 1), herbicide_saving_pct: r1(savingPct, 1),
      },
      rows: { detected: !!rows.detected, angle_deg: r1(rows.angle, 1), confidence: r1(rows.confidence, 3), spacing_px: r1(rows.spacing, 1), count: rows.centers.length },
      grid: { cols, rows: rws, weed_fraction: frac, flag_threshold: flagThr },
      method: { engine: "browser-js", classifier: used, exg_threshold: r1(thr, 3), sensitivity, min_area: minArea, mode_requested: mode },
      recommendations: recs,
      processing_ms: Math.round(performance.now() - t0),
    };
    return {
      result,
      layers: {
        crop: colorLayer(cropMask, w, h, [47, 168, 102], 120),
        weed: colorLayer(weedMask, w, h, [228, 87, 46], 150),
        mask: colorLayer(veg, w, h, [64, 214, 255], 170),
        heat: { data: heatTiny, w: cols, h: rws },
      },
    };
  }

  global.FieldSenseDetector = { analyze };
})(window);
