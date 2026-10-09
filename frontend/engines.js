/* FieldSense engines.
 *  - ApiEngine:     talks to the Flask backend (full-stack mode, SQLite history)
 *  - BrowserEngine: runs the detector in the browser (works on static hosting such as
 *                   GitHub Pages) and keeps scan history in IndexedDB.
 * The UI doesn't care which one is active; both return the same result schema.
 */
(function (global) {
  "use strict";
  const MAX_W = 1280;

  /* ------------------------------------------------------------------ API engine */
  const ApiEngine = {
    name: "server",
    label: "Flask API",
    note: "Analysis runs on the Flask + OpenCV server; history is stored in SQLite.",
    async analyze(file, p) {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("sensitivity", p.sensitivity);
      fd.append("min_area", p.min_area);
      fd.append("mode", p.mode);
      const r = await fetch("api/analyze", { method: "POST", body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || "Analysis failed");
      return data;
    },
    async list() { return (await fetch("api/analyses")).json(); },
    async get(id) {
      const r = await fetch(`api/analyses/${id}`);
      if (!r.ok) throw new Error("Scan not found");
      return r.json();
    },
    async remove(id) { await fetch(`api/analyses/${id}`, { method: "DELETE" }); },
    async stats() { return (await fetch("api/stats")).json(); },
  };

  /* ------------------------------------------------------------------ storage (IndexedDB + memory fallback) */
  const mem = new Map();
  let dbp = null;
  function openDB() {
    if (!dbp) {
      dbp = new Promise((resolve, reject) => {
        if (!global.indexedDB) return reject(new Error("no idb"));
        const rq = indexedDB.open("fieldsense", 1);
        rq.onupgradeneeded = () => rq.result.createObjectStore("scans", { keyPath: "id" });
        rq.onsuccess = () => resolve(rq.result);
        rq.onerror = () => reject(rq.error);
      });
    }
    return dbp;
  }
  async function withStore(mode, fn) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction("scans", mode);
      const rq = fn(t.objectStore("scans"));
      t.oncomplete = () => resolve(rq ? rq.result : undefined);
      t.onerror = t.onabort = () => reject(t.error);
    });
  }
  const store = {
    async put(rec) { try { await withStore("readwrite", (s) => s.put(rec)); } catch { mem.set(rec.id, rec); } },
    async all() {
      let rows = [];
      try { rows = (await withStore("readonly", (s) => s.getAll())) || []; } catch { /* fall through */ }
      return rows.concat([...mem.values()]).sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
    },
    async get(id) {
      try { const r = await withStore("readonly", (s) => s.get(id)); if (r) return r; } catch { /* fall through */ }
      return mem.get(id);
    },
    async del(id) { mem.delete(id); try { await withStore("readwrite", (s) => s.delete(id)); } catch { /* ignore */ } },
  };

  /* ------------------------------------------------------------------ browser engine */
  const tick = () => new Promise((r) => setTimeout(r, 30));

  async function decode(blob) {
    let bmp;
    try { bmp = await createImageBitmap(blob, { imageOrientation: "from-image" }); }
    catch { bmp = await createImageBitmap(blob); }
    return bmp;
  }
  function rgbaURL(data, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d").putImageData(new ImageData(data, w, h), 0, 0);
    return c.toDataURL("image/png");
  }
  function heatURL(tiny, w, h) {
    const t = document.createElement("canvas");
    t.width = tiny.w; t.height = tiny.h;
    t.getContext("2d").putImageData(new ImageData(tiny.data, tiny.w, tiny.h), 0, 0);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(t, 0, 0, w, h);
    return c.toDataURL("image/png");
  }
  const canvasBlob = (c, type, q) => new Promise((res) => c.toBlob(res, type, q));

  function toCSV(res) {
    const lines = ["id,label,confidence,x,y,width,height,area_px,area_pct_of_image"];
    for (const d of res.detections) lines.push([d.id, d.label, d.confidence, ...d.bbox, d.area_px, d.area_pct].join(","));
    return lines.join("\n") + "\n";
  }

  async function run(blob, params, meta) {
    const bmp = await decode(blob);
    const scale = Math.min(1, MAX_W / bmp.width);
    const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bmp, 0, 0, w, h);
    if (bmp.close) bmp.close();
    const img = ctx.getImageData(0, 0, w, h);

    await tick(); // let the "analysing" overlay paint before the heavy loop
    const out = global.FieldSenseDetector.analyze(img, { sensitivity: params.sensitivity, minArea: params.min_area, mode: params.mode });
    const res = out.result;
    res.id = meta.id; res.filename = meta.filename; res.created_at = meta.created_at;
    res.urls = {
      original: canvas.toDataURL("image/jpeg", 0.92),
      crop: rgbaURL(out.layers.crop, w, h),
      weed: rgbaURL(out.layers.weed, w, h),
      mask: rgbaURL(out.layers.mask, w, h),
      heat: heatURL(out.layers.heat, w, h),
      csv: URL.createObjectURL(new Blob([toCSV(res)], { type: "text/csv" })),
      json: URL.createObjectURL(new Blob([JSON.stringify(res, null, 2)], { type: "application/json" })),
    };
    res.names = { csv: `fieldsense_${res.id}.csv`, json: `fieldsense_${res.id}.json` };
    return { res, canvas };
  }

  const BrowserEngine = {
    name: "browser",
    label: "In-browser",
    note: "Analysis runs entirely in your browser; images never leave your device. History is stored locally (IndexedDB).",
    async analyze(file, params) {
      const meta = {
        id: Math.random().toString(36).slice(2, 14),
        filename: file.name || "field.jpg",
        created_at: new Date().toISOString(),
      };
      const { res, canvas } = await run(file, params, meta);
      // persist (resized JPEG + params + thumbnail) so the scan can be reopened later
      try {
        const jpeg = await canvasBlob(canvas, "image/jpeg", 0.9);
        const tw = 360, th = Math.round((canvas.height / canvas.width) * tw);
        const tc = document.createElement("canvas");
        tc.width = tw; tc.height = th;
        tc.getContext("2d").drawImage(canvas, 0, 0, tw, th);
        const s = res.summary;
        await store.put({
          id: meta.id, created_at: meta.created_at, filename: meta.filename, blob: jpeg, params,
          thumb: tc.toDataURL("image/jpeg", 0.7), crop_count: s.crop_count, weed_count: s.weed_count,
          weed_ratio: s.weed_ratio_pct, pressure: s.pressure_level, saving: s.herbicide_saving_pct,
        });
        const all = await store.all();
        for (const old of all.slice(60)) await store.del(old.id);
      } catch { /* history is best-effort */ }
      return res;
    },
    async list() {
      return (await store.all()).map((r) => ({
        id: r.id, created_at: r.created_at, filename: r.filename, crop_count: r.crop_count, weed_count: r.weed_count,
        weed_ratio: r.weed_ratio, pressure: r.pressure, saving: r.saving, thumb: r.thumb,
      }));
    },
    async get(id) {
      const rec = await store.get(id);
      if (!rec) throw new Error("Scan not found");
      const { res } = await run(rec.blob, rec.params, { id: rec.id, filename: rec.filename, created_at: rec.created_at });
      return res;
    },
    async remove(id) { await store.del(id); },
    async stats() {
      const all = await store.all();
      const n = all.length;
      return {
        analyses: n,
        plants_classified: all.reduce((a, r) => a + r.crop_count + r.weed_count, 0),
        avg_weed_ratio: n ? all.reduce((a, r) => a + r.weed_ratio, 0) / n : 0,
        avg_saving: n ? all.reduce((a, r) => a + r.saving, 0) / n : 0,
      };
    },
  };

  /* ------------------------------------------------------------------ selection */
  async function pick() {
    const forced = new URLSearchParams(location.search).get("engine");
    if (forced === "browser" || location.protocol === "file:" || /\.github\.io$/i.test(location.hostname)) return BrowserEngine;
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      const r = await fetch("api/health", { signal: ctrl.signal });
      clearTimeout(timer);
      const j = await r.json();
      if (j && j.status === "ok") return ApiEngine;
    } catch { /* no backend -> static hosting */ }
    return BrowserEngine;
  }

  global.FieldSenseEngines = { pick, ApiEngine, BrowserEngine };
})(window);
