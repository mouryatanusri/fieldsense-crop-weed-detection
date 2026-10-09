/* FieldSense frontend — vanilla JS, no dependencies */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const COLORS = { crop: "#2fa866", weed: "#e4572e" };
  const LEVELS = { Low: "#2fa866", Moderate: "#d99a0b", High: "#e4572e", Severe: "#a8300f" };

  const state = {
    file: null, result: null, filter: "all", mode: "auto",
    selected: null, hover: null, busy: false, layers: { crop: true, weed: true, boxes: true, heat: false, mask: false },
  };

  const el = {
    drop: $("#drop"), file: $("#file"), sampleBtn: $("#sampleBtn"),
    sens: $("#sens"), sensOut: $("#sensOut"), area: $("#area"), areaOut: $("#areaOut"),
    stage: $("#stage"), empty: $("#empty"), base: $("#base"), canvas: $("#boxes"), tip: $("#tip"),
    scan: $("#scan"), scanMsg: $("#scanMsg"), viewTitle: $("#viewTitle"), engineNote: $("#engineNote"),
    holdBtn: $("#holdBtn"), pngBtn: $("#pngBtn"), results: $("#results"),
  };
  const ctx = el.canvas.getContext("2d");

  let engine = null;
  const enginePromise = FieldSenseEngines.pick().then((e) => { engine = e; return e; });

  /* ------------------------------------------------------------------ utils */
  function toast(msg, err = false) {
    const t = document.createElement("div");
    t.className = "toast" + (err ? " err" : "");
    t.textContent = msg;
    $("#toasts").appendChild(t);
    setTimeout(() => t.remove(), 4200);
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const fmtDate = (iso) => new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

  function countUp(node, to, decimals = 0, ms = 700) {
    const start = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - start) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      node.textContent = (to * e).toFixed(decimals);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ------------------------------------------------------------------ input */
  el.drop.addEventListener("click", () => el.file.click());
  el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); } });
  el.file.addEventListener("change", () => el.file.files[0] && setFile(el.file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) setFile(f); });

  let sampleIdx = Math.floor(Math.random() * 4);
  el.sampleBtn.addEventListener("click", async () => {
    try {
      sampleIdx = (sampleIdx % 4) + 1;
      const r = await fetch(`samples/sample_${sampleIdx}.jpg`);
      if (!r.ok) throw new Error("missing sample");
      const blob = await r.blob();
      setFile(new File([blob], `sample_field_${sampleIdx}.jpg`, { type: "image/jpeg" }));
    } catch { toast("Could not load sample image.", true); }
  });

  el.sens.addEventListener("input", () => { el.sensOut.textContent = el.sens.value; rerun(); });
  el.area.addEventListener("input", () => { el.areaOut.textContent = el.area.value + " px"; rerun(); });
  $("#modeSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#modeSeg button").forEach((x) => x.classList.toggle("on", x === b));
    state.mode = b.dataset.v; rerun();
  });
  const rerun = debounce(() => { if (state.file) analyze(); }, 450);

  $("#toggles").addEventListener("change", (e) => {
    const cb = e.target.closest("input[data-layer]"); if (!cb) return;
    state.layers[cb.dataset.layer] = cb.checked;
    applyLayers();
  });

  function applyLayers() {
    $$(".layer", el.stage).forEach((img) => { img.hidden = !(state.layers[img.dataset.name] && img.getAttribute("src")); });
    draw();
  }

  /* ------------------------------------------------------------------ analysis */
  function setFile(file) {
    if (!file.type.startsWith("image/")) return toast("Please choose an image file.", true);
    if (file.size > 20 * 1024 * 1024) return toast("Image is larger than 20 MB.", true);
    state.file = file;
    state.result = null;
    $$(".layer", el.stage).forEach((i) => { i.removeAttribute("src"); i.hidden = true; });
    el.base.src = URL.createObjectURL(file);
    el.empty.hidden = true; el.stage.hidden = false;
    el.viewTitle.textContent = file.name;
    clearCanvas();
    el.base.onload = () => { sizeCanvas(); analyze(); };
  }

  const MSGS = ["Segmenting vegetation…", "Finding crop rows…", "Classifying plants…", "Building weed heat map…"];
  let msgTimer;
  function setBusy(on) {
    state.busy = on;
    el.scan.hidden = !on;
    el.holdBtn.disabled = el.pngBtn.disabled = on || !state.result;
    clearInterval(msgTimer);
    if (on) { let i = 0; el.scanMsg.textContent = MSGS[0]; msgTimer = setInterval(() => { el.scanMsg.textContent = MSGS[++i % MSGS.length]; }, 900); }
  }

  async function analyze() {
    if (!state.file) return;
    setBusy(true);
    try {
      const t0 = performance.now();
      await enginePromise;
      const data = await engine.analyze(state.file, { sensitivity: +el.sens.value, min_area: +el.area.value, mode: state.mode });
      // keep the scan animation visible long enough to feel intentional
      const wait = Math.max(0, 900 - (performance.now() - t0));
      await new Promise((res) => setTimeout(res, wait));
      showResult(data, true);
      loadHistory(); loadStats();
    } catch (err) {
      toast(err.message || "Could not reach the server.", true);
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------------ results */
  function showResult(res, scroll = false) {
    state.result = res; state.selected = null; state.hover = null;
    state.stageLoaded = new Promise((resolve) => {
      let pending = 1;
      const done = () => { if (--pending === 0) resolve(); };
      el.base.onload = () => { sizeCanvas(); draw(); done(); };
      el.base.src = res.urls.original;
      ["crop", "weed", "mask", "heat"].forEach((n) => {
        const img = $(`.layer[data-name="${n}"]`, el.stage);
        pending++;
        img.onload = img.onerror = done;
        img.src = res.urls[n];
      });
    });
    el.empty.hidden = true; el.stage.hidden = false;
    el.viewTitle.textContent = res.filename || "Field image";
    el.engineNote.textContent = `${res.processing_ms} ms · ${res.method.classifier === "rows" ? "row-geometry" : "size/shape"} classifier`;
    applyLayers();
    setBusy(false);
    el.holdBtn.disabled = el.pngBtn.disabled = false;
    el.results.hidden = false;
    renderSummary(res);
    renderTable();
    $("#csvBtn").href = res.urls.csv; $("#jsonBtn").href = res.urls.json;
    $("#csvBtn").download = (res.names && res.names.csv) || ""; $("#jsonBtn").download = (res.names && res.names.json) || "";
    // reflect the settings used
    el.sens.value = res.method.sensitivity; el.sensOut.textContent = res.method.sensitivity;
    el.area.value = res.method.min_area; el.areaOut.textContent = res.method.min_area + " px";
    state.mode = res.method.mode_requested;
    $$("#modeSeg button").forEach((b) => b.classList.toggle("on", b.dataset.v === state.mode));
    if (scroll) setTimeout(() => $("#verdict").scrollIntoView({ behavior: "smooth", block: "start" }), 150);
  }

  function renderSummary(res) {
    const s = res.summary, color = LEVELS[s.pressure_level] || LEVELS.Low;
    const v = $("#verdict");
    v.style.setProperty("--lvl", color);
    v.innerHTML = `<div><span class="badge">${esc(s.pressure_level)} weed pressure</span></div>
      <div><h2>${s.weed_ratio_pct.toFixed(1)}% of plant cover is weeds</h2><p>${esc(res.recommendations[0])}</p></div>`;

    const cards = [
      ["Crops detected", s.crop_count, 0, "", "plants on rows"],
      ["Weeds detected", s.weed_count, 0, "", "clusters & plants"],
      ["Weed cover", s.weed_cover_pct, 1, "%", "of the whole image"],
      ["Plant cover", s.vegetation_cover_pct, 1, "%", "crop + weed"],
      ["Herbicide saved", s.herbicide_saving_pct, 0, "%", "spot vs blanket spray"],
      ["Analysis time", res.processing_ms, 0, "ms", `${res.width}×${res.height}px`],
    ];
    $("#cards").innerHTML = cards.map(([k, , , u, d]) => `<div class="card"><div class="k">${k}</div><div class="v"><span class="n">0</span>${u ? `<small>${u}</small>` : ""}</div><div class="d">${d}</div></div>`).join("");
    $$("#cards .n").forEach((n, i) => countUp(n, cards[i][1], cards[i][2]));

    // donut
    const w = s.weed_ratio_pct, c = 100 - w, R = 52, C = 2 * Math.PI * R;
    const hasVeg = s.crop_count + s.weed_count > 0;
    $("#donut").innerHTML = `<svg width="150" height="150" viewBox="0 0 150 150" role="img" aria-label="Crop ${c.toFixed(1)}% weed ${w.toFixed(1)}%">
      <circle cx="75" cy="75" r="${R}" fill="none" stroke="#eae7d8" stroke-width="20"/>
      ${hasVeg ? `<circle cx="75" cy="75" r="${R}" fill="none" stroke="${COLORS.crop}" stroke-width="20" stroke-dasharray="${(c / 100) * C} ${C}" transform="rotate(-90 75 75)"/>
      <circle cx="75" cy="75" r="${R}" fill="none" stroke="${COLORS.weed}" stroke-width="20" stroke-dasharray="${(w / 100) * C} ${C}" stroke-dashoffset="${-(c / 100) * C}" transform="rotate(-90 75 75)"/>` : ""}
      <text x="75" y="75" text-anchor="middle" font-family="Georgia,serif" font-size="26" font-weight="600" fill="#14231a">${w.toFixed(0)}%</text>
      <text x="75" y="94" text-anchor="middle" font-size="11" fill="#62705f">weeds</text></svg>`;
    $("#donutLegend").innerHTML = `
      <li><i style="background:${COLORS.crop}"></i>Crop <b>${c.toFixed(1)}%</b></li>
      <li><i style="background:${COLORS.weed}"></i>Weed <b>${w.toFixed(1)}%</b></li>`;

    const r = res.rows;
    $("#rowInfo").innerHTML = r.detected
      ? `<b>${r.count} crop rows</b> detected at <b>${r.angle_deg}°</b>, spacing ≈ <b>${r.spacing_px}px</b> · pattern confidence <b>${Math.round(r.confidence * 100)}%</b>`
      : `No clear row pattern found — classification used plant <b>size &amp; shape</b>.`;

    // spray grid
    const g = res.grid, sp = $("#spray");
    sp.style.gridTemplateColumns = `repeat(${g.cols}, 1fr)`;
    let html = "";
    g.weed_fraction.forEach((row, y) => row.forEach((f, x) => {
      const hit = f > g.flag_threshold;
      const a = hit ? Math.min(1, 0.35 + f * 6) : 0;
      html += `<div class="${hit ? "hit" : ""}" ${hit ? `style="opacity:${a.toFixed(2)};background:var(--weed)"` : ""} data-t="${hit ? (f * 100).toFixed(1) + "% weeds" : "clear"}"></div>`;
    }));
    sp.innerHTML = html;
    $("#sprayFoot").innerHTML = `<div><b>${s.spray_area_pct}%</b>of field to treat</div><div><b>${s.herbicide_saving_pct}%</b>estimated herbicide saved</div>`;

    $("#recs").innerHTML = res.recommendations.slice(1).map((r) => `<li>${esc(r)}</li>`).join("");
  }

  /* ------------------------------------------------------------------ table */
  function renderTable() {
    const res = state.result; if (!res) return;
    const list = res.detections.filter((d) => state.filter === "all" || d.label === state.filter)
      .sort((a, b) => b.area_px - a.area_px);
    $("#detCount").textContent = `(${list.length})`;
    $("#dets tbody").innerHTML = list.map((d) => `<tr data-id="${d.id}" class="${d.id === state.selected ? "sel" : ""}">
      <td><b>${d.id}</b></td><td><span class="tag ${d.label}">${d.label}</span></td>
      <td><span class="bar"><i style="width:${d.confidence * 100}%"></i></span>${Math.round(d.confidence * 100)}%</td>
      <td>${Math.round(d.centroid[0])}, ${Math.round(d.centroid[1])}</td>
      <td>${d.bbox[2]}×${d.bbox[3]}</td><td>${d.area_pct.toFixed(2)}%</td></tr>`).join("");
  }
  $("#filterSeg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#filterSeg button").forEach((x) => x.classList.toggle("on", x === b));
    state.filter = b.dataset.v; renderTable();
  });
  $("#dets tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr"); if (!tr) return;
    state.selected = state.selected === tr.dataset.id ? null : tr.dataset.id;
    renderTable(); draw();
    if (state.selected) $("#detect").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  /* ------------------------------------------------------------------ canvas */
  function sizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    el.canvas.width = Math.round(el.stage.clientWidth * dpr);
    el.canvas.height = Math.round(el.stage.clientHeight * dpr);
  }
  function clearCanvas() { ctx.clearRect(0, 0, el.canvas.width, el.canvas.height); }

  function paintBoxes(c, scale, lw, opts = {}) {
    const res = state.result; if (!res) return;
    c.lineJoin = "round";
    const ordered = [...res.detections].sort((a, b) => (a.id === state.selected) - (b.id === state.selected));
    for (const d of ordered) {
      const [x, y, w, h] = d.bbox;
      const active = d.id === state.selected || d.id === state.hover;
      const col = COLORS[d.label];
      c.strokeStyle = col;
      c.lineWidth = lw * (active ? 2.2 : 1);
      c.globalAlpha = active ? 1 : 0.9;
      const pad = 2 * scale;
      c.strokeRect(x * scale - pad, y * scale - pad, w * scale + pad * 2, h * scale + pad * 2);
      if (active || opts.labels) {
        const label = `${d.id} · ${d.label} ${Math.round(d.confidence * 100)}%`;
        const fs = Math.max(11, 12 * (opts.fontScale || 1));
        c.font = `700 ${fs}px system-ui, sans-serif`;
        const tw = c.measureText(label).width + 12 * (opts.fontScale || 1);
        const ty = Math.max(0, y * scale - fs - 10);
        c.globalAlpha = 1; c.fillStyle = col;
        c.fillRect(x * scale - pad, ty, tw, fs + 8);
        c.fillStyle = "#fff"; c.textBaseline = "middle";
        c.fillText(label, x * scale - pad + 6, ty + (fs + 8) / 2);
      }
    }
    c.globalAlpha = 1;
  }

  function draw() {
    clearCanvas();
    if (!state.result || !state.layers.boxes) return;
    const dpr = window.devicePixelRatio || 1;
    const scale = (el.stage.clientWidth * dpr) / state.result.width;
    paintBoxes(ctx, scale, 1.6 * dpr, { fontScale: dpr });
  }

  function hit(e) {
    const rect = el.canvas.getBoundingClientRect();
    const k = state.result.width / rect.width;
    const x = (e.clientX - rect.left) * k, y = (e.clientY - rect.top) * k;
    let best = null;
    for (const d of state.result.detections) {
      const [bx, by, bw, bh] = d.bbox;
      if (x >= bx - 3 && x <= bx + bw + 3 && y >= by - 3 && y <= by + bh + 3) {
        if (!best || d.area_px < best.area_px) best = d;
      }
    }
    return { d: best, rect };
  }
  el.canvas.addEventListener("mousemove", (e) => {
    if (!state.result || !state.layers.boxes) return;
    const { d, rect } = hit(e);
    const id = d ? d.id : null;
    if (id !== state.hover) { state.hover = id; draw(); }
    if (d) {
      el.tip.hidden = false;
      el.tip.innerHTML = `${d.id} · ${d.label}<br><span>${Math.round(d.confidence * 100)}% confidence · ${d.area_pct.toFixed(2)}% of image</span>`;
      const tx = Math.min(e.clientX - rect.left + 14, rect.width - 190), ty = e.clientY - rect.top + 14;
      el.tip.style.left = tx + "px"; el.tip.style.top = ty + "px";
    } else el.tip.hidden = true;
  });
  el.canvas.addEventListener("mouseleave", () => { state.hover = null; el.tip.hidden = true; draw(); });
  el.canvas.addEventListener("click", (e) => {
    if (!state.result) return;
    const { d } = hit(e);
    state.selected = d && d.id !== state.selected ? d.id : null;
    renderTable(); draw();
    const tr = state.selected && $(`#dets tr[data-id="${state.selected}"]`);
    if (tr) tr.scrollIntoView({ block: "nearest" });
  });
  new ResizeObserver(() => { sizeCanvas(); draw(); }).observe(el.stage);

  /* hold to compare */
  const cmp = (on) => el.stage.classList.toggle("compare", on);
  ["mousedown", "touchstart"].forEach((ev) => el.holdBtn.addEventListener(ev, () => cmp(true), { passive: true }));
  ["mouseup", "mouseleave", "touchend"].forEach((ev) => el.holdBtn.addEventListener(ev, () => cmp(false)));
  window.addEventListener("keydown", (e) => { if (e.code === "Space" && state.result && !/INPUT|BUTTON|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); cmp(true); } });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") cmp(false); });

  /* save annotated PNG */
  el.pngBtn.addEventListener("click", () => {
    const res = state.result; if (!res) return;
    const cv = document.createElement("canvas");
    cv.width = res.width; cv.height = res.height;
    const c = cv.getContext("2d");
    c.drawImage(el.base, 0, 0, res.width, res.height);
    ["heat", "mask", "crop", "weed"].forEach((n) => {
      const img = $(`.layer[data-name="${n}"]`, el.stage);
      if (state.layers[n] && img.complete && img.naturalWidth) c.drawImage(img, 0, 0, res.width, res.height);
    });
    if (state.layers.boxes) {
      const sel = state.selected, hov = state.hover; state.selected = state.hover = null;
      paintBoxes(c, 1, Math.max(2, res.width / 700));
      state.selected = sel; state.hover = hov;
    }
    cv.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `fieldsense_${(res.filename || "field").replace(/\.[^.]+$/, "")}_annotated.png`;
      a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });
  });

  /* ------------------------------------------------------------------ history + stats */
  async function loadHistory() {
    try {
      await enginePromise;
      const list = await engine.list();
      const box = $("#hist");
      if (!list.length) { box.innerHTML = `<div class="hist-empty">No scans yet — analyze an image and it will appear here.</div>`; return; }
      box.innerHTML = list.map((a) => `<div class="hcard" data-id="${a.id}" style="--lvl:${LEVELS[a.pressure] || LEVELS.Low}">
        <span class="lvl">${esc(a.pressure)}</span><button class="del" title="Delete" aria-label="Delete scan">×</button>
        <img src="${a.thumb}" alt="" loading="lazy">
        <div class="m"><b>${esc(a.filename || "field")}</b>${a.crop_count} crops · ${a.weed_count} weeds · ${a.weed_ratio.toFixed(1)}% weed<br>${fmtDate(a.created_at)}</div></div>`).join("");
    } catch { /* offline */ }
  }
  $("#hist").addEventListener("click", async (e) => {
    const card = e.target.closest(".hcard"); if (!card) return;
    const id = card.dataset.id;
    if (e.target.closest(".del")) {
      e.stopPropagation();
      await engine.remove(id);
      toast("Scan deleted."); loadHistory(); loadStats(); return;
    }
    try {
      setBusy(true);
      const res = await engine.get(id);
      state.file = null;
      showResult(res, false);
      $("#detect").scrollIntoView({ behavior: "smooth" });
    } catch (err) { toast(err.message, true); setBusy(false); }
  });

  async function loadStats() {
    try {
      await enginePromise;
      const s = await engine.stats();
      $("#navStats").innerHTML = `<span><b>${s.analyses}</b> scans</span><span><b>${s.plants_classified.toLocaleString()}</b> plants classified</span>` +
        `<span class="eng" title="${esc(engine.note)}">● ${esc(engine.label)}</span>`;
    } catch { /* ignore */ }
  }

  loadHistory(); loadStats();
})();
