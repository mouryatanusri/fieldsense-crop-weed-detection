"""FieldSense - crop / weed detection API + static frontend server."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, abort, g, jsonify, request, send_from_directory

from detector import analyze
from sample_field import generate_field

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
DB_PATH = os.path.join(DATA_DIR, "fieldsense.db")
FRONTEND = os.path.join(os.path.dirname(BASE), "frontend")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


# ----------------------------------------------------------------------------- db
def db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT,
                width INTEGER, height INTEGER,
                crop_count INTEGER, weed_count INTEGER,
                weed_ratio REAL, pressure TEXT,
                saving REAL,
                result_json TEXT NOT NULL
            )"""
        )


init_db()


# ----------------------------------------------------------------------------- helpers
def error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def save_run(run: dict, filename: str) -> dict:
    rid = uuid.uuid4().hex[:12]
    folder = os.path.join(DATA_DIR, rid)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "original.jpg"), run["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    for name, layer in run["layers"].items():
        cv2.imwrite(os.path.join(folder, f"{name}.png"), layer)

    result = run["result"]
    result["id"] = rid
    result["filename"] = filename
    result["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    s = result["summary"]
    db().execute(
        "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (rid, result["created_at"], filename, result["width"], result["height"],
         s["crop_count"], s["weed_count"], s["weed_ratio_pct"], s["pressure_level"],
         s["herbicide_saving_pct"], json.dumps(result)),
    )
    db().commit()
    return with_urls(result)


def with_urls(result: dict) -> dict:
    rid = result["id"]
    base = f"/api/analyses/{rid}/files"
    result["urls"] = {
        "original": f"{base}/original.jpg",
        "crop": f"{base}/crop.png",
        "weed": f"{base}/weed.png",
        "mask": f"{base}/mask.png",
        "heat": f"{base}/heat.png",
        "csv": f"/api/analyses/{rid}/export.csv",
        "json": f"/api/analyses/{rid}/export.json",
    }
    return result


# ----------------------------------------------------------------------------- api
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "engine": "classical-cv", "version": "1.0.0"})


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("image")
    if f is None or not f.filename:
        return error("No image uploaded. Send a multipart form field named 'image'.")
    img = decode_image(f.read())
    if img is None:
        return error("Could not read that file as an image. Use JPG, PNG, WEBP or BMP.")
    if min(img.shape[:2]) < 120:
        return error("Image is too small (minimum 120px on each side).")

    try:
        sens = int(request.form.get("sensitivity", 50))
        min_area = int(request.form.get("min_area", 60))
    except ValueError:
        return error("sensitivity and min_area must be integers.")
    mode = request.form.get("mode", "auto")
    if mode not in ("auto", "rows", "size"):
        return error("mode must be one of: auto, rows, size.")

    run = analyze(img, sensitivity=sens, min_area=min_area, mode=mode)
    return jsonify(save_run(run, os.path.basename(f.filename)))


@app.get("/api/sample")
def api_sample():
    try:
        seed = int(request.args.get("seed", 7))
    except ValueError:
        seed = 7
    img = generate_field(seed=seed)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), mimetype="image/jpeg",
                    headers={"Content-Disposition": f'inline; filename="sample_field_{seed}.jpg"'})


@app.get("/api/analyses")
def api_list():
    rows = db().execute(
        "SELECT id, created_at, filename, crop_count, weed_count, weed_ratio, pressure, saving "
        "FROM analyses ORDER BY created_at DESC LIMIT 60"
    ).fetchall()
    return jsonify([
        {**dict(r), "thumb": f"/api/analyses/{r['id']}/files/original.jpg"} for r in rows
    ])


@app.get("/api/analyses/<rid>")
def api_get(rid):
    row = db().execute("SELECT result_json FROM analyses WHERE id=?", (rid,)).fetchone()
    if not row:
        return error("Analysis not found.", 404)
    return jsonify(with_urls(json.loads(row["result_json"])))


@app.delete("/api/analyses/<rid>")
def api_delete(rid):
    cur = db().execute("DELETE FROM analyses WHERE id=?", (rid,))
    db().commit()
    if cur.rowcount == 0:
        return error("Analysis not found.", 404)
    if rid.isalnum():
        shutil.rmtree(os.path.join(DATA_DIR, rid), ignore_errors=True)
    return jsonify({"deleted": rid})


@app.get("/api/analyses/<rid>/files/<name>")
def api_file(rid, name):
    if not rid.isalnum() or name not in ("original.jpg", "crop.png", "weed.png", "mask.png", "heat.png"):
        abort(404)
    return send_from_directory(os.path.join(DATA_DIR, rid), name, max_age=3600)


@app.get("/api/analyses/<rid>/export.json")
def api_export_json(rid):
    row = db().execute("SELECT result_json FROM analyses WHERE id=?", (rid,)).fetchone()
    if not row:
        return error("Analysis not found.", 404)
    return Response(row["result_json"], mimetype="application/json",
                    headers={"Content-Disposition": f'attachment; filename="fieldsense_{rid}.json"'})


@app.get("/api/analyses/<rid>/export.csv")
def api_export_csv(rid):
    row = db().execute("SELECT result_json FROM analyses WHERE id=?", (rid,)).fetchone()
    if not row:
        return error("Analysis not found.", 404)
    result = json.loads(row["result_json"])
    out = io.StringIO()
    wr = csv.writer(out)
    wr.writerow(["id", "label", "confidence", "x", "y", "width", "height", "area_px", "area_pct_of_image"])
    for d in result["detections"]:
        x, y, w, h = d["bbox"]
        wr.writerow([d["id"], d["label"], d["confidence"], x, y, w, h, d["area_px"], d["area_pct"]])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="fieldsense_{rid}.csv"'})


@app.get("/api/stats")
def api_stats():
    r = db().execute(
        "SELECT COUNT(*) n, COALESCE(AVG(weed_ratio),0) avg_ratio, COALESCE(SUM(crop_count),0) crops, "
        "COALESCE(SUM(weed_count),0) weeds, COALESCE(AVG(saving),0) avg_saving FROM analyses"
    ).fetchone()
    return jsonify({
        "analyses": r["n"], "avg_weed_ratio": round(r["avg_ratio"], 1),
        "plants_classified": r["crops"] + r["weeds"], "avg_saving": round(r["avg_saving"], 1),
    })


@app.errorhandler(413)
def too_large(_e):
    return error("File too large (max 20 MB).", 413)


# ----------------------------------------------------------------------------- frontend
@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/<path:path>")
def static_files(path):
    if path.startswith("api/"):
        abort(404)
    return send_from_directory(FRONTEND, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  FieldSense running ->  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
