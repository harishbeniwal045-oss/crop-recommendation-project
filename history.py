"""History of every recommendation (inputs + output), stored in a SQLite file.

SQLite ships with Python, so no new package is needed. The file is
data/history.db (override with the HISTORY_DB environment variable).
Nothing personal is stored: no name, no IP address - only the farm conditions
that were typed in and the recommendation that came out.
"""
import csv
import hashlib
import io
import json
import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FIELDS = ["season", "region", "temperature", "rainfall", "humidity",
                "soil_ph", "water_availability", "soil_type"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,          -- UTC, ISO 8601
    source        TEXT NOT NULL,          -- web / api / cli / rerun
    season        TEXT, region TEXT,
    temperature   REAL, rainfall REAL, humidity REAL, soil_ph REAL,
    water_availability TEXT, soil_type TEXT,
    input_hash    TEXT NOT NULL,          -- same inputs => same hash
    drought_risk  TEXT, best_crop TEXT, best_variety TEXT, confidence TEXT,
    model_version TEXT,                   -- which model/data produced it
    result_json   TEXT NOT NULL           -- full stored output
);
CREATE INDEX IF NOT EXISTS idx_hist_created ON history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hist_hash ON history(input_hash);
"""


def db_path():
    return os.environ.get("HISTORY_DB") or os.path.join(BASE_DIR, "data", "history.db")


def _connect():
    path = db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _num(v):
    return round(float(v), 4)


def normalize_inputs(inputs):
    """Stable form of the inputs so equal inputs always give an equal hash."""
    out = {}
    for k in INPUT_FIELDS:
        v = inputs.get(k)
        out[k] = _num(v) if k in ("temperature", "rainfall", "humidity", "soil_ph") else str(v or "")
    return out


def input_hash(inputs):
    blob = json.dumps(normalize_inputs(inputs), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


_version = None


def model_version():
    """Short fingerprint of the trained model + variety data. If either changes
    (retrained, new varieties), history shows the entry came from an older version."""
    global _version
    if _version is None:
        h = hashlib.sha256()
        for rel in ("model/crop_model.pkl", "data/variety_data.csv"):
            try:
                with open(os.path.join(BASE_DIR, rel), "rb") as f:
                    h.update(f.read())
            except OSError:
                h.update(rel.encode())
        _version = h.hexdigest()[:8]
    return _version


def refresh_model_version():
    """Call after retraining inside a running process."""
    global _version
    _version = None


def _clean(result):
    r = dict(result)
    r.pop("metrics", None)          # large and identical for every run; re-attached on display
    return r


def save(inputs, result, source="web"):
    n = normalize_inputs(inputs)
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO history (created_at, source, season, region, temperature, rainfall,
               humidity, soil_ph, water_availability, soil_type, input_hash, drought_risk,
               best_crop, best_variety, confidence, model_version, result_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source,
             n["season"], n["region"], n["temperature"], n["rainfall"], n["humidity"],
             n["soil_ph"], n["water_availability"], n["soil_type"], input_hash(inputs),
             result["drought_risk"], result["best_crop"], result["best_variety"],
             result["confidence"], model_version(),
             json.dumps(_clean(result), default=float)))
        return cur.lastrowid


def _row(r):
    d = dict(r)
    d["result"] = json.loads(d.pop("result_json"))
    d["inputs"] = {k: d[k] for k in INPUT_FIELDS}
    d["is_current_model"] = d["model_version"] == model_version()
    return d


def get(entry_id):
    with _connect() as conn:
        r = conn.execute("SELECT * FROM history WHERE id=?", (entry_id,)).fetchone()
    return _row(r) if r else None


def _where(filters):
    clauses, params = [], []
    for col in ("season", "best_crop", "drought_risk", "region", "source"):
        v = (filters.get(col) or "").strip()
        if v:
            clauses.append(f"{col}=?")
            params.append(v)
    q = (filters.get("q") or "").strip()
    if q:
        clauses.append("(best_crop LIKE ? OR best_variety LIKE ? OR input_hash=?)")
        params += [f"%{q}%", f"%{q}%", q]
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def search(filters=None, page=1, per_page=15):
    where, params = _where(filters or {})
    page = max(1, int(page))
    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM history{where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM history{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [per_page, (page - 1) * per_page]).fetchall()
    pages = max(1, -(-total // per_page))
    return {"rows": [_row(r) for r in rows], "total": total, "page": page, "pages": pages}


def same_inputs(entry):
    """Every saved run with exactly the same inputs (including this one), oldest first."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM history WHERE input_hash=? ORDER BY id",
                            (entry["input_hash"],)).fetchall()
    return [_row(r) for r in rows]


def stats():
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        crops = conn.execute("SELECT best_crop, COUNT(*) n FROM history GROUP BY best_crop "
                             "ORDER BY n DESC LIMIT 5").fetchall()
        risks = conn.execute("SELECT drought_risk, COUNT(*) n FROM history "
                             "GROUP BY drought_risk").fetchall()
        distinct = conn.execute("SELECT COUNT(DISTINCT input_hash) FROM history").fetchone()[0]
        filters = {
            "seasons": [r[0] for r in conn.execute("SELECT DISTINCT season FROM history ORDER BY 1")],
            "crops": [r[0] for r in conn.execute("SELECT DISTINCT best_crop FROM history ORDER BY 1")],
        }
    return {"total": total, "distinct_inputs": distinct,
            "top_crops": [(r[0], r[1]) for r in crops],
            "risks": {r[0]: r[1] for r in risks}, "filters": filters}


def compare(old, new):
    """Compare a stored result with a fresh one for the same inputs."""
    diffs = []
    for key, label in (("best_crop", "Recommended crop"), ("best_variety", "Recommended variety"),
                       ("drought_risk", "Drought risk"), ("confidence", "Confidence")):
        if old.get(key) != new.get(key):
            diffs.append({"field": label, "old": old.get(key), "new": new.get(key)})
    old_rank = [c["crop"] for c in old.get("crops", [])]
    new_rank = [c["crop"] for c in new.get("crops", [])]
    if old_rank != new_rank:
        diffs.append({"field": "Crop ranking order", "old": " > ".join(old_rank),
                      "new": " > ".join(new_rank)})
    return diffs


def to_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at_utc", "source", *INPUT_FIELDS, "drought_risk", "best_crop",
                "best_variety", "confidence", "model_version", "input_hash"])
    for r in rows:
        w.writerow([r["id"], r["created_at"], r["source"], *[r[k] for k in INPUT_FIELDS],
                    r["drought_risk"], r["best_crop"], r["best_variety"], r["confidence"],
                    r["model_version"], r["input_hash"]])
    return buf.getvalue()
