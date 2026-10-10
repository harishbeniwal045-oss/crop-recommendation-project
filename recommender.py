"""
Core recommendation logic shared by the web app (app.py) and the command-line
tool (src/predict.py).

Pipeline
  1. Validate the farmer's inputs.
  2. Drought risk (season-aware, rule based).
  3. ML model (RandomForest) gives a suitability probability for every crop.
     Only crops that are grown in the chosen season are kept.
  4. Drought-resilient ranking: final score = suitability * (1 - a + a * tolerance),
     where a grows with drought risk (LOW 0.1, MEDIUM 0.3, HIGH 0.5). A crop must be
     suitable first; tolerance then scales its score down when the risk is high.
  5. Variety ranking for the best crop: drought tolerance, water need vs the water
     the farmer has, maturity time, season and region.
"""
import json
import os
import sys

import joblib
import pandas as pd
import sklearn

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from drought_risk import assess_drought_risk  # noqa: E402
from generate_dataset import (  # noqa: E402
    BASE_DIR, DATA_PATH, PROFILES, SEASONS, SOIL_TYPES, WATER_LEVELS,
)
from train_model import FEATURES, METRICS_PATH, MODEL_PATH, train  # noqa: E402

VARIETY_PATH = os.path.join(BASE_DIR, "data", "variety_data.csv")

REGIONS = ["North", "South", "East", "West", "Central"]
SEASON_LABELS = {"Kharif": "Kharif (monsoon, Jun-Oct)",
                 "Rabi": "Rabi (winter, Nov-Mar)",
                 "Zaid": "Zaid (summer, Mar-Jun)"}

TOLERANCE_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.2}
RISK_ALPHA = {"LOW": 0.10, "MEDIUM": 0.30, "HIGH": 0.50}      # weight of drought tolerance
VARIETY_WEIGHTS = {                                           # (tolerance, water fit, early maturity)
    "HIGH": (0.5, 0.3, 0.2),
    "MEDIUM": (0.4, 0.4, 0.2),
    "LOW": (0.2, 0.5, 0.3),
}
LEVEL = {"low": 0, "medium": 1, "high": 2}

LIMITS = {                           # (min, max) accepted for each numeric input
    "temperature": (0, 50),
    "rainfall": (0, 3000),
    "humidity": (0, 100),
    "soil_ph": (3, 10),
}
LABELS = {"temperature": "Temperature", "rainfall": "Rainfall",
          "humidity": "Humidity", "soil_ph": "Soil pH"}

_model = None
_varieties = None


# ------------------------------------------------------------------ loading
def load_model():
    """Load the saved model; retrain automatically if it is missing or was
    built with a different scikit-learn version."""
    global _model
    if _model is not None:
        return _model
    needs_training = not os.path.exists(MODEL_PATH)
    if not needs_training and os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                if json.load(f).get("sklearn_version") != sklearn.__version__:
                    needs_training = True
        except (OSError, ValueError):
            needs_training = True
    if not needs_training:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            needs_training = True
    if needs_training:
        print("Training model (first run)...")
        _model, _ = train(save=True, verbose=False)
    return _model


def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_varieties():
    global _varieties
    if _varieties is None:
        _varieties = pd.read_csv(VARIETY_PATH)
    return _varieties


def crops_for_season(season):
    return [c for c, info in PROFILES.items() if season in info["seasons"]]


# --------------------------------------------------------------- validation
def validate_inputs(form):
    """Return (clean_inputs, errors). `form` is any dict-like of strings."""
    clean, errors = {}, []
    for key, (lo, hi) in LIMITS.items():
        raw = str(form.get(key, "")).strip()
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{LABELS[key]} must be a number.")
            continue
        if not (lo <= value <= hi):
            errors.append(f"{LABELS[key]} must be between {lo} and {hi}.")
        else:
            clean[key] = value

    choices = {
        "water_availability": WATER_LEVELS,
        "soil_type": SOIL_TYPES,
        "season": SEASONS,
    }
    for key, allowed in choices.items():
        value = str(form.get(key, "")).strip()
        if key != "season":
            value = value.lower()
        if value not in allowed:
            errors.append("Please select a valid " + key.replace("_", " ") + ".")
        else:
            clean[key] = value

    region = str(form.get("region", "Any")).strip() or "Any"
    if region != "Any" and region not in REGIONS:
        errors.append("Please select a valid region.")
    else:
        clean["region"] = region
    return clean, errors


# ------------------------------------------------------------------ ranking
def rank_crops(inputs, risk_level):
    model = load_model()
    season = inputs["season"]
    row = pd.DataFrame([{k: inputs[k] for k in FEATURES}])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)

    allowed = crops_for_season(season)
    raw = {c: float(probabilities[classes.index(c)]) for c in allowed if c in classes}
    total = sum(raw.values())
    if total <= 0:                               # model has no opinion: treat all equally
        raw = {c: 1.0 for c in raw}
        total = float(len(raw))
    top = max(raw.values())
    alpha = RISK_ALPHA[risk_level]

    rows = []
    for crop, p in raw.items():
        tolerance = PROFILES[crop]["tolerance"]
        relative = (p / top) if top > 0 else 0.0
        final = relative * ((1 - alpha) + alpha * TOLERANCE_SCORE[tolerance])
        rows.append({
            "crop": crop,
            "suitability": round(100 * p / total, 1),
            "tolerance": tolerance,
            "final_score": round(100 * final, 1),
        })
    rows.sort(key=lambda r: (r["final_score"], r["suitability"]), reverse=True)
    return rows


def rank_varieties(crop, inputs, risk_level):
    df = load_varieties()
    season, region = inputs["season"], inputs["region"]
    water = LEVEL[inputs["water_availability"]]
    candidates = df[df["crop"] == crop].copy()
    note = None
    if candidates.empty:
        return [], "No varieties of this crop are listed in the variety database."

    in_season = candidates[candidates["season"].str.split(";").apply(lambda s: season in s)]
    if in_season.empty:
        note = (f"No {crop} variety in the database is listed specifically for {season}; "
                "showing other varieties of this crop. Check local recommendations.")
    else:
        candidates = in_season

    w_tol, w_water, w_mat = VARIETY_WEIGHTS[risk_level]
    shortest = candidates["maturity_days"].min()
    longest = candidates["maturity_days"].max()
    span = max(longest - shortest, 1)

    out = []
    for _, v in candidates.iterrows():
        tol = TOLERANCE_SCORE.get(str(v["drought_tolerance"]).lower(), 0.5)
        gap = LEVEL[str(v["water_requirement"]).lower()] - water
        water_fit = 1.0 if gap <= 0 else (0.4 if gap == 1 else 0.1)
        maturity = 0.5 if pd.isna(v["maturity_days"]) else 1 - (v["maturity_days"] - shortest) / span
        score = w_tol * tol + w_water * water_fit + w_mat * maturity

        zones = str(v["region"]).split(";")
        if region == "Any":
            region_text = "Not filtered"
        elif region in zones:
            score += 0.15
            region_text = "Suited to your region"
        elif "All" in zones:
            score += 0.05
            region_text = "Grown across India"
        else:
            region_text = "Released for " + ", ".join(zones) + " - may not suit your region"

        out.append({
            "variety": v["variety"],
            "tolerance": str(v["drought_tolerance"]),
            "maturity_days": None if pd.isna(v["maturity_days"]) else int(v["maturity_days"]),
            "water_requirement": str(v["water_requirement"]),
            "region": "All India" if "All" in zones else ", ".join(zones),
            "region_match": region_text,
            "remarks": str(v["remarks"]),
            "score": round(100 * score, 1),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:3], note


def range_check(crop, inputs):
    """Compare the farmer's values with the typical range of the crop in this season."""
    profile = PROFILES[crop]["seasons"][inputs["season"]]
    rows = []
    for key, unit in (("temperature", "C"), ("rainfall", "mm"), ("humidity", "%"), ("soil_ph", "")):
        lo, hi = profile[key]
        value = inputs[key]
        rows.append({
            "name": LABELS[key], "value": f"{value:g} {unit}".strip(),
            "typical": f"{lo:g} - {hi:g} {unit}".strip(), "ok": lo <= value <= hi,
        })
    water_ok = inputs["water_availability"] in profile["water"]
    soil_ok = inputs["soil_type"] in profile["soil"]
    rows.append({"name": "Water availability", "value": inputs["water_availability"].title(),
                 "typical": ", ".join(w.title() for w in profile["water"]), "ok": water_ok})
    rows.append({"name": "Soil type", "value": inputs["soil_type"].replace("_", " ").title(),
                 "typical": ", ".join(s.replace("_", " ").title() for s in profile["soil"]), "ok": soil_ok})
    return rows


def confidence_level(crops):
    """How clear-cut is the result? Based on the best crop's suitability and how far
    it is ahead of the runner-up (crops of one season overlap, so close calls happen)."""
    top = crops[0]["suitability"]
    lead = top - (crops[1]["suitability"] if len(crops) > 1 else 0)
    if top >= 50 and lead >= 15:
        return "High"
    if top >= 35 and lead >= 5:
        return "Medium"
    return "Low"


def recommend(inputs):
    """Run the full pipeline on validated inputs and return a result dict."""
    season = inputs["season"]
    drought = assess_drought_risk(inputs["temperature"], inputs["rainfall"],
                                  inputs["humidity"], inputs["water_availability"], season)
    crops = rank_crops(inputs, drought["level"])
    best = crops[0]["crop"]
    varieties, variety_note = rank_varieties(best, inputs, drought["level"])
    checks = range_check(best, inputs)

    notes = []
    if crops[0]["suitability"] < 30:
        notes.append("Suitability is low for every crop; your conditions are unusual for "
                     "this season. Treat the result with caution.")
    outside = [c["name"] for c in checks if not c["ok"]]
    if outside:
        notes.append("Outside the typical range for " + best + ": " + ", ".join(outside) + ".")

    confidence = confidence_level(crops)
    if confidence == "Low" and len(crops) > 1:
        notes.append(f"Confidence is low: {crops[0]['crop']} and {crops[1]['crop']} are close. "
                     "Consider both before deciding.")

    alpha = RISK_ALPHA[drought["level"]]
    explanation = list(drought["reasons"])
    explanation.append(
        f"Drought risk is {drought['level']}, so the drought-tolerance weight is {int(alpha * 100)}% "
        f"({season} season): crops with low tolerance are scored down."
    )
    return {
        "season": season,
        "drought_risk": drought["level"],
        "drought_score": drought["score"],
        "drought_max": drought["max_score"],
        "best_crop": best,
        "best_tolerance": crops[0]["tolerance"],
        "confidence": confidence,
        "crops": crops,
        "varieties": varieties,
        "best_variety": varieties[0]["variety"] if varieties else "Not available",
        "variety_note": variety_note,
        "explanation": explanation,
        "range_check": checks,
        "notes": notes,
        "metrics": load_metrics(),
    }
