"""
Generate the (synthetic) training dataset for the crop recommendation model.

Every crop has an agronomic profile for each season it is normally grown in
(Kharif, Rabi, Zaid). Rows are sampled from these profiles, so the model learns
season-specific conditions instead of a single hard-coded season.

NOTE: the data is synthetic (built from typical agronomic ranges), not field
survey data. Say this clearly in your report / presentation.
"""
import os
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "crop_data.csv")

SEED = 42
SAMPLES_PER_COMBO = 60          # rows per (crop, season) combination
NOISE = 0.08                    # ranges are widened by 8% so crops overlap a little

SEASONS = ["Kharif", "Rabi", "Zaid"]
SOIL_TYPES = ["sandy", "sandy_loam", "loamy", "clay"]
WATER_LEVELS = ["low", "medium", "high"]


def P(temp, rain, hum, ph, water, soil):
    """Profile of one crop in one season.
    temp = mean temperature (deg C) in the season
    rain = total rainfall (mm) during the growing season
    """
    return {
        "temperature": temp, "rainfall": rain, "humidity": hum,
        "soil_ph": ph, "water": water, "soil": soil,
    }


PROFILES = {
    "Sorghum": {"tolerance": "high", "seasons": {
        "Kharif": P((28, 36), (300, 650), (35, 60), (6.0, 7.5), ["low", "medium"], ["sandy", "sandy_loam", "loamy"]),
        "Rabi":   P((18, 28), (50, 250),  (35, 60), (6.0, 8.0), ["low", "medium"], ["loamy", "clay"]),
    }},
    "Pearl Millet": {"tolerance": "high", "seasons": {
        "Kharif": P((30, 38), (250, 550), (30, 55), (6.0, 7.8), ["low", "medium"], ["sandy", "sandy_loam", "loamy"]),
        "Zaid":   P((30, 40), (50, 200),  (25, 50), (6.5, 8.0), ["low", "medium"], ["sandy", "sandy_loam"]),
    }},
    "Finger Millet": {"tolerance": "high", "seasons": {
        "Kharif": P((25, 32), (500, 900), (45, 70), (5.5, 7.0), ["low", "medium"], ["sandy_loam", "loamy"]),
    }},
    "Chickpea": {"tolerance": "high", "seasons": {
        "Rabi":   P((15, 27), (60, 250),  (30, 60), (6.0, 8.0), ["low", "medium"], ["sandy_loam", "loamy", "clay"]),
    }},
    "Green Gram": {"tolerance": "high", "seasons": {
        "Kharif": P((27, 35), (350, 700), (40, 70), (6.0, 7.5), ["low", "medium"], ["sandy_loam", "loamy"]),
        "Zaid":   P((28, 38), (50, 200),  (30, 60), (6.0, 7.5), ["low", "medium"], ["sandy_loam", "loamy"]),
    }},
    "Barley": {"tolerance": "high", "seasons": {
        "Rabi":   P((12, 24), (60, 250),  (35, 65), (6.5, 8.0), ["low", "medium"], ["sandy_loam", "loamy"]),
    }},
    "Soybean": {"tolerance": "medium", "seasons": {
        "Kharif": P((24, 32), (600, 1000), (50, 75), (6.0, 7.5), ["medium", "high"], ["loamy", "clay"]),
    }},
    "Cotton": {"tolerance": "medium", "seasons": {
        "Kharif": P((25, 36), (500, 900), (40, 70), (6.0, 8.0), ["medium", "high"], ["sandy_loam", "loamy", "clay"]),
    }},
    "Groundnut": {"tolerance": "medium", "seasons": {
        "Kharif": P((25, 33), (500, 800), (45, 70), (6.0, 7.0), ["low", "medium"], ["sandy", "sandy_loam"]),
        "Rabi":   P((20, 28), (100, 300), (45, 70), (6.0, 7.0), ["medium"], ["sandy", "sandy_loam"]),
        "Zaid":   P((27, 35), (100, 300), (40, 65), (6.0, 7.0), ["medium", "high"], ["sandy", "sandy_loam"]),
    }},
    "Wheat": {"tolerance": "medium", "seasons": {
        "Rabi":   P((12, 25), (100, 350), (40, 70), (6.0, 7.8), ["medium", "high"], ["sandy_loam", "loamy", "clay"]),
    }},
    "Mustard": {"tolerance": "medium", "seasons": {
        "Rabi":   P((12, 25), (50, 250),  (35, 65), (6.0, 8.0), ["low", "medium"], ["sandy_loam", "loamy"]),
    }},
    "Rice": {"tolerance": "low", "seasons": {
        "Kharif": P((24, 35), (800, 1500), (60, 85), (5.5, 7.5), ["high"], ["clay", "loamy"]),
        "Rabi":   P((20, 30), (100, 300),  (60, 85), (5.5, 7.5), ["high"], ["clay", "loamy"]),
        "Zaid":   P((28, 38), (100, 300),  (55, 80), (5.5, 7.5), ["high"], ["clay", "loamy"]),
    }},
}


def _draw(rng, lo, hi, lo_clip=None, hi_clip=None):
    pad = NOISE * (hi - lo)
    value = rng.uniform(lo - pad, hi + pad)
    if lo_clip is not None:
        value = max(lo_clip, value)
    if hi_clip is not None:
        value = min(hi_clip, value)
    return value


def generate(save=True, verbose=True):
    rng = random.Random(SEED)
    rows = []
    for crop, info in PROFILES.items():
        for season, p in info["seasons"].items():
            for _ in range(SAMPLES_PER_COMBO):
                rows.append({
                    "temperature": round(_draw(rng, *p["temperature"]), 1),
                    "rainfall": int(round(_draw(rng, *p["rainfall"], lo_clip=0))),
                    "humidity": int(round(_draw(rng, *p["humidity"], lo_clip=10, hi_clip=100))),
                    "soil_ph": round(_draw(rng, *p["soil_ph"], lo_clip=4.0, hi_clip=9.5), 1),
                    "water_availability": rng.choice(p["water"]),
                    "soil_type": rng.choice(p["soil"]),
                    "season": season,
                    "crop": crop,
                    "drought_tolerance": info["tolerance"],
                })
    df = pd.DataFrame(rows)
    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    if verbose:
        print("=" * 50)
        print(" DATASET GENERATED SUCCESSFULLY")
        print("=" * 50)
        print("Total records:", len(df))
        print("\nRecords per season:")
        print(df["season"].value_counts().to_string())
        print("\nRecords per crop:")
        print(df["crop"].value_counts().to_string())
        print("\nSaved to:", DATA_PATH)
    return df


if __name__ == "__main__":
    generate()
