"""
Rule-based drought risk assessment (LOW / MEDIUM / HIGH).

Thresholds depend on the season, because "low rainfall" or "high temperature"
means something different in Kharif (monsoon), Rabi (winter) and Zaid (summer).
Each of 4 factors scores 1 (good) to 3 (bad); total 4-12.
"""

# (warm, hot) temperature limits in deg C and (very_low, low) rainfall limits in mm
SEASON_THRESHOLDS = {
    "Kharif": {"temp": (30, 35), "rain": (400, 700)},
    "Rabi":   {"temp": (25, 30), "rain": (100, 250)},
    "Zaid":   {"temp": (35, 40), "rain": (75, 150)},
}
HUMIDITY_LIMITS = (40, 60)          # below 40 = dry air, below 60 = moderate


def assess_drought_risk(temperature, rainfall, humidity, water_availability, season="Kharif"):
    """Return a dict: level, score and a list of human-readable reasons."""
    t = SEASON_THRESHOLDS.get(season, SEASON_THRESHOLDS["Kharif"])
    warm, hot = t["temp"]
    rain_low, rain_mid = t["rain"]
    reasons = []
    score = 0

    # Temperature
    if temperature >= hot:
        score += 3
        reasons.append(f"Temperature ({temperature:g} C) is high for {season}.")
    elif temperature >= warm:
        score += 2
        reasons.append(f"Temperature ({temperature:g} C) is moderately high for {season}.")
    else:
        score += 1
        reasons.append(f"Temperature ({temperature:g} C) is comfortable for {season}.")

    # Rainfall
    if rainfall < rain_low:
        score += 3
        reasons.append(f"Rainfall ({rainfall:g} mm) is very low for {season}.")
    elif rainfall < rain_mid:
        score += 2
        reasons.append(f"Rainfall ({rainfall:g} mm) is moderate for {season}.")
    else:
        score += 1
        reasons.append(f"Rainfall ({rainfall:g} mm) is good for {season}.")

    # Humidity
    if humidity < HUMIDITY_LIMITS[0]:
        score += 3
        reasons.append(f"Humidity ({humidity:g}%) is low (dry air).")
    elif humidity < HUMIDITY_LIMITS[1]:
        score += 2
        reasons.append(f"Humidity ({humidity:g}%) is moderate.")
    else:
        score += 1
        reasons.append(f"Humidity ({humidity:g}%) is high.")

    # Water availability (irrigation / stored moisture)
    if water_availability == "low":
        score += 3
        reasons.append("Water availability is low.")
    elif water_availability == "medium":
        score += 2
        reasons.append("Water availability is medium.")
    else:
        score += 1
        reasons.append("Water availability is high.")

    if score >= 10:
        level = "HIGH"
    elif score >= 7:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"level": level, "score": score, "max_score": 12, "reasons": reasons}


def calculate_drought_risk(temperature, rainfall, humidity, water_availability, season="Kharif"):
    """Backward-compatible helper that returns only the level string."""
    return assess_drought_risk(temperature, rainfall, humidity, water_availability, season)["level"]
