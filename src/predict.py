"""Command-line version of the recommender.   Run:  python src/predict.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import REGIONS, recommend, validate_inputs  # noqa: E402
import history  # noqa: E402


def ask(prompt, options=None):
    text = input(prompt).strip()
    return text


def main():
    print("\n" + "=" * 54)
    print("   AI-BASED DROUGHT-RESILIENT CROP/VARIETY SYSTEM")
    print("=" * 54 + "\n")
    form = {
        "temperature": ask("Mean temperature in the season (C): "),
        "rainfall": ask("Rainfall during the growing season (mm): "),
        "humidity": ask("Humidity (%): "),
        "soil_ph": ask("Soil pH: "),
        "water_availability": ask("Water availability (low/medium/high): "),
        "soil_type": ask("Soil type (sandy/sandy_loam/loamy/clay): "),
        "season": ask("Season (Kharif/Rabi/Zaid): ").title(),
        "region": ask("Region (" + "/".join(REGIONS) + ", or Any): ").title() or "Any",
    }
    inputs, errors = validate_inputs(form)
    if errors:
        print("\nPlease fix these inputs:")
        for e in errors:
            print(" -", e)
        return

    r = recommend(inputs)
    run_id = history.save(inputs, r, source="cli")
    print("\nDrought risk (" + r["season"] + "):", r["drought_risk"],
          f"(score {r['drought_score']}/{r['drought_max']})")
    print("\nCrop ranking")
    print("-" * 54)
    print(f"{'#':<3}{'Crop':<16}{'Suitability':>12}{'Tolerance':>11}{'Score':>8}")
    for i, c in enumerate(r["crops"], 1):
        print(f"{i:<3}{c['crop']:<16}{c['suitability']:>11.1f}%{c['tolerance']:>11}{c['final_score']:>8.1f}")

    print("\nRecommended varieties for", r["best_crop"])
    print("-" * 54)
    if r["variety_note"]:
        print("Note:", r["variety_note"])
    for i, v in enumerate(r["varieties"], 1):
        print(f"{i}. {v['variety']}  | {v['tolerance']} tolerance | {v['maturity_days']} days | "
              f"water: {v['water_requirement']} | {v['region_match']}")

    print("\nWhy this recommendation?")
    for line in r["explanation"]:
        print(" -", line)
    for n in r["notes"]:
        print(" !", n)
    print("\nFinal recommendation:", r["best_crop"], "-", r["best_variety"])
    print("(saved to history as record #%d - see it on the website's History page)" % run_id)
    print("=" * 54)


if __name__ == "__main__":
    main()
