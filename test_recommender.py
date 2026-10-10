"""Automated tests. Run from the project folder:  python -m unittest discover -s tests -v

Drought risk is rule based, so its level is tested exactly. The crop comes from the
ML model, so the tests check what must always hold (crop grown in the season, tolerant
crop under HIGH risk) and the two clear-cut cases (wet Kharif -> Rice, irrigated Rabi -> Wheat).
"""
import os
import sys
import tempfile
import unittest

# tests must never write into the real history file
os.environ["HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_history.db")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from recommender import crops_for_season, recommend, validate_inputs  # noqa: E402

CASES = [  # season, temp, rain, humidity, water, region, expected risk, crop rule
    ("Kharif", 35, 350, 40, "low", "West", "HIGH", None),
    ("Kharif", 28, 1100, 80, "high", "East", "LOW", "Rice"),
    ("Rabi", 18, 150, 50, "low", "South", "MEDIUM", None),
    ("Rabi", 18, 250, 55, "high", "North", "LOW", "Wheat"),
    ("Zaid", 36, 100, 35, "low", "West", "HIGH", None),
    ("Rabi", 33, 550, 40, "medium", "South", "MEDIUM", None),
]


def run(case):
    season, t, r, h, w, region, *_ = case
    form = dict(season=season, region=region, temperature=t, rainfall=r, humidity=h,
                soil_ph=7, water_availability=w, soil_type="loamy")
    inputs, errors = validate_inputs(form)
    assert not errors, errors
    return recommend(inputs)


class RecommenderTests(unittest.TestCase):
    def test_cases(self):
        for case in CASES:
            with self.subTest(case=case[:6]):
                out = run(case)
                self.assertEqual(out["drought_risk"], case[6])
                self.assertIn(out["best_crop"], crops_for_season(case[0]))
                if case[7]:
                    self.assertEqual(out["best_crop"], case[7])
                if case[6] == "HIGH":
                    self.assertIn(out["best_tolerance"], ("high", "medium"))
                self.assertTrue(out["varieties"])
                self.assertIn(out["confidence"], ("High", "Medium", "Low"))

    def test_rejects_bad_input(self):
        _, errors = validate_inputs(dict(season="Winter", temperature="abc", rainfall=-5,
                                         humidity=120, soil_ph=2, water_availability="x",
                                         soil_type="rock"))
        self.assertGreaterEqual(len(errors), 6)

    def test_api(self):
        from app import app
        client = app.test_client()
        ok = client.post("/api/recommend", json=dict(
            season="Rabi", region="South", temperature=18, rainfall=250, humidity=55,
            soil_ph=7, water_availability="high", soil_type="loamy"))
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(ok.get_json()["ok"])
        bad = client.post("/api/recommend", json=dict(season="Rabi"))
        self.assertEqual(bad.status_code, 400)


class HistoryTests(unittest.TestCase):
    FORM = dict(season="Rabi", region="South", temperature="18", rainfall="250", humidity="55",
                soil_ph="7", water_availability="high", soil_type="loamy")

    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_post_saves_and_survives_refresh(self):
        import history
        before = history.stats()["total"]
        r = self.client.post("/", data=self.FORM)
        self.assertEqual(r.status_code, 302)                      # redirect (no re-submit on refresh)
        page = self.client.get(r.headers["Location"])             # a refresh = GET again
        self.assertIn(b"Recommendation Result", page.data)
        self.assertEqual(history.stats()["total"], before + 1)

    def test_invalid_input_not_saved(self):
        import history
        before = history.stats()["total"]
        self.client.post("/", data=dict(self.FORM, humidity="500"))
        self.assertEqual(history.stats()["total"], before)

    def test_same_input_same_output(self):
        import history
        from recommender import validate_inputs
        inputs, _ = validate_inputs(self.FORM)
        a = history.save(inputs, recommend(inputs))
        b = history.save(inputs, recommend(inputs), source="rerun")
        ea, eb = history.get(a), history.get(b)
        self.assertEqual(ea["input_hash"], eb["input_hash"])
        self.assertEqual(history.compare(ea["result"], eb["result"]), [])
        self.assertGreaterEqual(len(history.same_inputs(ea)), 2)

    def test_pages_and_api(self):
        self.client.post("/", data=self.FORM)
        self.assertEqual(self.client.get("/history").status_code, 200)
        self.assertEqual(self.client.get("/history?season=Rabi&q=a").status_code, 200)
        first = self.client.get("/api/history").get_json()["rows"][0]
        self.assertEqual(self.client.get(f"/history/{first['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/history/{first['id']}").get_json()["ok"], True)
        self.assertEqual(self.client.post(f"/history/{first['id']}/rerun").status_code, 302)
        csv = self.client.get("/history/export.csv")
        self.assertIn(b"best_crop", csv.data)
        self.assertEqual(self.client.get("/history/999999").status_code, 404)

    def test_detects_changed_model(self):
        import history
        old = {"best_crop": "Wheat", "best_variety": "A", "drought_risk": "LOW",
               "confidence": "High", "crops": [{"crop": "Wheat"}, {"crop": "Barley"}]}
        new = dict(old, best_crop="Barley", crops=[{"crop": "Barley"}, {"crop": "Wheat"}])
        self.assertEqual({d["field"] for d in history.compare(old, new)},
                         {"Recommended crop", "Crop ranking order"})


if __name__ == "__main__":
    unittest.main()
