# AI-Based Drought-Resilient Crop/Variety Recommendation System

A machine-learning web app that recommends a crop and a variety for a farmer's
conditions, with the drought risk taken into account. Works for three Indian
cropping seasons: **Kharif, Rabi and Zaid**.

## How to run (Windows)

Option 1 - double-click `run.bat` (creates the environment and starts the app).

Option 2 - manual:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Command-line version: `python src/predict.py`

Retrain the model (optional, it also trains itself automatically if the model
file is missing): `python src/train_model.py`

Regenerate the dataset (optional): `python src/generate_dataset.py`

## Project structure

```
app.py                  Flask web app
requirements.txt        Python packages
tests/                  Automated tests (unittest)
run.bat                 One-click start for Windows
data/crop_data.csv      Training data (synthetic, 1140 rows, 12 crops, 3 seasons)
data/variety_data.csv   Variety database (42 varieties)
model/                  Saved model (crop_model.pkl) and metrics.json
src/generate_dataset.py Builds the dataset from season-wise crop profiles
src/drought_risk.py     Season-aware drought risk (LOW / MEDIUM / HIGH)
src/train_model.py      Trains and evaluates the Random Forest model
src/recommender.py      Validation, crop ranking, variety ranking, explanations
src/predict.py          Command-line interface
templates/index.html    Recommendation page
templates/history*.html History list and record-detail pages
static/style.css        Shared styles
src/history.py          Saves/searches history in SQLite (data/history.db, created automatically)
```

## How it works

1. **Input** - season, region, temperature, seasonal rainfall, humidity, soil pH,
   water availability, soil type (all validated).
2. **Drought risk** - four factors (temperature, rainfall, humidity, water) are
   scored 1-3 using thresholds that depend on the season. Total 4-12:
   LOW (up to 6), MEDIUM (7-9), HIGH (10-12).
3. **Crop suitability (ML)** - a Random Forest gives a probability for each crop.
   Only crops grown in the chosen season are kept.
4. **Drought-resilient ranking** - final score = suitability x (1 - a + a x tolerance),
   with a = 0.1 / 0.3 / 0.5 for LOW / MEDIUM / HIGH risk and tolerance = 1.0 / 0.6 / 0.2
   for high / medium / low. A crop must be suitable first; the higher the drought
   risk, the more a low-tolerance crop is scored down.
5. **Variety** - varieties of the best crop are filtered by season and ranked by
   drought tolerance, whether their water need fits the farmer's water, early
   maturity, and region match.
6. **Explanation** - the page shows why, and compares the farmer's values with the
   crop's typical range.

## Crops and seasons

| Crop | Kharif | Rabi | Zaid |
|---|---|---|---|
| Sorghum | Yes | Yes | |
| Pearl Millet | Yes | | Yes |
| Finger Millet | Yes | | |
| Chickpea | | Yes | |
| Green Gram | Yes | | Yes |
| Barley | | Yes | |
| Soybean | Yes | | |
| Cotton | Yes | | |
| Groundnut | Yes | Yes | Yes |
| Wheat | | Yes | |
| Mustard | | Yes | |
| Rice | Yes | Yes | Yes |

## Model results

Run `python src/train_model.py` to see the numbers. On the synthetic data:
about 68% top-1 accuracy and about 93% top-3 accuracy for the Random Forest (crops of the same season
overlap in their ranges, so the app shows a ranked list rather than one answer).

## What changed from the first version

- Season is no longer hard-coded to Kharif; the model is trained on Kharif, Rabi and Zaid.
- Chickpea is now a Rabi crop (it was wrongly labeled Kharif).
- Added Wheat, Mustard, Barley, Green Gram and Groundnut, so Rabi and Zaid have real choices.
- Drought tolerance now affects the crop ranking, weighted by the drought risk.
- Drought thresholds depend on the season.
- Variety database grew from 12 to 42 varieties, with season, water need, maturity and region.
- Rainfall is now clearly **seasonal rainfall**.
- Input validation, auto-training, clearer results page.

## What changed in version 3 (this version)

- **Model comparison**: Random Forest is compared with Gradient Boosting, Decision Tree,
  K-Nearest Neighbours, Logistic Regression and Naive Bayes on the same split
  (`python src/train_model.py` prints the table; it is also saved in `model/metrics.json`).
- **Feature importance**: shows which inputs the model relies on (humidity, temperature and
  rainfall lead), displayed on the result page.
- **Per-season accuracy** in `metrics.json` (Kharif, Rabi, Zaid).
- **Confidence level** (High / Medium / Low) for each recommendation, with a note when two crops are close.
- **JSON API**: `POST /api/recommend` returns the recommendation as JSON (for a mobile app or other tools).
- **Automated tests**: `python -m unittest discover -s tests -v` checks six season cases, input
  validation and the API.
- Developer name and ID added to the web page footer.

## What changed in version 4 - History system

- **Every run is saved** (inputs + full output + time + model version) in `data/history.db`
  (SQLite, built into Python - nothing new to install). No names or IP addresses are stored.
- **Refresh no longer loses the result**: after you submit, the page redirects to `/?run=<id>`,
  so refreshing shows the same inputs and output again.
- **History page** (`/history`): anyone visiting the site sees all past inputs and outputs, with
  search, season / crop / drought-risk filters, paging and CSV download (`/history/export.csv`).
- **Record page** (`/history/<id>`): shows the saved input and output, then re-runs the same
  inputs through today's model and says whether the result is the same or changed.
  Also lists every run that used exactly the same inputs, and has "Run again" and "Edit inputs".
- **Same input after years -> same output?** Yes, as long as the model and variety data are
  unchanged (the model is deterministic). Each record stores a model-version fingerprint, so if
  you retrain or edit `variety_data.csv` later, old records are tagged "older model" and the
  record page shows exactly what changed.
- **JSON**: `/api/history`, `/api/history/<id>`; `POST /api/recommend` now returns `history_id`.
  The command-line version also saves to history.
- Tests: 5 new history tests (8 total). They use a temporary database, never your real history.
- To start with empty history, delete `data/history.db` (stop the app first).
- When deploying publicly, back up `data/history.db`; some free hosts reset local files on restart.

## Limitations (mention these in your report)

- The training data is **synthetic**, generated from typical agronomic ranges, not
  from field surveys. Accuracy shows how well the model learns these ranges, not
  real-farm performance. To improve it, replace `data/crop_data.csv` with real data
  (ICAR, ICRISAT, state agriculture departments) with the same columns.
- Variety details (maturity days, water need, region) are approximate. Confirm
  with a local agricultural university or KVK before sowing.
- Drought risk is a rule-based score, not a weather forecast.

## Adding more crops or varieties

- New variety: add a row to `data/variety_data.csv`.
- New crop: add a profile in `src/generate_dataset.py` (PROFILES), run
  `python src/generate_dataset.py`, then `python src/train_model.py`.
