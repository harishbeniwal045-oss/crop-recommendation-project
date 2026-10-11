# 🌾 AI-Based Drought-Resilient Crop Recommendation System

A web app that tells a farmer **which crop and which variety to grow**, based on their land, weather and water, while taking **drought risk** into account.

Works for all three Indian cropping seasons: **Kharif, Rabi and Zaid**.

---

## ✨ What It Does

| You enter | You get |
|---|---|
| Season, region, temperature, rainfall, humidity, soil pH, water availability, soil type | Drought risk (Low / Medium / High) |
| | Ranked list of suitable crops |
| | Best variety for the top crop |
| | Simple explanation and confidence level |

Extra features:
- 🕘 **History**: every result is saved; search, filter and download as CSV
- 🔌 **JSON API** for use in other apps
- 📊 **Model comparison** and feature importance shown on the result page

---

## 🧠 How It Works (in 4 steps)

1. **Drought risk**: temperature, rainfall, humidity and water are scored (1–3 each, season-wise). Total 4–6 = Low, 7–9 = Medium, 10–12 = High.
2. **Crop suitability**: a Random Forest model gives a probability for each crop grown in the chosen season.
3. **Drought-aware ranking**: the higher the drought risk, the more crops with low drought tolerance are scored down.
4. **Variety pick**: varieties of the best crop are ranked by drought tolerance, water need, early maturity and region match.

---

## 🌱 Crops Covered

12 crops: Rice, Wheat, Sorghum, Pearl Millet, Finger Millet, Chickpea, Green Gram, Barley, Soybean, Cotton, Groundnut, Mustard.
The variety database has 42 varieties.

---

## 🛠️ Tech Stack

Python · Flask · Scikit-learn · Pandas · NumPy · SQLite · HTML/CSS

---

## ⚙️ How to Run

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

> 💡 On Windows you can simply double-click `run.bat`.

Other commands:

| Task | Command |
|---|---|
| Command-line version | `python src/predict.py` |
| Retrain the model | `python src/train_model.py` |
| Regenerate dataset | `python src/generate_dataset.py` |
| Run tests | `python -m unittest discover -s tests -v` |

---

## 📁 Project Structure

```
app.py          Flask web app
src/            Model training, drought risk, recommender, history
data/           Crop dataset and variety database
model/          Saved model and evaluation results
templates/      HTML pages
static/         CSS
tests/          Automated tests
```

---

## 🔌 API

- `POST /api/recommend`: get a recommendation as JSON
- `GET /api/history` and `GET /api/history/<id>`: view saved runs

---

## 📊 Model Results

Random Forest reaches about **68% top-1** and **93% top-3** accuracy. Several models (Gradient Boosting, Decision Tree, KNN, Logistic Regression, Naive Bayes) are compared on the same data; see `model/metrics.json`.

---

## ⚠️ Limitations

- The dataset is **synthetic** (built from typical crop ranges), so accuracy does not guarantee real-farm results.
- Variety details are approximate. Confirm with a local agricultural university or KVK before sowing.
- Drought risk is a rule-based score, not a weather forecast.

---

## 🔮 Future Improvements

- Real agricultural data (ICAR, ICRISAT) and live weather
- Regional language support
- Mobile app

---

## 👨‍💻 Author

**Harish Beniwal**
