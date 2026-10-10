"""
Train the RandomForest crop-suitability model.

Run:  python src/train_model.py
The trained pipeline is saved to model/crop_model.pkl and the metrics to
model/metrics.json.
"""
import json
import os
import sys

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dataset import BASE_DIR, DATA_PATH, generate  # noqa: E402

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

NUMERIC = ["temperature", "rainfall", "humidity", "soil_ph"]
CATEGORICAL = ["water_availability", "soil_type", "season"]
FEATURES = NUMERIC + CATEGORICAL


def build_pipeline(model=None, scale=False):
    """Pre-processing (one-hot encoding, optional scaling) followed by a classifier.
    The default classifier is the Random Forest used by the web app."""
    numeric = StandardScaler() if scale else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", numeric, NUMERIC),
        ],
    )
    if model is None:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# Algorithms compared in the project report: name -> (factory, needs scaling)
CANDIDATES = {
    "Random Forest": (lambda: RandomForestClassifier(n_estimators=200, random_state=42), False),
    "Gradient Boosting": (lambda: GradientBoostingClassifier(random_state=42), False),
    "Decision Tree": (lambda: DecisionTreeClassifier(random_state=42), False),
    "K-Nearest Neighbours": (lambda: KNeighborsClassifier(n_neighbors=7), True),
    "Logistic Regression": (lambda: LogisticRegression(max_iter=2000), True),
    "Naive Bayes": (lambda: GaussianNB(), True),
}


def compare_models(X_train, X_test, y_train, y_test, X, y, valid_by_season):
    """Train every candidate on the same split and report the same metrics."""
    rows = []
    for name, (factory, scale) in CANDIDATES.items():
        pipe = build_pipeline(factory(), scale)
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        season_acc = season_constrained_accuracy(pipe, X_test, y_test, valid_by_season)
        cv = cross_val_score(build_pipeline(factory(), scale), X, y, cv=5).mean()
        rows.append({
            "model": name,
            "top1": round(float(season_acc[1]), 4),
            "top3": round(float(season_acc[3]), 4),
            "cv": round(float(cv), 4),
        })
    rows.sort(key=lambda r: (r["top3"], r["top1"]), reverse=True)
    return rows


def feature_importance(pipeline):
    """Random Forest importance, with the one-hot columns added back together
    so every farmer input gets one value (in %)."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    totals = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, values):
        column = name.split("__", 1)[1]
        for f in FEATURES:
            if column == f or column.startswith(f + "_"):
                totals[f] += float(value)
                break
    total = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return {k: round(100 * v / total, 1) for k, v in ranked}


def per_season_accuracy(pipeline, X_test, y_test, valid_by_season):
    out = {}
    for season in sorted(valid_by_season):
        mask = (X_test["season"] == season).values
        acc = season_constrained_accuracy(
            pipeline, X_test[mask], y_test[mask], valid_by_season)
        out[season] = {"top1": round(float(acc[1]), 4), "top3": round(float(acc[3]), 4),
                       "test_records": int(mask.sum())}
    return out


def season_constrained_accuracy(pipeline, X_test, y_test, valid_by_season, k_values=(1, 3)):
    """Accuracy when only crops grown in the selected season are considered
    (this is exactly how the web app ranks crops)."""
    proba = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)
    hits = {k: 0 for k in k_values}
    for row_proba, season, truth in zip(proba, X_test["season"], y_test):
        allowed = valid_by_season[season]
        ranked = sorted(
            (c for c in classes if c in allowed),
            key=lambda c: row_proba[classes.index(c)],
            reverse=True,
        )
        for k in k_values:
            if truth in ranked[:k]:
                hits[k] += 1
    n = len(y_test)
    return {k: hits[k] / n for k in k_values}


def train(save=True, verbose=True):
    if not os.path.exists(DATA_PATH):
        generate(verbose=False)
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["crop"]
    strat = data["crop"] + "_" + data["season"]      # keep every crop-season in both splits

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(build_pipeline(), X, y, cv=5)

    valid_by_season = data.groupby("season")["crop"].apply(set).to_dict()
    season_acc = season_constrained_accuracy(pipeline, X_test, y_test, valid_by_season)

    comparison = compare_models(X_train, X_test, y_train, y_test, X, y, valid_by_season)
    metrics = {
        "records": int(len(data)),
        "crops": int(y.nunique()),
        "test_accuracy": round(float(accuracy), 4),
        "season_top1_accuracy": round(float(season_acc[1]), 4),
        "season_top3_accuracy": round(float(season_acc[3]), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "per_season": per_season_accuracy(pipeline, X_test, y_test, valid_by_season),
        "feature_importance": feature_importance(pipeline),
        "model_comparison": comparison,
        "sklearn_version": sklearn.__version__,
    }

    if verbose:
        print("Dataset records :", metrics["records"])
        print("Crops           :", metrics["crops"])
        print("Test accuracy (all crops)          : {:.2%}".format(accuracy))
        print("Top-1 accuracy (within the season) : {:.2%}".format(season_acc[1]))
        print("Top-3 accuracy (within the season) : {:.2%}".format(season_acc[3]))
        print("5-fold CV mean  : {:.2%}".format(cv_scores.mean()))
        print("\nModel comparison (top-1 / top-3 within the season / 5-fold CV)")
        for r in comparison:
            print("  {:<22} {:>6.1%} {:>6.1%} {:>6.1%}".format(r["model"], r["top1"], r["top3"], r["cv"]))
        print("\nFeature importance (%):", metrics["feature_importance"])
        print("Per-season:", metrics["per_season"])
        print("\nClassification report")
        print("-" * 50)
        print(classification_report(y_test, y_pred, zero_division=0))

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print("Model saved to:", MODEL_PATH)
    return pipeline, metrics


if __name__ == "__main__":
    train()
