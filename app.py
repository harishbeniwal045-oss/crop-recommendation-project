"""
AI-Based Drought-Resilient Crop/Variety Recommendation System - web app.

Run:  python app.py      then open  http://127.0.0.1:5000
"""
import os
import sys

from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, url_for)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from recommender import (  # noqa: E402
    REGIONS, SEASON_LABELS, SEASONS, SOIL_TYPES, WATER_LEVELS,
    load_metrics, load_model, recommend, validate_inputs,
)
import history  # noqa: E402

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

load_model()          # loads the saved model, or trains it automatically on first run


def _form_options():
    return dict(seasons=SEASONS, season_labels=SEASON_LABELS, regions=REGIONS,
                soil_types=SOIL_TYPES, water_levels=WATER_LEVELS)


@app.route("/", methods=["GET", "POST"])
def home():
    result, errors, form, entry = None, [], {}, None
    if request.method == "POST":
        form = request.form.to_dict()
        inputs, errors = validate_inputs(form)
        if not errors:
            # Save, then redirect: a page refresh no longer re-submits the form
            # or loses the result - the result is reloaded from history.
            run_id = history.save(inputs, recommend(inputs), source="web")
            return redirect(url_for("home", run=run_id))
    else:
        saved_id = request.args.get("run", type=int) or request.args.get("prefill", type=int)
        if saved_id:
            entry = history.get(saved_id)
            if entry is None:
                abort(404)
            form = {k: str(v) for k, v in entry["inputs"].items()}
            if request.args.get("run", type=int):
                result = dict(entry["result"], metrics=load_metrics())
            else:
                entry = None
    return render_template("index.html", result=result, errors=errors, form=form,
                           entry=entry, **_form_options())


@app.route("/history")
def history_page():
    filters = {k: request.args.get(k, "") for k in ("q", "season", "best_crop", "drought_risk")}
    page = history.search(filters, page=request.args.get("page", 1, type=int))
    return render_template("history.html", page=page, filters=filters,
                           stats=history.stats(), season_labels=SEASON_LABELS)


@app.route("/history/<int:entry_id>")
def history_detail(entry_id):
    entry = history.get(entry_id)
    if entry is None:
        abort(404)
    # Re-run the SAME inputs through today's model and compare with what was saved.
    fresh = recommend(entry["inputs"])
    diffs = history.compare(entry["result"], fresh)
    return render_template("history_detail.html", entry=entry, fresh=fresh, diffs=diffs,
                           same=history.same_inputs(entry), current_version=history.model_version(),
                           season_labels=SEASON_LABELS)


@app.route("/history/<int:entry_id>/rerun", methods=["POST"])
def history_rerun(entry_id):
    entry = history.get(entry_id)
    if entry is None:
        abort(404)
    new_id = history.save(entry["inputs"], recommend(entry["inputs"]), source="rerun")
    return redirect(url_for("history_detail", entry_id=new_id))


@app.route("/history/export.csv")
def history_export():
    filters = {k: request.args.get(k, "") for k in ("q", "season", "best_crop", "drought_risk")}
    rows = history.search(filters, page=1, per_page=100000)["rows"]
    return Response(history.to_csv(rows), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=crop_history.csv"})


@app.route("/api/history")
def api_history():
    filters = {k: request.args.get(k, "") for k in ("q", "season", "best_crop", "drought_risk")}
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    page = history.search(filters, page=request.args.get("page", 1, type=int), per_page=per_page)
    for r in page["rows"]:
        r.pop("result", None)
    return jsonify({"ok": True, **page})


@app.route("/api/history/<int:entry_id>")
def api_history_item(entry_id):
    entry = history.get(entry_id)
    if entry is None:
        return jsonify({"ok": False, "errors": ["not found"]}), 404
    return jsonify({"ok": True, "entry": entry})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """JSON version of the recommender, e.g. for a mobile app.
    POST a JSON body (or form data) with: season, region, temperature, rainfall,
    humidity, soil_ph, water_availability, soil_type. The run is saved to history."""
    data = request.get_json(silent=True) or request.form.to_dict()
    inputs, errors = validate_inputs(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    result = recommend(inputs)
    run_id = history.save(inputs, result, source="api")
    result.pop("metrics", None)
    return jsonify({"ok": True, "history_id": run_id, "result": result})


@app.errorhandler(404)
def not_found(_):
    return render_template("history.html", page={"rows": [], "total": 0, "page": 1, "pages": 1},
                           filters={}, stats=history.stats(), season_labels=SEASON_LABELS,
                           message="That record was not found."), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
