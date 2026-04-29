from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, render_template_string, request

from ml_pipeline import dashboard_chart_data, load_or_train_model, predict


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
MODEL_ARTIFACT = load_or_train_model()


@app.get("/")
def index():
    template_path = Path(app.template_folder or "templates") / "index.html"
    if template_path.exists():
        return render_template("index.html", metrics=MODEL_ARTIFACT["metrics"])

    root_index = Path("index.html")
    if root_index.exists():
        return render_template_string(root_index.read_text(encoding="utf-8"), metrics=MODEL_ARTIFACT["metrics"])

    return (
        "Missing frontend file. Upload templates/index.html to GitHub, then redeploy.",
        500,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "metrics": MODEL_ARTIFACT["metrics"]})


@app.post("/predict")
def predict_api():
    data = request.get_json(silent=True) or {}
    required = ["irradiation", "ambient_temp", "module_temp", "hour"]
    missing = [key for key in required if key not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
    try:
        return jsonify(predict(data, MODEL_ARTIFACT))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400


@app.get("/chart-data")
def chart_data_api():
    return jsonify(dashboard_chart_data(MODEL_ARTIFACT))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)
