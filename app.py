import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "diabetes.db"
MODEL_PATH_S1 = DATA_DIR / "diabetes_model_s1.pkl"
THRESHOLD_S1 = 150.0
MODEL_PATH_S2 = DATA_DIR / "diabetes_model_s2.pkl"
THRESHOLD_S2 = 250.0

FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
RISK_LABELS = {0: "Not Endangered", 1: "Endangered"}

app = Flask(__name__, template_folder="templates", static_folder="static")
model = None


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    print("Initializing database...")
    DATA_DIR.mkdir(exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            sex REAL,
            bmi REAL,
            bp REAL,
            s1 REAL,
            s2 REAL,
            s3 REAL,
            s4 REAL,
            s5 REAL,
            s6 REAL,
            target REAL
        )
        """
    )
    conn.commit()

    cursor.execute("SELECT COUNT(1) FROM patients")
    count = cursor.fetchone()[0]
    if count == 0:
        diabetes = load_diabetes()
        X = diabetes["data"]
        y = diabetes["target"]

        cursor.executemany(
            "INSERT INTO patients (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*X[i].tolist(), float(y[i])) for i in range(X.shape[0])],
        )
        conn.commit()

    conn.close()
    print("Database initialized successfully.")


def train_model(model_path:str, threshold):
    print("Training models...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target FROM patients")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No patient data available for model training.")

    X = np.array([tuple(row[:10]) for row in rows], dtype=float)
    y = np.array([row[10] > threshold for row in rows], dtype=float)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=600),
            ),
        ]
    )

    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    print(f"Model {model_path} trained and saved successfully.")
    return pipeline


def load_models():
    global model_s1
    global model_s2
    if MODEL_PATH_S1.exists():
        model_s1 = joblib.load(MODEL_PATH_S1)
    else:
        model_s1 = train_model(MODEL_PATH_S1, THRESHOLD_S1)

    if MODEL_PATH_S2.exists():
        model_s2 = joblib.load(MODEL_PATH_S2)
    else:
        model_s2 = train_model(MODEL_PATH_S2, THRESHOLD_S2)

    return model_s1, model_s2


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify(
        {
            "ready": model is not None,
            "database": str(DB_PATH),
            "model_file1": str(MODEL_PATH_S1),
            "model_file2": str(MODEL_PATH_S2),
            "features": FEATURE_NAMES,
        }
    )


@app.route("/api/summary")
def summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM patients")
    count = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(age), AVG(sex), AVG(bmi), AVG(bp), AVG(s1), AVG(s2), AVG(s3), AVG(s4), AVG(s5), AVG(s6), AVG(target) FROM patients")
    avgs = cursor.fetchone()
    
    cursor.execute("SELECT MIN(age), MIN(sex), MIN(bmi), MIN(bp), MIN(s1), MIN(s2), MIN(s3), MIN(s4), MIN(s5), MIN(s6), MIN(target) FROM patients")
    mins = cursor.fetchone()
    
    cursor.execute("SELECT MAX(age), MAX(sex), MAX(bmi), MAX(bp), MAX(s1), MAX(s2), MAX(s3), MAX(s4), MAX(s5), MAX(s6), MAX(target) FROM patients")
    maxs = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) as count FROM patients WHERE target > ?", (THRESHOLD_S1,))
    risk_counts_s1 = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) as count FROM patients WHERE target > ?", (THRESHOLD_S2,))
    risk_counts_s2 = cursor.fetchone()[0]

    return jsonify({
        "total_patients": count,
        "averages": dict(zip(FEATURE_NAMES + ["target"], avgs[:-1])),
        "mins": dict(zip(FEATURE_NAMES + ["target"], mins[:-1])),
        "maxs": dict(zip(FEATURE_NAMES + ["target"], maxs[:-1])),
        "risk_distribution": [risk_counts_s1, risk_counts_s2]
    })


@app.route("/api/visualizations")
def visualizations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT bmi, bp, target FROM patients")
    rows = cursor.fetchall()
    conn.close()
    
    data = {
        "bmi": [row[0] for row in rows],
        "bp": [row[1] for row in rows],
        "target": [row[2] for row in rows]
    }

    # Scatter plot of BMI vs Target
    bmi_scatter_data = [{
        "x": data["bmi"],
        "y": data["target"],
        "mode": "markers"
    }]
    
    bmi_layout = {
        "xaxis": {"range": [min(data["bmi"])-0.01, max(data["bmi"])+0.01], "title": "BMI"},
        "yaxis": {"range": [min(data["target"])-10, max(data["target"])+10], "title": "Target"},
        "title": "BMI vs Target Scatter"
    }

    # Scatter plot of BP vs Target
    bp_scatter_data = [{
        "x": data["bp"],
        "y": data["target"],
        "mode": "markers"
    }]

    bp_layout = {
        "xaxis": {"range": [min(data["bp"])-0.01, max(data["bp"])+0.01], "title": "Blood Pressure"},
        "yaxis": {"range": [min(data["target"])-10, max(data["target"])+10], "title": "Target"},
        "title": "Blood Pressure vs Target Scatter"
    }
    
    # Risk distribution pie chart
    x_pie = ["Not Endangered", "Endangered"]

    at_risk_s1 = 0
    at_risk_s2 = 0

    for t in data["target"]:
        if t > THRESHOLD_S1:
            at_risk_s1 += 1
            if t > THRESHOLD_S2:
                at_risk_s2 += 1
   
    return jsonify({
        "bmi_target_scatter": { 
            "data": bmi_scatter_data,
            "layout": bmi_layout
        },
        "bp_target_scatter": { 
            "data": bp_scatter_data,
            "layout": bp_layout
        },
        "risk_pie": {
            "x_array": x_pie,
            "y_array_s1": [at_risk_s1, (len(data["target"]) - at_risk_s1)],
            "y_array_s2": [at_risk_s2, (len(data["target"]) - at_risk_s2)]
        }
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model_s1 is None or model_s2 is None:
        load_models()

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload."}), 400

    features = []
    for name in FEATURE_NAMES:
        value = payload.get(name)
        if value is None:
            return jsonify({"error": f"Missing feature: {name}"}), 400
        try:
            features.append(float(value))
        except (TypeError, ValueError):
            return jsonify({"error": f"Invalid value for feature: {name}"}), 400

    X = np.array([features], dtype=float)
    prediction_s1 = model_s1.predict(X)[0]
    probabilities_s1 = model_s1.predict_proba(X)[0]
    risk_proba_s1 = float(probabilities_s1[1])

    prediction_s2 = model_s2.predict(X)[0]
    probabilities_s2 = model_s2.predict_proba(X)[0]
    risk_proba_s2 = float(probabilities_s2[1])

    return jsonify(
        {
            "risk_s1": int(prediction_s1),
            "risk_label_s1": RISK_LABELS[int(prediction_s1)],
            "risk_probability_s1": round(risk_proba_s1, 4),
            "risk_s2": int(prediction_s2),
            "risk_label_s2": RISK_LABELS[int(prediction_s2)],
            "risk_probability_s2": round(risk_proba_s2, 4),
            "features": dict(zip(FEATURE_NAMES, features)),
        }
    )

if __name__ == "__main__":
    init_db()
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
