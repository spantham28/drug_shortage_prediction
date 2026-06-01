"""Shared inference helpers for Vercel Python serverless functions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Allow imports when running as a script from api/
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def _encode_profit_flag(value) -> float:
    profit_map = {"Profit": 1.0, "Loss": 0.0, "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0}
    return float(profit_map.get(value, profit_map.get(str(value), 0.0)))


def load_shortage_artifacts():
    scaler = joblib.load(MODELS_DIR / "shortage_scaler.pkl")
    bundle = joblib.load(MODELS_DIR / "shortage_ensemble.pkl")
    return scaler, bundle


def predict_shortage(payload: dict) -> dict:
    scaler, bundle = load_shortage_artifacts()
    features = bundle["features"]
    row = []
    for feat in features:
        if feat not in payload:
            raise ValueError(f"Missing required field: {feat}")
        row.append(float(payload[feat]))

    X = np.array(row, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    scores = []
    for name in bundle["members"]:
        model = bundle["models"][name]
        if hasattr(model, "predict_proba"):
            scores.append(model.predict_proba(X_scaled)[0, 1])
        else:
            scores.append(float(model.decision_function(X_scaled)[0]))

    probability = float(np.mean(scores))
    threshold = float(bundle.get("threshold", 0.5))
    shortage_flag = int(probability >= threshold)

    return {
        "shortage_flag": shortage_flag,
        "shortage_probability_pct": round(probability * 100, 2),
        "shortage_label": "Shortage predicted" if shortage_flag == 1 else "No shortage predicted",
        "model_type": bundle.get("model_type", "EnsembleTop3"),
        "ensemble_members": bundle.get("members", []),
    }


def load_income_artifacts():
    model = joblib.load(MODELS_DIR / "income_model.pkl")
    with open(MODELS_DIR / "income_feature_config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, config


def transform_features(raw: dict, config: dict) -> np.ndarray:
    values = []
    for feat in config["features"]:
        name = feat["name"]
        if feat.get("type") == "categorical":
            values.append(_encode_profit_flag(raw.get(name, "Loss")))
            continue

        default = feat.get("median", 0.0)
        val = float(raw.get(name, default))
        if np.isnan(val):
            val = default

        transform = feat.get("transform", "none")
        if transform == "square":
            val = val ** 2
        elif transform == "sqrt":
            val = np.sqrt(abs(val) + feat.get("offset", 0.0))
        elif transform == "log":
            val = np.log(abs(val) + feat.get("offset", 1.0))
        values.append(val)

    return np.array(values, dtype=float).reshape(1, -1)


def predict_income(payload: dict) -> dict:
    model, config = load_income_artifacts()
    X = transform_features(payload, config)
    prediction = float(model.predict(X)[0])

    return {
        "net_income": prediction,
        "net_income_formatted": f"${prediction:,.0f}",
        "model_name": config.get("model_name", "RandomForestRegressor"),
        "target": config.get("target", "Net Income"),
    }


def get_income_feature_schema() -> dict:
    _, config = load_income_artifacts()
    return config
