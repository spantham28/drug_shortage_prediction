"""
Export trained models and feature configs for the web UI inference API.

Run from repository root:
    python scripts/export_inference_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_MODELS_DIR = REPO_ROOT / "ui" / "models"
UI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

SHORTAGE_FEATURES = [
    "avg_nadac",
    "manufacturer_num",
    "ingredient_num",
    "num_forms",
    "liquid_flag",
]

EXTRA_PREDICTORS = ["Profit"]
TRANSFORM_MAP = {
    "Square": "square",
    "Sqrt": "sqrt",
    "Original": "none",
    "Log": "log",
}


def _encode_profit_flag(value: str | float) -> float:
    profit_map = {"Profit": 1.0, "Loss": 0.0, "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0}
    return float(profit_map.get(value, profit_map.get(str(value), 0.0)))


def remove_outliers_compare(X, y, contamination=0.1):
    """Conservative class-aware outlier removal (Isolation Forest per class)."""
    from sklearn.ensemble import IsolationForest

    outlier_mask = np.zeros(len(X), dtype=bool)
    X_values = X.values
    y_values = y.astype(int).values
    for cls in [0, 1]:
        cls_idx = y_values == cls
        if cls_idx.sum() < 10:
            continue
        iso = IsolationForest(contamination=contamination, random_state=42)
        labels = iso.fit_predict(X_values[cls_idx])
        outlier_mask[np.where(cls_idx)[0]] = labels == -1
    clean = ~outlier_mask
    return X[clean].copy(), y[clean].copy()


def export_shortage_model() -> None:
    print("Exporting shortage prediction ensemble...")
    data_path = REPO_ROOT / "drug_shortage_timeline_prediction" / "price_signals_complete.csv"
    df = pd.read_csv(data_path)
    X = df[SHORTAGE_FEATURES]
    y = df["shortage_flag"]
    X_clean, y_clean = remove_outliers_compare(X, y, contamination=0.1)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, _ = next(sss.split(X_clean, y_clean.astype(int)))
    X_train_raw = X_clean.iloc[train_idx]
    y_train_raw = y_clean.iloc[train_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted[name] = model

    bundle = {
        "model_type": "EnsembleTop3",
        "members": list(fitted.keys()),
        "models": fitted,
        "features": SHORTAGE_FEATURES,
        "threshold": 0.5,
        "strategy": "soft_mean@0.5",
    }

    joblib.dump(scaler, UI_MODELS_DIR / "shortage_scaler.pkl")
    joblib.dump(bundle, UI_MODELS_DIR / "shortage_ensemble.pkl")
    print(f"  Saved shortage models to {UI_MODELS_DIR}")


def create_stratified_splits(df: pd.DataFrame) -> pd.DataFrame:
    target_col = "Net Income"
    try:
        quantiles = pd.qcut(df[target_col], q=4, labels=False, duplicates="drop")
        df = df.copy()
        df["stratum"] = [f"Q{q}" for q in quantiles]
    except ValueError:
        df = df.copy()
        df["stratum"] = "All"
    df["stratum"] = df["stratum"].fillna("All")
    return df


def stratified_train_val_test_split(df, test_size=0.2, val_size=0.2, random_state=42):
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

    stratum_counts = df["stratum"].value_counts()
    multi = stratum_counts[stratum_counts > 1].index
    train_indices, val_indices, test_indices = [], [], []

    if len(multi) > 0:
        multi_df = df[df["stratum"].isin(multi)].copy()
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(sss1.split(multi_df, multi_df["stratum"]))
        train_val_df = multi_df.iloc[train_val_idx]
        test_indices.extend(multi_df.iloc[test_idx].index.tolist())
        val_adj = val_size / (1 - test_size)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_adj, random_state=random_state + 1)
        train_idx, val_idx = next(sss2.split(train_val_df, train_val_df["stratum"]))
        train_indices.extend(train_val_df.iloc[train_idx].index.tolist())
        val_indices.extend(train_val_df.iloc[val_idx].index.tolist())

    single = stratum_counts[stratum_counts == 1].index
    if len(single) > 0:
        single_df = df[df["stratum"].isin(single)]
        idx = single_df.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        train_indices.extend(idx[:n_train])
        val_indices.extend(idx[n_train : n_train + n_val])
        test_indices.extend(idx[n_train + n_val :])

    return df.loc[train_indices].copy(), df.loc[val_indices].copy(), df.loc[test_indices].copy()


def build_feature_config(train_df: pd.DataFrame) -> tuple[dict, list[str]]:
    nonlinear_path = REPO_ROOT / "drug_shortage_cost_prediction" / "nonlinear_correlations.csv"
    nonlinear_df = pd.read_csv(nonlinear_path)
    final_features = nonlinear_df["Feature"].tolist()
    transformations = {}
    for _, row in nonlinear_df.iterrows():
        transformations[row["Feature"]] = TRANSFORM_MAP.get(row["Best_Transformation"], "none")
    for feat in EXTRA_PREDICTORS:
        if feat not in final_features:
            final_features.append(feat)
            transformations[feat] = "none"

    feature_meta = []
    for feat in final_features:
        if feat not in train_df.columns:
            continue
        transform = transformations.get(feat, "none")
        meta = {"name": feat, "transform": transform}
        if feat == "Profit":
            meta["type"] = "categorical"
            meta["options"] = ["Profit", "Loss"]
            feature_meta.append(meta)
            continue

        train_vals = train_df[feat].fillna(train_df[feat].median()).values.astype(float)
        meta["median"] = float(np.median(train_vals))
        if transform == "sqrt":
            offset = abs(float(np.min(train_vals))) if np.min(train_vals) < 0 else 0.0
            meta["offset"] = offset
        elif transform == "log":
            offset = abs(float(np.min(train_vals))) + 1 if np.min(train_vals) <= 0 else 0.0
            meta["offset"] = offset
        feature_meta.append(meta)

    return {"features": feature_meta}, [f["name"] for f in feature_meta]


def transform_row(raw: dict, feature_config: dict) -> np.ndarray:
    values = []
    for feat in feature_config["features"]:
        name = feat["name"]
        if feat.get("type") == "categorical":
            values.append(_encode_profit_flag(raw.get(name, "Loss")))
            continue
        val = float(raw.get(name, feat["median"]))
        if np.isnan(val):
            val = feat["median"]
        transform = feat["transform"]
        if transform == "square":
            val = val ** 2
        elif transform == "sqrt":
            val = np.sqrt(abs(val) + feat.get("offset", 0.0))
        elif transform == "log":
            val = np.log(abs(val) + feat.get("offset", 1.0))
        values.append(val)
    return np.array(values, dtype=float).reshape(1, -1)


def export_income_model() -> None:
    print("Exporting net income prediction model...")
    data_path = REPO_ROOT / "drug_shortage_cost_prediction" / "hospital_ops_updated.csv"
    df = pd.read_csv(data_path)
    target_col = "Net Income"
    exclude = ["Provider CCN", target_col, "Unnamed: 0"]
    feature_cols = [c for c in df.columns if c not in exclude + ["Profit"]]
    df = df.dropna(subset=[target_col] + feature_cols + EXTRA_PREDICTORS)
    df = create_stratified_splits(df)
    train_df, val_df, test_df = stratified_train_val_test_split(df)

    feature_config, feature_names = build_feature_config(train_df)

    def matrix(split_df):
        rows = []
        for _, row in split_df.iterrows():
            raw = row.to_dict()
            rows.append(transform_row(raw, feature_config)[0])
        return np.array(rows)

    X_train = matrix(train_df)
    y_train = train_df[target_col].values
    X_val = matrix(val_df)
    y_val = val_df[target_col].values

    from sklearn.ensemble import IsolationForest

    scaler_iso = StandardScaler()
    X_train_scaled = scaler_iso.fit_transform(X_train)
    X_val_scaled = scaler_iso.transform(X_val)
    train_xy = np.hstack([X_train_scaled, y_train.reshape(-1, 1)])
    val_xy = np.hstack([X_val_scaled, y_val.reshape(-1, 1)])
    iso = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    iso.fit(train_xy)
    train_mask = iso.predict(train_xy) == 1
    val_mask = iso.predict(val_xy) == 1
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    model = GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        n_estimators=100,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    joblib.dump(model, UI_MODELS_DIR / "income_model.pkl")
    config_path = UI_MODELS_DIR / "income_feature_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **feature_config,
                "model_name": "GradientBoostingRegressor",
                "target": "Net Income",
            },
            f,
            indent=2,
        )
    print(f"  Saved income model and config to {UI_MODELS_DIR}")


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    export_shortage_model()
    export_income_model()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
