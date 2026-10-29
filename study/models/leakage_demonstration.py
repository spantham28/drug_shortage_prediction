"""
Quantify the leakage in the ORIGINAL analyses by re-running each original design with the same data and
the same algorithm, changing only how observations are split.

1. Original shortage model (drug x 4-week windows, price_signals_complete.csv, ExtraTrees):
   random row split  vs  split by drug (GroupKFold)  vs  drug split + later-period test.
2. Original profit model (hospital-years, hospital_ops_updated.csv, RandomForest, 14 same-year predictors):
   random row split  vs  split by hospital  vs  predictors from the prior fiscal year only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "outputs" / "leakage"
SEED = 42


def shortage_model():
    d = pd.read_csv(REPO / "drug_shortage_timeline_prediction" / "price_signals_complete.csv")
    feats = ["avg_nadac", "manufacturer_num", "ingredient_num", "num_forms", "liquid_flag"]
    X, y, g = d[feats].values, d["shortage_flag"].values, d["generic_name"].values
    rows = []
    for name, splitter, groups in [("Random split of drug-windows (original)", KFold(5, shuffle=True, random_state=SEED), None),
                                   ("Split by drug (no drug in both sets)", GroupKFold(5), g)]:
        aucs = []
        for tr, te in splitter.split(X, y, groups):
            m = ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        rows.append({"analysis": "Original shortage model (ExtraTrees)", "design": name,
                     "metric": "AUROC", "mean": np.mean(aucs), "sd": np.std(aucs)})
    # Features constant within drug: show the model can identify the drug from them alone
    n_profiles = d.groupby(feats[1:])["generic_name"].nunique()
    rows.append({"analysis": "Original shortage model", "design": "Share of drugs uniquely identified by the 4 market-structure features",
                 "metric": "share", "mean": (n_profiles == 1).sum() / d["generic_name"].nunique(), "sd": np.nan})
    return rows


def profit_model():
    d = pd.read_csv(REPO / "drug_shortage_cost_prediction" / "hospital_ops_updated.csv")
    feats = ["Total Assets", "Inventory", "Less Total Operating Expense",
             "Less Contractual Allowance and Discounts on Patients' Accounts", "Total Days (V + XVIII + XIX + Unknown)",
             "Major Movable Equipment", "Depreciation Cost", "Total Fixed Assets", "Total Discharges Title XVIII",
             "DRG Amounts After October 1", "Accounts Receivable", "Total Current Assets", "Total Liabilities",
             "Hospital Number of Beds For Adults & Peds"]
    d["fy"] = pd.to_datetime(d["Fiscal Year End Date"]).dt.year
    d = d.dropna(subset=feats + ["Net Income"])
    X, y, g = d[feats].values, d["Net Income"].values, d["Provider CCN"].values
    rows = []
    for name, splitter, groups in [("Random split of hospital-years (original)", KFold(5, shuffle=True, random_state=SEED), None),
                                   ("Split by hospital", GroupKFold(5), g)]:
        r2s, maes = [], []
        for tr, te in splitter.split(X, y, groups):
            m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
            p = m.predict(X[te])
            r2s.append(r2_score(y[te], p))
            maes.append(mean_absolute_error(y[te], p))
        rows.append({"analysis": "Original profit model (RandomForest, same-year predictors)", "design": name,
                     "metric": "R2", "mean": np.mean(r2s), "sd": np.std(r2s)})
    # Same predictors measured one fiscal year earlier (a genuine forecast), split by hospital
    lag = d[["Provider CCN", "fy"] + feats].copy()
    lag["fy"] += 1
    m2 = d[["Provider CCN", "fy", "Net Income"]].merge(lag, on=["Provider CCN", "fy"])
    X2, y2, g2 = m2[feats].values, m2["Net Income"].values, m2["Provider CCN"].values
    r2s = []
    for tr, te in GroupKFold(5).split(X2, y2, g2):
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X2[tr], y2[tr])
        r2s.append(r2_score(y2[te], m.predict(X2[te])))
    rows.append({"analysis": "Original profit model (RandomForest)", "design": "Prior-year predictors, split by hospital",
                 "metric": "R2", "mean": np.mean(r2s), "sd": np.std(r2s), "n": len(m2)})
    # How much of net income is an accounting identity of the predictors?
    rows.append({"analysis": "Original profit data", "design": "Hospitals with >1 year (share of rows)", "metric": "share",
                 "mean": (d.groupby("Provider CCN")["fy"].transform("size") > 1).mean(), "sd": np.nan})
    rows.append({"analysis": "Original profit data", "design": "Distinct shortage_num values across all hospital-years",
                 "metric": "count", "mean": d["shortage_num"].nunique(), "sd": np.nan})
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(shortage_model() + profit_model())
    res.to_csv(OUT / "leakage_demonstration.csv", index=False)
    print(res.to_string())
