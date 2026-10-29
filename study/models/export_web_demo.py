"""
Export the revised models' outputs as static JSON for the web demo (ui/public/data/).

After validation, each final model is refit on all labeled data with the hyperparameters selected during
validation, then used to score the most recent period:
  * Aim 1: penalized logistic regression (best model on held-out ingredients) scores every generic product
    priced in NADAC in August 2025, the last month with a known shortage status. The fitted preprocessing
    and coefficients are exported so the browser can recompute risk when a user edits a product's inputs.
  * Aim 2b: gradient-boosted trees, feature set A (best model for next-year margin and operating loss)
    forecast each hospital's next fiscal year from its latest cost report.
  * Aim 2a: each hospital's exposure index history and the fixed-effects estimate of the drug expense
    associated with being above or below the national average exposure.
Validation metrics shown in the demo are the held-out results, not the refit models' in-sample fit.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aim1_shortage_forecast as a1
import aim2b_margin_forecast as a2
from common import SEED, dense_preprocessor

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
DEST = ROOT.parent / "ui" / "public" / "data"
HOSPITAL_NAMES_URL = ("https://data.cms.gov/provider-data/sites/default/files/resources/"
                      "893c372430d9d71a1c52737d01239d47_1785189955/Hospital_General_Information.csv")

FEATURE_LABELS = {
    "lp": "Log NADAC price per unit", "dlp_1": "Price change, 1 mo (log)", "dlp_3": "Price change, 3 mo (log)",
    "dlp_6": "Price change, 6 mo (log)", "dlp_12": "Price change, 12 mo (log)", "vol_12": "Price volatility, 12 mo",
    "max_jump_12": "Largest monthly price jump, 12 mo", "dispersion": "Price spread across manufacturers",
    "n_ndcs": "Number of NDCs priced", "n_labelers": "Number of manufacturers", "d_labelers_12":
    "Change in manufacturers, 12 mo", "d_ndcs_12": "Change in NDCs, 12 mo", "new_market_12m": "New to NADAC (<12 mo)",
    "prior_shortages": "Previous shortages of this product", "ever_active_before": "Ever in shortage before",
    "months_since_shortage": "Months since last shortage (120 = never)", "form_injectable": "Injectable",
    "form_oral_solid": "Tablet or capsule", "form_liquid": "Oral liquid", "form_topical": "Topical / inhaled / drops",
    "unit_ml": "Priced per mL", "unit_gm": "Priced per gram", "ingr_active_other":
    "Other forms of same ingredient in shortage", "labeler_short_share":
    "Share of manufacturers with a shortage elsewhere", "labeler_short_any": "Any manufacturer has a shortage elsewhere",
    "national_onsets_3m": "New shortages nationally, last 3 mo",
}
EDITABLE = ["n_labelers", "d_labelers_12", "dlp_6", "dlp_12", "labeler_short_share", "ingr_active_other",
            "months_since_shortage", "max_jump_12"]


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def linear_parts(pipe, features):
    """Unpack the fitted ColumnTransformer + logistic regression into per-feature parameters."""
    ct, lr = pipe.named_steps["prep"], pipe.named_steps["model"]
    coef = lr.coef_.ravel()
    params, k, intercept = {}, 0, float(lr.intercept_[0])
    for name, branch, cols in ct.transformers_:
        if name == "remainder":
            continue
        imp, sc = branch.named_steps["impute"], branch.named_steps["scale"]
        n = len(cols)
        for i, c in enumerate(cols):
            params[c] = {"slog": name == "log", "median": float(imp.statistics_[i]), "mean": float(sc.mean_[i]),
                         "scale": float(sc.scale_[i]), "coef": float(coef[k + i])}
        # Missing-value indicator columns are 0 for complete inputs; fold their constant term into the intercept
        n_ind = len(imp.indicator_.features_) if imp.indicator_ is not None else 0
        for j in range(n_ind):
            intercept += coef[k + n + j] * (0 - sc.mean_[n + j]) / sc.scale_[n + j]
        k += n + n_ind
    assert k == len(coef)
    return params, intercept


def export_shortage():
    panel = pd.read_csv(INTERIM / "drug_month_panel.csv.gz")
    panel["new_market_12m"] = (panel["months_in_nadac"] < 12).astype(int)
    best = json.loads(pd.read_csv(OUTPUTS / "aim1" / "tuning.csv").set_index("model").loc["Penalized linear", "best_params"])
    pipe = Pipeline([("prep", dense_preprocessor(a1.FEATURES, a1.LOG_FEATURES)),
                     ("model", LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED,
                                                  C=best["model__C"], penalty=best["model__penalty"]))])
    pipe.fit(panel[a1.FEATURES], panel["y"])
    params, intercept = linear_parts(pipe, a1.FEATURES)

    snap = pd.read_csv(INTERIM / "drug_scoring_snapshot.csv.gz")
    snap["new_market_12m"] = (snap["months_in_nadac"] < 12).astype(int)
    current = sorted(snap.loc[snap["active"] == 1, "market"].unique().tolist())
    s = snap[snap["active"] == 0].reset_index(drop=True)
    s["risk"] = pipe.predict_proba(s[a1.FEATURES])[:, 1]

    def contrib(row):
        out = []
        for f, p in params.items():
            x = row[f] if pd.notna(row[f]) else p["median"]
            g = np.sign(x) * np.log1p(abs(x)) if p["slog"] else x
            out.append((f, p["coef"] * (g - p["mean"]) / p["scale"]))
        return out

    base_rate = float(panel["y"].mean())
    products = []
    for _, row in s.sort_values("risk", ascending=False).iterrows():
        c = sorted(contrib(row), key=lambda t: -t[1])
        products.append({
            "market": row["market"], "ingredient": row["ingredient"], "risk": r(row["risk"], 5),
            "price": r(row["price"], 4), "unit": row["pricing_unit"],
            "x": {f: (None if pd.isna(row[f]) else r(row[f], 4)) for f in a1.FEATURES},
            "drivers": [FEATURE_LABELS[f] for f, v in c[:3] if v > 0],
        })
    perf = pd.read_csv(OUTPUTS / "aim1" / "performance.csv")
    lin = perf[perf["model"] == "Penalized linear"].set_index("test_set")
    out = {
        "asOf": "2025-08", "horizonMonths": 6, "baseRate": base_rate, "intercept": intercept,
        "features": [{"key": f, "label": FEATURE_LABELS[f], "editable": f in EDITABLE, **params[f]} for f in a1.FEATURES],
        "products": products, "currentShortages": current,
        "validation": {k.split(",")[0]: {"auroc": v["AUROC"], "top5Sensitivity": r(v["top5pct_sensitivity"], 3),
                                         "top5Lift": r(v["top5pct_lift"], 2)} for k, v in lin.iterrows()},
    }
    (DEST / "shortage.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"shortage.json: {len(products):,} products scored, {len(current):,} currently in shortage")


def export_hospitals():
    tune = pd.read_csv(OUTPUTS / "aim2" / "forecast_tuning.csv")
    sel = lambda task: json.loads(tune[(tune.horizon == 1) & (tune.task == task) &
                                       (tune.feature_set.str.startswith("A")) &
                                       (tune.model == "Gradient-boosted trees")].iloc[0]["best_params"])
    feats = a2.FEATURE_SETS["A: financial/operating history"]
    frame = a2.build_frame(1)
    reg = XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=8, **sel("regression")) \
        .fit(frame[feats], frame["y_margin_pp"])
    clf = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=8,
                        **sel("classification")).fit(frame[feats], frame["y_loss"])

    d = a2.build_features()
    latest = d.sort_values("fy").groupby("ccn").tail(1)
    latest = latest[latest["fy"] >= 2023].copy()
    latest["forecast_margin"] = reg.predict(latest[feats])
    latest["loss_prob"] = clf.predict_proba(latest[feats])[:, 1]

    ex = pd.read_csv(INTERIM / "hospital_exposure.csv", dtype={"ccn": str})
    ex = ex[(ex["has_baseline"] == 1) & ex["fy"].between(2016, 2024)]
    nat = ex.groupby("fy")["exposure"].mean()
    fe = pd.read_csv(OUTPUTS / "aim2" / "fixed_effects_results.csv")
    beta = fe[(fe.outcome == "log_drug_expense") & (fe.specification == "Main: service-weighted exposure")].iloc[0]
    panel = pd.read_csv(INTERIM / "hospital_panel.csv", dtype={"ccn": str})

    # Structural traits from each hospital's first (baseline) report: since FY2019 many urban hospitals report
    # a rural reclassification on Worksheet S-2, so the latest report overstates rurality
    first = panel.sort_values("fy").groupby("ccn").head(1).set_index("ccn")
    titled = lambda t: re.sub(r"'S\b", "'s", t.title())
    names_path = ROOT / "data" / "raw" / "hospital_general_information.csv"
    if not names_path.exists():  # CMS Care Compare "Hospital General Information" (names, cities)
        subprocess.run(["curl", "-sfL", HOSPITAL_NAMES_URL, "-o", str(names_path)], check=True)
    names = pd.read_csv(names_path, dtype=str)
    names = names.set_index(names["Facility ID"].str.zfill(6))[["Facility Name", "City/Town", "State"]]

    hospitals = []
    for ccn, g in ex.groupby("ccn"):
        if ccn not in latest["ccn"].values:
            continue
        L = latest[latest["ccn"] == ccn].iloc[0]
        hp = panel[(panel["ccn"] == ccn) & panel["fy"].between(2016, 2025)].set_index("fy")
        e24 = g.set_index("fy")["exposure"]
        last_e_year = int(e24.index.max())
        dev = float(e24.loc[last_e_year] - nat.loc[last_e_year])
        drug = float(hp["drug_expense"].get(last_e_year, np.nan)) if last_e_year in hp.index else np.nan
        nm = names.loc[ccn] if ccn in names.index else None
        hospitals.append({
            "ccn": ccn,
            "name": (titled(nm["Facility Name"]) if nm is not None else f"Hospital {ccn}"),
            "city": (titled(nm["City/Town"]) if nm is not None else ""),
            "state": (nm["State"] if nm is not None else str(L["state"])),
            "beds": int(L["beds"]), "ownership": L["ownership"], "teaching": int(L["teaching"]),
            "rural": int(first.loc[ccn, "rural"]), "system": int(L["system_member"]),
            "exposure": {int(k): r(v, 1) for k, v in e24.items()},
            "exposureYear": last_e_year, "exposureVsNational": r(dev, 1),
            "exposurePercentile": r(float((ex[ex["fy"] == last_e_year]["exposure"] < e24.loc[last_e_year]).mean() * 100), 0),
            "drugExpense": r(drug, 0),
            "drugExpenseVsAverage": r(drug * (np.exp(beta["coef"] * dev) - 1), 0) if np.isfinite(drug) else None,
            "margin": {int(k): r(100 * v, 2) for k, v in hp["operating_margin"].items() if abs(v) <= 1},
            "forecastYear": int(L["fy"]) + 1, "forecastMargin": r(L["forecast_margin"], 2),
            "lossProbability": r(L["loss_prob"], 3),
        })
    fp = pd.read_csv(OUTPUTS / "aim2" / "forecast_performance.csv")
    A = fp[(fp.horizon == 1) & fp.feature_set.str.startswith("A")]
    gbt_reg = A[(A.task == "regression") & (A.model == "Gradient-boosted trees")].iloc[0]
    gbt_cls = A[(A.task == "classification") & (A.model == "Gradient-boosted trees")].iloc[0]
    out = {
        "years": list(range(2016, 2025)), "nationalExposure": {int(k): r(v, 1) for k, v in nat.items()},
        "betaLogPerUnit": float(beta["coef"]), "betaCi": [float(beta["ci_lo"]), float(beta["ci_hi"])],
        "validation": {"marginMAE": gbt_reg["MAE_pp"], "marginR2": gbt_reg["R2"], "lossAUROC": gbt_cls["AUROC"]},
        "hospitals": sorted(hospitals, key=lambda h: h["name"]),
    }
    (DEST / "hospitals.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"hospitals.json: {len(hospitals):,} hospitals")


def export_validation():
    leak = pd.read_csv(OUTPUTS / "leakage" / "leakage_demonstration.csv")
    perf1 = pd.read_csv(OUTPUTS / "aim1" / "performance.csv")
    fp = pd.read_csv(OUTPUTS / "aim2" / "forecast_performance.csv")
    inc = pd.read_csv(OUTPUTS / "tables" / "etable_aim2b_incremental_value.csv")
    fe = pd.read_csv(OUTPUTS / "aim2" / "fixed_effects_results.csv")
    num = json.loads((OUTPUTS / "numbers.json").read_text())
    keep_fe = fe[fe.outcome.isin(["log_drug_expense", "operating_margin_pp"]) & (fe.term == "exposure")]
    out = {
        "leakage": leak[["analysis", "design", "metric", "mean"]].to_dict(orient="records"),
        "aim1": perf1[["test_set", "model", "events", "AUROC", "AUPRC", "top5pct_sensitivity", "top5pct_lift"]]
        .to_dict(orient="records"),
        "aim2a": keep_fe[["outcome", "specification", "coef", "ci_lo", "ci_hi", "identifying_variation_share"]]
        .to_dict(orient="records"),
        "aim2b": fp[["horizon", "task", "feature_set", "model", "MAE_pp", "R2", "AUROC"]].fillna("").to_dict(orient="records"),
        "aim2bIncremental": inc.to_dict(orient="records"),
        "headline": {k: num[k] for k in ["n_markets", "n_market_months", "n_hosp", "n_hosp_years", "fe_pct",
                                          "fe_pct_lo", "fe_pct_hi", "fe_usd_mean", "cv_ratio", "a1_all_auc_ci",
                                          "a1_new_auc_ci", "a1_top5_sens", "a1_top5_lift", "drug_share_opex_median"]},
    }
    (DEST / "validation.json").write_text(json.dumps(out, separators=(",", ":"), default=str))
    print("validation.json written")


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    export_shortage()
    export_hospitals()
    export_validation()
