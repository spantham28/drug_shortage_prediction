"""
Aim 2b: does hospital-specific shortage exposure improve forecasts of a hospital's FUTURE operating margin?

Prediction timing
  A forecast is made at the close of fiscal year t using only cost-report data for years <= t and shortage
  data available by the end of year t.  Outcomes are operating margin in year t+h (h = 1 or 2 years) and an
  operating loss (margin < 0) in year t+h.  No component of the year t+h income statement is a predictor.

Splits (no hospital and no outcome year appears on both sides)
  * 20% of hospitals are held out entirely.
  * Fixed origin: models are trained and tuned on development hospitals with outcome years <= 2022 - h
    (so every training label was observable when the test forecasts would have been made) and tested on
    held-out hospitals with outcome years 2022-2024.
  * Rolling origin (h = 1): for each test year Y = 2020..2024, retrain on development hospitals with outcome
    years <= Y - 1, test on held-out hospitals in year Y.
  * Hyperparameters: randomized search inside the training data only, using up to 3 forward-chaining folds
    that are disjoint in hospitals (validate on a later outcome year for one third of hospitals; train on the
    remaining hospitals' outcomes observable h years earlier).

Feature sets
  A  financial and operating history (no shortage information)
  B  A + shortage exposure known at forecast time (year-t exposure, change in exposure, shortages still
     active at fiscal year end)
  C  B + realized exposure in the outcome year (not knowable in advance; an upper bound on the value of
     perfect shortage foresight)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, calibration_intercept_slope, clf_metrics, cluster_bootstrap, fmt_ci, forward_group_cv,
                    model_specs, reg_metrics)

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs" / "aim2"
N_ITER = 20
N_ITER_ROLLING = 10
N_BOOT = 1000

REGION = {s: r for r, states in {
    "northeast": "CT ME MA NH RI VT NJ NY PA",
    "midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "south": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX PR",
    "west": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items() for s in states.split()}

FIN = ["operating_margin", "total_margin", "d_operating_margin", "log_npr", "npr_growth", "opex_growth",
       "days_cash", "current_ratio", "debt_ratio", "log_npr_per_apd", "log_opex_per_apd", "log_drug_cost_per_apd",
       "drug_share_of_cost", "discount_rate", "other_income_share"]
OPS = ["log_beds", "occupancy", "log_discharges", "fte_per_bed", "medicare_day_share", "medicaid_day_share",
       "icu_day_share", "periop_share", "ed_share", "outpatient_rev_share"]
STRUCT = ["teaching", "system_member", "rural", "own_for_profit", "own_government", "region_midwest",
          "region_south", "region_west"]
EXPOSURE_KNOWN = ["exposure", "d_exposure", "carry_in", "exposure_parenteral", "exposure_limited_supply"]
EXPOSURE_FUTURE = ["exposure_target_year"]
EXPOSURE_RELATIVE = [f"{c}_rel" for c in EXPOSURE_KNOWN]
LOG_FEATURES = ["days_cash", "current_ratio", "debt_ratio", "fte_per_bed"]
FEATURE_SETS = {
    "A: financial/operating history": FIN + OPS + STRUCT,
    "B: + shortage exposure known at forecast": FIN + OPS + STRUCT + EXPOSURE_KNOWN,
    "C: + realized target-year exposure (oracle)": FIN + OPS + STRUCT + EXPOSURE_KNOWN + EXPOSURE_FUTURE,
}


def build_features():
    """Hospital-year predictors known at the close of each fiscal year (no outcome attached)."""
    h = pd.read_csv(INTERIM / "hospital_panel.csv", dtype={"ccn": str})
    e = pd.read_csv(INTERIM / "hospital_exposure.csv", dtype={"ccn": str})
    d = h.merge(e.drop(columns=["fyb", "fye"]), on=["ccn", "fy"])
    d = d[d["has_baseline"] == 1].sort_values(["ccn", "fy"])
    # Pre-specified data-quality rule: implausible operating margins (beyond +/-100%) are treated as errors
    d = d[d["operating_margin"].abs() <= 1].copy()

    d["log_npr"] = np.log(d["npr"])
    d["log_beds"] = np.log(d["beds"])
    d["log_discharges"] = np.log1p(d["discharges"].clip(lower=0))
    d["log_npr_per_apd"] = np.log(d["npr_per_apd"].clip(lower=1))
    d["log_opex_per_apd"] = np.log(d["opex_per_apd"].clip(lower=1))
    d["log_drug_cost_per_apd"] = np.log1p(d["drug_cost_per_apd"].clip(lower=0))
    d["discount_rate"] = d["contractual"] / d["tpr"]
    d["other_income_share"] = d["other_income"].fillna(0) / d["npr"]
    d["own_for_profit"] = (d["ownership"] == "for_profit").astype(int)
    d["own_government"] = (d["ownership"] == "government").astype(int)
    region = d["state"].map(REGION).fillna("south")
    for r in ["midwest", "south", "west"]:
        d[f"region_{r}"] = (region == r).astype(int)

    prev = d[["ccn", "fy", "operating_margin", "npr", "opex", "exposure"]].copy()
    prev["fy"] += 1
    d = d.merge(prev.rename(columns={"operating_margin": "om_prev", "npr": "npr_prev", "opex": "opex_prev",
                                     "exposure": "exposure_prev"}), on=["ccn", "fy"], how="left")
    d["d_operating_margin"] = d["operating_margin"] - d["om_prev"]
    d["npr_growth"] = d["npr"] / d["npr_prev"] - 1
    d["opex_growth"] = d["opex"] / d["opex_prev"] - 1
    d["d_exposure"] = d["exposure"] - d["exposure_prev"]
    # Hospital-relative exposure: deviation from the mean of all hospitals in the same fiscal year
    # (removes the national trend, which every hospital shares and which is known at forecast time)
    for c in ["exposure", "d_exposure", "carry_in", "exposure_parenteral", "exposure_limited_supply"]:
        d[f"{c}_rel"] = d[c] - d.groupby("fy")[c].transform("mean")
    for c in ["npr_growth", "opex_growth", "discount_rate", "other_income_share", "days_cash", "current_ratio",
              "debt_ratio", "fte_per_bed", "occupancy", "total_margin"]:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    d["total_margin"] = d["total_margin"].clip(-1, 1)
    d["npr_growth"] = d["npr_growth"].clip(-0.9, 3)
    d["opex_growth"] = d["opex_growth"].clip(-0.9, 3)
    return d


def build_frame(horizon):
    d = build_features()
    target = d[["ccn", "fy", "operating_margin", "exposure"]].copy()
    target["fy"] -= horizon
    target = target.rename(columns={"operating_margin": "y_margin", "exposure": "exposure_target_year"})
    f = d.merge(target, on=["ccn", "fy"], how="inner")
    f["target_fy"] = f["fy"] + horizon
    f["y_margin_pp"] = 100 * f["y_margin"]
    f["y_loss"] = (f["y_margin"] < 0).astype(int)
    f = f[(f["fy"] >= 2016) & (f["target_fy"] <= 2024)]
    return f.reset_index(drop=True)


def held_out_hospitals(frame):
    rng = np.random.default_rng(SEED)
    ccns = np.sort(frame["ccn"].unique())
    return set(rng.choice(ccns, size=int(round(0.2 * len(ccns))), replace=False))


def tune(task, features, train, n_iter, horizon):
    # Forward-chaining folds, disjoint in hospitals: validate on a later outcome year for one third of hospitals,
    # train on the other hospitals' outcome years that were observable h years before that validation year
    years = sorted(train["target_fy"].unique())
    blocks = [y for y in years if any(t <= y - horizon for t in years)][-3:]
    cv = forward_group_cv(train.reset_index(drop=True), "target_fy", "ccn", [(y, y) for y in blocks],
                          lambda y: y - horizon)
    train = train.reset_index(drop=True)
    y = train["y_loss"] if task == "classification" else train["y_margin_pp"]
    scoring = "roc_auc" if task == "classification" else "neg_mean_absolute_error"
    fitted = {}
    for name, (est, grid) in model_specs(task, features, LOG_FEATURES).items():
        s = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring=scoring, cv=cv, random_state=SEED,
                               n_jobs=1 if name.startswith("Gradient") else 8)
        s.fit(train[features], y)
        fitted[name] = (s.best_estimator_, s.best_score_, s.best_params_)
    return fitted


def persistence(task, train, test):
    if task == "regression":
        return 100 * test["operating_margin"].values
    m = LogisticRegression().fit(train[["operating_margin"]], train["y_loss"])
    return m.predict_proba(test[["operating_margin"]])[:, 1]


def evaluate(task, test, preds):
    rows = []
    for name, p in preds.items():
        d = test[["ccn", "y_margin_pp", "y_loss"]].assign(p=p)
        if task == "regression":
            ci = cluster_bootstrap(d, "ccn", lambda q: reg_metrics(q["y_margin_pp"].values, q["p"].values), N_BOOT)
            rows.append({"model": name, "n": len(d), "MAE_pp": fmt_ci(ci["MAE"], 2), "RMSE_pp": fmt_ci(ci["RMSE"], 2),
                         "R2": fmt_ci(ci["R2"], 3), "MAE_est": ci["MAE"][0], "R2_est": ci["R2"][0]})
        else:
            ci = cluster_bootstrap(d, "ccn", lambda q: clf_metrics(q["y_loss"].values, q["p"].values), N_BOOT)
            a, b = calibration_intercept_slope(d["y_loss"].values, d["p"].values)
            rows.append({"model": name, "n": len(d), "events": int(d["y_loss"].sum()), "AUROC": fmt_ci(ci["AUROC"]),
                         "AUPRC": fmt_ci(ci["AUPRC"]), "Brier": fmt_ci(ci["Brier"], 4), "AUROC_est": ci["AUROC"][0],
                         "cal_intercept": a, "cal_slope": b})
    return rows


def paired_difference(task, test, p_a, p_b):
    """Hospital-cluster bootstrap of the change in MAE (regression) or AUROC (classification), B minus A."""
    d = test[["ccn", "y_margin_pp", "y_loss"]].assign(pa=p_a, pb=p_b)
    if task == "regression":
        fn = lambda q: {"delta": np.mean(np.abs(q["y_margin_pp"] - q["pb"])) - np.mean(np.abs(q["y_margin_pp"] - q["pa"]))}
    else:
        from sklearn.metrics import roc_auc_score
        fn = lambda q: {"delta": roc_auc_score(q["y_loss"], q["pb"]) - roc_auc_score(q["y_loss"], q["pa"])}
    return cluster_bootstrap(d, "ccn", fn, N_BOOT)["delta"]


def fixed_origin(horizon, held_out, results, deltas, tuning_log, importance):
    f = build_frame(horizon)
    train = f[(~f["ccn"].isin(held_out)) & (f["target_fy"] <= 2022 - horizon)]
    test = f[(f["ccn"].isin(held_out)) & (f["target_fy"] >= 2022)].reset_index(drop=True)
    print(f"h={horizon}: train {len(train):,} rows / {train.ccn.nunique():,} hospitals; "
          f"test {len(test):,} rows / {test.ccn.nunique():,} held-out hospitals")
    sets = {k: v for k, v in FEATURE_SETS.items() if horizon == 1 or not k.startswith("C")}
    for task in ["regression", "classification"]:
        preds_by_set = {}
        for set_name, feats in sets.items():
            fitted = tune(task, feats, train, N_ITER, horizon)
            preds = {}
            for name, (m, score, params) in fitted.items():
                p = m.predict(test[feats]) if task == "regression" else m.predict_proba(test[feats])[:, 1]
                preds[name] = p
                tuning_log.append({"horizon": horizon, "task": task, "feature_set": set_name, "model": name,
                                   "cv_score": score, "best_params": json.dumps(params, default=str)})
                if set_name.startswith("B") and name.startswith("Gradient"):
                    booster = m.get_booster()
                    gain = booster.get_score(importance_type="total_gain")
                    tot = sum(gain.values())
                    for k, v in gain.items():
                        importance.append({"horizon": horizon, "task": task, "feature": k, "share_of_gain": v / tot})
            preds_by_set[set_name] = preds
            if set_name.startswith("A"):
                preds = {"Reference: persistence (last year's margin)": persistence(task, train, test), **preds}
            for r in evaluate(task, test, preds):
                results.append({"horizon": horizon, "task": task, "feature_set": set_name, **r})
        for name in preds_by_set["A: financial/operating history"]:
            for other in [k for k in sets if not k.startswith("A")]:
                est, lo, hi = paired_difference(task, test, preds_by_set["A: financial/operating history"][name],
                                                preds_by_set[other][name])
                deltas.append({"horizon": horizon, "task": task, "model": name, "comparison": f"{other} vs A",
                               "metric": "change in MAE (pp)" if task == "regression" else "change in AUROC",
                               "delta": est, "ci_lo": lo, "ci_hi": hi})
    return f


def rolling_origin(held_out):
    f = build_frame(1)
    rows = []
    for year in range(2020, 2025):
        train = f[(~f["ccn"].isin(held_out)) & (f["target_fy"] <= year - 1)]
        test = f[(f["ccn"].isin(held_out)) & (f["target_fy"] == year)].reset_index(drop=True)
        for task in ["regression", "classification"]:
            for set_name in ["A: financial/operating history", "B: + shortage exposure known at forecast"]:
                feats = FEATURE_SETS[set_name]
                fitted = tune(task, feats, train, N_ITER_ROLLING, 1)
                for name, (m, _, _) in fitted.items():
                    if task == "regression":
                        met = reg_metrics(test["y_margin_pp"].values, m.predict(test[feats]))
                    else:
                        met = clf_metrics(test["y_loss"].values, m.predict_proba(test[feats])[:, 1])
                    rows.append({"test_year": year, "task": task, "feature_set": set_name, "model": name,
                                 "n_train": len(train), "n_test": len(test), **met})
            ref = persistence(task, train, test)
            met = reg_metrics(test["y_margin_pp"].values, ref) if task == "regression" else \
                clf_metrics(test["y_loss"].values, ref)
            rows.append({"test_year": year, "task": task, "feature_set": "reference",
                         "model": "Reference: persistence", "n_train": len(train), "n_test": len(test), **met})
        print("rolling origin done:", year)
    pd.DataFrame(rows).to_csv(OUT / "rolling_origin.csv", index=False)


def posthoc_relative():
    """Post hoc (disclosed): exposure expressed relative to all hospitals in the same fiscal year.
    Motivated by the primary result, in which the raw exposure level - largely a national trend that rose
    beyond its training range in 2022-2024 - degraded forecasts."""
    held_out = held_out_hospitals(build_frame(1))
    rows, deltas = [], []
    feats_a = FEATURE_SETS["A: financial/operating history"]
    feats_r = feats_a + EXPOSURE_RELATIVE
    for horizon in (1, 2):
        f = build_frame(horizon)
        train = f[(~f["ccn"].isin(held_out)) & (f["target_fy"] <= 2022 - horizon)]
        test = f[(f["ccn"].isin(held_out)) & (f["target_fy"] >= 2022)].reset_index(drop=True)
        for task in ["regression", "classification"]:
            fa = tune(task, feats_a, train, N_ITER, horizon)
            fr = tune(task, feats_r, train, N_ITER, horizon)
            for name in fa:
                pa = fa[name][0].predict(test[feats_a]) if task == "regression" else fa[name][0].predict_proba(test[feats_a])[:, 1]
                pr = fr[name][0].predict(test[feats_r]) if task == "regression" else fr[name][0].predict_proba(test[feats_r])[:, 1]
                for r in evaluate(task, test, {name: pr}):
                    rows.append({"horizon": horizon, "task": task, "feature_set": "B-rel: + hospital-relative exposure", **r})
                est, lo, hi = paired_difference(task, test, pa, pr)
                deltas.append({"horizon": horizon, "task": task, "model": name,
                               "comparison": "B-rel: + hospital-relative exposure vs A",
                               "metric": "change in MAE (pp)" if task == "regression" else "change in AUROC",
                               "delta": est, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(OUT / "forecast_performance_posthoc_relative.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "forecast_incremental_value_posthoc_relative.csv", index=False)
    print(pd.DataFrame(deltas).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    held_out = held_out_hospitals(build_frame(1))
    results, deltas, tuning_log, importance = [], [], [], []
    f1 = fixed_origin(1, held_out, results, deltas, tuning_log, importance)
    fixed_origin(2, held_out, results, deltas, tuning_log, importance)
    pd.DataFrame(results).to_csv(OUT / "forecast_performance.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "forecast_incremental_value.csv", index=False)
    pd.DataFrame(tuning_log).to_csv(OUT / "forecast_tuning.csv", index=False)
    pd.DataFrame(importance).to_csv(OUT / "forecast_importance_xgb.csv", index=False)
    f1.to_csv(OUT / "forecast_frame_h1.csv.gz", index=False, compression="gzip")
    json.dump({"held_out_hospitals": len(held_out), "n_hospitals": int(f1["ccn"].nunique())},
              open(OUT / "forecast_summary.json", "w"))
    print(pd.DataFrame(results).drop(columns=[c for c in ["MAE_est", "R2_est", "AUROC_est"] if c in results[0]],
                                     errors="ignore").to_string())
    print(pd.DataFrame(deltas).to_string())
    rolling_origin(held_out)


if __name__ == "__main__":
    posthoc_relative() if "--posthoc" in sys.argv else main()
