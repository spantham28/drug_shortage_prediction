"""
Aim 1: forecast the onset of a new generic drug shortage within the next 6 months.

Design (prevents the leakage present in the original pipeline)
  * Unit: drug market (NADAC product) x month. Features use only data observed at or before month t.
  * Temporal split: development months 2015-01..2021-06 (labels end 2021-12);
                    test months 2022-01..2025-01 (labels end 2025-07). No label window spans both periods.
  * Drug-level split: 20% of active ingredients are held out completely. Every strength / dosage form of a
    held-out ingredient is excluded from development, so the model cannot memorise drug identity.
  * Hyperparameters: randomized search scored by AUROC on 3 forward-chaining folds inside the development
    period. Each fold validates on a later 12-month block for a disjoint third of ingredients and trains only on
    the other ingredients' months whose 6-month labels closed before the block began. The refit model is
    evaluated once on the test period.
  * No oversampling and no outlier removal; test data are never altered.
  * Three model classes + two reference models (shortage history only; random).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SEED, calibration_intercept_slope, clf_metrics, cluster_bootstrap, dense_preprocessor,
                    fmt_ci, forward_group_cv, model_specs)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "aim1"
N_ITER = 30
N_BOOT = 1000

PRICE_FEATURES = ["lp", "dlp_1", "dlp_3", "dlp_6", "dlp_12", "vol_12", "max_jump_12", "dispersion"]
MARKET_FEATURES = ["n_ndcs", "n_labelers", "d_labelers_12", "d_ndcs_12", "new_market_12m"]
HISTORY_FEATURES = ["prior_shortages", "ever_active_before", "months_since_shortage"]
PRODUCT_FEATURES = ["form_injectable", "form_oral_solid", "form_liquid", "form_topical", "unit_ml", "unit_gm"]
SPILLOVER_FEATURES = ["ingr_active_other", "labeler_short_share", "labeler_short_any", "national_onsets_3m"]
FEATURES = PRICE_FEATURES + MARKET_FEATURES + HISTORY_FEATURES + PRODUCT_FEATURES + SPILLOVER_FEATURES
LOG_FEATURES = ["n_ndcs", "n_labelers", "prior_shortages", "ingr_active_other"]
VAL_BLOCKS = [("2018-07", "2019-06"), ("2019-07", "2020-06"), ("2020-07", "2021-06")]


def split(panel):
    rng = np.random.default_rng(SEED)
    ingredients = np.sort(panel["ingredient"].unique())
    held_out = set(rng.choice(ingredients, size=int(round(0.2 * len(ingredients))), replace=False))
    panel["held_out_ingredient"] = panel["ingredient"].isin(held_out).astype(int)
    dev = panel[(panel["month"] <= "2021-06") & (panel["held_out_ingredient"] == 0)].copy()
    test = panel[panel["month"] >= "2022-01"].copy()
    return dev, test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(ROOT / "data" / "interim" / "drug_month_panel.csv.gz")
    # Months since a market first appeared in NADAC rises mechanically with calendar time, so only a
    # time-stationary version (entered within the past 12 months) is used
    panel["new_market_12m"] = (panel["months_in_nadac"] < 12).astype(int)
    dev, test = split(panel)
    test_new = test[test["held_out_ingredient"] == 1]
    print(f"dev rows={len(dev):,} pos={dev.y.sum():,} | test rows={len(test):,} pos={test.y.sum():,} | "
          f"test held-out ingredients rows={len(test_new):,} pos={test_new.y.sum():,}")

    dev = dev.reset_index(drop=True)
    gap = lambda start: str(pd.Period(start, "M") - 7)
    cv = forward_group_cv(dev, "month", "ingredient", VAL_BLOCKS, gap)
    X, y = dev[FEATURES], dev["y"].values

    fitted, tuning = {}, []
    specs = model_specs("classification", FEATURES, LOG_FEATURES)
    for name, (est, grid) in specs.items():
        search = RandomizedSearchCV(est, grid, n_iter=N_ITER, scoring="roc_auc", cv=cv, random_state=SEED,
                                    n_jobs=1 if name.startswith("Gradient") else 5, refit=True)
        search.fit(X, y)
        fitted[name] = search.best_estimator_
        tuning.append({"model": name, "cv_auroc": search.best_score_,
                       "best_params": json.dumps({k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                                  for k, v in search.best_params_.items()}, default=str)})
        print(f"{name}: CV AUROC={search.best_score_:.3f}")

    # Reference model: shortage history only (no price / market signals)
    hist = Pipeline([("prep", dense_preprocessor(HISTORY_FEATURES, ["prior_shortages"])),
                     ("model", LogisticRegression(max_iter=2000))]).fit(dev[HISTORY_FEATURES], y)
    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)

    rows, preds = [], {}
    for label, frame in [("Held-out ingredients, 2022-2025", test_new), ("All ingredients, 2022-2025", test)]:
        frame = frame.reset_index(drop=True)
        scores = {name: m.predict_proba(frame[FEATURES])[:, 1] for name, m in fitted.items()}
        scores["Reference: shortage history only"] = hist.predict_proba(frame[HISTORY_FEATURES])[:, 1]
        for name, p in scores.items():
            d = frame[["ingredient", "y"]].assign(p=p)
            ci = cluster_bootstrap(d, "ingredient", lambda q: clf_metrics(q["y"].values, q["p"].values), N_BOOT)
            a, b = calibration_intercept_slope(d["y"].values, d["p"].values)
            top = d["p"] >= d["p"].quantile(0.95)
            rows.append({
                "test_set": label, "model": name, "n": len(d), "events": int(d["y"].sum()),
                "prevalence": d["y"].mean(),
                "AUROC": fmt_ci(ci["AUROC"]), "AUPRC": fmt_ci(ci["AUPRC"]), "Brier": fmt_ci(ci["Brier"], 4),
                "AUROC_est": ci["AUROC"][0], "AUPRC_est": ci["AUPRC"][0],
                "cal_intercept": a, "cal_slope": b,
                "top5pct_sensitivity": d.loc[top, "y"].sum() / max(d["y"].sum(), 1),
                "top5pct_ppv": d.loc[top, "y"].mean(),
                "top5pct_lift": d.loc[top, "y"].mean() / d["y"].mean(),
            })
            preds[(label, name)] = p
        if label.startswith("Held-out"):
            frame.assign(**{f"p_{k}": v for k, v in scores.items()})[["market", "ingredient", "month", "y"] +
                [f"p_{k}" for k in scores]].to_csv(OUT / "test_predictions_heldout.csv.gz", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "performance.csv", index=False)
    print(res[["test_set", "model", "events", "AUROC", "AUPRC", "cal_slope", "top5pct_lift"]].to_string())

    # Permutation importance (AUROC drop) for the best model on held-out-ingredient test data
    best = res[res["test_set"].str.startswith("Held-out") & ~res["model"].str.startswith("Reference")] \
        .sort_values("AUROC_est", ascending=False)["model"].iloc[0]
    tn = test_new.reset_index(drop=True)
    imp = permutation_importance(fitted[best], tn[FEATURES], tn["y"], scoring="roc_auc", n_repeats=10,
                                 random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "auroc_drop_mean": imp.importances_mean, "auroc_drop_sd": imp.importances_std}) \
        .sort_values("auroc_drop_mean", ascending=False).to_csv(OUT / f"permutation_importance.csv", index=False)
    json.dump({"best_model": best, "features": FEATURES, "n_dev": len(dev), "n_test": len(test),
               "n_test_heldout": len(test_new), "dev_events": int(dev.y.sum()), "dev_markets": dev.market.nunique(),
               "dev_ingredients": dev.ingredient.nunique(), "test_markets": test.market.nunique(),
               "heldout_ingredients": int(test_new.ingredient.nunique())},
              open(OUT / "summary.json", "w"), indent=2)

    # Calibration curve data for the figure
    cal = []
    for (label, name), p in preds.items():
        frame = test_new if label.startswith("Held-out") else test
        d = pd.DataFrame({"y": frame["y"].values, "p": p})
        d["bin"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
        c = d.groupby("bin").agg(mean_pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
        cal.append(c.assign(test_set=label, model=name))
    pd.concat(cal).to_csv(OUT / "calibration_deciles.csv", index=False)


if __name__ == "__main__":
    main()
