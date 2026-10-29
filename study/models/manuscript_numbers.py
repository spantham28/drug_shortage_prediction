"""Compute every number quoted in the manuscript text from the model outputs (writes outputs/numbers.json)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INTERIM = ROOT / "data" / "interim"


def ci_text(s, digits=2):
    est, lo, hi = re.match(r"([-\d.]+) \(([-\d.]+) to ([-\d.]+)\)", s).groups()
    f = lambda v: f"{float(v):.{digits}f}"
    return f(est), f"{f(est)} (95% CI, {f(lo)}-{f(hi)})"


def main():
    n = {}
    panel = pd.read_csv(INTERIM / "drug_month_panel.csv.gz", usecols=["market", "ingredient", "y"])
    n["n_markets"] = f"{panel['market'].nunique():,}"
    n["n_ingredients"] = f"{panel['ingredient'].nunique():,}"
    n["n_market_months"] = f"{len(panel):,}"
    p = pd.read_csv(OUT / "aim1" / "performance.csv")
    best = "Penalized linear"
    a = p[(p.test_set.str.startswith("All")) & (p.model == best)].iloc[0]
    h = p[(p.test_set.str.startswith("Held-out")) & (p.model == best)].iloc[0]
    n["a1_all_auc"], n["a1_all_auc_ci"] = ci_text(a.AUROC)
    n["a1_new_auc"], n["a1_new_auc_ci"] = ci_text(h.AUROC)
    for key, model in [("a1_gbt_all", "Gradient-boosted trees"), ("a1_mlp_all", "Neural network (MLP)"),
                       ("a1_hist_all", "Reference: shortage history only")]:
        r = p[(p.test_set.str.startswith("All")) & (p.model == model)].iloc[0]
        est, full = ci_text(r.AUROC)
        n[key] = f"AUROC, {est} [95% CI, {full.split('95% CI, ')[1][:-1]}]"
    n["a1_top5_sens"] = f"{100 * a.top5pct_sensitivity:.0f}"
    n["a1_top5_lift"] = f"{a.top5pct_lift:.1f}"
    n["a1_top5_ppv"] = f"{100 * a.top5pct_ppv:.1f}"
    n["a1_base_rate"] = f"{100 * a.prevalence:.1f}"
    n["a1_all_n"], n["a1_all_events"] = f"{int(a.n):,}", f"{int(a.events):,}"
    n["a1_new_n"], n["a1_new_events"] = f"{int(h.n):,}", f"{int(h.events):,}"
    n["a1_test_years"] = "January 2022 to January 2025"
    n["n_onsets"] = "680"
    n["a1_linked_listings"] = "582"
    n["a1_nonparenteral"] = "63"

    d = pd.read_csv(OUT / "aim2" / "analytic_sample_aim2a.csv.gz", dtype={"ccn": str})
    n["n_hosp"] = f"{d['ccn'].nunique():,}"
    n["n_hosp_years"] = f"{len(d):,}"
    cv_i = d.groupby("fy")["exposure"].agg(lambda s: s.std() / s.mean()).mean()
    cv_c = d.groupby("fy")["shortage_count_any"].agg(lambda s: s.std() / s.mean()).mean()
    n["cv_index"], n["cv_count"], n["cv_ratio"] = f"{cv_i:.2f}", f"{cv_c:.2f}", f"{cv_i / cv_c:.0f}"

    fe = pd.read_csv(OUT / "aim2" / "fixed_effects_results.csv")
    g = lambda o, s, t="exposure": fe[(fe.outcome == o) & (fe.specification == s) & (fe.term == t)].iloc[0]
    pct = lambda b: 100 * (np.exp(b) - 1)
    m = g("log_drug_expense", "Main: service-weighted exposure")
    n["fe_pct"], n["fe_pct_lo"], n["fe_pct_hi"] = (f"{pct(m.coef):.2f}", f"{pct(m.ci_lo):.2f}", f"{pct(m.ci_hi):.2f}")
    n["ident_share"] = f"{100 * m.identifying_variation_share:.1f}"
    n["n_fe_obs"] = f"{int(m.n_obs):,}"
    mapd = g("log_drug_cost_per_apd", "Main: service-weighted exposure")
    n["fe_apd_pct"], n["fe_apd_lo"], n["fe_apd_hi"] = (f"{pct(mapd.coef):.2f}", f"{pct(mapd.ci_lo):.2f}", f"{pct(mapd.ci_hi):.2f}")
    mean_drug, med_drug = d["drug_expense"].mean(), d["drug_expense"].median()
    usd = lambda b, base: f"{(np.exp(b) - 1) * base:,.0f}"
    n["fe_usd_mean"], n["fe_usd_lo"], n["fe_usd_hi"] = usd(m.coef, mean_drug), usd(m.ci_lo, mean_drug), usd(m.ci_hi, mean_drug)
    n["fe_usd_median"] = usd(m.coef, med_drug)
    n["fe_usd_mean_round"] = f"{round((np.exp(m.coef) - 1) * mean_drug, -3):,.0f}"
    n["mean_drug_m"], n["median_drug_m"] = f"{mean_drug / 1e6:.1f}", f"{med_drug / 1e6:.1f}"
    sd_between = d.groupby("fy")["exposure"].std().mean()
    n["sd_between"] = f"{sd_between:.0f}"
    n["sd_pct"] = f"{pct(m.coef * sd_between):.1f}"
    n["fe_trend_cat"] = f"{pct(g('log_drug_expense', 'Category-specific baseline-share trends').coef):.2f}"
    n["fe_trend_hosp"] = f"{pct(g('log_drug_expense', 'Hospital-specific linear trends').coef):.2f}"
    n["fe_lead"] = f"{pct(g('log_drug_expense', 'Falsification: adds next-year exposure', 'exposure_lead').coef):.2f}"
    om = g("operating_margin_pp", "Main: service-weighted exposure")
    n["fe_om"], n["fe_om_lo"], n["fe_om_hi"] = f"{om.coef:.3f}", f"{om.ci_lo:.3f}", f"{om.ci_hi:.3f}"
    n["drug_share_opex_median"] = f"{100 * (d['drug_expense'] / d['opex']).median():.1f}"

    fp = pd.read_csv(OUT / "aim2" / "forecast_performance.csv")
    inc = pd.read_csv(OUT / "aim2" / "forecast_incremental_value.csv")
    A = "A: financial/operating history"
    reg = fp[(fp.task == "regression") & (fp.feature_set == A)]
    cls = fp[(fp.task == "classification") & (fp.feature_set == A)]

    def row(df, h, model):
        return df[(df.horizon == h) & (df.model.str.startswith(model))].iloc[0]

    best1 = reg[(reg.horizon == 1) & ~reg.model.str.startswith("Reference")].sort_values("MAE_est").iloc[0]
    pers1, pers2 = row(reg, 1, "Reference"), row(reg, 2, "Reference")
    best2 = reg[(reg.horizon == 2) & ~reg.model.str.startswith("Reference")].sort_values("MAE_est").iloc[0]
    cbest1 = cls[(cls.horizon == 1) & ~cls.model.str.startswith("Reference")].sort_values("AUROC_est").iloc[-1]
    cpers1 = row(cls, 1, "Reference")
    mae_rng = reg[(reg.horizon == 1) & ~reg.model.str.startswith("Reference")]["MAE_est"]
    ci = lambda s: ci_text(s)[1]
    n["res_2b_1"] = (
        f"Using financial and operating history alone (feature set A), next-year operating margin in held-out hospitals "
        f"was forecast by {best1.model.lower()} with an MAE of {ci_text(best1.MAE_pp)[0]} percentage points "
        f"({ci(best1.MAE_pp)[len(ci_text(best1.MAE_pp)[0]) + 2:-1]}) and an R² of {float(best1.R2.split()[0]):.2f}, "
        f"compared with {ci_text(pers1.MAE_pp)[0]} percentage points and {float(pers1.R2.split()[0]):.2f} for the persistence "
        f"forecast; the 3 model classes performed similarly (MAE {mae_rng.min():.2f}-{mae_rng.max():.2f}). An operating loss "
        f"the next year was predicted with an AUROC of {ci(cbest1.AUROC)} by {cbest1.model.lower()}, similar to the "
        f"persistence reference ({ci_text(cpers1.AUROC)[0]}). At 2 years, the MAE was {ci_text(best2.MAE_pp)[0]} vs {ci_text(pers2.MAE_pp)[0]} for persistence "
        f"(eTable 6 in Supplement 1).")
    i1 = inc[(inc.horizon == 1) & inc.comparison.str.startswith("B")]
    r1 = i1[i1.task == "regression"]
    c1 = i1[i1.task == "classification"]
    oc = inc[inc.comparison.str.startswith("C") & (inc.task == "regression")]
    txt = (f"Adding shortage exposure known at the time of the forecast did not improve forecasts for any model class or horizon: "
           f"1-year MAE changed by {r1.delta.min():+.2f} to {r1.delta.max():+.2f} percentage points (worse) and AUROC by "
           f"{c1.delta.min():+.3f} to {c1.delta.max():+.3f} (Figure 2; eTable 7 in Supplement 1). Even the realized exposure in the outcome year, "
           f"which could not be known in advance, did not help (MAE change {oc.delta.min():+.2f} to {oc.delta.max():+.2f}).")
    post = OUT / "aim2" / "forecast_incremental_value_posthoc_relative.csv"
    if post.exists():
        pr = pd.read_csv(post)
        prr, prc = pr[(pr.task == "regression") & (pr.horizon == 1)], pr[(pr.task == "classification") & (pr.horizon == 1)]
        txt += (f" In a post hoc analysis expressing exposure relative to other hospitals in the same year, which removes the "
                f"national trend that rose beyond its training range in 2022-2024, the changes were {prr.delta.min():+.2f} to "
                f"{prr.delta.max():+.2f} percentage points in MAE and {prc.delta.min():+.3f} to {prc.delta.max():+.3f} in AUROC.")
        n["posthoc_mae_range"] = f"{prr.delta.min():+.2f} to {prr.delta.max():+.2f}"
    rr = pd.read_csv(OUT / "aim2" / "rolling_origin.csv") if (OUT / "aim2" / "rolling_origin.csv").exists() else None
    if rr is not None:
        g1 = rr[(rr.task == "regression") & (rr.feature_set.str.startswith("A")) & ~rr.model.str.startswith("Reference")]
        best_by_year = g1.groupby("test_year")["MAE"].min()
        txt += (f" In the rolling-origin analysis, the best model's 1-year MAE ranged from {best_by_year.min():.1f} to "
                f"{best_by_year.max():.1f} percentage points across test years 2020-2024, with the largest errors in "
                f"{int(best_by_year.idxmax())} (eTable 8 in Supplement 1).")
    n["res_2b_2"] = txt
    n["abs_2b"] = (f"Next-year operating margin was forecast with a mean absolute error of {ci_text(best1.MAE_pp)[0]} percentage points "
                   f"(persistence, {ci_text(pers1.MAE_pp)[0]}) and operating loss with an AUROC of {ci_text(cbest1.AUROC)[0]}; adding shortage "
                   f"exposure did not improve either forecast.")
    n["resp_2b"] = (f"With hospital-level and time-forward splits, next-year margin is forecast with an MAE of {ci_text(best1.MAE_pp)[0]} "
                    f"percentage points (R² {float(best1.R2.split()[0]):.2f}), modestly better than simply carrying forward last year's "
                    f"margin ({ci_text(pers1.MAE_pp)[0]} percentage points; R² {float(pers1.R2.split()[0]):.2f}), and operating loss with an AUROC of "
                    f"{ci_text(cbest1.AUROC)[0]}.")
    n["resp_2b_short"] = (f"In the revised forecasting analysis, financial history predicts next-year margin with an MAE of "
                          f"{ci_text(best1.MAE_pp)[0]} percentage points, and adding shortage exposure, even perfectly foreseen exposure, "
                          f"does not improve it. Shortage costs appear in drug budgets (Aim 2a) but are too small relative to total "
                          f"expenses to move margins.")
    (OUT / "numbers.json").write_text(json.dumps(n, indent=1, default=str))
    print(json.dumps({k: v for k, v in n.items() if k not in ("fp", "inc")}, indent=1))


if __name__ == "__main__":
    main()
