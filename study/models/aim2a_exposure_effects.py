"""
Aim 2a: within-hospital association between shortage exposure and hospital drug expense and margins.

Model (two-way fixed effects, OLS, SEs clustered by hospital):
    Y_h,t = beta * Exposure_h,t + X_h,t' gamma + alpha_h + tau_t + e_h,t
Hospital fixed effects (alpha_h) remove stable differences in size, service mix, purchasing contracts and
accounting practice; year fixed effects (tau_t) remove national trends (drug inflation, COVID-19, policy).
beta is identified from hospitals whose services use the drugs that went into shortage experiencing larger
increases in exposure than other hospitals in the same year (a shift-share design).

Outcomes: log drug expense per adjusted patient day (primary), drug expense per adjusted patient day ($),
log total drug expense, operating margin (percentage points).
Robustness: alternative exposure definitions, category-specific baseline-share x linear-trend controls,
a lead (t+1) exposure falsification test, excluding FY2020-2021 (COVID-19), excluding FY2016.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs" / "aim2"
CATEGORIES = ["perioperative", "critical_care", "oncology", "anti_infective", "fluids_nutrition_electrolytes",
              "other_parenteral", "non_parenteral"]
CONTROLS = ["log_beds", "occupancy", "medicare_day_share", "medicaid_day_share", "log_apd"]


def load():
    h = pd.read_csv(INTERIM / "hospital_panel.csv", dtype={"ccn": str})
    e = pd.read_csv(INTERIM / "hospital_exposure.csv", dtype={"ccn": str})
    d = h.merge(e.drop(columns=["fyb", "fye"]), on=["ccn", "fy"])
    d = d[(d["has_baseline"] == 1) & (d["fy"] > d["baseline_fy"]) & d["fy"].between(2016, 2024)].copy()
    d["log_beds"] = np.log(d["beds"])
    d["log_apd"] = np.log(d["adj_patient_days"])
    d["log_drug_cost_per_apd"] = np.log(d["drug_cost_per_apd"].where(d["drug_cost_per_apd"] > 0))
    d["log_drug_expense"] = np.log(d["drug_expense"].where(d["drug_expense"] > 0))
    d["operating_margin_pp"] = 100 * d["operating_margin"]
    d["fye_month"] = pd.to_datetime(d["fye"]).dt.month
    # Winsorise outcomes at the 1st/99th percentiles to limit the influence of cost-report entry errors
    for c in ["log_drug_cost_per_apd", "drug_cost_per_apd", "log_drug_expense", "operating_margin_pp"]:
        lo, hi = d[c].quantile([0.01, 0.99])
        d[c] = d[c].clip(lo, hi)
    for c in ["occupancy", "medicare_day_share", "medicaid_day_share"]:
        d[c] = d[c].clip(0, 1.5)
    # Lead exposure for the falsification test
    lead = e[["ccn", "fy", "exposure"]].rename(columns={"exposure": "exposure_lead"})
    lead["fy"] -= 1
    d = d.merge(lead, on=["ccn", "fy"], how="left")
    return d


def _detrend_within(frame, groups, t):
    """Residualise every column on a hospital-specific intercept and linear time trend."""
    out = frame.copy()
    tc = t - t.groupby(groups).transform("mean")
    denom = (tc ** 2).groupby(groups).transform("sum").replace(0, np.nan)
    for c in frame.columns:
        v = frame[c] - frame[c].groupby(groups).transform("mean")
        slope = (v * tc).groupby(groups).transform("sum") / denom
        out[c] = v - slope.fillna(0) * tc
    return out


def _residual_share(raw, demeaned, period_dummies):
    """Share of the exposure variable's variance left after hospital and period fixed effects."""
    beta, *_ = np.linalg.lstsq(period_dummies.values, demeaned.values, rcond=None)
    resid = demeaned.values - period_dummies.values @ beta
    return float(resid.var() / raw.var())


def twfe(d, y, x, controls=CONTROLS, extra=None, hospital_trends=False):
    cols = [y] + x + controls + (extra or [])
    s = d.dropna(subset=cols).copy()
    s = s[s.groupby("ccn")["ccn"].transform("size") >= (3 if hospital_trends else 2)]
    # Period fixed effects: fiscal-year-end year x month, so hospitals are compared only with hospitals whose
    # fiscal years cover exactly the same calendar months (absorbs national price trends and shortage counts)
    period = s["fy"].astype(str) + "-" + s["fye_month"].astype(str)
    periods = pd.get_dummies(period, prefix="p", drop_first=True, dtype=float)
    X = pd.concat([s[x + controls + (extra or [])], periods], axis=1)
    Y = s[y]
    if hospital_trends:
        Xd = _detrend_within(X, s["ccn"], s["fy"].astype(float))
        Yd = _detrend_within(Y.to_frame(), s["ccn"], s["fy"].astype(float))[y]
    else:
        # Within transformation by hospital (exact two-way FE because period dummies are demeaned too)
        Xd = X - X.groupby(s["ccn"]).transform("mean")
        Yd = Y - Y.groupby(s["ccn"]).transform("mean")
    keep = Xd.columns[Xd.abs().sum() > 1e-9]
    Xd = Xd[keep]
    fit = sm.OLS(Yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s["ccn"])[0]})
    n_h = s["ccn"].nunique()
    k_fe = n_h * (2 if hospital_trends else 1)
    # Degrees-of-freedom correction for absorbed hospital effects
    dof = (len(s) - 1) / (len(s) - Xd.shape[1] - k_fe)
    se = fit.bse * np.sqrt(dof)
    res = []
    for v in x:
        if v not in fit.params:
            res.append({"outcome": y, "term": v, "coef": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "p": np.nan, "n_obs": len(s), "n_hospitals": n_h, "mean_y": Y.mean(),
                        "sd_x_within": 0.0, "note": "no variation within hospital and fiscal period"})
            continue
        b, sev = fit.params[v], se[v]
        res.append({"outcome": y, "term": v, "coef": b, "se": sev, "ci_lo": b - 1.96 * sev, "ci_hi": b + 1.96 * sev,
                    "p": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(b / sev))), "n_obs": len(s),
                    "n_hospitals": n_h, "mean_y": Y.mean(), "sd_x_within": Xd[v].std(),
                    "identifying_variation_share": _residual_share(s[v], Xd[v], Xd[[c for c in Xd if c.startswith("p_")]])})
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f"analytic sample: {len(d):,} hospital-years, {d['ccn'].nunique():,} hospitals, FY{d.fy.min()}-{d.fy.max()}")
    outcomes = ["log_drug_cost_per_apd", "drug_cost_per_apd", "log_drug_expense", "operating_margin_pp"]

    rows = []
    for y in outcomes:
        for spec, x, extra, sub, trends in [
            ("Main: service-weighted exposure", ["exposure"], None, d, False),
            ("Unweighted shortage-years", ["exposure_unweighted"], None, d, False),
            ("Original measure: count of active shortages", ["shortage_count_any"], None, d, False),
            ("Parenteral shortages only", ["exposure_parenteral"], None, d, False),
            ("Limited-supply shortages only", ["exposure_limited_supply"], None, d, False),
            ("Exposure x baseline drug intensity", ["exposure_drug_intensity"], None, d, False),
            ("Category-specific baseline-share trends", ["exposure"], [f"trend_{c}" for c in CATEGORIES], d, False),
            ("Excluding FY2020-2021", ["exposure"], None, d[~d["fy"].isin([2020, 2021])], False),
            ("Excluding FY2016", ["exposure"], None, d[d["fy"] >= 2017], False),
            ("Hospital-specific linear trends", ["exposure"], None, d, True),
            ("Falsification: adds next-year exposure", ["exposure", "exposure_lead"], None, d, False),
        ]:
            if extra:
                sub = sub.copy()
                t = sub["fy"] - 2016
                for c in CATEGORIES:
                    sub[f"trend_{c}"] = sub[c] * t
            for r in twfe(sub, y, x, extra=extra, hospital_trends=trends):
                r["specification"] = spec
                rows.append(r)
    res = pd.DataFrame(rows)

    # Dollar translation of the main estimate: change in annual drug expense per additional exposure unit
    main_spec = res[(res["specification"] == "Main: service-weighted exposure") & (res["term"] == "exposure")]
    log_total = main_spec[main_spec["outcome"] == "log_drug_expense"].iloc[0]
    mean_drug = d["drug_expense"].mean()
    median_drug = d["drug_expense"].median()
    pct = np.exp(log_total["coef"]) - 1
    dollars = {
        "pct_per_unit": pct, "pct_ci": (np.exp(log_total["ci_lo"]) - 1, np.exp(log_total["ci_hi"]) - 1),
        "usd_per_unit_at_mean": pct * mean_drug,
        "usd_ci_at_mean": ((np.exp(log_total["ci_lo"]) - 1) * mean_drug, (np.exp(log_total["ci_hi"]) - 1) * mean_drug),
        "usd_per_unit_at_median": pct * median_drug,
        "mean_drug_expense": mean_drug, "median_drug_expense": median_drug,
        "exposure_sd_between": d.groupby("fy")["exposure"].std().mean(),
        "exposure_sd_within": (d["exposure"] - d.groupby("ccn")["exposure"].transform("mean")).std(),
        "exposure_iqr": d["exposure"].quantile(0.75) - d["exposure"].quantile(0.25),
    }
    res.to_csv(OUT / "fixed_effects_results.csv", index=False)
    pd.Series({k: (v if not isinstance(v, tuple) else f"{v[0]:.6g};{v[1]:.6g}") for k, v in dollars.items()}) \
        .to_csv(OUT / "dollar_translation.csv", header=["value"])

    # Descriptive table inputs
    d.to_csv(OUT / "analytic_sample_aim2a.csv.gz", index=False, compression="gzip")
    show = res[res["term"] != "exposure_lead"][["outcome", "specification", "coef", "ci_lo", "ci_hi", "p", "n_obs"]]
    print(show.to_string())
    print(res[res["term"] == "exposure_lead"][["outcome", "coef", "ci_lo", "ci_hi", "p"]].to_string())
    print(dollars)


if __name__ == "__main__":
    main()
