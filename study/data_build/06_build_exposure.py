"""
Construct hospital-specific drug shortage exposure for every hospital fiscal year.

Step 1. Duration within the hospital's own fiscal year.
        For each shortage s and hospital-year (h, t), d_s,h,t = days s was active during h's fiscal year / FY days.
        D_c,h,t = sum over shortages in therapeutic category c of d_s,h,t ("shortage-years" in category c).

Step 2. Service-line relevance.
        Each category is linked to the hospital services that use those drugs, measured from the hospital's
        BASELINE cost report (its first full-year report, FY2014 or FY2015, held fixed thereafter so that
        exposure cannot respond to the outcomes being studied):
          perioperative                 -> OR + recovery + anesthesiology share of direct costs
          critical_care                 -> mean of ICU share of inpatient days and ED share of direct costs
          oncology                      -> mean of radiation-oncology cost share and outpatient share of drug charges
          anti_infective, fluids_nutrition_electrolytes, other_parenteral -> inpatient share of patient revenue
          non_parenteral (oral, topical, ophthalmic, inhaled) -> outpatient share of patient revenue
        Each measure is divided by its mean across hospitals, so a hospital with an average service mix has
        weight 1 in every category.

Step 3. Exposure index  E_h,t = sum_c w_h,c * D_c,h,t.
        One unit = one shortage active for a full fiscal year in a service the hospital uses at the national
        average intensity.

Variants written for sensitivity analyses
  exposure_unweighted     sum_c D_c,h,t (duration-weighted count, no service-mix weights)
  exposure_parenteral     service-weighted, parenteral (hospital-administered) products only
  exposure_limited_supply service-weighted, shortages with <=2 listed suppliers (few substitutes)
  exposure_drug_intensity service-weighted index x hospital's baseline drug share of total cost / mean
  shortage_count_any      number of shortages active at any point in the fiscal year (the original measure)
  carry_in                service-weighted shortages still active on the last day of the fiscal year
                          (known at the time a next-year forecast is made)

Output: data/interim/hospital_exposure.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
CATEGORIES = ["perioperative", "critical_care", "oncology", "anti_infective", "fluids_nutrition_electrolytes",
              "other_parenteral", "non_parenteral"]


def baseline_weights(h):
    base = h.sort_values("fy").groupby("ccn").head(1).set_index("ccn")
    base = base[base["fy"] <= 2015]

    def norm(s):
        s = s.clip(lower=0).fillna(0)
        return s / s.mean()

    w = pd.DataFrame(index=base.index)
    w["perioperative"] = norm(base["periop_share"])
    w["critical_care"] = (norm(base["icu_day_share"]) + norm(base["ed_share"])) / 2
    w["oncology"] = (norm(base["rad_onc_share"]) + norm(base["op_drug_share"])) / 2
    inpatient = norm(base["inpatient_rev_share"])
    for c in ["anti_infective", "fluids_nutrition_electrolytes", "other_parenteral"]:
        w[c] = inpatient
    w["non_parenteral"] = norm(base["outpatient_rev_share"])
    w["drug_intensity"] = norm(base["drug_share_of_cost"])
    w["baseline_fy"] = base["fy"]
    return w


def overlap_matrix(fyb, fye, start, end):
    """Fraction of each fiscal year (rows) during which each shortage (columns) was active."""
    fyb = fyb.values.astype("datetime64[D]")[:, None]
    fye = fye.values.astype("datetime64[D]")[:, None]
    s = start.values.astype("datetime64[D]")[None, :]
    e = end.values.astype("datetime64[D]")[None, :]
    lo = np.maximum(fyb, s)
    hi = np.minimum(fye, e)
    days = (hi - lo).astype(int) + 1
    fy_days = (fye - fyb).astype(int) + 1
    return np.clip(days, 0, None) / fy_days


def main():
    h = pd.read_csv(INTERIM / "hospital_panel.csv", dtype={"ccn": str}, parse_dates=["fyb", "fye"])
    ev = pd.read_csv(INTERIM / "shortage_events.csv", parse_dates=["start", "end"])
    w = baseline_weights(h)

    frac = overlap_matrix(h["fyb"], h["fye"], ev["start"], ev["end"])
    active_any = (frac > 0)
    active_end = ((ev["start"].values[None, :] <= h["fye"].values[:, None]) &
                  (ev["end"].values[None, :] >= h["fye"].values[:, None]))

    out = h[["ccn", "fy", "fyb", "fye"]].copy()
    cat = ev["category"].values
    parenteral = ev["parenteral"].values == 1
    limited = ev["limited_supply"].values == 1
    for c in CATEGORIES:
        m = cat == c
        out[f"D_{c}"] = frac[:, m].sum(axis=1)
        out[f"D_{c}_parenteral"] = frac[:, m & parenteral].sum(axis=1)
        out[f"D_{c}_limited"] = frac[:, m & limited].sum(axis=1)
        out[f"carry_{c}"] = active_end[:, m].sum(axis=1)
    out["shortage_count_any"] = active_any.sum(axis=1)
    out["new_shortages_in_fy"] = ((ev["start"].values[None, :] >= h["fyb"].values[:, None]) &
                                  (ev["start"].values[None, :] <= h["fye"].values[:, None])).sum(axis=1)

    out = out.merge(w, left_on="ccn", right_index=True, how="left")
    out["exposure_unweighted"] = sum(out[f"D_{c}"] for c in CATEGORIES)
    out["exposure"] = sum(out[c] * out[f"D_{c}"] for c in CATEGORIES)
    out["exposure_parenteral"] = sum(out[c] * out[f"D_{c}_parenteral"] for c in CATEGORIES)
    out["exposure_limited_supply"] = sum(out[c] * out[f"D_{c}_limited"] for c in CATEGORIES)
    out["exposure_drug_intensity"] = out["exposure"] * out["drug_intensity"]
    out["carry_in"] = sum(out[c] * out[f"carry_{c}"] for c in CATEGORIES)
    out["has_baseline"] = out["baseline_fy"].notna().astype(int)
    out.to_csv(INTERIM / "hospital_exposure.csv", index=False)

    s = out[(out["has_baseline"] == 1) & (out["fy"] >= 2016)]
    print(s.groupby("fy")[["shortage_count_any", "exposure_unweighted", "exposure"]].describe().T.round(1).to_string())
    print("within-year CV of exposure:", (s.groupby("fy")["exposure"].std() / s.groupby("fy")["exposure"].mean()).round(3).to_dict())
    print("within-year CV of shortage count:",
          (s.groupby("fy")["shortage_count_any"].std() / s.groupby("fy")["shortage_count_any"].mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
