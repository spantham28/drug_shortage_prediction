"""
Build the hospital-fiscal-year panel from HCRIS cost reports (Form CMS-2552-10).

Population: short-term acute care hospitals (CCN suffix 0001-0879) filing a full-year cost report
(360-371 days).  When a hospital filed more than one report for the same fiscal year, the most recently
processed report is kept.  Fiscal year (fy) = calendar year of the fiscal-year end date.

Worksheet references (line / column):
  G-3  L1 total patient revenue, L2 contractual allowances, L3 net patient revenue, L4 operating expenses,
       L5 net income from patient services, L25 other income, L29 net income
  G-2  L28 total patient revenue: C1 inpatient, C2 outpatient, C3 total
  G    L1 cash, L11 current assets, L36 total assets, L45 current liabilities, L51 total liabilities
  S-3  L14 totals: C2 beds, C3 bed-days available, C6 Medicare days, C7 Medicaid days, C8 total days,
       C9 intern/resident FTEs, C10 employee FTEs, C15 discharges; L8-L12 C8 intensive-care days
  A    C7 (net expenses for cost allocation): L15 pharmacy, L73.xx drugs charged to patients,
       L50 operating room, L51 recovery room, L53 anesthesiology, L55 radiology-therapeutic,
       L91 emergency, L118 subtotal, L200 total
  C    Pt I L73 drugs charged to patients: C6 inpatient charges, C7 outpatient charges, C8 total
  S-2  L21 type of control, L26 urban (1) / rural (2), L56 teaching (Y/N), L140 chain/system member (Y/N)

Output: data/interim/hospital_panel.csv
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
ONLY_YEARS = set(sys.argv[1:])  # optional: restrict to specific HCRIS FY files (debugging)
RPT_COLS = ["rpt_rec_num", "ctrl_type", "ccn", "npi", "status", "fyb", "fye", "proc_dt", "initl", "last", "trns",
            "fi", "adr", "fi_creat", "util", "npr_dt", "spec", "fi_rcpt"]

NUMERIC_CELLS = {
    # name: (worksheet, line, col)
    "tpr": ("G300000", "00100", "00100"),
    "contractual": ("G300000", "00200", "00100"),
    "npr": ("G300000", "00300", "00100"),
    "opex": ("G300000", "00400", "00100"),
    "net_patient_income": ("G300000", "00500", "00100"),
    "other_income": ("G300000", "02500", "00100"),
    "net_income": ("G300000", "02900", "00100"),
    "rev_inpatient": ("G200000", "02800", "00100"),
    "rev_outpatient": ("G200000", "02800", "00200"),
    "rev_total": ("G200000", "02800", "00300"),
    "cash": ("G000000", "00100", "00100"),
    "current_assets": ("G000000", "01100", "00100"),
    "total_assets": ("G000000", "03600", "00100"),
    "current_liabilities": ("G000000", "04500", "00100"),
    "total_liabilities": ("G000000", "05100", "00100"),
    "beds": ("S300001", "01400", "00200"),
    "bed_days": ("S300001", "01400", "00300"),
    "days_medicare": ("S300001", "01400", "00600"),
    "days_medicaid": ("S300001", "01400", "00700"),
    "days_total": ("S300001", "01400", "00800"),
    "ir_fte": ("S300001", "01400", "00900"),
    "fte": ("S300001", "01400", "01000"),
    "discharges": ("S300001", "01400", "01500"),
    "cost_pharmacy": ("A000000", "01500", "00700"),
    "cost_or": ("A000000", "05000", "00700"),
    "cost_recovery": ("A000000", "05100", "00700"),
    "cost_anesthesia": ("A000000", "05300", "00700"),
    "cost_rad_onc": ("A000000", "05500", "00700"),
    "cost_ed": ("A000000", "09100", "00700"),
    "cost_subtotal": ("A000000", "11800", "00700"),
    "cost_total": ("A000000", "20000", "00700"),
    "drug_chg_ip": ("C000001", "07300", "00600"),
    "drug_chg_op": ("C000001", "07300", "00700"),
    "drug_chg_total": ("C000001", "07300", "00800"),
    "urban_rural": ("S200001", "02600", "00100"),
}
ALPHA_CELLS = {
    "state": ("S200001", "00200", "00200"),
    "control": ("S200001", "02100", "00100"),
    "teaching_flag": ("S200001", "05600", "00100"),
    "chain_flag": ("S200001", "14000", "00100"),
}


def load_year(path_cells, path_rpt):
    rpt = pd.read_csv(path_rpt, header=None, names=RPT_COLS, dtype=str)
    cells = pd.read_csv(path_cells, dtype=str)
    num = cells[cells["kind"] == "n"].copy()
    num["value"] = pd.to_numeric(num["value"], errors="coerce")

    out = rpt[["rpt_rec_num", "ccn", "status", "fyb", "fye", "proc_dt"]].set_index("rpt_rec_num")
    num["key"] = num["wksht"] + "|" + num["line"] + "|" + num["col"]
    names = {"|".join(k): n for n, k in NUMERIC_CELLS.items()}
    sel = num[num["key"].isin(names)].assign(name=lambda x: x["key"].map(names))
    wide = sel.pivot_table(index="rpt_rec_num", columns="name", values="value", aggfunc="sum")
    for name in NUMERIC_CELLS:
        out[name] = wide[name] if name in wide else np.nan
    # Drugs charged to patients: line 73 plus any subscripted lines 73.01-73.99
    drug = num[(num["wksht"] == "A000000") & num["line"].str.startswith("073") & (num["col"] == "00700")]
    out["cost_drugs_charged"] = drug.groupby("rpt_rec_num")["value"].sum()
    icu = num[(num["wksht"] == "S300001") & num["line"].isin(["00800", "00900", "01000", "01100", "01200"])
              & (num["col"] == "00800")]
    out["days_icu"] = icu.groupby("rpt_rec_num")["value"].sum()

    alpha = cells[cells["kind"] == "a"].copy()
    alpha["key"] = alpha["wksht"] + "|" + alpha["line"] + "|" + alpha["col"]
    for name, key in ALPHA_CELLS.items():
        out[name] = alpha[alpha["key"] == "|".join(key)].drop_duplicates("rpt_rec_num").set_index("rpt_rec_num")["value"]
    return out.reset_index()


def main():
    frames = []
    for f in sorted(glob.glob(str(INTERIM / "hcris_cells_FY*.csv.gz"))):
        year = f.split("FY")[-1][:4]
        if ONLY_YEARS and year not in ONLY_YEARS:
            continue
        frames.append(load_year(f, INTERIM / f"hcris_rpt_FY{year}.csv"))
        print("loaded", year)
    h = pd.concat(frames, ignore_index=True)

    h["fyb"] = pd.to_datetime(h["fyb"], format="%m/%d/%Y")
    h["fye"] = pd.to_datetime(h["fye"], format="%m/%d/%Y")
    h["proc_dt"] = pd.to_datetime(h["proc_dt"], format="%m/%d/%Y", errors="coerce")
    h["fy_days"] = (h["fye"] - h["fyb"]).dt.days + 1
    h["ccn"] = h["ccn"].str.zfill(6)
    suffix = pd.to_numeric(h["ccn"].str[2:], errors="coerce")
    n0 = len(h)
    h = h[suffix.between(1, 879)]
    n1 = len(h)
    h = h[h["fy_days"].between(360, 371)]
    n2 = len(h)
    h["fy"] = h["fye"].dt.year
    h = h.sort_values("proc_dt").drop_duplicates(["ccn", "fy"], keep="last")
    n3 = len(h)

    # Derived measures
    zero_fill = ["cost_pharmacy", "cost_drugs_charged", "cost_or", "cost_recovery", "cost_anesthesia",
                 "cost_rad_onc", "cost_ed", "days_icu", "ir_fte", "drug_chg_ip", "drug_chg_op", "drug_chg_total"]
    h[zero_fill] = h[zero_fill].fillna(0)
    h["drug_expense"] = h["cost_pharmacy"] + h["cost_drugs_charged"]
    ip_share = (h["rev_inpatient"] / h["rev_total"]).where(h["rev_total"] > 0)
    h["inpatient_rev_share"] = ip_share
    h["outpatient_rev_share"] = 1 - ip_share
    h["adj_patient_days"] = h["days_total"] / ip_share.where(ip_share > 0.05)
    h["drug_cost_per_apd"] = h["drug_expense"] / h["adj_patient_days"]
    h["operating_margin"] = h["net_patient_income"] / h["npr"]
    h["total_margin"] = h["net_income"] / (h["npr"] + h["other_income"].fillna(0))
    h["days_cash"] = h["cash"] / (h["opex"] / 365)
    h["current_ratio"] = h["current_assets"] / h["current_liabilities"]
    h["debt_ratio"] = h["total_liabilities"] / h["total_assets"]
    h["occupancy"] = h["days_total"] / h["bed_days"]
    h["medicare_day_share"] = h["days_medicare"] / h["days_total"]
    h["medicaid_day_share"] = h["days_medicaid"] / h["days_total"]
    h["fte_per_bed"] = h["fte"] / h["beds"]
    h["opex_per_apd"] = h["opex"] / h["adj_patient_days"]
    h["npr_per_apd"] = h["npr"] / h["adj_patient_days"]
    h["drug_share_of_cost"] = h["drug_expense"] / h["cost_total"]
    h["periop_share"] = (h["cost_or"] + h["cost_recovery"] + h["cost_anesthesia"]) / h["cost_subtotal"]
    h["rad_onc_share"] = h["cost_rad_onc"] / h["cost_subtotal"]
    h["ed_share"] = h["cost_ed"] / h["cost_subtotal"]
    h["icu_day_share"] = h["days_icu"] / h["days_total"]
    h["op_drug_share"] = (h["drug_chg_op"] / h["drug_chg_total"]).where(h["drug_chg_total"] > 0)
    h["teaching"] = ((h["teaching_flag"] == "Y") | (h["ir_fte"] > 0)).astype(int)
    h["system_member"] = (h["chain_flag"] == "Y").astype(int)
    h["rural"] = (pd.to_numeric(h["urban_rural"], errors="coerce") == 2).astype(int)
    ctrl = pd.to_numeric(h["control"], errors="coerce")
    h["ownership"] = np.select([ctrl.between(1, 2), ctrl.between(3, 6), ctrl.between(7, 13)],
                               ["nonprofit", "for_profit", "government"], "unknown")
    # 1-2 voluntary nonprofit; 3-6 proprietary; 7-13 governmental (CMS-2552-10 S-2 line 21 codes)

    # Data-quality exclusions defined before modelling
    valid = (h["npr"] > 0) & (h["opex"] > 0) & (h["beds"] >= 1) & (h["days_total"] > 0) & h["adj_patient_days"].notna()
    n4 = int(valid.sum())
    h = h[valid].copy()

    h.to_csv(INTERIM / "hospital_panel.csv", index=False)
    flow = pd.DataFrame({"step": ["all HCRIS reports FY2014-2025 files", "short-term acute care CCN",
                                  "full-year (360-371 day) report", "one report per hospital-year",
                                  "positive revenue, expenses, beds, patient days"],
                         "n": [n0, n1, n2, n3, n4]})
    flow.to_csv(INTERIM / "hospital_panel_flow.csv", index=False)
    print(flow.to_string())
    print(h.groupby("fy").agg(n=("ccn", "size"), om=("operating_margin", "median"),
                              drug_apd=("drug_cost_per_apd", "median")))


if __name__ == "__main__":
    main()
