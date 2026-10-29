"""
Build the drug-market x month panel used to forecast new generic drug shortages.

Market  = one NADAC product description (ingredient + strength + dosage form), pooling every
          manufacturer's NDC for that product.
Month t = calendar month.  Every feature uses NADAC data observed in months <= t only.
Label   = 1 if a new ASHP shortage listing that includes one of the market's NDCs begins in
          months t+1 ... t+6 (6-month forecast horizon).
Risk set: market-months that are priced in NADAC and not already in an active shortage.

Inputs
  data/interim/nadac_ndc_month_YYYY.csv.gz   (02_download_nadac.py)
  data/interim/shortage_events.csv, shortage_ndcs.csv (03_build_shortage_events.py)
Output
  data/interim/drug_month_panel.csv.gz
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
HORIZON = 6
FIRST_ONSET = pd.Period("2015-10", "M")   # ASHP listings created before this are the initial back-load
LAST_LABEL_MONTH = pd.Period("2025-07", "M")  # ASHP extract taken August 2025
SCORING_MONTH = pd.Period("2025-08", "M")  # latest month with known shortage status (web demo snapshot)


def load_nadac():
    n = pd.concat(
        [pd.read_csv(f, dtype={"ndc": str}) for f in sorted(glob.glob(str(INTERIM / "nadac_ndc_month_*.csv.gz")))],
        ignore_index=True,
    )
    n["market"] = n["ndc_description"].str.upper().str.split().str.join(" ")
    n["month"] = pd.PeriodIndex(n["month"], freq="M")
    n["labeler"] = n["ndc"].str[:5]
    # Generic markets only: the market's NDCs are predominantly classified as generic for rate setting
    gshare = n.groupby("market")["classification"].apply(lambda s: (s == "G").mean())
    n = n[n["market"].isin(gshare[gshare >= 0.5].index)]
    return n


def market_month(n):
    g = n.groupby(["market", "month"])
    mm = g.agg(
        price=("nadac", "median"),
        price_min=("nadac", "min"),
        price_max=("nadac", "max"),
        n_ndcs=("ndc", "nunique"),
        n_labelers=("labeler", "nunique"),
        pricing_unit=("pricing_unit", "first"),
    ).reset_index()
    return mm


def shortage_months(n):
    s = pd.read_csv(INTERIM / "shortage_ndcs.csv", dtype=str)
    ev = pd.read_csv(INTERIM / "shortage_events.csv", parse_dates=["start", "end"])
    s["shortage_id"] = s["shortage_id"].astype(int)
    ndc_market = n[["ndc", "market"]].drop_duplicates()
    ndc_market["product9"] = ndc_market["ndc"].str[:9]
    exact = s.merge(ndc_market[["ndc", "market"]], left_on="ndc11", right_on="ndc")
    by_product = s.merge(ndc_market[["product9", "market"]].drop_duplicates(), on="product9")
    links = pd.concat([exact[["shortage_id", "market"]], by_product[["shortage_id", "market"]]]).drop_duplicates()
    links = links.merge(ev[["shortage_id", "start", "end"]], on="shortage_id")
    links["start_m"] = links["start"].dt.to_period("M")
    links["end_m"] = links["end"].dt.to_period("M")
    return links


def build():
    n = load_nadac()
    mm = market_month(n)
    links = shortage_months(n)

    # Full monthly grid per market between its first and last NADAC month
    span = mm.groupby("market")["month"].agg(["min", "max"])
    grid = pd.concat(
        [pd.DataFrame({"market": m, "month": pd.period_range(r["min"], r["max"], freq="M")}) for m, r in span.iterrows()],
        ignore_index=True,
    )
    p = grid.merge(mm, on=["market", "month"], how="left").sort_values(["market", "month"]).reset_index(drop=True)
    p["observed"] = p["price"].notna().astype(int)
    g = p.groupby("market", sort=False)

    # Price features (log scale, computed only from current and past months)
    p["lp"] = np.log(p["price"])
    p["lp_ffill"] = g["lp"].ffill()
    for k in (1, 3, 6, 12):
        p[f"dlp_{k}"] = p["lp_ffill"] - g["lp_ffill"].shift(k)
    p["dlp_month"] = g["lp_ffill"].diff()
    g = p.groupby("market", sort=False)
    p["vol_12"] = g["dlp_month"].transform(lambda s: s.rolling(12, min_periods=6).std())
    p["max_jump_12"] = g["dlp_month"].transform(lambda s: s.rolling(12, min_periods=3).max())
    p["dispersion"] = np.log(p["price_max"] / p["price_min"])
    p["n_labelers_ffill"] = g["n_labelers"].ffill()
    p["d_labelers_12"] = p["n_labelers_ffill"] - g["n_labelers_ffill"].shift(12)
    p["d_ndcs_12"] = g["n_ndcs"].ffill() - g["n_ndcs"].ffill().shift(12)
    p["months_in_nadac"] = g.cumcount()

    desc = p["market"]
    p["form_injectable"] = desc.str.contains(r"VIAL|AMPUL|SYRINGE|SYRG|INJ|CARPUJECT|\bIV\b", regex=True).astype(int)
    p["form_oral_solid"] = desc.str.contains(r"TAB|CAP", regex=True).astype(int)
    p["form_liquid"] = desc.str.contains(r"SOLN|SOLUTION|SUSP|LIQUID|SYRUP|ELIXIR|ORAL SOL", regex=True).astype(int)
    p["form_topical"] = desc.str.contains(r"CREAM|OINT|GEL|LOTION|PATCH|DROP|SPRAY|INHAL|NEB", regex=True).astype(int)
    p["unit_ml"] = (p["pricing_unit"] == "ML").astype(int)
    p["unit_gm"] = (p["pricing_unit"] == "GM").astype(int)
    p["ingredient"] = desc.str.split().str[0]

    # Shortage status, onset label, and shortage history
    p["active"] = 0
    p["onset_now"] = 0
    onsets = links[links["start_m"] >= FIRST_ONSET]
    idx = p.set_index(["market", "month"]).index
    act = set()
    for r in links.itertuples():
        for m in pd.period_range(r.start_m, r.end_m, freq="M"):
            act.add((r.market, m))
    p["active"] = np.fromiter(((k in act) for k in idx), dtype=bool, count=len(p)).astype(int)
    onset_set = set(zip(onsets["market"], onsets["start_m"]))
    p["onset_now"] = np.fromiter(((k in onset_set) for k in idx), dtype=bool, count=len(p)).astype(int)
    g = p.groupby("market", sort=False)
    future = sum(g["onset_now"].shift(-k).fillna(0) for k in range(1, HORIZON + 1))
    p["y"] = (future > 0).astype(int)
    p["prior_shortages"] = g["onset_now"].cumsum() - p["onset_now"]
    p["ever_active_before"] = g["active"].transform(lambda s: s.shift(1).fillna(0).cummax())
    # Months since the market last left shortage (capped), 0 for never
    last_end = p["month"].where((g["active"].shift(1).fillna(0) == 1) & (p["active"] == 0))
    p["last_end"] = last_end
    p["last_end"] = p.groupby("market", sort=False)["last_end"].ffill()
    p["months_since_shortage"] = [
        (m - le).n if isinstance(le, pd.Period) else 999 for m, le in zip(p["month"], p["last_end"])
    ]
    p["months_since_shortage"] = np.minimum(p["months_since_shortage"], 120)

    # Spillover signals that are public at month t (the ASHP list is published in real time)
    #   ingr_active_other: other strengths / forms of the same ingredient currently in shortage
    #   labeler_short_share: share of the market's current suppliers with a shortage in another market
    #   national_onsets_3m: new shortage listings nationally in months t-2..t
    ingr = p.groupby(["ingredient", "month"])["active"].transform("sum")
    p["ingr_active_other"] = ingr - p["active"]
    lab = n[["market", "month", "labeler"]].drop_duplicates()
    lab_short = lab.merge(p.loc[p["active"] == 1, ["market", "month"]], on=["market", "month"]) \
        .groupby(["labeler", "month"])["market"].nunique().rename("lab_n_short").reset_index()
    lab = lab.merge(lab_short, on=["labeler", "month"], how="left").fillna({"lab_n_short": 0})
    own = p.loc[p["active"] == 1, ["market", "month"]].assign(own=1)
    lab = lab.merge(own, on=["market", "month"], how="left").fillna({"own": 0})
    lab["elsewhere"] = (lab["lab_n_short"] - lab["own"]) > 0
    spill = lab.groupby(["market", "month"]).agg(labeler_short_share=("elsewhere", "mean"),
                                                   labeler_short_any=("elsewhere", "max")).reset_index()
    p = p.merge(spill, on=["market", "month"], how="left")
    p["labeler_short_any"] = p["labeler_short_any"].fillna(False).astype(int)
    nat = p.groupby("month")["onset_now"].sum().sort_index()
    p["national_onsets_3m"] = p["month"].map(nat.rolling(3, min_periods=1).sum())

    risk = (p["observed"] == 1) & (p["active"] == 0) & (p["month"] >= pd.Period("2015-01", "M")) \
        & (p["month"] + HORIZON <= LAST_LABEL_MONTH)
    out = p[risk].copy()
    out["month"] = out["month"].astype(str)
    keep = ["market", "ingredient", "month", "y", "lp", "dlp_1", "dlp_3", "dlp_6", "dlp_12", "vol_12",
            "max_jump_12", "dispersion", "n_ndcs", "n_labelers", "d_labelers_12", "d_ndcs_12", "months_in_nadac",
            "prior_shortages", "ever_active_before", "months_since_shortage", "form_injectable", "form_oral_solid",
            "form_liquid", "form_topical", "unit_ml", "unit_gm", "ingr_active_other", "labeler_short_share",
            "labeler_short_any", "national_onsets_3m"]
    out[keep].to_csv(INTERIM / "drug_month_panel.csv.gz", index=False, compression="gzip")

    # Scoring snapshot for the web demo: the last month with a known shortage status (ASHP extract month).
    # Products already in shortage that month are listed separately rather than scored.
    snap = p[(p["observed"] == 1) & (p["month"] == SCORING_MONTH)].copy()
    snap["month"] = snap["month"].astype(str)
    snap[[c for c in keep if c != "y"] + ["active", "pricing_unit", "price"]] \
        .to_csv(INTERIM / "drug_scoring_snapshot.csv.gz", index=False, compression="gzip")
    print(f"markets={out['market'].nunique():,} rows={len(out):,} positives={out['y'].sum():,} "
          f"onset events in markets={p['onset_now'].sum():,}")
    print(out.groupby(out["month"].str[:4])["y"].agg(["size", "sum", "mean"]))


if __name__ == "__main__":
    build()
