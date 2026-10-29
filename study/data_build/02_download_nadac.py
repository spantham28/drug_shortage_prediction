"""
Download CMS NADAC (National Average Drug Acquisition Cost) weekly files and
collapse them to one row per NDC per calendar month.

NADAC is a weekly survey-based acquisition price for retail community pharmacies.
For each NDC-month we keep the median per-unit NADAC, the generic/brand
classification and the pricing unit.  Output: data/interim/nadac_ndc_month_YYYY.csv.gz

Source: https://data.medicaid.gov (NADAC yearly datasets)
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

URLS = {
    2014: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.ba0c3734-8012-549a-8f50-2ff389d0e0ef.csv",
    2015: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.4d7af295-2132-55a8-b40c-d6630061f3e8.csv",
    2016: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.7656fc17-f1b4-566b-9a2d-c4a4f2ac7ae1.csv",
    2017: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.1c5d0fc9-693a-534a-8240-4627d9362b0d.csv",
    2018: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.8de1b213-73c5-552b-b84e-ac795f34d056.csv",
    2019: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.76a1984a-6d69-5e4d-86c8-65eb31f0506d.csv",
    2020: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost.a4y5-998d.c933dc16-7de9-52b6-8971-4b75992673e0.csv",
    2021: "https://download.medicaid.gov/data/national-average-drug-acquisition-cost-12-29-2021.csv",
    2022: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-2022.csv",
    2023: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-12-27-2023.csv",
    2024: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-12-25-2024.csv",
    2025: "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-12-31-2025.csv",
}


def norm(c):
    return c.strip().lower().replace(" ", "_").replace("/", "_")


def process_year(year):
    out = INTERIM / f"nadac_ndc_month_{year}.csv.gz"
    if out.exists():
        print(f"{year}: already processed")
        return
    raw = RAW / f"nadac_{year}.csv"
    if not raw.exists():
        subprocess.run(["curl", "-sfL", URLS[year], "-o", str(raw)], check=True)
    df = pd.read_csv(raw, dtype=str)
    df.columns = [norm(c) for c in df.columns]
    df = df.rename(columns={"classification_for_rate_setting": "classification"})
    df["ndc"] = df["ndc"].str.zfill(11)
    df["nadac_per_unit"] = pd.to_numeric(df["nadac_per_unit"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["month"] = df["effective_date"].dt.to_period("M").astype(str)
    agg = (
        df.dropna(subset=["nadac_per_unit", "month"])
        .groupby(["ndc", "month"], as_index=False)
        .agg(
            nadac=("nadac_per_unit", "median"),
            ndc_description=("ndc_description", "first"),
            classification=("classification", "first"),
            pricing_unit=("pricing_unit", "first"),
        )
    )
    agg.to_csv(out, index=False, compression="gzip")
    raw.unlink()
    print(f"{year}: {len(df):,} weekly rows -> {len(agg):,} NDC-months")


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in sys.argv[1:]] or sorted(URLS)
    for y in years:
        process_year(y)
