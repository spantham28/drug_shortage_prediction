"""
Download CMS HCRIS hospital cost reports (Form CMS-2552-10) and keep only the
worksheet cells used in the study.

Raw zip files are ~140 MB each and the numeric file unzips to ~700 MB, so each
year is streamed from the zip, filtered, written to data/interim/hcris_cells_FYxxxx.csv.gz,
and the zip is deleted.

Source: https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports
"""

import csv
import gzip
import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
URL = "https://downloads.cms.gov/FILES/HCRIS/HOSP10FY{year}.ZIP"

# Worksheet -> set of line prefixes to keep (None = keep every line).
# Line codes are 5 characters: 3-digit line + 2-digit subscript.
KEEP_NMRC = {
    "S300001": None,          # S-3 Pt I: beds, patient days, discharges, FTEs
    "G300000": None,          # G-3: statement of revenues and expenses
    "G200000": None,          # G-2: patient revenues
    "G000000": None,          # G: balance sheet
    "A000000": None,          # A: cost centres (pharmacy, drugs charged, OR, ICU, ...)
    "C000001": None,          # C Pt I: cost and charges by cost centre
    "S200001": {"02600", "02700"},  # S-2 Pt I numeric: urban(1)/rural(2) indicator, CBSA
}
KEEP_ALPHA = {
    "S200001": {"00200", "02100", "02600", "05600", "14000"},  # state, control, urban/rural, teaching, chain
    "A000000": None,                                   # cost-centre labels for subscripted lines
}


def keep(row, spec):
    wk = row[1]
    if wk not in spec:
        return False
    lines = spec[wk]
    return lines is None or row[2] in lines


def process_year(year):
    out = INTERIM / f"hcris_cells_FY{year}.csv.gz"
    rpt_out = INTERIM / f"hcris_rpt_FY{year}.csv"
    if out.exists() and rpt_out.exists():
        print(f"FY{year}: already processed")
        return
    zpath = RAW / f"HOSP10FY{year}.zip"
    if not zpath.exists():
        print(f"FY{year}: downloading")
        subprocess.run(["curl", "-sfL", URL.format(year=year), "-o", str(zpath)], check=True)
    with zipfile.ZipFile(zpath) as z:
        names = {n.lower(): n for n in z.namelist()}
        rpt = next(n for k, n in names.items() if k.endswith("_rpt.csv"))
        nmrc = next(n for k, n in names.items() if k.endswith("_nmrc.csv"))
        alpha = next(n for k, n in names.items() if k.endswith("_alpha.csv"))
        rpt_out.write_bytes(z.read(rpt))
        n_kept = 0
        with gzip.open(out, "wt", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["rpt_rec_num", "wksht", "line", "col", "value", "kind"])
            for member, spec, kind in [(nmrc, KEEP_NMRC, "n"), (alpha, KEEP_ALPHA, "a")]:
                with z.open(member) as fh:
                    for row in csv.reader(io.TextIOWrapper(fh, encoding="latin-1")):
                        if len(row) >= 5 and keep(row, spec):
                            w.writerow(row[:5] + [kind])
                            n_kept += 1
    zpath.unlink()
    print(f"FY{year}: kept {n_kept:,} cells")


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in sys.argv[1:]] or list(range(2014, 2025))
    for y in years:
        process_year(y)
