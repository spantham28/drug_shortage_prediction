# Revised study pipeline (2026 revision)

This folder replaces the analyses in `drug_shortage_cost_prediction/` and `drug_shortage_timeline_prediction/`,
which are kept only for the record. The revision responds to external review: the original analyses split
repeated hospital-year and drug-month observations at random (leakage), used a national shortage count that is
identical for every hospital in a year, predicted current-year profit from line items of the same income
statement, compared 12-15 algorithms without motivation, and removed "outliers" from test data.

Everything here is built from public data and runs end to end with the scripts below.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `data_build/01_download_hcris.py` | Medicare cost report cells, FY2014-2025 (`data/interim/hcris_*`) |
| 2 | `data_build/02_download_nadac.py` | NADAC prices collapsed to NDC-month (`data/interim/nadac_ndc_month_*`) |
| 3 | `data_build/03_build_shortage_events.py` | One row per ASHP shortage with dates, route, therapeutic category, supplier count |
| 4 | `data_build/04_build_drug_panel.py` | Drug-market x month panel with 6-month shortage-onset label |
| 5 | `data_build/05_build_hospital_panel.py` | Hospital-fiscal-year panel (financials, payer mix, service mix, drug expense) |
| 6 | `data_build/06_build_exposure.py` | Hospital-specific shortage exposure index and variants |
| 7 | `models/aim1_shortage_forecast.py` | Aim 1: shortage-onset forecasting, held-out drugs and years |
| 8 | `models/aim2a_exposure_effects.py` | Aim 2a: within-hospital fixed-effects estimates |
| 9 | `models/aim2b_margin_forecast.py` | Aim 2b: next-year margin forecasting with held-out hospitals and horizons |
| 10 | `models/make_tables_figures.py` | Manuscript tables and figures (`outputs/`) |
| 11 | `models/manuscript_numbers.py` | Every number quoted in the manuscript (`outputs/numbers.json`) |
| 12 | `models/export_web_demo.py` | Final models refit on all labeled data; JSON for the web demo (`ui/public/data/`) |
| — | `models/leakage_demonstration.py` | Re-runs the original models with unit-level splits to quantify leakage |

Run from this folder with Python 3.11 (`pip install -r requirements.txt`):

```bash
python data_build/01_download_hcris.py && python data_build/02_download_nadac.py && python data_build/03_build_shortage_events.py && python data_build/04_build_drug_panel.py && python data_build/05_build_hospital_panel.py && python data_build/06_build_exposure.py && python models/aim1_shortage_forecast.py && python models/aim2a_exposure_effects.py && python models/aim2b_margin_forecast.py && python models/aim2b_margin_forecast.py --posthoc && python models/leakage_demonstration.py && python models/make_tables_figures.py && python models/manuscript_numbers.py && python models/export_web_demo.py
```

## Design safeguards

* **No unit appears on both sides of a split.** Aim 1 holds out 20% of active ingredients (all strengths and
  forms) and evaluates only on later months; Aim 2b holds out 20% of hospitals and evaluates only on later
  fiscal years.
* **Prediction timing is explicit.** Features are measured at or before the forecast date; outcomes are 6
  months (Aim 1) or 1-2 fiscal years (Aim 2b) later. No line item of the outcome-year income statement is a
  predictor.
* **Tuning uses training data only** (grouped 5-fold cross-validation by ingredient or hospital).
* **Preprocessing is fitted inside each training fold** (imputation, signed-log transform, scaling for the linear
  and neural models; none for gradient-boosted trees).
* **Three model classes** (penalized linear, gradient-boosted trees, neural network) plus simple reference models.
* **No oversampling and no outlier removal of test data.**
* **Uncertainty** is reported with cluster bootstrap 95% CIs (resampling ingredients or hospitals).
