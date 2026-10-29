# Generic drug shortages: early warning and hospital financial exposure

**Current analysis: [`study/`](study/README.md).** It rebuilds the project from public data (ASHP shortage
listings, CMS NADAC prices, and Medicare cost reports for every US short-term acute care hospital) and fixes the
issues raised in external review: hospital-specific shortage exposure, hospital- and drug-level hold-outs with
time-forward (horizon) evaluation, explicit prediction timing with no same-year accounting predictors, and three
prespecified model classes.

**Web demo: [`ui/`](ui/README.md)** - shortage watch list, hospital explorer, and validation results from the
revised models (static Next.js site; data exported by `study/models/export_web_demo.py`).

Superseded (kept for the record; do not use for results):

* `drug_shortage_cost_prediction/` - original profit models (random hospital-year splits, same-year accounting predictors)
* `drug_shortage_timeline_prediction/` - original shortage classifier (random drug-window splits, 2025 NDC Directory counts)

`study/models/leakage_demonstration.py` quantifies how much the original results depended on those splits.
