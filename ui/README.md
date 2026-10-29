# Drug Shortage Platform UI

Web demo of the revised study in [`../study`](../study/README.md). It is a static Next.js site: all model outputs
are precomputed JSON in `public/data/`, so there is no Python runtime or serverless function to deploy.

## Tabs

- **Shortage Watch List** — 6-month probability of a new ASHP shortage listing for every generic product priced in
  NADAC in August 2025 (the last month with known shortage status), from the penalized logistic regression validated
  on held-out drugs and years. The "what if" panel recomputes risk in the browser from the exported coefficients
  (`lib/riskModel.ts`).
- **Hospital Explorer** — each hospital's shortage exposure index (FY2016-2024), the drug expense associated with
  being above or below the national average (within-hospital fixed-effects estimate), and a next-year operating margin
  and loss forecast (gradient-boosted trees on financial history; exposure did not improve forecasts).
- **Methods & Validation** — held-out performance and the comparison with the original, leaky design.

## Updating the data

After re-running the study pipeline, regenerate the JSON from the repository root:

```bash
cd study && python models/export_web_demo.py
```

## Local development

```bash
cd ui
npm install
npm run dev
```

Open http://localhost:3000.

## Deploy (Vercel)

Set the Vercel project's root directory to `ui/`. The build is a standard Next.js static build (`next build`);
no environment variables or Python dependencies are required.
