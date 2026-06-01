# Drug Shortage Platform UI

Interactive web interface for generic drug shortage prediction and hospital net income forecasting.

**Page title:** Building Models to Tackle Generic Drug Shortages in the US

## Features

- **Shortage Prediction** — Enter drug acquisition cost, manufacturer count, ingredients, dosage forms, and formulation type. Returns `shortage_flag` and shortage probability from the EnsembleTop3 classifier.
- **Net Income Prediction** — Enter all 15 model features (with automatic nonlinear transforms). Returns predicted hospital Net Income in dollars.
- **Model Methodology** — Synopsis of outlier removal, information gain, cross-correlation analysis, and model performance for both pipelines.

## Local Development

### 1. Export inference models (required once, or after retraining)

From the repository root:

```bash
pip install -r requirements.txt imbalanced-learn joblib
python scripts/export_inference_models.py
```

This writes serialized models to `ui/models/`.

### 2. Run the UI

```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

> **Note:** Python API routes (`/api/shortage`, `/api/income`) run as Vercel serverless functions locally via `vercel dev`, or deploy to Vercel for full stack testing. During `next dev`, API calls to `/api/*` require Vercel CLI:

```bash
npm i -g vercel
cd ui
vercel dev
```

## Deploy to Vercel

1. Push the repository to GitHub.
2. Import the project in [Vercel](https://vercel.com/new).
3. Set the **Root Directory** to `ui`.
4. Ensure model artifacts exist in `ui/models/` (run the export script before deploying, and commit the `.pkl` files).
5. Deploy — Vercel auto-detects Next.js and Python functions in `api/`.

### Environment

No environment variables required. Models are loaded from `ui/models/` at runtime.

## Project Structure

```
ui/
├── app/                  # Next.js App Router pages
├── api/                  # Python serverless prediction endpoints
│   ├── shortage.py
│   ├── income.py
│   └── _inference.py
├── components/           # React components
├── lib/                  # Types and model methodology content
├── models/               # Serialized ML artifacts (generated)
├── requirements.txt      # Python dependencies for Vercel
└── vercel.json
```

## Models

| Tab | Model | Source |
|-----|-------|--------|
| Shortage | EnsembleTop3 (ExtraTrees + RandomForest + GradientBoosting) | `drug_shortage_timeline_prediction/` |
| Net Income | GradientBoostingRegressor | `drug_shortage_cost_prediction/` |
