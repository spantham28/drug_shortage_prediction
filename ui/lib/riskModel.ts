import { ShortageData, ShortageProduct } from "./types";

// Penalized logistic regression exported from study/models/export_web_demo.py:
// logit = intercept + sum_j coef_j * (g(x_j) - mean_j) / scale_j, with g = signed log for skewed counts
// and missing inputs replaced by the training median (as in the fitted pipeline).
function transform(x: number, slog: boolean) {
  return slog ? Math.sign(x) * Math.log1p(Math.abs(x)) : x;
}

export function shortageRisk(data: ShortageData, x: Record<string, number | null>) {
  let logit = data.intercept;
  for (const f of data.features) {
    const raw = x[f.key];
    const v = raw === null || raw === undefined || Number.isNaN(raw) ? f.median : raw;
    logit += (f.coef * (transform(v, f.slog) - f.mean)) / f.scale;
  }
  return 1 / (1 + Math.exp(-logit));
}

// The exported product risk is exact; edits are applied as a change on the logit scale so a product whose
// inputs had missing values keeps its exact baseline.
export function editedRisk(data: ShortageData, p: ShortageProduct, edits: Record<string, number>) {
  const base = shortageRisk(data, p.x);
  const edited = shortageRisk(data, { ...p.x, ...edits });
  const logit = (q: number) => Math.log(q / (1 - q));
  const z = logit(p.risk) + logit(edited) - logit(base);
  return 1 / (1 + Math.exp(-z));
}

export const pct = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}%`;
export const usd = (v: number) =>
  `${v < 0 ? "−" : ""}$${Math.abs(v) >= 1e6 ? (Math.abs(v) / 1e6).toFixed(2) + "M" : Math.round(Math.abs(v)).toLocaleString("en-US")}`;
