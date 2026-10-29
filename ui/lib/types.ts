export type TabId = "shortage" | "hospital" | "methods";

export interface ShortageFeature {
  key: string;
  label: string;
  editable: boolean;
  slog: boolean;
  median: number;
  mean: number;
  scale: number;
  coef: number;
}

export interface ShortageProduct {
  market: string;
  ingredient: string;
  risk: number;
  price: number;
  unit: string;
  x: Record<string, number | null>;
  drivers: string[];
}

export interface ShortageData {
  asOf: string;
  horizonMonths: number;
  baseRate: number;
  intercept: number;
  features: ShortageFeature[];
  products: ShortageProduct[];
  currentShortages: string[];
  validation: Record<string, { auroc: string; top5Sensitivity: number; top5Lift: number }>;
}

export interface Hospital {
  ccn: string;
  name: string;
  city: string;
  state: string;
  beds: number;
  ownership: string;
  teaching: number;
  rural: number;
  system: number;
  exposure: Record<string, number>;
  exposureYear: number;
  exposureVsNational: number;
  exposurePercentile: number;
  drugExpense: number | null;
  drugExpenseVsAverage: number | null;
  margin: Record<string, number>;
  forecastYear: number;
  forecastMargin: number;
  lossProbability: number;
}

export interface HospitalData {
  years: number[];
  nationalExposure: Record<string, number>;
  betaLogPerUnit: number;
  betaCi: [number, number];
  validation: { marginMAE: string; marginR2: string; lossAUROC: string };
  hospitals: Hospital[];
}

export interface ValidationData {
  leakage: { analysis: string; design: string; metric: string; mean: number }[];
  aim1: { test_set: string; model: string; events: number; AUROC: string; AUPRC: string; top5pct_sensitivity: number; top5pct_lift: number }[];
  aim2a: { outcome: string; specification: string; coef: number; ci_lo: number; ci_hi: number; identifying_variation_share: number }[];
  aim2b: { horizon: number; task: string; feature_set: string; model: string; MAE_pp: string; R2: string; AUROC: string }[];
  aim2bIncremental: { horizon: number; task: string; model: string; comparison: string; metric: string; delta: number; ci_lo: number; ci_hi: number }[];
  headline: Record<string, string>;
}
