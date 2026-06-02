"use client";

import { useCallback, useEffect, useState } from "react";
import { modelSynopsis } from "@/lib/synopsis";
import {
  IncomeFeatureSchema,
  IncomeResult,
  SHORTAGE_FIELDS,
  ShortageResult,
  TabId,
} from "@/lib/types";

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: "shortage", label: "Shortage Prediction", icon: "⚠" },
  { id: "income", label: "Net Income Prediction", icon: "$" },
  { id: "synopsis", label: "Model Methodology", icon: "◈" },
];

function formatTransform(t: string) {
  const map: Record<string, string> = {
    none: "Original",
    square: "Square",
    sqrt: "Square Root",
    log: "Log",
  };
  return map[t] ?? t;
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>("shortage");

  // Shortage state
  const [shortageForm, setShortageForm] = useState<Record<string, string>>({
    avg_nadac: "",
    manufacturer_num: "",
    ingredient_num: "",
    num_forms: "",
    liquid_flag: "0",
  });
  const [shortageResult, setShortageResult] = useState<ShortageResult | null>(null);
  const [shortageLoading, setShortageLoading] = useState(false);
  const [shortageError, setShortageError] = useState<string | null>(null);

  // Income state
  const [featureSchema, setFeatureSchema] = useState<IncomeFeatureSchema | null>(null);
  const [incomeForm, setIncomeForm] = useState<Record<string, string>>({});
  const [incomeResult, setIncomeResult] = useState<IncomeResult | null>(null);
  const [incomeLoading, setIncomeLoading] = useState(false);
  const [incomeError, setIncomeError] = useState<string | null>(null);

  const loadFeatureSchema = useCallback(async () => {
    try {
      const res = await fetch("/api/income");
      if (!res.ok) throw new Error("Failed to load feature schema");
      const data: IncomeFeatureSchema = await res.json();
      setFeatureSchema(data);
      const defaults: Record<string, string> = {};
      data.features.forEach((f) => {
        if (f.type === "categorical") {
          defaults[f.name] = f.options?.[0] ?? "Loss";
        } else {
          defaults[f.name] = String(f.median ?? 0);
        }
      });
      setIncomeForm(defaults);
    } catch {
      setFeatureSchema(null);
    }
  }, []);

  useEffect(() => {
    loadFeatureSchema();
  }, [loadFeatureSchema]);

  async function handleShortageSubmit(e: React.FormEvent) {
    e.preventDefault();
    setShortageLoading(true);
    setShortageError(null);
    setShortageResult(null);
    try {
      const payload: Record<string, number> = {};
      for (const field of SHORTAGE_FIELDS) {
        const val = shortageForm[field.key];
        if (val === "" || val === undefined) {
          throw new Error(`Please enter ${field.label}`);
        }
        payload[field.key] = Number(val);
      }
      const res = await fetch("/api/shortage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed");
      setShortageResult(data);
    } catch (err) {
      setShortageError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setShortageLoading(false);
    }
  }

  async function handleIncomeSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIncomeLoading(true);
    setIncomeError(null);
    setIncomeResult(null);
    try {
      const payload: Record<string, string | number> = {};
      featureSchema?.features.forEach((f) => {
        const val = incomeForm[f.name];
        if (f.type === "categorical") {
          payload[f.name] = val ?? "Loss";
        } else {
          payload[f.name] = Number(val ?? f.median ?? 0);
        }
      });
      const res = await fetch("/api/income", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed");
      setIncomeResult(data);
    } catch (err) {
      setIncomeError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setIncomeLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <header className="relative overflow-hidden border-b border-white/10">
        <div className="absolute inset-0 opacity-40" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.04) 1px, transparent 0)`,
          backgroundSize: "32px 32px",
        }} />
        <div className="relative mx-auto max-w-6xl px-6 py-14 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-teal-500/30 bg-teal-500/10 px-4 py-1.5 text-xs font-medium uppercase tracking-widest text-teal-400">
            <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-teal-400" />
            ML-Powered Drug Shortage Platform
          </div>
          <h1 className="font-display text-3xl font-semibold leading-tight tracking-tight text-white sm:text-4xl md:text-5xl">
            Building Models to Tackle Generic Drug Shortages in the US
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-slate-400">
            Predict generic drug shortages from pricing signals and estimate hospital net income
            from operational data — powered by ensemble classification and random forest regression.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        {/* Tabs */}
        <nav className="mb-8 flex flex-wrap gap-2 rounded-2xl border border-white/10 bg-navy-900/50 p-2">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            >
              <span className="mr-2 opacity-70">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Shortage Tab */}
        {activeTab === "shortage" && (
          <div className="animate-fade-up grid gap-8 lg:grid-cols-5">
            <form onSubmit={handleShortageSubmit} className="glass-card p-8 lg:col-span-3">
              <h2 className="font-display text-2xl font-semibold text-white">
                Shortage Risk Assessment
              </h2>
              <p className="mt-2 text-sm text-slate-400">
                Enter drug characteristics to predict shortage_flag and the probability of a shortage.
              </p>

              <div className="mt-8 grid gap-5 sm:grid-cols-2">
                {SHORTAGE_FIELDS.map((field) => (
                  <div
                    key={field.key}
                    className={field.key === "liquid_flag" ? "sm:col-span-2" : ""}
                  >
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-slate-400">
                      {field.label}
                    </label>
                    {field.type === "select" ? (
                      <select
                        className="input-field"
                        value={shortageForm[field.key]}
                        onChange={(e) =>
                          setShortageForm((prev) => ({ ...prev, [field.key]: e.target.value }))
                        }
                      >
                        {field.options?.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        step={field.step}
                        placeholder={field.placeholder}
                        className="input-field"
                        value={shortageForm[field.key]}
                        onChange={(e) =>
                          setShortageForm((prev) => ({ ...prev, [field.key]: e.target.value }))
                        }
                        required
                      />
                    )}
                    <p className="mt-1 text-xs text-slate-500">{field.hint}</p>
                  </div>
                ))}
              </div>

              {shortageError && (
                <p className="mt-4 rounded-lg border border-coral-500/30 bg-coral-500/10 px-4 py-3 text-sm text-coral-400">
                  {shortageError}
                </p>
              )}

              <button type="submit" className="btn-primary mt-8 w-full sm:w-auto" disabled={shortageLoading}>
                {shortageLoading ? "Running model…" : "Predict Shortage Risk"}
              </button>
            </form>

            <div className="glass-card flex flex-col p-8 lg:col-span-2">
              <h3 className="text-sm font-medium uppercase tracking-widest text-slate-400">
                Prediction Result
              </h3>

              {!shortageResult && !shortageLoading && (
                <div className="flex flex-1 flex-col items-center justify-center py-12 text-center">
                  <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-2xl">
                    ⚠
                  </div>
                  <p className="text-sm text-slate-500">
                    Submit drug characteristics to see shortage_flag and probability.
                  </p>
                </div>
              )}

              {shortageLoading && (
                <div className="flex flex-1 items-center justify-center py-12">
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-teal-500/30 border-t-teal-400" />
                </div>
              )}

              {shortageResult && (
                <div className="mt-6 flex flex-1 flex-col gap-6">
                  <div
                    className={`result-badge-shortage ${
                      shortageResult.shortage_flag === 1 ? "yes" : "no"
                    } self-start`}
                  >
                    {shortageResult.shortage_flag === 1 ? "● Shortage" : "● No Shortage"}
                    <span className="font-normal opacity-80">
                      (shortage_flag = {shortageResult.shortage_flag})
                    </span>
                  </div>

                  <div>
                    <p className="text-xs uppercase tracking-widest text-slate-500">
                      Probability of Shortage
                    </p>
                    <p className="mt-1 font-display text-5xl font-semibold text-white">
                      {shortageResult.shortage_probability_pct}
                      <span className="text-2xl text-teal-400">%</span>
                    </p>
                  </div>

                  <div className="relative h-3 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                      style={{
                        width: `${shortageResult.shortage_probability_pct}%`,
                        background:
                          shortageResult.shortage_flag === 1
                            ? "linear-gradient(90deg, #f43f5e, #fb7185)"
                            : "linear-gradient(90deg, #0d9488, #2dd4bf)",
                      }}
                    />
                  </div>

                  <p className="text-sm text-slate-400">{shortageResult.shortage_label}</p>

                  <div className="mt-auto rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-slate-500">
                    <p>
                      Model: <span className="text-slate-300">{shortageResult.model_type}</span>
                    </p>
                    <p className="mt-1">
                      Ensemble:{" "}
                      <span className="text-slate-300">
                        {shortageResult.ensemble_members.join(", ")}
                      </span>
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Income Tab */}
        {activeTab === "income" && (
          <div className="animate-fade-up">
            <form onSubmit={handleIncomeSubmit}>
              <div className="glass-card p-8">
                <h2 className="font-display text-2xl font-semibold text-white">
                  Hospital Net Income Prediction
                </h2>
                <p className="mt-2 text-sm text-slate-400">
                  Enter all model features (selected via information gain and nonlinear correlation
                  analysis). Values are transformed automatically before prediction.
                </p>

                {!featureSchema ? (
                  <div className="mt-8 flex items-center gap-3 text-sm text-slate-500">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-teal-500/30 border-t-teal-400" />
                    Loading feature schema…
                  </div>
                ) : (
                  <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {featureSchema.features.map((feat) => (
                      <div key={feat.name}>
                        <label className="mb-1.5 block text-xs font-medium text-slate-400 leading-snug">
                          {feat.name}
                          {feat.transform !== "none" && (
                            <span className="ml-1 rounded bg-teal-500/15 px-1.5 py-0.5 text-[10px] uppercase text-teal-400">
                              {formatTransform(feat.transform)}
                            </span>
                          )}
                        </label>
                        {feat.type === "categorical" ? (
                          <select
                            className="input-field"
                            value={incomeForm[feat.name] ?? "Loss"}
                            onChange={(e) =>
                              setIncomeForm((prev) => ({ ...prev, [feat.name]: e.target.value }))
                            }
                          >
                            {feat.options?.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            step="any"
                            className="input-field"
                            value={incomeForm[feat.name] ?? ""}
                            onChange={(e) =>
                              setIncomeForm((prev) => ({ ...prev, [feat.name]: e.target.value }))
                            }
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {incomeError && (
                  <p className="mt-4 rounded-lg border border-coral-500/30 bg-coral-500/10 px-4 py-3 text-sm text-coral-400">
                    {incomeError}
                  </p>
                )}

                <button
                  type="submit"
                  className="btn-primary mt-8"
                  disabled={incomeLoading || !featureSchema}
                >
                  {incomeLoading ? "Running model…" : "Predict Net Income"}
                </button>
              </div>
            </form>

            {(incomeResult || incomeLoading) && (
              <div className="glass-card mt-6 p-8 text-center">
                {incomeLoading ? (
                  <div className="flex justify-center py-8">
                    <div className="h-10 w-10 animate-spin rounded-full border-2 border-teal-500/30 border-t-teal-400" />
                  </div>
                ) : incomeResult ? (
                  <>
                    <p className="text-xs font-medium uppercase tracking-widest text-slate-500">
                      Predicted {incomeResult.target}
                    </p>
                    <p
                      className={`mt-3 font-display text-5xl font-semibold ${
                        incomeResult.net_income >= 0 ? "text-teal-400" : "text-coral-400"
                      }`}
                    >
                      {incomeResult.net_income_formatted}
                    </p>
                    <p className="mt-3 text-sm text-slate-400">
                      {incomeResult.model_name} · Dollar Net Income (no transform on target)
                    </p>
                  </>
                ) : null}
              </div>
            )}
          </div>
        )}

        {/* Synopsis Tab */}
        {activeTab === "synopsis" && (
          <div className="animate-fade-up grid gap-8 lg:grid-cols-2">
            {/* Shortage synopsis */}
            <section className="glass-card p-8">
              <div className="mb-6 flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-coral-500/15 text-lg">
                  ⚠
                </div>
                <div>
                  <h2 className="font-display text-xl font-semibold text-white">
                    {modelSynopsis.shortage.title}
                  </h2>
                  <p className="mt-1 text-sm text-slate-400">{modelSynopsis.shortage.subtitle}</p>
                </div>
              </div>

              <h3 className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Pipeline Steps
              </h3>
              <div className="space-y-0">
                {modelSynopsis.shortage.pipeline.map((step) => (
                  <div key={step.step} className="synopsis-step">
                    <p className="text-sm font-semibold text-slate-200">{step.step}</p>
                    <p className="mt-1 text-sm leading-relaxed text-slate-400">{step.detail}</p>
                  </div>
                ))}
              </div>

              <h3 className="mb-3 mt-8 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Feature Importance (EnsembleTop3)
              </h3>
              <p className="mb-3 text-xs text-slate-500">
                Permutation importance on the test set · source:{" "}
                {modelSynopsis.shortage.featureImportanceSource}
              </p>
              <div className="mb-6 overflow-hidden rounded-xl border border-white/10">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/10 bg-white/5 text-left text-slate-500">
                      <th className="px-3 py-2 font-medium">Rank</th>
                      <th className="px-3 py-2 font-medium">Feature</th>
                      <th className="px-3 py-2 font-medium">Importance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelSynopsis.shortage.featureImportance.map((row) => {
                      const maxImp = modelSynopsis.shortage.featureImportance[0].permutationImportance;
                      const barPct = (row.permutationImportance / maxImp) * 100;
                      return (
                        <tr key={row.feature} className="border-b border-white/5 last:border-0">
                          <td className="px-3 py-2 font-mono text-slate-500">{row.rank}</td>
                          <td className="px-3 py-2 text-slate-300">{row.label}</td>
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <div className="h-1.5 min-w-[80px] flex-1 overflow-hidden rounded-full bg-white/10">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-coral-500 to-teal-500"
                                  style={{ width: `${barPct}%` }}
                                />
                              </div>
                              <span className="shrink-0 font-mono text-teal-400">
                                {row.permutationImportance.toFixed(3)}
                              </span>
                            </div>
                            <p className="mt-0.5 text-[10px] text-slate-600">
                              ± {row.permutationStd.toFixed(4)}
                            </p>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <h3 className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Results
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {modelSynopsis.shortage.results.map((r) => (
                  <div key={r.metric} className="metric-pill">
                    <p className="text-[10px] uppercase tracking-wide text-slate-500">{r.metric}</p>
                    <p className="mt-0.5 text-sm font-semibold text-white">{r.value}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Income synopsis */}
            <section className="glass-card p-8">
              <div className="mb-6 flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-500/15 text-lg">
                  $
                </div>
                <div>
                  <h2 className="font-display text-xl font-semibold text-white">
                    {modelSynopsis.income.title}
                  </h2>
                  <p className="mt-1 text-sm text-slate-400">{modelSynopsis.income.subtitle}</p>
                </div>
              </div>

              <h3 className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Pipeline Steps
              </h3>
              <div className="space-y-0">
                {modelSynopsis.income.pipeline.map((step) => (
                  <div key={step.step} className="synopsis-step">
                    <p className="text-sm font-semibold text-slate-200">{step.step}</p>
                    <p className="mt-1 text-sm leading-relaxed text-slate-400">{step.detail}</p>
                  </div>
                ))}
              </div>

              <h3 className="mb-3 mt-8 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Top Information Gain Features
              </h3>
              <div className="mb-6 overflow-hidden rounded-xl border border-white/10">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/10 bg-white/5 text-left text-slate-500">
                      <th className="px-3 py-2 font-medium">Feature</th>
                      <th className="px-3 py-2 font-medium">IG</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelSynopsis.income.topInformationGain.map((row) => (
                      <tr key={row.feature} className="border-b border-white/5 last:border-0">
                        <td className="px-3 py-2 text-slate-300">{row.feature}</td>
                        <td className="px-3 py-2 font-mono text-teal-400">{row.ig}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="mb-3 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Top Nonlinear Correlations with Net Income
              </h3>
              <div className="mb-6 overflow-hidden rounded-xl border border-white/10">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/10 bg-white/5 text-left text-slate-500">
                      <th className="px-3 py-2 font-medium">Feature</th>
                      <th className="px-3 py-2 font-medium">Transform</th>
                      <th className="px-3 py-2 font-medium">|r|</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelSynopsis.income.topCorrelations.map((row) => (
                      <tr key={row.feature} className="border-b border-white/5 last:border-0">
                        <td className="px-3 py-2 text-slate-300">{row.feature}</td>
                        <td className="px-3 py-2 text-slate-400">{row.transform}</td>
                        <td className="px-3 py-2 font-mono text-teal-400">{row.r}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
                Results
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {modelSynopsis.income.results.map((r) => (
                  <div key={r.metric} className="metric-pill">
                    <p className="text-[10px] uppercase tracking-wide text-slate-500">{r.metric}</p>
                    <p className="mt-0.5 text-sm font-semibold text-white">{r.value}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="border-t border-white/10 py-8 text-center text-xs text-slate-600">
        Drug Shortage Platform · Models trained on NADAC pricing signals &amp; CMS hospital cost reports
      </footer>
    </div>
  );
}
