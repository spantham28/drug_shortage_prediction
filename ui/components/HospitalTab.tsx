"use client";

import { useMemo, useState } from "react";
import LineChart from "./LineChart";
import { pct, usd } from "@/lib/riskModel";
import { Hospital, HospitalData } from "@/lib/types";

const HOSPITAL = "#3987e5";
const NATIONAL = "#d95926";

const OWNERSHIP: Record<string, string> = { nonprofit: "Nonprofit", for_profit: "For-profit", government: "Government", unknown: "Unknown" };

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <p className="text-[10px] uppercase tracking-widest text-slate-500">{label}</p>
      <p className="mt-1 font-display text-2xl font-semibold text-white">{value}</p>
      {note && <p className="mt-1 text-xs leading-snug text-slate-400">{note}</p>}
    </div>
  );
}

export default function HospitalTab({ data }: { data: HospitalData }) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Hospital | null>(null);
  const mae = parseFloat(data.validation.marginMAE);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    return data.hospitals.filter((h) =>
      h.name.toLowerCase().includes(q) || h.city.toLowerCase().includes(q) || h.ccn.includes(q) || h.state.toLowerCase() === q
    ).slice(0, 25);
  }, [data, query]);

  const h = selected;
  const exposureSeries = h ? [
    { name: "This hospital", color: HOSPITAL, points: Object.entries(h.exposure).map(([x, y]) => ({ x: +x, y })) },
    { name: "National average", color: NATIONAL, dashed: true, points: Object.entries(data.nationalExposure).map(([x, y]) => ({ x: +x, y })) },
  ] : [];
  const marginSeries = h ? [{ name: "Operating margin", color: HOSPITAL, points: Object.entries(h.margin).map(([x, y]) => ({ x: +x, y })) }] : [];
  const betaPct = 100 * (Math.exp(data.betaLogPerUnit) - 1);

  return (
    <div className="animate-fade-up space-y-6">
      <section className="glass-card p-6">
        <h2 className="font-display text-2xl font-semibold text-white">Hospital explorer</h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          For each US short-term acute care hospital with Medicare cost reports: its drug shortage exposure (shortages
          weighted by how long they lasted and how much the hospital uses the affected services), the drug expense
          associated with being above or below the national average, and a forecast of next year&apos;s operating margin.
        </p>
        <div className="relative mt-5 max-w-xl">
          <input className="input-field" placeholder="Search by hospital name, city, state (e.g. MA), or CMS number" value={query}
            onChange={(e) => setQuery(e.target.value)} aria-label="Search hospitals" />
          {matches.length > 0 && (
            <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-xl border border-white/10 bg-navy-900 text-sm shadow-xl">
              {matches.map((m) => (
                <li key={m.ccn}>
                  <button type="button" className="w-full px-4 py-2 text-left hover:bg-white/5"
                    onClick={() => { setSelected(m); setQuery(""); }}>
                    <span className="text-slate-200">{m.name}</span>
                    <span className="ml-2 text-xs text-slate-500">{m.city}{m.city ? ", " : ""}{m.state} · {m.ccn}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <p className="mt-3 text-xs text-slate-500">{data.hospitals.length.toLocaleString()} hospitals with a recent full-year cost report.</p>
      </section>

      {!h && (
        <section className="glass-card p-10 text-center text-sm text-slate-500">Search for a hospital to see its exposure and forecast.</section>
      )}

      {h && (
        <>
          <section className="glass-card p-6">
            <h3 className="font-display text-xl font-semibold text-white">{h.name}</h3>
            <p className="mt-1 text-sm text-slate-400">
              {h.city}{h.city ? ", " : ""}{h.state} · CMS {h.ccn} · {h.beds} beds · {OWNERSHIP[h.ownership] ?? h.ownership}
              {h.teaching ? " · teaching" : ""}{h.rural ? " · rural" : " · urban"}{h.system ? " · health system member" : ""}
            </p>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label={`Exposure index, FY${h.exposureYear}`} value={h.exposure[h.exposureYear]?.toFixed(0) ?? "—"}
                note={`${h.exposureVsNational >= 0 ? "+" : ""}${h.exposureVsNational.toFixed(0)} vs national average · ${h.exposurePercentile.toFixed(0)}th percentile`} />
              <Stat label="Drug expense vs average-exposure hospital" value={h.drugExpenseVsAverage === null ? "—" : usd(h.drugExpenseVsAverage)}
                note={`Estimated per year, from ${betaPct.toFixed(2)}% per exposure unit${h.drugExpense ? ` × ${usd(h.drugExpense)} drug expense` : ""}`} />
              <Stat label={`Forecast operating margin, FY${h.forecastYear}`} value={`${h.forecastMargin.toFixed(1)}%`}
                note={`Typical error ± ${mae.toFixed(1)} percentage points`} />
              <Stat label={`Chance of operating loss, FY${h.forecastYear}`} value={pct(h.lossProbability, 0)}
                note={`Model AUROC ${data.validation.lossAUROC.split(" ")[0]} in held-out hospitals`} />
            </div>
          </section>
          <div className="grid gap-6 lg:grid-cols-2">
            <section className="glass-card p-6">
              <h4 className="text-sm font-semibold text-slate-200">Shortage exposure index</h4>
              <p className="mb-3 mt-1 text-xs text-slate-500">1 unit = one shortage active for a full fiscal year in a service used at national-average intensity</p>
              <LineChart series={exposureSeries} yLabel="Shortage exposure index by fiscal year" format={(v) => v.toFixed(0)} />
            </section>
            <section className="glass-card p-6">
              <h4 className="text-sm font-semibold text-slate-200">Operating margin and forecast</h4>
              <p className="mb-3 mt-1 text-xs text-slate-500">Patient-care margin from Medicare cost reports: net patient revenue minus operating expenses, as a share of net patient revenue. It excludes other income (research grants, investments, relief funds), so academic medical centers often show large negative values.</p>
              <LineChart series={marginSeries} zeroLine yLabel="Operating margin by fiscal year with forecast"
                format={(v) => `${v.toFixed(0)}%`}
                forecast={{ x: h.forecastYear, y: h.forecastMargin, error: mae, color: HOSPITAL, label: "Forecast" }} />
            </section>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-500">
            The drug expense figure applies the study&apos;s average within-hospital estimate ({betaPct.toFixed(2)}% per unit;
            95% CI {(100 * (Math.exp(data.betaCi[0]) - 1)).toFixed(2)}%-{(100 * (Math.exp(data.betaCi[1]) - 1)).toFixed(2)}%) to this
            hospital; it is an estimate, not a measurement of its purchases. Shortage exposure did not improve margin forecasts, so the
            forecast uses financial and operating history only.
          </p>
        </>
      )}
    </div>
  );
}
