"use client";

import { useMemo, useState } from "react";
import { editedRisk, pct } from "@/lib/riskModel";
import { ShortageData, ShortageProduct } from "@/lib/types";

const FORMS: { id: string; label: string; key?: string }[] = [
  { id: "all", label: "All forms" },
  { id: "inj", label: "Injectable", key: "form_injectable" },
  { id: "oral", label: "Tablet / capsule", key: "form_oral_solid" },
  { id: "liq", label: "Oral liquid", key: "form_liquid" },
  { id: "top", label: "Topical / inhaled / drops", key: "form_topical" },
];

function RiskBar({ risk, max }: { risk: number; max: number }) {
  return (
    <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
      <div className="h-full rounded-full bg-gradient-to-r from-teal-500 to-coral-500" style={{ width: `${Math.min(100, (risk / max) * 100)}%` }} />
    </div>
  );
}

export default function ShortageTab({ data }: { data: ShortageData }) {
  const [query, setQuery] = useState("");
  const [form, setForm] = useState("all");
  const [selected, setSelected] = useState<ShortageProduct | null>(data.products[0] ?? null);
  const [edits, setEdits] = useState<Record<string, number>>({});

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    const key = FORMS.find((f) => f.id === form)?.key;
    return data.products.filter((p) => (!q || p.market.includes(q)) && (!key || p.x[key] === 1));
  }, [data, query, form]);
  const shown = filtered.slice(0, 60);
  const maxRisk = data.products[0]?.risk ?? 1;
  const rank = selected ? data.products.indexOf(selected) + 1 : 0;
  const top5Cut = data.products[Math.floor(data.products.length * 0.05)]?.risk ?? 1;
  const whatIf = selected ? editedRisk(data, selected, edits) : 0;
  const editable = data.features.filter((f) => f.editable);

  return (
    <div className="animate-fade-up space-y-6">
      <section className="glass-card p-6">
        <h2 className="font-display text-2xl font-semibold text-white">Shortage watch list</h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          Estimated probability that each generic drug product gets a new ASHP shortage listing in the next{" "}
          {data.horizonMonths} months, using public NADAC price and supplier data as of {data.asOf}. The average
          product&apos;s risk is {pct(data.baseRate)}. In 2022-2025 testing, the 5% of products the model rated
          highest risk accounted for {pct(data.validation["All ingredients"].top5Sensitivity, 0)} of new shortages
          ({data.validation["All ingredients"].top5Lift.toFixed(1)}× the base rate). {data.currentShortages.length} products
          already in shortage that month are excluded.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <input className="input-field max-w-sm" placeholder="Search products, e.g. AMOXICILLIN" value={query}
            onChange={(e) => setQuery(e.target.value)} aria-label="Search products" />
          <select className="input-field max-w-[220px]" value={form} onChange={(e) => setForm(e.target.value)} aria-label="Dosage form">
            {FORMS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
          </select>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-5">
        <section className="glass-card overflow-hidden lg:col-span-3">
          <div className="max-h-[620px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-navy-900">
                <tr className="border-b border-white/10 text-left text-slate-500">
                  <th className="px-4 py-3 font-medium">Product</th>
                  <th className="px-4 py-3 font-medium">6-month risk</th>
                  <th className="px-4 py-3 font-medium">vs average</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((p) => (
                  <tr key={p.market} onClick={() => { setSelected(p); setEdits({}); }}
                    className={`cursor-pointer border-b border-white/5 transition hover:bg-white/5 ${selected?.market === p.market ? "bg-teal-500/10" : ""}`}>
                    <td className="px-4 py-2.5 text-slate-200">{p.market}</td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2"><RiskBar risk={p.risk} max={maxRisk} /><span className="font-mono text-slate-300">{pct(p.risk)}</span></div>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-slate-400">{(p.risk / data.baseRate).toFixed(1)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length > shown.length && (
              <p className="px-4 py-3 text-xs text-slate-500">Showing the {shown.length} highest-risk of {filtered.length.toLocaleString()} matching products. Narrow the search to see more.</p>
            )}
            {!filtered.length && <p className="px-4 py-6 text-sm text-slate-500">No matching products. Products already in shortage are not listed.</p>}
          </div>
        </section>

        {selected && (
          <section className="glass-card p-6 lg:col-span-2">
            <p className="text-xs uppercase tracking-widest text-slate-500">Rank {rank.toLocaleString()} of {data.products.length.toLocaleString()}{selected.risk >= top5Cut ? " · top 5%" : ""}</p>
            <h3 className="mt-1 font-display text-lg font-semibold text-white">{selected.market}</h3>
            <p className="mt-1 text-xs text-slate-500">NADAC ${selected.price.toFixed(4)} per {selected.unit} · {selected.x.n_labelers ?? "?"} manufacturer(s) priced</p>
            <div className="mt-5 flex items-end gap-6">
              <div>
                <p className="text-[10px] uppercase tracking-widest text-slate-500">Model risk</p>
                <p className="font-display text-4xl font-semibold text-white">{pct(selected.risk)}</p>
              </div>
              {Object.keys(edits).length > 0 && (
                <div>
                  <p className="text-[10px] uppercase tracking-widest text-slate-500">With your changes</p>
                  <p className="font-display text-4xl font-semibold text-teal-400">{pct(whatIf)}</p>
                </div>
              )}
            </div>
            {selected.drivers.length > 0 && (
              <div className="mt-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-teal-400">What raises this product&apos;s risk</p>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-300">{selected.drivers.map((d) => <li key={d}>{d}</li>)}</ul>
              </div>
            )}
            <div className="mt-6">
              <p className="text-xs font-semibold uppercase tracking-widest text-teal-400">What if…</p>
              <p className="mt-1 text-xs text-slate-500">Change an input to see how the model responds. Log price changes: 0.1 ≈ +10%, -0.1 ≈ −10%.</p>
              <div className="mt-3 grid grid-cols-2 gap-3">
                {editable.map((f) => (
                  <label key={f.key} className="block">
                    <span className="mb-1 block text-[11px] leading-snug text-slate-400">{f.label}</span>
                    <input type="number" step="any" className="input-field !py-2"
                      value={edits[f.key] ?? selected.x[f.key] ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        setEdits((prev) => {
                          const next = { ...prev };
                          if (v === "") delete next[f.key]; else next[f.key] = Number(v);
                          return next;
                        });
                      }} />
                  </label>
                ))}
              </div>
              {Object.keys(edits).length > 0 && (
                <button type="button" className="mt-3 text-xs text-teal-400 underline" onClick={() => setEdits({})}>Reset inputs</button>
              )}
            </div>
            <p className="mt-6 text-[11px] leading-relaxed text-slate-500">
              Associations, not causes: changing an input shows how the model&apos;s estimate moves, not what would happen
              in the market. For example, products with more manufacturers score slightly higher because shortage listings
              name individual product codes, and more manufacturers means more codes that can be listed. NADAC covers
              retail pharmacy prices, so hospital injectables are under-represented.
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
