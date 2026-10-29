"use client";

import { useEffect, useState } from "react";
import HospitalTab from "./HospitalTab";
import MethodsTab from "./MethodsTab";
import ShortageTab from "./ShortageTab";
import { HospitalData, ShortageData, TabId, ValidationData } from "@/lib/types";

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: "shortage", label: "Shortage Watch List", icon: "⚠" },
  { id: "hospital", label: "Hospital Explorer", icon: "✚" },
  { id: "methods", label: "Methods & Validation", icon: "◈" },
];

async function load<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path} (${res.status})`);
  return res.json() as Promise<T>;
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>("shortage");
  const [shortage, setShortage] = useState<ShortageData | null>(null);
  const [hospitals, setHospitals] = useState<HospitalData | null>(null);
  const [validation, setValidation] = useState<ValidationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      load<ShortageData>("/data/shortage.json"),
      load<HospitalData>("/data/hospitals.json"),
      load<ValidationData>("/data/validation.json"),
    ])
      .then(([s, h, v]) => { setShortage(s); setHospitals(h); setValidation(v); })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load data"));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="relative overflow-hidden border-b border-white/10">
        <div className="absolute inset-0 opacity-40" style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(255,255,255,0.04) 1px, transparent 0)`,
          backgroundSize: "32px 32px",
        }} />
        <div className="relative mx-auto max-w-6xl px-6 py-14 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-teal-500/30 bg-teal-500/10 px-4 py-1.5 text-xs font-medium uppercase tracking-widest text-teal-400">
            <span className="h-1.5 w-1.5 animate-pulse-soft rounded-full bg-teal-400" />
            Public-data models, validated on unseen drugs, hospitals, and years
          </div>
          <h1 className="font-display text-3xl font-semibold leading-tight tracking-tight text-white sm:text-4xl md:text-5xl">
            Generic Drug Shortages: Early Warning and Hospital Costs
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-slate-400">
            See which generic drugs are at elevated risk of a new shortage, how exposed each US hospital is to shortages
            given the services it provides, and what that exposure means for its drug budget and margins.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <nav className="mb-8 flex flex-wrap gap-2 rounded-2xl border border-white/10 bg-navy-900/50 p-2">
          {TABS.map((tab) => (
            <button key={tab.id} type="button" onClick={() => setActiveTab(tab.id)}
              className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}>
              <span className="mr-2 opacity-70">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {error && (
          <p className="rounded-lg border border-coral-500/30 bg-coral-500/10 px-4 py-3 text-sm text-coral-400">{error}</p>
        )}
        {!error && (!shortage || !hospitals || !validation) && (
          <div className="flex items-center gap-3 py-16 text-sm text-slate-500">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-teal-500/30 border-t-teal-400" />
            Loading model outputs…
          </div>
        )}
        {activeTab === "shortage" && shortage && <ShortageTab data={shortage} />}
        {activeTab === "hospital" && hospitals && <HospitalTab data={hospitals} />}
        {activeTab === "methods" && validation && <MethodsTab data={validation} />}
      </main>

      <footer className="border-t border-white/10 py-8 text-center text-xs text-slate-600">
        Data: ASHP drug shortage listings · CMS NADAC · CMS Healthcare Cost Report Information System. Estimates for research
        and planning; not purchasing or clinical advice.
      </footer>
    </div>
  );
}
