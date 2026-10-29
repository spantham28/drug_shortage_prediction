"use client";

import { ValidationData } from "@/lib/types";

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="glass-card p-6">
      <h3 className="font-display text-xl font-semibold text-white">{title}</h3>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-slate-400">{children}</div>
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-white/10 bg-white/5 text-left text-slate-500">
            {head.map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-white/5 last:border-0">
              {r.map((c, j) => <td key={j} className={`px-3 py-2 ${j === 0 ? "text-slate-300" : "font-mono text-slate-300"}`}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MethodsTab({ data }: { data: ValidationData }) {
  const H = data.headline;
  const leak = (d: string) => data.leakage.find((r) => r.design.startsWith(d))?.mean ?? NaN;
  const a2b = data.aim2b.filter((r) => r.horizon === 1 && r.feature_set.startsWith("A"));
  const fe = (o: string) => data.aim2a.find((r) => r.outcome === o && r.specification.startsWith("Main"));
  const om = fe("operating_margin_pp");

  return (
    <div className="animate-fade-up space-y-6">
      <Card title="What this platform does">
        <p>
          Two questions, using only public data: <span className="text-slate-200">can new generic drug shortages be
          anticipated?</span> and <span className="text-slate-200">how much do shortages cost an individual hospital?</span>{" "}
          The data are ASHP drug shortage listings (2015-2025), CMS National Average Drug Acquisition Cost (NADAC) prices
          ({H.n_markets} generic products, {H.n_market_months} product-months), and Medicare cost reports for {H.n_hosp} hospitals
          ({H.n_hosp_years} hospital-years, FY2016-2024).
        </p>
        <p>
          Every model was tested only on data it never saw during development: later years, plus drugs or hospitals held
          out entirely. Earlier versions of this platform split repeated observations of the same drug or hospital at random
          across training and test sets, which inflated performance (below).
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Shortage early warning">
          <p>
            A penalized logistic regression predicts whether a product gets a new shortage listing within 6 months, from
            price trends, the number of manufacturers, shortage history, and spillovers (other forms of the drug, or the
            same manufacturers, already in shortage). It was chosen over gradient-boosted trees and a neural network
            because it held up best in later years.
          </p>
          <Table head={["Test set (2022-2025)", "Model", "AUROC (95% CI)", "Top 5%: share of shortages"]}
            rows={data.aim1.map((r) => [r.test_set.split(",")[0], r.model, r.AUROC,
              r.model.startsWith("Reference") ? "— (tied scores)" : `${(100 * r.top5pct_sensitivity).toFixed(0)}%`])} />
          <p className="text-xs text-slate-500">AUROC: the chance the model ranks a product that went into shortage above one that did not (0.5 = coin flip, 1 = perfect).</p>
        </Card>

        <Card title="Hospital drug expense and margins">
          <p>
            Each hospital&apos;s exposure index counts shortages by how long they lasted within its fiscal year and weights them by
            how much the hospital uses the services that need those drugs (surgery, intensive care, oncology, inpatient vs
            outpatient care). Comparing each hospital with itself over time (fixed effects), each extra unit of exposure was
            associated with {H.fe_pct}% higher drug expense (95% CI {H.fe_pct_lo}%-{H.fe_pct_hi}%), about ${H.fe_usd_mean} a year at
            the average hospital. Operating margin showed no consistent change
            {om ? ` (main estimate ${om.coef.toFixed(3)} percentage points per unit; 95% CI ${om.ci_lo.toFixed(3)} to ${om.ci_hi.toFixed(3)})` : ""};
            drugs are a median {H.drug_share_opex_median}% of hospital operating expenses.
          </p>
          <Table head={["Next-year margin model", "Error (MAE, pp)", "R²", "Loss AUROC"]}
            rows={a2b.filter((r) => r.task === "regression").map((r) => {
              const c = a2b.find((q) => q.task === "classification" && q.model === r.model);
              return [r.model.startsWith("Reference") ? "Persistence (last year's margin)" : r.model, r.MAE_pp, r.R2, c?.AUROC ?? ""];
            })} />
          <p className="text-xs text-slate-500">Adding shortage exposure did not improve these forecasts, so the forecast shown in the hospital explorer uses financial history only.</p>
        </Card>
      </div>

      <Card title="Why the numbers changed from earlier versions">
        <p>
          Re-running the original models on the original data, changing only how observations were split between training
          and testing:
        </p>
        <Table head={["Original model", "Random split (original)", "No drug/hospital in both sets", "Prior-year predictors"]}
          rows={[
            ["Shortage classifier, AUROC", leak("Random split of drug").toFixed(2), leak("Split by drug").toFixed(2), "—"],
            ["Profit regression, R²", leak("Random split of hospital-years").toFixed(2), leak("Split by hospital").toFixed(2), leak("Prior-year predictors").toFixed(2)],
          ]} />
        <p>
          When the same drug or hospital appears in both training and test data, a model can score well by recognizing it
          rather than by learning anything that generalizes. The original profit model also used parts of the same
          year&apos;s income statement as predictors, which partly reconstructs profit by arithmetic. Full methods:{" "}
          <a className="text-teal-400 underline" href="https://github.com/spantham28/drug_shortage_prediction/tree/main/study">study/ on GitHub</a>.
        </p>
      </Card>
    </div>
  );
}
