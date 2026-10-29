"""Assemble manuscript tables (CSV) and figures (PNG) from the model outputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"
INTERIM = ROOT / "data" / "interim"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
MODEL_COLORS = {"Penalized linear": "#2a6f97", "Gradient-boosted trees": "#c75b39", "Neural network (MLP)": "#6a994e",
                "Reference": "#8d8d8d"}


def color_for(name):
    return next((c for k, c in MODEL_COLORS.items() if name.startswith(k)), "#8d8d8d")


def table1_hospitals():
    d = pd.read_csv(OUT / "aim2" / "analytic_sample_aim2a.csv.gz", dtype={"ccn": str})
    hosp = d.sort_values("fy").groupby("ccn").agg(
        exposure=("exposure", "mean"), beds=("beds", "first"), teaching=("teaching", "first"),
        system_member=("system_member", "first"), rural=("rural", "first"), ownership=("ownership", "first"),
        operating_margin=("operating_margin", "median"), drug_cost_per_apd=("drug_cost_per_apd", "median"),
        drug_expense=("drug_expense", "median"), periop=("periop_share", "first"), icu=("icu_day_share", "first"),
        medicaid=("medicaid_day_share", "median"), medicare=("medicare_day_share", "median"),
        years=("fy", "size"))
    hosp["tertile"] = pd.qcut(hosp["exposure"], 3, labels=["Low", "Middle", "High"])

    def n_pct(mask):
        return f"{int(mask.sum()):,} ({100 * mask.mean():.1f})"

    def summarize(g):
        # JAMA style: frequencies as No. (%), continuous as mean (SD) or median (IQR), categories alphabetical
        def q(c, f=1, scale=1):
            lo, hi = scale * g[c].quantile(.25), scale * g[c].quantile(.75)
            sep = " to " if lo < 0 else "-"  # AMA: use "to" when a range includes a negative value
            return f"{scale*g[c].median():.{f}f} ({lo:.{f}f}{sep}{hi:.{f}f})"
        return pd.Series({
            "Hospitals, No.": f"{len(g):,}",
            "Hospital-years, No.": f"{g['years'].sum():,}",
            "Exposure index, mean (SD)": f"{g['exposure'].mean():.1f} ({g['exposure'].std():.1f})",
            "Beds, median (IQR)": q("beds", 0),
            "Health system member, No. (%)": n_pct(g["system_member"] == 1),
            "Ownership, No. (%)": "",
            "   For-profit": n_pct(g["ownership"] == "for_profit"),
            "   Government": n_pct(g["ownership"] == "government"),
            "   Nonprofit": n_pct(g["ownership"] == "nonprofit"),
            "Rural, No. (%)": n_pct(g["rural"] == 1),
            "Teaching, No. (%)": n_pct(g["teaching"] == 1),
            "Medicaid share of inpatient days, median (IQR), %": q("medicaid", 1, 100),
            "Medicare share of inpatient days, median (IQR), %": q("medicare", 1, 100),
            "ICU share of inpatient days at baseline, median (IQR), %": q("icu", 1, 100),
            "Perioperative share of direct costs at baseline, median (IQR), %": q("periop", 1, 100),
            "Annual drug expense, median (IQR), $ million": q("drug_expense", 1, 1e-6),
            "Drug expense per adjusted patient day, median (IQR), $": q("drug_cost_per_apd", 0),
            "Operating margin, median (IQR), %": q("operating_margin", 1, 100),
        })

    t = pd.concat({"All": summarize(hosp), **{f"{k} exposure tertile": summarize(g)
                                               for k, g in hosp.groupby("tertile", observed=True)}}, axis=1)
    t.to_csv(TAB / "table1_hospital_characteristics.csv")
    return t


def table2_aim1():
    p = pd.read_csv(OUT / "aim1" / "performance.csv")
    cols = ["test_set", "model", "n", "events", "AUROC", "AUPRC", "Brier", "cal_intercept", "cal_slope",
            "top5pct_sensitivity", "top5pct_ppv", "top5pct_lift"]
    t = p[cols].copy()
    for c in ["cal_intercept", "cal_slope", "top5pct_lift"]:
        t[c] = t[c].round(2)
    for c in ["top5pct_sensitivity", "top5pct_ppv"]:
        t[c] = (100 * t[c]).round(1)
    t.to_csv(TAB / "table2_aim1_performance.csv", index=False)
    return t


def table3_aim2a():
    r = pd.read_csv(OUT / "aim2" / "fixed_effects_results.csv")
    r["estimate (95% CI)"] = [f"{c:.4f} ({lo:.4f} to {hi:.4f})" for c, lo, hi in zip(r["coef"], r["ci_lo"], r["ci_hi"])]
    r["p"] = r["p"].map(lambda v: "<.001" if v < 0.001 else f"{v:.3f}")
    r[["outcome", "specification", "term", "estimate (95% CI)", "p", "n_obs", "n_hospitals"]] \
        .to_csv(TAB / "table3_aim2a_fixed_effects.csv", index=False)
    return r


def table4_aim2b():
    p = pd.read_csv(OUT / "aim2" / "forecast_performance.csv")
    p.to_csv(TAB / "table4_aim2b_forecast_performance.csv", index=False)
    inc = load_incremental()
    inc.to_csv(TAB / "etable_aim2b_incremental_value.csv", index=False)
    return p, inc


def figure_exposure():
    """Exposure heterogeneity: national shortage count is the same for every hospital; the index is not."""
    d = pd.read_csv(OUT / "aim2" / "analytic_sample_aim2a.csv.gz", dtype={"ccn": str})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    years = sorted(d["fy"].unique())
    ax = axes[0]
    data = [d.loc[d["fy"] == y, "exposure"].values for y in years]
    ax.boxplot(data, positions=range(len(years)), widths=0.55, showfliers=False,
               medianprops={"color": "#eb6834", "linewidth": 2}, boxprops={"color": "#2a78d6"},
               whiskerprops={"color": "#2a78d6"}, capprops={"color": "#2a78d6"})
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.set_ylabel("Exposure index, shortage-years")
    ax.set_xlabel("Fiscal year")
    ax.set_title("A. Hospital exposure index", loc="left", fontsize=9)
    ax = axes[1]
    within = d.groupby("fy").agg(cv_index=("exposure", lambda s: s.std() / s.mean()),
                                 cv_count=("shortage_count_any", lambda s: s.std() / s.mean()))
    ax.plot(within.index, within["cv_index"], marker="o", ms=4, lw=2, color="#2a78d6", label="Hospital exposure index")
    ax.plot(within.index, within["cv_count"], marker="s", ms=4, lw=2, color="#eb6834", label="National shortage count")
    ax.text(within.index[-1] + 0.15, within["cv_index"].iloc[-1], "Exposure index", fontsize=7, va="center")
    ax.text(within.index[-1] + 0.15, within["cv_count"].iloc[-1], "National count", fontsize=7, va="center")
    ax.set_xlim(within.index[0] - 0.3, within.index[-1] + 2.2)
    ax.set_xticks(list(within.index)[::2])
    ax.set_ylim(0, None)
    ax.set_ylabel("Coefficient of variation across hospitals")
    ax.set_xlabel("Fiscal year")
    ax.set_title("B. Variation across hospitals within a year", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "figure1_exposure_heterogeneity.png", dpi=300)
    plt.close(fig)


def figure_aim1_calibration():
    c = pd.read_csv(OUT / "aim1" / "calibration_deciles.csv")
    c = c[c["test_set"].str.startswith("Held-out")]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = max(c["mean_pred"].max(), c["obs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=0.8, ls="--")
    for m, g in c.groupby("model"):
        ax.plot(g["mean_pred"], g["obs"], marker="o", ms=3, lw=1, color=color_for(m), label=m)
    ax.set_xlabel("Predicted 6-month shortage risk")
    ax.set_ylabel("Observed proportion")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "efigure_aim1_calibration.png", dpi=300)
    plt.close(fig)


def load_incremental():
    inc = pd.read_csv(OUT / "aim2" / "forecast_incremental_value.csv")
    post = OUT / "aim2" / "forecast_incremental_value_posthoc_relative.csv"
    if post.exists():
        inc = pd.concat([inc, pd.read_csv(post)], ignore_index=True)
    return inc.sort_values(["task", "horizon", "model", "comparison"]).reset_index(drop=True)


def figure_incremental():
    inc = load_incremental()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6))
    for ax, task, label in [(axes[0], "regression", "Change in MAE, percentage points"),
                            (axes[1], "classification", "Change in AUROC")]:
        g = inc[inc["task"] == task].copy()
        g["cmp"] = g["comparison"].str.split(":").str[0]
        order_m = {"Penalized linear": 0, "Gradient-boosted trees": 1, "Neural network (MLP)": 2}
        order_c = {"B": 0, "B-rel": 1, "C": 2}
        g = g.sort_values(["horizon", "model", "cmp"], key=lambda c: c.map(order_m) if c.name == "model"
                          else (c.map(order_c) if c.name == "cmp" else c), ascending=[False, False, False]).reset_index(drop=True)
        short = {"Penalized linear": "Linear", "Gradient-boosted trees": "GBT", "Neural network (MLP)": "MLP"}
        ylab = [f"{r.horizon}-y, {short[r.model]}, {r.cmp}" for r in g.itertuples()]
        y = np.arange(len(g))
        colors = {"B": "#2a6f97", "B-rel": "#6a994e", "C": "#c75b39"}
        for cmp_, gg in g.groupby("cmp"):
            ax.errorbar(gg["delta"], gg.index, xerr=[gg["delta"] - gg["ci_lo"], gg["ci_hi"] - gg["delta"]], fmt="o",
                        ms=3, color=colors[cmp_], ecolor=colors[cmp_], lw=1, label={"B": "B: exposure known at forecast",
                        "B-rel": "B-rel: relative exposure (post hoc)", "C": "C: realized exposure (oracle)"}[cmp_])
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ylab, fontsize=6)
        ax.set_xlabel(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center", ncol=3)
    axes[0].set_title("A. Margin (lower = better)", loc="left", fontsize=9)
    axes[1].set_title("B. Operating loss (higher = better)", loc="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "figure2_incremental_value.png", dpi=300)
    plt.close(fig)


def flow():
    f = pd.read_csv(INTERIM / "hospital_panel_flow.csv")
    f.to_csv(TAB / "etable_hospital_flow.csv", index=False)


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in [table2_aim1, figure_aim1_calibration, table1_hospitals, table3_aim2a, figure_exposure,
               table4_aim2b, figure_incremental, flow]:
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skipped", fn.__name__, e)
