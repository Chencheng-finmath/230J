#!/usr/bin/env python3
"""
Generate report figures for the mutual fund skill project.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pct(x):
    return x * 100.0


def main():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "report"
    fig_dir = report_dir / "figures"
    ensure_dir(fig_dir)

    # Inputs
    all_data = pd.read_csv(root / "all_data_clean.csv", usecols=["crsp_fundno", "caldt"])
    all_data["date"] = pd.to_datetime(all_data["caldt"])
    all_data["year"] = all_data["date"].dt.year

    alpha = pd.read_csv(report_dir / "results" / "alpha_results_by_fund.csv")
    timing = pd.read_csv(report_dir / "results" / "timing_results_by_fund.csv")
    persistence = pd.read_csv(report_dir / "results" / "persistence_summary.csv")
    regime = pd.read_csv(report_dir / "deep_dive" / "regime_alpha_by_fund.csv")
    downside = pd.read_csv(report_dir / "deep_dive" / "downside_timing_by_fund.csv")
    tail = pd.read_csv(report_dir / "deep_dive" / "tail_beta_shift_by_fund.csv")
    qsum = pd.read_csv(report_dir / "crisis_score" / "quintile_summary.csv")
    preg = pd.read_csv(report_dir / "crisis_score" / "predictive_regression.csv")
    vreg = pd.read_csv(report_dir / "crisis_score" / "resilience_validation_regression.csv")

    # Figure 1: active funds by year
    ycnt = all_data.groupby("year")["crsp_fundno"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ycnt.index, ycnt.values, marker="o", linewidth=1.8, color="#1f77b4")
    ax.set_title("Figure 1. Active Funds in Sample by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Active Funds")
    ax.grid(alpha=0.25)
    savefig(fig, fig_dir / "fig01_active_funds_by_year.png")

    # Figure 2: Carhart alpha distribution
    c4 = alpha[alpha["model"] == "Carhart4"].copy()
    c4_alpha = pct(c4["alpha_annual"].values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(c4_alpha, bins=16, color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(np.mean(c4_alpha), color="#d62728", linestyle="-", linewidth=1.5, label="Mean")
    ax.axvline(np.median(c4_alpha), color="#2ca02c", linestyle="-", linewidth=1.5, label="Median")
    ax.set_title("Figure 2. Distribution of Carhart Annual Alpha (50 Funds)")
    ax.set_xlabel("Annual Alpha (%)")
    ax.set_ylabel("Count of Funds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    savefig(fig, fig_dir / "fig02_carhart_alpha_distribution.png")

    # Figure 3: top/bottom 10 Carhart alpha
    c4s = c4.sort_values("alpha_annual")
    bot = c4s.head(10)
    top = c4s.tail(10)
    show = pd.concat([bot, top], axis=0).copy()
    show["label"] = show["crsp_fundno"].astype(str)
    show["alpha_pct"] = pct(show["alpha_annual"])
    colors = ["#d62728"] * len(bot) + ["#2ca02c"] * len(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(show))
    ax.barh(y, show["alpha_pct"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Figure 3. Bottom and Top 10 Funds by Carhart Annual Alpha")
    ax.set_xlabel("Annual Alpha (%)")
    ax.set_ylabel("CRSP Fund Number")
    ax.grid(axis="x", alpha=0.2)
    savefig(fig, fig_dir / "fig03_carhart_alpha_top_bottom.png")

    # Figure 4: timing gamma distributions by model
    tm = timing[timing["model"] == "TreynorMazuy"]["gamma"].values
    hm = timing[timing["model"] == "HenrikssonMerton"]["gamma"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(tm, bins=15, color="#f28e2b", alpha=0.85, edgecolor="white")
    axes[0].axvline(np.mean(tm), color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Treynor-Mazuy Gamma")
    axes[0].set_xlabel("Gamma")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)
    axes[1].hist(hm, bins=15, color="#e15759", alpha=0.85, edgecolor="white")
    axes[1].axvline(np.mean(hm), color="black", linestyle="--", linewidth=1.2)
    axes[1].set_title("Henriksson-Merton Gamma")
    axes[1].set_xlabel("Gamma")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Figure 4. Distribution of Timing Coefficients")
    savefig(fig, fig_dir / "fig04_timing_gamma_distributions.png")

    # Figure 5: persistence summary
    order = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]
    ps = persistence.set_index("portfolio").reindex(order).reset_index()
    vals = pct(ps["mean_alpha_annual"].values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["#4c78a8", "#4c78a8", "#4c78a8", "#4c78a8", "#4c78a8", "#d62728"]
    ax.bar(ps["portfolio"], vals, color=cols, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Figure 5. Next-Year Alpha by Prior-Year Alpha Quintile")
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Mean Annual Alpha (%)")
    ax.grid(axis="y", alpha=0.2)
    savefig(fig, fig_dir / "fig05_persistence_quintiles.png")

    # Figure 6: regime scatter
    r = regime.copy()
    r["low"] = pct(r["alpha_lowvol_annual"])
    r["high"] = pct(r["alpha_highvol_annual"])
    mn = min(r["low"].min(), r["high"].min())
    mx = max(r["low"].max(), r["high"].max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r["low"], r["high"], s=35, alpha=0.8, color="#4c78a8")
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1)
    ax.set_title("Figure 6. High-Vol vs Low-Vol Carhart Alpha by Fund")
    ax.set_xlabel("Low-Vol Alpha (% annual)")
    ax.set_ylabel("High-Vol Alpha (% annual)")
    share_high_better = np.mean(r["high"] > r["low"])
    ax.text(
        0.03,
        0.97,
        f"Share high-vol > low-vol: {share_high_better:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.grid(alpha=0.2)
    savefig(fig, fig_dir / "fig06_regime_alpha_scatter.png")

    # Figure 7: downside vs tail shift
    dt = downside.merge(tail, on="crsp_fundno", how="inner")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        dt["beta_down_minus_up"],
        dt["beta_tail_minus_normal"],
        s=35,
        alpha=0.85,
        color="#59a14f",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Figure 7. Defensive Shift Map (Downside vs Tail Beta Shift)")
    ax.set_xlabel("Downside Shift (beta_down - beta_up)")
    ax.set_ylabel("Tail Shift (beta_tail - beta_normal)")
    defensive_share = np.mean(
        (dt["beta_down_minus_up"] < 0) & (dt["beta_tail_minus_normal"] < 0)
    )
    ax.text(
        0.03,
        0.97,
        f"Double-defensive share: {defensive_share:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.grid(alpha=0.2)
    savefig(fig, fig_dir / "fig07_defensive_shift_scatter.png")

    # Figure 8: crisis score quintile future alpha
    qs = qsum.set_index("portfolio").reindex(order).reset_index()
    vals = pct(qs["mean_future_alpha_annual"].values)
    se = pct(qs["std_future_alpha_annual"].values / np.sqrt(qs["years"].values))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(qs["portfolio"], vals, yerr=se, capsize=3, color="#b07aa1", alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Figure 8. Future Alpha by Crisis-Score Quintile")
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Mean Future Annual Alpha (%)")
    ax.grid(axis="y", alpha=0.2)
    savefig(fig, fig_dir / "fig08_crisis_score_quintiles.png")

    # Figure 9: coefficient comparison (pooled models)
    pooled_alpha = preg[preg["model"] == "pooled"].iloc[0]
    pooled_val = vreg[vreg["model"] == "pooled"].copy()
    coeff = [
        ("Future Alpha", pooled_alpha["beta_score"], pooled_alpha["t_score"]),
        (
            "Future Downside Shift",
            pooled_val.loc[
                pooled_val["outcome"] == "downside_shift_year", "beta_score"
            ].iloc[0],
            pooled_val.loc[
                pooled_val["outcome"] == "downside_shift_year", "t_score"
            ].iloc[0],
        ),
        (
            "Future Tail Shift",
            pooled_val.loc[
                pooled_val["outcome"] == "tail_shift_year", "beta_score"
            ].iloc[0],
            pooled_val.loc[
                pooled_val["outcome"] == "tail_shift_year", "t_score"
            ].iloc[0],
        ),
    ]
    labels = [c[0] for c in coeff]
    betas = np.array([c[1] for c in coeff], dtype=float)
    tvals = np.array([c[2] for c in coeff], dtype=float)
    ses = np.abs(betas / np.where(tvals == 0, np.nan, tvals))
    cis = 1.96 * ses
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, betas, yerr=cis, capsize=4, color=["#4c78a8", "#59a14f", "#f28e2b"], alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Coefficient on Crisis Score Percentile")
    ax.set_title("Figure 9. Crisis Score Predictive Coefficients (Pooled Regressions)")
    ax.grid(axis="y", alpha=0.2)
    savefig(fig, fig_dir / "fig09_crisis_score_coefficients.png")

    # Figure notes markdown
    notes = f"""# Figure Guide for Final Report

This folder contains recommended figures for the final write-up.

## Figure List

1. `fig01_active_funds_by_year.png`  
   Shows panel coverage over time (how many funds are active each year).  
   Why it matters: demonstrates sample evolution and why panel is unbalanced in early years.

2. `fig02_carhart_alpha_distribution.png`  
   Histogram of annualized Carhart alpha across 50 funds, with mean/median markers.  
   Why it matters: shows cross-sectional alpha is centered near small positive values and is dispersed.

3. `fig03_carhart_alpha_top_bottom.png`  
   Bottom 10 and Top 10 funds by Carhart annual alpha.  
   Why it matters: highlights that winners and losers coexist, supporting the “heterogeneous skill” narrative.

4. `fig04_timing_gamma_distributions.png`  
   Distribution of timing coefficients for Treynor-Mazuy and Henriksson-Merton models.  
   Why it matters: visual evidence that timing signals are weak and often skewed toward negative values.

5. `fig05_persistence_quintiles.png`  
   Next-year alpha by prior-year alpha quintile (including Q5-Q1 spread).  
   Why it matters: directly visualizes persistence strength (or weakness).

6. `fig06_regime_alpha_scatter.png`  
   Fund-level high-vol alpha vs low-vol alpha with 45-degree reference line.  
   Why it matters: shows which funds adapt better/worse during volatile regimes.

7. `fig07_defensive_shift_scatter.png`  
   Downside beta shift vs tail beta shift map; lower-left quadrant indicates defensive behavior on both dimensions.  
   Why it matters: quantifies how rare true “double-defensive” funds are.

8. `fig08_crisis_score_quintiles.png`  
   Future alpha for crisis-score quintiles (with standard-error bars).  
   Why it matters: tests whether resilience score predicts higher future alpha economically.

9. `fig09_crisis_score_coefficients.png`  
   Coefficients of crisis-score predictive regressions (future alpha, future downside shift, future tail shift).  
   Why it matters: summarizes trade-off result: score predicts defense, not higher alpha.

## Recommended placement in the report
- Data section: Figure 1  
- Baseline results: Figures 2-5  
- Deep-dive section: Figures 6-7  
- Crisis score section: Figures 8-9
"""
    (fig_dir / "figure_notes.md").write_text(notes)

    print("Done.")
    print(f"Figure folder: {fig_dir}")
    print("Generated 9 figures + figure_notes.md")


if __name__ == "__main__":
    main()
