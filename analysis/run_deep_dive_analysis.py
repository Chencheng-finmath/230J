#!/usr/bin/env python3
"""
Deep-dive analysis for the 50-fund sample.

Novel directions implemented:
1) Regime-conditional alpha (high-vol vs low-vol market states)
2) Downside timing (beta shift on negative market days)
3) Tail-risk beta shift (beta shift on worst 5% market days)
4) Rolling 3-year alpha consistency
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252
MIN_OBS = 120
ROLL_WINDOW = 756  # ~3 years
ROLL_STEP = 63  # ~quarterly


def fit_ols(df: pd.DataFrame, y: str, x: list[str], min_obs: int = MIN_OBS):
    d = df[[y] + x].dropna()
    if len(d) < min_obs:
        return None, len(d)
    X = sm.add_constant(d[x], has_constant="add")
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return m, len(d)


def prepare_panel(root: Path) -> pd.DataFrame:
    merged_path = root / "report" / "results" / "analysis_panel_merged.csv"
    if merged_path.exists():
        panel = pd.read_csv(merged_path)
        panel["date"] = pd.to_datetime(panel["date"])
    else:
        # Fallback path: this script assumes phase-1 script has already run.
        raise FileNotFoundError(
            f"Missing merged panel: {merged_path}. Run run_skill_analysis.py first."
        )

    panel = panel.sort_values(["date", "crsp_fundno"]).reset_index(drop=True)

    # Market-state features from factor series.
    market = panel[["date", "mktrf"]].drop_duplicates("date").sort_values("date")
    market["mkt_vol_21"] = market["mktrf"].rolling(21).std()
    vol_cut = market["mkt_vol_21"].quantile(0.75)
    tail_cut = market["mktrf"].quantile(0.05)
    market["high_vol"] = (market["mkt_vol_21"] >= vol_cut).astype(int)
    market["neg_mkt"] = (market["mktrf"] < 0).astype(int)
    market["tail5"] = (market["mktrf"] <= tail_cut).astype(int)

    panel = panel.merge(
        market[["date", "high_vol", "neg_mkt", "tail5", "mkt_vol_21"]],
        on="date",
        how="left",
    )
    return panel


def run_regime_alpha(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = ["mktrf", "smb", "hml", "mom"]
    for fund, g in panel.groupby("crsp_fundno"):
        full, n_full = fit_ols(g, "excess_ret", x)
        low, n_low = fit_ols(g[g["high_vol"] == 0], "excess_ret", x)
        high, n_high = fit_ols(g[g["high_vol"] == 1], "excess_ret", x)
        if full is None or low is None or high is None:
            continue
        a_full = full.params["const"] * TRADING_DAYS
        a_low = low.params["const"] * TRADING_DAYS
        a_high = high.params["const"] * TRADING_DAYS
        rows.append(
            {
                "crsp_fundno": fund,
                "n_full": n_full,
                "n_lowvol": n_low,
                "n_highvol": n_high,
                "alpha_full_annual": a_full,
                "alpha_lowvol_annual": a_low,
                "alpha_highvol_annual": a_high,
                "alpha_high_minus_low": a_high - a_low,
            }
        )
    return pd.DataFrame(rows).sort_values("crsp_fundno").reset_index(drop=True)


def run_downside_timing(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = panel.copy()
    d["mktrf_neg_inter"] = d["mktrf"] * d["neg_mkt"]
    x = ["mktrf", "smb", "hml", "mom", "mktrf_neg_inter"]
    for fund, g in d.groupby("crsp_fundno"):
        m, n = fit_ols(g, "excess_ret", x)
        if m is None:
            continue
        beta_up = m.params["mktrf"]
        shift = m.params["mktrf_neg_inter"]
        beta_down = beta_up + shift
        rows.append(
            {
                "crsp_fundno": fund,
                "n_obs": n,
                "beta_up": beta_up,
                "beta_down": beta_down,
                "beta_down_minus_up": shift,
                "shift_t": m.tvalues["mktrf_neg_inter"],
                "shift_p": m.pvalues["mktrf_neg_inter"],
            }
        )
    return pd.DataFrame(rows).sort_values("crsp_fundno").reset_index(drop=True)


def run_tail_beta(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = panel.copy()
    d["mktrf_tail_inter"] = d["mktrf"] * d["tail5"]
    x = ["mktrf", "smb", "hml", "mom", "mktrf_tail_inter"]
    for fund, g in d.groupby("crsp_fundno"):
        m, n = fit_ols(g, "excess_ret", x)
        if m is None:
            continue
        beta_norm = m.params["mktrf"]
        shift = m.params["mktrf_tail_inter"]
        beta_tail = beta_norm + shift
        rows.append(
            {
                "crsp_fundno": fund,
                "n_obs": n,
                "beta_normal": beta_norm,
                "beta_tail5": beta_tail,
                "beta_tail_minus_normal": shift,
                "tail_shift_t": m.tvalues["mktrf_tail_inter"],
                "tail_shift_p": m.pvalues["mktrf_tail_inter"],
            }
        )
    return pd.DataFrame(rows).sort_values("crsp_fundno").reset_index(drop=True)


def run_alpha_consistency(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = ["mktrf", "smb", "hml", "mom"]
    for fund, g in panel.groupby("crsp_fundno"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < ROLL_WINDOW + MIN_OBS:
            continue
        alphas = []
        for start in range(0, n - ROLL_WINDOW + 1, ROLL_STEP):
            w = g.iloc[start : start + ROLL_WINDOW]
            m, wn = fit_ols(w, "excess_ret", x, min_obs=MIN_OBS)
            if m is None:
                continue
            alphas.append(m.params["const"] * TRADING_DAYS)
        if len(alphas) < 5:
            continue
        arr = np.array(alphas)
        rows.append(
            {
                "crsp_fundno": fund,
                "windows": len(arr),
                "alpha_mean_annual": float(arr.mean()),
                "alpha_median_annual": float(np.median(arr)),
                "alpha_std_annual": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "alpha_positive_share": float(np.mean(arr > 0)),
                "alpha_negative_share": float(np.mean(arr < 0)),
                # Higher is better: high mean alpha + high positive frequency + low volatility.
                "consistency_score": float(
                    (arr.mean() * np.mean(arr > 0)) / (arr.std(ddof=1) + 1e-6)
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("crsp_fundno").reset_index(drop=True)
    return out


def write_summary_report(
    path: Path,
    panel: pd.DataFrame,
    regime: pd.DataFrame,
    downside: pd.DataFrame,
    tail: pd.DataFrame,
    consistency: pd.DataFrame,
):
    lines = []
    lines.append("# Deep-Dive Findings: Novel Directions")
    lines.append("")
    lines.append("## Sample")
    lines.append(f"- Funds: **{panel['crsp_fundno'].nunique()}**")
    lines.append(f"- Date range: **{panel['date'].min().date()} to {panel['date'].max().date()}**")
    lines.append(f"- Rows: **{len(panel):,}**")
    lines.append("")

    lines.append("## 1) Regime Alpha (High-vol vs Low-vol)")
    if regime.empty:
        lines.append("- Not enough observations after regime split.")
    else:
        diff = regime["alpha_high_minus_low"]
        lines.append(f"- Mean (high - low) annual alpha: **{diff.mean():.4%}**")
        lines.append(f"- Median (high - low) annual alpha: **{diff.median():.4%}**")
        lines.append(f"- Share with worse high-vol alpha (diff < 0): **{np.mean(diff < 0):.2%}**")
    lines.append("")

    lines.append("## 2) Downside Timing (Negative-Market Beta Shift)")
    if downside.empty:
        lines.append("- Not enough observations.")
    else:
        s = downside["beta_down_minus_up"]
        lines.append(f"- Mean downside beta shift: **{s.mean():.4f}**")
        lines.append(f"- Median downside beta shift: **{s.median():.4f}**")
        lines.append(f"- Share with defensive shift (beta_down < beta_up): **{np.mean(s < 0):.2%}**")
        sig_def = np.mean((s < 0) & (downside["shift_p"] < 0.05))
        lines.append(f"- Share with statistically significant defensive shift: **{sig_def:.2%}**")
    lines.append("")

    lines.append("## 3) Tail Beta Shift (Worst 5% Market Days)")
    if tail.empty:
        lines.append("- Not enough observations.")
    else:
        s = tail["beta_tail_minus_normal"]
        lines.append(f"- Mean tail beta shift: **{s.mean():.4f}**")
        lines.append(f"- Median tail beta shift: **{s.median():.4f}**")
        lines.append(f"- Share with lower beta in tail days: **{np.mean(s < 0):.2%}**")
        sig_tail_def = np.mean((s < 0) & (tail["tail_shift_p"] < 0.05))
        lines.append(f"- Share with significant tail defense: **{sig_tail_def:.2%}**")
    lines.append("")

    lines.append("## 4) Rolling 3Y Alpha Consistency")
    if consistency.empty:
        lines.append("- Not enough observations for rolling windows.")
    else:
        lines.append(
            f"- Mean positive-alpha window share: **{consistency['alpha_positive_share'].mean():.2%}**"
        )
        lines.append(
            f"- Median positive-alpha window share: **{consistency['alpha_positive_share'].median():.2%}**"
        )
        top = consistency.sort_values("consistency_score", ascending=False).head(10)
        lines.append("")
        lines.append("Top 10 consistency-score funds:")
        lines.append(top.to_markdown(index=False))

    path.write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "report" / "deep_dive"
    out.mkdir(parents=True, exist_ok=True)

    panel = prepare_panel(root)
    regime = run_regime_alpha(panel)
    downside = run_downside_timing(panel)
    tail = run_tail_beta(panel)
    consistency = run_alpha_consistency(panel)

    regime.to_csv(out / "regime_alpha_by_fund.csv", index=False)
    downside.to_csv(out / "downside_timing_by_fund.csv", index=False)
    tail.to_csv(out / "tail_beta_shift_by_fund.csv", index=False)
    consistency.to_csv(out / "rolling_alpha_consistency_by_fund.csv", index=False)

    # Quick leaderboards.
    if not regime.empty:
        regime.sort_values("alpha_high_minus_low").head(10).to_csv(
            out / "top10_highvol_underperformers.csv", index=False
        )
        regime.sort_values("alpha_high_minus_low", ascending=False).head(10).to_csv(
            out / "top10_highvol_outperformers.csv", index=False
        )
    if not downside.empty:
        downside.sort_values("beta_down_minus_up").head(10).to_csv(
            out / "top10_defensive_downside_timing.csv", index=False
        )
    if not tail.empty:
        tail.sort_values("beta_tail_minus_normal").head(10).to_csv(
            out / "top10_tail_defensive_funds.csv", index=False
        )

    write_summary_report(
        out / "deep_dive_summary.md", panel, regime, downside, tail, consistency
    )

    print("Done.")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
