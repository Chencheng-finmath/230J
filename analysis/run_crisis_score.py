#!/usr/bin/env python3
"""
Build Crisis-Resilience Skill Score and test predictive power for next-year alpha.

Outputs:
- fund_year_score.csv
- predictive_sample.csv
- predictive_regression.csv
- quintile_future_alpha_by_year.csv
- quintile_summary.csv
- crisis_resilience_score_report.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

TRADING_DAYS = 252
MIN_OBS = 120
ROLL_WIN = 252
ROLL_STEP = 63


def fit_ols(df: pd.DataFrame, y: str, x: list[str], min_obs: int = MIN_OBS):
    d = df[[y] + x].dropna()
    if len(d) < min_obs:
        return None, len(d)
    X = sm.add_constant(d[x], has_constant="add")
    m = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return m, len(d)


def trailing_window(df: pd.DataFrame, year: int, years_back: int = 3) -> pd.DataFrame:
    lo = year - years_back + 1
    return df[(df["year"] >= lo) & (df["year"] <= year)].copy()


def carhart_alpha_annual(df: pd.DataFrame, min_obs: int = MIN_OBS) -> tuple[Optional[float], Optional[float], int]:
    m, n = fit_ols(df, "excess_ret", ["mktrf", "smb", "hml", "mom"], min_obs=min_obs)
    if m is None:
        return None, None, n
    return float(m.params["const"] * TRADING_DAYS), float(m.pvalues["const"]), n


def downside_shift(df: pd.DataFrame) -> tuple[Optional[float], Optional[float], int]:
    d = df.copy()
    d["neg"] = (d["mktrf"] < 0).astype(int)
    d["mktrf_neg_inter"] = d["mktrf"] * d["neg"]
    m, n = fit_ols(d, "excess_ret", ["mktrf", "smb", "hml", "mom", "mktrf_neg_inter"])
    if m is None:
        return None, None, n
    return float(m.params["mktrf_neg_inter"]), float(m.pvalues["mktrf_neg_inter"]), n


def tail_shift(df: pd.DataFrame, tail_cut: float) -> tuple[Optional[float], Optional[float], int]:
    d = df.copy()
    d["tail"] = (d["mktrf"] <= tail_cut).astype(int)
    d["mktrf_tail_inter"] = d["mktrf"] * d["tail"]
    m, n = fit_ols(d, "excess_ret", ["mktrf", "smb", "hml", "mom", "mktrf_tail_inter"])
    if m is None:
        return None, None, n
    return float(m.params["mktrf_tail_inter"]), float(m.pvalues["mktrf_tail_inter"]), n


def regime_alpha_gap(df: pd.DataFrame, vol_cut: float) -> tuple[Optional[float], int, int]:
    low = df[df["mkt_vol_21"] < vol_cut]
    high = df[df["mkt_vol_21"] >= vol_cut]
    a_low, _, n_low = carhart_alpha_annual(low)
    a_high, _, n_high = carhart_alpha_annual(high)
    if a_low is None or a_high is None:
        return None, n_low, n_high
    return float(a_high - a_low), n_low, n_high


def rolling_consistency(df: pd.DataFrame) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    g = df.sort_values("date").reset_index(drop=True)
    if len(g) < ROLL_WIN + MIN_OBS:
        return None, None, None, 0
    alphas = []
    for i in range(0, len(g) - ROLL_WIN + 1, ROLL_STEP):
        w = g.iloc[i : i + ROLL_WIN]
        a, _, n = carhart_alpha_annual(w, min_obs=MIN_OBS)
        if a is not None:
            alphas.append(a)
    if len(alphas) < 4:
        return None, None, None, len(alphas)
    arr = np.array(alphas, dtype=float)
    pos_share = float(np.mean(arr > 0))
    std = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))
    score = float((mean * pos_share) / (std + 1e-6))
    return score, pos_share, std, len(arr)


def compute_yearly_outcome_alpha(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fund, year), g in panel.groupby(["crsp_fundno", "year"]):
        a, p, n = carhart_alpha_annual(g, min_obs=MIN_OBS)
        if a is None:
            continue
        rows.append(
            {
                "crsp_fundno": fund,
                "year": int(year),
                "alpha_next_annual": a,
                "alpha_next_p": p,
                "n_obs_next": n,
            }
        )
    return pd.DataFrame(rows)


def compute_yearly_risk_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fund, year), g in panel.groupby(["crsp_fundno", "year"]):
        if len(g) < MIN_OBS:
            continue
        d_shift, _, _ = downside_shift(g)
        tail_cut = g["mktrf"].quantile(0.05)
        t_shift, _, _ = tail_shift(g, tail_cut=tail_cut)
        if d_shift is None or t_shift is None:
            continue
        rows.append(
            {
                "crsp_fundno": fund,
                "year": int(year),
                "downside_shift_year": float(d_shift),
                "tail_shift_year": float(t_shift),
            }
        )
    return pd.DataFrame(rows)


def zscore_by_year(df: pd.DataFrame, col: str) -> pd.Series:
    def z(s: pd.Series) -> pd.Series:
        mu = s.mean()
        sd = s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    return df.groupby("score_year")[col].transform(z)


def percentile_by_year(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("score_year")[col].rank(pct=True)


def build_score(panel: pd.DataFrame) -> pd.DataFrame:
    years = sorted(panel["year"].unique())
    rows = []

    # Market-only rolling vol from full panel dates.
    market = panel[["date", "mktrf"]].drop_duplicates("date").sort_values("date").copy()
    market["mkt_vol_21"] = market["mktrf"].rolling(21).std()
    panel2 = panel.merge(market[["date", "mkt_vol_21"]], on="date", how="left")

    for year in years:
        # Need 3-year trailing window for score year.
        if year < min(years) + 2:
            continue
        w = trailing_window(panel2, year, years_back=3)
        if w.empty:
            continue

        # Market thresholds based only on the trailing window.
        daily = w[["date", "mktrf", "mkt_vol_21"]].drop_duplicates("date")
        tail_cut = daily["mktrf"].quantile(0.05)
        vol_cut = daily["mkt_vol_21"].quantile(0.75)

        for fund, g in w.groupby("crsp_fundno"):
            d_shift, d_p, n_d = downside_shift(g)
            t_shift, t_p, n_t = tail_shift(g, tail_cut=tail_cut)
            rgap, n_low, n_high = regime_alpha_gap(g, vol_cut=vol_cut)
            c_score, c_pos, c_std, n_roll = rolling_consistency(g)

            rows.append(
                {
                    "crsp_fundno": fund,
                    "score_year": int(year),
                    "downside_shift": d_shift,
                    "downside_p": d_p,
                    "tail_shift": t_shift,
                    "tail_p": t_p,
                    "regime_alpha_gap": rgap,
                    "consistency_raw": c_score,
                    "consistency_pos_share": c_pos,
                    "consistency_std": c_std,
                    "n_downside": n_d,
                    "n_tail": n_t,
                    "n_lowvol": n_low,
                    "n_highvol": n_high,
                    "n_roll_windows": n_roll,
                }
            )

    score = pd.DataFrame(rows)
    score = score.dropna(
        subset=["downside_shift", "tail_shift", "regime_alpha_gap", "consistency_raw"]
    ).copy()

    # More defensive is better => negate downside/tail shifts.
    score["downside_def"] = -score["downside_shift"]
    score["tail_def"] = -score["tail_shift"]
    score["regime_skill"] = score["regime_alpha_gap"]
    score["consistency_skill"] = score["consistency_raw"]

    score["z_downside"] = zscore_by_year(score, "downside_def")
    score["z_tail"] = zscore_by_year(score, "tail_def")
    score["z_regime"] = zscore_by_year(score, "regime_skill")
    score["z_consistency"] = zscore_by_year(score, "consistency_skill")

    score["crisis_score_raw"] = score[
        ["z_downside", "z_tail", "z_regime", "z_consistency"]
    ].mean(axis=1)
    score["crisis_score_pct"] = percentile_by_year(score, "crisis_score_raw")
    return score.sort_values(["score_year", "crsp_fundno"]).reset_index(drop=True)


def run_predictive_tests(score: pd.DataFrame, panel: pd.DataFrame, out_dir: Path):
    # Next-year outcome alpha.
    yearly_alpha = compute_yearly_outcome_alpha(panel)

    # Score year t predicts year t+1 alpha.
    score2 = score.copy()
    score2["next_year"] = score2["score_year"] + 1
    yearly_alpha2 = yearly_alpha.rename(columns={"year": "next_year"})
    pred = score2.merge(yearly_alpha2, on=["crsp_fundno", "next_year"], how="inner")
    pred = pred.rename(columns={"alpha_next_annual": "future_alpha_annual"})
    pred = pred.dropna(subset=["crisis_score_pct", "future_alpha_annual"]).copy()

    # Regression 1: pooled future alpha on score.
    m1 = smf.ols("future_alpha_annual ~ crisis_score_pct", data=pred).fit(cov_type="HC1")
    # Regression 2: add year fixed effects.
    m2 = smf.ols("future_alpha_annual ~ crisis_score_pct + C(score_year)", data=pred).fit(
        cov_type="HC1"
    )

    reg = pd.DataFrame(
        [
            {
                "model": "pooled",
                "n_obs": int(m1.nobs),
                "beta_score": float(m1.params.get("crisis_score_pct", np.nan)),
                "t_score": float(m1.tvalues.get("crisis_score_pct", np.nan)),
                "p_score": float(m1.pvalues.get("crisis_score_pct", np.nan)),
                "r2": float(m1.rsquared),
            },
            {
                "model": "year_fe",
                "n_obs": int(m2.nobs),
                "beta_score": float(m2.params.get("crisis_score_pct", np.nan)),
                "t_score": float(m2.tvalues.get("crisis_score_pct", np.nan)),
                "p_score": float(m2.pvalues.get("crisis_score_pct", np.nan)),
                "r2": float(m2.rsquared),
            },
        ]
    )

    # Additional validation: does score predict future resilience mechanics?
    yearly_risk = compute_yearly_risk_metrics(panel).rename(columns={"year": "next_year"})
    pred = pred.merge(yearly_risk, on=["crsp_fundno", "next_year"], how="left")
    val_rows = []
    for y_col in ["downside_shift_year", "tail_shift_year"]:
        tmp = pred.dropna(subset=[y_col, "crisis_score_pct"]).copy()
        if len(tmp) < 50:
            continue
        mv1 = smf.ols(f"{y_col} ~ crisis_score_pct", data=tmp).fit(cov_type="HC1")
        mv2 = smf.ols(f"{y_col} ~ crisis_score_pct + C(score_year)", data=tmp).fit(
            cov_type="HC1"
        )
        val_rows.extend(
            [
                {
                    "outcome": y_col,
                    "model": "pooled",
                    "n_obs": int(mv1.nobs),
                    "beta_score": float(mv1.params.get("crisis_score_pct", np.nan)),
                    "t_score": float(mv1.tvalues.get("crisis_score_pct", np.nan)),
                    "p_score": float(mv1.pvalues.get("crisis_score_pct", np.nan)),
                    "r2": float(mv1.rsquared),
                },
                {
                    "outcome": y_col,
                    "model": "year_fe",
                    "n_obs": int(mv2.nobs),
                    "beta_score": float(mv2.params.get("crisis_score_pct", np.nan)),
                    "t_score": float(mv2.tvalues.get("crisis_score_pct", np.nan)),
                    "p_score": float(mv2.pvalues.get("crisis_score_pct", np.nan)),
                    "r2": float(mv2.rsquared),
                },
            ]
        )
    val_reg = pd.DataFrame(val_rows)

    # Quintile sort by score each year.
    qrows = []
    for y, g in pred.groupby("score_year"):
        if g["crisis_score_pct"].nunique() < 5 or len(g) < 15:
            continue
        gg = g.copy()
        gg["q"] = pd.qcut(
            gg["crisis_score_pct"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        )
        for q, h in gg.groupby("q", observed=True):
            qrows.append(
                {
                    "score_year": int(y),
                    "portfolio": f"Q{int(q)}",
                    "n_funds": int(len(h)),
                    "future_alpha_annual": float(h["future_alpha_annual"].mean()),
                }
            )
        # Q5-Q1 spread
        q1 = gg[gg["q"] == 1]["future_alpha_annual"].mean()
        q5 = gg[gg["q"] == 5]["future_alpha_annual"].mean()
        qrows.append(
            {
                "score_year": int(y),
                "portfolio": "Q5-Q1",
                "n_funds": int(min((gg["q"] == 1).sum(), (gg["q"] == 5).sum())),
                "future_alpha_annual": float(q5 - q1),
            }
        )

    qyear = pd.DataFrame(qrows).sort_values(["portfolio", "score_year"])
    qsum = (
        qyear.groupby("portfolio", as_index=False)
        .agg(
            years=("score_year", "nunique"),
            mean_future_alpha_annual=("future_alpha_annual", "mean"),
            median_future_alpha_annual=("future_alpha_annual", "median"),
            std_future_alpha_annual=("future_alpha_annual", "std"),
            pos_share=("future_alpha_annual", lambda s: float(np.mean(s > 0))),
        )
        .sort_values("portfolio")
    )
    qsum["tstat_mean"] = qsum["mean_future_alpha_annual"] / (
        qsum["std_future_alpha_annual"] / np.sqrt(qsum["years"])
    )

    # Save artifacts.
    score.to_csv(out_dir / "fund_year_score.csv", index=False)
    pred.to_csv(out_dir / "predictive_sample.csv", index=False)
    reg.to_csv(out_dir / "predictive_regression.csv", index=False)
    val_reg.to_csv(out_dir / "resilience_validation_regression.csv", index=False)
    qyear.to_csv(out_dir / "quintile_future_alpha_by_year.csv", index=False)
    qsum.to_csv(out_dir / "quintile_summary.csv", index=False)

    return pred, reg, qsum, val_reg


def write_report(
    out_dir: Path,
    score: pd.DataFrame,
    pred: pd.DataFrame,
    reg: pd.DataFrame,
    qsum: pd.DataFrame,
    val_reg: pd.DataFrame,
):
    latest_year = int(score["score_year"].max())
    latest_top = (
        score[score["score_year"] == latest_year]
        .sort_values("crisis_score_pct", ascending=False)
        .head(10)[
            [
                "crsp_fundno",
                "crisis_score_pct",
                "downside_shift",
                "tail_shift",
                "regime_alpha_gap",
                "consistency_raw",
            ]
        ]
    )

    lines = []
    lines.append("# Crisis-Resilience Skill Score: Construction and Validation")
    lines.append("")
    lines.append("## Score Design")
    lines.append("- Built each year using a trailing 3-year window for each fund.")
    lines.append("- Components:")
    lines.append("  - Downside beta shift (negative-market interaction): more negative is better.")
    lines.append("  - Tail beta shift (worst 5% market days): more negative is better.")
    lines.append("  - Regime alpha gap (high-vol alpha minus low-vol alpha): higher is better.")
    lines.append("  - Rolling alpha consistency: higher mean/positive share and lower volatility are better.")
    lines.append("- Each component is z-scored within score-year; final score is the average z-score.")
    lines.append("")

    lines.append("## Predictive Test Setup")
    lines.append("- Outcome: next-year Carhart annualized alpha.")
    lines.append("- Predictor: current-year crisis score percentile.")
    lines.append(f"- Fund-year observations in predictive sample: **{len(pred):,}**")
    lines.append("")

    lines.append("## Predictive Regressions")
    lines.append(reg.to_markdown(index=False))
    lines.append("")

    lines.append("## Mechanism Validation (Future Risk-Exposure Shifts)")
    if val_reg.empty:
        lines.append("- Not enough observations for risk-mechanism validation.")
    else:
        lines.append(val_reg.to_markdown(index=False))
    lines.append("")

    lines.append("## Portfolio Sort Test (By Score Quintiles)")
    lines.append(qsum.to_markdown(index=False))
    lines.append("")

    lines.append(f"## Top Funds by Score in {latest_year}")
    lines.append(latest_top.to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation")
    b = float(reg.loc[reg["model"] == "pooled", "beta_score"].iloc[0])
    p = float(reg.loc[reg["model"] == "pooled", "p_score"].iloc[0])
    spread = float(
        qsum.loc[qsum["portfolio"] == "Q5-Q1", "mean_future_alpha_annual"].iloc[0]
    )

    if b > 0:
        lines.append(
            f"- Score has positive relation with next-year alpha (pooled beta={b:.4f}, p={p:.4f})."
        )
    else:
        lines.append(
            f"- Score has negative relation with next-year alpha (pooled beta={b:.4f}, p={p:.4f})."
        )
        lines.append(
            "- This suggests a resilience-vs-alpha tradeoff: more defensive funds may sacrifice upside alpha."
        )
    lines.append(f"- Q5-Q1 mean future alpha spread: {spread:.4%}.")
    if not val_reg.empty:
        lines.append(
            "- Validation regressions show whether score predicts future defensive beta shifts as intended."
        )

    (out_dir / "crisis_resilience_score_report.md").write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "report" / "crisis_score"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = root / "report" / "results" / "analysis_panel_merged.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing {panel_path}. Run analysis/run_skill_analysis.py first."
        )
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year

    score = build_score(panel)
    pred, reg, qsum, val_reg = run_predictive_tests(score, panel, out_dir)
    write_report(out_dir, score, pred, reg, qsum, val_reg)

    print("Done.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
