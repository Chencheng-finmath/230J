#!/usr/bin/env python3
"""
Mutual fund skill analysis for 50 CRSP funds.

Outputs:
- Alpha model results (CAPM / FF3 / Carhart4)
- Timing model results (Treynor-Mazuy / Henriksson-Merton)
- Persistence analysis (year-ahead quintiles based on prior-year Carhart alpha)
- Markdown summary report
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm


TRADING_DAYS = 252
MIN_OBS_REG = 120
MIN_OBS_PERSIST = 120
MIN_FUNDS_PER_YEAR = 10

FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def bh_qvalues(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def download_french_zip(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="latin1")

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
    if not members:
        raise ValueError(f"No CSV found in zip from {url}")
    text = zf.read(members[0]).decode("latin1")
    cache_path.write_text(text, encoding="latin1")
    return text


def parse_french_daily_csv(text: str, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or len(parts) < 1 + len(value_cols):
            continue
        if len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts[: 1 + len(value_cols)])

    df = pd.DataFrame(rows, columns=["date"] + value_cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # Ken French daily factors are in percent.
    for c in value_cols:
        df[c] = df[c] / 100.0
    return df


def load_factors(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ff3_txt = download_french_zip(FF3_DAILY_URL, cache_dir / "ff3_daily.csv")
    mom_txt = download_french_zip(MOM_DAILY_URL, cache_dir / "mom_daily.csv")

    ff3 = parse_french_daily_csv(ff3_txt, ["mktrf", "smb", "hml", "rf"])
    mom = parse_french_daily_csv(mom_txt, ["mom"])
    factors = ff3.merge(mom, on="date", how="left")
    return factors.sort_values("date").reset_index(drop=True)


def load_funds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["crsp_fundno", "caldt", "dnav", "dret"])
    df["date"] = pd.to_datetime(df["caldt"], errors="coerce")
    df["dnav"] = pd.to_numeric(df["dnav"], errors="coerce")
    df["dret"] = pd.to_numeric(df["dret"], errors="coerce")
    df["crsp_fundno"] = (
        df["crsp_fundno"].astype(str).str.strip().str.lstrip("0").replace("", "0")
    )
    df["crsp_fundno"] = df["crsp_fundno"].astype(int).astype(str).str.zfill(6)
    df = df.dropna(subset=["date", "dret"])
    return df[["crsp_fundno", "date", "dnav", "dret"]].sort_values(
        ["crsp_fundno", "date"]
    )


def fit_ols(frame: pd.DataFrame, y_col: str, x_cols: list[str], min_obs: int = MIN_OBS_REG):
    cols = [y_col] + x_cols
    d = frame[cols].dropna()
    if len(d) < min_obs:
        return None, len(d)
    y = d[y_col]
    x = sm.add_constant(d[x_cols], has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model, len(d)


def run_alpha_models(merged: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "CAPM": ["mktrf"],
        "FF3": ["mktrf", "smb", "hml"],
        "Carhart4": ["mktrf", "smb", "hml", "mom"],
    }
    rows = []
    for fund, g in merged.groupby("crsp_fundno"):
        for model_name, x_cols in specs.items():
            model, n = fit_ols(g, "excess_ret", x_cols)
            if model is None:
                continue
            rows.append(
                {
                    "crsp_fundno": fund,
                    "model": model_name,
                    "n_obs": n,
                    "alpha_daily": model.params.get("const", np.nan),
                    "alpha_annual": model.params.get("const", np.nan) * TRADING_DAYS,
                    "alpha_t": model.tvalues.get("const", np.nan),
                    "alpha_p": model.pvalues.get("const", np.nan),
                    "r2": model.rsquared,
                    "beta_mktrf": model.params.get("mktrf", np.nan),
                    "beta_smb": model.params.get("smb", np.nan),
                    "beta_hml": model.params.get("hml", np.nan),
                    "beta_mom": model.params.get("mom", np.nan),
                }
            )
    out = pd.DataFrame(rows)
    out["alpha_q"] = np.nan
    for model_name, idx in out.groupby("model").groups.items():
        out.loc[idx, "alpha_q"] = bh_qvalues(out.loc[idx, "alpha_p"].values)
    return out.sort_values(["model", "crsp_fundno"]).reset_index(drop=True)


def run_timing_models(merged: pd.DataFrame) -> pd.DataFrame:
    data = merged.copy()
    data["mkt_sq"] = data["mktrf"] ** 2
    data["mkt_pos"] = np.maximum(data["mktrf"], 0.0)

    specs = {
        "TreynorMazuy": {
            "x_cols": ["mktrf", "smb", "hml", "mom", "mkt_sq"],
            "gamma_col": "mkt_sq",
        },
        "HenrikssonMerton": {
            "x_cols": ["mktrf", "smb", "hml", "mom", "mkt_pos"],
            "gamma_col": "mkt_pos",
        },
    }

    rows = []
    for fund, g in data.groupby("crsp_fundno"):
        for model_name, cfg in specs.items():
            model, n = fit_ols(g, "excess_ret", cfg["x_cols"])
            if model is None:
                continue
            gamma_col = cfg["gamma_col"]
            rows.append(
                {
                    "crsp_fundno": fund,
                    "model": model_name,
                    "n_obs": n,
                    "alpha_daily": model.params.get("const", np.nan),
                    "alpha_annual": model.params.get("const", np.nan) * TRADING_DAYS,
                    "alpha_p": model.pvalues.get("const", np.nan),
                    "gamma": model.params.get(gamma_col, np.nan),
                    "gamma_t": model.tvalues.get(gamma_col, np.nan),
                    "gamma_p": model.pvalues.get(gamma_col, np.nan),
                    "r2": model.rsquared,
                }
            )

    out = pd.DataFrame(rows)
    out["gamma_q"] = np.nan
    for model_name, idx in out.groupby("model").groups.items():
        out.loc[idx, "gamma_q"] = bh_qvalues(out.loc[idx, "gamma_p"].values)
    return out.sort_values(["model", "crsp_fundno"]).reset_index(drop=True)


def run_persistence(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = merged.copy()
    data["year"] = data["date"].dt.year
    years = sorted(data["year"].dropna().unique())

    yearly_rows = []
    carhart = ["mktrf", "smb", "hml", "mom"]

    for year in years[1:]:
        prev = year - 1
        prev_df = data[data["year"] == prev]
        cur_df = data[data["year"] == year]
        if prev_df.empty or cur_df.empty:
            continue

        prev_count = prev_df.groupby("crsp_fundno").size()
        cur_count = cur_df.groupby("crsp_fundno").size()
        eligible = set(prev_count[prev_count >= MIN_OBS_PERSIST].index).intersection(
            cur_count[cur_count >= MIN_OBS_PERSIST].index
        )

        if len(eligible) < MIN_FUNDS_PER_YEAR:
            continue

        prev_alpha = []
        for fund in sorted(eligible):
            g = prev_df[prev_df["crsp_fundno"] == fund]
            m, n = fit_ols(g, "excess_ret", carhart, min_obs=MIN_OBS_PERSIST)
            if m is None:
                continue
            prev_alpha.append({"crsp_fundno": fund, "alpha_prev": m.params["const"]})

        rank_df = pd.DataFrame(prev_alpha)
        if len(rank_df) < MIN_FUNDS_PER_YEAR:
            continue

        try:
            rank_df["quintile"] = pd.qcut(
                rank_df["alpha_prev"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )
        except ValueError:
            continue

        if rank_df["quintile"].nunique() < 3:
            continue

        q_portfolios: dict[int, pd.DataFrame] = {}
        factor_year = (
            cur_df[["date", "mktrf", "smb", "hml", "mom"]]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )

        for q in sorted(rank_df["quintile"].dropna().unique()):
            q_int = int(q)
            funds_q = rank_df.loc[rank_df["quintile"] == q, "crsp_fundno"].tolist()
            p = (
                cur_df[cur_df["crsp_fundno"].isin(funds_q)]
                .groupby("date", as_index=False)["excess_ret"]
                .mean()
                .sort_values("date")
            )
            p = p.merge(factor_year, on="date", how="inner")
            q_portfolios[q_int] = p
            m, n = fit_ols(p, "excess_ret", carhart, min_obs=MIN_OBS_PERSIST)
            if m is None:
                continue
            yearly_rows.append(
                {
                    "year": year,
                    "portfolio": f"Q{q_int}",
                    "n_funds": len(funds_q),
                    "n_obs": n,
                    "alpha_daily": m.params["const"],
                    "alpha_annual": m.params["const"] * TRADING_DAYS,
                    "alpha_t": m.tvalues["const"],
                    "alpha_p": m.pvalues["const"],
                }
            )

        if 1 in q_portfolios and 5 in q_portfolios:
            spread = q_portfolios[5][["date", "excess_ret"]].merge(
                q_portfolios[1][["date", "excess_ret"]],
                on="date",
                how="inner",
                suffixes=("_q5", "_q1"),
            )
            spread["excess_ret"] = spread["excess_ret_q5"] - spread["excess_ret_q1"]
            spread = spread[["date", "excess_ret"]].merge(factor_year, on="date", how="inner")
            m, n = fit_ols(spread, "excess_ret", carhart, min_obs=MIN_OBS_PERSIST)
            if m is not None:
                yearly_rows.append(
                    {
                        "year": year,
                        "portfolio": "Q5-Q1",
                        "n_funds": int(rank_df["quintile"].value_counts().min()),
                        "n_obs": n,
                        "alpha_daily": m.params["const"],
                        "alpha_annual": m.params["const"] * TRADING_DAYS,
                        "alpha_t": m.tvalues["const"],
                        "alpha_p": m.pvalues["const"],
                    }
                )

    yearly = pd.DataFrame(yearly_rows).sort_values(["portfolio", "year"]).reset_index(drop=True)
    if yearly.empty:
        return yearly, pd.DataFrame()

    summary = (
        yearly.groupby("portfolio", as_index=False)
        .agg(
            years=("year", "nunique"),
            mean_alpha_annual=("alpha_annual", "mean"),
            median_alpha_annual=("alpha_annual", "median"),
            std_alpha_annual=("alpha_annual", "std"),
            pos_alpha_share=("alpha_annual", lambda s: np.mean(s > 0)),
            sig_alpha_share=("alpha_p", lambda s: np.mean(s < 0.05)),
        )
        .sort_values("portfolio")
    )
    return yearly, summary


def build_summary_tables(alpha: pd.DataFrame, timing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_summary = (
        alpha.groupby("model", as_index=False)
        .agg(
            funds=("crsp_fundno", "nunique"),
            mean_alpha_annual=("alpha_annual", "mean"),
            median_alpha_annual=("alpha_annual", "median"),
            pos_alpha_share=("alpha_annual", lambda s: np.mean(s > 0)),
            sig_pos_alpha_share=("alpha_p", lambda s: np.mean((alpha.loc[s.index, "alpha_annual"] > 0) & (s < 0.05))),
            sig_neg_alpha_share=("alpha_p", lambda s: np.mean((alpha.loc[s.index, "alpha_annual"] < 0) & (s < 0.05))),
            fdr_pos_alpha_share=("alpha_q", lambda s: np.mean((alpha.loc[s.index, "alpha_annual"] > 0) & (s < 0.10))),
        )
        .sort_values("model")
    )

    timing_summary = (
        timing.groupby("model", as_index=False)
        .agg(
            funds=("crsp_fundno", "nunique"),
            mean_gamma=("gamma", "mean"),
            median_gamma=("gamma", "median"),
            sig_pos_gamma_share=("gamma_p", lambda s: np.mean((timing.loc[s.index, "gamma"] > 0) & (s < 0.05))),
            sig_neg_gamma_share=("gamma_p", lambda s: np.mean((timing.loc[s.index, "gamma"] < 0) & (s < 0.05))),
            fdr_pos_gamma_share=("gamma_q", lambda s: np.mean((timing.loc[s.index, "gamma"] > 0) & (s < 0.10))),
        )
        .sort_values("model")
    )
    return alpha_summary, timing_summary


def save_report(
    report_path: Path,
    merged: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    timing_summary: pd.DataFrame,
    persistence_summary: pd.DataFrame,
):
    start = merged["date"].min().date()
    end = merged["date"].max().date()
    n_funds = merged["crsp_fundno"].nunique()
    n_rows = len(merged)

    lines = []
    lines.append("# Mutual Fund Skill Analysis (50 Funds)")
    lines.append("")
    lines.append("## Data Window")
    lines.append(f"- Funds: **{n_funds}**")
    lines.append(f"- Observations used after factor merge: **{n_rows:,}**")
    lines.append(f"- Date range: **{start} to {end}**")
    lines.append("")

    lines.append("## Stock Selection (Alpha)")
    lines.append(alpha_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Market Timing")
    lines.append(timing_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Persistence (Year-ahead Quintiles from Prior-year Carhart Alpha)")
    if persistence_summary.empty:
        lines.append("- Not enough valid yearly windows for persistence test under current thresholds.")
    else:
        lines.append(persistence_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation Notes")
    lines.append("- Alpha columns are annualized from daily intercepts (multiply by 252).")
    lines.append("- Inference uses Newey-West HAC standard errors (lag=5).")
    lines.append("- `fdr_*` shares use Benjamini-Hochberg q-values at 10%.")
    lines.append("- Timing models include Carhart controls plus timing terms.")
    report_path.write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "all_data_clean.csv"
    out_dir = root / "report" / "results"
    cache_dir = out_dir / "factor_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    funds = load_funds(data_path)
    factors = load_factors(cache_dir)
    merged = funds.merge(factors, on="date", how="inner")
    merged["excess_ret"] = merged["dret"] - merged["rf"]
    merged = merged.dropna(subset=["excess_ret", "mktrf", "smb", "hml", "mom"]).copy()

    alpha = run_alpha_models(merged)
    timing = run_timing_models(merged)
    persistence_yearly, persistence_summary = run_persistence(merged)
    alpha_summary, timing_summary = build_summary_tables(alpha, timing)

    alpha.to_csv(out_dir / "alpha_results_by_fund.csv", index=False)
    timing.to_csv(out_dir / "timing_results_by_fund.csv", index=False)
    alpha_summary.to_csv(out_dir / "alpha_summary.csv", index=False)
    timing_summary.to_csv(out_dir / "timing_summary.csv", index=False)
    persistence_yearly.to_csv(out_dir / "persistence_yearly.csv", index=False)
    persistence_summary.to_csv(out_dir / "persistence_summary.csv", index=False)
    merged[["crsp_fundno", "date", "dret", "rf", "excess_ret", "mktrf", "smb", "hml", "mom"]].to_csv(
        out_dir / "analysis_panel_merged.csv", index=False
    )

    # Leaderboards for quick reading.
    carhart = alpha[alpha["model"] == "Carhart4"].copy()
    carhart.sort_values("alpha_annual", ascending=False).head(10).to_csv(
        out_dir / "top10_carhart_alpha.csv", index=False
    )
    carhart.sort_values("alpha_annual", ascending=True).head(10).to_csv(
        out_dir / "bottom10_carhart_alpha.csv", index=False
    )

    report_path = out_dir / "empirical_results_summary.md"
    save_report(report_path, merged, alpha_summary, timing_summary, persistence_summary)

    print("Done.")
    print(f"Output directory: {out_dir}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
