# Do U.S. Mutual Funds Have Stock-Selection and Market-Timing Skill?

## 1. Executive Summary
This project studies whether a basket of 50 popular U.S. mutual funds shows genuine stock-selection skill, market-timing skill, and persistence.

Main conclusions:
- Broad, robust stock-selection skill is limited after multiple-testing control.
- Market-timing ability is weak on average; negative timing signals appear more frequently than positive timing signals.
- Out-of-sample persistence is weak.
- A newly designed Crisis-Resilience Score strongly predicts future defensive risk behavior, but does **not** predict higher future alpha; instead, it points to a resilience-vs-alpha tradeoff.

## 2. Research Workflow
The work was completed in five stages.

### Stage A: Fund universe design
- Goal: keep the study investor-relevant and computationally manageable.
- Final universe: 50 CRSP funds (`crsp_fundno`) representing popular/high-visibility mutual funds.
- Universe file: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/fund_codes_50_crsp_fundno.txt`
- Story report: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/fund_selection_story_report.md`

### Stage B: Data cleaning and panel construction
- Raw file: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/all_data.csv`
- Clean file: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/all_data_clean.csv`
- Cleaning report: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/data_cleaning_report.md`
- Fund date ranges: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/fund_50_date_ranges.csv`

Cleaning rules:
- Keep columns: `crsp_fundno`, `caldt`, `dnav`, `dret`.
- Parse date and numeric fields.
- Drop non-numeric/missing `dret` rows (e.g., marker `R`).
- Drop missing/non-numeric `dnav` rows.
- Standardize fund IDs to zero-padded 6 digits.
- Check and remove duplicate `(fund, date)` rows (none removed).

Key cleaning stats:
- Raw rows: 292,994
- Clean rows: 292,957
- Funds: 50
- Date range: 1998-09-01 to 2025-12-31

### Stage C: Baseline empirical tests (proposal core)
Script:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_skill_analysis.py`

Factor data:
- Daily Fama-French factors + Momentum from Ken French library (auto-downloaded and cached).

Models:
- Stock selection: CAPM, FF3, Carhart4.
- Market timing: Treynor-Mazuy, Henriksson-Merton.
- Persistence: prior-year Carhart alpha quintiles predicting next-year alpha.
- Inference: Newey-West HAC (lag=5).
- Multiple testing: Benjamini-Hochberg FDR (q-values).

Outputs:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/results/`
- Main summary: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/results/empirical_results_summary.md`

### Stage D: Deep-dive analysis (novel extensions)
Script:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_deep_dive_analysis.py`

New angles:
- Regime alpha: high-volatility vs low-volatility markets.
- Downside timing: beta shift on negative market days.
- Tail defense: beta shift on worst 5% market days.
- 3-year rolling alpha consistency.

Outputs:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/deep_dive/`
- Summary: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/deep_dive/deep_dive_summary.md`

### Stage E: Crisis-Resilience Score and predictive validation
Script:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_crisis_score.py`

Score design (formed each year using prior 3 years, no look-ahead):
- Downside beta shift component (more negative is better)
- Tail beta shift component (more negative is better)
- Regime alpha gap component (high-vol alpha minus low-vol alpha, higher is better)
- Rolling alpha consistency component (higher mean/positive share and lower volatility)
- Year-wise z-score normalization, equal-weight average score

Outputs:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/crisis_score/`
- Main score report: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/crisis_score/crisis_resilience_score_report.md`

## 3. Baseline Results (Stage C)
Data used in merged panel:
- 50 funds
- 292,957 observations
- 1998-09-01 to 2025-12-31

### 3.1 Stock selection (alpha)
Summary from CAPM / FF3 / Carhart4:
- Mean annual alpha is small positive: about 0.67% to 0.70%.
- Nominally significant positive alpha share: about 14% to 16%.
- After FDR control, robust positive-alpha share drops to 2% to 8%.

Interpretation:
- Some funds show positive alpha, but broad “manager skill” evidence becomes much weaker once multiple testing is accounted for.

### 3.2 Market timing
- Treynor-Mazuy mean gamma is negative.
- Henriksson-Merton mean gamma is negative.
- Significant negative timing signals are more common than significant positive ones.

Interpretation:
- The sample does not support strong, widespread market-timing skill.

### 3.3 Persistence
- Prior-year alpha ranking has weak out-of-sample power.
- Q5-Q1 next-year alpha spread is small (~0.205% annualized average) and unstable.

Interpretation:
- Persistence exists at best weakly and is not robust across years.

## 4. Deep-Dive Results (Stage D)
### 4.1 Regime alpha
- Mean (high-vol minus low-vol) alpha: +0.5570% annual.
- Median: +0.4155% annual.
- 44.9% of funds perform worse in high-vol regime.

Interpretation:
- High-vol periods are heterogeneous: some funds adapt well, many do not.

### 4.2 Downside and tail beta behavior
- Mean downside beta shift: +0.0149.
- Mean tail (worst-5%) beta shift: +0.0131.
- Only 30% show defensive downside shift; only 28% show lower beta in tail days.

Interpretation:
- Most funds do not systematically reduce market exposure when markets are bad.

### 4.3 Rolling consistency
- Mean share of positive alpha windows: 52.01%.
- Median share: 55.10%.

Interpretation:
- Truly stable positive alpha appears only in a minority of funds.

## 5. Crisis-Resilience Score Results (Stage E)
Predictive sample:
- 1,043 fund-year observations

### 5.1 Does score predict next-year alpha?
Regression (`future_alpha_annual ~ score`):
- Pooled beta: -0.0145, p=0.0726
- Year-FE beta: -0.0145, p=0.0553

Quintile sort (by score):
- Q5-Q1 mean future alpha: -1.6613% annual

Interpretation:
- Higher crisis-resilience score does not predict higher next-year alpha.
- It suggests a resilience-vs-upside tradeoff in this sample.

### 5.2 Does score predict future defensive behavior (mechanism check)?
Regression outcomes:
- `future_downside_shift ~ score`: beta=-0.0347, p=0.00024
- `future_tail_shift ~ score`: beta=-0.0370, p=1.45e-05

Interpretation:
- Score works as intended for risk behavior: higher score predicts more defensive future beta shifts.

## 6. Integrated Conclusion
Across the full pipeline, the evidence supports three core conclusions:
- Broad and persistent stock-selection skill is limited after robust inference.
- Average market-timing skill is weak and often adverse.
- Defensive resilience is measurable and persistent in risk behavior, but not rewarded with higher subsequent alpha in this sample.

In practical terms:
- Investors can use resilience metrics for downside-risk management.
- But resilience score should not be interpreted as an alpha-forecasting signal without additional conditioning.

## 7. Reproducibility
Run in this order:

```bash
python3 /Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_skill_analysis.py
python3 /Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_deep_dive_analysis.py
python3 /Users/chenchengliu/Desktop/Python_MFE_Course/230J/analysis/run_crisis_score.py
```

All output folders:
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/results`
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/deep_dive`
- `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/crisis_score`

## 8. Limitations and Next Research Steps
Limitations:
- Universe is 50 funds, not full CRSP universe.
- Fee decomposition, share-class aggregation, and flow interaction are not fully modeled yet.
- Tail/regime definitions can be expanded to macro state variables.

Next steps:
- Add bootstrap skill-vs-luck decomposition (fund-level and cross-sectional).
- Build out-of-sample manager-selection strategy using resilience and alpha jointly.
- Test conditional alpha models with macro/liquidity state variables.
- Extend to full WRDS universe with share-class consolidation.
