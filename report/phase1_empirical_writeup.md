# Phase 1 Empirical Results (50 U.S. Mutual Funds)

## Scope
This phase implements the core tests in the proposal on the cleaned panel of 50 CRSP mutual funds (`all_data_clean.csv`) from 1998-09-01 to 2025-12-31, merged with daily Fama-French factors and momentum.

## Methods Implemented
- Stock-selection alpha models:
  - CAPM
  - Fama-French 3-factor (FF3)
  - Carhart 4-factor
- Market-timing models:
  - Treynor-Mazuy (quadratic market term)
  - Henriksson-Merton (up-market interaction term)
- Inference:
  - Newey-West HAC standard errors (lag = 5)
  - Benjamini-Hochberg FDR adjustment (q-values)
- Persistence:
  - For each year, rank funds by prior-year Carhart alpha into quintiles
  - Evaluate next-year quintile portfolio alpha
  - Report Q5-Q1 spread alpha

## Main Findings

### 1) Stock Selection (Alpha)
- Mean annualized alpha is small but positive across models (about 0.67% to 0.70%).
- A nontrivial fraction of funds shows nominally significant positive alpha at 5%:
  - CAPM: 14%
  - FF3: 16%
  - Carhart4: 16%
- After FDR correction, the share of robust positive-alpha funds falls materially:
  - CAPM: 2%
  - FF3: 8%
  - Carhart4: 4%

Interpretation: evidence of broad, strong stock-selection skill is limited once multiple testing is controlled.

### 2) Market Timing
- Treynor-Mazuy gamma is negative on average; significant negative gamma appears more often than significant positive gamma.
- Henriksson-Merton results are similar: weak evidence for positive timing and some evidence of adverse timing.

Interpretation: this sample does not support strong, widespread market-timing ability.

### 3) Persistence
- Year-ahead quintile alphas are noisy.
- Top-minus-bottom spread (Q5-Q1) has only a small average annualized alpha (~0.21%) and low significance frequency.

Interpretation: prior alpha has weak out-of-sample persistence in this 50-fund universe.

## Deliverables Produced
- Full by-fund alpha results: `report/results/alpha_results_by_fund.csv`
- Alpha summary table: `report/results/alpha_summary.csv`
- Full by-fund timing results: `report/results/timing_results_by_fund.csv`
- Timing summary table: `report/results/timing_summary.csv`
- Persistence yearly results: `report/results/persistence_yearly.csv`
- Persistence summary table: `report/results/persistence_summary.csv`
- Auto-generated summary report: `report/results/empirical_results_summary.md`

## Caveats and Next Steps
- This is a focused 50-fund sample, not the entire CRSP universe.
- Net-of-fee vs gross-of-fee distinctions and share-class aggregation can be expanded in the next phase.
- Next phase should add:
  - Bootstrap skill-vs-luck decomposition
  - Alternative factor sets (e.g., FF5)
  - Explicit out-of-sample rolling windows (1y/3y training, next-period testing)
