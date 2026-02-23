# Figure Guide for Final Report

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
