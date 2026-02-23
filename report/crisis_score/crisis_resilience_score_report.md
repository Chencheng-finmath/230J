# Crisis-Resilience Skill Score: Construction and Validation

## Score Design
- Built each year using a trailing 3-year window for each fund.
- Components:
  - Downside beta shift (negative-market interaction): more negative is better.
  - Tail beta shift (worst 5% market days): more negative is better.
  - Regime alpha gap (high-vol alpha minus low-vol alpha): higher is better.
  - Rolling alpha consistency: higher mean/positive share and lower volatility are better.
- Each component is z-scored within score-year; final score is the average z-score.

## Predictive Test Setup
- Outcome: next-year Carhart annualized alpha.
- Predictor: current-year crisis score percentile.
- Fund-year observations in predictive sample: **1,043**

## Predictive Regressions
| model   |   n_obs |   beta_score |   t_score |   p_score |         r2 |
|:--------|--------:|-------------:|----------:|----------:|-----------:|
| pooled  |    1043 |   -0.0145119 |  -1.79551 | 0.0725726 | 0.00442899 |
| year_fe |    1043 |   -0.0145347 |  -1.91646 | 0.055306  | 0.124515   |

## Mechanism Validation (Future Risk-Exposure Shifts)
| outcome             | model   |   n_obs |   beta_score |   t_score |     p_score |        r2 |
|:--------------------|:--------|--------:|-------------:|----------:|------------:|----------:|
| downside_shift_year | pooled  |    1043 |   -0.0347382 |  -3.67229 | 0.000240386 | 0.0183991 |
| downside_shift_year | year_fe |    1043 |   -0.0347247 |  -3.83675 | 0.000124674 | 0.115679  |
| tail_shift_year     | pooled  |    1043 |   -0.0369703 |  -4.33557 | 1.4538e-05  | 0.0249125 |
| tail_shift_year     | year_fe |    1043 |   -0.0369215 |  -4.47277 | 7.72137e-06 | 0.112656  |

## Portfolio Sort Test (By Score Quintiles)
| portfolio   |   years |   mean_future_alpha_annual |   median_future_alpha_annual |   std_future_alpha_annual |   pos_share |   tstat_mean |
|:------------|--------:|---------------------------:|-----------------------------:|--------------------------:|------------:|-------------:|
| Q1          |      25 |                 0.011826   |                   0.0120289  |                 0.0434155 |        0.68 |     1.36196  |
| Q2          |      25 |                -0.00214077 |                  -0.00522849 |                 0.0256746 |        0.4  |    -0.416905 |
| Q3          |      25 |                 0.00589864 |                   0.00403949 |                 0.01926   |        0.56 |     1.53132  |
| Q4          |      25 |                 0.0027405  |                   0.00490977 |                 0.0270776 |        0.56 |     0.506047 |
| Q5          |      25 |                -0.00478656 |                  -0.0046001  |                 0.0380956 |        0.4  |    -0.62823  |
| Q5-Q1       |      25 |                -0.0166126  |                  -0.0167429  |                 0.048188  |        0.32 |    -1.72373  |

## Top Funds by Score in 2025
|   crsp_fundno |   crisis_score_pct |   downside_shift |   tail_shift |   regime_alpha_gap |   consistency_raw |
|--------------:|-------------------:|-----------------:|-------------:|-------------------:|------------------:|
|         64961 |               1    |      -0.0396607  |   -0.0987428 |        0.377689    |       -0.00926921 |
|         36540 |               0.98 |      -0.0577065  |   -0.0979275 |        0.216237    |       -0          |
|          5846 |               0.96 |      -0.036182   |   -0.0849458 |        0.0904425   |        0.038021   |
|         11809 |               0.94 |      -0.0411789  |   -0.0192754 |        0.0247586   |        1.11635    |
|         21120 |               0.92 |       0.0469463  |   -0.0326832 |        0.251691    |        0.650069   |
|         31302 |               0.9  |       0.0232557  |   -0.0388107 |        0.0523102   |        1.05442    |
|         31304 |               0.88 |      -0.00560724 |   -0.0413562 |        0.0338129   |        0.673266   |
|         14031 |               0.86 |      -0.00107061 |   -0.0151429 |        0.0785106   |        0.788066   |
|         87002 |               0.84 |       0.0666931  |   -0.0332613 |        0.0359851   |        1.45756    |
|         26985 |               0.82 |      -0.0366718  |   -0.0325298 |        0.000997521 |        0.403492   |

## Interpretation
- Score has negative relation with next-year alpha (pooled beta=-0.0145, p=0.0726).
- This suggests a resilience-vs-alpha tradeoff: more defensive funds may sacrifice upside alpha.
- Q5-Q1 mean future alpha spread: -1.6613%.
- Validation regressions show whether score predicts future defensive beta shifts as intended.