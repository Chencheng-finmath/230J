# Mutual Fund Skill Analysis (50 Funds)

## Data Window
- Funds: **50**
- Observations used after factor merge: **292,957**
- Date range: **1998-09-01 to 2025-12-31**

## Stock Selection (Alpha)
| model    |   funds |   mean_alpha_annual |   median_alpha_annual |   pos_alpha_share |   sig_pos_alpha_share |   sig_neg_alpha_share |   fdr_pos_alpha_share |
|:---------|--------:|--------------------:|----------------------:|------------------:|----------------------:|----------------------:|----------------------:|
| CAPM     |      50 |          0.00700031 |            0.00958011 |              0.76 |                  0.14 |                  0.04 |                  0.02 |
| Carhart4 |      50 |          0.0070346  |            0.00844591 |              0.68 |                  0.16 |                  0.02 |                  0.04 |
| FF3      |      50 |          0.00666957 |            0.00825649 |              0.68 |                  0.16 |                  0.06 |                  0.08 |

## Market Timing
| model            |   funds |   mean_gamma |   median_gamma |   sig_pos_gamma_share |   sig_neg_gamma_share |   fdr_pos_gamma_share |
|:-----------------|--------:|-------------:|---------------:|----------------------:|----------------------:|----------------------:|
| HenrikssonMerton |      50 |   -0.0149019 |     -0.0100729 |                  0.08 |                  0.12 |                  0.06 |
| TreynorMazuy     |      50 |   -0.210687  |     -0.161219  |                  0.04 |                  0.2  |                  0.04 |

## Persistence (Year-ahead Quintiles from Prior-year Carhart Alpha)
| portfolio   |   years |   mean_alpha_annual |   median_alpha_annual |   std_alpha_annual |   pos_alpha_share |   sig_alpha_share |
|:------------|--------:|--------------------:|----------------------:|-------------------:|------------------:|------------------:|
| Q1          |      26 |          0.0102302  |           -0.0024609  |          0.0610642 |          0.423077 |         0.192308  |
| Q2          |      26 |          0.00950204 |            0.0049115  |          0.0258103 |          0.615385 |         0.115385  |
| Q3          |      26 |         -0.00423232 |           -0.00678658 |          0.0275625 |          0.423077 |         0.153846  |
| Q4          |      26 |          0.0040166  |           -0.00198027 |          0.0339696 |          0.461538 |         0.153846  |
| Q5          |      26 |          0.0122805  |            0.0128095  |          0.04593   |          0.615385 |         0.153846  |
| Q5-Q1       |      26 |          0.00205033 |           -0.0019146  |          0.0616814 |          0.5      |         0.0769231 |

## Interpretation Notes
- Alpha columns are annualized from daily intercepts (multiply by 252).
- Inference uses Newey-West HAC standard errors (lag=5).
- `fdr_*` shares use Benjamini-Hochberg q-values at 10%.
- Timing models include Carhart controls plus timing terms.