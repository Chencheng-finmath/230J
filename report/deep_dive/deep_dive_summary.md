# Deep-Dive Findings: Novel Directions

## Sample
- Funds: **50**
- Date range: **1998-09-01 to 2025-12-31**
- Rows: **292,957**

## 1) Regime Alpha (High-vol vs Low-vol)
- Mean (high - low) annual alpha: **0.5570%**
- Median (high - low) annual alpha: **0.4155%**
- Share with worse high-vol alpha (diff < 0): **44.90%**

## 2) Downside Timing (Negative-Market Beta Shift)
- Mean downside beta shift: **0.0149**
- Median downside beta shift: **0.0101**
- Share with defensive shift (beta_down < beta_up): **30.00%**
- Share with statistically significant defensive shift: **8.00%**

## 3) Tail Beta Shift (Worst 5% Market Days)
- Mean tail beta shift: **0.0131**
- Median tail beta shift: **0.0111**
- Share with lower beta in tail days: **28.00%**
- Share with significant tail defense: **12.00%**

## 4) Rolling 3Y Alpha Consistency
- Mean positive-alpha window share: **52.01%**
- Median positive-alpha window share: **55.10%**

Top 10 consistency-score funds:
|   crsp_fundno |   windows |   alpha_mean_annual |   alpha_median_annual |   alpha_std_annual |   alpha_positive_share |   alpha_negative_share |   consistency_score |
|--------------:|----------:|--------------------:|----------------------:|-------------------:|-----------------------:|-----------------------:|--------------------:|
|         52819 |        46 |           0.0226303 |             0.0265421 |          0.0115416 |               0.978261 |              0.0217391 |            1.91797  |
|         31363 |        75 |           0.0266511 |             0.0331975 |          0.0271318 |               0.813333 |              0.186667  |            0.798893 |
|         31235 |        98 |           0.0129096 |             0.0157714 |          0.0157028 |               0.826531 |              0.173469  |            0.679464 |
|         31304 |        98 |           0.0153709 |             0.0180123 |          0.0195827 |               0.795918 |              0.204082  |            0.6247   |
|         12009 |        98 |           0.018651  |             0.023555  |          0.0242392 |               0.714286 |              0.285714  |            0.549589 |
|          5846 |        89 |           0.022466  |             0.0307293 |          0.0392208 |               0.719101 |              0.280899  |            0.411896 |
|         31302 |        98 |           0.0252207 |             0.0273185 |          0.0458568 |               0.72449  |              0.27551   |            0.398452 |
|         12017 |        98 |           0.0333497 |             0.0278092 |          0.0683574 |               0.72449  |              0.27551   |            0.353454 |
|         26998 |        98 |           0.0185886 |             0.0230328 |          0.039441  |               0.744898 |              0.255102  |            0.351062 |
|         11974 |        98 |           0.0173219 |             0.0126566 |          0.0335556 |               0.673469 |              0.326531  |            0.347645 |