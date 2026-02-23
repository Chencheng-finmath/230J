# all_data.csv Quality & Structure Analysis

File: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/all_data.csv`

## 1) Basic Structure
- Rows: **292,994**
- Columns: **4**
- Columns present: `crsp_fundno`, `caldt`, `dnav`, `dret`
- Unique funds: **50**
- Date range: **1998-09-01 to 2025-12-31**
- Unique trading dates: **6,876**

## 2) Consistency vs Target Fund List
- Code list compared against: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/fund_codes_50_crsp_fundno.txt`
- In expected list: **50 / 50**
- Extra funds not in list: **0**
- Missing expected funds: **0**

## 3) Data Quality Checks
- Duplicate `(crsp_fundno, caldt)` rows: **0**
- Missing `dnav`: **16 rows**
- Non-numeric `dret`: **37 rows** (marker value appears as `R`)
- Most funds have one `R` row; one fund (`093577`) has 15 such rows.

## 4) Panel Coverage (Unbalanced Panel)
- Observations per fund:
  - Min: **742**
  - 25%: **5,395**
  - Median: **6,876**
  - 75%: **6,876**
  - Max: **6,876**
- Funds with full-history length (6,876 obs): **29**
- Shortest history fund: `100948` (2023-01-18 to 2025-12-31)

## 5) Return Distribution (after coercing `dret` to numeric and dropping non-numeric rows)
- Clean rows: **292,957**
- Mean daily return: **0.000432**
- Std daily return: **0.012111**
- 1% quantile: **-0.035238**
- 99% quantile: **0.033658**
- Min: **-0.138772**
- Max: **0.318126**

Cross-sectional daily mean return (across funds):
- Mean: **0.000427**
- Std: **0.010395**
- Worst day: **2020-03-16** (about **-10.27%**)
- Best day: **2008-10-13** (about **+9.73%**)

## 6) Interpretation
- The file is structurally correct and matches the intended 50-fund universe.
- It is usable for empirical work after standard cleaning:
  1. Parse `dret` with numeric coercion; treat `R` as missing.
  2. Drop rows where `dret` is missing; optionally keep `dnav`-missing rows only if not using NAV-level analysis.
  3. Decide whether to use an unbalanced panel (all funds) or a common sample window for comparability.

## 7) Recommended sample choices
- If your goal is maximum history: keep unbalanced panel and control for fund fixed effects / varying sample length.
- If your goal is comparability across funds: use a common start date (e.g., 2018+) where active-fund count is near full coverage.
