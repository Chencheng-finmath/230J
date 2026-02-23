# Data Cleaning Report for all_data.csv

## Files Generated
- Cleaned data: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/all_data_clean.csv`
- Fund date ranges: `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/report/fund_50_date_ranges.csv`

## Cleaning Rules Applied
1. Kept only required columns: `crsp_fundno`, `caldt`, `dnav`, `dret`.
2. Parsed `caldt` as date; dropped rows with invalid date.
3. Parsed `dret` and `dnav` as numeric; dropped rows where `dret` is non-numeric (e.g., marker `R`) or missing.
4. Dropped rows with missing/non-numeric `dnav`.
5. Checked duplicate `(crsp_fundno, caldt)` rows and removed duplicates if present (none removed in this run).
6. Standardized `crsp_fundno` to 6-digit zero-padded format and sorted by fund/date.

## Row Counts
- Raw rows: 292,994
- After valid fund/date filter: 292,994
- After numeric `dret` filter: 292,957
- After numeric `dnav` filter: 292,957
- Duplicate rows removed: 0
- Final cleaned rows: 292,957

## Fund Coverage
- Unique funds in raw data: 50
- Unique funds in cleaned data: 50
- Duplicate `(fund,date)` pairs in cleaned data: 0

## Notes
- This is a strict clean intended for return/NAV analysis.
- The date-range CSV includes both raw and cleaned start/end dates per fund.
