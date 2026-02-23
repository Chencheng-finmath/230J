# Why We Chose These 50 Mutual Funds

## Research Objective
Our project asks a practical question: **Do popular U.S. mutual funds actually have stock-selection and market-timing skill, or are investors mostly paying for factor exposure and luck?**

To answer that question, we intentionally built a fund basket that reflects what real investors actually buy and discuss, rather than selecting obscure funds.

## Selection Logic
We used a three-layer approach:

1. **Publicly recognized "popular" fund lists**
   - We started from investor-facing lists that summarize widely followed mutual funds.
   - Core source list: Kiplinger 25.

2. **Large flagship active funds from major families**
   - We added high-visibility flagship funds (for example, major Capital Group and other well-known active funds) so the sample reflects institutional and retail attention.

3. **Style and benchmark coverage**
   - We retained diversity across growth, value, blend, dividend/income, and broad-market exposure.
   - This allows us to test skill while controlling for style/factor effects and to compare active funds against low-cost benchmark-like funds.

## Why This Story Is Strong
This design makes the empirical result easy to interpret:

- If even this "popular" set fails to deliver persistent alpha after fees and factor controls, that is strong evidence against broad active skill.
- If only a small subset survives out-of-sample and multiple-testing corrections, we can identify where true skill may exist.

In other words, the sample is built to answer an investor-relevant question, not just a statistical exercise.

## Final Data-Ready Basket
- Final basket size: **50 funds**
- WRDS-ready identifier for extraction/joins: **CRSP fund number (`crsp_fundno`)**
- Upload file (one code per line):
  - `/Users/chenchengliu/Desktop/Python_MFE_Course/230J/fund_codes_50_crsp_fundno.txt`

## References
- Kiplinger 25: https://www.kiplinger.com/investing/mutual-funds/the-kiplinger-25
- Kiplinger low-fee mutual funds list: https://www.kiplinger.com/investing/mutual-funds/best-low-fee-mutual-funds-you-can-buy
- Capital Group flagship fund pages (examples):
  - AGTHX: https://www.capitalgroup.com/individual/investments/fund/agthx
  - AWSHX: https://www.capitalgroup.com/individual/investments/fund/awshx
  - ANCFX: https://www.capitalgroup.com/individual/investments/fund/ancfx
  - AMCPX: https://www.capitalgroup.com/individual/investments/fund/amcpx
- Morningstar Active/Passive Barometer: https://www.morningstar.com/lp/active-passive-barometer
- S&P SPIVA: https://www.spglobal.com/spdji/en/research-insights/spiva/
