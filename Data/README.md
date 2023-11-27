# Data Documentation

## Input Data
- File Name: all_inputs_cleaned.csv
- Number of columns (features): 15 (excluding Month)
- Number of rows (data points): 204
	- Monthly data from June 2006 to September 2023 (inclusive)

## Features

All data is in (percent (%) change)/100. For instance, 20% is expressed as 0.2.
1. CPI: Consumer Price Index ([source](https://fred.stlouisfed.org/series/CPIAUCSL)) (Seasonally Adjusted (SA): Yes (Y))
2. InterestRate: Effective Federal Funds Rate (EFFR) ([source](https://fred.stlouisfed.org/series/FEDFUNDS)) (SA: N)
3. GDP: Gross Domestic Product ([source](https://fred.stlouisfed.org/series/GDP)) (SA: Y)
4. ValAddConst: Value Added by the Construction Industry ([source](https://fred.stlouisfed.org/series/VAC)) (SA: Y)
5. ValAddInfo: Value Added by the Information Industry ([source](https://fred.stlouisfed.org/series/VAI)) (SA: Y)
6. Borrowing: Total Borrowings from all commercial banks ([source](https://fred.stlouisfed.org/series/H8B3094NCBA)) (SA: Y)
7. CommercialLoan: Commercial and Industrial Loans ([source](https://fred.stlouisfed.org/series/H8B1023NCBCMG)) (SA: Y)
8. ConsumerLoan: Consumer Loans ([source](https://fred.stlouisfed.org/series/H8B1029NCBCMG)) (SA: Y)
9. Deficit: Federal Surplus or Deficit ([source](https://fred.stlouisfed.org/series/MTSDS133FMS)) (SA: N)
10. ITBPrice: iShares US Home Construction ETF (ITB) price ([source](https://finance.yahoo.com/quote/ITB/history/)) (SA: N)
11. ITBVol: ITB volume of trade ([source](https://finance.yahoo.com/quote/ITB/history/)) (SA: N)
12. VGTPrice: Vanguard Information Technology Index Fund ETF (VGT) price ([source](https://finance.yahoo.com/quote/VGT/history/)) (SA: N)
13. VGTVol: VGT volume of trade ([source](https://finance.yahoo.com/quote/VGT/history/)) (SA: N)
14. S&P500Price: Standard and Poor's 500 (S&P 500) price ([source](https://finance.yahoo.com/quote/%5EGSPC/history)) (SA: N)
15. S&P500Vol: S&P 500 volume of trade ([source](https://finance.yahoo.com/quote/%5EGSPC/history)) (SA: N)


## Output Data
- File Name: all_outputs_cleaned.csv
- Number of columns: 3 (excluding Month)
- Number of rows: 204

## Outputs

The outputs are for employment by sector. All data is in (percent (%) change)/100. Data is not adjusted for seasonality.

1. Construction
2. Information
3. Total_Private: Private Sector
