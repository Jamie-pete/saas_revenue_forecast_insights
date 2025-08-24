# SaaS Revenue Analytics & Forecasting

## What this does

- Cleans & prepares SaaS customer + transaction + MRR data
- Visualizes revenue trends, customer dynamics, cohorts, segmentation
- Computes ARPU & LTV
- Models MRR Forecast with SARIMAX (ARIMA)
- Exports clean datasets for Power BI
- Auto-generates Executive Insights

## Key files

- notebooks/saas_revenue_project.ipynb
- data/processed/trans_cust_clean.csv
- data/processed/mrr_clean.csv
- data/processed/mrr_forecast.csv
- reports/ai_insights.md

## Run

1. pip install -r requirements.txt
2. Open the notebook and run cells in order

## Notes

- Keep raw files out of Git using .gitignore
- Parameterize file paths for other environments
