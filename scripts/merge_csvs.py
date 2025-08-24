import pandas as pd

# File paths
customers_path = "/Users/james_peter/Documents/saas_revenue_forcasting_gpt/data/raw/saas_customers.csv"
mrr_path = "/Users/james_peter/Documents/saas_revenue_forcasting_gpt/data/raw/saas_mrr_aggregated.csv"
transactions_path = "/Users/james_peter/Documents/saas_revenue_forcasting_gpt/data/raw/saas_transactions.csv"

# Load CSVs
customers_df = pd.read_csv(customers_path)
mrr_df = pd.read_csv(mrr_path)
transactions_df = pd.read_csv(transactions_path)

# Check columns in each file
print("Customers CSV columns:", customers_df.columns.tolist())
print("MRR CSV columns:", mrr_df.columns.tolist())
print("Transactions CSV columns:", transactions_df.columns.tolist())

