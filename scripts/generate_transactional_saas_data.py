# scripts/generate_transactional_saas_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------- Config ----------
START = "2022-01-01"
END = "2025-08-01"
np.random.seed(42)

INITIAL_CUSTOMERS = 200           # existing customers at start
AVG_MONTHLY_NEW = 90              # mean new signups per month (Poisson)
PLAN_DISTR = ["SMB", "Mid", "Enterprise"]
PLAN_PROBS = [0.70, 0.25, 0.05]
REGIONS = ["US", "EU", "APAC"]
REGION_PROBS = [0.5, 0.3, 0.2]
ACQ_CHANNELS = ["Organic", "Paid Search", "Social", "Referral"]
ACQ_PROBS = [0.4, 0.3, 0.2, 0.1]

# churn monthly probability by plan (approx)
CHURN_PROBS = {"SMB": 0.06, "Mid": 0.03, "Enterprise": 0.01}

# base mrr distributions by plan (mean, std)
MRR_PARAMS = {
    "SMB": (80, 25),
    "Mid": (600, 200),
    "Enterprise": (3500, 1200)
}

# Make sure data folder exists
os.makedirs("data/raw", exist_ok=True)

# ---------- Helper functions ----------
def month_iterator(start, end):
    cur = pd.to_datetime(start)
    endd = pd.to_datetime(end)
    while cur <= endd:
        yield cur
        cur += relativedelta(months=1)

def add_months(dt, months):
    return (pd.to_datetime(dt) + relativedelta(months=months)).replace(day=1)

# ---------- Simulate customers & transactions ----------
months = list(month_iterator(START, END))
transactions = []   # list of dicts for transaction-level rows
customers = []      # summary customer info (signup, churn, plan, region, acq_channel)

customer_counter = 0

# Helper to create a new customer record and their monthly billing rows
def create_customer(signup_date):
    global customer_counter
    customer_counter += 1
    cid = f"C{customer_counter:06d}"
    plan = np.random.choice(PLAN_DISTR, p=PLAN_PROBS)
    region = np.random.choice(REGIONS, p=REGION_PROBS)
    acq = np.random.choice(ACQ_CHANNELS, p=ACQ_PROBS)
    # initial MRR sampled by plan; ensure positive
    base_mrr = max(5, np.random.normal(*MRR_PARAMS[plan]))
    # sample churn lifetime (in months) from geometric distribution with plan-specific prob
    churn_p = CHURN_PROBS[plan]
    # geometric gives the number of trials until first success; subtract 1 to get "months active before churn" inclusive
    lifetime_months = np.random.geometric(churn_p)
    churn_date = add_months(signup_date, lifetime_months)  # churn occurs at this month (customer not active after this month)
    # cap churn date to END; if churn beyond END, treat as still active
    if churn_date > pd.to_datetime(END):
        churn_date = pd.NaT

    # Store customer-level info
    customers.append({
        "customer_id": cid,
        "signup_date": pd.to_datetime(signup_date),
        "churn_date": churn_date,
        "plan": plan,
        "region": region,
        "acquisition_channel": acq,
        "base_mrr": round(base_mrr, 2)
    })

    # generate monthly billing rows from signup until churn (exclusive if churn_date is NaT, else include last active month)
    current = pd.to_datetime(signup_date)
    last_month = pd.to_datetime(END)
    while current <= last_month:
        # stop if churn_date is a timestamp and current > churn_date - they are inactive after churn month
        if pd.notna(churn_date) and current > churn_date:
            break

        m = base_mrr

        # small month-to-month noise / growth
        monthly_noise = np.random.normal(loc=0.003, scale=0.02)  # small daily growth/noise
        m = m * (1 + monthly_noise)

        # occasional expansion or contraction
        if np.random.rand() < 0.035:  # expansion chance
            m *= (1 + np.random.uniform(0.10, 0.40))
        if np.random.rand() < 0.02:   # contraction chance
            m *= (1 - np.random.uniform(0.10, 0.30))

        # round and floor
        m = round(max(5, m), 2)

        transactions.append({
            "customer_id": cid,
            "date": current.replace(day=1),
            "mrr": m,
            "plan": plan,
            "region": region,
            "acquisition_channel": acq,
            "is_churn_month": (pd.notna(churn_date) and current == churn_date)
        })

        # increment month
        current = add_months(current, 1)

# 1) create an initial customer base in the first month
start_month = pd.to_datetime(START)
for _ in range(INITIAL_CUSTOMERS):
    create_customer(start_month)

# 2) simulate monthly arrivals
for month in months:
    # skip the first month (already seeded with initial customers)
    if month == start_month:
        continue
    # sample new signups this month
    new_count = np.random.poisson(AVG_MONTHLY_NEW)
    for _ in range(new_count):
        create_customer(month)

# Build dataframes
transactions_df = pd.DataFrame(transactions)
customers_df = pd.DataFrame(customers)

# ---------- Aggregate monthly metrics ----------
# Total MRR per month
agg_mrr = transactions_df.groupby('date')['mrr'].sum().reset_index().rename(columns={'mrr': 'mrr'})

# New signups per month (customer signup_date)
new_signups = customers_df.groupby(customers_df['signup_date'].dt.to_period('M')).size().reset_index(name='new_signups')
new_signups['date'] = new_signups['signup_date'].dt.to_timestamp()
new_signups = new_signups[['date', 'new_signups']]

# Churned customers per month (customers whose churn_date falls in that month)
churns = customers_df[pd.notna(customers_df['churn_date'])]
if not churns.empty:
    churns = churns.groupby(churns['churn_date'].dt.to_period('M')).size().reset_index(name='churned_customers')
    churns['date'] = churns['churn_date'].dt.to_timestamp()
    churns = churns[['date', 'churned_customers']]
else:
    churns = pd.DataFrame(columns=['date', 'churned_customers'])

# Active customers calculation: cumulative
# Start with zero then add new_signups and subtract churns month by month
all_months = pd.DataFrame({'date': pd.to_datetime(months)})
all_months = all_months.merge(new_signups, on='date', how='left').merge(churns, on='date', how='left')
all_months['new_signups'] = all_months['new_signups'].fillna(0).astype(int)
all_months['churned_customers'] = all_months['churned_customers'].fillna(0).astype(int)

# compute active customers
active = []
running = 0
for idx, row in all_months.iterrows():
    running = running + int(row['new_signups']) - int(row['churned_customers'])
    active.append(running)
all_months['active_customers'] = active

# combine aggregated MRR with customer metrics
agg = all_months.merge(agg_mrr, on='date', how='left')
agg['mrr'] = agg['mrr'].fillna(0)

# Simulate marketing spend correlated with new signups + noise
base_marketing = 8000
agg['marketing_spend'] = (base_marketing + agg['new_signups'] * np.random.uniform(60, 140, size=len(agg))).round(2)

# sanity columns
agg = agg[['date', 'mrr', 'marketing_spend', 'churned_customers', 'new_signups', 'active_customers']]

# ---------- Save files ----------
transactions_df.to_csv("data/raw/saas_transactions.csv", index=False)
customers_df.to_csv("data/raw/saas_customers.csv", index=False)
agg.to_csv("data/raw/saas_mrr_aggregated.csv", index=False)

print("✅ Transactional dataset saved to 'data/raw/saas_transactions.csv'")
print("✅ Customer table saved to 'data/raw/saas_customers.csv'")
print("✅ Aggregated monthly summary saved to 'data/raw/saas_mrr_aggregated.csv'")
print(f"Total customers simulated: {len(customers_df)}")
print(f"Total transaction rows: {len(transactions_df)}")