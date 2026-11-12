# %% Setup
from pathlib import Path
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"

p = pd.read_parquet(ROOT / "data" / "processed" / "purchases.parquet")
#   Set 2009-12-01 07:45:00 -> 2009-12-01 00:00:00
p["date"] = p["InvoiceDate"].dt.normalize()

# One row per customer per purchase day, used for days["gap"] below
days = (p[["Customer ID", "date"]].drop_duplicates()
          .sort_values(["Customer ID", "date"]).reset_index(drop=True))
last_date = days["date"].max()
print("purchase days:", len(days), "| last date:", last_date.date())

# %% Gaps between consecutive purchase days per Customer
days["gap"] = days.groupby("Customer ID")["date"].diff().dt.days
#   Removes first-purchase rows
gaps = days["gap"].dropna()
#   Prints the Count, Mean, Standard Deviation, Min, Max, and the Percentiles 
print(gaps.describe(percentiles=[.5, .75, .9, .95]).round(1))
#   Median 32.0, Mean 59.3, Distribution is Right-Skewed
for w in (14, 30, 42, 60, 90, 120):
    print(f"share of gaps longer than {w} days: {(gaps > w).mean():.1%}")
#   51.0% of gaps are longer than 30 days
#   Does not mean 51% of customers wait more than 30 days. Unit is gaps not customer.

# %% Churn base rate by snapshot month and window (W)
# At snapshot s: eligible = customers with a purchase before s.
# Churned = no purchase in [s, s + W).
first_purchase = days.groupby("Customer ID")["date"].min()
#   Snapshot dates, MS = Month Start
snaps = pd.date_range("2010-06-01", last_date, freq="MS")
rows = []
#   2 weeks, 1 month, 6 weeks, 2/3 month
for W in (14, 30, 42, 60, 90):
    for s in snaps:
        end = s + pd.Timedelta(days=W)  #   The end of the window (exclusive)
        if end > last_date: #   Skip any window that runs past the end of the data
            continue
        eligible = first_purchase[first_purchase < s].index  #   Customers with first purchase before s
        #   Rows of days that fall inside the window, reduced to the unique customer IDs
        #   This is everyone who bought during the window
        buyers = days.loc[(days["date"] >= s) & (days["date"] < end), "Customer ID"].unique()
        #   np.isin gives True or False for each eligible Customer (did they appear in buyers?)
        #   Example eligible = [A, B, C, D, E], buyers = [B, D, E, Y, Z]
        #   Y and Z first purchase does not fall in eligible so they are not counted
        #   In buyers? [A: False, B:True, C:False, D:True, E:True]
        #   True counts as 1, so the mean is the share who bought. 1-(2/5) = 0.60
        #   If W is 30, that means 60% of eligible customer did not purchase any product 
        #   between s to s + 30 (exclusive)
        rows.append({"snapshot": s.date(), "window": W, "eligible": len(eligible),
                     "churn_rate": 1 - np.isin(eligible, buyers).mean()})
#   Turn the list into a table
base = pd.DataFrame(rows)
print(base.pivot(index="snapshot", columns="window", values="churn_rate").round(3))
print("\naverage churn rate by window:\n", base.groupby("window")["churn_rate"].mean().round(3))
# %% Base rate among recently active customers only
LOOKBACK = 180  # days; "active" = bought within this many days before the snapshot
rows = []
for W in (14, 30, 42, 60, 90):
    for s in snaps:
        end = s + pd.Timedelta(days=W)
        if end > last_date:
            continue
        #   Latest purchase date before s
        last_buy = days.loc[days["date"] < s].groupby("Customer ID")["date"].max()
        #   Customers who made at least one purchase between s - 180d and s (exclusive)
        active = last_buy[last_buy >= s - pd.Timedelta(days=LOOKBACK)].index
        buyers = days.loc[(days["date"] >= s) & (days["date"] < end), "Customer ID"].unique()
        rows.append({"snapshot": s.date(), "window": W, "active": len(active),
                     "churn_rate": 1 - np.isin(active, buyers).mean()})
active_base = pd.DataFrame(rows)
print(active_base.pivot(index="snapshot", columns="window", values="churn_rate").round(3))
print("\naverage churn rate, active customers only:\n",
      active_base.groupby("window")["churn_rate"].mean().round(3))
print("\ncustomers per snapshot (min / median / max):",
      active_base["active"].min(), active_base["active"].median(), active_base["active"].max())
