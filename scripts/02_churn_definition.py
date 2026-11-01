# %% Setup
from pathlib import Path
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"

p = pd.read_parquet(ROOT / "data" / "processed" / "purchases.parquet")
p["date"] = p["InvoiceDate"].dt.normalize()

# One row per customer per purchase day
days = (p[["Customer ID", "date"]].drop_duplicates()
          .sort_values(["Customer ID", "date"]).reset_index(drop=True))
last_date = days["date"].max()
print("purchase days:", len(days), "| last date:", last_date.date())

# %% Gaps between consecutive purchase days
days["gap"] = days.groupby("Customer ID")["date"].diff().dt.days
gaps = days["gap"].dropna()
print(gaps.describe(percentiles=[.5, .75, .9, .95]).round(1))
for w in (14, 30, 42, 60, 90, 120):
    print(f"share of gaps longer than {w} days: {(gaps > w).mean():.1%}")

# %% Churn base rate by snapshot month and window
# At snapshot s: eligible = customers with a purchase before s.
# Churned = no purchase in [s, s + W).
first_purchase = days.groupby("Customer ID")["date"].min()
snaps = pd.date_range("2010-06-01", last_date, freq="MS")
rows = []
for W in (14, 30, 42, 60, 90):
    for s in snaps:
        end = s + pd.Timedelta(days=W)
        if end > last_date:
            continue
        eligible = first_purchase[first_purchase < s].index
        buyers = days.loc[(days["date"] >= s) & (days["date"] < end), "Customer ID"].unique()
        rows.append({"snapshot": s.date(), "window": W, "eligible": len(eligible),
                     "churn_rate": 1 - np.isin(eligible, buyers).mean()})
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
        last_buy = days.loc[days["date"] < s].groupby("Customer ID")["date"].max()
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
