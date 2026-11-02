# %% Setup
from pathlib import Path
import numpy as np
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"
P = ROOT / "data" / "processed"

purch = pd.read_parquet(P / "purchases.parquet")
rets = pd.read_parquet(P / "returns.parquet")
for t in (purch, rets):
    t["date"] = t["InvoiceDate"].dt.normalize()

# One row per invoice
orders = (purch.groupby(["Customer ID", "Invoice"])
               .agg(date=("date", "first"), revenue=("revenue", "sum"),
                    n_items=("Quantity", "sum"), n_products=("StockCode", "nunique"))
               .reset_index())
ret_orders = (rets.groupby(["Customer ID", "Invoice"])
                  .agg(date=("date", "first"), revenue=("revenue", "sum"))
                  .reset_index())

LOOKBACK, CHURN_W, LTV_W = 180, 60, 90
DATA_END = orders["date"].max()
SNAPS = pd.date_range("2010-06-01", "2011-10-01", freq="MS")
D = lambda n: pd.Timedelta(days=n)

# %% Feature + label builder (features use ONLY rows dated before the snapshot)
def build_snapshot(s, orders, ret_orders):
    hist = orders[orders["date"] < s]
    h_ret = ret_orders[ret_orders["date"] < s]

    # Purchase days -> recency, tenure, gaps
    days = (hist[["Customer ID", "date"]].drop_duplicates()
                .sort_values(["Customer ID", "date"]))
    days["gap"] = days.groupby("Customer ID")["date"].diff().dt.days
    g = days.groupby("Customer ID")
    f = pd.DataFrame({"first_date": g["date"].min(), "last_date": g["date"].max(),
                      "n_days": g["date"].size(), "last_gap": g["gap"].last(),
                      "gap_std": g["gap"].std(),   # needs >= 3 purchase days
                      "gap_max": g["gap"].max()})  # needs >= 2 purchase days

    # Population: bought within the lookback window
    f = f[f["last_date"] >= s - D(LOOKBACK)].copy()
    f["recency"] = (s - f["last_date"]).dt.days
    f["tenure"] = (s - f["first_date"]).dt.days
    f["avg_gap"] = (f["last_date"] - f["first_date"]).dt.days / (f["n_days"] - 1).replace(0, np.nan)
    f["recency_over_gap"] = f["recency"] / f["avg_gap"]
    f["recency_over_max_gap"] = f["recency"] / f["gap_max"]   # > 1 = longer silence than ever before
    f["gap_cv"] = f["gap_std"] / f["avg_gap"]                  # low = steady buyer

    # Frequency / monetary in recent windows
    for w in (30, 90, 180):
        hw = hist[hist["date"] >= s - D(w)]
        f = f.join(hw.groupby("Customer ID").agg(
            **{f"orders_{w}": ("Invoice", "nunique"), f"revenue_{w}": ("revenue", "sum")}))
    prev = hist[(hist["date"] >= s - D(180)) & (hist["date"] < s - D(90))]
    f = f.join(prev.groupby("Customer ID").agg(
        orders_prev90=("Invoice", "nunique"), revenue_prev90=("revenue", "sum")))
    win_cols = [c for c in f.columns if c.startswith(("orders_", "revenue_"))]
    f[win_cols] = f[win_cols].fillna(0)
    f["orders_trend"] = f["orders_90"] - f["orders_prev90"]
    f["revenue_trend"] = f["revenue_90"] - f["revenue_prev90"]
    f["revenue_trend_log"] = np.log1p(f["revenue_90"]) - np.log1p(f["revenue_prev90"])  # relative change
    f["orders_trend_ratio"] = (f["orders_90"] + 1) / (f["orders_prev90"] + 1)           # relative change

    # Lifetime-to-date totals
    f = f.join(hist.groupby("Customer ID").agg(
        orders_total=("Invoice", "nunique"), revenue_total=("revenue", "sum"),
        avg_items=("n_items", "mean"), avg_products=("n_products", "mean")))
    f["aov"] = f["revenue_total"] / f["orders_total"]

    # Returns in the last 180 days
    hr = h_ret[h_ret["date"] >= s - D(180)]
    f = f.join(hr.groupby("Customer ID").agg(
        returns_180=("Invoice", "nunique"), return_value_180=("revenue", "sum")))
    f[["returns_180", "return_value_180"]] = f[["returns_180", "return_value_180"]].fillna(0)

    # Labels (use data on/after the snapshot)
    fut = orders[(orders["date"] >= s) & (orders["date"] < s + D(CHURN_W))]
    f["churn_60"] = (~f.index.isin(fut["Customer ID"].unique())).astype(int)
    if s + D(LTV_W) <= DATA_END:
        wp = orders[(orders["date"] >= s) & (orders["date"] < s + D(LTV_W))].groupby("Customer ID")["revenue"].sum()
        wr = ret_orders[(ret_orders["date"] >= s) & (ret_orders["date"] < s + D(LTV_W))].groupby("Customer ID")["revenue"].sum()
        f["net_revenue_90"] = wp.reindex(f.index).fillna(0) + wr.reindex(f.index).fillna(0)
    else:
        f["net_revenue_90"] = np.nan

    f.insert(0, "snapshot", s)
    return f.drop(columns=["first_date", "last_date"]).reset_index()


df = pd.concat([build_snapshot(s, orders, ret_orders) for s in SNAPS], ignore_index=True)

# Split with embargo months (labels look 60 days ahead, so skip overlap months)
def split_of(s):
    if s <= pd.Timestamp("2011-03-01"): return "train"
    if s in (pd.Timestamp("2011-05-01"), pd.Timestamp("2011-06-01")): return "val"
    if s >= pd.Timestamp("2011-08-01"): return "test"
    return "embargo"

df["split"] = df["snapshot"].map(split_of)
print(df.shape)
print(df.groupby("split").agg(rows=("churn_60", "size"), churn_rate=("churn_60", "mean")).round(3))


# %% Check 1: churn rates must match the earlier base-rate analysis
chk = df.groupby("snapshot").agg(n=("churn_60", "size"), churn_60=("churn_60", "mean")).round(3)
print(chk)  # expect ~0.581 for 2010-06-01 and ~0.474 for 2011-10-01


# %% Check 2: leakage test - features must not change if all future data is deleted
s = pd.Timestamp("2011-03-01")
full = df[df["snapshot"] == s].set_index("Customer ID")
trunc = build_snapshot(s, orders[orders["date"] < s], ret_orders[ret_orders["date"] < s]).set_index("Customer ID")
label_cols = ["churn_60", "net_revenue_90", "split"]
feat_cols = [c for c in full.columns if c not in label_cols]
pd.testing.assert_frame_equal(full[feat_cols].sort_index(), trunc[feat_cols].sort_index(), check_dtype=False)
print("LEAKAGE TEST PASSED: features identical with and without future data")


# %% Feature summary and save
feature_cols = [c for c in df.columns if c not in ["Customer ID", "snapshot", "split", "churn_60", "net_revenue_90"]]
print(len(feature_cols), "features:", feature_cols)
print("\nnull share:\n", df[feature_cols].isna().mean().round(3)[lambda x: x > 0])
df.to_parquet(P / "snapshots.parquet")
print("saved", df.shape)


# %% Churn by order-count segment, and naive baselines to beat
from sklearn.metrics import roc_auc_score

d_ = df[df["split"] != "embargo"].assign(
    order_segment=lambda x: pd.cut(x["orders_total"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]))
seg = (d_.groupby(["split", "order_segment"], observed=True)
         .agg(rows=("churn_60", "size"), churn_rate=("churn_60", "mean")).round(3))
print(seg.unstack("split"))

for sp in ("train", "val", "test"):
    d = df[df["split"] == sp]
    print(sp,
          "| AUC recency only:", round(roc_auc_score(d["churn_60"], d["recency"]), 3),
          "| AUC fewer orders_180:", round(roc_auc_score(d["churn_60"], -d["orders_180"]), 3),
          "| AUC fewer orders_total:", round(roc_auc_score(d["churn_60"], -d["orders_total"]), 3))
    
# %% LTV target: how heavy is the tail?
y = df.loc[(df["split"] == "train") & df["net_revenue_90"].notna(), "net_revenue_90"]
print(y.describe(percentiles=[.5, .9, .99]).round(0))
print("share negative:", round((y < 0).mean(), 4))
ys = y.clip(lower=0).sort_values(ascending=False)
for q in (0.01, 0.05, 0.20):
    k = int(len(ys) * q)
    print(f"top {q:.0%} of customers hold {ys.iloc[:k].sum() / ys.sum():.1%} of 90-day revenue")