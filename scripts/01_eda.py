# %% Setup
from pathlib import Path
import pandas as pd

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running a cell interactively
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"

RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
CACHE = ROOT / "data" / "processed" / "transactions_raw.parquet"

# %% Load both sheets once, then cache as parquet (the Excel read takes a few minutes)
if CACHE.exists():
    df = pd.read_parquet(CACHE)
else:
    sheets = pd.read_excel(RAW, sheet_name=None)
    print("sheets:", list(sheets))  
    df = pd.concat(sheets.values(), ignore_index=True)
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype("string")  # mixed number/text columns break parquet
    df.to_parquet(CACHE)
# %% First look
print(df.shape)
print(df.dtypes)
print(df.head())

# %% Date range, nulls, cancellations
print("date range:", df["InvoiceDate"].min(), "->", df["InvoiceDate"].max())
print("\nnull share per column:\n", df.isna().mean().round(4))
print("\ncancellation lines:", df["Invoice"].astype(str).str.startswith("C").mean().round(4))
print("non-positive price lines:", (df["Price"] <= 0).sum())
print("negative quantity lines:", (df["Quantity"] < 0).sum())
print("unique customers:", df["Customer ID"].nunique())

# %% Cleaning: split into purchases and returns
n0 = len(df)
d = df.dropna(subset=["Customer ID"]).copy()
d["Customer ID"] = d["Customer ID"].astype("int64")
print("after dropping null Customer ID:", len(d), f"({len(d)/n0:.1%} kept)")

# Study non-product codes
pd.set_option("display.width", 200, "display.max_rows", 100, "display.max_colwidth", 40)

codes = df.assign(revenue=df["Quantity"] * df["Price"])
odd = codes[~codes["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$", na=False)]

summary = (odd.groupby("StockCode")
              .agg(lines=("Invoice", "size"),
                   example_description=("Description", "first"),
                   avg_price=("Price", "mean"),
                   total_revenue=("revenue", "sum"))
              .sort_values("lines", ascending=False))

print(len(summary), "non-standard codes\n")
print(summary.round(2))

# Check uncertain codes
check = ["C3", "GIFT", "SP1002", "47503J", "m"]
print(df[df["StockCode"].isin(check)]
        .groupby("StockCode")
        .agg(lines=("Invoice", "size"),
             description=("Description", "first"),
             avg_price=("Price", "mean"),
             avg_qty=("Quantity", "mean")))
print(repr(df.loc[df["StockCode"].str.contains("47503J", na=False), "StockCode"].iloc[0]))

# Non-product codes (postage, fees, manual adjustments) are not customer purchases
d["StockCode"] = d["StockCode"].str.strip().str.upper()
NON_PRODUCT = ["POST", "DOT", "C2", "M", "D", "S", "BANK CHARGES", "AMAZONFEE",
               "ADJUST", "ADJUST2", "B", "CRUK", "PADS", "TEST001", "TEST002"]
is_non_product = d["StockCode"].isin(NON_PRODUCT)
print("non-product lines dropped:", is_non_product.sum())
d = d[~is_non_product]

d["is_cancel"] = d["Invoice"].str.startswith("C")
d = d[d["Price"] > 0]
d["revenue"] = d["Quantity"] * d["Price"]

purchases = d[(~d["is_cancel"]) & (d["Quantity"] > 0)].copy()
returns = d[d["is_cancel"]].copy()
print("purchase lines:", len(purchases), "| return lines:", len(returns))
print("customers with purchases:", purchases["Customer ID"].nunique())
print("date range:", purchases["InvoiceDate"].min(), "->", purchases["InvoiceDate"].max())
print("total purchase revenue:", round(purchases["revenue"].sum()))
print("total return value:", round(returns["revenue"].sum()))

# %% Save cleaned tables
purchases.to_parquet(ROOT / "data" / "processed" / "purchases.parquet")
returns.to_parquet(ROOT / "data" / "processed" / "returns.parquet")