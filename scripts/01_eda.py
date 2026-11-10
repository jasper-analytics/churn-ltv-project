# %% Setup
from pathlib import Path
import pandas as pd

#   Project ROOT folder
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running a cell interactively
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"

#   Build file paths from ROOT
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
CACHE = ROOT / "data" / "processed" / "transactions_raw.parquet"

# %% Load both sheets once, then cache as parquet (the Excel read takes a few minutes)
if CACHE.exists():
    df = pd.read_parquet(CACHE)
else:
    sheets = pd.read_excel(RAW, sheet_name=None)
    print("sheets:", list(sheets))  
    df = pd.concat(sheets.values(), ignore_index=True)
#   Invoice column contains numbers and text (C489449), Parquet requires one type per column
#   Converts every text-like column to string dtype
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype("string")
    df.to_parquet(CACHE)
# %% First look
print(df.shape)
print(df.dtypes)
print(df.head())

# %% Date range, nulls, cancellations
#   Date Range
print("date range:", df["InvoiceDate"].min(), "->", df["InvoiceDate"].max())
#   Null share per column, Customer ID 0.2277
print("\nnull share per column:\n", df.isna().mean().round(4))
#   Cancellation share, 0.0183
print("\ncancellation lines:", df["Invoice"].astype(str).str.startswith("C").mean().round(4))
#   Non-positive  price lines, adjustments, bad-debt, free-items
print("non-positive price lines:", (df["Price"] <= 0).sum())
#   Negative Quantity lines, returns
print("negative quantity lines:", (df["Quantity"] < 0).sum())
#   Unique Customers
print("unique customers:", df["Customer ID"].nunique())

#   Missing Customer ID, drop these rows
#   Cancellations and nagative quantity, move to returns
#   Non-positive price, exclude from purchases
#   Non-standard stock codes, exclude fees, postage, adjustments
# %% Cleaning: split into purchases and returns
n0 = len(df)    #   starting row count
d = df.dropna(subset=["Customer ID"]).copy()
#   removes rows where Customer ID is missing, .copy() makes d independant of df, avoiding SettingWithCopyWarning
d["Customer ID"] = d["Customer ID"].astype("int64") #   convert IDs from float to integer
print("after dropping null Customer ID:", len(d), f"({len(d)/n0:.1%} kept)")

# Study non-product codes
pd.set_option("display.width", 200, "display.max_rows", 100, "display.max_colwidth", 40)

codes = df.assign(revenue=df["Quantity"] * df["Price"])
#   find StockCode that don't look like normal products (5 digits plus up to 2 letters)
odd = codes[~codes["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$", na=False)]

#   Build one-row-per-stock-code summary table
summary = (odd.groupby("StockCode")
              .agg(lines=("Invoice", "size"),
                   example_description=("Description", "first"),
                   avg_price=("Price", "mean"),
                   total_revenue=("revenue", "sum"))
              .sort_values("lines", ascending=False))

print(len(summary), "non-standard codes\n")
print(summary.round(2))

#   Clear non-products
#   POST: Postage charges, DOT: Dotcom postage, M: Manual adjustments, C2: Carriage,
#   D: Discounts, S: Samples, Bank Charges: Fees, ADJUST/2: Accounting Adjustments
#   PADS/CRUK: Odd item, charity, TEST: Test entries, AMAZONFEE: Amazon, B: Bad debt adjustment
#   gift_: gift vouchers w/ small revenue, DCGS: Separate product, small revenue

# Check uncertain codes
check = ["C3", "GIFT", "SP1002", "47503J", "m"]
print(df[df["StockCode"].isin(check)]
        .groupby("StockCode")
        .agg(lines=("Invoice", "size"),
             description=("Description", "first"),
             avg_price=("Price", "mean"),
             avg_qty=("Quantity", "mean")))
print(repr(df.loc[df["StockCode"].str.contains("47503J", na=False), "StockCode"].iloc[0]))

#   47503J: Real product, C3: negligible, GIFT" negligible, SP1002: Real product odd code, 
#   m: non-product lower-case of M

#   Non-product codes (postage, fees, manual adjustments) are not customer purchases
#   Removes leading, trailing whitespace, and uppercases
d["StockCode"] = d["StockCode"].str.strip().str.upper()
#   Define non-product list
NON_PRODUCT = ["POST", "DOT", "C2", "M", "D", "S", "BANK CHARGES", "AMAZONFEE",
               "ADJUST", "ADJUST2", "B", "CRUK", "PADS", "TEST001", "TEST002"]
is_non_product = d["StockCode"].isin(NON_PRODUCT)
print("non-product lines dropped:", is_non_product.sum())
#   Keep everything that is NOT a non-product.
d = d[~is_non_product]

#   Cancellations
d["is_cancel"] = d["Invoice"].str.startswith("C")
#   Remove non-positive prices
d = d[d["Price"] > 0]
#   Compute revenue
d["revenue"] = d["Quantity"] * d["Price"]

#   Split d into Purchases and Returns
purchases = d[(~d["is_cancel"]) & (d["Quantity"] > 0)].copy()
returns = d[d["is_cancel"]].copy()

print("purchase lines:", len(purchases), "| return lines:", len(returns))
#   Number of unique customer ID with purchase
print("customers with purchases:", purchases["Customer ID"].nunique())
#   Min and Max Invoice Date of Purchases
print("date range:", purchases["InvoiceDate"].min(), "->", purchases["InvoiceDate"].max())
#   Total $ Sum of Purchase Revenue
print("total purchase revenue:", round(purchases["revenue"].sum()))
#   Total $ Sum of Return Value
print("total return value:", round(returns["revenue"].sum()))

# %% Save cleaned tables to disk as Parquet files
purchases.to_parquet(ROOT / "data" / "processed" / "purchases.parquet")
returns.to_parquet(ROOT / "data" / "processed" / "returns.parquet")