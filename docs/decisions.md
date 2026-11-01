# Decisions log

Running notes: what I decided, why, and what didn't work. This becomes raw material for the README and the write-up.

## 2026-11-01
- Chose UCI Online Retail II over KKBox: non-contractual purchases fit BG/NBD + Gamma-Gamma, and the dataset is small enough to run locally.
- Working in Spyder with .py scripts (`# %%` cells) instead of notebooks.
- Open questions:
  - Churn window: 60 days (try 30 and 90 as sensitivity checks)?
  - Snapshot cadence for rolling training rows?
  - How to treat wholesale customers (very large baskets)?

## 2026-11-01 (data cleaning)
- Raw: 1,067,371 lines, 5,942 customers; 22.8% of lines have no Customer ID (dropped).
- Non-product StockCodes: found by listing codes not matching ^\d{5}[A-Za-z]{0,2}$ and reading descriptions/revenue. Dropped postage, fees, manual/adjustment, samples, test entries. Kept DCGS* products and gift vouchers (real customer spend).
- Purchases = non-"C" invoices with Quantity > 0 and Price > 0; returns kept as a separate table.
- Cleaned: 802,634 purchase lines, 5,852 customers, 2009-12-01 to 2011-12-09, revenue GBP 17.4M, returns GBP 0.72M.
