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

## 2026-11-02 (churn definition)
- Median gap between purchases is ~30 days; 29% of gaps exceed 60 days, 19% exceed 90.
- Among all past buyers the 60-day churn rate was 67%, dominated by long-dormant customers. Restricted to customers active in the last 180 days: 59% on average (47%-71% by snapshot). Higher than I expected: most customers are occasional buyers.
- Chose 60-day churn as primary label, 90-day as sensitivity check.
- Strong seasonality: highest churn after Dec-Apr snapshots (Christmas lull). Training sees one lull, test (Aug-Oct 2011) has none. Listed as a limitation.
- Split uses embargo months (skip 2011-04 and 2011-07) because labels look 60 days ahead.
