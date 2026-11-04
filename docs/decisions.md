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

## 2026-11-03 (features and baselines)
- Built 29 features per customer per monthly snapshot (recency, frequency, monetary, gaps, relative trends, returns). Leakage test passed: features identical with and without future data.
- Rows: train 30,673 / val 5,705 / test 8,453 (6,021 embargo rows unused). Churn rate 60.8% / 58.9% / 51.3%.
- Naive baselines to beat (AUC): recency only 0.65; fewer orders_180 0.71-0.72; fewer orders_total 0.72-0.74. Test AUC must clearly exceed ~0.72 to show value.
- One-time buyers churn ~79%, 5+ order customers ~34%.
- Drift: lifetime features (tenure, orders_total, revenue_total) grow with snapshot date; 5+ order customers are 28% of train but 52% of test. Plan: ablate these features.
- LTV target: median 0, mean 548, top 1% hold 32% of revenue, 1.3% negative. Plan: two-part model (buys at all x amount), log target clipped at 0, train-only cap at 99.5th percentile, report decile totals / Spearman / top-decile capture, not raw squared error.
- Seasonality: annual features (same window last year, Sep-Nov share) need 12 months of history, so they exist only from 2010-12 onward. Test as a secondary experiment; try month-of-year as an ablation.

## 2026-11-05 (churn models)
- Test AUC: naive (fewer orders) 0.719; logistic 0.758; LightGBM 0.760; LightGBM without lifetime features 0.759; with month-of-year 0.757. Per-snapshot AUC is within 0.002 of pooled AUC, so the result is not inflated by base-rate shifts.
- Top-10% risk flag: 82% precision on test vs 51% base rate (lift 1.6x of a 1.95x maximum).
- LightGBM and logistic are tied: signal is mostly simple. Report both.
- Dropped lifetime features (tenure, orders_total, revenue_total, n_days): no accuracy loss, less drift risk.
- Dropped month-of-year: shifts the base rate but lowers per-snapshot AUC on test (0.755 vs 0.761).
- Calibration: predictions run ~9 points high on test (0.606 vs 0.513) because of season and population shift. Isotonic recalibration on validation helps the low bins but worsens the top bins. For ranking this does not matter; for expected-value rules it does. Plan: periodic base-rate correction, monitor predicted vs actual per period.
- Feature importance (gain): orders_180 25%, gap_std 23%, n_days 13%, avg_gap 10%. recency_over_max_gap (1%) was much weaker than I expected.
