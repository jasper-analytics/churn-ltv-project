# Decisions log

Running notes: what I decided, why, and what didn't work. This becomes raw material for the README and the write-up.

## 2026-11-01
- Chose UCI Online Retail II over KKBox: non-contractual purchases fit BG/NBD + Gamma-Gamma, and the dataset is small enough to run locally.
- Working in Spyder with .py scripts (`# %%` cells) instead of notebooks.
- Open questions:
  - Churn window: 60 days (try 30 and 90 as sensitivity checks)?
  - Snapshot cadence for rolling training rows?
  - How to treat wholesale customers (very large baskets)?
