# Customer Churn and LTV Prediction (UCI Online Retail II)

Predict which customers will stop purchasing in the next 60 days and estimate their 90-day value, then use both to decide who to target for retention.

> Status: in progress. Results, metrics and the deployed API will be added as the project proceeds.

## Problem
A non-contractual retailer cannot observe cancellations, so churn has to be defined from behaviour. This project:
1. Predicts the probability a customer makes no purchase in the next 60 days.
2. Estimates expected spend over the next 90 days.
3. Combines both to rank customers for retention outreach, using an assumed cost per contact.

## Data
- UCI Online Retail II (UK online retailer, Dec 2009 - Dec 2011), roughly 1M invoice lines.
- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (Kaggle mirror also available).
- Raw data is not committed to the repo. See `data/README.md` for download steps.

## Definitions (fill in before modelling)
| Item | Decision |
|---|---|
| Churn | No purchase in the 60 days after the snapshot date |
| LTV horizon | Net revenue in the 90 days after the snapshot date |
| Population | Customers with a purchase in the 180 days before the snapshot |
| Snapshot dates | First of each month, 2010-06-01 to 2011-10-01 (6-month burn-in; last snapshot needs a full 60-day label window) |
| Train / validation / test split | Time-based with embargo months so label windows never overlap: train 2010-06 to 2011-03; validation 2011-05, 2011-06; test 2011-08, 2011-09, 2011-10 |
| Cleaning rules | Drop null Customer ID; treat invoices starting with "C" as cancellations; drop non-positive prices; revenue = Quantity x Price |

## Approach
1. Clean and explore (`scripts/01_eda.py`)
2. Feature engineering in SQL (`sql/`): recency, frequency, monetary, tenure, purchase gaps, trends
3. Baseline logistic regression, then LightGBM/XGBoost (`scripts/`, `src/`)
4. BG/NBD + Gamma-Gamma for LTV (`lifetimes` or `pymc-marketing`)
5. Evaluation: AUC, precision/recall at top decile, calibration, lift, feature importance, failure analysis
6. Targeting economics: expected value of contacting top-risk customers
7. Deployment: FastAPI + Docker scoring API, with simple drift and score-distribution monitoring (`app/`)

## Results
_To be completed._

## Repo structure
    sql/         feature and label queries
    scripts/     numbered analysis scripts (EDA, modelling, evaluation), run cell by cell in Spyder
    src/         reusable code (features, training, evaluation)
    app/         FastAPI scoring service and Dockerfile
    data/        data download instructions (raw data not committed)
    docs/        decisions log, one-page summary

## README checklist (tick as you go)
- [ ] Problem and business framing
- [ ] Data source and cleaning rules
- [ ] Churn / LTV definitions and time-split diagram
- [ ] Feature list with leakage notes
- [ ] Model comparison table (baseline vs boosting)
- [ ] Metrics: AUC, precision/recall at top decile, calibration, lift
- [ ] Failure analysis (who does the model get wrong, and why)
- [ ] Targeting recommendation with assumptions stated
- [ ] How to run locally and via Docker
- [ ] Monitoring approach
- [ ] Limitations
