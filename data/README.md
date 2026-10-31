# Data download

Raw data is not committed to this repo.

1. Download **Online Retail II** from the UCI repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii
   (a Kaggle mirror also exists).
2. Unzip and place the file at `data/raw/online_retail_II.xlsx`. It has two sheets: `Year 2009-2010` and `Year 2010-2011`.
3. Load both sheets, concatenate them, and save to `data/processed/transactions.parquet` (or load into BigQuery / PostgreSQL).

Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.

Notes:
- About a fifth of rows have no Customer ID. These are dropped for customer-level modelling.
- Invoices starting with "C" are cancellations/returns.
- Check the dataset's licence/terms on the UCI page before redistributing anything derived from it.
