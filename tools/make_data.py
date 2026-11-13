"""Rebuild the two data files in dags/data/ from the UCI download. Not needed to run the lab: both are committed.

    1. Download online_retail_II.zip from https://archive.ics.uci.edu/dataset/502/online+retail+ii and unzip it
    2. python3 -m venv .venv-data && .venv-data/bin/pip install pandas==2.3.3 openpyxl==3.1.5
    3. .venv-data/bin/python tools/make_data.py path/to/online_retail_II.xlsx

Writes:
  dags/data/online_retail_II.csv.gz  both sheets of the workbook (Dec 2009 - Dec 2011) as one CSV, unchanged
  dags/data/bad_sample.csv.gz        110,000 of those rows with three problems added, for the quality gate to reject
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "dags" / "data"
# mtime=0 keeps the gzip bytes identical between rebuilds, so the files' hashes do not change
GZIP = {"method": "gzip", "mtime": 0}


def main(xlsx_path):
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["InvoiceDate"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(DATA_DIR / "online_retail_II.csv.gz", index=False, compression=GZIP)
    print(f"online_retail_II.csv.gz: {len(df):,} rows from sheets {list(sheets)}")

    rng = np.random.default_rng(7)
    bad = df.head(110_000).copy()
    bad.loc[rng.random(len(bad)) < 0.40, "Customer ID"] = pd.NA     # 40% of rows lose their customer
    bad.loc[rng.choice(len(bad), 25, replace=False), "InvoiceDate"] = "2011-13-45 25:61:00"  # impossible dates
    bad.loc[rng.choice(len(bad), 12, replace=False), "Invoice"] = "INV-0000"   # not an invoice number
    bad.to_csv(DATA_DIR / "bad_sample.csv.gz", index=False, compression=GZIP)
    print(f"bad_sample.csv.gz: {len(bad):,} rows")


if __name__ == "__main__":
    main(sys.argv[1])
