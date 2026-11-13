"""The pipeline's steps as plain functions that read and write files, so they run and test without Airflow.

Each step writes its data to the run's folder and returns only something small (a file path, a number,
a few counts) for the DAG to hand to the next task through XCom.

Data: Online Retail II (UCI), every line of every invoice of a UK online shop from Dec 2009 to Dec 2011.
Model: each customer is described by recency, frequency and monetary value (RFM) and segmented with a
Gaussian mixture, whose number of components and covariance type are chosen by BIC.
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CUSTOMER = "Customer ID"
REQUIRED_COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", CUSTOMER, "Country"]
INVOICE_PATTERN = r"[A-Z]?\d{6}"  # 489434; C489449 is a cancellation, A506401 a bad-debt adjustment
FEATURES = ["recency", "frequency", "monetary"]
COUNT_FEATURES = [0, 1]  # recency (whole days) and frequency (number of invoices) are counts
COVARIANCE_TYPES = ["full", "diag", "tied", "spherical"]
GMM_KWARGS = {"n_init": 3, "max_iter": 500, "random_state": 42}
NOISE_DRAWS = 5  # each fit's BIC is averaged over this many dequantization draws (see Dequantize)
BIC_TOLERANCE = 10  # a BIC difference below 10 is not strong evidence for the bigger model
MIN_CONFIDENCE = 0.8  # a customer whose most likely segment is below this is reported as uncertain


def load_data(input_file, run_dir) -> str:
    """Copy a CSV (optionally gzipped) from the data folder into the run's folder as Parquet. Returns its path."""
    # Kept as text so validate_data can see values that are not dates or numbers instead of failing here
    df = pd.read_csv(DATA_DIR / input_file, dtype={c: str for c in ["Invoice", "StockCode", "InvoiceDate", CUSTOMER]})
    out_path = Path(run_dir) / "raw.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def validate_data(raw_path, run_dir, min_rows, max_missing_customer_fraction) -> dict:
    """Run the data-quality checks and save them to quality_report.json. Returns the report.

    The checks catch a broken export. Problems that are normal in this data (cancellations, returns,
    orders without a customer account) are counted under "profile" and handled later by build_features.
    """
    df = pd.read_parquet(raw_path)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks = {"required_columns": {"passed": not missing_columns, "missing": missing_columns}}
    profile = {}

    if not missing_columns:
        dates = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        bad_dates = int(dates.isna().sum())
        bad_invoices = df.loc[~df["Invoice"].fillna("").str.fullmatch(INVOICE_PATTERN), "Invoice"]
        no_customer = float(df[CUSTOMER].isna().mean())
        checks.update({
            "row_count": {"passed": len(df) >= min_rows, "rows": len(df), "min_rows": min_rows},
            "parseable_dates": {"passed": bad_dates == 0, "unparseable": bad_dates,
                                "first": str(dates.min()), "last": str(dates.max())},
            "invoice_format": {"passed": bad_invoices.empty, "bad_invoices": len(bad_invoices),
                               "examples": sorted(bad_invoices.astype(str).unique())[:5]},
            "missing_customer_id": {"passed": no_customer <= max_missing_customer_fraction,
                                    "fraction_of_rows": round(no_customer, 4), "max": max_missing_customer_fraction},
        })
        profile = {
            "cancellation_lines": int(df["Invoice"].str.startswith("C", na=False).sum()),
            "non_positive_quantity": int((df["Quantity"] <= 0).sum()),
            "non_positive_price": int((df["Price"] <= 0).sum()),
            "duplicate_lines": int(df.duplicated().sum()),
            "customers": int(df[CUSTOMER].nunique()),
        }

    report = {"passed": all(c["passed"] for c in checks.values()), "checks": checks, "profile": profile}
    with open(Path(run_dir) / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def publish_dataset(raw_path, datasets_dir, version) -> dict:
    """Copy validated data to datasets/transactions/<version>.parquet, a file that is never overwritten.

    Returns what the asset event records: where the version is, how many rows it has and its content hash.
    """
    out_dir = Path(datasets_dir) / "transactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version}.parquet"
    shutil.copyfile(raw_path, out_path)
    return {
        "path": str(out_path),
        "rows": len(pd.read_parquet(out_path)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()[:12],
    }


def build_features(transactions_path, run_dir) -> dict:
    """Turn invoice lines into one row per customer: recency, frequency and monetary value.

    Lines without a customer, free or adjustment lines (price <= 0) and exact duplicate lines are
    dropped, and how many of each is returned. Cancellations are not purchases, but they are subtracted
    from spend: otherwise an order that was placed and then cancelled would count in full.
    """
    df = pd.read_parquet(transactions_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")
    lines = len(df)

    # Applied in order, so each line is counted once, under the first reason that removes it
    dropped = {}
    for reason, is_dropped in [
        ("no_customer", lambda d: d[CUSTOMER].isna()),
        ("non_positive_price", lambda d: d["Price"] <= 0),
        ("duplicate_line", lambda d: d.duplicated()),
        ("non_positive_quantity", lambda d: ~d["Invoice"].str.startswith("C") & (d["Quantity"] <= 0)),
    ]:
        mask = is_dropped(df)
        dropped[reason] = int(mask.sum())
        df = df[~mask]

    df = df.assign(amount=df["Quantity"] * df["Price"])
    cancelled = df["Invoice"].str.startswith("C")
    purchases = df[~cancelled]

    # Recency is whole days from a customer's last purchase day to the day after the data ends, so the
    # latest buyers get 1
    snapshot = purchases["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = purchases.groupby(CUSTOMER).agg(
        recency=("InvoiceDate", lambda d: (snapshot - d.max().normalize()).days),
        frequency=("Invoice", "nunique"),
    )
    # Net spend: purchases plus cancellations (whose amounts are negative)
    rfm["monetary"] = df.groupby(CUSTOMER)["amount"].sum().reindex(rfm.index).round(2)
    # Customers who returned everything (or more, for orders placed before the data starts) have no spend left
    no_net_spend = rfm["monetary"] <= 0
    rfm = rfm[~no_net_spend]

    features_path = Path(run_dir) / "rfm.parquet"
    rfm.to_parquet(features_path)
    return {
        "features_path": str(features_path),
        "snapshot_date": snapshot.date().isoformat(),
        "lines": lines,
        "purchase_lines": len(purchases),
        "cancellation_lines": int(cancelled.sum()),
        "dropped_lines": dropped,
        "customers": len(rfm),
        "customers_without_net_spend": int(no_net_spend.sum()),
    }


class Dequantize(TransformerMixin, BaseEstimator):
    """Spread whole-number features over [n, n+1) while fitting, and use the midpoint n + 0.5 afterwards.

    27% of customers bought exactly once, so frequency piles up on a few exact values. A Gaussian
    mixture can put a component of almost zero width on one of those values, and its likelihood (and
    so BIC) then grows without limit and always picks more components. Uniform noise of one unit removes
    the spikes without changing the order of customers.
    """

    def __init__(self, columns=tuple(COUNT_FEATURES), random_state=0):
        self.columns = columns
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        X = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        X[:, list(self.columns)] += rng.uniform(0, 1, size=(len(X), len(self.columns)))
        return X

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[:, list(self.columns)] += 0.5
        return X


def make_preprocessing(random_state) -> Pipeline:
    """Dequantize, log-transform (spend and frequency are very skewed) and standardize the RFM features."""
    return Pipeline([
        ("dequantize", Dequantize(random_state=random_state)),
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])


def data_preprocessing(features_path, run_dir) -> dict:
    """Preprocess the RFM features once per noise draw, stacked into model_inputs.npy (draw, customer, feature).

    The fits compare models on all the draws; the final model is trained on the first one, and its fitted
    preprocessing is saved so the final model can take plain RFM values.
    """
    rfm = pd.read_parquet(features_path)[FEATURES]
    pipelines = [make_preprocessing(random_state=draw) for draw in range(NOISE_DRAWS)]
    inputs = np.stack([p.fit_transform(rfm) for p in pipelines])

    input_path = Path(run_dir) / "model_inputs.npy"
    np.save(input_path, inputs)
    preprocess_path = Path(run_dir) / "preprocess.pkl"
    with open(preprocess_path, "wb") as f:
        pickle.dump(pipelines[0], f)
    return {"input_path": str(input_path), "preprocess_path": str(preprocess_path)}


def fit_gmm(input_path, k, covariance_type) -> dict:
    """Fit one Gaussian mixture to each noise draw and return its mean BIC. One call per (k, covariance type),
    run in parallel by the DAG."""
    bics, converged = [], True
    # Many of these run at once; letting each use every core makes them fight over the CPU
    with threadpool_limits(limits=1):
        for data in np.load(input_path):
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, **GMM_KWARGS).fit(data)
            bics.append(gmm.bic(data))
            converged = converged and bool(gmm.converged_)
    return {"k": k, "covariance_type": covariance_type, "bic": round(float(np.mean(bics)), 1),
            "bic_sd": round(float(np.std(bics)), 1), "converged": converged}


def choose_model(fits, k_max, run_dir) -> dict:
    """Pick the simplest model the data supports, and save every fit's BIC for reference.

    That is the fewest components among fits within BIC_TOLERANCE of the lowest BIC, and of those the
    lowest BIC. If the lowest BIC is at the largest k tried, BIC may still be falling, so the range is
    too small to tell.
    """
    usable = [f for f in fits if f["converged"]]
    if not usable:
        raise ValueError("No Gaussian mixture converged")
    lowest = min(usable, key=lambda f: f["bic"])
    if lowest["k"] == k_max:
        raise ValueError(f"The lowest BIC is at k_max={k_max}, so it may still be falling; raise k_max")
    close = [f for f in usable if f["bic"] <= lowest["bic"] + BIC_TOLERANCE]
    best = min(close, key=lambda f: (f["k"], f["bic"]))
    choice = {**best, "lowest_bic_k": lowest["k"], "above_lowest_bic": round(best["bic"] - lowest["bic"], 1)}

    with open(Path(run_dir) / "bic.json", "w") as f:
        json.dump({"chosen": choice, "tolerance": BIC_TOLERANCE,
                   "fits": sorted(fits, key=lambda f: (f["covariance_type"], f["k"]))}, f, indent=2)
    print(f"Chosen: k={best['k']}, covariance_type={best['covariance_type']}; the lowest BIC is at "
          f"k={lowest['k']}, {choice['above_lowest_bic']} lower")
    return choice


def train_final_model(input_path, preprocess_path, k, covariance_type, run_dir) -> str:
    """Fit the chosen mixture on the first noise draw and save it behind that draw's preprocessing, so it
    takes plain RFM values."""
    with open(preprocess_path, "rb") as f:
        preprocess = pickle.load(f)
    gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, **GMM_KWARGS).fit(np.load(input_path)[0])

    model = Pipeline([("preprocess", preprocess), ("gmm", gmm)])
    model_path = Path(run_dir) / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return str(model_path)


def predict(model_path, features_path, run_dir) -> dict:
    """Give every customer a segment and its probability, and describe each segment.

    Writes segments.csv. Segments are listed by their share of revenue, largest first.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    rfm = pd.read_parquet(features_path)
    probabilities = model.predict_proba(rfm[FEATURES])
    rfm["segment"] = probabilities.argmax(axis=1)
    rfm["probability"] = probabilities.max(axis=1).round(4)
    rfm.to_csv(Path(run_dir) / "segments.csv")

    revenue = rfm["monetary"].sum()
    profiles = (
        rfm.groupby("segment")
        .agg(customers=("monetary", "size"), median_recency_days=("recency", "median"),
             median_invoices=("frequency", "median"), median_spend=("monetary", "median"),
             revenue=("monetary", "sum"), mean_probability=("probability", "mean"))
        .sort_values("revenue", ascending=False)
    )
    profiles["revenue_share"] = profiles.pop("revenue") / revenue
    print(profiles.round(3).to_string())

    gmm = model.named_steps["gmm"]
    return {
        "k": int(gmm.n_components),
        "covariance_type": gmm.covariance_type,
        "customers": len(rfm),
        "uncertain_customers": int((rfm["probability"] < MIN_CONFIDENCE).sum()),
        "segments": [{"segment": int(s), "customers": int(row["customers"]),
                      **{c: round(float(v), 3) for c, v in row.drop("customers").items()}}
                     for s, row in profiles.iterrows()],
    }
