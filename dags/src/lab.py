"""The pipeline's steps as plain functions that read and write files, so they run and test without Airflow.

Each step writes its data to the run's folder and returns only something small (a file path, a number,
a few counts) for the DAG to hand to the next task through XCom.
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ID_COLUMN = "CUST_ID"
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


def load_data(input_file, run_dir) -> str:
    """Copy a CSV from the data folder into the run's folder as Parquet. Returns its path."""
    df = pd.read_csv(DATA_DIR / input_file)
    out_path = Path(run_dir) / "raw.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def validate_data(raw_path, run_dir, min_rows, max_missing_fraction) -> dict:
    """Run the data-quality checks and save them to quality_report.json. Returns the report."""
    df = pd.read_parquet(raw_path)
    missing_columns = [c for c in [ID_COLUMN, *FEATURES] if c not in df.columns]
    checks = {"required_columns": {"passed": not missing_columns, "missing": missing_columns}}

    if not missing_columns:
        incomplete = float(df.isna().any(axis=1).mean())
        negative = {c: int((df[c] < 0).sum()) for c in FEATURES}
        duplicates = int(df[ID_COLUMN].duplicated().sum())
        checks.update({
            "row_count": {"passed": len(df) >= min_rows, "rows": len(df), "min_rows": min_rows},
            "missing_values": {"passed": incomplete <= max_missing_fraction,
                               "fraction_of_rows": round(incomplete, 4), "max": max_missing_fraction},
            "non_negative": {"passed": not any(negative.values()), "negative_values": negative},
            "unique_ids": {"passed": duplicates == 0, "duplicates": duplicates},
        })

    report = {"passed": all(c["passed"] for c in checks.values()), "checks": checks}
    with open(Path(run_dir) / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def publish_dataset(raw_path, datasets_dir, version) -> dict:
    """Copy validated data to datasets/customers/<version>.parquet, a file that is never overwritten.

    Returns what the asset event records: where the version is, how many rows it has and its content hash.
    """
    out_dir = Path(datasets_dir) / "customers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{version}.parquet"
    shutil.copyfile(raw_path, out_path)
    return {
        "path": str(out_path),
        "rows": len(pd.read_parquet(out_path)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()[:12],
    }


def data_preprocessing(raw_path, run_dir) -> dict:
    """Drop incomplete rows and min-max scale the clustering features.

    The fitted scaler is saved too: new data has to be scaled the same way before it is clustered.
    """
    df = pd.read_parquet(raw_path).dropna()
    scaler = MinMaxScaler().fit(df[FEATURES])

    scaled_path = Path(run_dir) / "scaled.npy"
    np.save(scaled_path, scaler.transform(df[FEATURES]))
    scaler_path = Path(run_dir) / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    return {"scaled_path": str(scaled_path), "scaler_path": str(scaler_path)}


def fit_kmeans(scaled_path, k) -> float:
    """Fit KMeans with k clusters and return its SSE (inertia). One call per k, run in parallel by the DAG."""
    data = np.load(scaled_path)
    # Many of these run at once; letting each use every core makes them fight over the CPU
    with threadpool_limits(limits=1):
        kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(data)
    return float(kmeans.inertia_)


def choose_k(k_values, sse, run_dir) -> int:
    """Pick k at the elbow of the SSE curve, and save the curve for reference."""
    elbow = KneeLocator(k_values, sse, curve="convex", direction="decreasing").elbow
    if elbow is None:
        raise ValueError("The SSE curve has no elbow; widen the range of k")
    with open(Path(run_dir) / "sse.json", "w") as f:
        json.dump({"k": k_values, "sse": sse, "elbow_k": int(elbow)}, f, indent=2)
    print(f"Optimal no. of clusters: {elbow}")
    return int(elbow)


def train_final_model(scaled_path, scaler_path, k, run_dir) -> str:
    """Fit the final KMeans with the chosen k and save it behind its scaler, so it takes raw feature values."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(np.load(scaled_path))

    model = Pipeline([("scaler", scaler), ("kmeans", kmeans)])
    model_path = Path(run_dir) / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return str(model_path)


def out_of_range_features(scaler, row) -> list:
    """Features of a row that fall outside the range seen in training, where the clusters say little."""
    scaled = scaler.transform(row[FEATURES])[0]
    return [f for f, v in zip(FEATURES, scaled) if v < 0 or v > 1]


def predict(model_path, raw_path, run_dir) -> dict:
    """Assign every customer to a cluster, and predict the cluster of the row in test.csv."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    customers = pd.read_parquet(raw_path).dropna()
    customers["cluster"] = model.predict(customers[FEATURES])
    customers[[ID_COLUMN, *FEATURES, "cluster"]].to_csv(Path(run_dir) / "clusters.csv", index=False)

    test = pd.read_csv(DATA_DIR / "test.csv")
    test_cluster = int(model.predict(test[FEATURES])[0])
    out_of_range = out_of_range_features(model.named_steps["scaler"], test)
    print(f"test.csv row {test.iloc[0].to_dict()} -> cluster {test_cluster}")
    if out_of_range:
        print(f"WARNING: {out_of_range} outside the training range, so this cluster is unreliable")

    sizes = customers["cluster"].value_counts().sort_index()
    return {
        "k": int(model.named_steps["kmeans"].n_clusters),
        "test_cluster": test_cluster,
        "test_out_of_range": out_of_range,
        "cluster_sizes": {str(c): int(n) for c, n in sizes.items()},
    }
