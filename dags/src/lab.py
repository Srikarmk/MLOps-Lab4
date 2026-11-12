"""The pipeline's steps as plain functions that read and write files, so they run and test without Airflow.

Each step writes its data to the run's folder and returns only something small (a file path, a number,
a few counts) for the DAG to hand to the next task through XCom.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
K_VALUES = list(range(1, 50))
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


def load_data(run_dir) -> str:
    """Copy the raw CSV into the run's folder as Parquet. Returns its path."""
    df = pd.read_csv(DATA_DIR / "file.csv")
    out_path = Path(run_dir) / "raw.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


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


def predict(model_path, raw_path, run_dir) -> dict:
    """Assign every customer to a cluster, and predict the cluster of the row in test.csv."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    customers = pd.read_parquet(raw_path).dropna()
    customers["cluster"] = model.predict(customers[FEATURES])
    customers[["CUST_ID", *FEATURES, "cluster"]].to_csv(Path(run_dir) / "clusters.csv", index=False)

    test = pd.read_csv(DATA_DIR / "test.csv")
    test_cluster = int(model.predict(test[FEATURES])[0])
    print(f"test.csv row {test.iloc[0].to_dict()} -> cluster {test_cluster}")

    sizes = customers["cluster"].value_counts().sort_index()
    return {
        "k": int(model.named_steps["kmeans"].n_clusters),
        "test_cluster": test_cluster,
        "cluster_sizes": {str(c): int(n) for c, n in sizes.items()},
    }
