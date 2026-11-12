"""The pipeline's steps as plain functions that read and write files, so they run and test without Airflow.

Each step writes its data to the run's folder and returns only something small (a file path, a list of
numbers) for the DAG to hand to the next task through XCom.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
K_VALUES = range(1, 50)
KMEANS_KWARGS = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}


def load_data(run_dir) -> str:
    """Copy the raw CSV into the run's folder as Parquet. Returns its path."""
    df = pd.read_csv(DATA_DIR / "file.csv")
    out_path = Path(run_dir) / "raw.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def data_preprocessing(raw_path, run_dir) -> str:
    """Drop incomplete rows and min-max scale the clustering features. Returns the path of the scaled array."""
    df = pd.read_parquet(raw_path).dropna()
    scaled = MinMaxScaler().fit_transform(df[FEATURES])
    out_path = Path(run_dir) / "scaled.npy"
    np.save(out_path, scaled)
    return str(out_path)


def build_save_model(scaled_path, run_dir) -> dict:
    """Fit KMeans for every k in K_VALUES to get the SSE (inertia) curve, and save a model.

    Like the original lab, the saved model is the last one fitted (k=49), not the one the elbow picks.
    """
    data = np.load(scaled_path)
    sse = []
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, **KMEANS_KWARGS)
        kmeans.fit(data)
        sse.append(float(kmeans.inertia_))

    model_path = Path(run_dir) / "model.sav"
    with open(model_path, "wb") as f:
        pickle.dump(kmeans, f)
    return {"model_path": str(model_path), "sse": sse}


def load_model_elbow(model_path, sse) -> dict:
    """Find the elbow of the SSE curve and predict the cluster of the first row of test.csv."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    elbow = KneeLocator(list(K_VALUES), sse, curve="convex", direction="decreasing").elbow
    print(f"Optimal no. of clusters: {elbow}")

    # Like the original lab, this predicts on the raw (unscaled) test row
    test = pd.read_csv(DATA_DIR / "test.csv")
    prediction = int(model.predict(test)[0])
    return {"elbow_k": int(elbow), "prediction": prediction}
