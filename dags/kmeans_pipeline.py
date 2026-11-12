"""KMeans clustering pipeline, written with the TaskFlow API.

Tasks pass each other file paths and small results through XCom. The data itself is written to
/opt/airflow/working_data/<run id>/, which is mounted from ./working_data on the host.
"""

import re
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task

from src import lab

WORKING_DIR = Path("/opt/airflow/working_data")


def run_dir() -> Path:
    """A folder per DAG run, so two runs never overwrite each other's files."""
    run_id = get_current_context()["run_id"]
    path = WORKING_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Mapped KMeans fits allowed at once. Each is its own process (~0.2 GB), so this keeps the run inside
# a 4 GB Docker VM; raise it on a bigger machine.
MAX_PARALLEL_FITS = 8


@dag(
    dag_id="Airflow_Lab1",
    description="KMeans clustering: one mapped task per k, then the elbow's k is trained and used",
    start_date=pendulum.datetime(2026, 1, 15, tz="UTC"),
    schedule=None,  # triggered manually
    catchup=False,
    default_args={"owner": "Ramin Mohammadi", "retries": 0, "retry_delay": pendulum.duration(minutes=5)},
    tags=["clustering"],
)
def kmeans_pipeline():
    @task
    def load_data() -> str:
        return lab.load_data(run_dir())

    @task
    def data_preprocessing(raw_path: str) -> dict:
        return lab.data_preprocessing(raw_path, run_dir())

    @task(max_active_tis_per_dagrun=MAX_PARALLEL_FITS)
    def fit_kmeans(k: int, scaled_path: str) -> float:
        return lab.fit_kmeans(scaled_path, k)

    @task
    def choose_k(sse: list[float]) -> int:
        # The mapped results arrive in map-index order, which is the order of lab.K_VALUES
        return lab.choose_k(lab.K_VALUES, list(sse), run_dir())

    @task
    def train_final_model(prep: dict, k: int) -> str:
        return lab.train_final_model(prep["scaled_path"], prep["scaler_path"], k, run_dir())

    @task
    def predict(model_path: str, raw_path: str) -> dict:
        return lab.predict(model_path, raw_path, run_dir())

    raw = load_data()
    prep = data_preprocessing(raw)
    # Fan out: one fit_kmeans task instance per k, created at run time from the list
    sse = fit_kmeans.partial(scaled_path=prep["scaled_path"]).expand(k=lab.K_VALUES)
    # Fan in: choose_k receives all 49 results at once
    k = choose_k(sse)
    model = train_final_model(prep, k)
    predict(model, raw)


pipeline = kmeans_pipeline()

if __name__ == "__main__":
    pipeline.test()
