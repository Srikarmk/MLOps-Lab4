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


@dag(
    dag_id="Airflow_Lab1",
    description="KMeans clustering with the elbow method",
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
    def data_preprocessing(raw_path: str) -> str:
        return lab.data_preprocessing(raw_path, run_dir())

    @task
    def build_save_model(scaled_path: str) -> dict:
        return lab.build_save_model(scaled_path, run_dir())

    @task
    def load_model_elbow(model: dict) -> dict:
        return lab.load_model_elbow(model["model_path"], model["sse"])

    # Calling the tasks wires both the dependencies and the XCom hand-offs
    load_model_elbow(build_save_model(data_preprocessing(load_data())))


pipeline = kmeans_pipeline()

if __name__ == "__main__":
    pipeline.test()
