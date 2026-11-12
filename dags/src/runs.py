"""What both DAGs share: the Assets that connect them, per-run folders, retry defaults and the failure callback."""

import json
import re
from pathlib import Path

import pendulum
from airflow.sdk import Asset, get_current_context

WORKING_DIR = Path("/opt/airflow/working_data")
DATASETS_DIR = WORKING_DIR / "datasets"

# customer_data_ingest updates CUSTOMERS only when the data passes the quality gate; customer_clustering
# is scheduled on it, so it runs once per validated version and never on rejected data
CUSTOMERS = Asset(name="customers", uri="file:///opt/airflow/working_data/datasets/customers")
SEGMENTS = Asset(name="customer_segments", uri="file:///opt/airflow/working_data/datasets/segments")


def run_folder(run_id) -> Path:
    """A folder per DAG run, so two runs never overwrite each other's files."""
    path = WORKING_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir() -> Path:
    return run_folder(get_current_context()["run_id"])


def record_failure(context):
    """on_failure_callback: write the failed task and its error to failures/ in the run's folder."""
    ti = context["ti"]
    folder = run_folder(context["run_id"]) / "failures"
    folder.mkdir(exist_ok=True)
    name = ti.task_id if ti.map_index < 0 else f"{ti.task_id}[{ti.map_index}]"
    record = {
        "dag_id": ti.dag_id,
        "task_id": ti.task_id,
        "map_index": ti.map_index,
        "try_number": ti.try_number,
        "error": repr(context.get("exception")),
        "failed_at": pendulum.now("UTC").isoformat(),
    }
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2))


DEFAULT_ARGS = {
    "owner": "Ramin Mohammadi",
    # Transient failures (a full disk, a killed process) are retried; failed checks raise
    # AirflowFailException, which is never retried
    "retries": 2,
    "retry_delay": pendulum.duration(seconds=10),
    "execution_timeout": pendulum.duration(minutes=10),
    "on_failure_callback": record_failure,
}
