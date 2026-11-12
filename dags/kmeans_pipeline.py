"""KMeans clustering pipeline, written with the TaskFlow API.

Tasks pass each other file paths and small results through XCom. The data itself is written to
/opt/airflow/working_data/<run id>/, which is mounted from ./working_data on the host.

Trigger it with the defaults, or with a config such as {"input_file": "bad_sample.csv"} or {"k_max": 30}.
"""

import json
import re
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from src import lab

WORKING_DIR = Path("/opt/airflow/working_data")

# Mapped KMeans fits allowed at once. Each is its own process (~0.2 GB), so this keeps the run inside
# a 4 GB Docker VM; raise it on a bigger machine.
MAX_PARALLEL_FITS = 8


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
        "task_id": ti.task_id,
        "map_index": ti.map_index,
        "try_number": ti.try_number,
        "error": repr(context.get("exception")),
        "failed_at": pendulum.now("UTC").isoformat(),
    }
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2))


@dag(
    dag_id="Airflow_Lab1",
    description="KMeans clustering: a quality gate, one mapped task per k, then the elbow's k is trained and used",
    start_date=pendulum.datetime(2026, 1, 15, tz="UTC"),
    schedule=None,  # triggered manually
    catchup=False,
    # Validated when a run is triggered: a wrong type or an out-of-range value is rejected before anything runs
    params={
        "input_file": Param("file.csv", type="string", pattern=r"^[A-Za-z0-9_.-]+\.csv$",
                            description="CSV file in dags/data/ to cluster"),
        "k_min": Param(1, type="integer", minimum=1, description="smallest number of clusters to try"),
        "k_max": Param(49, type="integer", minimum=2, maximum=100, description="largest number of clusters to try"),
        "min_rows": Param(1000, type="integer", minimum=1, description="quality gate: fewest rows accepted"),
        "max_missing_fraction": Param(0.05, type="number", minimum=0, maximum=1,
                                      description="quality gate: largest fraction of rows with a blank value"),
    },
    default_args={
        "owner": "Ramin Mohammadi",
        # Transient failures (a full disk, a killed process) are retried; failed checks raise
        # AirflowFailException, which is never retried
        "retries": 2,
        "retry_delay": pendulum.duration(seconds=10),
        "execution_timeout": pendulum.duration(minutes=10),
        "on_failure_callback": record_failure,
    },
    tags=["clustering"],
)
def kmeans_pipeline():
    @task
    def k_values() -> list[int]:
        """Check the k range before doing any work, and return the list of k to fit."""
        params = get_current_context()["params"]
        k_min, k_max = params["k_min"], params["k_max"]
        if k_max - k_min < 4:
            raise AirflowFailException(f"k_min={k_min}, k_max={k_max}: the elbow needs at least 5 values of k")
        return list(range(k_min, k_max + 1))

    @task
    def load_data() -> str:
        return lab.load_data(get_current_context()["params"]["input_file"], run_dir())

    @task
    def validate_data(raw_path: str) -> dict:
        params = get_current_context()["params"]
        return lab.validate_data(raw_path, run_dir(), params["min_rows"], params["max_missing_fraction"])

    @task.branch
    def quality_gate(report: dict) -> str:
        """Choose which task runs next. The other one, and everything after it, is skipped."""
        return "data_preprocessing" if report["passed"] else "reject_data"

    @task
    def reject_data(report: dict) -> None:
        failed = {name: check for name, check in report["checks"].items() if not check["passed"]}
        raise AirflowFailException(f"Data-quality checks failed, nothing was trained: {failed}")

    @task
    def data_preprocessing(raw_path: str) -> dict:
        return lab.data_preprocessing(raw_path, run_dir())

    @task(max_active_tis_per_dagrun=MAX_PARALLEL_FITS, execution_timeout=pendulum.duration(minutes=2))
    def fit_kmeans(k: int, scaled_path: str) -> float:
        return lab.fit_kmeans(scaled_path, k)

    @task
    def choose_k(ks: list[int], sse: list[float]) -> int:
        # The mapped results arrive in map-index order, which is the order of ks
        return lab.choose_k(ks, list(sse), run_dir())

    @task
    def train_final_model(prep: dict, k: int) -> str:
        return lab.train_final_model(prep["scaled_path"], prep["scaler_path"], k, run_dir())

    @task
    def predict(model_path: str, raw_path: str) -> dict:
        return lab.predict(model_path, raw_path, run_dir())

    ks = k_values()
    raw = load_data()
    ks >> raw
    report = validate_data(raw)
    prep = data_preprocessing(raw)
    quality_gate(report) >> [prep, reject_data(report)]

    # Fan out: one fit_kmeans task instance per k, created at run time from k_values' output
    sse = fit_kmeans.partial(scaled_path=prep["scaled_path"]).expand(k=ks)
    # Fan in: choose_k receives all the results at once
    k = choose_k(ks, sse)
    model = train_final_model(prep, k)
    predict(model, raw)


pipeline = kmeans_pipeline()

if __name__ == "__main__":
    pipeline.test()
