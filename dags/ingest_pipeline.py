"""Producer: loads the customer data, checks it and, only if it passes, publishes it as the `customers` Asset.

Publishing writes an immutable copy under working_data/datasets/customers/ and records an asset event
carrying its path, row count and hash. That event is what starts customer_clustering.

Trigger it with the defaults, or with a config such as {"input_file": "bad_sample.csv"}.
"""

import re

import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from src import lab
from src.runs import CUSTOMERS, DATASETS_DIR, DEFAULT_ARGS, run_dir


@dag(
    dag_id="customer_data_ingest",
    description="Load and validate the customer data, then publish it as the customers Asset",
    start_date=pendulum.datetime(2026, 1, 15, tz="UTC"),
    schedule="@daily",
    catchup=False,
    # Validated when a run is triggered: a wrong type or an out-of-range value is rejected before anything runs
    params={
        "input_file": Param("file.csv", type="string", pattern=r"^[A-Za-z0-9_.-]+\.csv$",
                            description="CSV file in dags/data/ to load"),
        "min_rows": Param(1000, type="integer", minimum=1, description="quality gate: fewest rows accepted"),
        "max_missing_fraction": Param(0.05, type="number", minimum=0, maximum=1,
                                      description="quality gate: largest fraction of rows with a blank value"),
    },
    default_args=DEFAULT_ARGS,
    tags=["clustering", "producer"],
)
def customer_data_ingest():
    @task
    def load_data() -> str:
        return lab.load_data(get_current_context()["params"]["input_file"], run_dir())

    @task
    def validate_data(raw_path: str) -> dict:
        params = get_current_context()["params"]
        return lab.validate_data(raw_path, run_dir(), params["min_rows"], params["max_missing_fraction"])

    @task.branch
    def quality_gate(report: dict) -> str:
        """Choose which task runs next. The other one is skipped."""
        return "publish_dataset" if report["passed"] else "reject_data"

    @task
    def reject_data(report: dict) -> None:
        failed = {name: check for name, check in report["checks"].items() if not check["passed"]}
        raise AirflowFailException(f"Data-quality checks failed, nothing was published: {failed}")

    @task(outlets=[CUSTOMERS])
    def publish_dataset(raw_path: str) -> dict:
        context = get_current_context()
        version = re.sub(r"[^A-Za-z0-9_.-]", "_", context["run_id"])
        published = lab.publish_dataset(raw_path, DATASETS_DIR, version)
        # Attached to the asset event, so the consumer knows exactly which version started it
        context["outlet_events"][CUSTOMERS].extra = {
            **published,
            "source_file": context["params"]["input_file"],
            "producer_run_id": context["run_id"],
        }
        return published

    raw = load_data()
    report = validate_data(raw)
    quality_gate(report) >> [publish_dataset(raw), reject_data(report)]


pipeline = customer_data_ingest()

if __name__ == "__main__":
    pipeline.test()
