"""Consumer: segments the customers in each new version of `retail_transactions`, then publishes
`customer_segments`.

It has no time schedule. Airflow starts a run whenever retail_ingest publishes data that passed its
quality gate, and the run reads the exact version named in that asset event. Tasks pass each other file
paths and small results through XCom; the data itself stays in working_data/.

Trigger it by hand to re-segment the latest published version, optionally with {"k_max": 20}.
"""

from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from src import lab
from src.runs import DATASETS_DIR, DEFAULT_ARGS, SEGMENTS, TRANSACTIONS, run_dir

# Mapped mixture fits allowed at once. Each is its own process (~0.2 GB), so this keeps the run inside
# a 4 GB Docker VM; raise it on a bigger machine.
MAX_PARALLEL_FITS = 8


@dag(
    dag_id="customer_segmentation",
    description="RFM features per customer, then a Gaussian mixture chosen by BIC from one mapped fit per (k, covariance type)",
    start_date=pendulum.datetime(2026, 1, 15, tz="UTC"),
    schedule=[TRANSACTIONS],  # data-aware: runs when the retail_transactions Asset is updated
    catchup=False,
    params={
        "k_min": Param(1, type="integer", minimum=1, description="fewest mixture components to try"),
        "k_max": Param(12, type="integer", minimum=2, maximum=30, description="most mixture components to try"),
    },
    default_args=DEFAULT_ARGS,
    tags=["segmentation", "consumer"],
)
def customer_segmentation():
    @task
    def k_values() -> list[int]:
        """Check the k range before doing any work, and return the list of k to fit."""
        params = get_current_context()["params"]
        k_min, k_max = params["k_min"], params["k_max"]
        if k_max - k_min < 2:
            raise AirflowFailException(f"k_min={k_min}, k_max={k_max}: comparing BIC needs at least 3 values of k")
        return list(range(k_min, k_max + 1))

    @task
    def resolve_dataset() -> dict:
        """The transactions version this run should use: the one in the triggering asset event, or for a
        manual run the newest published version."""
        events = get_current_context()["triggering_asset_events"][TRANSACTIONS]
        if events:
            return dict(events[-1].extra)  # several updates while a run was queued: take the newest
        versions = sorted((DATASETS_DIR / "transactions").glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        if not versions:
            raise AirflowFailException("No transactions published yet. Run retail_ingest first.")
        return {"path": str(versions[-1]), "producer_run_id": None}

    @task
    def build_features(dataset: dict) -> dict:
        print(f"Using {dataset['path']} (published by {dataset.get('producer_run_id') or 'an earlier run'})")
        return lab.build_features(dataset["path"], run_dir())

    @task
    def data_preprocessing(features: dict) -> dict:
        return lab.data_preprocessing(features["features_path"], run_dir())

    @task(max_active_tis_per_dagrun=MAX_PARALLEL_FITS, execution_timeout=pendulum.duration(minutes=2))
    def fit_gmm(k: int, covariance_type: str, input_path: str) -> dict:
        return lab.fit_gmm(input_path, k, covariance_type)

    @task
    def choose_model(fits: list[dict], ks: list[int]) -> dict:
        return lab.choose_model(list(fits), max(ks), run_dir())

    @task
    def train_final_model(prep: dict, choice: dict) -> str:
        return lab.train_final_model(prep["input_path"], prep["preprocess_path"],
                                     choice["k"], choice["covariance_type"], run_dir())

    @task(outlets=[SEGMENTS])
    def predict(model_path: str, features: dict, dataset: dict) -> dict:
        result = lab.predict(model_path, features["features_path"], run_dir())
        context = get_current_context()
        context["outlet_events"][SEGMENTS].extra = {
            "k": result["k"],
            "covariance_type": result["covariance_type"],
            "segments_csv": str(Path(model_path).parent / "segments.csv"),
            "transactions_version": Path(dataset["path"]).stem,
        }
        return result

    ks = k_values()
    dataset = resolve_dataset()
    ks >> dataset
    features = build_features(dataset)
    prep = data_preprocessing(features)

    # Fan out: one fit_gmm task instance for every combination of k and covariance type, created at run
    # time from k_values' output (expand over two lists = their cross product)
    fits = fit_gmm.partial(input_path=prep["input_path"]).expand(k=ks, covariance_type=lab.COVARIANCE_TYPES)
    # Fan in: choose_model receives all the results at once
    choice = choose_model(fits, ks)
    model = train_final_model(prep, choice)
    predict(model, features, dataset)


pipeline = customer_segmentation()

if __name__ == "__main__":
    pipeline.test()
