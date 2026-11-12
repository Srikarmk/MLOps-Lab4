"""Consumer: clusters each new version of the `customers` Asset, then publishes `customer_segments`.

It has no time schedule. Airflow starts a run whenever customer_data_ingest publishes data that passed
its quality gate, and the run reads the exact version named in that asset event. Tasks pass each other
file paths and small results through XCom; the data itself stays in working_data/.

Trigger it by hand to re-cluster the latest published version, optionally with {"k_max": 30}.
"""

from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from src import lab
from src.runs import CUSTOMERS, DATASETS_DIR, DEFAULT_ARGS, SEGMENTS, run_dir

# Mapped KMeans fits allowed at once. Each is its own process (~0.2 GB), so this keeps the run inside
# a 4 GB Docker VM; raise it on a bigger machine.
MAX_PARALLEL_FITS = 8


@dag(
    dag_id="customer_clustering",
    description="KMeans on each new customers version: one mapped task per k, then the elbow's k is trained and used",
    start_date=pendulum.datetime(2026, 1, 15, tz="UTC"),
    schedule=[CUSTOMERS],  # data-aware: runs when the customers Asset is updated
    catchup=False,
    params={
        "k_min": Param(1, type="integer", minimum=1, description="smallest number of clusters to try"),
        "k_max": Param(49, type="integer", minimum=2, maximum=100, description="largest number of clusters to try"),
    },
    default_args=DEFAULT_ARGS,
    tags=["clustering", "consumer"],
)
def customer_clustering():
    @task
    def k_values() -> list[int]:
        """Check the k range before doing any work, and return the list of k to fit."""
        params = get_current_context()["params"]
        k_min, k_max = params["k_min"], params["k_max"]
        if k_max - k_min < 4:
            raise AirflowFailException(f"k_min={k_min}, k_max={k_max}: the elbow needs at least 5 values of k")
        return list(range(k_min, k_max + 1))

    @task
    def resolve_dataset() -> dict:
        """The customers version this run should cluster: the one in the triggering asset event, or for a
        manual run the newest published version."""
        events = get_current_context()["triggering_asset_events"][CUSTOMERS]
        if events:
            return dict(events[-1].extra)  # several updates while a run was queued: take the newest
        versions = sorted((DATASETS_DIR / "customers").glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        if not versions:
            raise AirflowFailException("No customers data published yet. Run customer_data_ingest first.")
        return {"path": str(versions[-1]), "producer_run_id": None}

    @task
    def data_preprocessing(dataset: dict) -> dict:
        print(f"Clustering {dataset['path']} (published by {dataset.get('producer_run_id') or 'an earlier run'})")
        return lab.data_preprocessing(dataset["path"], run_dir())

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

    @task(outlets=[SEGMENTS])
    def predict(model_path: str, dataset: dict) -> dict:
        result = lab.predict(model_path, dataset["path"], run_dir())
        context = get_current_context()
        context["outlet_events"][SEGMENTS].extra = {
            "k": result["k"],
            "clusters_csv": str(Path(model_path).parent / "clusters.csv"),
            "customers_version": Path(dataset["path"]).stem,
        }
        return result

    ks = k_values()
    dataset = resolve_dataset()
    ks >> dataset
    prep = data_preprocessing(dataset)

    # Fan out: one fit_kmeans task instance per k, created at run time from k_values' output
    sse = fit_kmeans.partial(scaled_path=prep["scaled_path"]).expand(k=ks)
    # Fan in: choose_k receives all the results at once
    k = choose_k(ks, sse)
    model = train_final_model(prep, k)
    predict(model, dataset)


pipeline = customer_clustering()

if __name__ == "__main__":
    pipeline.test()
