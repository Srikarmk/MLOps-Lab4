"""Structure tests: both DAGs import cleanly and are wired the way the pipeline relies on."""

import pytest
from airflow.dag_processing.dagbag import DagBag
from airflow.sdk.definitions.timetables.assets import AssetTriggeredTimetable

from conftest import DAGS_DIR
from src.runs import CUSTOMERS, SEGMENTS, record_failure


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR))


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == {"customer_data_ingest", "customer_clustering"}


def test_ingest_gate_branches_to_publish_or_reject(dagbag):
    dag = dagbag.dags["customer_data_ingest"]
    assert dag.get_task("quality_gate").downstream_task_ids == {"publish_dataset", "reject_data"}
    assert dag.get_task("validate_data").upstream_task_ids == {"load_data"}


def test_only_publish_updates_the_customers_asset(dagbag):
    dag = dagbag.dags["customer_data_ingest"]
    producers = [t.task_id for t in dag.tasks if CUSTOMERS in t.outlets]
    assert producers == ["publish_dataset"]


def test_clustering_is_scheduled_on_the_customers_asset(dagbag):
    dag = dagbag.dags["customer_clustering"]
    assert isinstance(dag.timetable, AssetTriggeredTimetable)
    assert dag.timetable.asset_condition.objects == (CUSTOMERS,)
    assert SEGMENTS in dag.get_task("predict").outlets


def test_kmeans_fits_are_mapped_and_reduced_by_choose_k(dagbag):
    dag = dagbag.dags["customer_clustering"]
    fit = dag.get_task("fit_kmeans")
    assert fit.is_mapped
    assert fit.max_active_tis_per_dagrun == 8
    assert fit.downstream_task_ids == {"choose_k"}
    assert dag.get_task("train_final_model").upstream_task_ids == {"data_preprocessing", "choose_k"}


@pytest.mark.parametrize("dag_id", ["customer_data_ingest", "customer_clustering"])
def test_every_task_retries_times_out_and_reports_failures(dagbag, dag_id):
    for task in dagbag.dags[dag_id].tasks:
        assert task.retries == 2, task.task_id
        assert task.execution_timeout is not None, task.task_id
        # Mapped tasks keep the callback as a function, the others as a list of callbacks
        callbacks = task.on_failure_callback if isinstance(task.on_failure_callback, list) else [task.on_failure_callback]
        assert callbacks == [record_failure], task.task_id


@pytest.mark.parametrize(
    ("dag_id", "param", "bad_value"),
    [
        ("customer_data_ingest", "input_file", "../../etc/passwd"),
        ("customer_data_ingest", "max_missing_fraction", 1.5),
        ("customer_clustering", "k_max", "lots"),
        ("customer_clustering", "k_min", 0),
    ],
)
def test_params_reject_bad_values(dagbag, dag_id, param, bad_value):
    with pytest.raises(Exception, match="Invalid input|is not|greater than|less than|does not match"):
        dagbag.dags[dag_id].params.get_param(param).resolve(value=bad_value)
