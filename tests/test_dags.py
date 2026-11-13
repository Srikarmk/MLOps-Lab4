"""Structure tests: both DAGs import cleanly and are wired the way the pipeline relies on."""

import pytest
from airflow.dag_processing.dagbag import DagBag
from airflow.sdk.definitions.timetables.assets import AssetTriggeredTimetable

from conftest import DAGS_DIR
from src.lab import COVARIANCE_TYPES
from src.runs import SEGMENTS, TRANSACTIONS, record_failure


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR))


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dags) == {"retail_ingest", "customer_segmentation"}


def test_ingest_gate_branches_to_publish_or_reject(dagbag):
    dag = dagbag.dags["retail_ingest"]
    assert dag.get_task("quality_gate").downstream_task_ids == {"publish_dataset", "reject_data"}
    assert dag.get_task("validate_data").upstream_task_ids == {"load_data"}


def test_only_publish_updates_the_transactions_asset(dagbag):
    dag = dagbag.dags["retail_ingest"]
    producers = [t.task_id for t in dag.tasks if TRANSACTIONS in t.outlets]
    assert producers == ["publish_dataset"]


def test_segmentation_is_scheduled_on_the_transactions_asset(dagbag):
    dag = dagbag.dags["customer_segmentation"]
    assert isinstance(dag.timetable, AssetTriggeredTimetable)
    assert dag.timetable.asset_condition.objects == (TRANSACTIONS,)
    assert SEGMENTS in dag.get_task("predict").outlets


def test_mixture_fits_are_mapped_over_k_and_covariance_type_and_reduced_by_choose_model(dagbag):
    dag = dagbag.dags["customer_segmentation"]
    fit = dag.get_task("fit_gmm")
    assert fit.is_mapped
    assert set(fit.op_kwargs_expand_input.value) == {"k", "covariance_type"}  # expanded over both: their cross product
    assert fit.op_kwargs_expand_input.value["covariance_type"] == COVARIANCE_TYPES
    assert fit.max_active_tis_per_dagrun == 8
    assert fit.downstream_task_ids == {"choose_model"}
    assert dag.get_task("train_final_model").upstream_task_ids == {"data_preprocessing", "choose_model"}
    assert dag.get_task("build_features").upstream_task_ids == {"resolve_dataset"}


@pytest.mark.parametrize("dag_id", ["retail_ingest", "customer_segmentation"])
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
        ("retail_ingest", "input_file", "../../etc/passwd"),
        ("retail_ingest", "max_missing_customer_fraction", 1.5),
        ("customer_segmentation", "k_max", "lots"),
        ("customer_segmentation", "k_min", 0),
    ],
)
def test_params_reject_bad_values(dagbag, dag_id, param, bad_value):
    with pytest.raises(Exception, match="Invalid input|is not|greater than|less than|does not match"):
        dagbag.dags[dag_id].params.get_param(param).resolve(value=bad_value)
