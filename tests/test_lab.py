"""Unit tests for the pipeline steps in src/lab.py. They run without Airflow's scheduler or database."""

import json

import numpy as np
import pandas as pd
import pytest

from src import lab


@pytest.fixture
def raw_path(tmp_path):
    return lab.load_data("file.csv", tmp_path)


def test_real_data_passes_every_check(raw_path, tmp_path):
    report = lab.validate_data(raw_path, tmp_path, min_rows=1000, max_missing_fraction=0.05)
    assert report["passed"]
    assert report["checks"]["missing_values"]["fraction_of_rows"] == pytest.approx(0.0351, abs=1e-4)
    assert json.loads((tmp_path / "quality_report.json").read_text()) == report


def test_bad_sample_fails_exactly_the_broken_checks(tmp_path):
    raw = lab.load_data("bad_sample.csv", tmp_path)
    report = lab.validate_data(raw, tmp_path, min_rows=1000, max_missing_fraction=0.05)
    failed = {name for name, check in report["checks"].items() if not check["passed"]}
    assert not report["passed"]
    assert failed == {"missing_values", "non_negative", "unique_ids"}
    assert report["checks"]["unique_ids"]["duplicates"] == 3
    assert report["checks"]["non_negative"]["negative_values"]["PURCHASES"] == 5


def test_missing_column_fails_without_crashing(tmp_path):
    raw = tmp_path / "raw.parquet"
    pd.DataFrame({"CUST_ID": ["a"], "BALANCE": [1.0]}).to_parquet(raw)
    report = lab.validate_data(raw, tmp_path, min_rows=1, max_missing_fraction=0.05)
    assert not report["passed"]
    assert report["checks"]["required_columns"]["missing"] == ["PURCHASES", "CREDIT_LIMIT"]


def test_published_versions_are_separate_files_with_a_content_hash(raw_path, tmp_path):
    first = lab.publish_dataset(raw_path, tmp_path / "datasets", "v1")
    second = lab.publish_dataset(raw_path, tmp_path / "datasets", "v2")
    assert first["path"] != second["path"]
    assert first["sha256"] == second["sha256"]  # same content, same hash
    assert first["rows"] == 8950


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")  # kneed, on the flat curve
def test_choose_k_finds_the_elbow_and_rejects_a_curve_without_one(tmp_path):
    ks = list(range(1, 11))
    sse = [100, 40, 15, 12, 10, 9, 8, 7.5, 7, 6.8]
    assert lab.choose_k(ks, sse, tmp_path) == 3
    assert json.loads((tmp_path / "sse.json").read_text())["elbow_k"] == 3
    with pytest.raises(ValueError, match="no elbow"):
        lab.choose_k(ks, [5.0] * 10, tmp_path)


def test_model_scales_raw_values_before_clustering(raw_path, tmp_path):
    prep = lab.data_preprocessing(raw_path, tmp_path)
    assert lab.fit_kmeans(prep["scaled_path"], 3) == lab.fit_kmeans(prep["scaled_path"], 3)  # seeded
    model_path = lab.train_final_model(prep["scaled_path"], prep["scaler_path"], 3, tmp_path)

    model = pd.read_pickle(model_path)
    raw = pd.read_parquet(raw_path).dropna()[lab.FEATURES].head(200)
    scaled = model.named_steps["scaler"].transform(raw)
    # The saved pipeline must give the same clusters as scaling first and then clustering
    np.testing.assert_array_equal(model.predict(raw), model.named_steps["kmeans"].predict(scaled))


def test_predict_writes_every_customer_and_flags_the_out_of_range_test_row(raw_path, tmp_path):
    prep = lab.data_preprocessing(raw_path, tmp_path)
    model_path = lab.train_final_model(prep["scaled_path"], prep["scaler_path"], 4, tmp_path)
    result = lab.predict(model_path, raw_path, tmp_path)

    clusters = pd.read_csv(tmp_path / "clusters.csv")
    assert len(clusters) == sum(result["cluster_sizes"].values()) == 8636  # 8950 rows minus 314 incomplete
    assert result["k"] == 4
    # test.csv has CREDIT_LIMIT 241,514; the largest in the data is 30,000
    assert result["test_out_of_range"] == ["CREDIT_LIMIT"]
