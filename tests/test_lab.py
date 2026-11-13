"""Unit tests for the pipeline steps in src/lab.py. They run without Airflow's scheduler or database."""

import json

import numpy as np
import pandas as pd
import pytest

from src import lab


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory):
    return lab.load_data("online_retail_II.csv.gz", tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def features(raw_path, tmp_path_factory):
    return lab.build_features(raw_path, tmp_path_factory.mktemp("features"))


def test_real_data_passes_every_check(raw_path, tmp_path):
    report = lab.validate_data(raw_path, tmp_path, min_rows=100_000, max_missing_customer_fraction=0.30)
    assert report["passed"]
    assert report["checks"]["row_count"]["rows"] == 1_067_371
    assert report["checks"]["missing_customer_id"]["fraction_of_rows"] == pytest.approx(0.2277, abs=1e-4)
    # Normal in this data, so counted but not failed
    assert report["profile"]["cancellation_lines"] == 19_494
    assert json.loads((tmp_path / "quality_report.json").read_text()) == report


def test_bad_sample_fails_exactly_the_broken_checks(tmp_path):
    raw = lab.load_data("bad_sample.csv.gz", tmp_path)
    report = lab.validate_data(raw, tmp_path, min_rows=100_000, max_missing_customer_fraction=0.30)
    failed = {name for name, check in report["checks"].items() if not check["passed"]}
    assert not report["passed"]
    assert failed == {"parseable_dates", "invoice_format", "missing_customer_id"}
    assert report["checks"]["parseable_dates"]["unparseable"] == 25
    assert report["checks"]["invoice_format"]["examples"] == ["INV-0000"]


def test_missing_column_fails_without_crashing(tmp_path):
    raw = tmp_path / "raw.parquet"
    pd.DataFrame({"Invoice": ["489434"], "Quantity": [1]}).to_parquet(raw)
    report = lab.validate_data(raw, tmp_path, min_rows=1, max_missing_customer_fraction=0.30)
    assert not report["passed"]
    assert report["checks"]["required_columns"]["missing"] == ["StockCode", "InvoiceDate", "Price", "Customer ID", "Country"]


def test_published_versions_are_separate_files_with_a_content_hash(raw_path, tmp_path):
    first = lab.publish_dataset(raw_path, tmp_path / "datasets", "v1")
    second = lab.publish_dataset(raw_path, tmp_path / "datasets", "v2")
    assert first["path"] != second["path"]
    assert first["sha256"] == second["sha256"]  # same content, same hash
    assert first["rows"] == 1_067_371


def test_features_count_purchases_and_net_out_cancellations(features):
    # Every line is a purchase, a cancellation, or dropped for exactly one reason
    accounted = features["purchase_lines"] + features["cancellation_lines"] + sum(features["dropped_lines"].values())
    assert accounted == features["lines"]
    assert features["purchase_lines"] == 779_425
    assert features["customers"] == 5_837
    assert features["customers_without_net_spend"] == 41
    rfm = pd.read_parquet(features["features_path"])
    assert rfm.index.is_unique and len(rfm) == 5_837
    assert rfm["recency"].min() == 1 and rfm["frequency"].min() == 1 and (rfm["monetary"] > 0).all()
    # 12346 placed one huge order and cancelled it, so nothing is left to count
    assert "12346" not in rfm.index


def test_dequantize_spreads_counts_while_fitting_and_uses_the_midpoint_after():
    X = np.array([[1, 1, 10.25], [3, 2, 99.5]])
    fitted = lab.Dequantize().fit_transform(X)
    assert ((fitted[:, :2] >= X[:, :2]) & (fitted[:, :2] < X[:, :2] + 1)).all()
    assert (fitted[:, 2] == X[:, 2]).all()  # monetary is not a count, so it is left alone
    np.testing.assert_array_equal(lab.Dequantize().transform(X), X + [0.5, 0.5, 0])


def test_choose_model_takes_the_simplest_fit_close_to_the_lowest_bic(tmp_path):
    fits = [
        {"k": 2, "covariance_type": "full", "bic": 120.0, "converged": True},
        {"k": 3, "covariance_type": "diag", "bic": 104.0, "converged": True},
        {"k": 3, "covariance_type": "full", "bic": 100.0, "converged": True},
        {"k": 4, "covariance_type": "full", "bic": 95.0, "converged": True},  # the lowest
        {"k": 4, "covariance_type": "diag", "bic": 80.0, "converged": False},  # ignored: did not converge
        {"k": 5, "covariance_type": "full", "bic": 110.0, "converged": True},
    ]
    choice = lab.choose_model(fits, 5, tmp_path)
    # k=3 is within 10 of the lowest BIC (k=4), so the simpler model wins; full beats diag at k=3
    assert (choice["k"], choice["covariance_type"]) == (3, "full")
    assert (choice["lowest_bic_k"], choice["above_lowest_bic"]) == (4, 5.0)
    assert json.loads((tmp_path / "bic.json").read_text())["chosen"] == choice

    falling = [{"k": k, "covariance_type": "full", "bic": 100.0 - k, "converged": True} for k in range(1, 6)]
    with pytest.raises(ValueError, match="raise k_max"):
        lab.choose_model(falling, 5, tmp_path)


def test_model_applies_the_saved_preprocessing(features, tmp_path):
    prep = lab.data_preprocessing(features["features_path"], tmp_path)
    assert lab.fit_gmm(prep["input_path"], 3, "full") == lab.fit_gmm(prep["input_path"], 3, "full")  # seeded
    model_path = lab.train_final_model(prep["input_path"], prep["preprocess_path"], 3, "full", tmp_path)

    model = pd.read_pickle(model_path)
    rfm = pd.read_parquet(features["features_path"])[lab.FEATURES].head(300)
    transformed = model.named_steps["preprocess"].transform(rfm)
    # The saved model must give the same segments as preprocessing first and then the mixture
    np.testing.assert_array_equal(model.predict(rfm), model.named_steps["gmm"].predict(transformed))


def test_predict_segments_every_customer(features, tmp_path):
    prep = lab.data_preprocessing(features["features_path"], tmp_path)
    model_path = lab.train_final_model(prep["input_path"], prep["preprocess_path"], 4, "full", tmp_path)
    result = lab.predict(model_path, features["features_path"], tmp_path)

    segments = pd.read_csv(tmp_path / "segments.csv")
    assert len(segments) == sum(s["customers"] for s in result["segments"]) == 5_837
    assert segments["probability"].between(0, 1).all()
    assert (result["k"], result["covariance_type"]) == (4, "full")
    shares = [s["revenue_share"] for s in result["segments"]]
    assert shares == sorted(shares, reverse=True) and sum(shares) == pytest.approx(1, abs=0.01)
