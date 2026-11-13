# Airflow Lab: Customer Segmentation Pipeline

This lab segments the 5,837 customers of a UK online shop from 1,067,371 invoice lines ([Online Retail II](#the-data)), orchestrated by **Apache Airflow 3.3.2**. Each customer is described by recency, frequency and monetary value (RFM) and segmented with a **Gaussian mixture** whose size is chosen by **BIC**. It covers:
- two DAGs connected by an **Asset**: a daily ingest DAG that publishes data only after it passes a **data-quality gate**, and a segmentation DAG that runs whenever new data is published
- **dynamic task mapping** to fit one Gaussian mixture per (number of components, covariance type) in parallel
- the **TaskFlow API**, **Params**, **branching**, **retries**, **timeouts** and a **failure callback**
- **tests** for the DAGs and the pipeline code

It is based on the course's Airflow Lab 1 (`Labs/Airflow_Labs/Lab_1`), a single 4-task DAG running KMeans on credit-card data. The original write-up is kept unchanged in [ORIGINAL_LAB.md](ORIGINAL_LAB.md). That version doesn't run as shipped, and its model has two bugs. [What changed](#what-changed-from-the-original-lab) lists every change and why. The Airflow side was rebuilt first on the original data; the data and model were then replaced.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the whole lab in one command](#run-the-whole-lab-in-one-command)
- [Run it yourself in the Airflow UI](#run-it-yourself-in-the-airflow-ui)
- [The data](#the-data)
- [How the pipeline works](#how-the-pipeline-works)
- [What a successful run looks like](#what-a-successful-run-looks-like)
- [Results and findings](#results-and-findings)
- [What changed from the original lab](#what-changed-from-the-original-lab)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

You only need **Docker** and **Git**, plus a bash shell to run the two scripts. Airflow, Python and every Python package run inside the containers, so nothing else is installed on your machine. The data is in the repository, so the lab needs no download beyond the Docker images.

| Requirement | Version tested | Install |
|---|---|---|
| Docker Desktop (Docker Engine + Compose v2) | Engine 29.8.2, Compose 5.5.1 | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/). On Linux, Docker Engine plus the `docker compose` plugin is enough |
| Git | any | macOS: `xcode-select --install`; Ubuntu: `sudo apt install git`; Windows: `winget install Git.Git` |
| bash | any | built into macOS and Linux. On Windows, use **WSL 2** (recommended) or Git Bash |

**Docker resources.** Give Docker's VM at least **4 GB of memory**: Docker Desktop → Settings → Resources. The lab peaks at about 2.8 GB, and Airflow recommends 8 GB. You'll also need about **5 GB of free disk**: the built Airflow image is 3.3 GB and Postgres is 0.7 GB.

Check Docker is running:

```bash
docker version --format 'Engine {{.Server.Version}}' && docker compose version --short
```

This should print an engine version and a Compose version (2.x or later). If it says it can't connect to the Docker daemon, start Docker Desktop.

**What runs inside the image.** The `Dockerfile` builds on `apache/airflow:3.3.2-python3.12` and adds the packages in `requirements.txt`:

| Package | Version | Source | Used for |
|---|---|---|---|
| apache-airflow | 3.3.2 | base image (pinned again in the `Dockerfile`) | scheduler, API server, DAG processor |
| Python | 3.12.14 | base image | |
| pandas / numpy / pyarrow | 3.0.5 / 2.5.3 / 25.0.1 | base image | reading the data, Parquet files, RFM features |
| scikit-learn / scipy | 1.9.1 / 1.18.1 | base image | `GaussianMixture`, `StandardScaler`, pipelines |
| pytest | 9.1.1 | `requirements.txt` | the tests in `tests/` |
| Postgres | 16 | `postgres:16` image | Airflow's metadata database |

## Setup

```bash
git clone https://github.com/Srikarmk/MLOps-Lab4.git
cd MLOps-Lab4
./setup.sh
```

`setup.sh` does six things:
1. It checks that Docker is installed and running.
2. It removes any containers and metadata database left by a previous run.
3. It writes `.env` with your user ID, so files Airflow writes into `working_data/` and `logs/` belong to you.
4. It builds the image.
5. It migrates the database.
6. It starts Postgres, the API server, the scheduler and the DAG processor, then waits until all of them report healthy.

The first run downloads about 1.3 GB of images and takes a few minutes; later runs take about 35 seconds. It ends with:

```text
Airflow is running at http://localhost:8080 (no login needed)
```

Open http://localhost:8080. You should see the two DAGs, **retail_ingest** and **customer_segmentation**, both paused, with no import errors. If port 8080 is already in use, run `AIRFLOW_PORT=8081 ./setup.sh` and use that port instead.

> **No login, local only.** The UI and API are bound to `127.0.0.1` and every visitor is an admin (Airflow's `SimpleAuthManager` with `all_admins`). That suits a lab on your own machine; don't expose this setup to a network.

## Run the whole lab in one command

```bash
./run_lab.sh
```

`run_lab.sh` runs three stages and stops at the first failure:
1. **Fresh start:** runs `./setup.sh`.
2. **Tests:** runs the 20 pytest tests inside the Airflow image.
3. **End-to-end check:** runs `tests/end_to_end.py` against the live scheduler, which:
   - unpauses both DAGs
   - confirms the daily ingest run passes the quality gate and publishes the data
   - confirms that publish starts a segmentation run **by itself**, on exactly the version that was published
   - checks the segmentation run fits all 48 mixtures and trains the one `choose_model` picked
   - triggers an ingest of `bad_sample.csv.gz` and confirms it is rejected and starts **no** segmentation run

A full run takes about 2.5 minutes once the images are downloaded. Airflow is left running so you can look at the runs in the UI. `./run_lab.sh --down` stops it and deletes the database at the end instead. `AIRFLOW_PORT=8081 ./run_lab.sh` uses another port.

## Run it yourself in the Airflow UI

With Airflow running (from `./setup.sh`):

1. **Unpause `customer_segmentation`**, then **unpause `retail_ingest`**, using the toggles on the DAGs page. The ingest DAG is scheduled `@daily` with `catchup=False`, so unpausing it starts the latest daily run straight away.
2. Watch **retail_ingest** (about 10 s). Its Graph view shows `load_data → validate_data → quality_gate`, then `publish_dataset` running and `reject_data` skipped.
3. A run of **customer_segmentation** appears by itself, with run type *asset_triggered* (about 65 s). Its graph shows `fit_gmm [48]`; click it to see the 48 mapped task instances (12 values of k × 4 covariance types), 8 running at a time.
4. **Browse → Assets** shows `retail_transactions` (produced by ingest, consumed by segmentation) and `customer_segments`, with an event per publish and its metadata.
5. **Reject bad data:** on retail_ingest choose **Trigger** and set `input_file` to `bad_sample.csv.gz` in the form.
   - The run fails at `reject_data`, `publish_dataset` is skipped, and no segmentation run follows.
   - `working_data/<run id>/quality_report.json` says which checks failed.
6. **Change the search:** trigger customer_segmentation with `k_max` = `20`. It creates 80 mapped tasks instead of 48 and re-segments the newest published version.

The same from the command line:

```bash
docker compose run --rm airflow-cli airflow dags unpause customer_segmentation
docker compose run --rm airflow-cli airflow dags unpause retail_ingest
docker compose run --rm airflow-cli airflow dags trigger retail_ingest --conf '{"input_file": "bad_sample.csv.gz"}'
docker compose run --rm airflow-cli airflow dags list-runs customer_segmentation
```

Run the tests on their own:

```bash
docker compose run --rm --no-deps airflow-cli bash -c "cd /opt/airflow && python -m pytest tests -v"
```

Debug one DAG in a single process, with no scheduler: `docker compose exec airflow-scheduler python /opt/airflow/dags/segmentation_pipeline.py`. That calls `dag.test()`.

**Stopping:** `docker compose down` keeps the metadata database, and `docker compose up -d --wait` brings it back. `docker compose down -v` deletes it. Files in `working_data/` stay either way; delete the folder to clear them.

## The data

**Online Retail II** (UCI Machine Learning Repository, dataset 502): every line of every invoice of a UK online gift shop, many of whose customers are wholesalers, from 1 Dec 2009 to 9 Dec 2011. That's 1,067,371 lines from customers in 43 countries.

| Column | Meaning |
|---|---|
| `Invoice` | invoice number, 6 digits; `C` in front marks a cancellation, `A` a bad-debt adjustment |
| `StockCode`, `Description` | the product |
| `Quantity`, `Price` | units and unit price in pounds (negative quantities on cancellations) |
| `InvoiceDate` | date and time of the invoice |
| `Customer ID` | blank on 22.8% of lines (orders without a customer account) |
| `Country` | the customer's country |

- **Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed **CC BY 4.0**.
- **In this repository:** `dags/data/online_retail_II.csv.gz` (14.9 MB) holds both sheets of the UCI workbook as one CSV, with no rows changed. `tools/make_data.py` rebuilds it, and `bad_sample.csv.gz`, from the UCI download. You don't need to run it; its docstring has the steps.
- `dags/data/file.csv` and `test.csv` are the original lab's credit-card data. They are no longer used, and are kept for reference.

## How the pipeline works

```text
retail_ingest  (@daily, or triggered)
  load_data ──> validate_data ──> quality_gate ─┬─> publish_dataset ──> [Asset: retail_transactions]
                                                └─> reject_data   (fails the run; nothing is published)

[Asset: retail_transactions] ──> customer_segmentation  (no clock schedule: runs on each published version)
  k_values ──> resolve_dataset ──> build_features ──> data_preprocessing ──> fit_gmm ×N ──> choose_model ──> train_final_model ──> predict ──> [Asset: customer_segments]
                                    (RFM per customer)  (5 noise draws)   (k × covariance,  (BIC)        (preprocessing + mixture)
                                                                            8 at once)
```

**Data-quality gate** (`validate_data`, then `@task.branch quality_gate`). The checks look for a broken export. All five must pass for the data to be published:

| Check | Rule | The real data |
|---|---|---|
| `required_columns` | `Invoice`, `StockCode`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` present | present |
| `row_count` | at least `min_rows` lines (default 100,000) | 1,067,371 |
| `parseable_dates` | every `InvoiceDate` is a real `YYYY-MM-DD HH:MM:SS` | all, 2009-12-01 to 2011-12-09 |
| `invoice_format` | every `Invoice` is 6 digits with an optional letter in front | all |
| `missing_customer_id` | at most `max_missing_customer_fraction` of lines lack a customer (default 30%) | 22.8% |

Problems that are normal in this data are counted in the report's `profile`, not failed: 19,494 cancellation lines, 22,950 lines with quantity ≤ 0, 6,207 with price ≤ 0 and 34,335 exact duplicate lines. `build_features` deals with them.

`dags/data/bad_sample.csv.gz` is the first 110,000 lines of the real data, deliberately broken. It fails three checks: 55.8% of lines lack a customer, 25 dates are impossible (`2011-13-45 25:61:00`), and 12 invoice numbers are `INV-0000`.

**Assets.** `publish_dataset` writes an immutable copy to `working_data/datasets/transactions/<run id>.parquet` and records an asset event carrying its path, row count, content hash and source file. `customer_segmentation` is scheduled on that Asset (`schedule=[retail_transactions]`). Its `resolve_dataset` task reads the triggering event, so a run always uses the version that started it, even if a newer one is published meanwhile. A manual run uses the newest version. Rejected data emits no event, so it can never reach training.

**Features** (`build_features`). Invoice lines become one row per customer:

| Step | Lines or customers |
|---|---|
| Lines without a customer, dropped | 243,007 |
| Lines with price ≤ 0 (free items, adjustments), dropped | 71 |
| Exact duplicate lines, dropped | 26,478 |
| Purchase lines kept | 779,425 |
| Cancellation lines, **subtracted from spend** | 18,390 |
| Customers with no net spend left (they returned everything), dropped | 41 |
| Customers segmented | **5,837** |

- **Recency** is the number of days from the customer's last purchase to 10 Dec 2011, the day after the data ends, so the latest buyers get 1.
- **Frequency** is the number of purchase invoices.
- **Monetary** is net spend: purchases minus cancellations. Without the subtraction, customer 12346, who placed one order of £77,184 and cancelled it, would count as one of the biggest spenders.

**Preprocessing** (`data_preprocessing`) has three steps:
- **Dequantize:** recency and frequency are whole numbers, and 27% of customers bought exactly once. A Gaussian mixture can put a component of almost zero width on one exact value, which sends its likelihood, and so BIC, off without limit. Adding uniform noise in [0, 1) spreads each count over its unit interval and removes those spikes. A trained model uses the midpoint, n + 0.5, so its predictions are deterministic.
- **Log:** `log1p`, because frequency and spend are very skewed (spend runs from £2 to £570,000).
- **Standardize:** `StandardScaler`.

This is done for 5 independent noise draws, stacked into `model_inputs.npy`.

**Dynamic task mapping.** `fit_gmm.partial(input_path=…).expand(k=k_values, covariance_type=COVARIANCE_TYPES)` expands over two lists, so Airflow creates one task instance for every combination. That's 12 values of k (from the `k_min`…`k_max` Params) × 4 covariance types (`full`, `diag`, `tied`, `spherical`) = 48 tasks, created at run time. They run 8 at a time (`max_active_tis_per_dagrun`), each limited to one CPU thread so they don't compete for cores. Each fits the mixture on all 5 noise draws and returns the mean BIC and its spread.

**Choosing the model** (`choose_model`). It receives all 48 results at once (fan-in):
- If the lowest BIC is at `k_max`, BIC may still be falling, so the run fails and asks for a larger `k_max`.
- Otherwise it takes the **simplest model within 10 BIC of the lowest**: the fewest components, then the lowest BIC. A BIC difference under 10 is not strong evidence for the extra components, and here it is smaller than the noise between draws. [Results](#results-and-findings) shows why this matters.
- All 48 BICs are saved to `bic.json`.

**The model.** `train_final_model` fits the chosen mixture and saves it **behind its fitted preprocessing** as one scikit-learn pipeline (`Dequantize → log1p → StandardScaler → GaussianMixture`), so it takes plain RFM values. `predict` writes every customer's segment and its probability to `segments.csv`. It also describes each segment (size, median recency, invoices and spend, share of revenue), largest share of revenue first.

**Params** are validated when a run is triggered, so a wrong value is rejected with an error before anything runs:

| DAG | Param | Default | Rule |
|---|---|---|---|
| retail_ingest | `input_file` | `online_retail_II.csv.gz` | a `.csv` or `.csv.gz` name in `dags/data/` (no paths) |
| | `min_rows` | 100000 | ≥ 1 |
| | `max_missing_customer_fraction` | 0.30 | 0 to 1 |
| customer_segmentation | `k_min` | 1 | ≥ 1 |
| | `k_max` | 12 | 2 to 30, and at least 2 above `k_min` |

**Failure handling** (`dags/src/runs.py`):
- Every task retries twice, 10 s apart, and has a 10-minute timeout (2 minutes per `fit_gmm`).
- Failed quality checks and invalid k ranges raise `AirflowFailException`, which is never retried.
- `record_failure`, the `on_failure_callback`, writes the task, try number and error to `working_data/<run id>/failures/<task>.json`.

**XCom carries only small values:** file paths, the 48 BIC results and the summaries. The data stays in files. Each run gets its own folder:

```text
working_data/
├── datasets/transactions/<ingest run id>.parquet   published versions (never overwritten)
└── <run id>/
    ├── raw.parquet, quality_report.json           ingest
    ├── rfm.parquet                                segmentation: one row per customer
    ├── model_inputs.npy, preprocess.pkl, bic.json     the 5 preprocessed draws, the preprocessing, every fit's BIC
    ├── model.pkl, segments.csv                    the trained pipeline, every customer's segment and probability
    └── failures/<task>.json                       only if a task failed
```

## What a successful run looks like

The end of `./run_lab.sh`:

```text
==> Unit and DAG tests (pytest, inside the Airflow image)
....................                                                     [100%]
20 passed in 9.48s

==> End-to-end run through the scheduler
1. Unpausing both DAGs (the consumer first, so it is listening when data is published)
2. The daily ingest run starts by itself and publishes validated data
  ok   ingest run scheduled__2026-11-13T00:00:00+00:00 (scheduled) succeeded
  ok   the quality gate skipped reject_data
       published 1067371 rows, sha256 b3ce44532ec9: /opt/airflow/working_data/datasets/transactions/scheduled__....parquet
       quality checks: row_count passed, invoice_format passed, parseable_dates passed, required_columns passed, missing_customer_id passed
3. The publish starts customer_segmentation on that exact version
  ok   segmentation run asset_triggered__... was started by the asset event
  ok   segmentation run succeeded
  ok   it used the version that ingest published
       779425 purchase lines -> 5837 customers; 18390 cancellations netted out; dropped {'no_customer': 243007, 'duplicate_line': 26478, 'non_positive_price': 71, 'non_positive_quantity': 0}
  ok   48 mapped fit_gmm tasks succeeded
  ok   the final model is the chosen fit: k=6, full covariance (the lowest BIC is at k=10, 1.2 lower)
  ok   every customer got a segment
       2933 of 5837 customers have no segment with probability >= 0.8
       segment 2:   271 customers, 47.0% of revenue, median invoices 28, last bought 11 days ago
       segment 0:  1113 customers, 23.7% of revenue, median invoices 9, last bought 12 days ago
       segment 4:  1059 customers, 16.6% of revenue, median invoices 6, last bought 82 days ago
       segment 5:  1022 customers, 6.0% of revenue, median invoices 3, last bought 389.5 days ago
       segment 3:  1285 customers, 4.6% of revenue, median invoices 2, last bought 54 days ago
       segment 1:  1087 customers, 2.1% of revenue, median invoices 1, last bought 471 days ago
4. Bad data is rejected and starts nothing
  ok   the bad_sample.csv.gz ingest was rejected
  ok   nothing was published
       failed checks: ['invoice_format', 'parseable_dates', 'missing_customer_id']
       watching 45s for a segmentation run ...
  ok   no segmentation run was started

End-to-end check passed.
```

Run IDs and dates will differ, and the order of checks in a line can too. The counts, the chosen model, the segment sizes and the data hash should match: the noise draws and the mixtures are all seeded. If any check fails, the script prints `FAIL` next to it and exits with an error.

## Results and findings

**Six segments.** The run picks 6 components with full covariance. Ordered by share of revenue:

| Segment | Customers | Median days since last purchase | Median invoices | Median net spend | Share of revenue |
|---|---|---|---|---|---|
| Best customers | 271 | 11 | 28 | £14,687 | 47.0% |
| Regular, active | 1,113 | 12 | 9 | £2,542 | 23.7% |
| Regular, cooling off | 1,059 | 82 | 6 | £1,980 | 16.6% |
| Lapsed regulars | 1,022 | 390 | 3 | £716 | 6.0% |
| Recent, occasional | 1,285 | 54 | 2 | £414 | 4.6% |
| One-off, long gone | 1,087 | 471 | 1 | £215 | 2.1% |

The names are my reading of the medians; the model only numbers the segments. Revenue is very concentrated: **4.6% of customers bring in 47% of net spend**.

**How many segments the data supports.** BIC rules out 5 or fewer components: k = 5 is 61 worse than the best, and k = 4 is 400 worse. Above that, it can't tell k = 6 to 12 apart:
- **One noise draw is not enough.** With a single draw, the lowest BIC lands anywhere from k = 6 to k = 12 depending on the noise seed, and once exactly on k_max, which would fail the run.
- **The spread is bigger than the differences.** With 5 or more components, a fit's BIC varies by 8–47 between noise draws, more than the gaps between k = 6, 8 and 10.
- **Averaging alone doesn't settle it.** Four independent sets of 5 draws put the lowest averaged BIC at k = 10, 8, 7 and 6.
- **The simplest-within-10 rule is stable.** It picks k = 6 in all four sets, because k = 6 is always within 5.1 of the lowest. In this run the lowest BIC is at k = 10, just 1.2 below k = 6.

That is why `choose_model` averages over draws and prefers the simpler model. More segments than six would describe noise, not customers.

**Without dequantization, BIC is confidently wrong.** On the raw whole-number features, BIC picks k = 11 with diagonal covariance, ahead of every other k by 2,325. That is the spike artifact described in [preprocessing](#how-the-pipeline-works), and it looks far more decisive than the real answer.

**The segments overlap.** For half of the customers (2,933 of 5,837), no segment has a probability of 0.8 or more. RFM behaviour is a continuum rather than separate groups, so the boundaries between neighbouring segments are soft. A Gaussian mixture makes this visible, where KMeans would assign every customer with the same apparent certainty. `segments.csv` keeps each customer's probability so downstream use can filter on it.

**Full covariance wins clearly.** It beats the best diagonal, tied and spherical fits by 187 or more. Recency, frequency and spend are correlated (frequent buyers spend more and bought recently), and only full covariance can model that.

**Cancellations matter.** 41 customers have no net spend once cancellations are subtracted. Counting only purchases would have overstated their spend: customer 12346's by £77,184, an order cancelled 16 minutes after it was placed.

**Run time.** An ingest run takes about 10 s and a segmentation run about 65 s, most of it in the 48 × 5 mixture fits.

Findings about the original lab, from rebuilding it on its credit-card data before the swap:
- **It trained the right curve but saved the wrong model.** It fitted KMeans for k = 1…49 to draw the elbow (at k = 8), then saved the k = 49 model, the last one fitted.
- **It predicted on unscaled data.** The model was trained on min-max scaled features, but `test.csv` was passed in raw. For ordinary customers, predicting on raw values agreed with the correct cluster only **1.9%** of the time.
- **Its only test input was out of range.** `test.csv` has a credit limit of 241,514, 8× the largest in the data (30,000).
- **Its elbow depended on the range searched.** k = 1…30 gave an elbow at 5, and 1…49 gave 8.
- **XCom held the data.** The DataFrame went through XCom as base64-encoded pickles, 1.79 MB and 0.28 MB per run, stored in the metadata database.

## What changed from the original lab

| Area | Original lab | This version | Airflow concept |
|---|---|---|---|
| Running it | No `docker-compose.yaml`; the tutorial links the Airflow **2.5.1** compose file while the DAG uses Airflow 3 imports. Packages were to be installed at container start, and a 110 KB `airflow.cfg` dump was deleted by its own `setup.sh` | A pinned Airflow **3.3.2** `docker-compose.yaml` trimmed for a laptop (LocalExecutor + Postgres, no Redis or Celery), with packages baked into an image. `setup.sh` resets, builds, migrates and waits for health | Running Airflow on Docker |
| DAG style | `PythonOperator` with `op_args` and `.output` wiring | TaskFlow (`@dag`/`@task` from `airflow.sdk`), where calling the tasks defines the dependencies and the XCom hand-offs. The DAG file `airflow.py`, which could shadow the `airflow` package, is replaced | TaskFlow API |
| Passing data | Base64-encoded pickled DataFrames through XCom (about 2 MB per run in the database) | Files in a per-run folder; XCom carries paths and small results | What XCom is for |
| Model search | One task fitting k = 1…49 in a loop | 48 **mapped** task instances in parallel, expanded over k × covariance type, reduced by `choose_model` | Dynamic task mapping, fan-out/fan-in |
| Data | 8,950 credit-card customers, 3 columns used, one test row | 1,067,371 Online Retail II invoice lines turned into RFM features for 5,837 customers, with cancellations netted out | |
| Model | KMeans, k by the `kneed` elbow; saved the k = 49 model and predicted on unscaled data | Gaussian mixture, k and covariance type by BIC averaged over noise draws (simplest within 10 of the lowest), saved behind its preprocessing; gives each customer a segment probability | |
| Data quality | None | Five checks and a branch that publishes or rejects | Branching (`@task.branch`) |
| Configuration | Hard-coded file and k range | Params, validated at trigger time | Params |
| Failures | `retries: 0` | Retries, per-task timeouts, non-retryable `AirflowFailException` for failed checks, and an `on_failure_callback` | Retries, timeouts, callbacks |
| Scheduling | One manually triggered DAG | A daily ingest DAG and a segmentation DAG scheduled on the `retail_transactions` **Asset**, which publishes `customer_segments` | Assets (data-aware scheduling) |
| Tests | None | 20 pytest tests (DAG structure, Params, the pipeline functions) and an end-to-end check against the scheduler | DAG testing |
| Automation | Manual UI steps | `run_lab.sh`: fresh start, tests, end-to-end check | |

## Project layout

```text
MLOps-Lab4/
├── run_lab.sh                     whole lab end to end
├── setup.sh                       build and start Airflow from a clean state
├── docker-compose.yaml            Airflow 3.3.2 + Postgres (LocalExecutor)
├── Dockerfile                     apache/airflow:3.3.2-python3.12 + requirements.txt
├── requirements.txt               pytest
├── dags/
│   ├── ingest_pipeline.py         retail_ingest: load, check, publish the retail_transactions Asset
│   ├── segmentation_pipeline.py   customer_segmentation: RFM, mapped mixture fits, BIC choice, final model, segments
│   ├── src/
│   │   ├── lab.py                 the pipeline steps as plain functions (no Airflow imports)
│   │   └── runs.py                Assets, per-run folders, retry defaults, failure callback
│   └── data/
│       ├── online_retail_II.csv.gz    1,067,371 invoice lines (the data)
│       ├── bad_sample.csv.gz          deliberately broken sample for the quality gate
│       └── file.csv, test.csv         the original lab's data (not used)
├── tests/
│   ├── test_dags.py               the DAGs import and are wired as intended
│   ├── test_lab.py                the pipeline functions
│   └── end_to_end.py              run by run_lab.sh against the live scheduler
├── tools/make_data.py             rebuilds the two .csv.gz files from the UCI download (optional)
├── ORIGINAL_LAB.md                the original write-up (unchanged)
└── working_data/, logs/           created at run time (git-ignored)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Docker is installed but not running` | Start Docker Desktop and wait for it to say it's running |
| `port is already allocated` / `address already in use` on 8080 | Something else uses 8080 (a common default, e.g. an MLflow or Jupyter server). Run `AIRFLOW_PORT=8081 ./setup.sh` |
| A run stays **queued** and never starts | Its DAG is paused, and new DAGs start paused. Unpause it with the toggle or `airflow dags unpause <dag_id>` |
| No segmentation run after an ingest | Check the ingest run's `publish_dataset` succeeded (a rejected file publishes nothing), and that customer_segmentation is unpaused |
| `choose_model` fails with `The lowest BIC is at k_max` | BIC was still falling at the largest k tried. Trigger customer_segmentation with a larger `k_max` |
| Containers restart or tasks are killed (exit code 137) | Docker is out of memory. Give its VM at least 4 GB, or lower `MAX_PARALLEL_FITS` in `dags/segmentation_pipeline.py` |
| `Permission denied` writing `working_data/` or `logs/` (Linux) | Run `./setup.sh`, which writes your user ID to `.env`, and don't run it with `sudo` |
| `./setup.sh: Permission denied` | `chmod +x setup.sh run_lab.sh`, or run `bash setup.sh` |
| `$'\r': command not found` (Windows) | The scripts were checked out with Windows line endings. Clone inside WSL, or run `git config core.autocrlf input` and clone again (`.gitattributes` keeps `*.sh` as LF) |
| A DAG shows an import error | Open it in the UI or run `docker compose logs airflow-dag-processor`. The tests catch this too: `docker compose run --rm --no-deps airflow-cli bash -c "cd /opt/airflow && python -m pytest tests"` |
