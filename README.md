# Airflow Lab: Customer Segmentation Pipeline

This lab segments 8,950 credit-card customers with KMeans, orchestrated by **Apache Airflow 3.3.2**. It covers:
- two DAGs connected by an **Asset**: a daily ingest DAG that publishes data only after it passes a **data-quality gate**, and a clustering DAG that runs whenever new data is published
- **dynamic task mapping** to fit one KMeans model per number of clusters in parallel
- the **TaskFlow API**, **Params**, **branching**, **retries**, **timeouts** and a **failure callback**
- **tests** for the DAGs and the pipeline code

It is based on the course's Airflow Lab 1 (`Labs/Airflow_Labs/Lab_1`), a single 4-task DAG. The original write-up is kept unchanged in [ORIGINAL_LAB.md](ORIGINAL_LAB.md). That version doesn't run as shipped, and its model has two bugs; [What changed](#what-changed-from-the-original-lab) lists every change and why.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Run the whole lab in one command](#run-the-whole-lab-in-one-command)
- [Run it yourself in the Airflow UI](#run-it-yourself-in-the-airflow-ui)
- [How the pipeline works](#how-the-pipeline-works)
- [What a successful run looks like](#what-a-successful-run-looks-like)
- [Results and findings](#results-and-findings)
- [What changed from the original lab](#what-changed-from-the-original-lab)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## Prerequisites

You only need **Docker** and **Git**, plus a bash shell to run the two scripts. Airflow, Python and every Python package run inside the containers, so nothing else is installed on your machine.

| Requirement | Version tested | Install |
|---|---|---|
| Docker Desktop (Docker Engine + Compose v2) | Engine 29.8.2, Compose 5.5.1 | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/). On Linux, Docker Engine plus the `docker compose` plugin is enough |
| Git | any | macOS: `xcode-select --install`; Ubuntu: `sudo apt install git`; Windows: `winget install Git.Git` |
| bash | any | built into macOS and Linux. On Windows, use **WSL 2** (recommended) or Git Bash |

**Docker resources.** Give Docker's VM at least **4 GB of memory**: Docker Desktop → Settings → Resources. The lab peaks at about 2.7 GB, and Airflow recommends 8 GB. You'll also need about **5 GB of free disk**: the built Airflow image is 3.3 GB and Postgres is 0.7 GB.

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
| pandas / numpy / scikit-learn / scipy | 3.0.5 / 2.5.3 / 1.9.1 / 1.18.1 | base image | data handling, MinMaxScaler, KMeans |
| kneed | 0.8.5 | `requirements.txt` | finding the elbow of the SSE curve |
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

Open http://localhost:8080. You should see the two DAGs, **customer_data_ingest** and **customer_clustering**, both paused, with no import errors. If port 8080 is already in use, run `AIRFLOW_PORT=8081 ./setup.sh` and use that port instead.

> **No login, local only.** The UI and API are bound to `127.0.0.1` and every visitor is an admin (Airflow's `SimpleAuthManager` with `all_admins`). That suits a lab on your own machine; don't expose this setup to a network.

## Run the whole lab in one command

```bash
./run_lab.sh
```

`run_lab.sh` runs three stages and stops at the first failure:
1. **Fresh start:** runs `./setup.sh`.
2. **Tests:** runs the 18 pytest tests inside the Airflow image.
3. **End-to-end check:** runs `tests/end_to_end.py` against the live scheduler, which:
   - unpauses both DAGs
   - confirms the daily ingest run passes the quality gate and publishes the data
   - confirms that publish starts a clustering run **by itself**, on exactly the version that was published
   - checks the clustering run fits all 49 models and trains with the elbow's k
   - triggers an ingest of `bad_sample.csv` and confirms it is rejected and starts **no** clustering run

A full run takes about 3 minutes once the images are downloaded. Airflow is left running so you can look at the runs in the UI. `./run_lab.sh --down` stops it and deletes the database at the end instead. `AIRFLOW_PORT=8081 ./run_lab.sh` uses another port.

## Run it yourself in the Airflow UI

With Airflow running (from `./setup.sh`):

1. **Unpause `customer_clustering`**, then **unpause `customer_data_ingest`**, using the toggles on the DAGs page. The ingest DAG is scheduled `@daily` with `catchup=False`, so unpausing it starts the latest daily run straight away.
2. Watch **customer_data_ingest**. Its Graph view shows `load_data → validate_data → quality_gate`, then `publish_dataset` running and `reject_data` skipped.
3. A run of **customer_clustering** appears by itself, with run type *asset_triggered*. Its graph shows `fit_kmeans [49]`; click it to see the 49 mapped task instances, 8 running at a time.
4. **Browse → Assets** shows `customers` (produced by ingest, consumed by clustering) and `customer_segments`, with an event per publish and its metadata.
5. **Reject bad data:** on customer_data_ingest choose **Trigger** and set `input_file` to `bad_sample.csv` in the form.
   - The run fails at `reject_data`, `publish_dataset` is skipped, and no clustering run follows.
   - `working_data/<run id>/quality_report.json` says which checks failed.
6. **Change the search:** trigger customer_clustering with `k_max` = `30`. It creates 30 mapped tasks instead of 49 and re-clusters the newest published version.

The same from the command line:

```bash
docker compose run --rm airflow-cli airflow dags unpause customer_clustering
docker compose run --rm airflow-cli airflow dags unpause customer_data_ingest
docker compose run --rm airflow-cli airflow dags trigger customer_data_ingest --conf '{"input_file": "bad_sample.csv"}'
docker compose run --rm airflow-cli airflow dags list-runs customer_clustering
```

Run the tests on their own:

```bash
docker compose run --rm --no-deps airflow-cli bash -c "cd /opt/airflow && python -m pytest tests -v"
```

Debug one DAG in a single process, with no scheduler: `docker compose exec airflow-scheduler python /opt/airflow/dags/kmeans_pipeline.py`. That calls `dag.test()`.

**Stopping:** `docker compose down` keeps the metadata database, and `docker compose up -d --wait` brings it back. `docker compose down -v` deletes it. Files in `working_data/` stay either way; delete the folder to clear them.

## How the pipeline works

```text
customer_data_ingest  (@daily, or triggered)
  load_data ──> validate_data ──> quality_gate ─┬─> publish_dataset ──> [Asset: customers]
                                                └─> reject_data   (fails the run; nothing is published)

[Asset: customers] ──> customer_clustering  (no clock schedule: runs on each published version)
  k_values ──> resolve_dataset ──> data_preprocessing ──> fit_kmeans ×N ──> choose_k ──> train_final_model ──> predict ──> [Asset: customer_segments]
                                                           (mapped, 8 at once)  (elbow)    (scaler + KMeans)
```

**Data-quality gate** (`validate_data`, then `@task.branch quality_gate`). All five checks must pass for the data to be published:

| Check | Rule | The real data |
|---|---|---|
| `required_columns` | `CUST_ID`, `BALANCE`, `PURCHASES`, `CREDIT_LIMIT` present | present |
| `row_count` | at least `min_rows` rows (default 1,000) | 8,950 |
| `missing_values` | at most `max_missing_fraction` of rows have a blank (default 5%) | 3.5% |
| `non_negative` | no negative balance, purchases or credit limit | none |
| `unique_ids` | no repeated `CUST_ID` | none |

`dags/data/bad_sample.csv` is 1,200 rows of the real data, deliberately broken. It fails three checks: 19% of rows have a blank, 5 purchases are negative, and 3 IDs are duplicated.

**Assets.** `publish_dataset` writes an immutable copy to `working_data/datasets/customers/<run id>.parquet` and records an asset event carrying its path, row count, content hash and source file. `customer_clustering` is scheduled on that Asset (`schedule=[customers]`). Its `resolve_dataset` task reads the triggering event, so a run always clusters the version that started it, even if a newer one is published meanwhile. A manual run uses the newest version. Rejected data emits no event, so it can never reach training.

**Dynamic task mapping.** `fit_kmeans.partial(scaled_path=…).expand(k=k_values)` creates one task instance per k at run time, from the `k_min`…`k_max` Params. They run 8 at a time (`max_active_tis_per_dagrun`), each limited to one CPU thread so they don't compete for cores. `choose_k` receives all the SSE values in order and finds the elbow with `kneed`.

**The model.** `train_final_model` fits KMeans with the chosen k and saves it **behind its fitted scaler** as one scikit-learn pipeline (`MinMaxScaler → KMeans`), so it takes raw feature values. `predict` writes every customer's cluster to `clusters.csv`, predicts the row in `test.csv`, and warns when an input is outside the range seen in training.

**Params** are validated when a run is triggered, so a wrong value is rejected with an error before anything runs:

| DAG | Param | Default | Rule |
|---|---|---|---|
| customer_data_ingest | `input_file` | `file.csv` | a `.csv` name in `dags/data/` (no paths) |
| | `min_rows` | 1000 | ≥ 1 |
| | `max_missing_fraction` | 0.05 | 0 to 1 |
| customer_clustering | `k_min` | 1 | ≥ 1 |
| | `k_max` | 49 | 2 to 100, and at least 4 above `k_min` |

**Failure handling** (`dags/src/runs.py`):
- Every task retries twice, 10 s apart, and has a 10-minute timeout (2 minutes per `fit_kmeans`).
- Failed quality checks and invalid k ranges raise `AirflowFailException`, which is never retried.
- `record_failure`, the `on_failure_callback`, writes the task, try number and error to `working_data/<run id>/failures/<task>.json`.

**XCom carries only small values:** file paths, the SSE list and the results. The data stays in files. Each run gets its own folder:

```text
working_data/
├── datasets/customers/<ingest run id>.parquet   published versions (never overwritten)
└── <run id>/
    ├── raw.parquet, quality_report.json        ingest
    ├── scaled.npy, scaler.pkl, sse.json         clustering: inputs and the SSE curve with its elbow
    ├── model.pkl, clusters.csv                  the trained pipeline and every customer's cluster
    └── failures/<task>.json                     only if a task failed
```

## What a successful run looks like

The end of `./run_lab.sh`:

```text
==> Unit and DAG tests (pytest, inside the Airflow image)
..................                                                       [100%]
18 passed in 9.66s

==> End-to-end run through the scheduler
1. Unpausing both DAGs (the consumer first, so it is listening when data is published)
2. The daily ingest run starts by itself and publishes validated data
  ok   ingest run scheduled__2026-11-12T00:00:00+00:00 (scheduled) succeeded
  ok   the quality gate skipped reject_data
       published 8950 rows, sha256 8c1b84552e71: /opt/airflow/working_data/datasets/customers/scheduled__....parquet
       quality checks: row_count passed, unique_ids passed, non_negative passed, missing_values passed, required_columns passed
3. The publish starts customer_clustering on that exact version
  ok   clustering run asset_triggered__... was started by the asset event
  ok   clustering run succeeded
  ok   it clustered the version that ingest published
  ok   49 mapped fit_kmeans tasks succeeded
  ok   the final model uses the elbow's k (8)
       test.csv row -> cluster 6, outside the training range: ['CREDIT_LIMIT']
       cluster sizes: {'0': 1204, '1': 1251, '2': 255, '3': 770, '4': 1737, '5': 3151, '6': 23, '7': 245}
4. Bad data is rejected and starts nothing
  ok   the bad_sample.csv ingest was rejected
  ok   nothing was published
       failed checks: ['unique_ids', 'non_negative', 'missing_values']
       watching 45s for a clustering run ...
  ok   no clustering run was started

End-to-end check passed.
```

Run IDs and dates will differ. The checks, k, the cluster sizes and the data hash should match, because KMeans is seeded (`random_state=42`). If any check fails, the script prints `FAIL` next to it and exits with an error.

## Results and findings

- **The elbow is at k = 8** for k = 1…49. That gives 8 segments, from 3,151 customers down to a 23-customer group with very high credit limits.
- **The original lab trained the right curve but saved the wrong model.** It fitted k = 1…49 to draw the elbow, then saved the k = 49 model (the last one fitted). This version trains the elbow's k.
- **The original predicted on unscaled data.** The model was trained on min-max scaled features, but `test.csv` was passed in raw. For ordinary customers, predicting on raw values agrees with the correct cluster only **1.9%** of the time; it puts everyone in two clusters. Saving the scaler inside the model fixes this.
- **The lab's only test input is out of range.** `test.csv` has a credit limit of 241,514, 8× the largest in the data (30,000), so its cluster says little. `predict` now flags this (`test_out_of_range: ['CREDIT_LIMIT']`).
- **The elbow depends on the range you search.** k = 1…30 gives an elbow at **5**, and 1…49 gives 8. It's a heuristic, not a property of the data. That's why the range is a Param and is saved with the curve in `sse.json`.
- **XCom used to hold the data.** The original pushed the DataFrame through XCom as base64-encoded pickles, 1.79 MB and 0.28 MB per run, stored in the metadata database. Now XCom holds file paths and small results.
- **Speed.** A clustering run takes about 23 s on its own (the 49 fits run in parallel). The original's serial loop of 49 fits took 34–40 s.

## What changed from the original lab

| Area | Original lab | This version | Airflow concept |
|---|---|---|---|
| Running it | No `docker-compose.yaml`; the tutorial links the Airflow **2.5.1** compose file while the DAG uses Airflow 3 imports. Packages were to be installed at container start, and a 110 KB `airflow.cfg` dump was deleted by its own `setup.sh` | A pinned Airflow **3.3.2** `docker-compose.yaml` trimmed for a laptop (LocalExecutor + Postgres, no Redis or Celery), with packages baked into an image. `setup.sh` resets, builds, migrates and waits for health | Running Airflow on Docker |
| DAG style | `PythonOperator` with `op_args` and `.output` wiring | TaskFlow (`@dag`/`@task` from `airflow.sdk`), where calling the tasks defines the dependencies and the XCom hand-offs. The DAG file is renamed from `airflow.py`, which could shadow the `airflow` package | TaskFlow API |
| Passing data | Base64-encoded pickled DataFrames through XCom (about 2 MB per run in the database) | Files in a per-run folder; XCom carries paths and small results | What XCom is for |
| Model search | One task fitting k = 1…49 in a loop | 49 **mapped** task instances in parallel (`expand`), reduced by `choose_k` | Dynamic task mapping, fan-out/fan-in |
| Model | Saved the k = 49 model and predicted on unscaled data | Trains the elbow's k and saves it behind its scaler; flags out-of-range inputs | |
| Data quality | None | Five checks and a branch that publishes or rejects | Branching (`@task.branch`) |
| Configuration | Hard-coded file and k range | Params, validated at trigger time | Params |
| Failures | `retries: 0` | Retries, per-task timeouts, non-retryable `AirflowFailException` for failed checks, and an `on_failure_callback` | Retries, timeouts, callbacks |
| Scheduling | One manually triggered DAG | A daily ingest DAG and a clustering DAG scheduled on the `customers` **Asset**, which publishes `customer_segments` | Assets (data-aware scheduling) |
| Tests | None | 18 pytest tests (DAG structure, Params, the pipeline functions) and an end-to-end check against the scheduler | DAG testing |
| Automation | Manual UI steps | `run_lab.sh`: fresh start, tests, end-to-end check | |

## Project layout

```text
MLOps-Lab4/
├── run_lab.sh                 whole lab end to end
├── setup.sh                   build and start Airflow from a clean state
├── docker-compose.yaml        Airflow 3.3.2 + Postgres (LocalExecutor)
├── Dockerfile                 apache/airflow:3.3.2-python3.12 + requirements.txt
├── requirements.txt           kneed, pytest
├── dags/
│   ├── ingest_pipeline.py     customer_data_ingest: load, check, publish the customers Asset
│   ├── kmeans_pipeline.py     customer_clustering: mapped KMeans fits, elbow, final model, predictions
│   ├── src/
│   │   ├── lab.py             the pipeline steps as plain functions (no Airflow imports)
│   │   └── runs.py            Assets, per-run folders, retry defaults, failure callback
│   └── data/
│       ├── file.csv           8,950 customers, 18 columns (the lab's data)
│       ├── test.csv           the lab's single prediction input
│       └── bad_sample.csv     deliberately broken sample for the quality gate
├── tests/
│   ├── test_dags.py           the DAGs import and are wired as intended
│   ├── test_lab.py            the pipeline functions
│   └── end_to_end.py          run by run_lab.sh against the live scheduler
├── ORIGINAL_LAB.md            the original write-up (unchanged)
└── working_data/, logs/       created at run time (git-ignored)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Docker is installed but not running` | Start Docker Desktop and wait for it to say it's running |
| `port is already allocated` / `address already in use` on 8080 | Something else uses 8080 (a common default, e.g. an MLflow or Jupyter server). Run `AIRFLOW_PORT=8081 ./setup.sh` |
| A run stays **queued** and never starts | Its DAG is paused, and new DAGs start paused. Unpause it with the toggle or `airflow dags unpause <dag_id>` |
| No clustering run after an ingest | Check the ingest run's `publish_dataset` succeeded (a rejected file publishes nothing), and that customer_clustering is unpaused |
| Containers restart or tasks are killed (exit code 137) | Docker is out of memory. Give its VM at least 4 GB, or lower `MAX_PARALLEL_FITS` in `dags/kmeans_pipeline.py` |
| `Permission denied` writing `working_data/` or `logs/` (Linux) | Run `./setup.sh`, which writes your user ID to `.env`, and don't run it with `sudo` |
| `./setup.sh: Permission denied` | `chmod +x setup.sh run_lab.sh`, or run `bash setup.sh` |
| `$'\r': command not found` (Windows) | The scripts were checked out with Windows line endings. Clone inside WSL, or run `git config core.autocrlf input` and clone again (`.gitattributes` keeps `*.sh` as LF) |
| A DAG shows an import error | Open it in the UI or run `docker compose logs airflow-dag-processor`. The tests catch this too: `docker compose run --rm --no-deps airflow-cli bash -c "cd /opt/airflow && python -m pytest tests"` |
