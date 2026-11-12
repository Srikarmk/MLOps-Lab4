"""End-to-end check against the running Airflow, used by run_lab.sh (not collected by pytest).

1. Unpause both DAGs. The daily ingest DAG then starts its latest scheduled run by itself.
2. That run must pass the quality gate and publish a version of the customers Asset.
3. The publish must start customer_clustering (an asset-triggered run) on exactly that version.
4. An ingest of bad_sample.csv must be rejected, and must not start any clustering run.

Run inside the image: docker compose run --rm airflow-cli python /opt/airflow/tests/end_to_end.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = os.environ.get("AIRFLOW_API", "http://airflow-apiserver:8080/api/v2")
INGEST, CLUSTERING = "customer_data_ingest", "customer_clustering"
NO_TRIGGER_WAIT = 45  # seconds to watch for a clustering run that must not appear
problems = []


def call(method, path, body=None):
    request = urllib.request.Request(API + path, method=method, headers={"Content-Type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:300]}")


def quote(run_id):
    return urllib.parse.quote(run_id, safe="")


def run_ids(dag_id):
    return {r["dag_run_id"] for r in call("GET", f"/dags/{dag_id}/dagRuns?limit=100")["dag_runs"]}


def wait_for_new_run(dag_id, known, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = [r for r in call("GET", f"/dags/{dag_id}/dagRuns?order_by=-start_date&limit=20")["dag_runs"]
               if r["dag_run_id"] not in known]
        if new:
            return new[0]
        time.sleep(2)
    return None


def wait_until_done(dag_id, run_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = call("GET", f"/dags/{dag_id}/dagRuns/{quote(run_id)}")
        if run["state"] in ("success", "failed"):
            return run
        time.sleep(3)
    sys.exit(f"{dag_id} run {run_id} did not finish within {timeout}s")


def task_states(dag_id, run_id):
    states = {}
    for ti in call("GET", f"/dags/{dag_id}/dagRuns/{quote(run_id)}/taskInstances?limit=200")["task_instances"]:
        states.setdefault(ti["task_id"], set()).add(ti["state"])
    return states


def xcom(dag_id, run_id, task_id):
    return call("GET", f"/dags/{dag_id}/dagRuns/{quote(run_id)}/taskInstances/{task_id}/xcomEntries/return_value")["value"]


def expect(condition, message):
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        problems.append(message)


def main():
    for _ in range(60):
        try:
            urllib.request.urlopen(API + "/monitor/health", timeout=5)
            break
        except OSError:
            time.sleep(2)
    # Right after a fresh start the DAG processor may not have registered the DAGs yet
    deadline = time.time() + 180
    while {INGEST, CLUSTERING} - {d["dag_id"] for d in call("GET", "/dags")["dags"]}:
        if time.time() > deadline:
            sys.exit("The DAGs were not registered within 180s; check `docker compose logs airflow-dag-processor`")
        time.sleep(2)

    print("1. Unpausing both DAGs (the consumer first, so it is listening when data is published)")
    known_clustering = run_ids(CLUSTERING)
    for dag_id in (CLUSTERING, INGEST):
        call("PATCH", f"/dags/{dag_id}", {"is_paused": False})

    print("2. The daily ingest run starts by itself and publishes validated data")
    ingest = wait_for_new_run(INGEST, set(), timeout=120)
    if ingest is None:
        sys.exit("No scheduled ingest run appeared within 120s")
    ingest = wait_until_done(INGEST, ingest["dag_run_id"])
    states = task_states(INGEST, ingest["dag_run_id"])
    expect(ingest["state"] == "success", f"ingest run {ingest['dag_run_id']} ({ingest['run_type']}) succeeded")
    expect(states.get("reject_data") == {"skipped"}, "the quality gate skipped reject_data")
    published = xcom(INGEST, ingest["dag_run_id"], "publish_dataset")
    print(f"       published {published['rows']} rows, sha256 {published['sha256']}: {published['path']}")
    report = xcom(INGEST, ingest["dag_run_id"], "validate_data")
    print("       quality checks: " + ", ".join(f"{name} {'passed' if c['passed'] else 'FAILED'}"
                                                  for name, c in report["checks"].items()))

    print("3. The publish starts customer_clustering on that exact version")
    clustering = wait_for_new_run(CLUSTERING, known_clustering, timeout=120)
    if clustering is None:
        sys.exit("No clustering run appeared within 120s of the publish")
    clustering = wait_until_done(CLUSTERING, clustering["dag_run_id"])
    run_id = clustering["dag_run_id"]
    expect(clustering["run_type"] == "asset_triggered", f"clustering run {run_id} was started by the asset event")
    expect(clustering["state"] == "success", "clustering run succeeded")
    dataset = xcom(CLUSTERING, run_id, "resolve_dataset")
    expect(dataset["path"] == published["path"] and dataset["sha256"] == published["sha256"],
           "it clustered the version that ingest published")
    fits = len([t for t in call("GET", f"/dags/{CLUSTERING}/dagRuns/{quote(run_id)}/taskInstances?limit=200")
                ["task_instances"] if t["task_id"] == "fit_kmeans" and t["state"] == "success"])
    k = xcom(CLUSTERING, run_id, "choose_k")
    result = xcom(CLUSTERING, run_id, "predict")
    expect(fits == 49, f"{fits} mapped fit_kmeans tasks succeeded")
    expect(result["k"] == k, f"the final model uses the elbow's k ({k})")
    print(f"       test.csv row -> cluster {result['test_cluster']}, outside the training range: {result['test_out_of_range']}")
    print(f"       cluster sizes: {result['cluster_sizes']}")

    print("4. Bad data is rejected and starts nothing")
    known_clustering = run_ids(CLUSTERING)
    bad = call("POST", f"/dags/{INGEST}/dagRuns", {"logical_date": None, "conf": {"input_file": "bad_sample.csv"}})
    bad = wait_until_done(INGEST, bad["dag_run_id"])
    states = task_states(INGEST, bad["dag_run_id"])
    expect(bad["state"] == "failed" and states.get("reject_data") == {"failed"}, "the bad_sample.csv ingest was rejected")
    expect(states.get("publish_dataset") == {"skipped"}, "nothing was published")
    failed_checks = [name for name, c in xcom(INGEST, bad["dag_run_id"], "validate_data")["checks"].items() if not c["passed"]]
    print(f"       failed checks: {failed_checks}")
    print(f"       watching {NO_TRIGGER_WAIT}s for a clustering run ...")
    expect(wait_for_new_run(CLUSTERING, known_clustering, timeout=NO_TRIGGER_WAIT) is None,
           "no clustering run was started")

    if problems:
        sys.exit(f"\n{len(problems)} check(s) failed: {problems}")
    print("\nEnd-to-end check passed.")


if __name__ == "__main__":
    main()
