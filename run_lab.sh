#!/usr/bin/env bash
# Runs the whole lab end to end on a fresh Airflow:
#   start Airflow -> unit and DAG tests -> daily ingest publishes the customers Asset
#   -> that starts customer_clustering -> a bad file is rejected and starts nothing
#
#   ./run_lab.sh                 leave Airflow running afterwards, to look at the runs in the UI
#   ./run_lab.sh --down          stop Airflow and delete its database at the end
#   AIRFLOW_PORT=8081 ./run_lab.sh
set -euo pipefail
cd "$(dirname "$0")"

DOWN=false
for arg in "$@"; do
    case "$arg" in
        --down) DOWN=true ;;
        *) echo "Unknown option: $arg (use --down)" >&2; exit 1 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }

step "Starting a fresh Airflow 3.3.2 (setup.sh)"
./setup.sh

step "Unit and DAG tests (pytest, inside the Airflow image)"
docker compose run --rm --no-deps airflow-cli bash -c "cd /opt/airflow && python -m pytest tests -q -p no:cacheprovider"

step "End-to-end run through the scheduler"
docker compose run --rm --no-deps airflow-cli python /opt/airflow/tests/end_to_end.py

if $DOWN; then
    step "Stopping Airflow"
    docker compose down -v
else
    step "Done"
    echo "Airflow is still running at http://localhost:${AIRFLOW_PORT:-8080}. Stop it with: docker compose down"
fi
