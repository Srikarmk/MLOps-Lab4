#!/bin/bash
# Builds the image and starts Airflow from a clean state.
#
#   ./setup.sh               start Airflow on http://localhost:8080
#   AIRFLOW_PORT=8081 ./setup.sh
#
# Stop it with `docker compose down`; add -v to also delete the metadata database.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but not running. Start Docker Desktop and try again." >&2
    exit 1
fi

# Remove containers and the metadata database from a previous run
docker compose down -v --remove-orphans

mkdir -p ./logs ./plugins
# Containers run as this UID so files they write to the mounted folders belong to you
echo "AIRFLOW_UID=$(id -u)" > .env
[ -n "${AIRFLOW_PORT:-}" ] && echo "AIRFLOW_PORT=$AIRFLOW_PORT" >> .env

docker compose build
docker compose up airflow-init --abort-on-container-exit --exit-code-from airflow-init
docker compose up -d --wait

echo
echo "Airflow is running at http://localhost:${AIRFLOW_PORT:-8080} (no login needed)"
