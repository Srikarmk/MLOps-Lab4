# Airflow plus the lab's Python packages, installed once at build time instead of
# on every container start (_PIP_ADDITIONAL_REQUIREMENTS)
FROM apache/airflow:3.3.2-python3.12

COPY requirements.txt /requirements.txt
# Pinning apache-airflow in the same install stops pip from upgrading or downgrading Airflow itself
RUN pip install --no-cache-dir "apache-airflow==3.3.2" -r /requirements.txt
