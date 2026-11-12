import sys
from pathlib import Path

# The DAG files import `src`, which Airflow finds because the dags folder is on its path
DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
sys.path.insert(0, str(DAGS_DIR))
