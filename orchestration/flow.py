
from prefect import flow, task
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


@task(name="Train Employee Attrition Models")
def train_models():
    script = PROJECT_DIR / "scripts" / "train_models.py"

    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        check=True
    )

    print("Model training completed successfully.")


@task(name="Tune Employee Attrition Model")
def tune_model():
    script = PROJECT_DIR / "scripts" / "tune_model.py"

    if not script.exists():
        raise FileNotFoundError(f"Tuning script not found: {script}")

    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_DIR),
        check=True
    )

    print("Model tuning completed successfully.")


@flow(name="Employee Attrition MLOps Pipeline")
def employee_attrition_pipeline():
    train_models()
    tune_model()


if __name__ == "__main__":
    employee_attrition_pipeline()
