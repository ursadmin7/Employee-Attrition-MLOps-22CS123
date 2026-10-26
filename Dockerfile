FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY feature_repo/data/employee_features.parquet feature_repo/data/employee_features.parquet

RUN mkdir -p artifacts

CMD ["python", "scripts/train_models.py"]