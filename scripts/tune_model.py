import pandas as pd
import mlflow
import mlflow.sklearn
import os
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_parquet(
    "feature_repo/data/employee_features.parquet"
)

df = df.drop(
    columns=["Employee_ID", "feature_timestamp"]
)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# --------------------------------------------------
# 2. Feature types
# --------------------------------------------------

categorical_features = [
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Overtime"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Random Forest pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])


# --------------------------------------------------
# 6. Hyperparameter grid
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 8, 12],
    "model__min_samples_split": [2, 5],
    "model__class_weight": [None, "balanced"]
}


# --------------------------------------------------
# 7. Grid Search
# --------------------------------------------------

print("\nStarting Hyperparameter Tuning...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(grid_search.best_score_, 4))


# --------------------------------------------------
# 8. Test evaluation
# --------------------------------------------------

predictions = best_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

print("\nTUNED RANDOM FOREST")
print("=" * 50)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# --------------------------------------------------
# 9. MLflow
# --------------------------------------------------

mlflow.set_experiment(
    "Employee_Attrition_22CS123"
)

with mlflow.start_run(
    run_name="Random Forest Hyperparameter Tuning"
):

    # Parameters
    for key, value in grid_search.best_params_.items():
        mlflow.log_param(key, value)

    mlflow.log_param(
        "cv",
        5
    )

    mlflow.log_param(
        "scoring",
        "f1"
    )

    # Metrics
    mlflow.log_metric(
        "cv_f1_score",
        grid_search.best_score_
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    os.makedirs("artifacts", exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Tuned Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    confusion_path = (
        "artifacts/"
        "tuned_random_forest_confusion_matrix.png"
    )

    plt.savefig(confusion_path)
    plt.close()

    mlflow.log_artifact(
        confusion_path
    )

    # Save best parameters
    params_path = (
        "artifacts/"
        "best_random_forest_params.json"
    )

    with open(params_path, "w") as f:
        json.dump(
            grid_search.best_params_,
            f,
            indent=4
        )

    mlflow.log_artifact(
        params_path
    )

    # Log model
    mlflow.sklearn.log_model(
        best_model,
        name="model",
        skops_trusted_types=["numpy.dtype"]
    )


print("\nHyperparameter tuning completed!")