import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

# Drop ID and timestamp
df = df.drop(
    columns=["Employee_ID", "feature_timestamp"]
)

# Target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# --------------------------------------------------
# 2. Define feature types
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
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
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
# 5. MLflow setup
# --------------------------------------------------

mlflow.set_experiment(
    "Employee_Attrition_22CS123"
)


# --------------------------------------------------
# 6. Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
}


# --------------------------------------------------
# 7. Train models
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(
        run_name=model_name
    ):

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

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

        # Log parameters
        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # Log metrics
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

        plt.figure()
        plt.imshow(cm)
        plt.title(model_name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()

        artifact_path = (
            "artifacts/"
            + model_name.replace(" ", "_")
            + "_confusion_matrix.png"
        )

        os.makedirs("artifacts", exist_ok=True)

        plt.savefig(artifact_path)
        plt.close()

        mlflow.log_artifact(
            artifact_path
        )

        # Log model
        mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=["numpy.dtype"]
)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))


# --------------------------------------------------
# 8. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

results_df.to_csv(
    "artifacts/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to artifacts/model_comparison.csv")