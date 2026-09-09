import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load data
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
# 2. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Feature types
# --------------------------------------------------

categorical_features = [
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Overtime"
]


# --------------------------------------------------
# 4. MLflow experiment
# --------------------------------------------------

mlflow.set_experiment(
    "Employee_Attrition_22CS123"
)


# --------------------------------------------------
# 5. Run experiment with and without
#    CareerStagnationScore
# --------------------------------------------------

for use_original_feature in [True, False]:

    if use_original_feature:

        X_train_exp = X_train.copy()
        X_test_exp = X_test.copy()

        run_name = "With CareerStagnationScore"

    else:

        X_train_exp = X_train.drop(
            columns=["CareerStagnationScore"]
        )

        X_test_exp = X_test.drop(
            columns=["CareerStagnationScore"]
        )

        run_name = "Without CareerStagnationScore"


    # Find categorical and numeric features
    categorical = [
        col for col in categorical_features
        if col in X_train_exp.columns
    ]

    numeric = [
        col for col in X_train_exp.columns
        if col not in categorical
    ]


    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])


    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42
    )


    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])


    # --------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------

    with mlflow.start_run(run_name=run_name):

        pipeline.fit(
            X_train_exp,
            y_train
        )

        predictions = pipeline.predict(
            X_test_exp
        )

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


        # --------------------------------------------------
        # MLflow logging
        # --------------------------------------------------

        mlflow.log_param(
            "model",
            "Random Forest"
        )

        mlflow.log_param(
            "original_feature_used",
            use_original_feature
        )

        mlflow.log_param(
            "original_feature",
            "CareerStagnationScore"
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_param(
            "max_depth",
            5
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
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


        # --------------------------------------------------
        # Log model
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"]
        )


        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print("\n" + "=" * 55)
        print(run_name)
        print("=" * 55)

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))


print("\nAblation experiment completed!")