import mlflow
from mlflow.tracking import MlflowClient

# Model name must contain roll number
MODEL_NAME = "EmployeeAttritionClassifier_22CS123"

# Run IDs
baseline_run_id = "460eaf105e6944daa70dde406fe37c6e"
tuned_run_id = "6368ea160e7f46fd837c2977c480dfb2"

client = MlflowClient()

# --------------------------------------------------
# Register Version 1 - Baseline Random Forest
# --------------------------------------------------

baseline_uri = f"runs:/{baseline_run_id}/model"

version1 = mlflow.register_model(
    model_uri=baseline_uri,
    name=MODEL_NAME
)

print("Version 1 registered successfully!")
print("Model:", MODEL_NAME)
print("Version:", version1.version)


# --------------------------------------------------
# Register Version 2 - Tuned Random Forest
# --------------------------------------------------

tuned_uri = f"runs:/{tuned_run_id}/model"

version2 = mlflow.register_model(
    model_uri=tuned_uri,
    name=MODEL_NAME
)

print("\nVersion 2 registered successfully!")
print("Model:", MODEL_NAME)
print("Version:", version2.version)


# --------------------------------------------------
# Set Champion Alias to Version 2
# --------------------------------------------------

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    version2.version
)

print("\nChampion alias created successfully!")
print("Champion Version:", version2.version)