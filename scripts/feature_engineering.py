import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Convert target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Feature 1: Years at company relative to age
df["YearsPerCompanyAge"] = (
    df["Years_at_Company"] / (df["Age"] + 1)
)

# Feature 2: Combined satisfaction
df["JobSatisfactionScore"] = (
    df["Job_Satisfaction"] +
    df["Work_Environment_Satisfaction"]
)

# Feature 3: Promotion ratio
df["PromotionRatio"] = (
    df["Years_Since_Last_Promotion"] /
    (df["Years_at_Company"] + 1)
)

# Feature 4: Workload score
df["WorkloadScore"] = (
    df["Project_Count"] *
    df["Average_Hours_Worked_Per_Week"]
)

# Feature 5: ORIGINAL FEATURE
# Measures possible career stagnation:
# long tenure + long time since promotion,
# adjusted by job level.
df["CareerStagnationScore"] = (
    df["Years_at_Company"] *
    (df["Years_Since_Last_Promotion"] + 1)
    / (df["Job_Level"] + 1)
)

# Create Feast timestamp
df["feature_timestamp"] = pd.Timestamp.now()

# Create output directory
os.makedirs("feature_repo/data", exist_ok=True)

# Save engineered dataset
df.to_parquet(
    "feature_repo/data/employee_features.parquet",
    index=False
)

print("Feature engineering completed!")
print("Dataset shape:", df.shape)

print("\nNew features:")
print([
    "YearsPerCompanyAge",
    "JobSatisfactionScore",
    "PromotionRatio",
    "WorkloadScore",
    "CareerStagnationScore"
])

print("\nSample engineered features:")
print(
    df[
        [
            "Employee_ID",
            "YearsPerCompanyAge",
            "JobSatisfactionScore",
            "PromotionRatio",
            "WorkloadScore",
            "CareerStagnationScore",
            "Attrition"
        ]
    ].head()
)