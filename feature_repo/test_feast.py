from feast import FeatureStore

store = FeatureStore(repo_path=".")

features = store.get_online_features(
    features=[
        "employee_features:Age",
        "employee_features:Job_Level",
        "employee_features:Monthly_Income",
        "employee_features:Years_at_Company",
        "employee_features:JobSatisfactionScore",
        "employee_features:PromotionRatio",
        "employee_features:WorkloadScore",
        "employee_features:CareerStagnationScore",
    ],
    entity_rows=[
        {"Employee_ID": 1}
    ],
).to_dict()

print("Feast Online Feature Retrieval:")
print(features)