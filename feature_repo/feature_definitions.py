from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Int64, Float64
from feast.infra.offline_stores.file_source import FileSource


# Entity
employee = Entity(
    name="employee",
    join_keys=["Employee_ID"]
)


# Data source
employee_source = FileSource(
    name="employee_features_source",
    path="data/employee_features.parquet",
    timestamp_field="feature_timestamp"
)


# Feature View
employee_features = FeatureView(
    name="employee_features",
    entities=[employee],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="Job_Level", dtype=Int64),
        Field(name="Monthly_Income", dtype=Int64),
        Field(name="Years_at_Company", dtype=Int64),
        Field(name="Years_in_Current_Role", dtype=Int64),
        Field(name="Years_Since_Last_Promotion", dtype=Int64),
        Field(name="Work_Life_Balance", dtype=Int64),
        Field(name="Job_Satisfaction", dtype=Int64),
        Field(name="Performance_Rating", dtype=Int64),
        Field(name="Training_Hours_Last_Year", dtype=Int64),
        Field(name="Project_Count", dtype=Int64),
        Field(name="Average_Hours_Worked_Per_Week", dtype=Int64),
        Field(name="Absenteeism", dtype=Int64),
        Field(name="Work_Environment_Satisfaction", dtype=Int64),
        Field(name="Relationship_with_Manager", dtype=Int64),
        Field(name="Job_Involvement", dtype=Int64),
        Field(name="Distance_From_Home", dtype=Int64),
        Field(name="Number_of_Companies_Worked", dtype=Int64),
        Field(name="YearsPerCompanyAge", dtype=Float64),
        Field(name="JobSatisfactionScore", dtype=Int64),
        Field(name="PromotionRatio", dtype=Float64),
        Field(name="WorkloadScore", dtype=Int64),
        Field(name="CareerStagnationScore", dtype=Float64),
        Field(name="Attrition", dtype=Int64),
    ],
    source=employee_source,
)