# Employee Attrition Prediction using End-to-End MLOps

## Student Details

- Roll Number: 22CS123
- Project: Employee Attrition Prediction using End-to-End MLOps

## 1. Problem Statement

The objective of this project is to build an end-to-end MLOps pipeline for predicting employee attrition. The project integrates data versioning, feature engineering, feature storage, experiment tracking, hyperparameter tuning, and model registration.

## 2. Technologies Used

- Python
- Pandas
- Scikit-learn
- DVC
- Feast
- MLflow
- Git
- GitHub
- PyArrow

## 3. Dataset

The employee attrition dataset contains 1000 employee records and employee-related attributes such as:

- Age
- Gender
- Department
- Job Role
- Job Level
- Monthly Income
- Years at Company
- Job Satisfaction
- Overtime
- Project Count
- Average Hours Worked Per Week
- Absenteeism
- Distance From Home
- Attrition

The target variable is `Attrition`.

## 4. DVC Dataset Versioning

DVC was used to version the employee attrition dataset.

### Version 1

Initial dataset was added using:

```text
python -m dvc add data\employee_attrition.csv