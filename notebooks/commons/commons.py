import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_features():
    return {
        "numeric_columns": [
            "Age", "Years_At_Company", "Performance_Score",
            "Work_Hours_Per_Week", "Projects_Handled",
            "Overtime_Hours", "Sick_Days", "Remote_Work_Frequency",
            "Team_Size", "Training_Hours", "Promotions"
        ],
        "one_hot_encode_columns": [
            "Department", "Gender", "Job_Title", "Education_Level"
        ],
        "target": [
            "Employee_Satisfaction_Score"
        ]
    }

def one_hot_encode(df, columns: list):
    """One Hot Encode a column and return to main data."""
    return pd.get_dummies(df, columns=columns,
                          drop_first=True)

def standardize(df, columns: list,
                pre_scaler: StandardScaler=None,
                return_scaler: bool=False):
    """Standardize columns of the dataset."""
    if pre_scaler is None:
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df[columns])
    else:
        scaler = pre_scaler
        standardized_data = scaler.transform(df[columns])

    standardized_df = pd.DataFrame(standardized_data,
                                   columns=columns,
                                   index=df.index)

    if return_scaler:
        return standardized_df, scaler
    return standardized_df