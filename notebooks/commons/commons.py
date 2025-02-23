import pandas as pd
import tempfile
import mlflow
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


def log_figure(fig, artifact_path):
    """
    Log a matplotlib figure as an artifact in MLflow.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Provide the figure object to log.
    artifact_path : str, optional
        Specify the filename for the artifact (default is 'figure.png').

    Returns
    -------
    None
        Log the figure as an artifact in the active MLflow run.
    
    Notes
    -----
    URI setting is assumed to be done outside this function. Set the
    correct MLFlow URI prior to execution.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/{artifact_path}"
        fig.savefig(file_path, format="png", dpi=300)
        mlflow.log_artifact(file_path)