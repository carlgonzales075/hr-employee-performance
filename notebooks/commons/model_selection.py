import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

## Functions branched out from from 
## https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html


def plot_residuals(model: object, y_test, y_pred,
                   save_path=None):
    """
    Plot the residuals of the model predictions against the true values.

    Parameters
    ----------
    model : object
        The trained model with a `.predict()` method.
    X_test : array-like or pd.DataFrame
        The test features used for prediction.
    y_test : array-like or pd.Series
        The true target values for the test set.
    save_path : str, optional
        File path to save the plot as a PNG. If None, the plot is only displayed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated residuals plot.
    """

    residuals = y_test - y_pred
    sns.set_style("whitegrid",
                  {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(y_test, residuals,
                color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path,
                    format="png", dpi=600)

    # Show the plot
    plt.close(fig)

    return fig

def plot_feature_importance(model, feature_names=None):
    """
    Plots feature importance for tree-based models (XGBoost, LightGBM, 
    RandomForest, GradientBoosting).

    Parameters
    ----------
    model : object
        A trained tree-based model (XGBoost, LightGBM, RandomForest, 
        or GradientBoosting).
    feature_names : list of str, optional
        List of feature names corresponding to the input features.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if isinstance(model, xgb.Booster) or isinstance(model, xgb.XGBModel):
        importance_type = "weight" if hasattr(model, "booster") and model.booster == "gblinear" else "gain"
        xgb.plot_importance(
            model, importance_type=importance_type, ax=ax,
            title=f"Feature Importance based on {importance_type}"
        )

    elif isinstance(model, lgb.Booster) or isinstance(model, lgb.LGBMModel):
        importance = model.feature_importance(importance_type="gain")
        features = feature_names if feature_names else model.feature_name_
        sns.barplot(x=importance, y=features, ax=ax)
        ax.set_title("Feature Importance based on Gain")

    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier, 
                            GradientBoostingRegressor, GradientBoostingClassifier)):
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        sns.barplot(x=importance, y=feature_names, ax=ax)
        ax.set_title("Feature Importance")

    else:
        raise ValueError("Unsupported model type for feature importance plotting.")

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.close(fig)
    
    return fig