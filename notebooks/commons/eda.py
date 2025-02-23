import matplotlib.pyplot as plt
import seaborn as sns
from .commons import get_features

## Functions branched out from from 
## https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html

def plot_correlation_with_scores(data_df,
                                 save_path=None):
    """
    Plot the correlation of each variable in the dataframe with the target column.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing features and the target variable.
    save_path : str, optional
        File path to save the plot as a PNG. If None, the plot is only displayed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated correlation plot.
    """
    target = get_features()['target'][0]
    correlations = data_df.corr()[target].drop(target).sort_values()
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index,
             correlations.values, color=color_mapped)
    plt.title(f"Correlation with {target.replace("_", " ")}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png",
                    dpi=600)
    plt.close(fig)
    return fig