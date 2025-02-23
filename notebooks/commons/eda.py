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


def plot_correlation_matrix(data_df, save_path=None):
    """
    Plot the correlation matrix for all features in the DataFrame, including 
    the target variable, with negative correlations in red.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing features and the target variable.
    save_path : str, optional
        File path to save the plot as a PNG. If None, the plot is only displayed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated correlation heatmap.
    """
    sns.set(style="whitegrid")

    # Compute the correlation matrix
    corr_matrix = data_df.corr()

    # Adjust figure size dynamically based on number of features
    fig_size = max(10, len(corr_matrix) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Define colormap
    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    # Function to format text color (red for negative, black for positive)
    def color_negative_values(val):
        return f'color: {"red" if val < 0 else "black"}'

    # Create the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5,
                ax=ax, cbar=True, square=True,
                annot_kws={"size": 8, "color": "black"})  # Default text color

    # Apply color formatting for negative values
    for text in ax.texts:
        val = float(text.get_text())
        if val < 0 and val > -0.5:
            text.set_color("red")

    plt.title("Feature Correlation Matrix", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)
    return fig