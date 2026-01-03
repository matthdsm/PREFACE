"""Plotting utilities for PREFACE."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error

# Define the public API for this module
__all__ = [
    "plot_regression_performance",
    "plot_ffx",
    "plot_pca",
    "plot_tsne",
    "plot_cv_splits",
]


# Consistent color palette for plots
COLOR_A: str = "#8DD1C6"
COLOR_B: str = "#E3C88A"
COLOR_C: str = "#C87878"


def _calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Calculate regression metrics (MAE, R², slope, intercept)."""
    mae = mean_absolute_error(y_true, y_pred)
    sd_diff = float(np.std(y_pred - y_true, ddof=1))

    if len(np.unique(y_pred)) > 1:
        # Reshape y_pred to be a 2D array for LinearRegression
        reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
        intercept = float(reg.intercept_)
        slope = float(reg.coef_[0])
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        intercept = float(np.mean(y_true))
        slope = 0.0
        correlation = 0.0

    return {
        "intercept": intercept,
        "slope": slope,
        "mae": mae,
        "sd_diff": sd_diff,
        "correlation": correlation,
    }


def _plot_pca_importance(
    ax: plt.Axes,  # type: ignore
    pca_explained_variance_ratio: np.ndarray,
    n_feat: int,
) -> None:
    """Plot the PCA explained variance."""
    y_vals = pca_explained_variance_ratio
    x_vals = np.arange(1, len(y_vals) + 1)

    # Use a mask to plot only non-zero values on a log scale
    mask = y_vals > 0
    ax.plot(np.log(x_vals[mask]), np.log(y_vals[mask]), color=COLOR_A, linewidth=2)
    ax.set_xlabel("Principal components (log scale)")
    ax.set_ylabel("Proportion of variance (log scale)")
    ax.set_title("PCA")

    # Add a vertical line to indicate the number of features used
    log_n_feat = np.log(n_feat)
    ylim = ax.get_ylim()
    ax.vlines(
        log_n_feat,
        ylim[0],
        ylim[1],
        colors=COLOR_C,
        linestyles="dotted",
        linewidth=3,
    )
    ax.text(log_n_feat, ylim[1], "n_feat", color=COLOR_C, ha="center", va="bottom")


def _plot_scatter(
    ax: plt.Axes,  # type: ignore
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlab: str,
    ylab: str,
    metrics: dict,
) -> None:
    """Plot the regression scatter plot with f(x)=x and OLS fit lines."""
    mx = max(np.max(y_true), np.max(y_pred))
    ax.scatter(y_pred, y_true, s=10, c="black", alpha=0.6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_title("Scatter Plot")

    # Plot Ordinary Least Squares (OLS) fit line
    fit_line = metrics["intercept"] + metrics["slope"] * np.array([0, mx])
    ax.plot(
        [0, mx], fit_line, color=COLOR_A, linestyle="--", linewidth=2, label="OLS fit"
    )

    # Plot identity line for reference
    ax.plot([0, mx], [0, mx], color=COLOR_B, linestyle=":", linewidth=3, label="f(x)=x")
    ax.legend()
    ax.text(0, mx * 1.03, f"(r = {metrics['correlation']:.3g})", fontsize=9, ha="left")


def _plot_error_histogram(
    ax: plt.Axes,  # type: ignore
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlab: str,
    ylab: str,
    metrics: dict,
) -> None:
    """Plot the histogram of prediction errors."""
    diff = y_pred - y_true
    n_bins = max(20, len(y_true) // 10)
    counts, bins, _ = ax.hist(diff, bins=n_bins, density=True, color="black", alpha=0.5)
    ax.set_xlabel(f"{xlab} - {ylab}")
    ax.set_ylabel("Density")
    ax.set_title("Error Histogram")

    mx_hist = float(np.max(counts)) if len(counts) > 0 else 0.1
    ax.vlines(
        metrics["mae"],
        0,
        mx_hist,
        colors=COLOR_A,
        linestyles="--",
        linewidth=3,
        label="Mean Error",
    )
    ax.vlines(0, 0, mx_hist, colors=COLOR_B, linestyles=":", linewidth=3, label="x=0")
    ax.legend()

    min_bin = float(min(bins)) if len(bins) > 0 else 0.0
    ax.text(
        min_bin,
        mx_hist * 1.03,
        f"(MAE = {metrics['mae']:.3g} ± {metrics['sd_diff']:.3g})",
        fontsize=9,
        ha="left",
    )


def plot_regression_performance(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pca_explained_variance_ratio: np.ndarray,
    n_feat: int,
    xlab: str,
    ylab: str,
    path: Path,
) -> dict[str, float]:
    """
    Plot a comprehensive regression performance dashboard and return statistics.

    This function generates a 3-panel plot:
    1. PCA explained variance.
    2. A scatter plot of predicted vs. true values.
    3. A histogram of the prediction errors.
    """
    y_true_1d = np.ravel(y_true)
    y_pred_1d = np.ravel(y_pred)

    metrics = _calculate_regression_metrics(y_true_1d, y_pred_1d)

    # Create a 3-panel plot
    # Arguments: nrows, ncols, figsize in inches
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    _plot_pca_importance(axes[0], pca_explained_variance_ratio, n_feat)
    _plot_scatter(axes[1], y_true_1d, y_pred_1d, xlab, ylab, metrics)
    _plot_error_histogram(axes[2], y_true_1d, y_pred_1d, xlab, ylab, metrics)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return metrics


def plot_ffx(
    x_values: np.ndarray,
    y_values: np.ndarray,
    intercept: float,
    slope: float,
    output: Path,
) -> None:
    """
    Plot FFX (Fetal Fraction from X) vs FF, before and after RLM correction.
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Before correction
    ax1 = axes[0]
    ax1.scatter(x_values, y_values, s=10, c="black", alpha=0.6)
    ax1.set_xlabel("FF (%)")
    ax1.set_ylabel("μ(ratio X)")
    mx = np.max(x_values) if x_values.size > 0 else 1
    ax1.set_xlim(0, mx)

    x_range = np.array([0, mx])
    y_range = intercept + slope * x_range
    ax1.plot(
        x_range, y_range, color=COLOR_A, linestyle="--", linewidth=2, label="RLM fit"
    )
    ax1.legend()

    # Plot 2: After correction
    ax2 = axes[1]
    y_values_corrected = (y_values - intercept) / slope if slope != 0 else y_values
    ax2.scatter(x_values, y_values_corrected, s=10, c="black", alpha=0.6)
    ax2.set_xlabel("FF (%)")
    ax2.set_ylabel("FFX (%)")
    ax2.set_xlim(0, mx)
    ax2.plot(
        [0, mx], [0, mx], color=COLOR_B, linestyle=":", linewidth=3, label="f(x)=x"
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_pca(
    pca: PCA,
    principal_components: npt.NDArray,
    output: Path,
    labels: list[str] | None = None,
    title: str = "PCA Plot",
) -> None:
    """Generate and save a PCA plot, optionally coloring points by labels."""
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            color=COLOR_A,
            alpha=0.7,
        )
    else:
        # Use a dictionary for color mapping to ensure consistency if needed,
        # but for now, rely on matplotlib's default color cycle.
        unique_labels = sorted(list(set(labels)))
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(
                principal_components[indices, 0],
                principal_components[indices, 1],
                label=label,
                alpha=0.7,
            )
        plt.legend()

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_tsne(
    data: npt.NDArray | pd.DataFrame,
    output: Path,
    labels: list[str] | None = None,
    perplexity: float = 30.0,
    title: str = "t-SNE Plot",
) -> None:
    """Generate and save a t-SNE plot, optionally coloring points by labels."""
    n_samples = data.shape[0]

    # Perplexity must be less than the number of samples.
    eff_perplexity = min(perplexity, n_samples - 1.0) if n_samples > 1 else 1.0

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color=COLOR_A, alpha=0.7)
    else:
        unique_labels = sorted(list(set(labels)))
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(
                tsne_results[indices, 0],
                tsne_results[indices, 1],
                label=label,
                alpha=0.7,
            )
        plt.legend()

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_cv_splits(
    cv: object,
    X: npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
    output: Path,
    lw: int = 10,
) -> None:
    """Plot the indices of a cross-validation object."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Generate the training/testing visualizations for each CV split
    # group_shuffle_split.split returns (train, test) indices
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
        # Fill in indices with the training set groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,  # type: ignore
            vmin=-0.2,
            vmax=1.2,
        )

    # ADD Legend
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-0.2, vmax=1.2)
    legend_elements = [
        mpatches.Patch(color=cmap(norm(0)), label="Training"),
        mpatches.Patch(color=cmap(norm(1)), label="Testing"),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    # Correctly format the plot
    n_splits = cv.get_n_splits(X, y, groups)
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(len(yticklabels)) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[len(yticklabels) + 0.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title(type(cv).__name__)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
