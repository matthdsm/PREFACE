from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import onnx
import pandas as pd
import statsmodels.api as sm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_absolute_error,
)


COLOR_A: str = "#8DD1C6"
COLOR_B: str = "#E3C88A"
COLOR_C: str = "#C87878"


def preprocess_ratios(ratios_df: pd.DataFrame, exclude_chrs: list[str]) -> pd.DataFrame:
    """Preprocess ratios DataFrame by excluding chromosomes, adding region column and transposing.
    returns a x by 1 dataframe with regions as columns.
    """
    # sanitize columns
    ratios_df = ratios_df[["chr", "start", "end", "ratio"]].copy()
    # santize chr column
    ratios_df["chr"] = ratios_df["chr"].astype(str).str.replace("chr", "", regex=False)
    # exclude chromosomes
    ratios_df = ratios_df[~ratios_df["chr"].isin(exclude_chrs)].copy()
    # add region column
    ratios_df["region"] = (
        ratios_df["chr"]
        + ":"  # type: ignore
        + ratios_df["start"].astype(str)
        + "-"
        + ratios_df["end"].astype(str)
    )  # type: ignore
    # drop chr, start, end columns
    ratios_df.drop(columns=["chr", "start", "end"], inplace=True)
    # set region as index and transpose
    ratios_df = ratios_df.set_index("region").T

    return ratios_df


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
    Plot performance metrics and return statistics.
    """
    # Ensure 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    diff = y_pred - y_true
    sd_diff = float(np.std(diff, ddof=1))

    # Linear Regression and Correlation
    if len(np.unique(y_pred)) > 1:
        # Use sklearn LinearRegression
        reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
        intercept = float(reg.intercept_)
        slope = float(reg.coef_[0])
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        intercept = float(np.mean(y_true))
        slope = 0.0
        correlation = 0.0

    # Plotting
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: PCA Importance
    ax = axes[0]
    y_vals = pca_explained_variance_ratio
    x_vals = np.arange(1, len(y_vals) + 1)

    # Filter zeros for log scale
    mask = y_vals > 0
    ax.plot(np.log(x_vals[mask]), np.log(y_vals[mask]), color=COLOR_A, linewidth=2)
    ax.set_xlabel("Principal components (log scale)")
    ax.set_ylabel("Proportion of variance (log scale)")
    ax.set_title("PCA")

    # Vertical line at n_feat
    log_n_feat = np.log(n_feat)
    ylim = ax.get_ylim()
    ax.vlines(
        log_n_feat,
        ylim[0],
        ylim[1] * 0.99,
        colors=COLOR_C,
        linestyles="dotted",
        linewidth=3,
    )
    ax.text(
        log_n_feat,
        ylim[1],
        "Number of features",
        color=COLOR_C,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    # Plot 2: Scatter Plot
    ax = axes[1]
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.scatter(y_pred, y_true, s=10, c="black", alpha=0.6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_title("Scatter plot")

    # Fit line
    if slope != 0:
        fit_line = intercept + slope * np.array([0, mx])
        ax.plot(
            [0, mx],
            fit_line,
            color=COLOR_A,
            linestyle="--",
            linewidth=2,
            label="OLS fit",
        )
    else:
        ax.plot(
            [0, mx],
            [intercept, intercept],
            color=COLOR_A,
            linestyle="--",
            linewidth=2,
            label="Mean fit",
        )

    # Identity line
    ax.plot([0, mx], [0, mx], color=COLOR_B, linestyle=":", linewidth=3, label="f(x)=x")
    ax.legend()
    ax.text(0, mx * 1.03, f"(r = {correlation:.3g})", fontsize=9, ha="left")

    # Plot 3: Histogram of errors
    ax = axes[2]
    n_bins = max(20, len(y_true) // 10)
    counts, bins, _ = ax.hist(diff, bins=n_bins, density=True, color="black", alpha=0.5)
    ax.set_xlabel(f"{xlab} - {ylab}")
    ax.set_ylabel("Density")
    ax.set_title("Histogram")

    mx_hist = float(np.max(counts)) if len(counts) > 0 else 0.1
    ax.vlines(
        mae,
        0,
        mx_hist,
        colors=COLOR_A,
        linestyles="--",
        linewidth=3,
        label="mean error",
    )
    ax.vlines(0, 0, mx_hist, colors=COLOR_B, linestyles=":", linewidth=3, label="x=0")
    ax.legend()

    min_bin = float(min(bins)) if len(bins) > 0 else 0.0
    ax.text(
        min_bin,
        mx_hist * 1.03,
        f"(MAE = {mae:.3g} ± {sd_diff:.3g})",
        fontsize=9,
        ha="left",
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return {
        "intercept": intercept,
        "slope": slope,
        "mae": mae,
        "sd_diff": sd_diff,
        "correlation": correlation,
    }


def fit_rlm(x_values: np.ndarray, y_values: np.ndarray):
    """
    Fit robust linear model (RLM) and return intercept and slope.
    """

    x_rlm = sm.add_constant(x_values)
    rlm_model = sm.RLM(y_values, x_rlm, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    fit_params = rlm_results.params
    intercept: float = fit_params[0]
    slope: float = fit_params[1]
    return intercept, slope


def plot_ffx(
    x_values: np.ndarray,
    y_values: np.ndarray,
    intercept: float,
    slope: float,
    output: Path,
):
    """
    Plot RLM fit results.
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(x_values, y_values, s=10, c="black", alpha=0.6)
    ax.set_xlabel("FF (%)")
    ax.set_ylabel("μ(ratio X)")
    mx = max(x_values) if len(x_values) > 0 else 1
    ax.set_xlim(0, mx)
    x_range = np.array(
        [
            min(x_values) if len(x_values) > 0 else 0,
            max(x_values) if len(x_values) > 0 else 1,
        ]
    )
    y_range = intercept + slope * x_range
    ax.plot(
        x_range,
        y_range,
        color=COLOR_A,
        linestyle="--",
        linewidth=2,
        label="RLM fit",
    )
    ax.legend()
    ax = axes[1]
    y_values_corrected = (y_values - intercept) / slope if slope != 0 else y_values
    ax.scatter(x_values, y_values_corrected, s=10, c="black", alpha=0.6)
    ax.set_xlabel("FF (%)")
    ax.set_ylabel("FFX (%)")
    ax.set_xlim(0, mx)
    ax.plot(
        [x_range[0], x_range[1]],
        [x_range[0], x_range[1]],
        color=COLOR_B,
        linestyle=":",
        linewidth=3,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_pca(
    pca: PCA,
    principal_components: npt.NDArray,
    output: Path,
    labels: list | None = None,
    title: str = "PCA Plot",
) -> None:
    """
    Generate and save a PCA plot.
    """

    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)
            plt.scatter(
                principal_components[indices, 0],
                principal_components[indices, 1],
                label=str(label),
                alpha=0.7,
            )
        plt.legend()
    else:
        plt.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            color=COLOR_A,
            alpha=0.7,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_tsne(
    data: np.ndarray | pd.DataFrame,
    output: Path,
    labels: list | None = None,
    perplexity: float = 30.0,
    title: str = "t-SNE Plot",
) -> None:
    """
    Generate and save a t-SNE plot.
    """
    # t-SNE requires fewer samples than perplexity usually, handled by sklearn but good to know.
    # If samples < perplexity, sklearn warns or adjusts.
    n_samples = data.shape[0]
    eff_perplexity = min(perplexity, n_samples - 1) if n_samples > 1 else 1.0

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne_results = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)
            plt.scatter(
                tsne_results[indices, 0],
                tsne_results[indices, 1],
                label=str(label),
                alpha=0.7,
            )
        plt.legend()
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color=COLOR_A, alpha=0.7)

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def pca_export(pca: PCA, input_dim: int) -> onnx.ModelProto:
    """Export PCA model to ONNX format."""
    initial_type = [("input", FloatTensorType([None, input_dim]))]
    pca_onnx = convert_sklearn(pca, initial_types=initial_type, target_opset=18)
    return pca_onnx  # type: ignore
