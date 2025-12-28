from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
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
    ratios_df["region"] = (ratios_df["chr"] + ":" + ratios_df["start"].astype(str) + "-" + ratios_df["end"].astype(str))  # type: ignore
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


def plot_classification_performance(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    path: Path,
) -> None:
    """
    Plot classification performance (ROC and Confusion Matrix).
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=COLOR_A, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color=COLOR_B, lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    # Confusion Matrix
    ax = axes[1]
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Female", "Male"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


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
    out_dir_path: Path,
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
    plt.savefig(out_dir_path / "FFX.png", dpi=300)
    plt.close()
