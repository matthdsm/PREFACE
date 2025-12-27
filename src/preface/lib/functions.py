from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import (  # pylint: disable=no-name-in-module,import-error # type: ignore
    Model,
    layers,
)

COLOR_A: str = "#8DD1C6"
COLOR_B: str = "#E3C88A"
COLOR_C: str = "#C87878"


def preprocess_ratios(ratios_df: pd.DataFrame, exclude_chrs: list[str]) -> pd.DataFrame:
    """Preprocess ratios DataFrame by excluding chromosomes, adding region column and transposing.
    returns a x by 1 dataframe with regions as columns.
    """

    # santize chr column
    ratios_df["chr"] = ratios_df["chr"].astype(str).str.replace("chr", "", regex=False)
    # exclude chromosomes
    masked_ratios = ratios_df[~ratios_df["chr"].isin(exclude_chrs)].copy()
    # add region column
    masked_ratios["region"] = (
        masked_ratios["chr"]
        + ":"
        + masked_ratios["start"].astype(str)
        + "-"
        + masked_ratios["end"].astype(str)
    )
    # drop chr, start, end columns
    masked_ratios = masked_ratios.drop(columns=["chr", "start", "end"])
    # set region as index
    masked_ratios = masked_ratios.set_index("region")
    # transpose to have regions as columns
    masked_ratios = masked_ratios.T

    return masked_ratios


def build_ensemble(n_feat: int, pca: PCA, models: list[Model]) -> Model:
    """
    Build an ensemble model that averages predictions from multiple fold models.
    Each fold model is assumed to have two outputs: regression and classification.
    """

    # Add input layer
    ensemble_input = layers.Input(shape=(n_feat,), name="input")

    # Add PCA layer
    class PCALayer(layers.Layer):
        def __init__(self, pca, **kwargs):
            super(
                PCALayer,
                self,
            ).__init__(**kwargs)
            # Convert Scikit-Learn attributes to TensorFlow constants
            self.components = tf.constant(pca.components_.T, dtype=tf.float32)

        def call(self, inputs):
            # PCA: Matrix multiplication with components
            pca_data = tf.matmul(inputs, self.components)
            return pca_data

    pca_feat = PCALayer(pca, name="pca")(ensemble_input)

    # Add each fold model as a sub-network
    reg_outputs = []
    class_outputs = []

    for i, fold_model in enumerate(models):
        fold_model.name = f"fold_model_{i}"  # Ensure unique names

        # Pass the PCA features through the fold model
        reg_out, class_out = fold_model(pca_feat)
        reg_outputs.append(reg_out)
        class_outputs.append(class_out)

    # Average regression outputs
    avg_reg_output = layers.Average(name="ff_pred")(reg_outputs)
    avg_class_output = layers.Average(name="sex_pred")(class_outputs)

    # Build and return ensemble model
    return Model(
        inputs=ensemble_input,
        outputs=[avg_reg_output, avg_class_output],
        name="PREFACE_model",
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


def plot_classification_performance():
    pass


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
