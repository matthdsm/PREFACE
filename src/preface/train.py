"""
Training module for PREFACE.
"""

import os
import time
from pathlib import Path
import logging
from enum import Enum
import pandas as pd
import typer
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore

from preface.lib.functions import (
    plot_regression_performance,
    plot_classification_performance,
    preprocess_ratios,
)
from preface.lib.xgboost import xgboost_tune, xgboost_fit
from preface.lib.neural import neural_tune, neural_fit
from preface.lib.impute import ImputeOptions, impute_nan
from preface.lib.ensemble import build_ensemble

# Constants
EXCLUDE_CHRS: list[str] = ["13", "18", "21", "X", "Y"]


class ModelOptions(Enum):
    NEURAL = "neural"
    XGBOOST = "xgboost"


def preface_train(
    samplesheet: Path = typer.Option(
        ..., "--samplesheet", help="Path to samplesheet file"
    ),
    out_dir: Path = typer.Option(os.getcwd(), "--outdir", help="Output directory"),
    # Data handling
    impute: ImputeOptions = typer.Option(
        ImputeOptions.ZERO, "--impute", help="Impute missing values"
    ),
    exclude_chrs: list[str] = typer.Option(
        EXCLUDE_CHRS, "--exclude-chrs", help="Chromosomes to exclude from training"
    ),
    # cross validation options
    n_folds: int = typer.Option(
        5, "--nfolds", help="Number of folds for cross-validation"
    ),
    # PCA options
    n_feat: int = typer.Option(
        50, "--nfeat", help="Number of features (PCA components)"
    ),
    # Mode options
    tune: bool = typer.Option(
        False, "--tune", help="Enable automatic hyperparameter tuning"
    ),
    # Model options
    model_type: ModelOptions = typer.Option(
        ModelOptions.NEURAL, "--model", help="Type of model to train"
    ),
) -> None:
    """
    Train and optionally tune the PREFACE model.
    """
    start_time: float = time.time()

    # Load samplesheet
    samplesheet_data: pd.DataFrame = pd.read_csv(
        samplesheet, comment="#", sep="\t", dtype={"sex": str, "ID": str}
    )

    # Check samples
    if len(samplesheet_data) < n_feat:
        logging.error(f"Please provide at least {n_feat} labeled samples.")
        raise typer.Exit(code=1)

    # Load all sample data
    logging.info("Loading samples...")
    # instantiate lists for ratios
    ratios_list: list[pd.DataFrame] = []

    # instantiate number of bins checker
    number_of_bins: int = -1

    # parse data
    for i, sample in samplesheet_data.iterrows():
        logging.info(
            f"Processing sample {sample['ID']} ({i + 1}/{len(samplesheet_data)})..."  # type: ignore
        )
        if (
            not Path(sample["filepath"]).exists()
            or not Path(sample["filepath"]).is_file()  # noqa: W503
        ):
            logging.error(f"File '{sample['filepath']}' does not exist.")
            raise typer.Exit(code=1)
        # load ratios (bed format)
        ratios = pd.read_csv(
            sample["filepath"],
            dtype={"chr": str, "start": int, "end": int, "ratio": float},
            sep="\t",
            header=0,
        )

        # check number of bins consistency
        number_of_bins_current = len(ratios)
        if number_of_bins == -1:
            number_of_bins = number_of_bins_current
        elif number_of_bins != number_of_bins_current:
            logging.error("Input BED files have different numbers of bins.")
            raise typer.Exit(code=1)

        # preprocess ratios
        masked_ratios = preprocess_ratios(ratios, exclude_chrs)

        # add sample metadata columns to transposed ratios
        masked_ratios["id"] = sample["ID"]
        masked_ratios["sex"] = 1 if sample["sex"] == "M" else 0
        masked_ratios["ff"] = sample["FF"]

        # add to list
        ratios_list.append(masked_ratios)

    # Stack dataframes horizontally
    logging.info("Merging sample data...")
    ratios_per_sample: pd.DataFrame = pd.concat(ratios_list, axis=0)

    # set index to ID column
    ratios_per_sample = ratios_per_sample.set_index("id")

    logging.info("Creating training frame...")

    # Split into features and labels
    x_all: npt.NDArray = ratios_per_sample.drop(columns=["sex", "ff"]).to_numpy()
    y_all: npt.NDArray = ratios_per_sample[["sex", "ff"]].to_numpy()

    train_params = {}
    if tune:
        # Enable hyperparameter tuning
        logging.info("Tuning hyperparameters...")
        tuner = neural_tune if model_type == ModelOptions.NEURAL else xgboost_tune
        train_params = tuner(x_all, y_all, n_feat, out_dir, impute)

    # Set up training (k-fold cross-validation)
    # Create directory to store fold metrics
    os.makedirs(out_dir / "training_folds", exist_ok=True)
    fold_metrics = []
    fold_models: list[tuple[object, PCA, keras.Model]] = []

    # Set up k-fold cross-validation
    kf: KFold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_all), 1):
        logging.info(f"Processing Fold {fold}/{n_folds}...")

        # split into train and test sets
        x_train, x_test = x_all[train_idx], x_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        y_test_class: npt.NDArray = y_test[:, 0]
        y_test_reg: npt.NDArray = y_test[:, 1]

        # impute data
        x_train, imputer = impute_nan(x_train, impute)
        x_test, _ = impute_nan(x_test, impute)

        # reduce dimensionality with PCA for each fold to prevent data leakage
        fold_pca = PCA(n_components=n_feat)
        x_train = fold_pca.fit_transform(x_train)
        x_test = fold_pca.transform(x_test)

        # Train
        logging.info(f"Training fold {fold}...")
        if model_type == ModelOptions.NEURAL:
            model, predictions = neural_fit(
                x_train,
                x_test,
                y_train,
                y_test,
                train_params,
            )
            # Save fold model
            model.save(out_dir / "training_folds" / f"fold_{fold}.keras")  # type: ignore

        elif model_type == ModelOptions.XGBOOST:
            model, predictions = xgboost_fit(
                x_train, x_test, y_train, y_test, train_params
            )
            model.save_model(out_dir / "training_folds" / f"fold_{fold}.bin")  # type: ignore

        fold_models.append((imputer, fold_pca, model))

        # Plot regression performance
        reg_perf = plot_regression_performance(
            predictions["regression_predictions"],
            y_test_reg,
            fold_pca.explained_variance_ratio_,
            n_feat,
            "PREFACE (%)",
            "FF (%)",
            out_dir / "training_folds" / f"fold_{fold}_regression.png",
        )

        # Plot classification performance
        plot_classification_performance(
            predictions["class_probabilities"],
            y_test_class,
            out_dir / "training_folds" / f"fold_{fold}_classification.png",
        )

        # return metrics
        metrics: dict = {
            # fold number
            "fold": fold,
            # regression metrics
            "ff_mae": mean_absolute_error(
                y_test_reg, predictions["regression_predictions"]
            ),
            "ff_r2": r2_score(y_test_reg, predictions["regression_predictions"]),
            "ff_intercept": reg_perf["intercept"],
            "ff_slope": reg_perf["slope"],
            # classification metrics
            "sex_f1": f1_score(y_test_class, predictions["class_predictions"]),
            "sex_auc": roc_auc_score(y_test_class, predictions["class_probabilities"]),
        }
        fold_metrics.append(metrics)

    # Save fold metrics to a DataFrame
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(out_dir / "training_fold_metrics.csv", index=False)

    # Build ensemble model from fold models
    logging.info("Building ensemble model from fold models...")
    build_ensemble(
        fold_models,
        x_all.shape[1],
        out_dir / "PREFACE.onnx",
        metadata={"exclude_chrs": ",".join(exclude_chrs)},
    )

    # Final evaluation on all training data
    logging.info("Evaluating final model on all training data...")

    # Load ONNX model
    sess = ort.InferenceSession(out_dir / "PREFACE.onnx")
    input_name = sess.get_inputs()[0].name

    # Handle NaNs for evaluation if ZERO strategy was used (since ONNX graph might expect clean input for that case)
    if impute == ImputeOptions.ZERO:
        x_all_eval = np.nan_to_num(x_all, nan=0.0)
    else:
        x_all_eval = x_all

    x_all_eval = x_all_eval.astype(np.float32)

    predictions = sess.run(None, {input_name: x_all_eval})
    y_ff_pred = predictions[0].flatten()  # type: ignore

    y_ff_all = y_all[:, 1]

    # Use first fold's PCA for visualization
    first_pca = fold_models[0][1]

    info_overall = plot_regression_performance(
        y_ff_pred,
        y_ff_all,
        first_pca.explained_variance_ratio_,
        n_feat,
        "PREFACE (%)",
        "FF (%)",
        out_dir / "overall_performance.png",
    )

    with open(out_dir / "training_statistics.txt", "w", encoding="utf-8") as f:
        f.write(
            f"""PREFACE - PREdict FetAl ComponEnt
            Training time: {time.time() - start_time:.0f} seconds
            Overall correlation (r): {info_overall["correlation"]:.4f}
            Overall mean absolute error (MAE): {info_overall["mae"]:.4f} ± {info_overall["sd_diff"]:.4f}
            """
        )

    logging.info(
        f"Finished! Consult '{out_dir / 'training_statistics.txt'}' "
        "to analyse your model's performance."
    )
