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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore
from preface.lib.plot import (
    plot_pca,
    plot_tsne,
    plot_regression_performance,
    plot_cv_splits,
)
from preface.lib.functions import preprocess_ratios, ensemble_export
from preface.lib.xgboost import xgboost_tune, xgboost_fit
from preface.lib.svm import svm_tune, svm_fit
from preface.lib.neural import neural_tune, neural_fit
from preface.lib.impute import ImputeOptions, impute_nan


# Constants
EXCLUDE_CHRS: list[str] = ["13", "18", "21", "X", "Y"]


class ModelOptions(Enum):
    NEURAL = "neural"
    XGBOOST = "xgboost"
    SVM = "svm"


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
    n_splits: int = typer.Option(
        10, "--nsplits", help="Number of splits for cross-validation"
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
    # Check if samplesheet exists
    if not samplesheet.exists() or not samplesheet.is_file():
        logging.error(f"Samplesheet file '{samplesheet}' does not exist.")
        raise typer.Exit(code=1)
    samplesheet_dir: Path = samplesheet.parent.resolve()
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
        data_path = samplesheet_dir / Path(sample["filepath"])
        if (
            not data_path.exists() or not data_path.is_file()  # noqa: W503
        ):
            logging.error(f"File '{data_path}' does not exist.")
            raise typer.Exit(code=1)
        # load ratios (bed format)
        ratios = pd.read_csv(
            data_path,
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

    # Check missingness
    # Drop columns with more than 1% missing values
    missingness = ratios_per_sample.isnull().mean()
    cols_to_drop = missingness[missingness > 0.01].index
    if len(cols_to_drop) > 0:
        logging.warning(
            f"Dropping {len(cols_to_drop)} regions with more than 1% missing values."
        )
        ratios_per_sample = ratios_per_sample.drop(columns=cols_to_drop)

    logging.info("Creating training frame...")

    # Split into features and labels
    target_cols = ["sex", "ff"]
    x: npt.NDArray = ratios_per_sample.drop(columns=target_cols).to_numpy()
    # x_male: npt.NDArray = ratios_per_sample[ratios_per_sample["sex"] == 1].drop(columns=target_cols).to_numpy()
    # x_female: npt.NDArray = ratios_per_sample[ratios_per_sample["sex"] == 0].drop(columns=target_cols).to_numpy()
    y: npt.NDArray = ratios_per_sample[["ff"]].to_numpy()
    # y_male: npt.NDArray = ratios_per_sample[ratios_per_sample["sex"] == 1][["ff"]].to_numpy()
    # y_female: npt.NDArray = ratios_per_sample[ratios_per_sample["sex"] == 0][["ff"]].to_numpy()

    # Generate bins for the target
    # prevent data leakage and keep the distribution
    groups = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))

    train_params = {}
    if tune:
        # Enable hyperparameter tuning
        logging.info("Tuning hyperparameters...")
        if model_type == ModelOptions.NEURAL:
            logging.info("Tuning neural network hyperparameters...")
            tuner = neural_tune
        elif model_type == ModelOptions.XGBOOST:
            logging.info("Tuning XGBoost hyperparameters...")
            tuner = xgboost_tune
        elif model_type == ModelOptions.SVM:
            logging.info("Tuning SVM hyperparameters...")
            tuner = svm_tune
        else:
            logging.error("Invalid model type specified for tuning.")
            raise typer.Exit(code=1)

        train_params = tuner(
            x=x,
            y=y,
            groups=groups,
            n_components=n_feat,
            outdir=out_dir,
            impute_option=impute,
        )

    # Set up training (cross-validation)
    # Create directory to store split metrics
    os.makedirs(out_dir / "training_splits", exist_ok=True)
    split_metrics = []
    split_models: list[tuple[object, PCA, keras.Model]] = []

    # Set up cross-validation
    gss: GroupShuffleSplit = GroupShuffleSplit(
        n_splits=n_splits, test_size=0.2, random_state=42
    )

    # Visualize CV splits
    plot_cv_splits(
        cv=gss,
        X=x,
        y=y,
        groups=groups,
        output=out_dir / "cv_splits.png",
    )

    for split, (train_idx, test_idx) in enumerate(gss.split(x, y, groups)):
        logging.info(f"Processing split {split}/{n_splits}...")

        # split into train and test sets
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # impute data
        x_train, imputer = impute_nan(x_train, impute)
        x_test, _ = impute_nan(x_test, impute)

        # reduce dimensionality with PCA for each split to prevent data leakage
        current_n_feat = min(n_feat, x_train.shape[0], x_train.shape[1])
        split_pca = PCA(n_components=current_n_feat)
        x_train = split_pca.fit_transform(x_train)
        x_test = split_pca.transform(x_test)

        train_labels = ratios_per_sample.index.to_numpy()[train_idx].tolist()

        plot_pca(
            split_pca,
            principal_components=x_train,
            output=out_dir / "training_splits" / f"pca_split_{split}.png",
            title=f"PCA of training split {split}",
        )

        plot_tsne(
            data=x_train,
            labels=train_labels,
            output=out_dir / "training_splits" / f"tsne_split_{split}.png",
            title=f"t-SNE of training split {split}",
        )

        # Train
        logging.info(f"Training split {split}...")
        if model_type == ModelOptions.NEURAL:
            model, predictions = neural_fit(
                x_train,
                x_test,
                y_train,
                y_test,
                train_params,
            )

        elif model_type == ModelOptions.XGBOOST:
            model, predictions = xgboost_fit(
                x_train, x_test, y_train, y_test, train_params
            )

        elif model_type == ModelOptions.SVM:
            model, predictions = svm_fit(x_train, x_test, y_train, y_test, train_params)

        else:
            logging.error("Invalid model type specified for training.")
            raise typer.Exit(code=1)
        split_models.append((imputer, split_pca, model))

        # Plot regression performance
        reg_perf = plot_regression_performance(
            predictions,
            y_test,
            split_pca.explained_variance_ratio_,
            n_feat,
            "PREFACE (%)",
            "FF (%)",
            out_dir / "training_splits" / f"split_{split}_regression.png",
        )

        # return metrics
        metrics: dict = {
            # split number
            "split": split,
            # regression metrics
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "intercept": reg_perf["intercept"],
            "slope": reg_perf["slope"],
        }
        split_metrics.append(metrics)

    # Save split metrics to a DataFrame
    split_metrics_df = pd.DataFrame(split_metrics)
    split_metrics_df.to_csv(out_dir / "training_split_metrics.csv", index=False)

    # Build ensemble model from split models
    logging.info("Building ensemble model from split models...")
    ensemble_export(
        split_models,
        x.shape[1],
        out_dir / "PREFACE.onnx",
        metadata={"exclude_chrs": ",".join(exclude_chrs)},
    )

    # Final evaluation on all training data
    logging.info("Evaluating final model on all training data...")

    # Load ONNX model
    sess = ort.InferenceSession(out_dir / "PREFACE.onnx")
    input_name = sess.get_inputs()[0].name
    predictions = sess.run(None, {input_name: x.astype(np.float32)})

    # Use first split's PCA for visualization
    first_pca = split_models[0][1]

    info_overall = plot_regression_performance(
        predictions[0],  # type: ignore
        y,
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
