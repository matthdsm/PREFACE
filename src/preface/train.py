"""
Training module for PREFACE.
"""

import os
# import time
from pathlib import Path
import logging
from enum import Enum
import pandas as pd
import typer
import sklearn
import numpy.typing as npt
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=unused-import # noqa: F401  # type: ignore
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
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
# from preface.lib.ensemble import build_ensemble

# Constants
EXCLUDE_CHRS: list[str] = ["13", "18", "21", "X", "Y"]


class ModelOptions(Enum):
    NEURAL = "neural"
    XGBOOST = "xgboost"


class ImputeOptions(Enum):
    ZERO = "zero"  # assume missing values are zero
    MICE = "mice"  # impute missing values using MICE
    MEAN = "mean"  # impute missing values by calculating mean
    KNN = "knn"    # impute missing values using k-nearest neighbors


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
    model: ModelOptions = typer.Option(
        ModelOptions.NEURAL, "--model", help="Type of model to train"
    ),
) -> None:
    """
    Train and optionally tune the PREFACE model.
    """
    # start_time: float = time.time()

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
        logging.info(f"Processing sample {sample['ID']} ({i + 1}/{len(samplesheet_data)})...")  # type: ignore
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

    # Handle NaN values
    # Identify the type of missingness (MCAR, MAR, MNAR). Here we assume MAR.
    # Since the input log2 ratios indicate relative coverage to a reference,
    # we can either impute missing values with the mean ratio of that feature
    # or assume zero (no change).
    # Option 1: Impute NaN through MICE (Multiple Imputation by Chained Equations)
    if impute == ImputeOptions.MICE:
        # Check sklearn version for compatibility
        sk_version = sklearn.__version__
        if sk_version != "1.8.0":
            logging.warning(f"""PREFACE uses imputation and was developed using scikit-learn version 1.8.0.
                        Since imputation is still experimental, it may be subject to change in other versions.
                        You are using version {sk_version}. Proceed with caution.""")

        logging.info("Imputing missing values using MICE... This might take a while.")
        imputer = IterativeImputer(
            random_state=42, max_iter=10, initial_strategy="mean", verbose=2
        )
        logging.info("Fitting imputer to data...")
        training_df_array = imputer.fit_transform(ratios_per_sample)
        logging.info(f"Imputation completed. Sample of imputed data:\n{training_df_array[:10]}")

        ratios_per_sample = pd.DataFrame(
            training_df_array,
            index=ratios_per_sample.index,
            columns=ratios_per_sample.columns,
        )
    # Option 2: Assume missing values are zero (no change)
    elif impute == ImputeOptions.ZERO:
        logging.info("Assuming missing values are zero...")
        ratios_per_sample = ratios_per_sample.fillna(0.0)

    # Option 3: Impute missing values by calculating mean
    elif impute == ImputeOptions.MEAN:
        logging.info("Imputing missing values using mean strategy...")
        imputer = SimpleImputer(strategy="mean")
        ratios_per_sample = pd.DataFrame(
            imputer.fit_transform(ratios_per_sample),
            index=ratios_per_sample.index,
            columns=ratios_per_sample.columns,
        )

    # Option 4: Impute missing values using k-nearest neighbors
    elif impute == ImputeOptions.KNN:
        logging.info("Imputing missing values using k-nearest neighbors...")
        imputer = KNNImputer(n_neighbors=5)
        ratios_per_sample = pd.DataFrame(
            imputer.fit_transform(ratios_per_sample),
            index=ratios_per_sample.index,
            columns=ratios_per_sample.columns,
        )

    # Split into features and labels
    x_all: npt.NDArray = ratios_per_sample.drop(columns=["sex", "ff"]).to_numpy()
    y_all: npt.NDArray = ratios_per_sample[["sex", "ff"]].to_numpy()
    # labels for regression (fetal fraction)
    y_ff_all = y_all[:, 1]
    # labels for classification (sex)
    y_sex_all = y_all[:, 0]

    # Global PCA fit on all data for later use in ensemble model
    logging.info("Fitting global PCA...")
    global_pca = PCA(n_components=n_feat)
    global_pca.fit(x_all)

    params = {}
    if tune:
        # Enable hyperparameter tuning
        logging.info("Tuning hyperparameters...")
        if model == ModelOptions.NEURAL:
            params = neural_tune(x_all, y_all, n_feat, out_dir)
        elif model == ModelOptions.XGBOOST:
            params = xgboost_tune(x_all, y_all, n_feat, out_dir)

    # Set up training (k-fold cross-validation)
    # Create directory to store fold metrics
    os.makedirs(out_dir / "training_folds", exist_ok=True)
    fold_metrics = []
    fold_models: list[keras.Model] = []

    # Set up k-fold cross-validation
    kf: KFold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_all), 1):
        logging.info(f"Processing Fold {fold}/{n_folds}...")

        # split into train and test sets
        # reduce dimensionality with PCA for each fold to prevent data leakage
        training_pca = PCA(n_components=n_feat)
        x_train_pca = training_pca.fit_transform(x_all[train_idx])
        x_test_pca = training_pca.transform(x_all[test_idx])
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        y_train_reg, y_test_reg = y_ff_all[train_idx], y_ff_all[test_idx]
        y_train_class, y_test_class = y_sex_all[train_idx], y_sex_all[test_idx]

        # Train
        logging.info(f"Training fold {fold}...")
        if model == ModelOptions.NEURAL:
            model, predictions = neural_fit(
                x_train_pca,
                x_test_pca,
                y_train_reg,
                y_train_class,
                y_test_reg,
                y_test_class,
                params,
            )
            # Save fold model
            model.save(out_dir / "training_folds" / f"fold_{fold}.keras")  # type: ignore

        elif model == ModelOptions.XGBOOST:
            model, predictions = xgboost_fit(
                x_train_pca, x_test_pca, y_train, y_test, params
            )
            model.save_model(out_dir / "training_folds" / f"fold_{fold}.bin")  # type: ignore

        fold_models.append(model)

        # Plot regression performance
        reg_perf = plot_regression_performance(
            predictions["regression_predictions"],
            y_test_reg,
            training_pca.explained_variance_ratio_,
            n_feat,
            "PREFACE (%)",
            "FF (%)",
            out_dir / "training_folds" / f"fold_{fold}_regression.png",
        )

        # Plot classification performance
        plot_classification_performance()

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

    # # Build ensemble model from fold models
    # logging.info("Building ensemble model from fold models...")
    # ensemble_model = build_ensemble(global_pca, fold_models, x_all.shape[1], out_dir / "PREFACE.onnx")

    # # Final evaluation on all training data
    # logging.info("Evaluating final model on all training data...")
    # predictions = ensemble_model.run()
    # info_overall = plot_regression_performance(
    #     predictions[0][0][0].flatten(),
    #     y_ff_all,
    #     global_pca.explained_variance_ratio_,
    #     n_feat,
    #     "PREFACE (%)",
    #     "FF (%)",
    #     out_dir / "overall_performance.png",
    # )

    # with open(out_dir / "training_statistics.txt", "w", encoding="utf-8") as f:
    #     f.write(
    #         f"""PREFACE - PREdict FetAl ComponEnt
    #         Training time: {time.time() - start_time:.0f} seconds
    #         Overall correlation (r): {info_overall["correlation"]:.4f}
    #         Overall mean absolute error (MAE): {info_overall["mae"]:.4f} ± {info_overall["sd_diff"]:.4f}
    #         """
    #     )

    logging.info(
        f"Finished! Consult '{out_dir / 'training_statistics.txt'}' "
        "to analyse your model's performance."
    )
