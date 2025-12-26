"""
Training module for PREFACE.
"""

import os
import time
from pathlib import Path

import pandas as pd
import typer
from pandera.errors import SchemaError
import sklearn
from sklearn.experimental import enable_iterative_imputer  # pylint: disable=unused-import # noqa: F401  # type: ignore
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore

from preface.lib.functions import (
    build_ensemble,
    build_multi_output_nn,
    plot_regression_performance,
    preprocess_ratios,
)
from preface.lib.schemas import SampleDataSchema, SampleSchema

# Constants
EXCLUDE_CHRS: list[str] = ['13', '18', '21', 'X', 'Y']


def preface_train(
    samplesheet: Path = typer.Option(..., "--samplesheet", help="Path to samplesheet file"),
    out_dir: Path = typer.Option(..., "--outdir", help="Output directory"),
    n_feat: int = typer.Option(50, "--nfeat", help="Number of features (PCA components)"),
    n_folds: int = typer.Option(5, "--nfolds", help="Number of folds for cross-validation"),
    n_neurons: int = typer.Option(2, "--neurons", help="Number of initial neurons in neural network"),
    exclude_chrs: list[str] = typer.Option(EXCLUDE_CHRS, "--exclude-chrs", help="Chromosomes to exclude from training"),
    impute: bool = typer.Option(False, "--impute", help="Impute missing values instead of assuming zero")
) -> None:
    """
    Train the PREFACE model.
    """
    start_time: float = time.time()

    # Load samplesheet
    samplesheet_data: pd.DataFrame = pd.read_csv(
        samplesheet, comment='#', dtype={'sex': str, 'ID': str}, index_col='ID'
    )

    # Validate samplesheet
    samplesheet_schema = SampleSchema()
    try:
        samplesheet_schema.validate(samplesheet_data)
    except SchemaError as e:
        typer.echo(f"Error validating samplesheet: {e}")
        raise typer.Exit(code=1)

    # Check samples
    if len(samplesheet_data) < n_feat:
        typer.echo(f"Please provide at least {n_feat} labeled samples.")
        raise typer.Exit(code=1)

    # Load all sample data
    typer.echo("Loading samples...")

    # instantiate lists for ratios
    ratios_list: list[pd.DataFrame] = []

    # instantiate schema
    sample_data_schema = SampleDataSchema()

    # instantiate number of bins checker
    number_of_bins: int = -1

    # parse data
    for _, sample in samplesheet_data.iterrows():
        if not Path(sample['filepath']).exists() or not Path(sample['filepath']).is_file():
            typer.echo(f"Error: File '{sample['filepath']}' does not exist.")
            raise typer.Exit(code=1)
        # load ratios (bed format)
        ratios = pd.read_csv(sample['filepath'], dtype={'chr': str, 'start': int, 'end': int, 'ratio': float}, sep='\t', header=0)
        # validate ratios
        try:
            sample_data_schema.validate(ratios)
        except SchemaError as e:
            typer.echo(f"Error validating sample data for file {sample['filepath']}: {e}")
            raise typer.Exit(code=1)

        # check number of bins consistency
        number_of_bins_current = len(ratios)
        if number_of_bins == -1:
            number_of_bins = number_of_bins_current
        elif number_of_bins != number_of_bins_current:
            typer.echo("Error: Input BED files have different numbers of bins.")
            raise typer.Exit(code=1)

        # preprocess ratios
        masked_ratios = preprocess_ratios(ratios, exclude_chrs)

        # add sample metadata columns to transposed ratios
        masked_ratios['id'] = sample['ID']
        masked_ratios['sex'] = sample['sex']
        masked_ratios['ff'] = sample['FF']

        # add to list
        ratios_list.append(masked_ratios)

    # Stack dataframes horizontally
    typer.echo("Merging sample data...")
    ratios_per_sample: pd.DataFrame = pd.concat(ratios_list, axis=0)

    # set index to ID column
    ratios_per_sample = ratios_per_sample.set_index('id')

    typer.echo("Creating training frame...")

    # Handle NaN values
    # Identify the type of missingness (MCAR, MAR, MNAR). Here we assume MAR.
    # Since the input log2 ratios indicate relative coverage to a reference,
    # we can either impute missing values with the mean ratio of that feature
    # or assume zero (no change).
    # Option 1: Impute NaN through MICE (Multiple Imputation by Chained Equations)
    if impute:
        # Check sklearn version for compatibility
        sk_version = sklearn.__version__
        if sk_version != "1.8.0":
            typer.echo(f"""Warning: PREFACE uses imputation and was developed using scikit-learn version 1.8.0.
                        Since imputation is still experimental, it may be subject to change in other versions.
                        You are using version {sk_version}. Proceed with caution.""")

        typer.echo("Imputing missing values using MICE...")

        imputer = IterativeImputer(random_state=42, max_iter=10, initial_strategy='mean')
        training_df_array = imputer.fit_transform(ratios_per_sample)
        ratios_per_sample = pd.DataFrame(
            training_df_array,
            index=ratios_per_sample.index,
            columns=ratios_per_sample.columns
        )
    # Option 2: Assume missing values are zero (no change)
    else:
        typer.echo("Assuming missing values are zero...")
        ratios_per_sample = ratios_per_sample.fillna(0.0)

    # Split into features and labels
    x_all: pd.DataFrame = ratios_per_sample.drop(columns=['sex', 'ff'])
    # labels for regression (fetal fraction)
    y_ff_all = ratios_per_sample['ff']
    # labels for classification (sex)
    y_sex_all = (
        ratios_per_sample['sex']
        .map({"M": 1, "F": 0})
        .astype(float)
        .values
    )

    # Reduce dimensionality with PCA
    global_pca = PCA(n_components=n_feat)
    x_all_pca = global_pca.fit_transform(x_all)

    # Set up training
    # Create directory to store fold metrics
    os.makedirs(out_dir / 'training_folds', exist_ok=True)
    fold_metrics = []
    fold_models: list[keras.Model] = []

    # Set up k-fold cross-validation
    kf: KFold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_all_pca), 1):
        typer.echo(f"Processing Fold {fold}/{n_folds}...")

        # split into train and test sets
        x_train, x_test = x_all_pca[train_idx], x_all_pca[test_idx]
        y_ff_train, y_ff_test = y_ff_all[train_idx], y_ff_all[test_idx]
        y_sex_train, y_sex_test = y_sex_all[train_idx], y_sex_all[test_idx]

        # Create new model instance
        model = build_multi_output_nn(input_dim=n_feat, n_neurons=n_neurons)

        # Train
        typer.echo(f"Training fold {fold}...")
        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        # Fit model
        model.fit(
            x_train,
            {"reg_output": y_ff_train, "class_output": y_sex_train},
            validation_data=(
                x_test,
                {"reg_output": y_ff_test, "class_output": y_sex_test},
            ),
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[early_stop],
        )

        # Save fold model
        model.save(out_dir / 'training_folds' / f'fold_{fold}.keras')
        fold_models.append(model)

        # Evaluate
        predictions = model.predict(x_test)
        y_ff_pred = predictions[0].flatten()
        class_pred_probs = predictions[1].flatten()
        class_pred = (class_pred_probs >= 0.5).astype(int)

        # Plot regression performance
        reg_perf = plot_regression_performance(
            y_ff_pred,
            y_ff_test.to_numpy(),
            global_pca.explained_variance_ratio_,
            n_feat,
            "PREFACE (%)",
            "FF (%)",
            out_dir / "training_folds" / f"fold_{fold}_regression.png",
        )

        # Calculate metrics
        metrics: dict = {
            # fold number
            'fold': fold,
            # regression metrics
            'ff_mae': mean_absolute_error(y_ff_test, y_ff_pred),
            'ff_r2': r2_score(y_ff_test, y_ff_pred),
            'ff_intercept': reg_perf['intercept'],
            'ff_slope': reg_perf['slope'],
            # classification metrics
            'sex_f1': f1_score(y_sex_test, class_pred), # type: ignore
            'sex_auc': roc_auc_score(y_sex_test, class_pred_probs) # type: ignore
        }
        fold_metrics.append(metrics)

    # Save fold metrics to a DataFrame
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(
        out_dir / 'training_fold_metrics.csv', index=False
    )

    # Build ensemble model from fold models
    typer.echo("Building ensemble model from fold models...")
    ensemble_model = build_ensemble(len(x_all.columns), global_pca, fold_models)
    ensemble_model.save(out_dir / 'PREFACE')

    # Final evaluation on all training data
    typer.echo("Evaluating final model on all training data...")
    predictions = ensemble_model.predict(x_all)
    info_overall = plot_regression_performance(
        predictions[0].flatten(),
        y_ff_all.to_numpy(),
        global_pca.explained_variance_ratio_,
        n_feat,
        'PREFACE (%)',
        'FF (%)',
        out_dir / 'overall_performance.png'
    )

    with open(
        out_dir / 'training_statistics.txt', 'w', encoding='utf-8'
    ) as f:
        f.write(
            f"""PREFACE - PREdict FetAl ComponEnt
            Training time: {time.time() - start_time:.0f} seconds
            Overall correlation (r): {info_overall["correlation"]:.4f}
            Overall mean absolute error (MAE): {info_overall["mae"]:.4f} ± {info_overall["sd_diff"]:.4f}
            """
        )

    typer.echo(
        f"Finished! Consult '{out_dir / 'training_statistics.txt'}' "
        "to analyse your model's performance."
    )
