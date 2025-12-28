from pathlib import Path
import numpy as np
import numpy.typing as npt
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.keras import (  # type: ignore # pylint: disable=no-name-in-module,import-error
    Model,
)
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

from preface.lib.impute import impute_nan, ImputeOptions


def xgboost_tune(features: npt.NDArray, targets: npt.NDArray, n_components: int, outdir: Path, impute_option: ImputeOptions) -> dict:
    def objective(trial) -> float:
        params = {
            # number of boosting rounds
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            # maximum depth of each tree
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            # sampling ratios
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
            # random state for reproducibility
            "random_state": 42,
        }

        # Internal split for the tuner
        kf_internal = KFold(n_splits=3, shuffle=True)
        scores = []

        for t_idx, v_idx in kf_internal.split(features):
            x_train, x_val = features[t_idx], features[v_idx]
            y_train, y_val = targets[t_idx], targets[v_idx]

            # impute missing values
            x_train = impute_nan(x_train, impute_option)
            x_val = impute_nan(x_val, impute_option)

            # Reduce dimensionality with PCA
            pca = PCA(n_components=n_components)
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            # Train and evaluate model
            model = XGBRegressor(**params)
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            scores.append(mean_squared_error(y_val, preds))
        return np.mean(scores).astype(float)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(outdir / "xgboost_tuning_history.png")
    return study.best_params


def xgboost_fit(
    x_train: npt.NDArray,
    x_test: npt.NDArray,
    y_train: npt.NDArray,
    y_test: npt.NDArray,
    params: dict,
) -> tuple[Model, dict]:
    """Build a multi-output xgboost model for regression and classification."""
    # Training parameters
    xgb_default_params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "early_stopping_rounds": 10,
    }

    # Create model
    model = XGBRegressor(
        **{**xgb_default_params, **params}  # Merge default and tuned parameters
    )
    model._estimator_type = "regressor"  # type: ignore

    # Fit model
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

    # Evaluate
    preds = model.predict(x_test)
    reg_preds = preds[:, 0]
    class_probs = preds[:, 1]
    class_preds = (class_probs > 0.5).astype(int)

    return model, {
        "regression_predictions": reg_preds,
        "class_probabilities": class_probs,
        "class_predictions": class_preds,
    }
