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


def xgboost_tune(features: npt.NDArray, targets: npt.NDArray, outdir: Path) -> dict:
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
            model = XGBRegressor(**params)
            model.fit(features[t_idx], targets[t_idx])
            preds = model.predict(features[v_idx])
            scores.append(mean_squared_error(targets[v_idx], preds))
        return np.mean(scores).astype(float)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    optuna.visualization.plot_optimization_history(study).savefig(
        outdir / "xgboost_tuning_history.png"
    )
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

    # Fit model
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

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
