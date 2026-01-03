from pathlib import Path
import numpy as np
import numpy.typing as npt
import onnx
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from skl2onnx.convert import may_switch_bases_classes_order
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from preface.lib.impute import impute_nan, ImputeOptions


def xgboost_tune(
    x: npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
    n_components: int,
    outdir: Path,
    impute_option: ImputeOptions,
) -> dict:
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
            # random state for reproducibility
            "random_state": 42,
            # base score
            "base_score": 0.5,
        }
        model = XGBRegressor(**params)

        # Internal split for the tuner
        gss_internal = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        scores = []

        for _, (train_index, test_index) in enumerate(gss_internal.split(x, y, groups)):
            x_train, x_val = x[train_index], x[test_index]
            y_train, y_val = y[train_index], y[test_index]

            # impute missing values
            x_train, _ = impute_nan(x_train, impute_option)
            x_val, _ = impute_nan(x_val, impute_option)

            # Reduce dimensionality with PCA
            current_n_components = min(n_components, x_train.shape[0], x_train.shape[1])
            pca = PCA(n_components=current_n_components)
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            # Train and evaluate model
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
) -> tuple[XGBRegressor, npt.NDArray]:
    """Build a xgboost model for regression."""
    # Training parameters
    xgb_default_params = {
        "tree_method": "hist",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "early_stopping_rounds": 10,
        "base_score": 0.5,
    }

    # Create model
    model = XGBRegressor(
        **{**xgb_default_params, **params}  # Merge default and tuned parameters
    )
    model._estimator_type = "regressor"  # type: ignore

    # Fit model
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    return model, model.predict(x_test)


def xgboost_export(model: XGBRegressor) -> onnx.ModelProto:
    """Export XGBoost model to ONNX format."""

    update_registered_converter(
        XGBRegressor,
        "XGBoostXGBRegressor",
        calculate_linear_regressor_output_shapes,
        convert_xgboost,
    )

    # Sanitize base_score if it's not a scalar (skl2onnx issue with newer xgboost)
    if hasattr(model, "base_score") and model.base_score is not None:
        if not isinstance(model.base_score, (float, int)):
            try:
                # If it's an array/list, take the first element for single output regression
                model.base_score = float(model.base_score)
            except (TypeError, ValueError):
                 # Try taking first element of array-like
                 try:
                    model.base_score = float(model.base_score[0]) # type: ignore
                 except:
                    pass


    with may_switch_bases_classes_order(XGBRegressor):
        initial_type = [("xgboost_input", FloatTensorType([None, model.n_features_in_]))]
        onnx_model = to_onnx(model, initial_types=initial_type, target_opset=18)

    return onnx_model
