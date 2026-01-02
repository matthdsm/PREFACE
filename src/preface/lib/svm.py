import numpy as np
import numpy.typing as npt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from preface.lib.impute import ImputeOptions, impute_nan
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # type: ignore # noqa
import optuna
import onnxmltools
import onnx
from skl2onnx.common.data_types import FloatTensorType


def svm_tune(
    x: npt.NDArray,  # feature matrix
    y: npt.NDArray,  # target vector
    groups: npt.NDArray,  # group labels for splitting
    n_components: int,  # number of PCA components
    outdir: Path,  # output directory
    impute_option: ImputeOptions,  # imputation strategy
    n_trials: int = 30,  # number of optimization trials
) -> dict:
    def objective(trial) -> float:
        params = {
            "kernel": "linear",
            # Regularization parameter
            "C": trial.suggest_float("C", 0.001, 100, log=True),
        }
        model = SVR(**params)

        # Internal split for the tuner
        gss_internal = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        scores = []

        for _, (train_index, test_index) in enumerate(gss_internal.split(x, y, groups)):
            x_train, x_val = x[train_index], x[test_index]
            y_train, y_val = y[train_index], y[test_index]

            # impute missing values
            x_train, _ = impute_nan(x_train, impute_option)
            x_val, _ = impute_nan(x_val, impute_option)

            # reduce dimensionality with PCA
            current_n_components = min(n_components, x_train.shape[0], x_train.shape[1])
            pca = PCA(n_components=current_n_components)
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            model.fit(x_train, y_train.ravel())
            preds = model.predict(x_val)

        scores.append(mean_squared_error(y_val, preds))

        return np.mean(scores).astype(float)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(outdir / "svm_tuning_history.png")
    return study.best_params


def svm_fit(
    x_train: npt.NDArray,
    x_test: npt.NDArray,
    y_train: npt.NDArray,
    y_test: npt.NDArray,
    params: dict,
) -> tuple[SVR, npt.NDArray]:
    """Build a SVM linear model for regression"""
    # Training parameters
    svr_default_params = {
        "kernel": "linear",
    }

    # Create models
    model = SVR(**svr_default_params, **params)
    model.fit(x_train, y_train.ravel())

    return model, model.predict(x_test)


def svm_export(model: SVR) -> onnx.ModelProto:
    """Export SVM model to ONNX format."""
    initial_type = [("svm_input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = onnxmltools.convert_sklearn(
        model, initial_types=initial_type, target_opset=13
    )
    return onnx_model  # type: ignore
