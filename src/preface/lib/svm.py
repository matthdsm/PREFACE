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
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx


def svm_tune(
    x: npt.NDArray,  # feature matrix
    y: npt.NDArray,  # target vector
    groups: npt.NDArray,  # group labels for splitting
    n_components: float,  # percentage of variance to explain
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
            pca = PCA(n_components=n_components, svd_solver="full")
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

    # Workaround for SVR with 0 support vectors (causes skl2onnx crash)
    if hasattr(model, "n_support_"):
        # Sum of support vectors (checking attributes robustly)
        n_supports = (
            np.sum(model.n_support_)
            if isinstance(model.n_support_, (list, np.ndarray))
            else model.n_support_
        )

        if n_supports == 0:
            # Injecting a dummy support vector with 0 coefficient
            # This allows export but contributes 0 to prediction
            n_features = model.n_features_in_

            # Create dummy attributes if they are empty
            # These ARE required for skl2onnx to populate coefficients/vectors attributes
            model.support_vectors_ = np.zeros((1, n_features), dtype=np.float32)
            model.dual_coef_ = np.zeros((1, 1), dtype=np.float32)

            # Note: We cannot reliably set model.n_support_ as it is often read-only/hidden.
            # skl2onnx might ignore our attempts to set it, resulting in n_supports=0 in ONNX.
            # We will patch the ONNX graph directly below.

    initial_type = [("svm_input", FloatTensorType([None, model.n_features_in_]))]
    onnx_model = to_onnx(model, initial_types=initial_type, target_opset=18)

    # Post-processing: Fix n_supports if it was exported as 0 due to workaround
    if n_supports == 0:
        for node in onnx_model.graph.node:
            if node.op_type == "SVMRegressor":
                for attr in node.attribute:
                    if attr.name == "n_supports" and attr.i == 0:
                        attr.i = 1
    return onnx_model  # type: ignore
