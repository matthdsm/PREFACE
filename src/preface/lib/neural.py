from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import onnxmltools
import optuna
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore
from tensorflow.keras import (  # pylint: disable=no-name-in-module,import-error # type: ignore
    Model,
    layers,
)
from preface.lib.impute import ImputeOptions, impute_nan
from onnxmltools.convert.common.data_types import FloatTensorType


def create_model(
    input_dim: int,
    n_layers: int,
    hidden_size: int,
    learning_rate: float,
    dropout_rate: float,
) -> Model:
    input_layer = layers.Input(shape=(input_dim,))
    x = input_layer
    for i in range(n_layers):
        x = layers.Dense(hidden_size // (2**i), activation="relu")(x)  # type: ignore
        x = layers.Dropout(dropout_rate)(x)

    # regression output
    reg_out = layers.Dense(1, activation="linear", name="reg_output")(x)

    nn = Model(inputs=input_layer, outputs=reg_out)

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        loss_weights={"reg_output": 1.0},
    )
    return nn


def neural_tune(
    x: npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
    n_components: float,
    outdir: Path,
    impute_option: ImputeOptions,
    n_trials: int = 30,
) -> dict:
    def objective(trial) -> float:
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "hidden_size": trial.suggest_int("hidden_size", 16, 128, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
            "epochs": trial.suggest_int("epochs", 20, 100, step=10),
            "batch_size": trial.suggest_int("batch_size", 8, 64, step=8),
        }
        # Internal split for the tuner
        gss_internal = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        scores = []

        for t_idx, v_idx in gss_internal.split(x, y, groups):
            x_train, x_val = x[t_idx], x[v_idx]
            y_train, y_val = y[t_idx], y[v_idx]

            # impute missing values
            x_train, _ = impute_nan(x_train, impute_option)
            x_val, _ = impute_nan(x_val, impute_option)

            # reduce dimensionality with PCA
            pca = PCA(n_components=n_components, svd_solver="full")
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            model = create_model(
                input_dim=x_train.shape[1],
                n_layers=params["n_layers"],
                hidden_size=params["hidden_size"],
                learning_rate=params["learning_rate"],
                dropout_rate=params["dropout_rate"],
            )

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[
                    optuna.integration.TFKerasPruningCallback(trial, "val_loss")
                ],
            )
            scores.append(min((history.history["val_loss"])))

        return np.mean(scores).astype(float)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    outdir.mkdir(parents=True, exist_ok=True)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(outdir / "neural_tuning_history.png")
    return study.best_params


def neural_fit(
    x_train: npt.NDArray,
    x_test: npt.NDArray,
    y_train: npt.NDArray,
    y_test: npt.NDArray,
    params: dict,
) -> tuple[Model, npt.NDArray]:
    """Build a neural network for regression."""
    # Clear session to prevent "tf.function retracing" warning
    keras.backend.clear_session()

    # default parameters
    nn_default_params = {
        "n_layers": 3,
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
    }
    epochs = params.pop("epochs", 50)
    batch_size = params.pop("batch_size", 32)

    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Create model
    model = create_model(input_dim=x_train.shape[1], **{**nn_default_params, **params})

    # Fit model
    model.fit(
        x_train,
        y_train,
        validation_data=(
            x_test,
            y_test,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    return model, model.predict(x_test)


def neural_export(model: Model) -> onnx.ModelProto:
    """Export neural network to ONNX format."""
    initial_type = [("neural_input", FloatTensorType([None, model.input_shape[1]]))]
    onnx_model = onnxmltools.convert_keras(
        model, initial_types=initial_type, target_opset=18
    )
    return onnx_model
