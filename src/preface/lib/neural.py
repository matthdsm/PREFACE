from pathlib import Path

import numpy as np
import numpy.typing as npt
import optuna
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore
from tensorflow.keras import (  # pylint: disable=no-name-in-module,import-error # type: ignore
    Model,
    layers,
)
from preface.lib.impute import ImputeOptions, impute_nan


def neural_tune(
    features: npt.NDArray,
    targets: npt.NDArray,
    n_components: int,
    outdir: Path,
    impute_option: ImputeOptions,
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
        kf_internal = KFold(n_splits=3, shuffle=True)
        scores = []

        for t_idx, v_idx in kf_internal.split(features):
            # Clear session to prevent "tf.function retracing" warning
            keras.backend.clear_session()
            
            x_train, x_val = features[t_idx], features[v_idx]
            y_train, y_val = targets[t_idx], targets[v_idx]

            # impute missing values
            x_train, _ = impute_nan(x_train, impute_option)
            x_val, _ = impute_nan(x_val, impute_option)

            # reduce dimensionality with PCA
            pca = PCA(n_components=n_components)
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            model = multi_output_nn(
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
                ]
            )
            scores.append(min((history.history["val_loss"])))

        return np.mean(scores).astype(float)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=30)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(outdir / "neural_tuning_history.png")
    return study.best_params


def multi_output_nn(
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

    # Head 1: Regression
    reg_out = layers.Dense(1, activation="linear", name="reg_output")(x)
    # Head 2: Classification
    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)

    nn = Model(inputs=input_layer, outputs=[reg_out, class_out])

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"reg_output": "mse", "class_output": "binary_crossentropy"},
        loss_weights={"reg_output": 1.0, "class_output": 1.0},
    )
    return nn


def neural_fit(
    x_train: npt.NDArray,
    x_test: npt.NDArray,
    y_train: npt.NDArray,
    y_test: npt.NDArray,
    params: dict,
) -> tuple[Model, dict]:
    """Build a multi-output neural network for regression and classification."""
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
    # Split targets
    # Assume y[:, 0] = class (sex), y[:, 1] = regression (ff)
    # TODO: this is very brittle, make it more robust
    y_train_reg: npt.NDArray = y_train[:, 1]
    y_train_class: npt.NDArray = y_train[:, 0]
    y_test_reg: npt.NDArray = y_test[:, 1]
    y_test_class: npt.NDArray = y_test[:, 0]

    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Create model
    model = multi_output_nn(
        input_dim=x_train.shape[1], **{**nn_default_params, **params}
    )

    # Fit model
    model.fit(
        x_train,
        {"reg_output": y_train_reg, "class_output": y_train_class},
        validation_data=(
            x_test,
            {"reg_output": y_test_reg, "class_output": y_test_class},
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    # Evaluate
    predictions = model.predict(x_test)
    reg_preds = predictions[0].flatten()
    class_probs = predictions[1].flatten()
    class_preds = (class_probs >= 0.5).astype(int)

    return model, {
        "regression_predictions": reg_preds,
        "class_probabilities": class_probs,
        "class_predictions": class_preds,
    }
