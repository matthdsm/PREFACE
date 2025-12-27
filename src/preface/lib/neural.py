from pathlib import Path

import numpy as np
import numpy.typing as npt
import optuna
from sklearn.model_selection import KFold
from tensorflow import keras  # pylint: disable=no-name-in-module # type: ignore
from tensorflow.keras import (  # pylint: disable=no-name-in-module,import-error # type: ignore
    Model,
    layers,
)


def neural_tune(features: npt.NDArray, targets: npt.NDArray, outdir: Path) -> dict:
    def objective(trial) -> float:
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "hidden_size": trial.suggest_int("hidden_size", 16, 128, step=32),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
        }

        # Internal split for the tuner
        kf_internal = KFold(n_splits=3, shuffle=True)
        scores = []

        for t_idx, v_idx in kf_internal.split(features):
            model = multi_output_nn(
                input_dim=features.shape[1],
                n_layers=params["n_layers"],
                hidden_size=params["hidden_size"],
                learning_rate=params["learning_rate"],
                dropout_rate=params["dropout_rate"],
            )
            history = model.fit(
                features[t_idx],
                targets[t_idx],
                validation_data=(features[v_idx], targets[v_idx]),
                epochs=50,
                batch_size=16,
                callbacks=[
                    optuna.integration.TFKerasPruningCallback(trial, "val_loss")
                ],
            )
            scores.append(min((history.history["val_loss"])))

        return np.mean(scores).astype(float)

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=30)

    optuna.visualization.plot_optimization_history(study).savefig(
        outdir / "neural_tuning_history.png"
    )
    return study.best_params


def multi_output_nn(
    input_dim: int,
    n_layers: int,
    hidden_size: int,
    learning_rate: float,
    dropout_rate: float,
) -> Model:
    x = layers.Input(shape=(input_dim,))
    for _ in range(n_layers):
        x = layers.Dense(hidden_size, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # Head 1: Regression
    reg_out = layers.Dense(1, activation="linear", name="reg_output")(x)
    # Head 2: Classification
    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)

    nn = Model(inputs=x, outputs=[reg_out, class_out])

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"reg_output": "mse", "class_output": "binary_crossentropy"},
        loss_weights={"reg_output": 1.0, "class_output": 1.0},
    )
    return nn


def neural_fit(
    x_train: npt.NDArray,
    x_test: npt.NDArray,
    y_train_reg: npt.NDArray,
    y_train_class: npt.NDArray,
    y_test_reg: npt.NDArray,
    y_test_class: npt.NDArray,
    input_dim: int,
    params: dict,
) -> tuple[Model, dict]:
    """Build a multi-output neural network for regression and classification."""
    # default parameters
    nn_default_params = {
        "n_layers": 3,
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
    }

    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Create model
    model = multi_output_nn(input_dim=input_dim, **{**nn_default_params, **params})

    # Fit model
    model.fit(
        x_train,
        {"reg_output": y_train_reg, "class_output": y_train_class},
        validation_data=(
            x_test,
            {"reg_output": y_test_reg, "class_output": y_test_class},
        ),
        epochs=100,
        batch_size=32,
        verbose=1,
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
