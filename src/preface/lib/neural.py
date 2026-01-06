from pathlib import Path
import io

import numpy as np
import numpy.typing as npt
import onnx
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from preface.lib.impute import ImputeOptions, impute_nan


class NeuralNetwork(nn.Module):
    def __init__(
        self, input_dim: int, n_layers: int, hidden_size: int, dropout_rate: float
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            out_dim = max(hidden_size // (2**i), 1)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def neural_tune(
    x: npt.NDArray,
    y: npt.NDArray,
    groups: npt.NDArray,
    n_components: float,
    outdir: Path,
    impute_option: ImputeOptions,
    n_trials: int = 30,
) -> dict:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

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

        for _, (t_idx, v_idx) in enumerate(gss_internal.split(x, y, groups)):
            x_train, x_val = x[t_idx], x[v_idx]
            y_train, y_val = y[t_idx], y[v_idx]

            # impute missing values
            x_train, _ = impute_nan(x_train, impute_option)
            x_val, _ = impute_nan(x_val, impute_option)

            # reduce dimensionality with PCA
            pca = PCA(n_components=n_components, svd_solver="full")
            x_train = pca.fit_transform(x_train)
            x_val = pca.transform(x_val)

            # Convert to tensors
            x_train_t = torch.tensor(x_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            x_val_t = torch.tensor(x_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)

            train_ds = TensorDataset(x_train_t, y_train_t)
            val_ds = TensorDataset(x_val_t, y_val_t)
            train_loader = DataLoader(
                train_ds, batch_size=params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(val_ds, batch_size=params["batch_size"])

            model = NeuralNetwork(
                input_dim=x_train.shape[1],
                n_layers=params["n_layers"],
                hidden_size=params["hidden_size"],
                dropout_rate=params["dropout_rate"],
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

            # Training loop
            val_loss = float("inf")
            for epoch in range(params["epochs"]):
                train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = validate_epoch(model, val_loader, criterion, device)
                scores.append(val_loss)

        mean_val_loss = np.mean(scores).astype(float)
        return mean_val_loss

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
) -> tuple[NeuralNetwork, npt.NDArray]:
    """Build a neural network for regression."""
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    # default parameters
    nn_default_params = {
        "n_layers": 3,
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "dropout_rate": 0.3,
    }
    epochs = params.pop("epochs", 50)
    batch_size = params.pop("batch_size", 32)
    # Merge params
    final_params = {**nn_default_params, **params}

    # Data loaders
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Create model
    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        n_layers=final_params["n_layers"],
        hidden_size=final_params["hidden_size"],
        dropout_rate=final_params["dropout_rate"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=final_params["learning_rate"])

    # Early stopping
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Predict
    model.eval()
    with torch.no_grad():
        preds = model(x_val_t.to(device)).cpu().numpy()

    return model, preds


def neural_export(model: NeuralNetwork) -> onnx.ModelProto:
    """Export neural network to ONNX format."""
    # Since we need input dim, we check the first layer
    # model.model[0] is Linear
    input_dim = model.model[0].in_features  # type: ignore

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # Needs to be on CPU for export
    model.cpu()
    model.eval()

    f = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        f,
        input_names=["neural_input"],
        output_names=["reg_output"],
        dynamic_axes={
            "neural_input": {0: "batch_size"},
            "reg_output": {0: "batch_size"},
        },
        opset_version=18,
    )

    f.seek(0)
    model_proto = onnx.load_model_from_string(f.getvalue())
    return model_proto
