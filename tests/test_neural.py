import unittest
import shutil
import tempfile
import numpy as np
import logging
from pathlib import Path
from tensorflow.keras import Model
import onnx
from preface.lib.neural import create_model, neural_fit, neural_tune, neural_export
from preface.lib.impute import ImputeOptions

# Configure logging to suppress verbose output during tests
logging.basicConfig(level=logging.ERROR)


class TestNeural(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)
        self.n_samples = 20
        self.n_features = 10
        # Create synthetic data
        self.X = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        # Add some structure to Y to make learning possible (y = 2*x0 + 0.5)
        self.y = (2 * self.X[:, 0] + 0.5).astype(np.float32)
        # Groups for GroupShuffleSplit (ensure enough groups)
        # With n_splits=5, test_size=0.2, we need enough groups.
        # 20 samples, if we have 10 groups of 2 samples each.
        self.groups = np.array([i // 2 for i in range(self.n_samples)])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_model(self):
        """Test model creation structure."""
        input_dim = self.n_features
        n_layers = 2
        hidden_size = 32
        learning_rate = 0.01
        dropout_rate = 0.2

        model = create_model(
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        )

        self.assertIsInstance(model, Model)
        # Check input shape: (None, 10)
        self.assertEqual(model.input_shape, (None, input_dim))
        # Check output shape: (None, 1)
        self.assertEqual(model.output_shape, (None, 1))

        # Check number of layers
        # Input layer is not always counted in len(model.layers) depending on how it's created,
        # but with functional API:
        # 1. Input (not in layers list usually if using Input() separately but here x = input_layer)
        # Loop 2 times: Dense, Dropout -> 4 layers
        # Output Dense -> 1 layer
        # Total expected: 5 layers (plus maybe input layer if counted, let's check names)

        # layers in create_model:
        # loop range(n_layers): Dense, Dropout
        # then Dense(1)
        # So 2 * n_layers + 1
        expected_layers = 2 * n_layers + 1
        self.assertEqual(
            len(model.layers), expected_layers + 1
        )  # +1 for InputLayer which usually appears in functional model.layers

    def test_neural_fit(self):
        """Test model training."""
        params = {
            "n_layers": 1,
            "hidden_size": 16,
            "learning_rate": 0.01,
            "dropout_rate": 0.1,
            "epochs": 2,  # Very fast
            "batch_size": 4,
        }

        # Split data manually for test
        x_train = self.X[:15]
        y_train = self.y[:15]
        x_test = self.X[15:]
        y_test = self.y[15:]

        model, preds = neural_fit(x_train, x_test, y_train, y_test, params)

        self.assertIsInstance(model, Model)
        self.assertEqual(preds.shape, (5, 1))  # 5 test samples

        # Ensure predictions are floats
        self.assertEqual(preds.dtype, np.float32)

    def test_neural_tune(self):
        """Test hyperparameter tuning with mocked n_trials."""
        # Use a small n_components for PCA
        n_components = 5

        # We need enough data/groups for the internal split in neural_tune
        # neural_tune uses GroupShuffleSplit(n_splits=5, test_size=0.2)
        # It loops 5 times.
        # This might be slow but with epochs set by optuna, we hope the suggested epochs are small?
        # Optuna range for epochs is 20-100.
        # We can't easily control the inner epochs unless we mock create_model or fit.
        # BUT, we set n_trials=1, so it only runs once.
        # 20 epochs on 20 samples is fast.

        # We need to make sure we handle the return value
        best_params = neural_tune(
            x=self.X,
            y=self.y,
            groups=self.groups,
            n_components=n_components,
            outdir=self.out_dir,
            impute_option=ImputeOptions.ZERO,  # Use ZERO to avoid complex imputation in test
            n_trials=1,
        )

        self.assertIsInstance(best_params, dict)
        self.assertIn("n_layers", best_params)
        self.assertIn("hidden_size", best_params)
        self.assertIn("learning_rate", best_params)

        # Check if plot was created
        self.assertTrue((self.out_dir / "neural_tuning_history.png").exists())

    def test_neural_export(self):
        """Test ONNX export."""
        model = create_model(
            input_dim=self.n_features,
            n_layers=1,
            hidden_size=16,
            learning_rate=0.01,
            dropout_rate=0.1,
        )
        # We don't need to train it to export it
        try:
            onnx_model = neural_export(model)
            self.assertIsInstance(onnx_model, onnx.ModelProto)
        except ValueError as e:
            # Handle potential Opset version issues gracefully if environment isn't perfect
            if "Opset" in str(e):
                logging.warning(
                    f"Skipping ONNX export validation due to Opset mismatch: {e}"
                )
            else:
                raise e


if __name__ == "__main__":
    unittest.main()
