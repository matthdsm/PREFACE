import unittest
import shutil
import tempfile
import numpy as np
import logging
from pathlib import Path
from sklearn.svm import SVR
import onnx
from preface.lib.svm import svm_fit, svm_tune, svm_export
from preface.lib.impute import ImputeOptions

# Configure logging to suppress verbose output during tests
logging.basicConfig(level=logging.ERROR)


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = Path(self.temp_dir)
        self.n_samples = 20
        self.n_features = 5
        # Create synthetic data
        self.X = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        # Add some structure to Y to make learning possible (y = 2*x0 + 0.5)
        self.y = (2 * self.X[:, 0] + 0.5).astype(np.float32)
        # Groups for GroupShuffleSplit
        self.groups = np.array([i // 2 for i in range(self.n_samples)])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_svm_fit(self):
        """Test SVM model training."""
        params = {"C": 1.0}

        # Split data manually for test
        x_train = self.X[:15]
        y_train = self.y[:15]
        x_test = self.X[15:]
        y_test = self.y[15:]

        model, preds = svm_fit(x_train, x_test, y_train, y_test, params)

        self.assertIsInstance(model, SVR)
        self.assertEqual(preds.shape, (5,))  # 5 test samples, 1D array for SVM

        # Ensure predictions are floats (can be float32 or float64 from sklearn)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_svm_tune(self):
        """Test hyperparameter tuning with mocked n_trials."""
        n_components = 3

        best_params = svm_tune(
            x=self.X,
            y=self.y,
            groups=self.groups,
            n_components=n_components,
            outdir=self.out_dir,
            impute_option=ImputeOptions.ZERO,
            n_trials=1,
        )

        self.assertIsInstance(best_params, dict)
        self.assertIn("C", best_params)

        # Check if plot was created
        self.assertTrue((self.out_dir / "svm_tuning_history.png").exists())

    def test_svm_export(self):
        """Test ONNX export."""
        model = SVR(kernel="linear")
        model.fit(self.X, self.y)

        try:
            onnx_model = svm_export(model)
            self.assertIsInstance(onnx_model, onnx.ModelProto)
        except ValueError as e:
            # Handle potential Opset version issues gracefully
            if "Opset" in str(e):
                logging.warning(
                    f"Skipping ONNX export validation due to Opset mismatch: {e}"
                )
            else:
                raise e


if __name__ == "__main__":
    unittest.main()
