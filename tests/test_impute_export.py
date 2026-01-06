import unittest
import numpy as np
import onnx
from preface.lib.impute import impute_nan, impute_export, ImputeOptions


class TestImputeExport(unittest.TestCase):
    def setUp(self):
        self.n_samples = 50
        self.n_features = 5
        self.data = np.random.rand(self.n_samples, self.n_features).astype(np.float32)
        # Introduce NaNs
        self.data[5, 2] = np.nan
        self.data[10, 0] = np.nan
        self.data[25, 4] = np.nan

    def _run_export_test(self, imputer):
        """Helper to run ONNX export and validation."""
        self.assertIsNotNone(imputer)
        onnx_model = impute_export(imputer, self.n_features)

        self.assertIsInstance(onnx_model, onnx.ModelProto)
        self.assertGreater(len(onnx_model.graph.node), 0)
        self.assertEqual(len(onnx_model.graph.input), 1)
        onnx.checker.check_model(onnx_model)

    def test_export_simple_imputer_mean(self):
        """Test ONNX export for SimpleImputer with 'mean' strategy."""
        _, imputer = impute_nan(self.data.copy(), ImputeOptions.MEAN)
        self._run_export_test(imputer)

    def test_export_simple_imputer_median(self):
        """Test ONNX export for SimpleImputer with 'median' strategy."""
        _, imputer = impute_nan(self.data.copy(), ImputeOptions.MEDIAN)
        self._run_export_test(imputer)

    def test_export_simple_imputer_zero(self):
        """Test ONNX export for SimpleImputer with 'zero' (constant) strategy."""
        _, imputer = impute_nan(self.data.copy(), ImputeOptions.ZERO)
        self._run_export_test(imputer)

    def test_export_knn_imputer(self):
        """Test ONNX export for KNNImputer."""
        _, imputer = impute_nan(self.data.copy(), ImputeOptions.KNN)
        self._run_export_test(imputer)

    def test_export_iterative_imputer_raises_error(self):
        """Test that ONNX export raises an error for IterativeImputer."""
        _, imputer = impute_nan(self.data.copy(), ImputeOptions.MICE)
        with self.assertRaises(NotImplementedError):
            impute_export(imputer, self.n_features)


if __name__ == "__main__":
    unittest.main()
